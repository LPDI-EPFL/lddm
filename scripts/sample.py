import argparse
import logging
import warnings
from pathlib import Path
from functools import partial

from tqdm import tqdm
from torch.utils.data import DataLoader

from lddm import utils
from lddm.data.data_utils import TensorDict
from lddm.data.dataset import ProcessedDataset
from lddm.data.molecule_builder import mols_to_pdbfile
from lddm.inference.utils import load_model, prepare_input


def setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(level)
    if not verbose:
        warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")


def add_common_args(p):
    """Arguments shared by the 'design' and 'dock' modes."""
    # Inputs
    p.add_argument('--protein', type=str, required=True,
                   help="Path to the protein PDB file.")
    p.add_argument('--ref_ligand', type=str, required=True,
                   help="SDF file of a reference ligand used only to define the "
                        "binding pocket (its interacting residues). It is not part "
                        "of the generated output.")
    p.add_argument('--checkpoint', type=str, required=True,
                   help="Path to the trained LDDM model checkpoint (.ckpt).")

    # Output
    p.add_argument('--output', type=str, required=True,
                   help="Path to the output SDF file the samples are written to.")

    # Sampling
    p.add_argument('--n_samples', type=int, default=10,
                   help="Number of molecules to generate.")
    p.add_argument('--batch_size', type=int, default=None,
                   help="Number of samples generated per forward pass. "
                        "Defaults to --n_samples.")
    p.add_argument('--n_steps', type=int, default=100,
                   help="Number of integration steps of the sampling process.")
    p.add_argument('--sampler', type=str, default='ForwardEuler',
                   choices=['ForwardEuler', 'HeunSampler'],
                   help="ODE sampler used to integrate the generative process.")
    p.add_argument('--sampling_noise', type=float, default=5.0,
                   help="Scale of the stochastic noise injected into the atom "
                        "coordinates during sampling (overrides the checkpoint's "
                        "value).")

    # Trajectory visualization (optional)
    p.add_argument('--n_frames', type=int, default=None,
                   help="If set, save a sampling trajectory with this many frames "
                        "(one sample at a time) instead of a batch of final "
                        "molecules. Useful for visualizing the sampling process.")
    p.add_argument('--return_projected_final', action='store_true',
                   help="When saving a trajectory (--n_frames), project each frame "
                        "onto the final clean prediction rather than showing the "
                        "raw noisy state.")

    # Misc
    p.add_argument('--device', type=str, default='cuda:0',
                   help="Device to run inference on, e.g. 'cuda:0' or 'cpu'.")
    p.add_argument('--seed', type=int, default=None,
                   help="Random seed for reproducible sampling.")
    p.add_argument('--verbose', action='store_true',
                   help="Enable debug-level logging.")


def parse_args():
    p = argparse.ArgumentParser(
        description="Sample molecules with LDDM in either 'design' or 'dock' mode."
    )
    subparsers = p.add_subparsers(dest='mode', required=True,
                                  help="Sampling mode.")

    # --- Design ---
    design = subparsers.add_parser(
        'design', help="Generate new molecules for the pocket (optionally "
                       "growing/linking provided fragments).")
    add_common_args(design)
    design.add_argument('--ligand', type=str, default=None,
                        help="Optional SDF with fragment(s) to grow or link "
                             "(fragment-based design). The fragments are kept "
                             "fixed and new atoms are generated around them. "
                             "If omitted, molecules are designed de novo.")
    design.add_argument('--molecule_size', type=str, default=None,
                        help="Target ligand size. An integer for a fixed size, "
                             "'uniform_<low>_<high>' for a random size, or omit "
                             "to sample the size from the pocket-conditioned "
                             "histogram.")

    # --- Dock ---
    dock = subparsers.add_parser(
        'dock', help="Predict the binding pose of a given molecule.")
    add_common_args(dock)
    dock.add_argument('--ligand', type=str, required=True,
                      help="SDF of the molecule to dock. Its atom types and bonds "
                           "are kept fixed; only the positions are generated.")
    dock.add_argument('--atoms_to_dock', type=int, nargs='+', default=None,
                      help="Optional atom indices to dock (partial docking). "
                           "Only these atoms get new coordinates; the remaining "
                           "atoms keep their input positions. Docks the whole "
                           "molecule if omitted.")

    args = p.parse_args()
    args.batch_size = args.batch_size or args.n_samples
    return args


def parse_molecule_size(spec):
    """Turn the --molecule_size string into the spec expected by model.sample()."""
    if spec is None:
        return None
    if spec.isdigit():
        return int(spec)
    return spec  # e.g. 'uniform_5_10'


def main():
    args = parse_args()
    setup_logging(args.verbose)
    if args.seed is not None:
        utils.set_deterministic(seed=args.seed)
    utils.disable_rdkit_logging()

    logging.info("Loading model...")
    model = load_model(args)

    # Determine the ligand size handling for each mode.
    if args.mode == 'dock':
        # In docking the molecule is fixed: use the ground-truth size and disable
        # virtual nodes so that LDDM does not add any atoms.
        model.virtual_nodes = None
        num_nodes = "ground_truth"
    else:
        num_nodes = parse_molecule_size(args.molecule_size)

    logging.info("Preparing input...")
    ligand, pocket = prepare_input(args, model)

    Path(args.output).parent.absolute().mkdir(parents=True, exist_ok=True)
    logging.info(f"Generating {args.n_samples} samples ({args.mode})")

    if args.n_frames is None:
        # Batched sampling of final molecules.
        dataset = [{'ligand': ligand, 'pocket': pocket} for _ in range(args.n_samples)]
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            collate_fn=partial(ProcessedDataset.collate_fn, ligand_transform=None),
            pin_memory=True,
        )

        mol_samples = []
        for data in tqdm(dataloader):
            new_data = {
                'ligand': TensorDict(**data['ligand']).to(args.device),
                'pocket': TensorDict(**data['pocket']).to(args.device),
            }
            rdmols, _, _ = model.sample(
                new_data,
                n_samples=1,
                num_nodes=num_nodes,
            )
            mol_samples.extend(rdmols)

        utils.write_sdf_file(args.output, mol_samples, add_atomprops_as_molprops=True)

    else:
        # Trajectory sampling (sample_chain handles a single sample at a time).
        dataloader = DataLoader(
            dataset=[{'ligand': ligand, 'pocket': pocket}],
            batch_size=1,
            collate_fn=partial(ProcessedDataset.collate_fn, ligand_transform=None),
        )
        data = next(iter(dataloader))
        new_data = {
            'ligand': TensorDict(**data['ligand']).to(args.device),
            'pocket': TensorDict(**data['pocket']).to(args.device),
        }
        ligand_chain, pocket_chain, _ = model.sample_chain(
            new_data,
            keep_frames=args.n_frames,
            num_nodes=num_nodes,
            docking=(args.mode == 'dock'),
            project_final=args.return_projected_final,
        )

        utils.write_sdf_file(args.output, ligand_chain, add_atomprops_as_molprops=True)
        mols_to_pdbfile(pocket_chain, str(Path(args.output).with_suffix('.pdb')))

    logging.info(f"Done. Wrote output to {args.output}")


if __name__ == '__main__':
    main()
