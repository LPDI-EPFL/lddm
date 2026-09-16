import logging
from collections import defaultdict
from pathlib import Path

import torch
from rdkit import Chem
from Bio.PDB import PDBParser

from lddm.config.main import LDDMConfig
from lddm.constants import atom_encoder, bond_encoder
from lddm.data.data_utils import process_raw_pair, prepare_ligand, Ligand
from lddm.model.lightning import LDDM


def get_model_params(checkpoint_path, args):
    model_params = torch.load(checkpoint_path, map_location=args.device, weights_only=False)['hyper_parameters']
    model_params = LDDMConfig.from_dict(model_params, strict=False)

    # Update parameters
    update_params = defaultdict(dict)

    if hasattr(args, "n_steps") and args.n_steps is not None:
        update_params['simulation_params']['n_steps'] = args.n_steps

    if hasattr(args, "sampler") and args.sampler is not None:
        update_params['simulation_params']['sampler'] = args.sampler

    if hasattr(args, "sampling_noise") and args.sampling_noise is not None:
        update_params['simulation_params']['noise_scale_x'] = args.sampling_noise

    model_params.update(update_params)
    return model_params.to_dict()


def load_model(args):
    chkpt_path = Path(args.checkpoint)
    model_params = get_model_params(chkpt_path, args)
    model = LDDM.load_from_checkpoint(chkpt_path, map_location=args.device, weights_only=False, **model_params)
    
    logging.debug(f"Sampler: {type(model.sampler).__name__}")

    model.setup(stage='generation')
    model.batch_size = model.eval_batch_size = args.batch_size
    model.eval().to(args.device)
    model.all_masked = True
    
    return model


def prepare_input(args, model):
    """
    Build the (ligand, pocket) input for a sampling run.

    The pocket is always defined by ``--ref_ligand`` (the reference ligand is
    only used to select the interacting pocket residues, it is not part of the
    generated output). The meaning of ``--ligand`` depends on ``args.mode``:

    * ``design``: an optional SDF with fragment(s) that are kept fixed and
      grown/linked around (fragment-based design). If omitted, sampling starts
      from an empty ligand (de novo design).
    * ``dock``: the molecule to be docked. Its atom types and bonds are known,
      only the atom positions are generated. ``--atoms_to_dock`` optionally
      restricts docking to a subset of atoms (partial docking); the remaining
      atoms keep their input coordinates.
    """
    config = model.featurization_config

    # The pocket is defined by the reference ligand.
    pdb_model = PDBParser(QUIET=True).get_structure('', args.protein)[0]
    ref_rdmol = Chem.SDMolSupplier(str(args.ref_ligand), sanitize=False)[0]
    _, pocket = process_raw_pair(pdb_model, ref_rdmol, config=config)

    # The ligand carries the (partial) information conditioned on during sampling.
    if getattr(args, 'ligand', None) is not None:
        rdmol = Chem.SDMolSupplier(str(args.ligand), sanitize=False)[0]
        ligand = Ligand(**prepare_ligand(rdmol, config=config))
    else:
        ligand = Ligand.empty(config)

    # By default nothing is known (everything is generated).
    ligand['known_x'] = torch.zeros_like(ligand['mask']).bool()
    ligand['known_h'] = torch.zeros_like(ligand['mask']).bool()
    ligand['known_e'] = torch.zeros_like(ligand['bond_mask']).bool()

    if args.mode == 'design':
        if getattr(args, 'ligand', None) is not None:
            # Fragment-based design: fix the provided fragments (positions, types
            # and bonds) and grow/link new atoms around them.
            ligand['known_x'] = torch.ones_like(ligand['mask']).bool()
            ligand['known_h'] = torch.ones_like(ligand['mask']).bool()
            ligand['known_e'] = torch.ones_like(ligand['bond_mask']).bool()

    elif args.mode == 'dock':
        # Atom types and bonds are known; positions are (re)generated.
        ligand['known_h'] = torch.ones_like(ligand['mask']).bool()
        ligand['known_e'] = torch.ones_like(ligand['bond_mask']).bool()

        if getattr(args, 'atoms_to_dock', None):
            # Partial docking: only the selected atoms are docked, the rest keep
            # their input coordinates.
            known_x = torch.ones_like(ligand['mask']).bool()
            dock_idx = torch.tensor(args.atoms_to_dock, dtype=torch.long)
            known_x[dock_idx] = False
            ligand['known_x'] = known_x

    else:
        raise ValueError(f"Unknown mode: {args.mode!r}")

    ligand['name'] = 'ligand'
    return ligand, pocket
