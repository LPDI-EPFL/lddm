from rdkit import Chem
import argparse
import warnings
from pathlib import Path
import logging
import yaml

warnings.filterwarnings("ignore")

from lddm import utils
from lddm.inference.utils import load_model
from lddm.utils import merge_args_and_yaml, set_default, setup_logging, write_smpls_to_dir
from lddm.inference.programmable_generation import ProgrammableGeneration, check_config
from lddm.inference.synthesizable_generation import SynthesizableGeneration

def main():
    utils.disable_rdkit_logging()

    p = argparse.ArgumentParser()
    p.add_argument('--protein', type=str)
    p.add_argument('--ligand', type=str)
    p.add_argument('--output', type=str, help='Output prefix for the samples SDF, pocket PDB and results CSV')
    p.add_argument('config', type=str, help='Path to the YAML configuration file for controlled sampling')
    
    p.add_argument('--synthesizable', action='store_true', default=None, help='Use synthesizable sampling')
    p.add_argument('--starting_fragments', type=str, help='Starting fragments for iterative sampling')
    p.add_argument('--save_all', action='store_true', default=None, help='Save all intermediate steps')
    p.add_argument('--verbose', action='store_true', default=None, help='Verbose logging')
    args = p.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    cfg = merge_args_and_yaml(args, config)

    set_default(cfg, 'verbose', False)
    setup_logging(cfg.verbose)
    logging.info("Starting controlled generation ...")
    logging.info(f'Using config file {args.config}')

    set_default(cfg, 'ligand', None)
    set_default(cfg, 'protein', None)
    if cfg.ligand is None or not Path(cfg.ligand).exists():
        raise ValueError('No ligand provided or file does not exist.')
    else:
        logging.info(f'Using ligand at {cfg.ligand}')
    if cfg.protein is None or not Path(cfg.protein).exists():
        raise ValueError('No protein provided or file does not exist.')
    else:
        logging.info(f'Using protein at {cfg.protein}')

    cfg = check_config(cfg)

    if getattr(cfg, 'seed', None) is not None:
        utils.set_deterministic(seed=cfg.seed)

    logging.info("Loading model...")
    model = load_model(cfg)
    model.masked_modeling = True

    if cfg.synthesizable:
        logging.info("Using synthesizable sampling")
        controlled_sampler = SynthesizableGeneration(
            model, 
            cfg.sampling_params, 
            cfg.itergen_params,
            cfg.docking_params,
        )
    else:
        controlled_sampler = ProgrammableGeneration(
            model, 
            cfg.sampling_params, 
            cfg.itergen_params,
        )
    
    logging.info("Preparing input...")
    controlled_sampler.setup_input(cfg.ligand, cfg.protein, starting_frag_p=cfg.starting_fragments)

    logging.info(f'Generating samples with {cfg.itergen_params.max_sampling_iter} sampling iterations')
    controlled_sampler.sample()

    sample_df = controlled_sampler.retrieve_results(return_format='table')
    if sample_df.empty:
        raise RuntimeError('No molecules passed selection; increase the sampling budget.')

    set_default(cfg, 'output', 'generated_samples')
    cfg.output = Path(cfg.output)
    parent_dir = cfg.output.parent
    if not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
    write_smpls_to_dir(
        cfg.output, 
        sample_df['mol'],
        controlled_sampler.pocket_p
    )
    complete_p = Path(str(cfg.output) + '_complete.csv')
    sample_df.drop(columns=['mol','frags'], errors='ignore').to_csv(complete_p, index=False)
    
    if cfg.save_all:
        out_dir = Path(str(cfg.output) + '_steps')
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f'Saving intermediate steps to {out_dir}')
        for path in controlled_sampler.fragment_tree.get_generated_paths():
            name = path[-1].GetProp('_Name')
            wint = Chem.SDWriter(f'{out_dir}/{name}.sdf')
            for mol in path:
                utils.add_mol_to_sdwriter(wint, mol, catch_errors=True)
            wint.close()

        frag_stats = controlled_sampler.get_fragment_stats()
        frag_stats_p = Path(str(cfg.output) + '_frag_stats.csv')
        frag_stats.to_csv(frag_stats_p, index=False)

if __name__ == "__main__":
    main()
