import warnings
import json
import logging
import time
import tempfile
import shutil

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
import numpy as np
import pandas as pd

from functools import partial
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from argparse import Namespace
from dataclasses import replace

from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

from lddm import utils
from lddm.constants import INT_TYPE, atom_encoder, bond_encoder
from lddm.utils import dict_to_namespace, namespace_to_dict, set_default
from lddm.data.dataset import ProcessedDataset
from lddm.data.data_utils import TensorDict, prepare_ligand, process_raw_pair
from lddm.sbdd_metrics.metrics import FullEvaluator, FullLocalEvaluator, LOCAL_EVALUATORS, is_nan
from lddm.inference.programmable_design_utils import *


def check_config(cfg):   
    if getattr(cfg, 'checkpoint', None) is None or not Path(cfg.checkpoint).exists():
        raise ValueError('No checkpoint provided or file does not exist.')
    else:
        logging.info(f'Using checkpoint at {cfg.checkpoint}')

    # setting default values
    set_default(cfg, 'synthesizable', False)
    set_default(cfg, 'sampling_params', dict_to_namespace({}))
    set_default(cfg, 'itergen_params', dict_to_namespace({}))
    set_default(cfg, 'docking_params', dict_to_namespace({}))
    set_default(cfg, 'device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    set_default(cfg, 'datadir', None)
    set_default(cfg, 'save_all', False)
    set_default(cfg, 'starting_fragments', None)
    set_default(cfg, 'batch_size', 16)

    # Forward sampling options to the shared model loader.
    if hasattr(cfg.sampling_params, 'n_steps'):
        cfg.n_steps = cfg.sampling_params.n_steps
    if hasattr(cfg.sampling_params, 'sampler'):
        cfg.sampler = cfg.sampling_params.sampler
    if hasattr(cfg.sampling_params, 'batch_size'):
        cfg.batch_size = cfg.sampling_params.batch_size
    if hasattr(cfg.sampling_params, 'sampling_noise'):
        cfg.sampling_noise = cfg.sampling_params.sampling_noise
    
    return cfg


class ProgrammableGeneration:
    ##################################### Initialization #####################################
    def __init__(self, model,
                 sampling_params: Namespace,
                 itergen_params: Namespace):
        self.model = model
        self.pocket_p = None

        set_default(sampling_params, 'pocket_distance_cutoff', 8.0)
        set_default(sampling_params, 'sample_with_ground_truth_size', False)
        set_default(sampling_params, 'molecule_size', None)
        self.pocket_distance_cutoff = sampling_params.pocket_distance_cutoff
        self.molecule_size = sampling_params.molecule_size
        self.sample_with_ground_truth_size = sampling_params.sample_with_ground_truth_size
        if self.sample_with_ground_truth_size:
            if sampling_params.molecule_size is not None:
                logging.warning('Overwriting molecule_size with ground truth size')
        self.featurization_config = replace(
            model.featurization_config, dist_cutoff=self.pocket_distance_cutoff,
        )
        if hasattr(sampling_params, 'batch_size'):
            self.model.batch_size = sampling_params.batch_size
        if hasattr(sampling_params, 'n_steps'):
            self.model.T_sampling = sampling_params.n_steps

        set_default(itergen_params, 'max_sampling_iter', 20)
        set_default(itergen_params, 'stop_after_n_mols', None)
        set_default(itergen_params, 'n_samples_per_iter', 64)
        set_default(itergen_params, 'use_ucb', True)
        set_default(itergen_params, 'exploration_weight', 1.0)
        set_default(itergen_params, 'min_root_sample_proportion', 0.1)
        set_default(itergen_params, 'max_tree_depth', 10)
        set_default(itergen_params, 'drop_random', None)
        set_default(itergen_params, 'restrict_fragment_size_threshold', None)
        set_default(itergen_params, 'score_history_window', 50)
        set_default(itergen_params, 'fragmentation_method', 'brics')
        self.max_sampling_iter = itergen_params.max_sampling_iter
        self.stop_after_n_mols = itergen_params.stop_after_n_mols
        self.n_samples_per_iter = itergen_params.n_samples_per_iter
        self.use_ucb = itergen_params.use_ucb
        self.min_root_sample_proportion = itergen_params.min_root_sample_proportion
        self.drop_random = itergen_params.drop_random
        self.restrict_fragment_size_threshold = itergen_params.restrict_fragment_size_threshold
        self.exploration_weight = itergen_params.exploration_weight
        self.max_tree_depth = itergen_params.max_tree_depth
        self.score_history_window = itergen_params.score_history_window
        self.fragmentation_method = itergen_params.fragmentation_method
        self.pocket_p = None
        self.setup_evaluators(itergen_params)

    def __del__(self):
        # clean up tmp files
        if getattr(self, 'pocket_p', None) and self.pocket_p.exists():
            self.pocket_p.unlink()

    def setup_input(self, ligand_p, pocket_p, starting_frag_p=None):
        if not Path(ligand_p).exists():
            raise ValueError(f'Ligand file not found: {ligand_p}')
        if not Path(pocket_p).exists():
            raise ValueError(f'Pocket file not found: {pocket_p}')

        pdb_model = PDBParser(QUIET=True).get_structure('', pocket_p)[0]
        rdmol = Chem.SDMolSupplier(str(ligand_p), sanitize=False)[0]
        starting_frags = None
        if starting_frag_p is not None:
            if not Path(starting_frag_p).exists():
                raise ValueError(f'Starting fragments file not found: {starting_frag_p}')
            starting_frags = Chem.SDMolSupplier(str(starting_frag_p), sanitize=False)[0]
            try:
                starting_frags2 = Chem.RemoveHs(starting_frags)
                Chem.SanitizeMol(starting_frags2)
                starting_frags = starting_frags2
            except:
                logging.warning('Failed to sanitize starting fragments. Continuing with possibly invalid molecule.')
            logging.info(f'Loaded starting fragment {Chem.MolToSmiles(starting_frags)} from {starting_frag_p}')

        ligand, pocket = process_raw_pair(
            pdb_model, rdmol,
            config=self.featurization_config,
            return_pocket_pdb=True,
        )
        ligand['name'] = 'ligand_0'
        logging.info(f'Input pocket size: {pocket["size"].item()}')
        logging.info(f'Input ligand size: {ligand["size"].item()}')
        self.init_ligand = ligand
        if self.sample_with_ground_truth_size:
            self.molecule_size = int(ligand['size'].item())
        self.pocket = pocket
        self.starting_frags = starting_frags
        # save tmp pocket pdb
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb') as tmp:
            processed_pocket = self.pocket.pop('pocket_pdb')
            io = PDBIO()
            io.set_structure(processed_pocket)
            io.save(tmp.name)
            self.pocket_p = Path(tmp.name)

        # update evaluators with pocket
        if self.frag_evaluator is not None:
            self.frag_evaluator.update_protein(self.pocket_p)
        if self.global_evaluator is not None:
            self.global_evaluator.update_protein(self.pocket_p)
        if self.local_evaluator is not None:
            self.local_evaluator.update_protein(self.pocket_p)
        if self.final_evaluator is not None:
            self.final_evaluator.update_protein(self.pocket_p)

        self.mol_table = pd.DataFrame(columns=
            [
                'iteration',
                'ligand_name',
                'selected_for_fragmentation',
                'passed_filters',
                'representation.smiles',
                'parent_smiles',
                'mol',
            ]
        )
        self.fragment_tree = FragmentTree(self.starting_frags)
        self.smiles = set()

    def setup_evaluators(self, itergen_params):
        set_default(itergen_params, 'fragment_filtering_criterion', 'basic_filter')
        set_default(itergen_params, 'filtering_criterion_global', 'basic_filter')
        set_default(itergen_params, 'adaptive_thresholds_global', {
            'uncertainty.mean_uncertainty': (DEFAULT_UNCERTAINTY_THRESHOLD, 'min'),
        })
        self.adaptive_thresholds_global = namespace_to_dict(itergen_params.adaptive_thresholds_global)
        set_default(itergen_params, 'filter_criterion_path', None)
        set_default(itergen_params, 'local_filtering_evaluators', ['posebusters_local'])
        if isinstance(itergen_params.local_filtering_evaluators, str):
            itergen_params.local_filtering_evaluators = itergen_params.local_filtering_evaluators.split(',')

        # final filtering (optional)
        set_default(itergen_params, 'filter_results', False)
        set_default(itergen_params, 'filtering_criterion_final', None)
        self.filter_results = itergen_params.filter_results
        assert not self.filter_results or itergen_params.filtering_criterion_final is not None, \
            'Final filtering criterion required if filter_results is True'

        # evaluator configs
        set_default(itergen_params, 'frag_filtering_constraints_path', None)
        set_default(itergen_params, 'pb_conf', 'dock')
        set_default(itergen_params, 'pb_conf_local', 'dock')
        set_default(itergen_params, 'reference_mols_validity3d', None)
        set_default(itergen_params, 'interactions', None)
        set_default(itergen_params, 'uncertainty_threshold', None)
        set_default(itergen_params, 'threshold_quantile', 0.5)
        set_default(itergen_params, 'max_unsatisfied_bonds', 4)
        set_default(itergen_params, 'gnina', 'gnina')
        set_default(itergen_params, 'reduce', None)
        self.reduce = itergen_params.reduce
        self.uncertainty_threshold = itergen_params.uncertainty_threshold
        self.threshold_quantile = itergen_params.threshold_quantile
        self.fix_uncertainty_threshold = isinstance(self.uncertainty_threshold, float)
        self.max_unsatisfied_bonds = itergen_params.max_unsatisfied_bonds

        if itergen_params.frag_filtering_constraints_path is not None:
            frag_constraints = json.load(open(itergen_params.frag_filtering_constraints_path))
        else:
            frag_constraints = None

        logging.info('Setting up evaluators')
        if itergen_params.reduce is not None:
            executable = shutil.which(str(itergen_params.reduce))
            if executable is None:
                raise FileNotFoundError(f'Reduce executable not found: {itergen_params.reduce}')
            itergen_params.reduce = Path(executable)
        reference = itergen_params.reference_mols_validity3d
        if reference is not None:
            reference = Path(reference)
            if not reference.is_file():
                raise FileNotFoundError(f'Validity3D reference SDF not found: {reference}')
            itergen_params.reference_mols_validity3d = reference
        if 'validity3d_local' in itergen_params.local_filtering_evaluators and reference is None:
            raise ValueError('validity3d_local requires reference_mols_validity3d')
        if 'interactions_local' in itergen_params.local_filtering_evaluators and itergen_params.reduce is None:
            raise ValueError('interactions_local requires a Reduce executable')
        if itergen_params.interactions is not None:
            if isinstance(itergen_params.interactions, str):
                itergen_params.interactions = itergen_params.interactions.split(',')
            self.interactions = itergen_params.interactions
        else:
            logging.info('No interactions specified, using default interactions')
            self.interactions = DEFAULT_INTERACTION_LIST
        self.filter_config = FilterConfig(itergen_params.filter_criterion_path)
        for criterion in (itergen_params.fragment_filtering_criterion,
                          itergen_params.filtering_criterion_global,
                          itergen_params.filtering_criterion_final):
            if criterion is not None and reference is None and 'validity3d' not in self.filter_config.get_exclude_evaluators(criterion):
                raise ValueError(f'{criterion} requires reference_mols_validity3d')
        logging.info(f'filtering criterion (local): {itergen_params.local_filtering_evaluators}')
        logging.info(f'filtering criterion (global): {itergen_params.filtering_criterion_global}')
        logging.info(f'fragment filtering criterion: {itergen_params.fragment_filtering_criterion}')
        
        exclude_evaluators_global = self.filter_config.get_exclude_evaluators(itergen_params.filtering_criterion_global)
        exclude_evaluators_frag = self.filter_config.get_exclude_evaluators(itergen_params.fragment_filtering_criterion)
        exclude_evaluators_local = set(LOCAL_EVALUATORS) - \
                                   set(itergen_params.local_filtering_evaluators)
        self.filtering_criterion_global = itergen_params.filtering_criterion_global
        self.local_filtering_evaluators = itergen_params.local_filtering_evaluators
        self.fragment_filtering_criterion = itergen_params.fragment_filtering_criterion
        self.filtering_criterion_final = itergen_params.filtering_criterion_final
        logging.info('Setting up fragment evaluator')
        eval_params_frag = {
            'pb_conf': itergen_params.pb_conf, 
            'exclude_evaluators': exclude_evaluators_frag,
            'gnina': itergen_params.gnina,
            'reduce': itergen_params.reduce,
            'reference_mols_validity3d': itergen_params.reference_mols_validity3d,
            'protein': self.pocket_p,
        }
        self.frag_evaluator = FullEvaluator(**eval_params_frag)
        logging.info('Setting up molecule prefilter evaluators')
        eval_params_global = eval_params_frag.copy()
        eval_params_global.update({
            'exclude_evaluators': exclude_evaluators_global,
        })
        self.global_evaluator = FullEvaluator(**eval_params_global)
        logging.info('Setting up local evaluators')
        self.local_evaluator = FullLocalEvaluator(
            pb_conf=itergen_params.pb_conf_local,
            reduce=itergen_params.reduce,
            interaction_list=self.interactions,
            residue_constraints=frag_constraints,
            reference_mols_validity3d=itergen_params.reference_mols_validity3d,
            exclude_evaluators=exclude_evaluators_local,
            protein=self.pocket_p,
        )
        if self.filtering_criterion_final is not None:
            exclude_evaluators_final = self.filter_config.get_exclude_evaluators(itergen_params.filtering_criterion_final)
            logging.info('Setting up final evaluator')
            final_eval_params = eval_params_global.copy()
            final_eval_params['exclude_evaluators'] = exclude_evaluators_final
            self.final_evaluator = FullEvaluator(**final_eval_params)
        else:
            self.final_evaluator = None

    ##################################### Sampling #####################################
    def prepare_dataset(self):
        ligands_prepared = []
        for frag_node in self.fragment_tree.node_list:
            if hasattr(frag_node, 'is_terminal') and frag_node.is_terminal:
                continue
            starting_fragment = frag_node.mol
            if starting_fragment is None:
                ligand = self.init_ligand.copy()
                ligand['fragments'] = torch.zeros_like(ligand['mask']).long()
                ligand['known_x'] = torch.zeros_like(ligand['mask']).bool()
                ligand['known_h'] = torch.zeros_like(ligand['mask']).bool()
                ligand['known_e'] = torch.zeros_like(ligand['bond_mask']).bool()
            else:
                try:
                    Chem.SanitizeMol(starting_fragment)
                except Exception as e:
                    logging.debug(f'Failed to sanitize fragment {frag_node.idx}: {e}')
                    continue
                if frag_node.processed_ligand is None:
                    ligand = prepare_ligand(
                        starting_fragment, config=self.featurization_config,
                    )
                    frag_node.processed_ligand = ligand
                else:
                    ligand = frag_node.processed_ligand
                ligand['fragments'] = torch.zeros_like(ligand['mask']).to(INT_TYPE)
                ligand['known_x'] = torch.ones_like(ligand['mask']).bool()
                ligand['known_h'] = torch.ones_like(ligand['mask']).bool()
                ligand['known_e'] = torch.ones_like(ligand['bond_mask']).bool()

            ligand['name'] = f'ligand_{frag_node.idx}'
            ligand['affinity'] = 0.0
            ligands_prepared.append((ligand, frag_node))
        if len(ligands_prepared) == 0:
            raise ValueError('No fragments to sample from')
        return ligands_prepared

    def get_dataloader(self, ligands, allocations):
        dataset = []
        for l, alloc in zip(ligands, allocations):
            dataset.extend([{'ligand': l[0], 'pocket': self.pocket} for _ in range(alloc)])

        logging.debug(f'len(dataset)={len(dataset)}, len(ligands)={len(ligands)}, sum(allocations)={sum(allocations)}')
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.model.batch_size, 
            collate_fn=partial(ProcessedDataset.collate_fn, ligand_transform=None),
            pin_memory=True
        )
        return dataloader
    
    def sample_dataloader(self, dataloader):
        sampled_mols = []
        frag_idx_counter = Counter()
        for data in tqdm(dataloader, total=len(dataloader), desc='Sampling'):
            new_data = {
                'ligand': TensorDict(**data['ligand']).to(self.model.device),
                'pocket': TensorDict(**data['pocket']).to(self.model.device),
            }
            starting_frag_idxs = [int(name.split('_')[1]) for name in new_data['ligand']['name']]
            rdmols, _, _ = self.model.sample(
                new_data,
                n_samples=1,
                timesteps=self.model.T_sampling,
                num_nodes=self.molecule_size,
            )
            self.global_sampling_counter += 1
            for idx, mol in enumerate(rdmols):
                mol.SetProp('frag_idx', str(starting_frag_idxs[idx]))
                frag_idx_counter[starting_frag_idxs[idx]] += 1
            sampled_mols.extend(rdmols)
        return sampled_mols, frag_idx_counter

    ##################################### Fragmentation #####################################
    def update_tree(self, select_table, frag_idx_counter):
        # insert filtered mols
        for i, row in select_table.iterrows():
            mol = row['mol']
            frag_idx = mol.GetProp('frag_idx')
            node = self.fragment_tree.get_node_from_index(frag_idx)
            if node is None:
                logging.warning(f'Fragment not found in tree: {frag_idx}')
                continue
            select_table.loc[i, 'parent_smiles'] = Chem.MolToSmiles(node.mol) if node.mol is not None else ''
            sample_num = ['s' + str(len(node.filtered_mols))] if row.passed_filters else ['f' + str(i)]
            tree_trace = '>'.join([str(n.idx) for n in node.get_path()] + sample_num)
            mol.SetProp('_Name', f'ligand_{tree_trace}')
            select_table.loc[i, 'ligand_name'] = tree_trace
            if row.passed_filters:
                node.filtered_mols.append(mol)

        # update sampling counter for nodes
        for frag_idx, count in frag_idx_counter.items():
            node = self.fragment_tree.get_node_from_index(frag_idx)
            if node is not None:
                node.sampling_counter += count
        return select_table
                
    def allocate_samples_to_fragments(self, ligands_prepared, budget=None):
        if budget is None:
            budget = self.n_samples_per_iter
        root_alloc = (0,0)
        if self.min_root_sample_proportion > 0:
            root_samples = int(budget * self.min_root_sample_proportion)
            budget = max(0, budget - root_samples)
            root_idx = 0
            for i, (l, f) in enumerate(ligands_prepared):
                if f.parent is None:
                    root_idx = i
                    break
            root_alloc = (root_idx, root_samples)
        if self.use_ucb:
            # Computing upper confidence bound
            sampling_successes = [len(f.filtered_mols) for l,f in ligands_prepared]
            sampling_calls = [f.sampling_counter for l,f in ligands_prepared]
            parent_calls = [
                f.parent.total_sampling_counter if f.parent is not None
                else f.total_sampling_counter
                for l,f in ligands_prepared
            ]

            ucb_scores = np.zeros(len(ligands_prepared))
            for i in range(len(ligands_prepared)):
                success_rate = sampling_successes[i] / sampling_calls[i] if sampling_calls[i] > 0 else 0
                exploration_term = self.exploration_weight * np.sqrt(
                    np.log(parent_calls[i] + 1) / (sampling_calls[i] + 1)
                )
                ucb_scores[i] = success_rate + exploration_term

            # Allocating budget proportionally to UCB scores
            if np.sum(ucb_scores) > 0:
                scaled_scores = ucb_scores / np.sum(ucb_scores) * budget
            else:
                scaled_scores = np.ones(len(ligands_prepared)) * budget / len(ligands_prepared)
            int_allocations = np.floor(scaled_scores).astype(int)

            remaining_budget = budget - sum(int_allocations)
            if remaining_budget > 0:
                fractional_parts = scaled_scores - int_allocations
                top_indices = np.argsort(-fractional_parts)[:remaining_budget]
                int_allocations[top_indices] += 1

            for i, alloc in enumerate(int_allocations):
                logging.debug(f'Fragment {i}: {ligands_prepared[i][1].smiles}, UCB={ucb_scores[i]:.3f}, alloc {alloc} samples')
        else:
            int_allocations = np.ones(len(ligands_prepared)) * (budget // len(ligands_prepared))
            remaining_budget = budget % len(ligands_prepared)
            if remaining_budget > 0:
                top_indices = np.argsort(-np.random.rand(len(ligands_prepared)))[:remaining_budget]
                int_allocations[top_indices] += 1
            int_allocations = int_allocations.astype(int)
        
        if root_alloc[1] > 0:
            logging.debug(f'update: alloc {root_alloc[1]} additional samples to tree root')
            int_allocations[root_alloc[0]] += root_alloc[1]
        logging.info(f'Allocating {budget+root_alloc[1]} samples to {len(ligands_prepared)} fragments')

        assert sum(int_allocations) == budget + root_alloc[1], f'Allocations do not sum to budget: {int_allocations}'
        return int_allocations

    def fragment_mols(self, select_table):
        for i, row in select_table.loc[select_table.selected_for_fragmentation].iterrows():
            mol = row['mol']
            logging.debug(f'Fragmenting molecule {Chem.MolToSmiles(mol)}')
            for atom in mol.GetAtoms():
                atom.SetProp('_InitialIndex', str(atom.GetIdx()))
            
            if self.fragmentation_method == 'brics':
                frags, fragment_mask = get_fragments_brics(mol)
            elif self.fragmentation_method == 'no_fragmentation':
                frags = [None]
                fragment_mask = np.zeros_like(mol.GetAtoms())
            else:
                raise NotImplementedError(f'Fragmentation method {self.fragmentation_method} not implemented')

            select_table.at[i, 'frags'] = frags
            select_table.at[i, 'fragment_mask'] = fragment_mask
        return select_table

    def process_fragments(self, select_table, 
                          check_substruct_match=True,
                          drop_random=None):
        selected_fragmols = []
        frag_smiles = {}
        has_frags = select_table['frags'].apply(lambda x: x is not None and any(x))
        sel_rows = select_table.loc[select_table.selected_for_fragmentation & has_frags]
        for i, frag_row in sel_rows.iterrows():
            mol = frag_row['mol']
            frags = frag_row['frags']
            fragment_mask = frag_row['fragment_mask']
            
            frag_table = filter_mols(
                frags, 
                self.pocket_p,
                evaluator=self.frag_evaluator,
                criterion=self.fragment_filtering_criterion,
                filter_manager=self.filter_config
            )
            if isinstance(drop_random, float):
                # drop random fragments
                drop_idxs = np.random.choice(frag_table.index, int(drop_random * len(frag_table)), replace=False)
                frag_table.loc[drop_idxs, 'passed_filters'] = False
            filt_idxs = frag_table.loc[frag_table.passed_filters].idx
            if len(filt_idxs) == 0: continue
            fragmol = filter_fragments(
                mol,
                fragment_mask,
                filt_idxs,
                fixed_frags=self.starting_frags
            )
            if fragmol is None:
                continue
            if fragmol.GetNumAtoms() == 0:
                continue
            if frags[-1] is not None:
                # add properties from the last fragment
                for prop in frags[-1].GetPropNames(includePrivate=True):
                    fragmol.SetProp(prop, frags[-1].GetProp(prop))
            smi = Chem.MolToSmiles(fragmol)
            if smi in frag_smiles.values() or smi in self.fragment_tree.frag_smiles:
                continue
            selected_fragmols.append(fragmol)
            frag_smiles[fragmol] = smi

        added_frags = []
        avg_mol_size = self.mol_table['mol'].apply(lambda x: x.GetNumAtoms()).mean()
        for fragmol in selected_fragmols:
            parent_frag = fragmol.GetProp('frag_idx')
            parent = self.fragment_tree.get_node_from_index(parent_frag)
            if parent is None:
                logging.warning(f'Fragment not found in tree: {parent_frag}')
                continue
            if check_substruct_match and parent.mol is not None and \
                (
                    not fragmol.HasSubstructMatch(parent.mol) or \
                    fragmol.GetNumAtoms() <= parent.mol.GetNumAtoms()
                ):
                continue
            if Chem.MolToSmiles(fragmol) in self.fragment_tree.frag_smiles:
                continue
            size = fragmol.GetNumAtoms()
            size_threshold = avg_mol_size * self.restrict_fragment_size_threshold if \
                                isinstance(self.restrict_fragment_size_threshold, float) else np.inf
            if not is_nan(size_threshold) and size > size_threshold:
                logging.debug(
                    f'Skipping fragment {Chem.MolToSmiles(fragmol)} due to size (avg mol size: {avg_mol_size:.2f}, fragment size: {size} > {size_threshold:.2f})')
                continue
            tree_trace = '>'.join([str(n.idx) for n in parent.get_path()] + \
                                  [str(len(self.fragment_tree.node_list))])
            fragmol.SetProp('_Name', f'frag_{tree_trace}')
            n = self.fragment_tree.add_child(fragmol, parent)
            if len(n.get_path()) > self.max_tree_depth:
                n.is_terminal = True
            added_frags.append(n)
            logging.debug(f'Added fragment {Chem.MolToSmiles(fragmol)} to tree')
        logging.info(f'Added {len(added_frags)} fragments')
        return added_frags

    ##################################### Filtering #####################################
    def filter_fragments_local(self, select_table):
        # local filtering
        for i, frag_row in select_table.loc[select_table.selected_for_fragmentation].iterrows():
            mol = frag_row['mol']
            fragment_mask = frag_row['fragment_mask']
            frags = frag_row['frags']
            if frags is None or not any(frags): continue

            # local evaluation
            eval_res_all = {}
            eval_res_per_frag = {}
            updated_frags = frags.copy()
            for evaluator_id in self.local_filtering_evaluators:
                if not any(updated_frags):
                    logging.debug(f'No fragments left for local evaluation with {evaluator_id}')
                    break
                part_evaluator = self.local_evaluator.get_evaluator(evaluator_id)
                eval_res = part_evaluator(mol, self.pocket_p)
                eval_res.update(aggregate_interaction_profile(eval_res, self.interactions))
                eval_res_all.update(eval_res)
                eval_res_per_frag = aggregate_local_eval_per_fragment(
                    eval_res, fragment_mask,
                    uncertainty_threshold=self.uncertainty_threshold
                )
                updated_frags = [
                    None if not eval_res_frag
                    else updated_frags[frag_idx] 
                        for frag_idx, eval_res_frag in eval_res_per_frag['local_eval.all'].items()
                ]
            select_table.at[i, 'frags'] = updated_frags

            # aggregating results
            eval_res_global = aggregate_local_eval_per_fragment(
                eval_res_all, np.zeros_like(fragment_mask),
                uncertainty_threshold=self.uncertainty_threshold
            )
            eval_res_global = {k: v[0] for k,v in eval_res_global.items()}
            
            # inserting results into table
            if 'mean_uncertainty_local' in eval_res_global:
                eval_res_global['mean_uncertainty'] = eval_res_global.pop('mean_uncertainty_local')
            eval_res_global['local_eval.all'] = True
            for k,v in eval_res_global.items():
                if not k in select_table.columns:
                    select_table[k] = None
                    select_table[k] = select_table[k].astype(object)
                select_table.at[i, k] = v
                if 'pass_' in k and not k == 'pass_unsatisfied_bonds':
                    eval_res_global['local_eval.all'] &= v
                elif k == 'sum_unsatisfied_bonds' and 'sum_interactions_local' in eval_res_global:
                    # exception for unsatisfied bonds, should be less than num interactions
                    crit = eval_res_global[k] <= eval_res_global['sum_interactions_local']
                    eval_res_global['local_eval.all'] &= crit
                elif k == 'sum_unsatisfied_bonds' and not 'sum_interactions_local' in eval_res_global:
                    eval_res_global['local_eval.all'] &= v <= self.max_unsatisfied_bonds
            if not eval_res_global['local_eval.all']:
                select_table.loc[i, 'passed_filters'] = False
                select_table.loc[i, 'local_eval.all'] = False
            if 'mean_uncertainty_local' in eval_res_per_frag:
                if not 'mean_uncertainty_local' in select_table.columns:
                    select_table['mean_uncertainty_local'] = None
                    select_table['mean_uncertainty_local'] = select_table['mean_uncertainty_local'].astype(object)
                local_uncert_res = list(eval_res_per_frag['mean_uncertainty_local'].values())
                select_table.at[i, 'mean_uncertainty_local'] = local_uncert_res
        return select_table

    def update_thresholds(self):
        for criterion in self.adaptive_thresholds_global:
            if not criterion in self.mol_table.columns: continue
            criterion_values = list(self.mol_table[criterion].tail(
                self.score_history_window,
            ).dropna())
            if len(criterion_values) == 0: continue
            # get quantile
            if self.adaptive_thresholds_global[criterion][1] == 'min':
                new_threshold = np.quantile(criterion_values, self.threshold_quantile)
                if is_nan(new_threshold): continue
                self.adaptive_thresholds_global[criterion] = (new_threshold, 'min')
                new_bounds = (-np.inf, new_threshold)
            elif self.adaptive_thresholds_global[criterion][1] == 'max':
                new_threshold = np.quantile(criterion_values, 1 - self.threshold_quantile)
                if is_nan(new_threshold): continue
                self.adaptive_thresholds_global[criterion] = (new_threshold, 'max')
                new_bounds = (new_threshold, np.inf)
            else:
                logging.warning(f'Invalid threshold type for {criterion}')
                continue

            self.filter_config.update_filter_threshold(
                criterion, 
                new_bounds
            )
        if 'mean_uncertainty_local' in self.mol_table.columns and not self.fix_uncertainty_threshold:
            last_uncertainty = self.mol_table.loc[self.mol_table.selected_for_fragmentation].mean_uncertainty_local.dropna()
            last_uncertainty = last_uncertainty.tail(
                self.score_history_window,
            )
            last_uncertainty = [item 
                for sublist in last_uncertainty for item in sublist 
                if not is_nan(item)
            ]
            if len(last_uncertainty) > 0:
                last_uncertainty = np.quantile(last_uncertainty, self.threshold_quantile)
                self.uncertainty_threshold = last_uncertainty
                logging.info(f'Updated uncertainty threshold to <{self.uncertainty_threshold:.3f}')


    ##################################### Controlled Sampling #####################################
    def run_iteration(self, iteration):
        # Preparing ligands for current iteration
        self.fragment_tree.update_total_sampling_counter()
        ligands = self.prepare_dataset()
        allocations = self.allocate_samples_to_fragments(ligands)
        dataloader = self.get_dataloader(ligands, allocations)
        
        self.update_thresholds()

        # Sampling and filtering
        smpl_time = time.time()
        sampled_mols, frag_idx_counter = self.sample_dataloader(dataloader)
        smpl_time = time.time() - smpl_time

        select_time = time.time()
        select_table = filter_mols(
            sampled_mols,
            self.pocket_p,
            smiles=self.smiles,
            evaluator=self.global_evaluator,
            criterion=self.filtering_criterion_global,
            select_unique=True,
            filter_manager=self.filter_config
        )
        select_table['selected_for_fragmentation'] = select_table['passed_filters']
        select_table['iteration'] = iteration
        select_table['sampling_time'] = smpl_time
        select_table['select_time'] = time.time() - select_time
        # insert additional columns
        select_table['frags'] = None
        select_table['fragment_mask'] = None
        select_table['frags'] = select_table['frags'].astype(object)
        select_table['fragment_mask'] = select_table['fragment_mask'].astype(object)

        fragment_time = time.time()
        select_table = self.fragment_mols(select_table)
        select_table['fragment_time'] = time.time() - fragment_time
        
        local_filt_time = time.time()
        select_table = self.filter_fragments_local(select_table)
        select_table['local_filtering_time'] = time.time() - local_filt_time
        select_table = self.update_tree(select_table, frag_idx_counter)

        # Fragment quality filtering and processing
        fragment_filtering_time = time.time()
        if iteration < self.max_sampling_iter - 1:
            self.process_fragments(
                select_table,
                check_substruct_match=False if self.drop_random is not None else True,
                drop_random=self.drop_random)
        select_table['fragment_filtering_time'] = time.time() - fragment_filtering_time
        
        self.global_mol_counter += len(select_table.loc[select_table.passed_filters])
        logging.info(f'Found {len(select_table.loc[select_table.passed_filters])} molecules in iteration {iteration}')
        self.smiles.update(select_table['representation.smiles'].dropna())
        return select_table

    def sample(self):
        utils.disable_rdkit_logging()
        self.global_sampling_counter = 0
        self.global_mol_counter = 0

        pbar = tqdm(total=self.stop_after_n_mols, desc='Selected mols') if \
                    self.stop_after_n_mols is not None else \
               tqdm(total=self.max_sampling_iter, desc='Sampling iterations')
        for iteration in range(self.max_sampling_iter):
            if self.stop_after_n_mols is not None and self.global_mol_counter >= self.stop_after_n_mols:
                logging.info(f'Stopping after having generated {self.global_mol_counter} molecules with {self.global_sampling_counter} sampling calls.')
                break
            
            total_time_per_iter = time.time()
            select_table = self.run_iteration(iteration)
            select_table['total_time_per_iter'] = time.time() - total_time_per_iter
            self.mol_table = pd.concat([self.mol_table, select_table], ignore_index=True)

            pbar.update(1 if self.stop_after_n_mols is None else len(select_table.loc[select_table.passed_filters]))
        pbar.close()

        # Adding ligand index to mols that passed filters
        self.mol_table.insert(0, 'ligand_idx', self.mol_table.index)
        for i, row in self.mol_table.iterrows():
            self.mol_table.loc[i, 'mol'].SetProp('ligand_idx', str(row['ligand_idx']))

        logging.info(f'Called sampling {self.global_sampling_counter} times, generated {self.global_mol_counter} filtered molecules')

    ##################################### Results #####################################
    def retrieve_results(self, return_format='samples'):
        if self.filter_results:
            return_table = self.filter_final_mols()
        else:
            return_table = self.mol_table.copy()

        if not return_table.empty:
            return_table = return_table.dropna(axis=1, how='all')
        return_table.drop(columns=['fragment_mcs_results'], inplace=True, errors='ignore')

        if return_format == 'samples':
            return list(return_table.mol)
        elif return_format == 'table':
            return return_table
        else:
            raise ValueError(f'Invalid return format: {return_format}')

    def get_fragment_stats(self):
        stats = []
        for node in self.fragment_tree.node_list:
            row = {
                'node_idx': node.idx,
                'tree_trace': '>'.join([str(n.idx) for n in node.get_path()]),
                'smiles': node.smiles,
                'num_atoms': node.mol.GetNumAtoms() if node.mol is not None else None,
                'mols_generated': len(node.filtered_mols),
                'n_sampled': node.sampling_counter,
                'success_rate': len(node.filtered_mols) / node.sampling_counter if node.sampling_counter > 0 else 0,
            }
            stats.append(row)
        stats = pd.DataFrame(stats)
        return stats
    
    def filter_final_mols(self):
        table = self.mol_table.loc[self.mol_table.passed_filters].copy()
        
        filtered_mols = list(table.mol)
        filtered_idxs = list(table.ligand_idx)
        select_table = filter_mols(
            filtered_mols, 
            self.pocket_p, 
            evaluator=self.final_evaluator,
            criterion=self.filtering_criterion_final,
            filter_manager=self.filter_config,
        )
        filtered_idxs_sel = [filtered_idxs[i] for i in select_table.idx]
        select_table['ligand_idx'] = filtered_idxs_sel
        select_table = select_table.loc[select_table.passed_filters]

        # Final evaluations replace preliminary metrics, which may be missing for products.
        replaced_columns = [col for col in select_table.columns
                            if col != 'ligand_idx' and col in self.mol_table.columns]
        return pd.merge(self.mol_table.drop(columns=replaced_columns), select_table,
                        on='ligand_idx', how='inner')
