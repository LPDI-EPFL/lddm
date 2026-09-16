import sys
import os
import warnings
import logging
import time
import json
import pickle
from pathlib import Path
from argparse import Namespace
from operator import xor

from rdkit import Chem
import torch
import numpy as np
import pandas as pd

from collections import Counter
from tqdm import tqdm

basedir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(basedir))
warnings.filterwarnings("ignore")

from lddm.constants import INT_TYPE, atom_encoder, bond_encoder
from lddm.utils import set_default
from lddm.model import samplers
from lddm.data.data_utils import TensorDict, prepare_ligand
from lddm.inference.programmable_generation import ProgrammableGeneration
from lddm.inference.programmable_design_utils import *
from lddm.reactions.process_reactions import ReactionsProcessor
from lddm.reactions.process_reactions_enamine import ReactionsProcessorEnamine
from lddm.reactions.reaction_utils import get_react_trace_building_block

class SynthesizableGeneration(ProgrammableGeneration):
    ##################################### Initialization #####################################
    def __init__(self, model,
                 sampling_params: Namespace,
                 itergen_params: Namespace,
                 docking_params: Namespace):
        set_default(itergen_params, 'fp_size', 2048)
        set_default(itergen_params, 'fragmentation_method', 'building_blocks')
        set_default(itergen_params, 'max_tree_depth', 3)
        set_default(itergen_params, 'min_root_sample_proportion', 0.25)
        set_default(itergen_params, 'min_product_set_size', 1)
        set_default(itergen_params, 'local_filtering_evaluators', [])
        set_default(itergen_params, 'filtering_criterion_global', 'basic_filter')
        set_default(itergen_params, 'building_blocks_path', None)
        set_default(itergen_params, 'reaction_path', None)
        set_default(itergen_params, 'reaction_path_enamine', None)
        set_default(itergen_params, 'reaction_to_compound_path', None)
        set_default(itergen_params, 'return_all_samples', False)
        assert itergen_params.fragmentation_method == 'building_blocks', 'Only building blocks fragmentation allowed'
        assert itergen_params.building_blocks_path is not None, \
            'Building blocks path must be provided for synthesizable sampling'
        assert xor(
            itergen_params.reaction_path is not None,
            itergen_params.reaction_path_enamine is not None
        ), 'Only one reaction path can be provided (custom or Enamine)'
        self.fp_size = itergen_params.fp_size
        self.building_blocks = None
        self.return_all_samples = itergen_params.return_all_samples
        self.min_product_set_size = itergen_params.min_product_set_size

        set_default(docking_params, 'use_docking', False)
        self.use_docking = docking_params.use_docking
        if self.use_docking:
            logging.info('Setting up docking')
            set_default(docking_params, 'n_docking_samples_per_iter', 256)
            set_default(docking_params, 'max_selected_docking_samples', 64)
            set_default(docking_params, 'min_dock_fp_similarity', 0.25)
            set_default(docking_params, 'min_dock_match_atoms_frac', 0.40)
            set_default(docking_params, 'max_mols_in_docking_pool',
                int(docking_params.n_docking_samples_per_iter / 4))
            set_default(docking_params, 'dock_completed_only', False)
            set_default(docking_params, 'sampler', 'HeunSampler')
            set_default(docking_params, 'n_sampling_steps', 50)

            self.n_docking_samples_per_iter = docking_params.n_docking_samples_per_iter
            self.min_dock_fp_similarity = docking_params.min_dock_fp_similarity
            self.min_dock_match_atoms_frac = docking_params.min_dock_match_atoms_frac
            self.max_selected_docking_samples = docking_params.max_selected_docking_samples
            self.max_mols_in_docking_pool = docking_params.max_mols_in_docking_pool
            self.n_docking_steps = docking_params.n_sampling_steps
            self.dock_completed_only = docking_params.dock_completed_only

        super().__init__(model, sampling_params, itergen_params)

        if self.use_docking:
            self.docking_sampler = getattr(samplers, docking_params.sampler)(self.model)

    def setup_input(self, ligand_p, pocket_p, starting_frag_p=None):
        assert starting_frag_p is None, 'Starting fragments are not supported for synthesizable sampling'
        super().setup_input(ligand_p, pocket_p)

        self.fragment_tree = SynthonTree(self.fragment_tree.root.mol)
        self.run_reactions()

    def setup_evaluators(self, itergen_params):
        super().setup_evaluators(itergen_params)
        paths_to_check = [
            'reaction_to_compound_path',
            'building_blocks_path',
            'reaction_path',
            'reaction_path_enamine',
        ]
        for attr in paths_to_check:
            path = getattr(itergen_params, attr)
            if path is not None:
                path = Path(path)
                setattr(itergen_params, attr, path)
                if not path.is_file():
                    raise FileNotFoundError(f'{attr} file not found: {path}')

        assert itergen_params.building_blocks_path is not None, \
            'Building blocks must be provided for synthesizable sampling'
        if not itergen_params.building_blocks_path.exists() or not itergen_params.building_blocks_path.suffix == '.pkl':
            raise ValueError(f'Building blocks file not found or invalid format: {itergen_params.building_blocks_path}')

        use_enamine_reactions = itergen_params.reaction_path_enamine is not None
        self.use_enamine_reactions = use_enamine_reactions
        if not use_enamine_reactions:
            logging.info('Using custom reactions for synthesis')
            with open(itergen_params.building_blocks_path, "rb") as f:
                self.building_blocks = pickle.load(f)
                if not 'synthon_smiles' in self.building_blocks.columns:
                    self.building_blocks['synthon_smiles'] = self.building_blocks['smiles']
                if 'completed' not in self.building_blocks.columns:
                    self.building_blocks['completed'] = False
                self.building_blocks = self.building_blocks.drop_duplicates(subset=['synthon_smiles'])
            logging.info(f'Loaded {len(self.building_blocks)} building blocks from {itergen_params.building_blocks_path}')
            reactions_p = Path(itergen_params.reaction_path)
            if not reactions_p.exists() or not reactions_p.suffix == '.json':
                raise ValueError(f'Reactions file not found or invalid format: {reactions_p}')
            with open(reactions_p) as f:
                reactions = json.load(f)
            self.reactions_processor = ReactionsProcessor(
                reactions,
                self.building_blocks,
                fp_size=self.fp_size
            )
            self.reactions_processor.load_reactant_to_building_blocks(
                itergen_params.reaction_to_compound_path
            )
            logging.info(f'Loaded {len(self.reactions_processor.reactions)} reactions from {reactions_p}')
        else:
            logging.info('Using Enamine reactions for synthesis')
            reactions_p = Path(itergen_params.reaction_path_enamine)
            if not reactions_p.exists():
                raise ValueError(f'Enamine reactions file not found: {reactions_p}')
            reactions_df = pd.read_csv(reactions_p, sep='\t')
            with open(itergen_params.building_blocks_path, "rb") as f:
                synthon_df = pickle.load(f)
            self.reactions_processor = ReactionsProcessorEnamine(
                reactions_df,
                synthon_df,
                fp_size=self.fp_size,
            )
            self.building_blocks = self.reactions_processor.processed_synthon_df
            logging.info(f'Loaded {len(self.building_blocks)} unique synthons')


    ##################################### Reactions #####################################
    def run_reactions(self):
        available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
        max_workers = max(available_cpus // 8, 1)
        total_products = 0
        default_df = pd.DataFrame(columns=['fp', 'react_trace', 'smiles', 'synthon_smiles', 'checked', 'completed'])

        for node in self.fragment_tree.node_list:
            if node.product_set is not None:
                # fragment already processed
                continue
            node.product_set = default_df.copy()
            # empty root node
            if node.mol is None:
                node.product_set = self.building_blocks.copy()
                node.product_set['checked'] = True
                node.product_set['react_trace'] = node.product_set['synthon_smiles'].apply(get_react_trace_building_block)
                node.is_terminal = False
                logging.info(f'Root node initialized with {len(node.product_set)} building blocks')
                continue
            elif node.is_terminal or node.completed:
                node.product_set = default_df.copy()
                continue

            if not hasattr(node, 'react_trace'):
                node.react_trace = None
            node.product_set = self.reactions_processor.run_reactions_for_node(
                node.synthon_smiles,
                node.smiles,
                node.react_trace,
                num_workers=max_workers
            )
            
            if len(node.product_set) < self.min_product_set_size:
                logging.info(f'Fragment {node.idx} ({node.smiles}) has too few products ({len(node.product_set)})')
                node.is_terminal = True
                node.product_set = default_df.copy()
            else:
                node.is_terminal = False
            total_products += len(node.product_set)

        logging.info('Checking reaction products')
        self.add_synthon_nodes()
        logging.info(f'Generated {total_products} new building blocks')

    def add_synthon_nodes(self):
        for node in tqdm(self.fragment_tree.node_list, desc='Checking reaction products'):
            if node.product_set is None or node.is_terminal:
                continue

            if 'checked' not in node.product_set.columns:
                node.product_set['checked'] = False
            node.product_set['checked'] = node.product_set['checked'].astype(bool).fillna(False)
            prod_set_to_check = node.product_set.loc[~node.product_set['checked']]
            if len(prod_set_to_check) == 0: continue
            assert node.parent is not None, \
                'Root node needs to be initialized before checking reaction products'
            for _, prod_set in prod_set_to_check.groupby('smiles'):
                product = prod_set.iloc[0]
                # find mcs
                mcs = Chem.rdFMCS.FindMCS(
                    [Chem.MolFromSmiles(product['smiles']), node.mol],
                    completeRingsOnly=True,
                    timeout=5,
                )
                if mcs.numAtoms == 0 or mcs.canceled:
                    logging.warning(f'MCS search for {product["smiles"]} and {node.smiles} was canceled')
                    node.product_set.drop(prod_set.index, inplace=True)
                    continue
                elif mcs.numAtoms == node.mol.GetNumAtoms():
                    node.product_set.loc[prod_set.index, 'checked'] = True
                elif mcs.numAtoms < node.mol.GetNumAtoms():
                    logging.debug(f'Fragment {node.smiles} does not match product {product["smiles"]}, generating new frags')
                    node.product_set.drop(prod_set.index, inplace=True)
                    matches = node.mol.GetSubstructMatches(Chem.MolFromSmarts(mcs.smartsString))
                    if len(matches) == 0: continue
                    match_atmos = matches[0]
                    rwmol = Chem.RWMol(node.mol)
                    not_present_idxs = [atm.GetIdx() for atm in node.mol.GetAtoms() if atm.GetIdx() not in match_atmos]
                    for atm_idx in sorted(not_present_idxs, reverse=True):
                        rwmol.RemoveAtom(atm_idx)
                    frag = rwmol.GetMol()
                    frag_smiles = Chem.MolToSmiles(frag)

                    # check if fragment is already present
                    n = self.fragment_tree.get_node_from_smiles(frag_smiles)
                    if n is not None:
                        logging.debug(f'Extending reactions for fragment {frag_smiles} from fragment {node.smiles}')
                        prod_set['checked'] = True
                        n.product_set = pd.concat([n.product_set, prod_set], ignore_index=True)
                    else:
                        logging.debug(f'Adding fragment {frag_smiles} from fragment {node.smiles}')
                        n = self.fragment_tree.add_child(
                            frag, node.parent,
                            synthon_smiles=node.synthon_smiles,
                            react_trace=node.react_trace,
                            completed=node.completed,
                        )
                        prod_set['checked'] = True
                        n.product_set = prod_set
        for node in self.fragment_tree.node_list:
            if node.product_set is None or node.is_terminal: continue
            if len(node.product_set) < self.min_product_set_size:
                logging.info(f'Fragment {node.idx} ({node.smiles}) has too few products ({len(node.product_set)}) after extracting synthons')
                node.is_terminal = True

    #################################### Molecule docking ###################################
    def prepare_mols_to_dock(self, select_table):
        sel_rows = select_table.dropna(subset=['fragment_mcs_results'])
        sel_rows = sel_rows.loc[sel_rows['fragment_mcs_results'].apply(lambda x: len(x) > 0)]
        mols_to_dock = []
        for i, row in sel_rows.iterrows():
            frag_idx = row['mol'].GetProp('frag_idx')
            # skip empty root fragment
            if self.fragment_tree.get_node_from_index(frag_idx).mol is None:
                continue

            mcs_results = row['fragment_mcs_results']
            for mcs in mcs_results:
                if mcs['similarity'] < self.min_dock_fp_similarity: continue
                db_smiles = mcs['db_smiles']
                if db_smiles in self.smiles: continue # skip docking of already sampled molecules
                if not mcs['completed'] and self.dock_completed_only:
                    continue
                db_mol = Chem.MolFromSmiles(db_smiles)

                mol_match_atoms = mcs['match_atoms']
                match_atoms_frac = len(mol_match_atoms) / mcs['db_mol_size']
                if match_atoms_frac < self.min_dock_match_atoms_frac:
                    # logging.debug(f'Skipping docking of {db_smiles} due to low match atoms fraction ({match_atoms_frac})')
                    continue
                mol_conformer = row['mol'].GetConformer()
                db_match_atoms = mcs['db_match_atoms']
                # MCS matches need not preserve extra bonds in the target. Both
                # ends of a new ring closure must not be frozen at unrelated positions.
                atom_map = dict(zip(db_match_atoms, mol_match_atoms))
                compatible = True
                for bond in db_mol.GetBonds():
                    begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    if begin not in atom_map or end not in atom_map:
                        continue
                    source_bond = row['mol'].GetBondBetweenAtoms(atom_map[begin], atom_map[end])
                    if source_bond is None or source_bond.GetBondType() != bond.GetBondType():
                        compatible = False
                        break
                if not compatible:
                    logging.debug(f'Skipping incompatible fixed-atom mapping for {db_smiles}')
                    continue
                # transfer coordinates from mol to db_mol
                fixed_coords = {
                    db_atm_idx: mol_conformer.GetAtomPosition(mol_atm_idx) \
                        for db_atm_idx, mol_atm_idx in zip(db_match_atoms, mol_match_atoms)
                }
                fixed_coords_idxs = list(fixed_coords.keys())
                # create conformer
                db_mol.AddConformer(Chem.Conformer(db_mol.GetNumAtoms()), assignId=True)
                for atm_idx, coords in fixed_coords.items():
                    db_mol.GetConformer().SetAtomPosition(atm_idx, coords)

                score = mcs['similarity'] + len(mol_match_atoms)/mcs['db_mol_size']
                mols_to_dock.append({
                    'mol': db_mol,
                    'score_match': score,
                    'fixed_coords': fixed_coords_idxs,
                    'react_trace': mcs['react_trace'],
                    'synthon_smiles': mcs['synthon_smiles'],
                    'completed': mcs['completed'],
                    'frag_idx': frag_idx,
                })
        return mols_to_dock

    def prepare_docking_dataset(self, mols_to_dock):
        mols_to_dock = sorted(mols_to_dock, key=lambda x: x['score_match'], reverse=True)
        ligands_prepared = []
        ligand_info = {}
        for j, mol_info in enumerate(mols_to_dock):
            if len(ligands_prepared) >= self.max_mols_in_docking_pool:
                logging.info(f'Reached maximum number of molecules in docking pool: {self.max_mols_in_docking_pool}')
                break
            mol = mol_info['mol']
            fixed_coords = mol_info['fixed_coords']
            react_trace = mol_info['react_trace']
            synthon_smiles = mol_info['synthon_smiles']
            completed = mol_info['completed']
            frag_node = self.fragment_tree.get_node_from_index(mol_info['frag_idx'])
            
            try:
                ligand = prepare_ligand(
                    mol, config=self.featurization_config,
                )
            except Exception as e:
                logging.warning(f'Failed to prepare ligand {Chem.MolToSmiles(mol)} for docking, {e}')
                continue
            ligand['fragments'] = torch.zeros_like(ligand['mask']).to(INT_TYPE)
            ligand['known_x'] = torch.zeros_like(ligand['mask']).bool()
            ligand['known_x'][fixed_coords] = True
            ligand['known_h'] = torch.ones_like(ligand['mask']).bool()
            ligand['known_e'] = torch.ones_like(ligand['bond_mask']).bool()
            ligand['name'] = f'docked_{frag_node.idx}_{j}'
            ligand_info[ligand['name']] = {
                'react_trace': react_trace,
                'synthon_smiles': synthon_smiles,
                'completed': completed,
                'frag_idx': frag_node.idx,
            }
            ligand['smiles'] = Chem.MolToSmiles(mol)
            ligand['affinity'] = 0.0
            ligands_prepared.append((ligand, frag_node))
        return ligands_prepared, ligand_info

    def allocate_docking_samples_to_fragments(self, docking_dataset):
        logging.info('Allocating docking samples to fragments')
        # create fake dataset with unique fragment nodes
        used_nodes = set()
        fake_dataset = []
        docking_dataset_dict = {}
        for lig, frag_node in docking_dataset:
            docking_dataset_dict.setdefault(frag_node.idx, []).append((lig, frag_node))
            if frag_node.idx in used_nodes: continue
            used_nodes.add(frag_node.idx)
            fake_dataset.append((None, frag_node))

        node_allocations = self.allocate_samples_to_fragments(fake_dataset, budget=self.n_docking_samples_per_iter)

        # reallocate samples to all nodes
        logging.debug('Reallocating samples to all nodes')
        allocations = []
        for idx, frag_alloc in enumerate(node_allocations):
            frag_idx = fake_dataset[idx][1].idx
            len_ligands = len(docking_dataset_dict[frag_idx])
            int_allocations = np.ones(len_ligands) * (frag_alloc // len_ligands)
            remaining_budget = frag_alloc % len_ligands
            if remaining_budget > 0:
                top_indices = np.argsort([
                    int(ligand['name'].split('_')[2]) for ligand, _ in docking_dataset_dict[frag_idx]
                ])[:remaining_budget]
                int_allocations[top_indices] += 1
            allocations.extend(int_allocations)
            for i, alloc in enumerate(int_allocations):
                if alloc == 0: continue
                fragsmi = docking_dataset_dict[frag_idx][i][1].smiles
                dockmolname = docking_dataset_dict[frag_idx][i][0]["name"]
                dockmolsmi = docking_dataset_dict[frag_idx][i][0]["smiles"]
                logging.debug(f'Fragment {frag_idx} ({fragsmi}), docking {dockmolname} ({dockmolsmi}), alloc {alloc}')
        allocations = np.array(allocations, dtype=int)
        return allocations

    def sample_dataloader_docking(self, dataloader, ligand_info):
        prev_mol_size = self.molecule_size
        prev_n_steps = self.model.T_sampling
        prev_virtual_nodes = self.model.virtual_nodes
        prev_sampler = self.model.sampler
        self.molecule_size = 'ground_truth'
        self.model.virtual_nodes = None
        self.model.sampler = self.docking_sampler
        self.model.T_sampling = self.n_docking_steps

        try:
            sampled_mols = []
            frag_idx_counter = Counter()
            for data in tqdm(dataloader, total=len(dataloader), desc='Sampling'):
                new_data = {
                    'ligand': TensorDict(**data['ligand']).to(self.model.device),
                    'pocket': TensorDict(**data['pocket']).to(self.model.device),
                }
                ligand_idx_to_info = [
                    ligand_info[name] for name in new_data['ligand']['name']
                ]
                rdmols, _, _ = self.model.sample(
                    new_data,
                    n_samples=1,
                    timesteps=self.model.T_sampling,
                    num_nodes=self.molecule_size,
                )
                self.global_sampling_counter += 1
                for idx, mol in enumerate(rdmols):
                    for prop in ligand_idx_to_info[idx]:
                        mol.SetProp(prop, str(ligand_idx_to_info[idx][prop]))
                    frag_idx_counter[int(ligand_idx_to_info[idx]['frag_idx'])] += 1
                sampled_mols.extend(rdmols)

        finally:
            self.molecule_size = prev_mol_size
            self.model.virtual_nodes = prev_virtual_nodes
            self.model.sampler = prev_sampler
            self.model.T_sampling = prev_n_steps
        return sampled_mols, frag_idx_counter

    def process_docking_results(self, docked_mols):
        docking_res_table = filter_mols(
            docked_mols,
            self.pocket_p, 
            evaluator=self.global_evaluator,
            criterion=self.filtering_criterion_global,
            select_unique=False,
            filter_manager=self.filter_config
        )
        docking_res_table = docking_res_table.loc[docking_res_table['passed_filters']]
        # deduplicate by smiles
        docking_res_table = docking_res_table.loc[~docking_res_table['smi'].isin(self.smiles)]
        if not 'uncertainty.mean_uncertainty' in docking_res_table.columns:
            logging.warning('Uncertainty column not found in docking results, setting to 0.0')
            docking_res_table['uncertainty.mean_uncertainty'] = 0.0
        docking_res_table = docking_res_table.sort_values(
            by='uncertainty.mean_uncertainty', ascending=True
        )
        docking_res_table = docking_res_table.drop_duplicates(
            subset='smi', keep='first'
        )

        docking_res_table['selected_for_fragmentation'] = docking_res_table['passed_filters']
        docking_res_table['is_docked'] = True
        docking_res_table['added_row'] = True
        # insert additional columns
        docking_res_table['frags'] = None
        docking_res_table['fragment_mask'] = None
        docking_res_table['frags'] = docking_res_table['frags'].astype(object)
        docking_res_table['fragment_mask'] = docking_res_table['fragment_mask'].astype(object)
        for idx, row in docking_res_table.iterrows():
            if not row['selected_for_fragmentation']: continue
            mol = row['mol']
            parent_mol = self.fragment_tree.get_node_from_index(mol.GetProp('frag_idx')).mol
            num_atmos = mol.GetNumAtoms()
            mask = np.ones(num_atmos, dtype=int)
            added_mask = get_added_atoms(parent_mol, mol, mask)
            # adding artificial mask with format [0==rest, 1==parent frag atoms, 2==new atoms]
            docking_res_table.at[idx, 'frags'] = [None, mol, -1]
            docking_res_table.at[idx, 'fragment_mask'] = added_mask

        return docking_res_table
    

    ##################################### Fragmentation #####################################
    def fragment_mols(self, select_table):
        added_rows = []
        for i, row in select_table.loc[select_table.selected_for_fragmentation].iterrows():
            mol = row['mol']
            num_atoms = mol.GetNumAtoms()
            logging.debug(f'Fragmenting molecule {Chem.MolToSmiles(mol)}')
            for atom in mol.GetAtoms():
                atom.SetProp('_InitialIndex', str(atom.GetIdx()))
            
            frag_idx = mol.GetProp('frag_idx')
            reaction_products = self.fragment_tree.get_node_from_index(frag_idx).product_set
            parent_mol = self.fragment_tree.get_node_from_index(frag_idx).mol
            if len(reaction_products) == 0:
                logging.debug(f'No reaction products for fragment {frag_idx}')
                continue
            frags, fragment_mask, mcs_results = get_fragments_building_blocks(
                mol,
                reaction_products,
                fp_size=self.fp_size,
                num_exact_search=int(100 * (15 / num_atoms)), # limit MCS comp of large molecules
                only_complete_matches=True,
            )
            # store MCS results for docking
            select_table.at[i, 'frags'] = None
            select_table.at[i, 'fragment_mask'] = np.zeros(num_atoms, dtype=int)
            select_table.at[i, 'fragment_mcs_results'] = mcs_results
            # add additional rows for new product substructure matches
            for idx, f in enumerate(frags):
                if f is None: continue
                # get "added atoms" mask (compared to parent mol)
                # adding artificial mask with format [0==rest, 1==parent frag atoms, 2==new atoms]
                added_mask = get_added_atoms(parent_mol, f, fragment_mask[idx])

                updated_row = row.copy()
                updated_row['added_row'] = True
                updated_row['frags'] = [None, f, -1]
                updated_row['fragment_mask'] = added_mask
                added_rows.append(updated_row)
        added_rows = pd.DataFrame(added_rows)
        if len(added_rows) > 0:
            select_table = pd.concat([select_table, added_rows], ignore_index=True)
        return select_table

    def process_fragments(self, select_table):
        selected_fragmols = []
        frag_smiles = {}
        has_frags = select_table['frags'].apply(lambda x: x is not None and any(x))
        sel_rows = select_table.loc[select_table.selected_for_fragmentation & has_frags]
        # only top samples for docking
        if 'is_docked' in sel_rows.columns and \
                'uncertainty.mean_uncertainty' in sel_rows.columns and \
                hasattr(self, 'max_selected_docking_samples'):
            sel_rows['is_docked'] = sel_rows['is_docked'].fillna(False)
            non_docked = sel_rows.loc[~sel_rows.is_docked]
            docked = sel_rows.loc[sel_rows.is_docked]
            docked = docked.sort_values(by='uncertainty.mean_uncertainty')
            docked = docked.iloc[:self.max_selected_docking_samples]
            sel_rows = pd.concat([non_docked, docked])
            logging.info(f'Selected {len(sel_rows)} samples for fragment evaluation, {len(docked)} of them are docked')
        else:
            logging.info(f'Selected {len(sel_rows)} samples for fragment evaluation')
        for _, frag_row in sel_rows.iterrows():
            frags = frag_row['frags']
            
            frag_table = filter_mols(
                frags, 
                self.pocket_p,
                evaluator=self.frag_evaluator,
                criterion=self.fragment_filtering_criterion,
                filter_manager=self.filter_config
            )
            filt_idxs = frag_table.loc[frag_table.passed_filters].idx
            if len(filt_idxs) == 0: continue
            fragmol = frags[-1]
            if fragmol is None: continue
            fragmol.SetProp('is_docked', str(frag_row.get('is_docked', False)))
            smi = Chem.MolToSmiles(fragmol)
            if smi in frag_smiles.values() or smi in self.fragment_tree.frag_smiles:
                continue
            selected_fragmols.append(fragmol)
            frag_smiles[fragmol] = smi

        added_frags = []
        for fragmol in selected_fragmols:
            parent_frag = fragmol.GetProp('frag_idx')
            parent = self.fragment_tree.get_node_from_index(parent_frag)
            if parent is None:
                logging.warning(f'Fragment not found in tree: {parent_frag}')
                continue
            if '.' in Chem.MolToSmiles(fragmol):
                continue
            if Chem.MolToSmiles(fragmol) in self.fragment_tree.frag_smiles:
                continue
            tree_trace = '>'.join([str(n.idx) for n in parent.get_path()] + \
                                  [str(len(self.fragment_tree.node_list))])
            fragmol.SetProp('_Name', f'frag_{tree_trace}')
            n = self.fragment_tree.add_child(
                fragmol, 
                parent,
                synthon_smiles=fragmol.GetProp('synthon_smiles'),
                react_trace=fragmol.GetProp('react_trace'),
                completed=str(fragmol.GetProp('completed')) == 'True',
            )
            if len(n.get_path()) > self.max_tree_depth:
                n.is_terminal = True
            added_frags.append(n)
            logging.debug(f'Added fragment {n.synthon_smiles} (completed: {n.completed}) to tree')
        logging.info(f'Added {len(added_frags)} fragments')
        return added_frags

    ##################################### Filtering #####################################
    def filter_fragments_local(self, select_table):
        # preprocess
        for i, frag_row in select_table.loc[select_table.selected_for_fragmentation].iterrows():
            mol = frag_row['mol']
            fragment_mask = frag_row['fragment_mask']
            frags = frag_row['frags']
            if frags is None: continue
            if not len(frags) == 3 or not frags[0] is None or not frags[2] == -1:
                logging.warning(f'Fragmentation mask has not scheme [0==rest, 1==parent, 2==new] for {Chem.MolToSmiles(mol)}')
                select_table.at[i, 'frags'] = None
            else:
                select_table.at[i, 'frag_original'] = frags[1]
        logging.info(f'Filtering {len(select_table.loc[select_table.frags.notna()])} fragments')
        # filter fragments
        select_table = super().filter_fragments_local(select_table)

        # postprocess
        for i, frag_row in select_table.loc[select_table.selected_for_fragmentation].iterrows():
            fragment_mask = frag_row['fragment_mask']
            frags = frag_row['frags']

            if frags is None or not any([f is not None for f in frags]) or len(frags) != 3:
                continue
            # if added frag is still is -1, it was selected -> restore fragmentation
            if frags[2] == -1:
                new_frags = [None,frag_row['frag_original']]
                new_mask = np.where(fragment_mask != 0, 1, 0)
            else:
                new_frags = None
                new_mask = np.zeros_like(fragment_mask)
            select_table.at[i, 'fragment_mask'] = new_mask
            select_table.at[i, 'frags'] = new_frags
        n_filt = len(select_table.loc[select_table.frags.notna()])
        logging.info(f'Filtered {n_filt} fragments after local filtering')
        return select_table

    ################################# Synthesis-Controlled Sampling #################################
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
        select_table['synthgen_sample'] = False
        select_table['frags'] = None
        select_table['fragment_mask'] = None
        select_table['frags'] = select_table['frags'].astype(object)
        select_table['fragment_mask'] = select_table['fragment_mask'].astype(object)
        select_table['fragment_mcs_results'] = None
        select_table['fragment_mcs_results'] = select_table['fragment_mcs_results'].astype(object)

        fragment_time = time.time()
        select_table = self.fragment_mols(select_table)
        select_table['fragment_time'] = time.time() - fragment_time
        
        # docking best non-exact matches
        if self.use_docking:
            mols_to_dock = self.prepare_mols_to_dock(select_table)
            dataset, ligand_info = self.prepare_docking_dataset(mols_to_dock)

            if len(dataset) > 0:
                allocations = self.allocate_docking_samples_to_fragments(dataset)
                dataloader = self.get_dataloader(dataset, allocations)
                
                docking_time = time.time()
                docked_mols, frag_idx_counter_docking = self.sample_dataloader_docking(
                    dataloader,
                    ligand_info
                )
                for k,v in frag_idx_counter_docking.items():
                    if k in frag_idx_counter: frag_idx_counter[k] += v
                    else: frag_idx_counter[k] = v

                docking_res_table = self.process_docking_results(docked_mols)
                docking_res_table['iteration'] = iteration
                docking_res_table['docking_time'] = time.time() - docking_time

                select_table = pd.concat([select_table, docking_res_table], ignore_index=True)
            else:
                logging.info('No molecules to dock in current iteration')

        local_filt_time = time.time()
        select_table = self.filter_fragments_local(select_table)
        select_table['local_filtering_time'] = time.time() - local_filt_time

        # Fragment quality filtering and processing
        fragment_filtering_time = time.time()
        added_frags = self.process_fragments(select_table)
        if 'added_row' in select_table.columns:
            select_table = select_table.loc[select_table['added_row'] != True]
        select_table['fragment_filtering_time'] = time.time() - fragment_filtering_time
        select_table['passed_filters'] = False
        new_rows = []
        for frag_node in added_frags:
            parent_node = frag_node.parent
            frag_node.mol.SetProp('frag_idx', str(parent_node.idx))
            # convert string to boolean for is_docked
            is_docked = str(frag_node.mol.GetProp('is_docked')) == 'True' if frag_node.mol.HasProp('is_docked') else False
            new_rows.append({
                'iteration': iteration,
                'ligand_name': frag_node.mol.GetProp('_Name'),
                'passed_filters': True,
                'representation.smiles': frag_node.smiles,
                'parent_smiles': parent_node.smiles,
                'mol': frag_node.mol,
                'react_trace': frag_node.react_trace,
                'synthon_smiles': frag_node.synthon_smiles,
                'completed': frag_node.completed,
                'is_docked': is_docked,
                'synthgen_sample': True,
                })
            if hasattr(parent_node, 'product_set'):
                parent_node.product_set = parent_node.product_set.loc[parent_node.product_set['smiles'] != frag_node.smiles]
            
        if len(new_rows) > 0:
            select_table = pd.concat([select_table, pd.DataFrame(new_rows)], ignore_index=True)
        select_table = self.update_tree(select_table, frag_idx_counter)

        self.global_mol_counter += len(select_table.loc[select_table.passed_filters])
        logging.info(f'Found {len(select_table.loc[select_table.passed_filters])} molecules in iteration {iteration}')
        self.smiles.update(select_table.loc[select_table.synthgen_sample]['representation.smiles'].dropna())
        if iteration < self.max_sampling_iter - 1:
            self.run_reactions()
        return select_table
    
    def retrieve_results(self, return_format='samples', only_completed=True):
        if not self.return_all_samples:
            filter_col = 'completed' if only_completed else 'synthgen_sample'
            if not filter_col in self.mol_table.columns:
                self.mol_table[filter_col] = False
            self.mol_table[filter_col] = self.mol_table[filter_col].fillna(False).astype(bool)
            self.mol_table = self.mol_table.loc[self.mol_table[filter_col]]
            self.mol_table.reset_index(drop=True, inplace=True)

            self.mol_table['ligand_idx'] = self.mol_table.index
            for i, row in self.mol_table.iterrows():
                self.mol_table.loc[i, 'mol'].SetProp('ligand_idx', str(row['ligand_idx']))

        return super().retrieve_results(return_format=return_format)
