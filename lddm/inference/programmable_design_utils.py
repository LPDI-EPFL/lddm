import sys
import logging
import signal
from typing import List, Union, Callable
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, DataStructs
from rdkit.Chem.rdmolops import FragmentOnBRICSBonds, GetMolFrags
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

basedir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(basedir))

from lddm.sbdd_metrics.metrics import ALL_EVALUATORS, LocalInteractionsEvaluator
from lddm.data.molecule_builder import remove_dummy_atoms
from lddm.utils import get_largest_connected_component, fix_aromaticity

class FragmentTree:
    def __init__(self, mol):
        self.root = FragmentNode(mol, idx=0)
        self.node_list = [self.root]
        if mol is not None:
            self.frag_smiles = set([Chem.MolToSmiles(mol)])
        else: # root is empty
            self.frag_smiles = set([''])

    def add_child(self, mol, parent):
        if parent is None:
            parent = self.root
        new_node = FragmentNode(mol, idx=len(self.node_list), parent=parent)
        parent.children.append(new_node)
        self.node_list.append(new_node)
        self.frag_smiles.add(Chem.MolToSmiles(mol))
        return new_node

    def get_generated_paths(self):
        paths = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            frag_path = node.get_frag_path()
            for gen_mol in node.filtered_mols:
                paths.append(frag_path + [gen_mol])
            stack.extend(node.children)
        return paths

    def update_total_sampling_counter(self):
        for node in self.node_list:
            node.total_sampling_counter = -1
        for node in self.node_list[::-1]:
            if node.total_sampling_counter != -1:
                continue
            node.total_sampling_counter = node.sampling_counter
            for child in node.children:
                node.total_sampling_counter += child.total_sampling_counter

    def get_node_from_index(self, idx):
        idx = int(idx)
        if idx < len(self.node_list):
            return self.node_list[idx]
        return None

    def get_node_from_smiles(self, smiles):
        for node in self.node_list:
            if node.smiles == smiles:
                return node
        return None

    def __len__(self):
        return len(self.node_list)
    

class SynthonTree(FragmentTree):
    def __init__(self, mol):
        self.root = SynthonNode(mol, idx=0)
        self.node_list = [self.root]
        if mol is not None:
            self.frag_smiles = set([Chem.MolToSmiles(mol)])
        else: # root is empty
            self.frag_smiles = set([''])

    def add_child(self, mol, parent=None, synthon_smiles=None, react_trace=None, completed=None):
        if parent is None:
            parent = self.root
        new_node = SynthonNode(
            mol, 
            idx=len(self.node_list), 
            parent=parent, 
            synthon_smiles=synthon_smiles,
            react_trace=react_trace,
            completed=completed,
        )
        parent.children.append(new_node)
        self.node_list.append(new_node)
        self.frag_smiles.add(Chem.MolToSmiles(mol))
        return new_node

class FragmentNode:
    def __init__(self, mol, idx, parent=None):
        self.mol = mol
        self.smiles = Chem.MolToSmiles(mol) if mol is not None else ''
        self.idx = idx
        self.processed_ligand = None
        self.filtered_mols = []
        self.sampling_counter = 0
        self.parent = parent
        self.children = []
        self.is_terminal = False
        self.total_sampling_counter = 0

    def get_frag_path(self):
        path = []
        node = self
        while node is not None:
            path.append(node.mol)
            node = node.parent
        return path[::-1]

    def get_path(self):
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]
    
class SynthonNode(FragmentNode):
    def __init__(self, mol, idx, parent=None, synthon_smiles=None, react_trace=None, completed=None):
        super().__init__(mol, idx, parent)
        self.synthon_smiles = synthon_smiles if synthon_smiles is not None else Chem.MolToSmiles(mol) if mol is not None else ''
        self.react_trace = None
        self.product_set = None
        self.react_trace = react_trace
        self.completed = completed

############################################ Fragmentation ############################################
def get_fragments_brics(mol):
    init_frags = GetMolFrags(FragmentOnBRICSBonds(mol), asMols=True)
    # init_frags = get_fragment_mols_MacFrag(mol, maxSR=8, minFragAtoms=1)
    fragment_mask = np.zeros(mol.GetNumAtoms(), dtype=int)
    for fragment_idx, fragment in enumerate(init_frags):
        for atom in fragment.GetAtoms():
            idx = atom.GetPropsAsDict().get('_InitialIndex')
            if idx is not None:
                fragment_mask[int(idx)] = fragment_idx

    # removing dummy atoms and Hs
    frags_processed = []
    for fragment in init_frags:
        fproc = remove_dummy_atoms(fragment, sanitize=False)
        if Chem.SanitizeMol(fproc, catchErrors=True) != 0:
            logging.debug('Failed sanitizing molecule in get_fragments')
            frags_processed.append(None)
            continue
        # only frags > 1 atom
        if fproc.GetNumAtoms() <= 1:
            frags_processed.append(None)
            continue
        fproc = Chem.RemoveHs(fproc)
        frags_processed.append(fproc)
    return frags_processed, fragment_mask

def get_fragments_building_blocks(
        mol: Chem.Mol,
        building_blocks_db: pd.DataFrame,
        fp_size=2048,
        num_exact_search=100,
        only_complete_matches=False,
        max_matched_substructures=3,
    ):
    required_cols = {'smiles', 'fp', 'react_trace'}
    if not required_cols.issubset(building_blocks_db.columns):
        raise ValueError(f'Building blocks database must contain {required_cols}')
    if 'synthon_smiles' not in building_blocks_db.columns:
        building_blocks_db['synthon_smiles'] = building_blocks_db['smiles']
    fragment_mask = np.zeros(mol.GetNumAtoms(), dtype=int)
    frags_processed = [None] # init with invalid fragment

    # Compute fingerprints and get top-k matches
    fpgen = AllChem.GetMorganGenerator(2, fpSize=fp_size)
    mol_fp = fpgen.GetFingerprint(mol)
    db_fps = building_blocks_db['fp']
    scores = np.array(DataStructs.BulkTanimotoSimilarity(mol_fp, list(db_fps)))
    top_indices = np.argsort(scores)[::-1][:min(num_exact_search, len(scores))]
    selected = building_blocks_db.iloc[top_indices]
    selected_scores = scores[top_indices]

    # MCS params
    params = rdFMCS.MCSParameters()
    params.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    params.AtomCompareParameters.MatchFormalCharge = True
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = True
    params.Timeout = 4

    # find MCS for top hits
    mcs_data = []
    for i, (_, row) in enumerate(selected.iterrows()):
        db_smiles = row['smiles']
        db_synthon_smiles = row['synthon_smiles']
        db_react_trace = row['react_trace']
        db_completed = row.get('completed', True)
        db_similarity = selected_scores[i]

        db_mol = Chem.MolFromSmiles(db_smiles)
        if db_mol is None:
            continue

        try:
            mcs_res = rdFMCS.FindMCS([mol, db_mol], params)
        except Exception as e:
            logging.error(f"MCS search failed for SMILES {db_smiles}: {e}")
            continue
        if mcs_res.canceled:
            logging.warning(f'MCS search timed out for SMILES {db_smiles}')
        if mcs_res.numAtoms == 0:
            continue

        smarts_mol = Chem.MolFromSmarts(mcs_res.smartsString)
        matches = mol.GetSubstructMatches(smarts_mol)
        if not matches:
            continue
        if len(matches) > max_matched_substructures:
            matches = matches[:max_matched_substructures]

        db_match_atoms = db_mol.GetSubstructMatch(smarts_mol)
        match_complete = len(db_match_atoms) == db_mol.GetNumAtoms()

        for match in matches:
            mcs_data.append({
                'match_atoms': match,
                'db_match_atoms': db_match_atoms,
                'match_complete': match_complete,
                'react_trace': db_react_trace,
                'db_smiles': db_smiles,
                'synthon_smiles': db_synthon_smiles,
                'completed': db_completed,
                'db_mol_size': db_mol.GetNumAtoms(),
                'db_similarity': db_similarity
            })
    if not mcs_data:
        return frags_processed, [], []

    # Sort MCS data by similarity and completeness
    mcs_df = pd.DataFrame(mcs_data)
    mcs_df['n_match_atoms'] = mcs_df['match_atoms'].apply(len)
    mcs_df['n_match_atoms'].fillna(0, inplace=True)
    mcs_df.sort_values(
        by=['match_complete', 'n_match_atoms'],
        ascending=[False, False],
        inplace=True
    )
    # print df stats
    logging.debug(
        f'Found {len(mcs_df)} MCS matches for {Chem.MolToSmiles(mol)}, '
        f'of which {mcs_df["match_complete"].sum()} are complete.'
    )
    used_atom_idxs = set()
    mcs_data = []

    for _, mcs in mcs_df.iterrows():
        # exctract fragment from mol
        atm_idxs = list(mcs['match_atoms'])
        rwmol = Chem.RWMol(mol)
        for idx in sorted(set(range(mol.GetNumAtoms())) - set(atm_idxs), reverse=True):
            rwmol.RemoveAtom(idx)
        frag = fix_aromaticity(rwmol)
        frag = get_largest_connected_component(frag, strict=True)
        if frag is None or Chem.SanitizeMol(frag, catchErrors=True) != 0:
            logging.debug("Fragment sanitization failed.")
            continue

        # check if fragment overlaps with already used atoms/is not complete
        same_structure = Chem.MolToSmiles(frag) == Chem.CanonSmiles(mcs['db_smiles'])
        if any(idx in used_atom_idxs for idx in atm_idxs) or (only_complete_matches and
                (not mcs['match_complete'] or not same_structure)):
            substruct_similarity = DataStructs.TanimotoSimilarity(
                fpgen.GetFingerprint(frag), mol_fp
            )
            mcs['similarity'] = substruct_similarity
            mcs_data.append(mcs.to_dict())
            continue
        elif only_complete_matches and frag.GetNumAtoms() != mcs['db_mol_size']:
            logging.warning(f"Fragment {Chem.MolToSmiles(frag)} atom count mismatch vs DB mol {mcs['db_smiles']}")
        else:
            for prop in mol.GetPropNames(includePrivate=True):
                frag.SetProp(prop, mol.GetProp(prop))
            frag.SetProp('react_trace', str(mcs['react_trace']))
            frag.SetProp('synthon_smiles', str(mcs['synthon_smiles']))
            frag.SetProp('completed', str(mcs['completed']))
            fragment_mask[atm_idxs] = len(frags_processed)
            frags_processed.append(frag)
            used_atom_idxs.update(atm_idxs)

    all_masks = [np.array(fragment_mask == idx, dtype=int) for idx in range(len(frags_processed))]
    return frags_processed, all_masks, mcs_data

def coords_overlap(mol1, mol2, eps=0.05):
    coords1 = mol1.GetConformer().GetPositions()
    coords2 = mol2.GetConformer().GetPositions()
    pairwise_distances = np.linalg.norm(
        coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :], axis=2
    )
    return np.any(pairwise_distances < eps)

def coords_overlap_atm(atm_idx, mol1, mol2, eps=0.05):
    """Check if the atom's coordinates overlap with any atom in the molecule."""
    positions = mol1.GetConformer().GetAtomPosition(atm_idx)
    atom_coords = np.array([positions.x, positions.y, positions.z])
    mol_coords = mol2.GetConformer().GetPositions()
    pairwise_distances = np.linalg.norm(
        mol_coords - atom_coords, axis=1
    )
    return np.any(pairwise_distances < eps)

def get_added_atoms(mol_parent, mol_gen, fragment_mask):
    """
    Returns an atom mask of mol_gen with 0=rest, 2=atoms from fragments_mask, 1=atoms that are in mol_parent and fragments_mask.
    """
    assert mol_gen is not None, "Molecule must not be None to generate mask."
    if mol_parent is None:
        new_mask = 2*(fragment_mask > 0) # new atoms have index 2, others are 0
        return new_mask

    gen_atoms = list(mol_gen.GetAtoms())
    parent_atoms = list(mol_parent.GetAtoms())

    # Use the molecule's conformer to get atom positions
    gen_conf = mol_gen.GetConformer()
    parent_conf = mol_parent.GetConformer()

    gen_coords = np.array([
        list(gen_conf.GetAtomPosition(atm.GetIdx()))
        for atm in gen_atoms
    ])
    parent_coords = np.array([
        list(parent_conf.GetAtomPosition(atm.GetIdx()))
        for atm in parent_atoms
    ])

    gen_indices = np.array([atm.GetIdx() for atm in gen_atoms])
    match_mask = np.any(
        np.all(np.isclose(gen_coords[:, None, :], parent_coords[None, :, :]), axis=2),
        axis=1
    )

    added_mask = ~match_mask
    new_mask = fragment_mask.copy()
    new_mask[gen_indices[added_mask]] += fragment_mask[gen_indices[added_mask]] > 0

    return new_mask


def filter_fragments(
        mol,
        fragment_mask,
        filtered_fragments_idxs,
        fixed_frags: Chem.Mol = None,
    ):
    frags_to_keep = set(filtered_fragments_idxs)
    new_mol = Chem.RWMol(mol)
    atms_to_remove = [
        atm.GetIdx()
        for atm in mol.GetAtoms()
        if fragment_mask[atm.GetIdx()] not in frags_to_keep
    ]
    for atm_idx in sorted(atms_to_remove, reverse=True):
        new_mol.RemoveAtom(atm_idx)
    new_mol = new_mol.GetMol()
    new_mol = fix_aromaticity(new_mol)
    if new_mol is None:
        logging.warning('Fragmentation resulted in an invalid molecule.')
        return None
    if fixed_frags is not None:
        for frag in Chem.GetMolFrags(fixed_frags, asMols=True):
            # check if the fragment is already in the molecule by comparing coordinates
            if not coords_overlap(frag, new_mol):
                new_mol = Chem.CombineMols(new_mol, frag)
    else:
        # check number of atoms in the new molecule
        atms_to_keep = mol.GetNumAtoms() - len(atms_to_remove)
        if new_mol.GetNumAtoms() != atms_to_keep:
            logging.warning(f'Fragmentation resulted in a molecule with {new_mol.GetNumAtoms()} != {atms_to_keep}.')
            return None
    return new_mol

############################################ Molecule filtering ############################################
def filter_func_wrapper(filter_func):
    '''catches KeyError and returns False for these cases'''
    def wrapper(row):
        try:
            return filter_func(row)
        except KeyError:
            return False
    return wrapper

# Default uncertainty based on previous runs
DEFAULT_UNCERTAINTY_THRESHOLD = 0.880
DEFAULT_EVALUATOR_LIST = [
    'representation', 'medchem', 'uncertainty',
]
DEFAULT_INTERACTION_LIST = [
    'HBAcceptor', 'HBDonor', 'XBAcceptor', 'XBDonor', 'CationPi', 'PiCation', 'PiStacking',
]

DEFAULT_FILTERS = {
    'filter_results': {
        'posebusters.all': (1, 1),
        'medchem.valid': (1, 1),
        'medchem.connected': (1, 1),
        'clashes.passed_clash_score_ligands': (1, 1),
        'clashes.passed_clash_score_between': (1, 1),
        'validity3d.all': (1, 1),
    },
    'basic_filter': {
        'medchem.valid': (1, 1),
        'medchem.connected': (1, 1),
    },
    'filter_pb': {
        'medchem.valid': (1, 1),
        'medchem.connected': (1, 1),
        'posebusters.all': (1, 1),
    },
}

class FilterConfig:
    def __init__(self, config_path: str = None):
        self.sampling_filters = {}
        for name, filter_conditions in DEFAULT_FILTERS.items():
            # group by evaluator
            evaluators = set()
            grouped_conditions = {}
            for col, bounds in filter_conditions.items():
                evaluator = col.split('.')[0]
                evaluators.add(evaluator)
                if evaluator not in grouped_conditions:
                    grouped_conditions[evaluator] = {'rank': len(grouped_conditions), 'conditions': {}}
                grouped_conditions[evaluator]['conditions'][col] = bounds
            # create filter functions
            for evaluator, cond in grouped_conditions.items():
                conditions = cond['conditions']
                if not conditions:
                    continue
                # create filter function
                grouped_conditions[evaluator]['filter_func'] = self._create_filter_lambda(conditions)

            self.sampling_filters[name] = {
                'grouped_conditions': grouped_conditions,
                'evaluators': list(evaluators)
            }
        if config_path:
            self._load_yaml_config(config_path)

    def _load_yaml_config(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f).get('filters', {})

        for name, details in config.items():
            filter_conditions = {
                col: tuple(bounds)
                for col, bounds in details.items()
            }
            # group by evaluator
            evaluators = set()
            grouped_conditions = {}
            for col, bounds in filter_conditions.items():
                evaluator = col.split('.')[0]
                evaluators.add(evaluator)
                if evaluator not in grouped_conditions:
                    grouped_conditions[evaluator] = {'rank': len(grouped_conditions), 'conditions': {}}
                grouped_conditions[evaluator]['conditions'][col] = bounds
            # sort evaluators by rank
            grouped_conditions = dict(sorted(grouped_conditions.items(), key=lambda item: item[1]['rank']))
            # create filter functions
            for evaluator, cond in grouped_conditions.items():
                conditions = cond['conditions']
                if not conditions:
                    continue
                # create filter function
                grouped_conditions[evaluator]['filter_func'] = self._create_filter_lambda(conditions)
            self.sampling_filters[name] = {
                'grouped_conditions': grouped_conditions,
                'evaluators': list(evaluators)
            }

    def _create_filter_lambda(self, conditions):
        def filter_func(row):
            return all(
                lower <= row.get(col, float('-inf')) <= upper
                for col, (lower, upper) in conditions.items()
            )
        return filter_func

    def update_filter_threshold(self, column, new_bounds):
        for filter_name, filter_data in self.sampling_filters.items():
            evaluator = column.split('.')[0]
            if evaluator not in filter_data['grouped_conditions']:
                continue
            # Update the conditions for the specific column
            current_conditions = filter_data['grouped_conditions'][evaluator]['conditions']
            if column not in current_conditions: continue
            if not isinstance(new_bounds, tuple) or len(new_bounds) != 2:
                raise ValueError(f'New bounds for {column} must be a tuple of two values.')
            current_conditions[column] = tuple(new_bounds)
            self.sampling_filters[filter_name]['grouped_conditions'][evaluator]['conditions'] = current_conditions
            # Update the filter function
            self.sampling_filters[filter_name]['grouped_conditions'][evaluator]['filter_func'] = self._create_filter_lambda(current_conditions)
            logging.info(f'Updated filter {filter_name} for column {column} to new bounds {new_bounds}')

    def get_evaluators(self, criterion: str) -> List[str]:
        if criterion not in self.sampling_filters:
            raise ValueError(f'Unknown filter criterion: {criterion}')
        return_list = []
        return_list.extend(DEFAULT_EVALUATOR_LIST)
        for evaluator in self.sampling_filters[criterion].get('evaluators', []):
            if evaluator not in return_list:
                return_list.append(evaluator)
        return return_list

    def get_filter(self, criterion: str, evaluator_id: str) -> Callable:
        if criterion not in self.sampling_filters:
            raise ValueError(f'Unknown filter criterion: {criterion}')
        
        grouped_conditions = self.sampling_filters[criterion]['grouped_conditions']
        if not grouped_conditions:
            raise ValueError(f'No conditions defined for filter criterion: {criterion}')

        # Combine all filter functions from evaluators
        if evaluator_id not in grouped_conditions:
            return lambda row: True  # No specific filter for this evaluator, return True
        
        filter_func = grouped_conditions[evaluator_id]['filter_func']
        return filter_func

    def get_exclude_evaluators(self, criterion: str) -> List[str]:
        if criterion not in self.sampling_filters:
            raise ValueError(f'Unknown filter criterion: {criterion}')

        return list(
            set(ALL_EVALUATORS) - 
            set(self.sampling_filters.get(criterion, {}).get('evaluators', [])).union(
                set(DEFAULT_EVALUATOR_LIST)
            )
        )


def aggregate_interaction_profile(profile, interactions_to_aggregate=None):
    if interactions_to_aggregate is None:
        interactions_to_aggregate = DEFAULT_INTERACTION_LIST
    res = {}
    evaluated_interactions_cols = [
        col for col in profile.keys()
            if col.startswith('interactions_local.')
    ]
    if not evaluated_interactions_cols:
        # no interaction eval was performed
        return res
    # collect interactions to aggregate
    tmp_interactions = [f'interactions_local.{interaction}'
        for interaction in interactions_to_aggregate]

    # aggregate unsatisfied bonds metrics
    unsatisfied_bonds_cols = [f'interactions_local.{c}' 
        for c in LocalInteractionsEvaluator.UNSATISFIED_HBOND_METRICS]
    unsatisfied_bonds_cols = set(unsatisfied_bonds_cols) \
        .intersection(set(evaluated_interactions_cols)) \
        .intersection(set(tmp_interactions))
    unsatisfied_bonds = {i: 0
        for i in range(len(profile[evaluated_interactions_cols[0]]))}
    if unsatisfied_bonds_cols:
        for col in unsatisfied_bonds_cols:
            tmp_interactions.remove(col)
            unsatisfied_bonds = {i: unsatisfied_bonds[i] + v
                for i, v in profile[col].items()}
    res['interactions_local.unsatisfied_bonds'] = {
        i: unsatisfied_bonds[i] for i in range(len(unsatisfied_bonds))
    }

    # aggregate other interactions metrics
    if tmp_interactions:
        sel_interactions_cols = set(tmp_interactions).intersection(set(evaluated_interactions_cols))
        sel_interactions = [
            list(profile[col].values()) for col in sel_interactions_cols
        ]
        arr = np.array(sel_interactions, dtype=int)
        sum_interactions = np.sum(arr, axis=0)
        res['interactions_local.sum'] = {
            i: sum_interactions[i] for i in range(len(sum_interactions))
        }
    return res


def aggregate_local_eval_per_fragment(eval_per_atom, fragment_mask, uncertainty_threshold=None):
    uncertainty_threshold = uncertainty_threshold or DEFAULT_UNCERTAINTY_THRESHOLD
    frag_mask_arr = np.array(fragment_mask)
    aggregated = {}
    aggregated['local_eval.all'] = {
        idx: True for idx in range(0, max(frag_mask_arr)+1)
    }

    def check_fragment_condition(condition_dict, condition_func):
        if len(condition_dict) == 0:
            return None
        res = {}
        for frag_idx in np.unique(frag_mask_arr):
            frag_atoms = np.where(frag_mask_arr == frag_idx)[0]       
            frag_values = [condition_dict.get(atom_idx, 0) for atom_idx in frag_atoms]
            res[frag_idx] = condition_func(frag_values)
            if isinstance(res[frag_idx], bool):
                aggregated['local_eval.all'][frag_idx] &= res[frag_idx]
        return res

    # Interactions: keep fragment if it has any interactions
    aggregated['pass_interactions_local'] = check_fragment_condition(
        eval_per_atom.get('interactions_local.sum', {}),
        lambda values: bool(sum(values) > 0),
    )
    aggregated['sum_interactions_local'] = check_fragment_condition(
        eval_per_atom.get('interactions_local.sum', {}),
        sum,
    )
    # Unsatisfied bonds: exclude fragment if it has unsatisfied bonds
    aggregated['pass_unsatisfied_bonds'] = check_fragment_condition(
        eval_per_atom.get('interactions_local.unsatisfied_bonds', {}),
        lambda values: bool(sum(values) == 0),
    )
    aggregated['sum_unsatisfied_bonds'] = check_fragment_condition(
        eval_per_atom.get('interactions_local.unsatisfied_bonds', {}),
        sum,
    )
    # Posebusters: exclude fragment if it any atom fails posebuster
    aggregated['pass_posebusters_local'] = check_fragment_condition(
        eval_per_atom.get('posebusters_local.all', {}),
        lambda values: all(v == 1 for v in values),
    )
    # Validity3D: like posebusters
    aggregated['pass_validity3d_local'] = check_fragment_condition(
        eval_per_atom.get('validity3d_local.all', {}),
        lambda values: all(v == 1 for v in values),
    )
    # Clashes between: exclude fragment if it has clashes
    aggregated['pass_clashes_local_between'] = check_fragment_condition(
        eval_per_atom.get('clashes_local.passed_clash_score_between', {}),
        lambda values: all(v == 1 for v in values),
    )
    # Clashes with ligands: exclude fragment if it has clashes
    aggregated['pass_clashes_local_ligands'] = check_fragment_condition(
        eval_per_atom.get('clashes_local.passed_clash_score_ligands', {}),
        lambda values: all(v == 1 for v in values),
    )
    # Confidence: exclude fragment if it has high uncertainty on avg
    aggregated['pass_uncertainty_local'] = check_fragment_condition(
        eval_per_atom.get('uncertainty_local.sigma_x', {}),
        lambda values:
            bool(np.mean([v for v in values if v is not None]) < uncertainty_threshold)
                if any(v is not None for v in values)
            else False
    )
    aggregated['mean_uncertainty_local'] = check_fragment_condition(
        eval_per_atom.get('uncertainty_local.sigma_x', {}),
        lambda values:
            np.mean([v for v in values if v is not None and v>0])
                if any(v is not None and v>0 for v in values)
            else None
    )

    aggregated = {k: v for k, v in aggregated.items() if v is not None}
    return aggregated


def filter_mols(
        rdmols,
        protein_ps,
        evaluator,
        smiles=None,
        criterion='basic_filter',
        select_unique=True,
        filter_manager=None
    ):
    filter_manager = filter_manager or FilterConfig()
    smiles = set() if smiles is None else set(smiles)
    if not isinstance(protein_ps, list):
        protein_ps = [protein_ps] * len(rdmols)

    passed_data = []
    evaluator_ids = filter_manager.get_evaluators(criterion)
    for idx, mol in enumerate(rdmols):
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        if smi in smiles and select_unique:
            logging.debug(f'Skipping molecule {idx} with SMILES {smi} as it is already in the set')
            continue
        all_metrics = {}
        prot_p = protein_ps[idx]
        passed_all_metrics = True
        for evaluator_id in evaluator_ids:
            part_evaluator = evaluator.get_evaluator(evaluator_id)
            results = part_evaluator(mol, prot_p)
            all_metrics.update(results)

            filter_crit = filter_manager.get_filter(criterion, evaluator_id)
            if not filter_crit(results):
                logging.debug(f'Failed {evaluator_id} filter for molecule {idx}: {results}')
                passed_all_metrics = False
                break

        all_metrics['passed_filters'] = passed_all_metrics
        if passed_all_metrics:
            smiles.add(smi)
            logging.debug(f'Passed all filters for molecule {idx}')
        all_metrics['mol'] = Chem.Mol(mol)
        if Chem.SanitizeMol(all_metrics['mol'], catchErrors=True) != 0:
            all_metrics['mol'] = Chem.Mol(mol) # fallback to original mol
        all_metrics['smi'] = smi
        all_metrics['idx'] = idx
        passed_data.append(all_metrics)

    if not passed_data:
        return pd.DataFrame({
            'mol': pd.Series(dtype=object),
            'smi': pd.Series(dtype=str),
            'representation.smiles': pd.Series(dtype=str),
            'idx': pd.Series(dtype=int),
            'passed_filters': pd.Series(dtype=bool),
        })
    table = pd.DataFrame(passed_data)
    table.fillna(value=np.nan, inplace=True)
    return table
