import sys
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
import gc
import pickle

from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import pandas as pd

basedir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(basedir))

from lddm.reactions.reaction_utils import (
    get_react_trace_building_block,
    get_react_trace_bimolecular,
    run_bimolecular_reaction_smarts,
)

class Reactant:
    def __init__(self, reactant_id, smarts, explicit_hs=False):
        assert isinstance(reactant_id, int), f'Reactant ID must be an integer, got {type(reactant_id)}'
        self.id = reactant_id
        self.smarts = smarts
        self.query = Chem.MolFromSmarts(smarts)
        self.allowed_building_blocks = None
        self.explicit_hs = explicit_hs

    def matches_mol(self, mol_smi):
        params = Chem.SubstructMatchParameters()
        params.numThreads = 1 # avoid issue with multiprocessing
        params.maxMatches = 1
        mol = Chem.MolFromSmiles(mol_smi)
        try:
            if self.explicit_hs:
                mol = Chem.AddHs(mol)
        except Exception as e:
            logging.error(f'Error adding hydrogens to molecule {mol_smi}: {e}')
        has_substruct_match = False
        if mol is not None:
            try:
                has_substruct_match = mol.HasSubstructMatch(self.query, params)
            except Exception as e:
                logging.error(f'Error checking substructure match for {self.smarts} in {mol_smi}: {e}')
                has_substruct_match = False
        return has_substruct_match

class ReactionsProcessor:
    def __init__(self, reactions, building_blocks, fp_size=2028):
        self.reactions = {}
        for reaction_info in reactions:
            assert 'id' in reaction_info, f'Reaction does not have an ID.'
            assert 'reaction' in reaction_info, f'Reaction does not have a reaction SMARTS.'
            assert 'explicit_hs' in reaction_info, f'Reaction does not have explicit_hs flag.'
            reaction_info['name'] = reaction_info.get('name', f'unk')
            reaction_info['id'] = str(reaction_info['id'])
            educts, product = reaction_info['reaction'].split('>>')
            educts = [e.strip() for e in educts.split('.')]
            reaction_info['reactants'] = [
                Reactant(i, e, reaction_info['explicit_hs']) for i, e in enumerate(educts)
            ]
            reaction_info['reaction_smarts'] = reaction_info['reaction']
            self.reactions[reaction_info['id']] = reaction_info

        self.fp_size = fp_size
        self.compatible_building_blocks = set()
        self.bb_id_to_smi = {bb_id: smi for bb_id, smi in zip(building_blocks['id'], building_blocks['smiles'])}

    def load_reactant_to_building_blocks(self, reactant_to_compound_path=None):
        if reactant_to_compound_path is not None:
            logging.info(f'Loading reaction to building blocks mapping from {reactant_to_compound_path}')
            self._load_reactant_to_building_blocks(reactant_to_compound_path)
        else:
            logging.warning('Reaction to compound mapping path is not provided, computing mapping ...')
            self._compute_reactant_to_building_blocks()

    def __len__(self):
        return len(self.reactions)

    def _load_reactant_to_building_blocks(self, reactant_to_compound_path):
        avail_bb_ids = set(self.bb_id_to_smi.keys())
        reactant_to_compound_path = Path(reactant_to_compound_path)
        if not reactant_to_compound_path.exists() or not reactant_to_compound_path.suffix == '.pkl':
            raise ValueError(f'Reaction to building blocks mapping file not found or invalid format: {reactant_to_compound_path}')
        with open(reactant_to_compound_path, 'rb') as f:
            reactant_to_compound = pickle.load(f)
        for reaction_id, reaction_info in tqdm(self.reactions.items(), desc='Loading reaction to building blocks mapping'):
            mapping = reactant_to_compound.get(reaction_id, {})
            if not mapping:
                raise ValueError(f'Reaction {reaction_id} not found in the provided mapping.')
            for reactant in reaction_info['reactants']:
                bbs = mapping.get(reactant.id, [])
                bbs = set(bbs)
                sel_bbs = bbs.intersection(avail_bb_ids)
                self.reactions[reaction_id]['reactants'][reactant.id].allowed_building_blocks = sel_bbs
                if not sel_bbs:
                    logging.warning(f'No compatible building blocks found for reactant {reactant.id} in reaction {reaction_id}.')
                    continue
                self.compatible_building_blocks.update(sel_bbs)
        logging.info(f'Loaded {len(self.compatible_building_blocks)} compatible building blocks for {len(self.reactions)} reactions.')
    
    def _compute_reactant_to_building_blocks(self):
        avail_bb_ids = set(self.bb_id_to_smi.keys())
        for reaction_id, reaction_info in tqdm(self.reactions.items(), desc='Computing reaction to building blocks mapping'):
            for reactant in reaction_info['reactants']:
                if reactant.allowed_building_blocks is not None:
                    logging.warning(f'Compatible building blocks for reactant {reactant.id} in reaction {reaction_id} are already set, skipping.')
                    continue
                compatible_bbs = set()
                for bb_id in avail_bb_ids:
                    bb_smi = self.bb_id_to_smi[bb_id]
                    if reactant.matches_mol(bb_smi):
                        compatible_bbs.add(bb_id)
                reactant.allowed_building_blocks = compatible_bbs
                self.compatible_building_blocks.update(compatible_bbs)
        logging.info(f'{len(self.compatible_building_blocks)} compatible building blocks for {len(self.reactions)} reactions.')

    def run_reactions_for_node(self, synthon_smi, smiles, react_trace, num_workers=1):
        """Run all applicable reactions on a synthon using multiprocessing."""
        logging.info(f'Running reactions for node {synthon_smi} with {num_workers} workers')
        assert synthon_smi == smiles, f'For normal reaction processing, synthon_smi must match smiles: {synthon_smi} != {smiles}'

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self._compute_products, synthon_smi, react_trace, rxn)
                       for rxn in self.reactions.values()]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        results_df = [
            pd.DataFrame(r) for r in results 
            if r and 'checked' in r and len(r['smiles']) > 0
        ]
        if not results_df:
            return pd.DataFrame(columns=['fp', 'react_trace', 'smiles', 'synthon_smiles', 'checked', 'completed'])

        product_set = pd.concat(results_df, ignore_index=True).drop_duplicates(subset='smiles')
        gc.collect()
        return product_set

    def _compute_products(self, smi, react_trace, reaction):
        """Compute products from a given SMILES and a reaction definition."""
        fpgen = AllChem.GetMorganGenerator(2, fpSize=self.fp_size)
        fps, ids, smiles_list = [], [], []

        matching = [
            i for i, reactant in enumerate(reaction['reactants'])
            if reactant.matches_mol(smi)
        ]

        if len(matching) != 1 or len(reaction['reactants']) != 2:
            return {'fp': [], 'react_trace': [], 'smiles': [], 'synthon_smiles': [], 'checked': []}

        matched_idx = matching[0]
        other_reactant = reaction['reactants'][1 - matched_idx]
        
        for bb_idx in other_reactant.allowed_building_blocks:
            bb_smi = self.bb_id_to_smi[bb_idx]
            reactants = [smi, bb_smi] if matched_idx == 0 else [bb_smi, smi]
            products_smi = run_bimolecular_reaction_smarts(
                reaction['reaction_smarts'],
                reactants[0],
                reactants[1],
                explicit_hs=reaction['explicit_hs']
            )
            product_smi = None
            if products_smi and len(products_smi) == 1:
                product_smi = products_smi[0]
            else: # only allow single product reactions
                # logging.debug(f'Reaction {reaction["id"]} with reactants {reactants} produced multiple/no products: {products_smi}')
                continue

            mol = Chem.MolFromSmiles(product_smi)
            if mol is None:
                continue

            fps.append(fpgen.GetFingerprint(mol))
            smiles_list.append(product_smi)

            bb_trace = get_react_trace_building_block(bb_smi, id=bb_idx)
            trace1, trace2 = (react_trace, bb_trace) if matched_idx == 0 else (bb_trace, react_trace)
            ids.append(get_react_trace_bimolecular(trace1, trace2, reaction["id"], product_smi))

        logging.debug(f'Generated {len(smiles_list)} new products for {smi} (reaction {reaction["id"]})')
        return {
            'fp': fps,
            'react_trace': ids,
            'smiles': smiles_list,
            'synthon_smiles': smiles_list,
            'checked': [False] * len(smiles_list),
            'completed': [True] * len(smiles_list),
        }
