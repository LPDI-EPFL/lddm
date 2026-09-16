import sys
import logging
from rdkit import Chem
from rdkit.Chem import AllChem
import gc
import pickle

from tqdm import tqdm
from pathlib import Path
import pandas as pd

basedir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(basedir))

from lddm.reactions.reaction_utils import (
    get_react_trace_building_block,
    get_react_trace_bimolecular,
    get_react_trace_trimolecular,
    find_connectors,
    connect_synthons,
    clean_smi,
    ReactionTree
)

class ReactionsProcessorEnamine:
    def __init__(self, reactions_df, synthon_df, fp_size=2028):
        self.reactions_df = reactions_df
        self.synthon_df = synthon_df

        assert 'components' in self.reactions_df.columns, 'No components column found in reactions_df'
        assert 'reaction_id' in self.reactions_df.columns, 'No reaction_id column found in reactions_df'
        assert 'synthon_smiles' in self.synthon_df.columns, 'No synthon_smiles column found in synthon_df'
        assert 'synton_id' in self.synthon_df.columns, 'No synton_id column found in synthon_df'
        assert 'synton#' in self.synthon_df.columns, 'No synton# column found in synthon_df'
        assert 'reaction_id' in self.synthon_df.columns, 'No reaction_id column found in synthon_df'

        reaction2components = dict(self.reactions_df[['reaction_id', 'components']].values)

        self.synthon_df['components'] = self.synthon_df['reaction_id'].map(reaction2components)
        self.synthon_df['completed'] = self.synthon_df['synthon_smiles'] == self.synthon_df['smiles']

        # deduplicate synthon_df
        self.processed_synthon_df = self.drop_duplicates_by_completeness(self.synthon_df)
        self.fp_size = fp_size

    def drop_duplicates_by_completeness(self, df):
        if 'completed' not in df.columns:
            logging.warning('No completed column found in DataFrame, cannot drop duplicates by completeness')
            return df.drop_duplicates(subset='smiles', keep='first').reset_index(drop=True)
        df_sorted = df.sort_values(by='completed', ascending=False)
        result = df_sorted.drop_duplicates(subset='smiles', keep='first').reset_index(drop=True)
        return result

    def run_reactions_for_node(self, synthon_smi, smiles, react_trace, num_workers=1): # TODO implement multiprocessing
        default_df = pd.DataFrame(columns=['fp', 'react_trace', 'smiles', 'synthon_smiles', 'checked', 'completed'])
        react_tree = ReactionTree(react_trace)
        if react_tree.tree['educts'] is None or len(react_tree.tree['educts']) == 0:
            num_react_steps = 1
        elif react_tree.tree['educts'] is not None:
            num_react_steps = len(react_tree.tree['educts'])
        if num_react_steps not in [1, 3]:
            logging.warning(f'Unexpected number of steps in reaction trace: {num_react_steps} for synthon {synthon_smi}')
            return default_df
        
        if num_react_steps == 1:  # completing a bimolecular reaction
            possible_synthon_smiles = self.processed_synthon_df[
                self.processed_synthon_df['smiles'] == smiles]['synthon_smiles'].values
            product_sets = []
            for synsmi in tqdm(possible_synthon_smiles, desc=f'Processing bimolecular reactions for {smiles}', total=len(possible_synthon_smiles)):
                product_set_synsmi = self._compute_products_2steps(synsmi, num_workers=num_workers)
                if len(product_set_synsmi) > 0:
                    product_sets.append(product_set_synsmi)
            product_set = pd.concat(product_sets, ignore_index=True) if product_sets else default_df
        elif num_react_steps == 3:  # completing a trimolecular reaction
            logging.info(f'Processing trimolecular reaction for synthon {synthon_smi} / trace {react_trace}')
            product_set = self._compute_products_3steps(synthon_smi, react_tree, num_workers=num_workers)
        if len(product_set) == 0:
            logging.warning(f'No products found for synthon {synthon_smi} / trace {react_trace}')
            return default_df
        product_set = self.drop_duplicates_by_completeness(product_set)
        return product_set

    def _compute_products_2steps(self, synthon_smi, num_workers=1):
        results = []
        fpgen = AllChem.GetMorganGenerator(2, fpSize=self.fp_size)
        relevant_synthons = self.synthon_df[self.synthon_df['synthon_smiles'] == synthon_smi]
        synthon_connectors = find_connectors(synthon_smi)
        iterator = tqdm(
            relevant_synthons[['synton_id', 'synthon_smiles', 'synton#', 'reaction_id', 'components']].values, 
            desc=f'Processing {synthon_smi} for reactions with 2 steps',
            total=len(relevant_synthons)
        )
        for synthon_id, synthon_smi, synthon_pos, reaction_id, n_components in iterator:
            possible_partners = self.synthon_df[(self.synthon_df['reaction_id'] == reaction_id) & (self.synthon_df['synton#'] != synthon_pos)]
            for partner_id, partner_smi, partner_pos in possible_partners[['synton_id', 'synthon_smiles', 'synton#']].values:
                partner_connectors = find_connectors(partner_smi)
                common_connectors = synthon_connectors & partner_connectors
                if len(common_connectors) == 0:
                    continue

                pos2id = {
                    synthon_pos: synthon_id,
                    partner_pos: partner_id,
                }
                pos2smi = {
                    synthon_pos: synthon_smi,
                    partner_pos: partner_smi,
                }
                combo_smi = connect_synthons(synthon_smi, partner_smi, connectors=common_connectors)
                try:
                    cleaned_smi = clean_smi(combo_smi)
                except Exception as e:
                    print(f'synthon_smiles={combo_smi}, exception: {e}')
                    continue
                fp = fpgen.GetFingerprint(Chem.MolFromSmiles(cleaned_smi))

                completed = (combo_smi == cleaned_smi)
                if completed and n_components != 2:
                    # logging.warning(f'Completed synthon {combo_smi} with n_components={n_components}, expected 2')
                    continue
                if not completed:
                    trace1 = get_react_trace_building_block(pos2smi.get(1), id=pos2id.get(1))
                    trace2 = get_react_trace_building_block(pos2smi.get(2), id=pos2id.get(2))
                    trace3 = get_react_trace_building_block(pos2smi.get(3), id=pos2id.get(3))
                    react_trace = get_react_trace_trimolecular(trace1, trace2, trace3, reaction_id, combo_smi)
                else:
                    trace1 = get_react_trace_building_block(pos2smi.get(1), id=pos2id.get(1))
                    trace2 = get_react_trace_building_block(pos2smi.get(2), id=pos2id.get(2))
                    react_trace = get_react_trace_bimolecular(trace1, trace2, reaction_id, combo_smi)

                results.append({
                    'fp': fp,
                    'react_trace': react_trace,
                    'smiles': cleaned_smi,
                    'synthon_smiles': combo_smi,
                    'checked': True,
                    'completed': completed,
                })
        return pd.DataFrame(results)
    
    def _compute_products_3steps(self, synthon_smi, react_tree, num_workers=1):
        results = []
        fpgen = AllChem.GetMorganGenerator(2, fpSize=self.fp_size)
        curr_connectors = find_connectors(synthon_smi)

        reaction_id = react_tree.tree['react_id']
        synthon_ids = [react_tree.tree['educts'][i]['prod_id'] for i in range(3)]
        synthon_smiles = [react_tree.tree['educts'][i]['product'] for i in range(3)]
        possible_positions = [i for i in [1, 2, 3] if synthon_smiles[i-1] is None]
        if len(possible_positions) != 1:
            logging.warning(f'Expected exactly one position to fill in synthon {synthon_smi}, found {len(possible_positions)}: {possible_positions}')
            return pd.DataFrame(results)
        position = possible_positions[0]
        possible_partners = self.synthon_df[(self.synthon_df['reaction_id'] == reaction_id) & (self.synthon_df['synton#'] == position)]

        for partner_id, partner_smi, n_components in possible_partners[['synton_id', 'synthon_smiles', 'components']].values:
            synthon_smiles[position-1] = partner_smi
            synthon_ids[position-1] = partner_id
            partner_connectors = find_connectors(partner_smi)
            combo_smi = connect_synthons(synthon_smi, partner_smi)

            if not curr_connectors == partner_connectors:
                logging.warning(f'Connector mismatch: {curr_connectors} vs {partner_connectors} for synthon {synthon_smi} and partner {partner_smi}')
                continue
            if len(find_connectors(combo_smi)) != 0 or n_components != 3:
                logging.warning(f'After connection still found connectors: {combo_smi} for synthon {synthon_smi} and partner {partner_smi}')
                continue

            fp = fpgen.GetFingerprint(Chem.MolFromSmiles(combo_smi))
            trace1 = get_react_trace_building_block(synthon_smiles[0], id=synthon_ids[0])
            trace2 = get_react_trace_building_block(synthon_smiles[1], id=synthon_ids[1])
            trace3 = get_react_trace_building_block(synthon_smiles[2], id=synthon_ids[2])
            react_trace = get_react_trace_trimolecular(trace1, trace2, trace3, reaction_id, combo_smi)

            results.append({
                'fp': fp,
                'react_trace': react_trace,
                'smiles': combo_smi,
                'synthon_smiles': combo_smi,
                'checked': True,
                'completed': True,
            })

        return pd.DataFrame(results)
