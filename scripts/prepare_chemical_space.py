"""Prepare building-block fingerprints and reaction-role mappings for LDDM."""
import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions

from lddm.reactions.process_reactions import ReactionsProcessor


def prepare_space(blocks, reactions, output, memberships=None):
    if not {'id', 'smiles'}.issubset(blocks.columns):
        raise ValueError('Building-block CSV requires id and smiles columns')
    blocks = blocks[['id', 'smiles']].copy()
    if blocks.empty or blocks.isna().any().any():
        raise ValueError('Building blocks must contain nonempty IDs and SMILES')
    blocks['id'] = blocks['id'].astype(str)
    if blocks.id.duplicated().any() or blocks.id.str.strip().eq('').any():
        raise ValueError('Building-block IDs must be unique and nonempty')
    fpgen = AllChem.GetMorganGenerator(2, fpSize=2048)
    molecules = []
    for row in blocks.itertuples():
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError(f'Invalid or disconnected building block: {row.id}')
        molecules.append(Chem.RemoveHs(mol))
    blocks['smiles'] = [Chem.MolToSmiles(mol) for mol in molecules]
    if blocks.smiles.duplicated().any():
        raise ValueError('Duplicate canonical SMILES: keep one ID per building block')
    blocks['synthon_smiles'] = blocks.smiles
    blocks['completed'] = False
    blocks['fp'] = [fpgen.GetFingerprint(mol) for mol in molecules]

    definitions = []
    for reaction in reactions:
        if not {'id', 'reaction'}.issubset(reaction):
            raise ValueError('Each reaction requires id and reaction (SMARTS)')
        rxn = rdChemReactions.ReactionFromSmarts(reaction['reaction'])
        if (rxn is None or rxn.GetNumReactantTemplates() != 2 or
                rxn.GetNumProductTemplates() != 1 or
                len(reaction['reaction'].split('>>')[0].split('.')) != 2):
            raise ValueError('Custom spaces require two reactants and one product per reaction')
        definitions.append(dict(reaction, id=str(reaction['id']),
                                explicit_hs=reaction.get('explicit_hs', False)))
    ids = [r['id'] for r in definitions]
    if not ids or len(set(ids)) != len(ids) or any(not name.strip() for name in ids):
        raise ValueError('Provide reactions with unique, nonempty IDs')
    # The processor adds runtime objects to its definitions; keep the JSON serializable.
    processor = ReactionsProcessor([dict(r) for r in definitions], blocks, fp_size=2048)
    if memberships is None:
        processor.load_reactant_to_building_blocks()
    else:
        required = ['reaction_id', 'reactant_role', 'building_block_id']
        if not set(required).issubset(memberships.columns):
            raise ValueError(f'Membership CSV requires {", ".join(required)}')
        if memberships[required].isna().any().any():
            raise ValueError('Membership CSV contains missing values')
        for reaction in processor.reactions.values():
            for reactant in reaction['reactants']:
                reactant.allowed_building_blocks = set()
        for row in memberships[required].drop_duplicates().itertuples(index=False):
            reaction = processor.reactions.get(str(row.reaction_id))
            if reaction is None or row.reactant_role not in (0, 1):
                raise ValueError(f'Unknown reaction or role: {row.reaction_id}, {row.reactant_role}')
            block_id = str(row.building_block_id)
            reactant = reaction['reactants'][int(row.reactant_role)]
            if (block_id not in processor.bb_id_to_smi or
                    not reactant.matches_mol(processor.bb_id_to_smi[block_id])):
                raise ValueError(f'Building block {block_id} does not match the specified reaction role')
            reactant.allowed_building_blocks.add(block_id)

    mapping, rows = {}, []
    for reaction_id, reaction in processor.reactions.items():
        mapping[reaction_id] = {}
        for reactant in reaction['reactants']:
            allowed = reactant.allowed_building_blocks
            if not allowed:
                raise ValueError(f'No building blocks for {reaction_id}, role {reactant.id}')
            mapping[reaction_id][reactant.id] = set(allowed)
            rows.extend({'reaction_id': reaction_id, 'reactant_role': reactant.id,
                         'building_block_id': block_id} for block_id in sorted(allowed))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    blocks.to_pickle(output / 'building_blocks.pkl')
    blocks.drop(columns='fp').to_csv(output / 'building_blocks.csv', index=False)
    (output / 'reactions.json').write_text(json.dumps(definitions, indent=2) + '\n')
    pd.DataFrame(rows).to_csv(output / 'reaction_to_building_blocks.csv', index=False)
    with (output / 'reaction_to_building_blocks.pkl').open('wb') as handle:
        pickle.dump(mapping, handle, protocol=4)
    print(f'Wrote {len(blocks)} building blocks, {len(definitions)} reactions and {len(rows)} role memberships to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--building-blocks', type=Path, required=True, help='CSV with id and smiles columns')
    parser.add_argument('--reactions', type=Path, required=True, help='JSON list of reaction SMARTS definitions')
    parser.add_argument('--memberships', type=Path, help='Optional CSV of curated reaction-role assignments')
    parser.add_argument('--output', type=Path, default=Path('data/chemical_spaces/custom'))
    args = parser.parse_args()
    memberships = (pd.read_csv(args.memberships, dtype={'reaction_id': str, 'building_block_id': str})
                   if args.memberships else None)
    prepare_space(pd.read_csv(args.building_blocks, dtype={'id': str, 'smiles': str}),
                  json.loads(args.reactions.read_text()), args.output, memberships)


if __name__ == '__main__':
    main()
