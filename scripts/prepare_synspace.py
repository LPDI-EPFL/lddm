"""Convert a deterministic SynSpace subset to the existing LDDM format."""
import argparse
import bz2
import json
import pickle
import subprocess
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# whitead/synspace, MIT; see data/synspace/LICENSE and README.md.
SOURCE_COMMIT = 'b4b43e0437ec0002eed0fc8cec6d0c53974459ad'
REACTIONS = ('Schotten-Baumann_amide', 'sulfon_amide', 'reductive amination')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='SynSpace checkout at the documented commit')
    parser.add_argument('--output', type=Path, default=Path('data/synspace'))
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != SOURCE_COMMIT:
        parser.error(f'Expected SynSpace commit {SOURCE_COMMIT}, got {revision}')

    data = args.source / 'synspace' / 'rxn_data'
    reactions = json.loads((data / 'rxns.json').read_text())
    with bz2.open(data / 'blocks.pk.bz2', 'rb') as handle:
        blocks = pickle.load(handle)

    records, mapping, definitions, memberships = {}, {}, [], []
    fpgen = AllChem.GetMorganGenerator(2, fpSize=2048)
    for name in REACTIONS:
        definitions.append({'id': name, 'name': name, 'reaction': reactions[name], 'explicit_hs': False})
        mapping[name] = {}
        for role, mols in enumerate(blocks[name]):
            # Preserve upstream reaction/role membership; prefer small, connected blocks.
            smiles = set()
            for mol in mols:
                if Chem.SanitizeMol(mol, catchErrors=True) != 0:
                    continue
                smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
                if 5 <= mol.GetNumHeavyAtoms() <= 16 and '.' not in smi:
                    smiles.add(smi)
            chosen = sorted(smiles, key=lambda s: (Chem.MolFromSmiles(s).GetNumHeavyAtoms(), s))
            if not chosen:
                raise ValueError(f'No building blocks for {name}, role {role}')
            mapping[name][role] = set()
            for smi in chosen:
                if smi not in records:
                    records[smi] = {
                        'id': f'synspace_{len(records):04d}', 'smiles': smi,
                        'synthon_smiles': smi, 'completed': False,
                        'fp': fpgen.GetFingerprint(Chem.MolFromSmiles(smi)),
                    }
                bb_id = records[smi]['id']
                mapping[name][role].add(bb_id)
                memberships.append({'reaction_id': name, 'reactant_role': role, 'building_block_id': bb_id})

    args.output.mkdir(parents=True, exist_ok=True)
    for filename in ('LICENSE', 'NOTICE.txt'):
        (args.output / filename).write_text((args.source / filename).read_text())
    frame = pd.DataFrame(records.values())
    frame.to_pickle(args.output / 'building_blocks.pkl')
    frame.drop(columns='fp').to_csv(args.output / 'building_blocks.csv', index=False)
    pd.DataFrame(memberships).to_csv(args.output / 'reaction_to_building_blocks.csv', index=False)
    with open(args.output / 'reaction_to_building_blocks.pkl', 'wb') as handle:
        pickle.dump(mapping, handle, protocol=4)
    (args.output / 'reactions.json').write_text(json.dumps(definitions, indent=2) + '\n')
    print(f'Wrote {len(frame)} building blocks, {len(definitions)} reactions and {len(memberships)} role memberships to {args.output}')


if __name__ == '__main__':
    main()
