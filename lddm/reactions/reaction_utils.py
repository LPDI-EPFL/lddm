from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions
from typing import Collection, Set

from rdkit import Chem
from rdkit.Chem import Draw
import logging


#####################################################################
########################## Enamine reactions ########################
#####################################################################

def clean_smi(smi: str, replacement_atomic_num: int = 1) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")

    rw_mol = Chem.RWMol(mol)

    replacement_map = {
        92: replacement_atomic_num,  # U -> H (default replacement)
        93: replacement_atomic_num,  # Np -> H (default replacement)
        94: replacement_atomic_num,  # Pu -> H (default replacement)
    }

    for atom in rw_mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num in replacement_map:
            atom.SetAtomicNum(replacement_map[atomic_num])

    # Convert back to mol, remove Hs, sanitize
    mol = rw_mol.GetMol()
    mol = Chem.RemoveAllHs(mol)
    Chem.SanitizeMol(mol)
    return Chem.CanonSmiles(Chem.MolToSmiles(mol))

def find_connectors(smi: str) -> Set[str]:
    connectors = set()
    for conn in ['[U]', '[Np]', '[Pu]']:
        if conn in smi:
            connectors.add(conn)
    return connectors


def connect_synthons(
    smi1: str, 
    smi2: str, 
    connectors: Collection[str] = ['[U]', '[Np]', '[Pu]']
) -> str:
    # Replacing connectors with dummy atoms to enable RDKit functionality
    for i, conn in enumerate(connectors):
        repl = f'[*:{i+1}]'
        smi1 = smi1.replace(conn, repl)
        smi2 = smi2.replace(conn, repl)

    synthon1 = Chem.MolFromSmiles(smi1)
    synthon2 = Chem.MolFromSmiles(smi2)

    # Combining molecules based on the exit vectors
    combo = Chem.molzip(synthon1, synthon2)
    combo = Chem.RemoveAllHs(combo)
    return Chem.CanonSmiles(Chem.MolToSmiles(combo))

#####################################################################
######################### Explicit reactions ########################
#####################################################################

def run_bimolecular_reaction_smarts(
    reaction_smarts: str,
    bb_smiles_x: str,
    bb_smiles_y: str,
    explicit_hs: bool = False
) -> Collection[str]:
    rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
    bb_mol_x = Chem.MolFromSmiles(bb_smiles_x)
    bb_mol_y = Chem.MolFromSmiles(bb_smiles_y)
    reacts = (Chem.AddHs(bb_mol_x), Chem.AddHs(bb_mol_y)) if explicit_hs else (Chem.Mol(bb_mol_x), Chem.Mol(bb_mol_y))
    products = rxn.RunReactants(reacts)
    
    uniqps = {}
    for p in products:
        try:
            smi = Chem.MolToSmiles(Chem.RemoveAllHs(p[0]))
            uniqps[smi] = p[0]
        except Exception as e:
            continue
    if not uniqps:
        logging.debug(f'Reaction failed with {bb_smiles_x} and {bb_smiles_y}, reaction: {reaction_smarts}')
        return None
    uniqps = sorted(uniqps.keys())
    return uniqps

def run_unimolecular_reaction_smarts(
    reaction_smarts: str,
    bb_smiles: str,
    explicit_hs: bool = False
) -> Collection[str]:
    rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
    bb_mol = Chem.MolFromSmiles(bb_smiles)
    reacts = (Chem.AddHs(bb_mol) if explicit_hs else Chem.Mol(bb_mol),)
    products = rxn.RunReactants(reacts)

    uniqps = {}
    for p in products:
        try:
            smi = Chem.MolToSmiles(Chem.RemoveAllHs(p[0]))
        except Exception as e:
            continue
        uniqps[smi] = p[0]
    if not uniqps:
        logging.debug(f'Reaction failed with {bb_smiles}, reaction: {reaction_smarts}')
        return None
    
    uniqps = sorted(uniqps.keys())
    return uniqps


class ReactionTree:
    def __init__(self, react_trace: str):
        self.react_trace = react_trace
        self.tree = self._parse_trace(react_trace)

    def _parse_trace(self, trace: str) -> dict:
        if trace.startswith('<') and trace.endswith('>'):
            inner = trace[1:-1]
        else:
            inner = trace
        if ':' not in inner:
            educts_part, react_id, product_part = None, None, inner
        else:
            educts_part, react_id, product_part = inner.rsplit(':', 2)
        product, prod_id = product_part.split('-', 1) if '-' in product_part else (product_part, None)
        if product == 'NA':
            product = None
        if prod_id == 'NA':
            prod_id = None
        if educts_part is not None:
            educts = []
            buf = ''
            d = 0
            for ch in educts_part:
                if ch == '<':
                    d += 1
                elif ch == '>':
                    d -= 1
                if ch == ';' and d == 0:
                    educts.append(buf)
                    buf = ''
                else:
                    buf += ch
            if buf:
                educts.append(buf)
            nodes = []
            for ed in educts:
                ed = ed.strip()
                nodes.append(self._parse_trace(ed))
        else:
            nodes = None
        return {'educts': nodes, 'react_id': react_id, 'product': product, 'prod_id': prod_id}

    def depth(self) -> int:
        def d(node):
            eds = node.get('educts') or []
            if not eds:
                return 1
            return 1 + max(d(e) for e in eds)
        return d(self.tree)

    def draw(self):
        import matplotlib.pyplot as plt

        steps = []
        def collect(n):
            eds = n.get('educts') or []
            if eds:
                steps.append(( [e['product'] for e in eds], n['product'], n.get('react_id') ))
                for e in eds:
                    collect(e)
        collect(self.tree)
        for i, (educt_smiles, product_smiles, rid) in enumerate(steps,1):
            ed_mols = [Chem.MolFromSmiles(s) for s in educt_smiles]
            prod_mol = Chem.MolFromSmiles(product_smiles)
            n = len(ed_mols)
            fig, axs = plt.subplots(1, n+2, figsize=(4*(n+2),4))
            for j, mol in enumerate(ed_mols):
                axs[j].imshow(Draw.MolToImage(mol, size=(300,300)))
                axs[j].axis('off')
                axs[j].set_title(f"Educt {j+1}")
            axs[n].text(0.5,0.5,'→',fontsize=40,ha='center',va='center')
            axs[n].axis('off')
            axs[n+1].imshow(Draw.MolToImage(prod_mol, size=(300,300)))
            axs[n+1].axis('off')
            axs[n+1].set_title("Product")
            plt.suptitle(f"Step {i} (ID: {rid})")
            plt.tight_layout()
            plt.show()

def get_react_trace_building_block(smi, id=None):
    if smi is None:
        smi = 'NA'
    if id is None:
        id = 'NA'
    return f'<{smi}-{id}>'

def get_react_trace_unimolecular(tr, reaction_id, product_smi, prod_id=None):
    if id is None:
        id = 'NA'
    if prod_id is None:
        prod_id = 'NA'
    return f'<{tr}:{reaction_id}:{product_smi}-{prod_id}>'

def get_react_trace_bimolecular(tr1, tr2, reaction_id, product_smi, prod_id=None):
    if prod_id is None:
        prod_id = 'NA'
    educts = f'{tr1};{tr2}'
    return f'<{educts}:{reaction_id}:{product_smi}-{prod_id}>'

def get_react_trace_trimolecular(tr1, tr2, tr3, reaction_id, product_smi, prod_id=None):
    if prod_id is None:
        prod_id = 'NA'
    educts = f'{tr1};{tr2};{tr3}'
    return f'<{educts}:{reaction_id}:{product_smi}-{prod_id}>'
