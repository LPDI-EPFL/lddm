import torch
from rdkit import Chem

from lddm import constants
from lddm.data.misc import protein_letters_1to3


def remove_dummy_atoms(rdmol, sanitize=False):
    # find exit atoms to be removed
    dummy_inds = []
    for a in rdmol.GetAtoms():
        if a.GetSymbol() == '*':
            dummy_inds.append(a.GetIdx())

    dummy_inds = sorted(dummy_inds, reverse=True)
    new_mol = Chem.EditableMol(rdmol)
    for idx in dummy_inds:
        new_mol.RemoveAtom(idx)
    new_mol = new_mol.GetMol()
    if sanitize:
        Chem.SanitizeMol(new_mol)
    return new_mol


def build_molecule(coords, atom_types, bonds=None, bond_types=None,
                   atom_props=None, atom_decoder=None, bond_decoder=None):
    """
    Build RDKit molecule with given bonds
    :param coords: N x 3
    :param atom_types: N
    :param bonds: 2 x N_bonds
    :param bond_types: N_bonds
    :param atom_props: Dict, key: property name, value: list of float values (N,)
    :param atom_decoder: list
    :param bond_decoder: list
    :return: RDKit molecule
    """
    if atom_decoder is None:
        atom_decoder = constants.atom_decoder
    if bond_decoder is None:
        bond_decoder = constants.bond_decoder
    assert len(coords) == len(atom_types)
    assert bonds is None or bonds.size(1) == len(bond_types)

    mol = Chem.RWMol()
    for i, atom in enumerate(atom_types):
        element = atom_decoder[atom.item()]
        charge = None
        explicitHs = None

        if len(element) > 1 and element.endswith('H'):
            explicitHs = 1
            element = element[:-1]
        elif element.endswith('+'):
            charge = 1
            element = element[:-1]
        elif element.endswith('-'):
            charge = -1
            element = element[:-1]

        if element in {'NOATOM', 'MASK'}:
            # element = 'Xe'  # debug
            element = '*'

        a = Chem.Atom(element)

        if explicitHs is not None:
            a.SetNumExplicitHs(explicitHs)
        if charge is not None:
            a.SetFormalCharge(charge)

        if atom_props is not None:
            for k, vals in atom_props.items():
                a.SetDoubleProp(k, vals[i].item())

        mol.AddAtom(a)

    # add coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (coords[i, 0].item(),
                                 coords[i, 1].item(),
                                 coords[i, 2].item()))
    mol.AddConformer(conf)

    # add bonds
    if bonds is not None:
        for bond, bond_type in zip(bonds.T, bond_types):
            bond_type = bond_decoder[bond_type]
            src = bond[0].item()
            dst = bond[1].item()

            if bond_type in {'NOBOND', 'MASK'} or mol.GetAtomWithIdx(src).GetSymbol() == '*' or mol.GetAtomWithIdx(dst).GetSymbol() == '*':
                continue
            if mol.GetBondBetweenAtoms(src, dst) is not None:
                assert mol.GetBondBetweenAtoms(src, dst).GetBondType() == bond_type, \
                    "Trying to assign two different types to the same bond."
                continue

            if bond_type is None or src == dst:
                continue
            mol.AddBond(src, dst, bond_type)

    mol = remove_dummy_atoms(mol, sanitize=False)
    return mol


def pocket_to_rdkit(pocket, pocket_representation, atom_encoder=None,
                    atom_decoder=None, aa_decoder=None,
                    aa_atom_index=None):

    rdpockets = []
    for i in torch.unique(pocket['mask']):

        pdb_infos = []

        if pocket_representation == 'CA+':

            node_coord = pocket['x'][pocket['mask'] == i]
            h = pocket['one_hot'][pocket['mask'] == i]
            atom_mask = pocket['atom_mask'][pocket['mask'] == i]

            aa_types = [aa_decoder[b] for b in h.argmax(-1)]
            side_chain_vec = pocket['v'][pocket['mask'] == i]

            coord = []
            atom_types = []
            for resi, (xyz, aa, vec, am) in enumerate(zip(node_coord, aa_types, side_chain_vec, atom_mask)):

                # CA not treated differently with updated atom dictionary
                for atom_name, idx in aa_atom_index[aa].items():

                    if ~am[idx]:
                        # warnings.warn(f"Missing atom {atom_name} in {aa}:{resi}")
                        continue

                    coord.append(xyz + vec[idx])
                    atom_types.append(atom_name[0])

                    info = Chem.AtomPDBResidueInfo()
                    # info.SetChainId('A')
                    info.SetResidueName(protein_letters_1to3[aa])
                    info.SetResidueNumber(resi + 1)
                    info.SetOccupancy(1.0)
                    info.SetTempFactor(0.0)
                    info.SetName(f' {atom_name:<3}')
                    pdb_infos.append(info)

            coord = torch.stack(coord, dim=0)

        else:
            raise NotImplementedError(f"{pocket_representation} residue representation not supported")

        atom_types = torch.tensor([atom_encoder[a] for a in atom_types])
        rdmol = build_molecule(coord, atom_types, atom_decoder=atom_decoder)

        if len(pdb_infos) == len(rdmol.GetAtoms()):
            for a, info in zip(rdmol.GetAtoms(), pdb_infos):
                a.SetPDBResidueInfo(info)

        if 'name' in pocket:
            rdmol.SetProp('_PDB', f"{pocket['name'][i]}")

        rdpockets.append(rdmol)

    return rdpockets


def mols_to_pdbfile(rdmols, filename, flavor=0):
    pdb_str = ""
    for i, mol in enumerate(rdmols):
        pdb_str += f"MODEL{i + 1:>9}\n"
        block = Chem.MolToPDBBlock(mol, flavor=flavor)
        block = "\n".join(block.split("\n")[:-2])  # remove END
        pdb_str += block + "\n"
        pdb_str += f"ENDMDL\n"
    pdb_str += f"END\n"

    with open(filename, 'w') as f:
        f.write(pdb_str)

    return pdb_str


def mol_as_pdb(rdmol, filename=None, bfactor=None):

    _rdmol = Chem.Mol(rdmol)  # copy
    for a in _rdmol.GetAtoms():
        a.SetIsAromatic(False)
    for b in _rdmol.GetBonds():
        b.SetIsAromatic(False)

    if bfactor is not None:
        for a in _rdmol.GetAtoms():
            val = a.GetPropsAsDict()[bfactor]

            info = Chem.AtomPDBResidueInfo()
            info.SetResidueName('UNL')
            info.SetResidueNumber(1)
            info.SetName(f' {a.GetSymbol():<3}')
            info.SetIsHeteroAtom(True)
            info.SetOccupancy(1.0)
            info.SetTempFactor(val)
            a.SetPDBResidueInfo(info)

    pdb_str = Chem.MolToPDBBlock(_rdmol)

    if filename is not None:
        with open(filename, 'w') as f:
            f.write(pdb_str)

    return pdb_str
