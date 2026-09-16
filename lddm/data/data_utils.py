from itertools import accumulate, chain
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from torch_geometric.utils import to_dense_adj
from Bio.PDB import StructureBuilder, Model, PDBParser
from scipy.ndimage import gaussian_filter
from rdkit.Chem.rdmolops import FragmentOnBRICSBonds, GetMolFrags
from typing import Dict, Union, List

from lddm.constants import FLOAT_TYPE, INT_TYPE
from lddm import utils
from lddm.data.misc import protein_letters_3to1, is_aa
from lddm.scatter import scatter_mean
from lddm.config.data import FeaturizationConfig


class TensorDict(dict):
    def __init__(self, **kwargs):
        super(TensorDict, self).__init__(**kwargs)

    def _apply(self, func: str, *args, **kwargs):
        """ Apply function to all tensors. """
        for k, v in self.items():
            if torch.is_tensor(v):
                self[k] = getattr(v, func)(*args, **kwargs)
        return self

    def cuda(self):
        return self.to('cuda')

    def cpu(self):
        return self.to('cpu')
    
    def to(self, device):
        return self._apply("to", device)
    
    def detach(self):
        return self._apply("detach")
    
    def copy(self):
        raise NotImplementedError(
            f"By default, dict.copy returns a shallow copy. To avoid unintended " \
            f"behavior use {type(self).__name__}.shallow_copy or " \
            f"{type(self).__name__}.deepcopy instead."
        )
    
    def shallow_copy(self):
        """ 
        Return a shallow copy of the dictionary similar to dict.copy().
        https://docs.python.org/3/library/stdtypes.html#dict.copy
        """
        data = super().copy()
        return type(self)(**data)

    def deepcopy(self):
        data = {k: v.clone() if torch.is_tensor(v) else deepcopy(v)
                for k, v in self.items()}
        return type(self)(**data)
    
    def to_dict(self):
        return dict(**self)
    
    @classmethod
    def from_dict(cls, input_dict, *args, **kwargs):
        return cls(**input_dict)

    def __repr__(self):
        def val_to_str(val):
            if isinstance(val, torch.Tensor):
                return "%r" % list(val.size())
            if isinstance(val, list):
                return "[%r,]" % len(val)
            else:
                return "?"

        return f"{type(self).__name__}({', '.join(f'{k}={val_to_str(v)}' for k, v in self.items())})"


class MoleculeTensorDict(TensorDict):
    NODE_MASK = 'mask'
    EDGE_MASK = 'bond_mask'

    # graph-level features
    GLOBAL_FEATURES = {'name', 'size', 'n_bonds'}

    # node-level features
    NODE_FEATURES = {'x', 'h'} | {NODE_MASK}
    NODE_FEATURES = NODE_FEATURES | {'one_hot'}  # legacy node feature keys

    # edge-level features
    EDGE_FEATURES = {'e'} | {EDGE_MASK}
    EDGE_FEATURES = EDGE_FEATURES | {'bond_one_hot'}  # legacy edge feature keys

    # edges
    EDGE_INDICES = {'bonds'}

    EXPECTED_KEYS = {NODE_MASK, EDGE_MASK} | GLOBAL_FEATURES | NODE_FEATURES | EDGE_FEATURES | EDGE_INDICES

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_keys()

    @classmethod
    def from_dict(cls, input_dict, strict=True):
        if strict:
            cls._check_keys(input_dict.keys())
        _dict = {}
        for k, v in input_dict.items():
            if k not in cls.EXPECTED_KEYS:
                print(f"[{cls.__name__}] Ignoring '{k}' which is not supported.")
                continue
            _dict[k] = v
        return super().from_dict(_dict)
    
    @classmethod
    def _check_keys(cls, keys):
        if not hasattr(cls, 'EXPECTED_KEYS'):
            return
        invalid_keys = set(keys) - cls.EXPECTED_KEYS
        assert len(invalid_keys) <= 0, f"Key(s) {invalid_keys} are not valid for object of type {cls.__name__}"

    def check_keys(self, keys=None):
        if keys is None:
            keys = set(self.keys())
        self._check_keys(keys)
    
    def update(self, *args, **kwargs):
        out = super().update(*args, **kwargs)
        self.check_keys()
        return out
    
    def __setitem__(self, key, value):
        self.check_keys({key})
        return super().__setitem__(key, value)


class Ligand(MoleculeTensorDict):
    """
    Dictionary-like container for ligands that supports masking.
    """

    # graph-level features
    GLOBAL_FEATURES = MoleculeTensorDict.GLOBAL_FEATURES | {'size', 'smiles', 'affinity'}

    # node-level features
    NODE_FEATURES = MoleculeTensorDict.NODE_FEATURES | {'known_x', 'known_h', 'true_x', 'true_h', 'virtual_mask', 'fragments'}

    # edge-level features
    EDGE_FEATURES = MoleculeTensorDict.EDGE_FEATURES | {'known_e', 'true_e'}

    # all keys
    EXPECTED_KEYS = MoleculeTensorDict.EXPECTED_KEYS | GLOBAL_FEATURES | NODE_FEATURES | EDGE_FEATURES

    _maskable_variables = {"x", "h", "e"}

    @classmethod
    def from_rdmol(cls, rdmol, config: FeaturizationConfig):
        return cls(**prepare_ligand(rdmol, config=config))

    @classmethod
    def from_sdf(cls, sdf_file, config: FeaturizationConfig):
        rdmol = Chem.SDMolSupplier(str(sdf_file))[0]
        return cls.from_rdmol(rdmol, config=config)
        
    @classmethod
    def empty(cls, config: FeaturizationConfig, device='cpu'):
        ligand = Ligand(**{
            'x': torch.zeros(0, 3, dtype=FLOAT_TYPE, device=device),
            'one_hot': torch.zeros(0, len(config.atom_encoder), dtype=FLOAT_TYPE, device=device),
            'mask': torch.zeros(0, dtype=INT_TYPE, device=device),
            'bonds': torch.zeros(2, 0, dtype=INT_TYPE, device=device),
            'bond_one_hot': torch.zeros(0, len(config.bond_encoder), dtype=FLOAT_TYPE, device=device),
            'bond_mask': torch.zeros(0, dtype=INT_TYPE, device=device),
            'size': torch.tensor([0], dtype=INT_TYPE, device=device),
            'n_bonds': torch.tensor([0], dtype=INT_TYPE, device=device),
        })
        if config.compute_fragment_mask is not None:
            ligand['fragments'] = torch.zeros(0, dtype=INT_TYPE, device=device)
        return ligand

    def register_known_variables(self, *, known_x=None, true_x=None, known_h=None, true_h=None, known_e=None, true_e=None):
        """ 
        Register binary masks for the known parts of the ligand (e.g. known_x) 
        and corresponding fixed, true values (e.g. true_x).
        """

        def _register_known_variable(name, known_mask=None, known_values=None):
            assert not ((known_mask is None) ^ (known_values is None)), "Invalid mask specification"
            if known_x is not None:
                self[f"known_{name}"] = known_mask
                self[f"true_{name}"] = known_values

        _register_known_variable("x", known_x, true_x)
        _register_known_variable("h", known_h, true_h)
        _register_known_variable("e", known_e, true_e)
        
    def insert_known_variables(self):
        for k in self._maskable_variables:
            if f"known_{k}" in self.keys():
                known_mask = self[f"known_{k}"]
                known_values = self[f"true_{k}"]
                self.replace_values(self[k], known_values, known_mask)
        return self
    
    @staticmethod
    def replace_values(target_tensor, new_values, where):
        target_tensor[where] = new_values[where]
        return target_tensor

    def rigid_transform(self, rot=None, tran=None):
        """
        Apply a roto-translation in-place.
        :param rot: (b, 3, 3) rotation matrix per batch element.
        :param trans: (b, 3) translation vector per batch element.
        """

        batch_size = 1 if isinstance(self['size'], int) else len(self['size'])
        rot, tran = unify_transform(rot, tran, batch_size)

        # repeat for items in the same batch
        rot = rot[self['mask']]
        tran = tran[self['mask']]

        self['x'] = torch.einsum('boi,bi->bo', rot, self['x']) + tran
    

class Residues(MoleculeTensorDict):
    """
    Dictionary-like container for residues that supports some basic transformations.
    """
    
    # node-level features
    NODE_FEATURES = MoleculeTensorDict.NODE_FEATURES | {'v', 'atom_mask'}

    # all keys
    EXPECTED_KEYS = MoleculeTensorDict.EXPECTED_KEYS | NODE_FEATURES

    @classmethod
    def from_pdb(cls, pdb_file, config: FeaturizationConfig, autoselect_ligand: Chem.Mol | None = None):
    
        pdb_model = PDBParser(QUIET=True).get_structure('', pdb_file)[0]

        # Find interacting pocket residues based on distance cutoff
        if autoselect_ligand is None:
            pocket_residues = [r for r in pdb_model.get_residues()]
        else:
            pocket_residues = extract_pocket_residues(pdb_model, autoselect_ligand, config=config)

        pocket, _ = prepare_pocket(pocket_residues, config=config)
        return cls(**pocket)

    @classmethod
    def empty(cls, config: FeaturizationConfig, device='cpu'):
        pocket = Residues(**{
            'x': torch.zeros(0, 3, dtype=FLOAT_TYPE, device=device),
            'one_hot': torch.zeros(0, len(config.aa_encoder), dtype=FLOAT_TYPE, device=device),
            'size': torch.tensor([0], dtype=INT_TYPE, device=device),
            'mask': torch.zeros(0, dtype=INT_TYPE, device=device),
            'bonds': torch.zeros(2, 0, dtype=INT_TYPE, device=device),
            'bond_one_hot': torch.zeros(0, len(config.residue_bond_encoder), dtype=FLOAT_TYPE, device=device),
            'bond_mask': torch.zeros(0, dtype=INT_TYPE, device=device),
            'n_bonds': torch.tensor([0], dtype=INT_TYPE, device=device),
            'v': torch.zeros(0, config.max_num_atoms_per_residue, 3, dtype=FLOAT_TYPE, device=device),
            'atom_mask': torch.zeros(0, config.max_num_atoms_per_residue, dtype=bool, device=device),
        })

        return pocket

    @property
    def batch_size(self):
        if 'size' in self:
            return 1 if isinstance(self['size'], int) else len(self['size'])
        return None
    
    def get_com(self):
        return scatter_mean(
            self['x'], self['mask'], dim=0, 
            out=utils.create_scatter_output_tensor(like=self['x'], dim_size=self.batch_size, fill_value=0.0)
        )

    def center(self):
        com = self.get_com()
        self['x'] = self['x'] - com[self['mask']]
        return com

    def rigid_transform(self, rot=None, tran=None):
        """
        Apply a roto-translation in-place.
        :param rot: (b, 3, 3) rotation matrix per batch element.
        :param trans: (b, 3) translation vector per batch element.
        """

        batch_size = 1 if isinstance(self['size'], int) else len(self['size'])
        rot, tran = unify_transform(rot, tran, batch_size)

        # repeat for items in the same batch
        rot = rot[self['mask']]
        tran = tran[self['mask']]

        self['x'] = torch.einsum('boi,bi->bo', rot, self['x']) + tran
        self['v'] = torch.einsum('boi,bai->bao', rot, self['v'])


def unify_transform(rot, tran, batch_size):
    """
    Replaces None values and makes sure that a rigid transform has the expected 
    shape by broadcasting tensors accordingly.
    rot = (batch_size, 3, 3)
    tran = (batch_size, 3)
    """
    if rot is None:
        rot = torch.eye(3)
    if tran is None:
        tran = torch.zeros(3)

    # add batch dimension if missing
    rot = rot.view(-1, 3, 3)
    tran = tran.view(-1, 3)
    if rot.size(0) == 1:
        rot = rot.expand(batch_size, 3, 3)
    if tran.size(0) == 1:
        tran = tran.expand(batch_size, 3)

    return rot, tran


def collate_entity(
        batch,
        *,
        global_types=Ligand.GLOBAL_FEATURES | Residues.GLOBAL_FEATURES,
        index_types=Ligand.EDGE_INDICES | Residues.EDGE_INDICES,
        batch_mask_key=MoleculeTensorDict.NODE_MASK, 
        edge_mask_key=MoleculeTensorDict.EDGE_MASK,
    ):

    out = {}
    for prop in batch[0].keys():

        if prop in {'name', 'smiles'}:  # exception for string-variables
            out[prop] = [x[prop] for x in batch if prop in x]

        elif prop in global_types:
            out[prop] = torch.tensor([x[prop] for x in batch])

        elif prop in index_types:
            # index offset
            offset = list(accumulate([x['size'] for x in batch], initial=0))
            out[prop] = torch.cat([x[prop] + offset[i] for i, x in enumerate(batch)], dim=1)

        # elif prop == 'residues':
        #     out[prop] = list(chain.from_iterable(x[prop] for x in batch))

        elif prop in {batch_mask_key, edge_mask_key}:
            pass  # batch masks will be written later

        else:
            out[prop] = torch.cat([x[prop] for x in batch], dim=0)

        # Create batch masks
        # make sure indices in batch start at zero (needed for torch_scatter)
        if prop == 'x':
            out[batch_mask_key] = torch.cat([i * torch.ones(len(x[prop]), dtype=torch.int64, device=x[prop].device)
                                             for i, x in enumerate(batch)], dim=0)
        if prop == 'bond_one_hot':
            out[edge_mask_key] = torch.cat([i * torch.ones(len(x[prop]), dtype=torch.int64, device=x[prop].device)
                                            for i, x in enumerate(batch)], dim=0)

    return out


def split_entity(
        batch,
        *,
        global_types=Ligand.GLOBAL_FEATURES | Residues.GLOBAL_FEATURES,
        edge_types=Ligand.EDGE_FEATURES | Residues.EDGE_FEATURES,
        index_types=Ligand.EDGE_INDICES | Residues.EDGE_INDICES,
        batch_mask_key=MoleculeTensorDict.NODE_MASK, 
        edge_mask_key=MoleculeTensorDict.EDGE_MASK,
    ):
    """ Splits a batch into items and returns a list. """

    batch_mask = batch[batch_mask_key]
    edge_mask = batch[edge_mask_key]
    sizes = batch['size'] if 'size' in batch else torch.unique(batch_mask, return_counts=True)[1].tolist()

    batch_size = len(sizes)
    out = {}
    for prop in batch.keys():

        if prop in global_types:
            out[prop] = batch[prop]  # already a list, no split required

        elif prop in index_types:
            offsets = list(accumulate(sizes[:-1], initial=0))
            out[prop] = utils.batch_to_list_for_indices(batch[prop], edge_mask, offsets, batch_size=batch_size)

        elif prop in edge_types:
            out[prop] = utils.batch_to_list(batch[prop], edge_mask, batch_size=batch_size)

        else:
            out[prop] = utils.batch_to_list(batch[prop], batch_mask, batch_size=batch_size)

    # Without batch mask should be always zero
    for prop in [batch_mask_key, edge_mask_key]:
        out[prop] = [values * 0 for values in out[prop]]

    out = [{k: v[i] for k, v in out.items()} for i in range(batch_size)]
    return out


def repeat_items(batch: TensorDict, repeats: int) -> TensorDict:
    device = batch['x'].device
    batch_list = split_entity(batch)
    out = collate_entity([x for _ in range(repeats) for x in batch_list])
    return type(batch)(**out).to(device)


def extract_substructure(ligand: Ligand, atoms_to_keep: torch.Tensor) -> Ligand:
    """
    Returns a new ligand keeping only the atoms marked True in atoms_to_keep.
    :param ligand: the original ligand
    :atoms_to_keep: Boolean mask with n_atoms elements
    :returns: new ligand object containing the extracted substructure
    """
    row, col = ligand["bonds"]
    bonds_to_keep = atoms_to_keep[row] & atoms_to_keep[col]

    # Re-index bonds correctly
    new_size = atoms_to_keep.long().sum().item()
    old_to_new = -torch.ones(atoms_to_keep.size(0), dtype=torch.long, device=row.device)
    old_to_new[atoms_to_keep] = torch.arange(new_size, device=row.device)

    new_bonds = ligand['bonds'][:, bonds_to_keep]
    new_bonds = old_to_new[new_bonds]

    substructure = {}    
    for prop in ligand.keys():

        # Special cases
        if prop == "bonds":
            substructure[prop] = new_bonds
        elif prop == "size":
            substructure[prop] = new_size
        elif prop == "n_bonds":
            substructure[prop] = bonds_to_keep.long().sum().item()
        elif prop == "name":
            substructure[prop] = ligand['name']
        elif prop == "smiles":
            substructure[prop] = '[substructure of] ' + ligand['smiles']
        elif prop == "affinity":
            substructure[prop] = 0.0  # affinity of substructure isn't known
        
        # Default cases
        elif prop in Ligand.NODE_FEATURES:
            substructure[prop] = ligand[prop][atoms_to_keep]
        elif prop in Ligand.EDGE_FEATURES:
            substructure[prop] = ligand[prop][bonds_to_keep]

    return Ligand.from_dict(substructure)


def get_side_chain_vectors(res, index_dict, size=None):
    if size is None:
        size = max([x for aa in index_dict.values() for x in aa.values()]) + 1

    resname = protein_letters_3to1[res.get_resname()]

    out = np.zeros((size, 3))
    mask = np.zeros(size, dtype=bool)
    for atom in res.get_atoms():
        if atom.get_name() in index_dict[resname]:
            idx = index_dict[resname][atom.get_name()]
            out[idx] = atom.get_coord() - res['CA'].get_coord()
            mask[idx] = True
        # else:
        #     if atom.get_name() != 'CA' and not atom.get_name().startswith('H'):
        #         print(resname, atom.get_name())

    return out, mask


def prepare_pocket(biopython_residues, config: FeaturizationConfig):

    # sort residues
    biopython_residues = sorted(biopython_residues, key=lambda x: (x.parent.id, x.id[1]))

    if config.pocket_representation == 'CA+':
        ca_coords = np.zeros((len(biopython_residues), 3))
        ca_types = np.zeros(len(biopython_residues), dtype='int64')

        vec_feats = np.zeros((len(biopython_residues), config.max_num_atoms_per_residue, 3), dtype='float32')
        atom_mask_feats = np.zeros((len(biopython_residues), config.max_num_atoms_per_residue), dtype='bool')

        edges = []  # CA-CA and CA-side_chain
        edge_types = []
        last_res_id = None
        for i, res in enumerate(biopython_residues):
            aa = config.amino_acid_encoder[protein_letters_3to1[res.get_resname()]]
            ca_coords[i, :] = res['CA'].get_coord()
            ca_types[i] = aa

            vec_feats[i], atom_mask_feats[i] = get_side_chain_vectors(res, config.aa_atom_index, config.max_num_atoms_per_residue)

            # add edges between contiguous CA atoms
            if i > 0 and res.id[1] == last_res_id + 1:
                edges.append((i - 1, i))
                edge_types.append(config.residue_bond_encoder['CA-CA'])

            last_res_id = res.id[1]

        # Coordinates
        pocket_coords = torch.from_numpy(ca_coords)

        # Features
        pocket_onehot = F.one_hot(torch.from_numpy(ca_types),
                                  num_classes=len(config.amino_acid_encoder))

        vector_features = torch.from_numpy(vec_feats)
        atom_mask_features = torch.from_numpy(atom_mask_feats)

        # Bonds
        if len(edges) < 1:
            edges = torch.empty(2, 0)
            edge_types = torch.empty(0, len(config.residue_bond_encoder))
        else:
            edges = torch.tensor(edges).T
            edge_types = F.one_hot(torch.tensor(edge_types),
                                   num_classes=len(config.residue_bond_encoder))

    else:
        raise NotImplementedError(
            f"Pocket representation '{config.pocket_representation}' not implemented")

    # pocket_ids = [f'{res.parent.id}:{res.id[1]}' for res in biopython_residues]

    pocket = {
        'x': pocket_coords.to(dtype=FLOAT_TYPE),
        'one_hot': pocket_onehot.to(dtype=FLOAT_TYPE),
        # 'ids': pocket_ids,
        'size': torch.tensor([len(pocket_coords)], dtype=INT_TYPE),
        'mask': torch.zeros(len(pocket_coords), dtype=INT_TYPE),
        'bonds': edges.to(INT_TYPE),
        'bond_one_hot': edge_types.to(FLOAT_TYPE),
        'bond_mask': torch.zeros(edges.size(1), dtype=INT_TYPE),
        'n_bonds': torch.tensor([len(edge_types)], dtype=INT_TYPE),
    }

    if vector_features is not None:
        pocket['v'] = vector_features.to(dtype=FLOAT_TYPE)
        pocket['atom_mask'] = atom_mask_features

    return pocket, biopython_residues


def encode_atom(rd_atom, atom_encoder):
    element = rd_atom.GetSymbol().capitalize()

    explicitHs = rd_atom.GetNumExplicitHs()
    if explicitHs == 1 and f'{element}H' in atom_encoder:
        return atom_encoder[f'{element}H']

    charge = rd_atom.GetFormalCharge()
    if charge == 1 and f'{element}+' in atom_encoder:
        return atom_encoder[f'{element}+']
    if charge == -1 and f'{element}-' in atom_encoder:
        return atom_encoder[f'{element}-']

    return atom_encoder[element]


def get_fragment_mask(rdmol):
    # make a copy so that the original molecule won't be modified
    rdmol = Chem.Mol(rdmol)
    Chem.SanitizeMol(rdmol)  # sanitization is necessary for proper BRICS decomposition

    for atom in rdmol.GetAtoms():
        atom.SetIntProp('_InitialIndex', atom.GetIdx())

    fragments = GetMolFrags(FragmentOnBRICSBonds(rdmol), asMols=True)
    fragment_mask = torch.zeros(rdmol.GetNumAtoms())
    for fragment_idx, fragment in enumerate(fragments):
        for atom in fragment.GetAtoms():
            props = atom.GetPropsAsDict()
            idx = props.get('_InitialIndex')
            if idx is not None:
                fragment_mask[idx] = fragment_idx

    return fragment_mask


def prepare_ligand(rdmol: Chem.Mol, config: FeaturizationConfig):

    Chem.SanitizeMol(rdmol)

    # remove H atoms if not in atom_encoder
    if 'H' not in config.atom_encoder:
        rdmol = Chem.RemoveAllHs(rdmol)

    if config.kekulize:
        # NOTE: this can affect other RDKit functions such as BRICS decomposition
        Chem.Kekulize(rdmol, clearAromaticFlags=True)

    # Coordinates
    ligand_coord = rdmol.GetConformer().GetPositions()
    ligand_coord = torch.from_numpy(ligand_coord)

    # Features
    ligand_onehot = F.one_hot(
        torch.tensor([encode_atom(a, config.atom_encoder) for a in rdmol.GetAtoms()]),
        num_classes=len(config.atom_encoder)
    )

    # BRICS fragmentation
    fragment_mask = get_fragment_mask(rdmol) if config.compute_fragment_mask else None

    # Bonds
    adj = np.ones((rdmol.GetNumAtoms(), rdmol.GetNumAtoms())) * config.bond_encoder['NOBOND']
    for b in rdmol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        adj[i, j] = config.bond_encoder[str(b.GetBondType())]
        adj[j, i] = adj[i, j]  # undirected graph

    # molecular graph is undirected -> don't save redundant information
    bonds = np.stack(np.triu_indices(len(ligand_coord), k=1), axis=0)
    # bonds = np.stack(np.ones_like(adj).nonzero(), axis=0)
    bond_types = adj[bonds[0], bonds[1]].astype('int64')
    bonds = torch.from_numpy(bonds)
    bond_types = F.one_hot(torch.from_numpy(bond_types), num_classes=len(config.bond_encoder))

    ligand = {
        'x': ligand_coord.to(dtype=FLOAT_TYPE),
        'one_hot': ligand_onehot.to(dtype=FLOAT_TYPE),
        'mask': torch.zeros(len(ligand_coord), dtype=INT_TYPE),
        'bonds': bonds.to(INT_TYPE),
        'bond_one_hot': bond_types.to(FLOAT_TYPE),
        'bond_mask': torch.zeros(bonds.size(1), dtype=INT_TYPE),
        'size': torch.tensor([len(ligand_coord)], dtype=INT_TYPE),
        'n_bonds': torch.tensor([len(bond_types)], dtype=INT_TYPE),
    }
    if fragment_mask is not None:
        ligand['fragments'] = fragment_mask.to(INT_TYPE)

    return ligand


def process_raw_molecule_with_empty_pocket(rdmol, config: FeaturizationConfig):
    ligand = prepare_ligand(rdmol, config=config)
    pocket = Residues.empty().to_dict()
    return ligand, pocket


def structure_from_residues(biopython_residue_list):
    builder = StructureBuilder.StructureBuilder()
    builder.init_structure("")
    builder.init_model(0)
    new_struct = builder.get_structure()
    for residue in biopython_residue_list:
        chain = residue.get_parent().get_id()

        # init chain if necessary
        if not new_struct[0].has_id(chain):
            builder.init_chain(chain)

        # add residue
        new_struct[0][chain].add(residue)

    return new_struct


def extract_pocket_residues(biopython_model, rdmol, config: FeaturizationConfig):

    dist_cutoff = config.dist_cutoff

    pocket_residues = []
    rdmol = Chem.RemoveAllHs(rdmol)  # returns a copy, the molecule isn't modified outside the scope of this function
    ligand_coords = torch.from_numpy(rdmol.GetConformer().GetPositions())

    for residue in biopython_model.get_residues():

        # Remove non-standard amino acids and HETATMs
        if not is_aa(residue.get_resname(), standard=True):
            continue

        res_coords = torch.from_numpy(np.array([a.get_coord() for a in residue.get_atoms()]))

        is_interacting = dist_cutoff is None or (((res_coords[:, None, :] - ligand_coords[None, :, :]) ** 2).sum(-1) ** 0.5).min() < dist_cutoff

        if is_interacting:
            pocket_residues.append(residue)

    return pocket_residues


def process_raw_pair(
        biopython_model: Model.Model,
        rdmol: Chem.Mol,
        config: FeaturizationConfig,
        return_pocket_pdb: bool = False
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

    # Process ligand
    ligand = prepare_ligand(rdmol, config=config)

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = extract_pocket_residues(biopython_model, rdmol, config=config)
    pocket, pocket_residues = prepare_pocket(pocket_residues, config=config)

    if return_pocket_pdb:
        pocket['pocket_pdb'] = structure_from_residues(pocket_residues)

    return ligand, pocket


def rdmol_to_smiles(rdmol):
    mol = Chem.Mol(rdmol)
    Chem.RemoveStereochemistry(mol)
    mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


def get_n_nodes(lig_positions, pocket_positions, smooth_sigma=None):
    # Joint distribution of ligand's and pocket's number of nodes
    n_nodes_lig = [len(x) for x in lig_positions]
    n_nodes_pocket = [len(x) for x in pocket_positions]
    return get_joint_size_histogram(n_nodes_lig, n_nodes_pocket, smooth_sigma)


def get_joint_size_histogram(n_nodes_lig, n_nodes_pocket, smooth_sigma=None):
    joint_histogram = np.zeros((np.max(n_nodes_lig) + 1,
                                np.max(n_nodes_pocket) + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(f'Original histogram: {np.count_nonzero(joint_histogram)}/'
          f'{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled')

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram, sigma=smooth_sigma, order=0, mode='constant',
            cval=0.0, truncate=4.0)

        print(f'Smoothed histogram: {np.count_nonzero(filtered_histogram)}/'
              f'{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled')

        joint_histogram = filtered_histogram

    return joint_histogram


def get_type_histogram(one_hot, type_encoder):

    one_hot = np.concatenate(one_hot, axis=0)

    decoder = list(type_encoder.keys())
    counts = {k: 0 for k in type_encoder.keys()}
    for a in [decoder[x] for x in one_hot.argmax(1)]:
        counts[a] += 1

    return counts


def get_residue_with_resi(pdb_chain, resi):
    res = [x for x in pdb_chain.get_residues() if x.id[1] == resi]
    assert len(res) == 1
    return res[0]


def get_pocket_from_ligand(pdb_model, ligand, dist_cutoff=8.0):

    if ligand.endswith(".sdf"):
        # ligand as sdf file
        rdmol = Chem.SDMolSupplier(str(ligand))[0]
        ligand_coords = torch.from_numpy(rdmol.GetConformer().GetPositions()).float()
        resi = None
    else:
        # ligand contained in PDB; given in <chain>:<resi> format
        chain, resi = ligand.split(':')
        ligand = get_residue_with_resi(pdb_model[chain], int(resi))
        ligand_coords = torch.from_numpy(
            np.array([a.get_coord() for a in ligand.get_atoms()]))

    pocket_residues = []
    for residue in pdb_model.get_residues():
        if residue.id[1] == resi:
            continue  # skip ligand itself

        res_coords = torch.from_numpy(
            np.array([a.get_coord() for a in residue.get_atoms()]))
        if is_aa(residue.get_resname(), standard=True) \
                and torch.cdist(res_coords, ligand_coords).min() < dist_cutoff:
            pocket_residues.append(residue)

    return pocket_residues


def encode_residues(biopython_residues, type_encoder, level='atom',
                    remove_H=True):
    assert level in {'atom', 'residue'}

    if level == 'atom':
        entities = [a for res in biopython_residues for a in res.get_atoms()
                    if (a.element != 'H' or not remove_H)]
        types = [a.element.capitalize() for a in entities]
    else:
        entities = [res['CA'] for res in biopython_residues]
        types = [protein_letters_3to1[res.get_resname()] for res in biopython_residues]

    coord = torch.tensor(np.stack([e.get_coord() for e in entities]))
    one_hot = F.one_hot(torch.tensor([type_encoder[t] for t in types]),
                        num_classes=len(type_encoder))

    return coord, one_hot


def edge_mask_by_node_mask(node_mask, edges):
    """Masks edges between masked nodes"""
    return node_mask[edges[0]] * node_mask[edges[1]]
