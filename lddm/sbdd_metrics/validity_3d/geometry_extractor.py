from rdkit.Chem import (Bond, 
                        Mol,
                        Conformer,
                        GetPeriodicTable,
                        Atom)
from rdkit.Chem.rdMolTransforms import (GetBondLength, 
                                        GetAngleDeg, 
                                        GetDihedralDeg)
from rdkit.Chem.rdmolops import FindAllPathsOfLengthN
from rdkit.Chem import GetPeriodicTable


from .pattern import (CentralAtomTuple, 
                      NeighborAtomTuple, 
                      NeighborhoodTuple,
                      BondPattern,
                      AnglePattern,
                      TorsionPattern,
                      NonAromaticRingPattern,
                      RingSizeTuple,
                      sort_neighbor_tuples)

import numpy as np
import re

from typing import List
from rdkit import Chem
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles


class GeometryExtractor():
    
    def __init__(self, authorized_ring_sizes: list[int] = list(range(3, 10))) -> None:
        self.periodic_table = GetPeriodicTable()
        self.authorized_ring_sizes = authorized_ring_sizes
    
    
    def get_vdw_radius(self,
                       symbol: str):
        return self.periodic_table.GetRvdw(symbol)
        
    @staticmethod
    def get_bond_order_float(bond_order: str) -> float:
        if bond_order == '-':
            return 1
        elif bond_order == '=':
            return 2
        elif bond_order == '#':
            return 3
        else:
            return 1.5
        
    # @staticmethod
    # def get_atom_tuple(atom: Atom) -> tuple[str, str, int]:
    #     atom_symbol = atom.GetSymbol()
    #     atom_hybridization = atom.GetHybridization()
    #     atom_charge = atom.GetFormalCharge()
    #     atom_tuple = (atom_symbol, str(atom_hybridization), atom_charge)
    #     return atom_tuple
    
    @staticmethod
    def get_atom_tuple(atom: Atom,
                       ring_sizes: list[int] = None) -> CentralAtomTuple:
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        if ring_sizes is not None and len(ring_sizes) > 0:
            ring_sizes = sorted(ring_sizes, reverse=True)
            ring_size_tuple = RingSizeTuple(tuple(ring_sizes))
            atom_tuple = CentralAtomTuple(atomic_num, 
                                          formal_charge,
                                          ring_size_tuple)
        else:
            atom_tuple = CentralAtomTuple(atomic_num, formal_charge)
        return atom_tuple
    
    @staticmethod
    def get_neighborhood_tuple(atom: Atom,
                                excluded_atoms: list[Atom]
                                ) -> NeighborhoodTuple:
        excluded_atom_idxs = [a.GetIdx() for a in excluded_atoms]
        neighbors = atom.GetNeighbors()
        neighbors = [neighbor 
                    for neighbor in neighbors 
                    if neighbor.GetIdx() not in excluded_atom_idxs]
        # neighbor_symbols = [atom.GetSymbol() for atom in neighbors]
        # neighbor_symbols_1 = np.array(neighbor_symbols_1)
        # order = neighbor_symbols_1.argsort()
        # sorted_neighbors_1 = neighbor_symbols_1[order]
        atom_id = atom.GetIdx()
        mol = atom.GetOwningMol()
        neighbor_tuples = []
        for neighbor in neighbors:
            atomic_num = neighbor.GetAtomicNum()
            neighbor_id = neighbor.GetIdx()
            bond = mol.GetBondBetweenAtoms(atom_id, neighbor_id)
            bond_type = bond.GetBondTypeAsDouble()
            neighbor_tuple = NeighborAtomTuple(atomic_num, bond_type)
            neighbor_tuples.append(neighbor_tuple)
            
        # neighbor_bonds = [mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        #                     for neighbor in neighbors]
        # neighbor_bond_types = [bond.GetBondTypeAsDouble() for bond in neighbor_bonds]
        # neighbor_1_bonds = np.array()
        # sorting_idx = np.lexsort((neighbor_bond_types, neighbor_symbols))
        # # neighbor_tuple = tuple((self.get_atom_tuple(neighbors[i]), 
        # #                         self.get_bond_order_str(neighbor_bond_types[i]))
        # #                        for i in sorting_idx)
        # neighbor_tuple = tuple((neighbor_symbols[i], 
        #                         self.get_bond_order_str(neighbor_bond_types[i]))
        #                        for i in sorting_idx)
        
        if len(neighbor_tuples) == 0:
            neighborhood_tuple = NeighborhoodTuple(())
        else:
            sorted_neighbor_tuples = sort_neighbor_tuples(neighbor_tuples)
            if len(neighbor_tuples) > 1:
                import pdb
                assert sorted_neighbor_tuples[0].atomic_num >= sorted_neighbor_tuples[1].atomic_num, pdb.set_trace()
            neighborhood_tuple = NeighborhoodTuple(tuple(sorted_neighbor_tuples))
        
        return neighborhood_tuple
    
    
    def get_bonds(self,
                  mol: Mol,
                  consider_hydrogens: bool = True):
        bonds = mol.GetBonds()
        if not consider_hydrogens:
            returned_bonds = []
            for bond in bonds:
                atom_symbols = [bond.GetBeginAtom().GetSymbol(),
                                bond.GetEndAtom().GetSymbol()]
                if not 'H' in atom_symbols:
                    returned_bonds.append(bond)
        else:
            returned_bonds = bonds
            
        return returned_bonds
    
    
    def get_bond_length(self,
                        conf: Conformer,
                        bond: Bond):
        return GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    
    def get_bond_pattern(self, 
                        bond: Bond) -> BondPattern:
        bond_type = bond.GetBondTypeAsDouble()
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        ring_sizes = [[] for _ in range(2)]
        atom_ids = [atom.GetIdx() for atom in [begin_atom, end_atom]]
        ring_info = bond.GetOwningMol().GetRingInfo()
        ring_atoms = ring_info.AtomRings()
        for ring in ring_atoms:
            for i, atom_id in enumerate(atom_ids):
                if atom_id in ring:
                    ring_size = len(ring)
                    if ring_size in self.authorized_ring_sizes:
                        ring_sizes[i].append(ring_size)
        
        atom_tuple_1 = self.get_atom_tuple(begin_atom, ring_sizes[0])
        atom_tuple_2 = self.get_atom_tuple(end_atom, ring_sizes[1])
        
        neighborhood_tuple_1 = self.get_neighborhood_tuple(begin_atom, 
                                                            excluded_atoms=[end_atom])
        neighborhood_tuple_2 = self.get_neighborhood_tuple(end_atom,
                                                            excluded_atoms=[begin_atom])
        
        # Comparison order: central atom tuples, neighborhood_tuples
        ascending_atom_id_order = True # Default
        if atom_tuple_1 < atom_tuple_2:
            ascending_atom_id_order = False
        elif atom_tuple_1 == atom_tuple_2:
            if neighborhood_tuple_1 < neighborhood_tuple_2:
                ascending_atom_id_order = False
        
        if ascending_atom_id_order:
            bond_pattern = BondPattern(atom_tuple_1, 
                                       neighborhood_tuple_1, 
                                       bond_type,
                                       atom_tuple_2,
                                       neighborhood_tuple_2)
        else:
            bond_pattern = BondPattern(atom_tuple_2, 
                                       neighborhood_tuple_2, 
                                       bond_type,
                                       atom_tuple_1,
                                       neighborhood_tuple_1)
            
        return bond_pattern
    
    
    def get_angles_atom_ids(self,
                           mol: Mol,
                            consider_hydrogens: bool = True):
        # angles are defined by three consecutive atom in the graph, we take their idx
        paths = FindAllPathsOfLengthN(mol, 3, useBonds=False, useHs=consider_hydrogens)
        
        returned_tuples = [tuple(path) for path in paths]
            
        return returned_tuples
    
    
    def get_angle_pattern(self, 
                         mol: Mol,
                         begin_atom_idx: int, 
                         second_atom_idx: int, 
                         end_atom_idx: int) -> AnglePattern:
        
        begin_atom = mol.GetAtomWithIdx(begin_atom_idx)
        second_atom = mol.GetAtomWithIdx(second_atom_idx)
        end_atom = mol.GetAtomWithIdx(end_atom_idx)
        
        ring_sizes = [[] for _ in range(3)]
        atom_ids = [atom.GetIdx() for atom in [begin_atom, second_atom, end_atom]]
        ring_info = mol.GetRingInfo()
        ring_atoms = ring_info.AtomRings()
        for ring in ring_atoms:
            for i, atom_id in enumerate(atom_ids):
                if atom_id in ring:
                    ring_size = len(ring)
                    ring_sizes[i].append(ring_size)
        
        bond_12 = mol.GetBondBetweenAtoms(begin_atom_idx, second_atom_idx)
        bond_type_12 = bond_12.GetBondTypeAsDouble()
        bond_23 = mol.GetBondBetweenAtoms(second_atom_idx, end_atom_idx)
        bond_type_23 = bond_23.GetBondTypeAsDouble()
        
        atom_tuple_1 = self.get_atom_tuple(begin_atom, ring_sizes[0])
        atom_tuple_2 = self.get_atom_tuple(second_atom, ring_sizes[1])
        atom_tuple_3 = self.get_atom_tuple(end_atom, ring_sizes[2])
        
        neighborhood_tuple_1 = self.get_neighborhood_tuple(begin_atom, 
                                                            excluded_atoms=[end_atom, second_atom])
        neighborhood_tuple_2 = self.get_neighborhood_tuple(second_atom,
                                                            excluded_atoms=[begin_atom, end_atom])
        neighborhood_tuple_3 = self.get_neighborhood_tuple(end_atom,
                                                            excluded_atoms=[begin_atom, second_atom])
        
        # Comparison order: bond_types, central_atom_tuples(1-3), neighborhood_tuples(1-3)
        ascending_atom_id_order = True # Default
        if bond_type_12 < bond_type_23:
            ascending_atom_id_order = False
        elif bond_type_12 == bond_type_23:
            if atom_tuple_1 < atom_tuple_3:
                ascending_atom_id_order = False
            elif atom_tuple_1 == atom_tuple_3:
                if neighborhood_tuple_1 < neighborhood_tuple_3:
                    ascending_atom_id_order = False
                
        if ascending_atom_id_order:
            angle_pattern = AnglePattern(atom_tuple_1,
                                         neighborhood_tuple_1,
                                         bond_type_12,
                                         atom_tuple_2,
                                         neighborhood_tuple_2,
                                         bond_type_23,
                                         atom_tuple_3,
                                         neighborhood_tuple_3)
        else:
            angle_pattern = AnglePattern(atom_tuple_3,
                                         neighborhood_tuple_3,
                                         bond_type_23,
                                         atom_tuple_2,
                                         neighborhood_tuple_2,
                                         bond_type_12,
                                         atom_tuple_1,
                                         neighborhood_tuple_1)
            
        return angle_pattern
    
    
    def get_angle_value(self,
                        conf: Conformer,
                        begin_atom_idx: int,
                        second_atom_idx: int,
                        end_atom_idx: int,
                        ):
        return GetAngleDeg(conf, begin_atom_idx, second_atom_idx, end_atom_idx)
    
    
    def get_torsions_atom_ids(self,
                             mol: Mol,
                             consider_hydrogens: bool = True):
        # dihedrals are defined by four consecutive atom in the graph, we take their idx
        paths = FindAllPathsOfLengthN(mol, 4, useBonds=False, useHs=consider_hydrogens)
        
        returned_tuples = [tuple(path) for path in paths]
        
        return returned_tuples
    
    
    def get_torsion_pattern(self,
                            mol: Mol,
                            begin_atom_idx: int, 
                            second_atom_idx: int, 
                            third_atom_idx: int,
                            end_atom_idx: int) -> TorsionPattern:
        
        begin_atom = mol.GetAtomWithIdx(begin_atom_idx)
        second_atom = mol.GetAtomWithIdx(second_atom_idx)
        third_atom = mol.GetAtomWithIdx(third_atom_idx)
        end_atom = mol.GetAtomWithIdx(end_atom_idx)
        
        ring_sizes = [[] for _ in range(4)]
        atom_ids = [atom.GetIdx() for atom in [begin_atom, second_atom, third_atom, end_atom]]
        ring_info = mol.GetRingInfo()
        ring_atoms = ring_info.AtomRings()
        for ring in ring_atoms:
            for i, atom_id in enumerate(atom_ids):
                if atom_id in ring:
                    ring_size = len(ring)
                    ring_sizes[i].append(ring_size)
        
        bond_12 = mol.GetBondBetweenAtoms(begin_atom_idx, second_atom_idx)
        bond_type_12 = bond_12.GetBondTypeAsDouble()
        bond_23 = mol.GetBondBetweenAtoms(second_atom_idx, third_atom_idx)
        bond_type_23 = bond_23.GetBondTypeAsDouble()
        bond_34 = mol.GetBondBetweenAtoms(third_atom_idx, end_atom_idx)
        bond_type_34 = bond_34.GetBondTypeAsDouble()
        
        atom_tuple_1 = self.get_atom_tuple(begin_atom, ring_sizes[0])
        atom_tuple_2 = self.get_atom_tuple(second_atom, ring_sizes[1])
        atom_tuple_3 = self.get_atom_tuple(third_atom, ring_sizes[2])
        atom_tuple_4 = self.get_atom_tuple(end_atom, ring_sizes[3])
        
        neighborhood_tuple_1 = self.get_neighborhood_tuple(begin_atom, 
                                                            excluded_atoms=[second_atom, third_atom, end_atom])
        neighborhood_tuple_2 = self.get_neighborhood_tuple(second_atom,
                                                            excluded_atoms=[begin_atom, third_atom, end_atom])
        neighborhood_tuple_3 = self.get_neighborhood_tuple(third_atom,
                                                            excluded_atoms=[begin_atom, second_atom, end_atom])
        neighborhood_tuple_4 = self.get_neighborhood_tuple(end_atom,
                                                            excluded_atoms=[begin_atom, second_atom, third_atom])
        
        # Comparison order: central_atom_tuples (2-3), bond_types (12-34), 
        # central_atom_tuples (1-4), neighborhood_tuples (2-3), neighborhood_tuples (1-4)
        ascending_atom_id_order = True # Default
        if atom_tuple_2 < atom_tuple_3:
            ascending_atom_id_order = False
        elif atom_tuple_2 == atom_tuple_3:
            if neighborhood_tuple_2 < neighborhood_tuple_3:
                ascending_atom_id_order = False
            elif neighborhood_tuple_2 == neighborhood_tuple_3:
                if bond_type_12 < bond_type_34:
                    ascending_atom_id_order = False
                elif bond_type_12 == bond_type_34:
                    if atom_tuple_1 < atom_tuple_4:
                        ascending_atom_id_order = False
                    elif atom_tuple_1 == atom_tuple_4:
                        if neighborhood_tuple_1 < neighborhood_tuple_4:
                            ascending_atom_id_order = False
            
        if ascending_atom_id_order:
            torsion_pattern = TorsionPattern(atom_tuple_1,
                                             neighborhood_tuple_1,
                                             bond_type_12,
                                             atom_tuple_2,
                                             neighborhood_tuple_2,
                                             bond_type_23,
                                             atom_tuple_3,
                                             neighborhood_tuple_3,
                                             bond_type_34,
                                             atom_tuple_4,
                                             neighborhood_tuple_4)
        else:
            torsion_pattern = TorsionPattern(atom_tuple_4,
                                             neighborhood_tuple_4,
                                             bond_type_34,
                                             atom_tuple_3,
                                             neighborhood_tuple_3,
                                             bond_type_23,
                                             atom_tuple_2,
                                             neighborhood_tuple_2,
                                             bond_type_12,
                                             atom_tuple_1,
                                             neighborhood_tuple_1)
            
        return torsion_pattern
    
    
    def get_torsion_value(self,
                        conf: Conformer,
                        begin_atom_idx: int,
                        second_atom_idx: int,
                        third_atom_idx: int,
                        end_atom_idx: int,
                        ):
        return GetDihedralDeg(conf, begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx)
        
    def get_planar_rings_atom_ids(self, mol: Mol, authorized_ring_sizes: list[int] = [5, 6]):
        ring_info = mol.GetRingInfo()
        ring_atoms = ring_info.AtomRings()
        returned_rings = []
        for ring in ring_atoms:
            if len(ring) in authorized_ring_sizes:
                planar = True
                for atom_id in ring:
                    atom = mol.GetAtomWithIdx(atom_id)
                    if not atom.GetIsAromatic():
                        planar = False
                        break
                if planar:
                    returned_rings.append(ring)
        
        return returned_rings
    
    @staticmethod
    def get_non_aromatic_ring_pattern(mol: Mol, atoms: List[int]):
        newmol = Chem.Mol(mol)

        # "Safely" remove AROMATIC bond types from the molecule
        Chem.Kekulize(newmol)
        for bond in newmol.GetBonds(): bond.SetIsAromatic(False)
        for atom in newmol.GetAtoms(): atom.SetIsAromatic(False)
        ring_coord = newmol.GetConformer().GetPositions()[list(atoms)]

        # Just smiles of the ring itself
        # generalized_pattern = StandardizeSmiles(Chem.MolFragmentToSmiles(newmol, atoms))

        # More specific pattern: SMILES of the ring + its bonds going outside (with dummy atoms "*"" on the ends)
        bonds = []
        for atom in atoms:
            for bond in newmol.GetAtomWithIdx(atom).GetBonds():
                if bond.GetBeginAtomIdx() not in atoms or bond.GetEndAtomIdx() not in atoms:
                    bonds.append(bond.GetIdx())

        if len(bonds) == 0:
            frags = Chem.GetMolFrags(newmol, asMols=True)
        else:
            frags = Chem.GetMolFrags(Chem.FragmentOnBonds(newmol, bonds), asMols=True)
        
        found = None
        for frag in frags:
            frag_coord = frag.GetConformer().GetPositions()
            distances = np.linalg.norm(ring_coord[:, None, :] - frag_coord[None, :, :], axis=-1).min(axis=1)
            if np.all(distances < 1e-5):
                found = frag
                break

        assert found is not None
        smi = Chem.MolToSmiles(found)
        pattern = StandardizeSmiles(re.sub(r'\[\d+\*\]', '*', smi))

        return NonAromaticRingPattern(pattern)
    
    @staticmethod
    def get_non_aromatic_rings_atom_ids(mol, authorized_ring_sizes=[5, 6, 7]):
        ring_info = mol.GetRingInfo()
        ring_atoms = ring_info.AtomRings()
        returned_rings = []
        for ring in ring_atoms:
            if len(ring) in authorized_ring_sizes:
                planar = True
                for atom_id in ring:
                    atom = mol.GetAtomWithIdx(atom_id)
                    if not atom.GetIsAromatic():
                        planar = False
                        break
                if not planar:
                    returned_rings.append(ring)
        
        return returned_rings
    
    @staticmethod
    def get_max_distance_to_plane(conf, ring_atom_ids):
        # Algorithm extracted from PoseBusters : 
        # https://github.com/maabuu/posebusters/blob/main/posebusters/modules/flatness.py
        coords = conf.GetPositions()[ring_atom_ids,:]
        center = coords.mean(axis=0)
        centred_coords = coords - center
        # singular value decomposition
        _, _, V = np.linalg.svd(centred_coords)
        # last vector in V is normal vector to plane
        n = V[-1]
        # distances to plane are projections onto normal
        distance_to_plane = np.dot(centred_coords, n)
        max_distance = np.max(distance_to_plane)
        return max_distance
