from rdkit import Chem
from rdkit.Chem import Conformer, Mol
from typing import NamedTuple

from .geometry_extractor import GeometryExtractor
# from genbench3d.data.structure import Pocket

class Clash(NamedTuple):
    i: int
    symbol_i: str
    j: int
    symbol_j: str
    distance: float
    min_distance: float
    

class ClashChecker():
    
    def __init__(self,
                 safety_ratio: float,
                 consider_hs: bool) -> None:
        self.safety_ratio = safety_ratio
        self.consider_hs = consider_hs
        
        self.vdw_min_distances = {}
        self.geometry_extractor = GeometryExtractor()
        
        
    def get_vdw_min_distance(self,
                             symbol1, 
                             symbol2):
        vdw1 = self.geometry_extractor.get_vdw_radius(symbol1)
        vdw2 = self.geometry_extractor.get_vdw_radius(symbol2)
        
        if symbol1 == 'H':
            min_distance = vdw2
        elif symbol2 == 'H':
            min_distance = vdw1
        else:
            # min_distance = vdw1 + vdw2 - self.clash_tolerance
            min_distance = vdw1 + vdw2
            
        min_distance = min_distance * self.safety_ratio
        
        return min_distance
    
    
    def get_clashes(self,
                    mol: Mol,
                    conf_id: int,
                    excluded_pairs: list[tuple[int, int]] = []
                    ) -> list[Clash]:
        
        assert all(i <= j for i, j in excluded_pairs)
        distance_matrix = Chem.Get3DDistanceMatrix(mol=mol, confId=conf_id)
        atoms = [atom for atom in mol.GetAtoms()]
        
        clashes = []
        for i, atom1 in enumerate(atoms):
            for j, atom2 in enumerate(atoms[i+1:]):
                j = i + 1 + j
                if not ((i, j) in excluded_pairs):
                    symbol1 = atom1.GetSymbol()
                    symbol2 = atom2.GetSymbol()
                    
                    if symbol1 > symbol2:
                        symbol1, symbol2 = symbol2, symbol1
                            
                    symbol_tuple = (symbol1, symbol2)
                    
                    evaluate = True
                    if not self.consider_hs and 'H' in symbol_tuple:
                        evaluate = False
                        
                    if evaluate:
                        
                        if symbol_tuple in self.vdw_min_distances:
                            min_distance = self.vdw_min_distances[symbol_tuple]
                        else:
                            min_distance = self.get_vdw_min_distance(symbol1, symbol2)
                            self.vdw_min_distances[symbol_tuple] = min_distance
                            
                        distance = distance_matrix[i, j]
                        if distance < min_distance:
                            clash = Clash(i, symbol1, 
                                        j, symbol2, 
                                        distance, min_distance)
                            clashes.append(clash)
                        
        return clashes
        
        
    def get_pocket_ligand_clashes(self,
                                   pocket, #: Pocket,
                                   ligand: Mol,
                                   conf_id: int):
        
        clashes = []
            
        mol = Mol(ligand, confId=conf_id)
        complx = Chem.CombineMols(pocket.mol, mol)
        atoms = [atom for atom in complx.GetAtoms()]
        pocket_atoms = atoms[:pocket.mol.GetNumAtoms()]
        ligand_atoms = atoms[pocket.mol.GetNumAtoms():]
        distance_matrix = Chem.Get3DDistanceMatrix(mol=complx)
    
        for i, atom1 in enumerate(pocket_atoms):
            for j, atom2 in enumerate(ligand_atoms):
                atom2_idx = atom2.GetIdx()
                symbol1 = atom1.GetSymbol()
                symbol2 = atom2.GetSymbol()
                
                if symbol1 > symbol2:
                    symbol1, symbol2 = symbol2, symbol1
                        
                symbol_tuple = (symbol1, symbol2)
                
                evaluate = True
                if not self.consider_hs and 'H' in symbol_tuple:
                    evaluate = False
                    
                if evaluate:
                    
                    if symbol_tuple in self.vdw_min_distances:
                        min_distance = self.vdw_min_distances[symbol_tuple]
                    else:
                        min_distance = self.get_vdw_min_distance(symbol1, symbol2)
                        self.vdw_min_distances[symbol_tuple] = min_distance
                        
                    distance = distance_matrix[i, atom2_idx]
                    if distance < min_distance:
                        clash = Clash(i, symbol1, 
                                    atom2_idx, symbol2, 
                                    distance, min_distance)
                        clashes.append(clash)
                    
        return clashes
        