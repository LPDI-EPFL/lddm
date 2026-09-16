import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Tuple, List, Dict, Any

from rdkit import Chem
from rdkit.Chem import Mol, Conformer
from scipy.stats.mstats import gmean
from collections import defaultdict

from .clash_checker import ClashChecker
from .geometry_extractor import GeometryExtractor
from .reference_geometry import ReferenceGeometry

class Validity3D:
    
    def __init__(self,
        reference_geometry: ReferenceGeometry,
        q_value_threshold: float,
        steric_clash_safety_ratio: float,
        maximum_ring_plane_distance: float,
        include_torsions: float,
        consider_hydrogens: float,
        generalize: bool,
    ) -> None:

        self.geometry_extractor = GeometryExtractor()
        self.reference_geometry = reference_geometry
        
        self.q_value_threshold = q_value_threshold
        self.max_ring_plane_distance = maximum_ring_plane_distance
        self.include_torsions = include_torsions
        self.consider_hs = consider_hydrogens
        self.generalize = generalize
        
        self.clash_checker = ClashChecker(safety_ratio=steric_clash_safety_ratio,
                                          consider_hs=self.consider_hs)
        
        self.valid_conf_ids = {}
        self.new_patterns: dict[str, tuple[str, str]] = {}
        self.validities = {}
        self.clashes = {}
        self.n_valid_confs = None
        self.invalidity_df = None
    
    

    def evaluate(self, mol, analyze_torsions=True, analyze_clashes=True):

        valid_bond_conf_ids, bond_validities, new_bond_patterns = self.analyze_bonds(mol)
        valid_conf_ids = set(valid_bond_conf_ids)
        validities = list(bond_validities)
        new_patterns = [('bond', p) for p in new_bond_patterns]
        
        valid_angle_conf_ids, angle_validities, new_angle_patterns = self.analyze_angles(mol)
        valid_conf_ids = valid_conf_ids.intersection(valid_angle_conf_ids)
        validities.extend(angle_validities)
        new_patterns.extend([('angle', p) for p in new_angle_patterns])
        
        if analyze_torsions: # this is slower than bonds and angles
            valid_torsion_conf_ids, torsion_validities, new_torsion_patterns = self.analyze_torsions(mol)
            validities.extend(torsion_validities)
            new_patterns.extend([('torsion', p) for p in new_torsion_patterns])
            if self.include_torsions:
                valid_conf_ids = valid_conf_ids.intersection(valid_torsion_conf_ids)

        valid_nar_conf_ids, nar_validities, new_nar_patterns = self.analyze_non_aromatic_rings(mol)
        valid_conf_ids = valid_conf_ids.intersection(valid_nar_conf_ids)
        validities.extend(nar_validities)
        new_patterns.extend([('non_aromatic_ring', p) for p in new_nar_patterns])
                
        valid_ring_conf_ids, ring_validities = self.analyze_rings(mol)
        valid_conf_ids = valid_conf_ids.intersection(valid_ring_conf_ids)
        validities.extend(ring_validities)
        
        clashes = None
        if analyze_clashes:
            valid_nonbonddistance_conf_ids, clashes = self.analyze_clashes(mol)
            valid_conf_ids = valid_conf_ids.intersection(valid_nonbonddistance_conf_ids)

        return validities, new_patterns, clashes
    
    
    def get_valid_conf_ids_for_mol(self, 
                                mol: Mol,
                                name: str = None,
                                analyze_torsions: bool = True,
                                ) -> List[int]:
        assert name is not None, 'Name must be given'
        assert name not in self.valid_conf_ids, 'Name already tested'
        
        valid_bond_conf_ids, bond_validities, new_bond_patterns = self.analyze_bonds(mol)
        valid_conf_ids = set(valid_bond_conf_ids)
        validities = list(bond_validities)
        new_patterns = [('bond', p) for p in new_bond_patterns]
        
        valid_angle_conf_ids, angle_validities, new_angle_patterns = self.analyze_angles(mol)
        valid_conf_ids = valid_conf_ids.intersection(valid_angle_conf_ids)
        validities.extend(angle_validities)
        new_patterns.extend([('angle', p) for p in new_angle_patterns])
        
        if analyze_torsions: # this is slower than bonds and angles
            valid_torsion_conf_ids, torsion_validities, new_torsion_patterns = self.analyze_torsions(mol)
            validities.extend(torsion_validities)
            new_patterns.extend([('torsion', p) for p in new_torsion_patterns])
            if self.include_torsions:
                valid_conf_ids = valid_conf_ids.intersection(valid_torsion_conf_ids)
                
        valid_ring_conf_ids, ring_validities = self.analyze_rings(mol)
        valid_conf_ids = valid_conf_ids.intersection(valid_ring_conf_ids)
        validities.extend(ring_validities)
        
        # clashes = []
        valid_nonbonddistance_conf_ids, clashes = self.analyze_clashes(mol)
        valid_conf_ids = valid_conf_ids.intersection(valid_nonbonddistance_conf_ids)
        
        # validity = len(valid_conf_ids) / mol.GetNumConformers()
        
        self.valid_conf_ids[name] = valid_conf_ids
        self.validities[name] = validities
        self.new_patterns[name] = new_patterns
        self.clashes[name] = clashes
        
        return valid_conf_ids
    
    
    def analyze_bonds(self, mol: Mol):
        bonds = self.geometry_extractor.get_bonds(mol, consider_hydrogens=self.consider_hs)

        valid_bond_conf_ids = []
        bond_validities = []
        new_bond_patterns = []
        for conf in mol.GetConformers():
            conf_is_valid = True
            conf_id = conf.GetId()
        
            for bond in bonds:
                bond_pattern = self.geometry_extractor.get_bond_pattern(bond)
                bond_value = self.geometry_extractor.get_bond_length(conf, bond)
                
                q_value, new_pattern = self.reference_geometry.geometry_is_valid(
                    geometry_pattern=bond_pattern, value=bond_value, geometry='bond', generalize=self.generalize,
                )
                
                if not new_pattern:
                    bond_is_valid = q_value > self.q_value_threshold
                else: # When pattern is new, we cannot state on validity, default behaviour is to accept
                    assert np.isnan(q_value)
                    bond_is_valid = True
                
                validity_d = {
                    'conf_id': conf_id,
                    'geometry_type': 'bond',
                    'pattern': bond_pattern.to_string(),
                    'value': bond_value,
                    'q-value': q_value,
                    'valid': bond_is_valid,
                    'atoms': [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()],
                }
                
                bond_validities.append(validity_d)
                
                if not bond_is_valid:
                    conf_is_valid = False
                    
                if new_pattern:
                    new_bond_patterns.append(bond_pattern.to_string())
            
            if conf_is_valid:
                valid_bond_conf_ids.append(conf_id)
                
        return valid_bond_conf_ids, bond_validities, new_bond_patterns
    
    
    def analyze_angles(self, mol: Mol):
        
        angles_atom_ids = self.geometry_extractor.get_angles_atom_ids(mol, consider_hydrogens=self.consider_hs)
        
        valid_angle_conf_ids = []
        angle_validities = []
        new_angle_patterns = []
        for conf in mol.GetConformers():
            conf_is_valid = True
            conf_id = conf.GetId()
        
            for begin_atom_idx, second_atom_idx, end_atom_idx in angles_atom_ids:
                angle_pattern = self.geometry_extractor.get_angle_pattern(mol, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        end_atom_idx)
                angle_value = self.geometry_extractor.get_angle_value(conf, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        end_atom_idx)
                
                q_value, new_pattern = self.reference_geometry.geometry_is_valid(
                    geometry_pattern=angle_pattern, value=angle_value, geometry='angle', generalize=self.generalize,
                )
                
                if not new_pattern:
                    angle_is_valid = q_value > self.q_value_threshold
                else: # When pattern is new, we cannot state on validity, default behaviour is to accept
                    assert np.isnan(q_value)
                    angle_is_valid = True
                
                validity_d = {
                    'conf_id': conf_id,
                    'geometry_type': 'angle',
                    'pattern': angle_pattern.to_string(),
                    'value': angle_value,
                    'q-value': q_value,
                    'valid': angle_is_valid,
                    'atoms': [begin_atom_idx, second_atom_idx, end_atom_idx],
                }
                angle_validities.append(validity_d)
                
                if not angle_is_valid:
                    conf_is_valid = False
                    
                if new_pattern:
                    new_angle_patterns.append(angle_pattern.to_string())
            
            if conf_is_valid:
                valid_angle_conf_ids.append(conf_id)
                
        return valid_angle_conf_ids, angle_validities, new_angle_patterns
        
    
    def analyze_torsions(self, mol: Mol) -> Tuple[List[int], List[Dict[str, Any]], List[str]]:
        torsions_atom_ids = self.geometry_extractor.get_torsions_atom_ids(mol, consider_hydrogens=self.consider_hs)
        valid_torsion_conf_ids = []
        torsion_validities = []
        new_torsion_patterns = []
        for conf in mol.GetConformers():
            conf_is_valid = True
            conf_id = conf.GetId()
        
            for begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx in torsions_atom_ids:
                torsion_pattern = self.geometry_extractor.get_torsion_pattern(mol, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        third_atom_idx,
                                                                        end_atom_idx)
                torsion_value = self.geometry_extractor.get_torsion_value(conf, 
                                                                        begin_atom_idx, 
                                                                        second_atom_idx, 
                                                                        third_atom_idx,
                                                                        end_atom_idx)
                
                q_value, new_pattern = self.reference_geometry.geometry_is_valid(
                    geometry_pattern=torsion_pattern, value=torsion_value, geometry='torsion', generalize=self.generalize,
                )
                
                if not new_pattern:
                    torsion_is_valid = q_value > self.q_value_threshold
                else: # When pattern is new, we cannot state on validity, default behaviour is to accept
                    assert np.isnan(q_value)
                    torsion_is_valid = True
                
                validity_d = {
                        'conf_id': conf_id,
                        'geometry_type': 'torsion',
                        'pattern': torsion_pattern.to_string(),
                        'value': torsion_value,
                        'q-value': q_value,
                        'valid': torsion_is_valid,
                        'atoms': [begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx],
                    }
                torsion_validities.append(validity_d)
                
                if not torsion_is_valid:
                    conf_is_valid = False
                    
                if new_pattern:
                    new_torsion_patterns.append(torsion_pattern.to_string())
            
            if conf_is_valid:
                valid_torsion_conf_ids.append(conf_id)
                
        return valid_torsion_conf_ids, torsion_validities, new_torsion_patterns
    

    def analyze_non_aromatic_rings(self, mol: Mol) -> Tuple[List[int], List[Dict[str, Any]], List[str]]:
        ring_atom_ids = self.geometry_extractor.get_non_aromatic_rings_atom_ids(mol)
        valid_ring_conf_ids = []
        ring_validities = []
        new_ring_patterns = []
        for conf in mol.GetConformers():
            conf_is_valid = True
            conf_id = conf.GetId()
            for curr_ring_atom_ids in ring_atom_ids:
                try:
                    ring_pattern = self.geometry_extractor.get_non_aromatic_ring_pattern(mol, curr_ring_atom_ids)
                except Chem.KekulizeException:
                    # Known issue: fails on molecules with aromatic (?) N+ in a non-aromatic ring
                    continue
                except Chem.AtomValenceException:
                    # Not very known issue: apparently caused by O2+ atoms
                    continue

                ring_value = self.geometry_extractor.get_max_distance_to_plane(conf, curr_ring_atom_ids)
                
                q_value, new_pattern = self.reference_geometry.geometry_is_valid(
                    geometry_pattern=ring_pattern, value=ring_value, geometry='non_aromatic_ring', 
                    generalize=self.generalize,
                )

                if not new_pattern:
                    ring_is_valid = q_value > self.q_value_threshold
                else: # When pattern is new, we cannot state on validity, default behaviour is to accept
                    assert np.isnan(q_value)
                    ring_is_valid = True
                
                validity_d = {
                    'conf_id': conf_id,
                    'geometry_type': 'non_aromatic_ring',
                    'pattern': ring_pattern.to_string(),
                    'value': ring_value,
                    'q-value': q_value,
                    'valid': ring_is_valid,
                    'atoms': curr_ring_atom_ids,
                }
                ring_validities.append(validity_d)
                
                if not ring_is_valid:
                    conf_is_valid = False
                    
                if new_pattern:
                    new_ring_patterns.append(ring_pattern.to_string())
            
            if conf_is_valid:
                valid_ring_conf_ids.append(conf_id)
                
        return valid_ring_conf_ids, ring_validities, new_ring_patterns
    
    
    def analyze_rings(self,
                      mol: Mol) -> tuple[list[int], dict[str, Any]]:
        
        rings_atom_ids = self.geometry_extractor.get_planar_rings_atom_ids(mol)
        
        ring_validities = []
        if len(rings_atom_ids) == 0:
            valid_ring_conf_ids = [conf.GetId() for conf in mol.GetConformers()]
            
        else:
            valid_ring_conf_ids = []
            for conf in mol.GetConformers():
                conf_is_valid = True
                conf_id = conf.GetId()
            
                max_distances = []
                ring_is_planars = []
                for atom_ids in rings_atom_ids:
                    max_distance = self.get_max_distance_to_plane(conf, atom_ids)
                    ring_is_planar = max_distance < self.max_ring_plane_distance
                    max_distances.append(max_distance)
                    ring_is_planars.append(ring_is_planar)
                        
                    validity_d = {
                        'conf_id': conf_id,
                        'geometry_type': 'ring',
                        'pattern': len(atom_ids),
                        'value': max_distance,
                        'q-value': float(ring_is_planar),
                        'valid': ring_is_planar,
                        'atoms': list(atom_ids),
                    }
                    ring_validities.append(validity_d)
                
                    if not ring_is_planar:
                        conf_is_valid = False
                
                if conf_is_valid:
                    valid_ring_conf_ids.append(conf_id)
                
        return valid_ring_conf_ids, ring_validities
    
    
    def get_max_distance_to_plane(self,
                                    conf: Conformer,
                                    ring_atom_ids: list[int]):
        # Algorithm extracted from PoseBusters : 
        # https://github.com/maabuu/posebusters/blob/main/posebusters/modules/flatness.py
        coords = conf.GetPositions()[ring_atom_ids,:]
        center = coords.mean(axis=0)
        centred_coords = coords - center
        # singular value decomposition
        # start_time = time.time()
        _, _, V = np.linalg.svd(centred_coords)
        # time_elapsed = time.time() - start_time
        # print(time_elapsed)
        # if time_elapsed > 5:
        # last vector in V is normal vector to plane
        n = V[-1]
        # distances to plane are projections onto normal
        distance_to_plane = np.dot(centred_coords, n)
        max_distance = np.max(distance_to_plane)
        return max_distance
    
    
    def analyze_clashes(self,
                        mol: Mol) -> Tuple[List[int], 
                                            Dict[str, Any]]:
        
        conf_clashes = {}
        
        try:
            
            if not self.consider_hs:
                mol = Chem.RemoveAllHs(mol, sanitize=False) # Saves time by not computing H interatomic distances
                
            clashes = []
            valid_interbonddist_conf_ids = []
            
            bonds = self.geometry_extractor.get_bonds(mol,
                                                      consider_hydrogens=self.consider_hs)
            bond_idxs = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        for bond in bonds]
            bond_idxs = [t 
                        if t[0] < t[1] 
                        else (t[1], t[0])
                        for t in bond_idxs ]
            angle_idxs = self.geometry_extractor.get_angles_atom_ids(mol,
                                                                     consider_hydrogens=self.consider_hs)
            two_hop_idxs = [(t[0], t[2])
                            if t[0] < t[2]
                            else (t[2], t[0])
                            for t in angle_idxs]
            torsion_idxs = self.geometry_extractor.get_torsions_atom_ids(mol,
                                                                         consider_hydrogens=self.consider_hs)
            three_hop_idxs = [(t[0], t[3])
                                if t[0] < t[3]
                                else (t[3], t[0])
                                for t in torsion_idxs]
            # atoms = [atom for atom in mol.GetAtoms()]
            
            excluded_pairs = bond_idxs + two_hop_idxs + three_hop_idxs
            
            for conf in mol.GetConformers():
                conf_id = conf.GetId()
                clashes = self.clash_checker.get_clashes(mol, conf_id, excluded_pairs)
                if len(clashes) > 0:
                    conf_clashes[conf_id] = clashes
                else:
                    valid_interbonddist_conf_ids.append(conf_id)
                    
        except Exception as e:
            print(f'Validity3D: failed to analyze clashes: {e}')
            return [], []
                
        return valid_interbonddist_conf_ids, conf_clashes
            
    
    def get_invalid_geometries(self):
        all_n_invs = defaultdict(list)
        invalidity_dfs = []
        for name, validity_list in self.validities.items():
            # columns: name, conf_id, geometry_type, string, value
            validity_df = pd.DataFrame(validity_list) 
            if len(validity_df) > 0:
                invalidity_df = validity_df[validity_df['q-value'] < self.q_value_threshold].copy()
                invalidity_df['name'] = name
                if invalidity_df.shape[0] > 0:
                    invalid_numbers_df = invalidity_df.pivot_table(values='value', 
                                                                    index=['name', 'conf_id'],
                                                                    columns=['geometry_type'],
                                                                    aggfunc='count',
                                                                    fill_value=0)
                    for geometry in ['bond', 'angle', 'torsion', 'ring']:#
                        if geometry in invalid_numbers_df:
                            if np.isnan(invalid_numbers_df[geometry].values[0]):
                                n_inv = 0
                            else:
                                n_inv = int(invalid_numbers_df[geometry].values[0])
                        else:
                            n_inv = 0
                        all_n_invs[geometry].append(n_inv)
                    
                    invalidity_dfs.append(invalid_numbers_df)
            
        if len(invalidity_dfs) > 0:
            self.invalidity_df = pd.concat(invalidity_dfs)
        
        return all_n_invs
    
    
    def get_invalid_bonds_angles(self):
        all_n_inv_bonds = []
        all_n_inv_angles = []
        invalidity_dfs = []
        for name, validity_list in self.validities.items():
            # columns: name, conf_id, geometry_type, string, value
            validity_df = pd.DataFrame(validity_list) 
            if len(validity_df) > 0:
                invalidity_df = validity_df[validity_df['q-value'] < self.q_value_threshold].copy()
                invalidity_df['name'] = name
                if invalidity_df.shape[0] > 0:
                    invalid_numbers_df = invalidity_df.pivot_table(values='value', 
                                                                    index=['name', 'conf_id'],
                                                                    columns=['geometry_type'],
                                                                    aggfunc='count')
                    if 'bond' in invalid_numbers_df.columns:
                        all_n_inv_bonds.extend(invalid_numbers_df['bond'].values.tolist())
                    if 'angle' in invalid_numbers_df.columns: 
                        all_n_inv_angles.extend(invalid_numbers_df['angle'].values.tolist())
                    
                    invalidity_dfs.append(invalid_numbers_df)
            
        if len(invalidity_dfs) > 0:
            self.invalidity_df = pd.concat(invalidity_dfs)
                    
            # if self.show_plots:
            #     sns.histplot(x=n_invalid_bonds_l)
            #     plt.xlabel('Number of invalid bonds')
            #     plt.show()
            
            #     sns.histplot(x=n_invalid_angles_l)
            #     plt.xlabel('Number of invalid angles')
            #     plt.show()
        
        # self.median_n_inv_bonds = np.nanmedian(all_n_inv_bonds)
        # self.median_n_inv_angles = np.nanmedian(all_n_inv_angles)
        
        return all_n_inv_bonds, all_n_inv_angles
    
    
    def get_min_q_values_for_geometry(self,
                                     geometry_type: str = 'bond'):
        return self.get_agg_q_values_for_geometries(agg_func=np.min,
                                                  geometry_types=[geometry_type])
    
    
    def get_geo_mean_q_values_for_geometry(self,
                                          geometry_type: str = 'bond'):
        return self.get_agg_q_values_for_geometries(agg_func=gmean,
                                                  geometry_types=[geometry_type])
    
    
    def get_min_q_values_for_geometries(self,
                                        geometry_types: list[str] = ['bond', 'angle', 'torsion']):
        return self.get_agg_q_values_for_geometries(agg_func=np.min,
                                                  geometry_types=geometry_types)
    
    
    def get_geo_mean_q_values_for_geometries(self,
                                            geometry_types: list[str] = ['bond', 'angle', 'torsion']):
        return self.get_agg_q_values_for_geometries(agg_func=gmean,
                                                  geometry_types=geometry_types)
    
    
    def get_agg_q_values_for_geometries(self,
                                        agg_func: callable,
                                        geometry_types: list[str] = ['bond', 'angle', 'torsion']):
        
        agg_q_values = []
        for name, validity_list in self.validities.items():
            validity_df = pd.DataFrame(validity_list) 
            if len(validity_df) > 0:
                geometry_validity_df = validity_df[validity_df['geometry_type'].isin(geometry_types)]
                geometry_validity_df = geometry_validity_df.dropna(subset=['q-value'])
                if geometry_validity_df.shape[0] > 1:
                    # try:
                    #     q_value_series = geometry_validity_df.groupby('conf_id')['q-value'].agg(agg_func)
                    # except Exception as e:
                    #     print(e)
                    q_value_series = geometry_validity_df.groupby('conf_id')['q-value'].agg(agg_func)
                    agg_q_values.extend(q_value_series.values.tolist())
                else:
                    agg_q_values.extend([float('nan')] * len(validity_df['conf_id'].unique()))
        
        return agg_q_values
        
    def get_new_patterns(self):
        returned_patterns = []
        for name, new_patterns in self.new_patterns.items():
            patterns = [t[1] for t in new_patterns]
            returned_patterns.extend(patterns)
        return returned_patterns