import os
import numpy as np
import pickle
import logging

from tqdm import tqdm
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Mol
from sklearn.neighbors import KernelDensity
from dataclasses import dataclass
from scipy.optimize import minimize, OptimizeResult
from scipy.special import iv

from .data_source import DataSource
from .geometry_extractor import GeometryExtractor
from .pattern import BondPattern, AnglePattern, TorsionPattern, GeometryPattern, NonAromaticRingPattern
from .von_mises_kde import VonMisesKDE

@dataclass
class GeometryKernelDensity:
    kernel_density: KernelDensity
    max_likelihood: float
    num_values_used_for_estimation: int
    
Values = dict[GeometryPattern, list[float]]
KernelDensities = dict[GeometryPattern, GeometryKernelDensity]

class ReferenceGeometry():
    
    def __init__(self, source: DataSource, root: str, minimum_pattern_values: int) -> None:
            
        self.source = source
        self.root = root
        self.minimum_pattern_values = minimum_pattern_values
        
        self.geometry_extractor = GeometryExtractor()
        
        self.values_filepath = os.path.join(root, f'{source.name}_geometry_values.p')
        self.kernel_densities_filepath = os.path.join(root, f'{source.name}_geometry_kernel_densities.p')
        
        self.kernel_densities = self.read_densities()
    
    
    def read_values(self) -> dict[str, Values]:
        if not os.path.exists(self.values_filepath):
            values = self.compute_values()
        else:
            with open(self.values_filepath, 'rb') as f:
                values = pickle.load(f)
        return values
    
    
    def read_densities(self) -> dict[str, KernelDensities]:
        if not os.path.exists(self.kernel_densities_filepath):
            kernel_densities = self.compute_densities(self.read_values())
        else:
            with open(self.kernel_densities_filepath, 'rb') as f:
                kernel_densities = pickle.load(f)
        return kernel_densities
    
    
    def compute_values(self) -> dict[str, Values]:
        
        logging.info(f'Compiling geometry values for {self.source.name}')
        
        all_bond_values = defaultdict(list)
        all_angle_values = defaultdict(list)
        all_torsion_values = defaultdict(list)
        all_non_aromatic_ring_values = defaultdict(list)
        
        # for mol in tqdm(mol_iterator):
        for mol in tqdm(self.source):
            if mol is not None:
                assert isinstance(mol, Mol)
                
                mol_bond_values = self.get_mol_bond_lengths(mol)
                for bond_pattern, bond_values in mol_bond_values.items():
                    all_bond_values[bond_pattern].extend(bond_values)
                
                mol_angle_values = self.get_mol_angle_values(mol)
                for angle_pattern, angle_values in mol_angle_values.items():
                    all_angle_values[angle_pattern].extend(angle_values)
                    
                mol_torsion_values = self.get_mol_torsion_values(mol)
                for torsion_pattern, torsion_values in mol_torsion_values.items():
                    all_torsion_values[torsion_pattern].extend(torsion_values)

                mol_non_aromatic_ring_values = self.get_mol_non_aromatic_ring_values(mol)
                for ring_pattern, ring_values in mol_non_aromatic_ring_values.items():
                    all_non_aromatic_ring_values[ring_pattern].extend(ring_values)
        
        values = {
            'bond': all_bond_values,
            'angle': all_angle_values,
            'torsion': all_torsion_values,
            'non_aromatic_ring': all_non_aromatic_ring_values,
        }
        
        with open(self.values_filepath, 'wb') as f:
            pickle.dump(values, f)
            
        return values
            
            
    def compute_densities(self, values: dict[str, Values]) -> dict[str, KernelDensities]:
        
        logging.info(f'Computing geometry kernel densities for {self.source.name}')
        
        bond_values: dict[BondPattern, list[float]] = values['bond']
        angle_values: dict[AnglePattern, list[float]] = values['angle']
        torsion_values: dict[TorsionPattern, list[float]] = values['torsion']
        non_aromatic_ring_values: dict[NonAromaticRingPattern, list[float]] = values['non_aromatic_ring']
        
        bond_bandwidth = 0.01
        angle_bandwidth = 1.0
        torsion_bandwidth = 200.0
        non_aromatic_ring_bandwith = 0.01
        
        # Bonds
        logging.info(f'Computing bond kernel densities for {self.source.name}')
        bond_kernel_densities: dict[BondPattern, GeometryKernelDensity] = {}
        for pattern, pattern_values in tqdm(bond_values.items()):
            if len(pattern_values) > self.minimum_pattern_values:
                # bandwidth = self.silverman_scott_bandwidth_estimation(pattern_values)
                # bandwidth = bandwidth * 10
                bandwidth = bond_bandwidth
                kernel_density = KernelDensity(bandwidth=bandwidth)
                kernel_density.fit(np.array(pattern_values).reshape(-1, 1))
                max_likelihood = self.get_max_likelihood(kernel_density, 
                                                         values=pattern_values, 
                                                         geometry='bond')
                geometry_kernel_density = GeometryKernelDensity(kernel_density, 
                                                                max_likelihood,
                                                                len(pattern_values))
                bond_kernel_densities[pattern] = geometry_kernel_density
                
        logging.info(f'Computing kernel densities for generalized bond patterns for {self.source.name}')
        generalized_bond_values = defaultdict(list)
        for pattern, values in bond_values.items():
            generalized_pattern = pattern.generalize()
            generalized_bond_values[generalized_pattern].extend(values)
            
        for pattern, pattern_values in tqdm(generalized_bond_values.items()):
            if len(pattern_values) > self.minimum_pattern_values:
                # bandwidth = self.silverman_scott_bandwidth_estimation(pattern_values)
                # bandwidth = bandwidth * 10
                bandwidth = bond_bandwidth
                kernel_density = KernelDensity(bandwidth=bandwidth)
                kernel_density.fit(np.array(pattern_values).reshape(-1, 1))
                max_likelihood = self.get_max_likelihood(kernel_density, 
                                                         pattern_values, 
                                                         geometry='bond')
                geometry_kernel_density = GeometryKernelDensity(kernel_density, 
                                                                max_likelihood,
                                                                len(pattern_values))
                bond_kernel_densities[pattern] = geometry_kernel_density
                
        # Angles
        logging.info(f'Computing angle kernel densities for {self.source.name}')
        angle_kernel_densities: dict[AnglePattern, GeometryKernelDensity] = {}
        for pattern, pattern_values in tqdm(angle_values.items()):
            if len(pattern_values) > self.minimum_pattern_values:
                # bandwidth = self.silverman_scott_bandwidth_estimation(pattern_values)
                # bandwidth = bandwidth * 5
                bandwidth = angle_bandwidth
                kernel_density = KernelDensity(bandwidth=bandwidth)
                kernel_density.fit(np.array(pattern_values).reshape(-1, 1))
                max_likelihood = self.get_max_likelihood(kernel_density, 
                                                         pattern_values, 
                                                         geometry='angle')
                geometry_kernel_density = GeometryKernelDensity(kernel_density, 
                                                                max_likelihood,
                                                                len(pattern_values))
                angle_kernel_densities[pattern] = geometry_kernel_density
                
        # Generalized angle pattern by removing only outer neighborhoods (default)
        # or both outer and inner neighborhoods
        logging.info(f'Computing generalized angle kernel densities for {self.source.name}')
        inner_generalizations = [False, True]
        for generalize_inner in inner_generalizations:
            generalized_angle_values = defaultdict(list)
            for pattern, pattern_values in angle_values.items():
                generalized_pattern = pattern.generalize(inner_neighbors=generalize_inner)
                generalized_angle_values[generalized_pattern].extend(pattern_values)
                
            for pattern, pattern_values in tqdm(generalized_angle_values.items()):
                if len(pattern_values) > 50:
                    # bandwidth = self.silverman_scott_bandwidth_estimation(pattern_values)
                    # bandwidth = bandwidth * 5
                    bandwidth = angle_bandwidth
                    kernel_density = KernelDensity(bandwidth=bandwidth)
                    kernel_density.fit(np.array(pattern_values).reshape(-1, 1))
                    max_likelihood = self.get_max_likelihood(kernel_density, 
                                                            pattern_values, 
                                                            geometry='angle')
                    geometry_kernel_density = GeometryKernelDensity(kernel_density, 
                                                                    max_likelihood,
                                                                    len(pattern_values))
                    angle_kernel_densities[pattern] = geometry_kernel_density
            
        # Torsions
        logging.info(f'Computing torsion kernel densities for {self.source.name}')
        torsion_kernel_densities = {}
        for pattern, pattern_values in tqdm(torsion_values.items()):
            if len(pattern_values) > 50:
                torsion_rad = np.radians(pattern_values)
                # bandwidth = self.taylor_von_mises_bandwidth_estimation(torsion_rad)
                bandwidth = torsion_bandwidth
                kernel_density = VonMisesKDE(bandwidth=bandwidth)
                kernel_density.fit(np.array(torsion_rad).reshape(-1, 1))
                max_likelihood = self.get_max_likelihood(kernel_density, 
                                                        torsion_rad, 
                                                        geometry='torsion')
                geometry_kernel_density = GeometryKernelDensity(kernel_density, 
                                                                max_likelihood,
                                                                len(pattern_values))
                torsion_kernel_densities[pattern] = geometry_kernel_density
                
        # Generalized torsion pattern by removing only outer neighborhoods (default)
        # or both outer and inner neighborhoods
        logging.info(f'Computing generalized torsion kernel densities for {self.source.name}')
        inner_generalizations = [False, True]
        for generalize_inner in inner_generalizations:
            generalized_torsion_values = defaultdict(list)
            for pattern, pattern_values in torsion_values.items():
                generalized_pattern = pattern.generalize(inner_neighbors=generalize_inner)
                generalized_torsion_values[generalized_pattern].extend(pattern_values)
                
            for pattern, pattern_values in tqdm(generalized_torsion_values.items()):
                if len(pattern_values) > 50:
                    torsion_rad = np.radians(pattern_values)
                    # bandwidth = self.taylor_von_mises_bandwidth_estimation(torsion_rad)
                    bandwidth = torsion_bandwidth
                    kernel_density = VonMisesKDE(bandwidth=bandwidth)
                    kernel_density.fit(np.array(torsion_rad).reshape(-1, 1))
                    max_likelihood = self.get_max_likelihood(kernel_density, 
                                                            torsion_rad, 
                                                            geometry='torsion')
                    geometry_kernel_density = GeometryKernelDensity(kernel_density, 
                                                                    max_likelihood,
                                                                    len(torsion_rad))
                    torsion_kernel_densities[pattern] = geometry_kernel_density


        # Non-aromatic rings
        logging.info(f'Computing non-aromatic ring kernel densities for {self.source.name}')
        ring_kernel_densities: dict[NonAromaticRingPattern, GeometryKernelDensity] = {}
        for pattern, pattern_values in tqdm(non_aromatic_ring_values.items()):
            if len(pattern_values) > self.minimum_pattern_values:
                bandwidth = non_aromatic_ring_bandwith
                kernel_density = KernelDensity(bandwidth=bandwidth)
                kernel_density.fit(np.array(pattern_values).reshape(-1, 1))
                max_likelihood = self.get_max_likelihood(kernel_density, pattern_values, geometry='non_aromatic_ring')
                geometry_kernel_density = GeometryKernelDensity(kernel_density, max_likelihood, len(pattern_values))
                ring_kernel_densities[pattern] = geometry_kernel_density
                
        kernel_densities = {
            'bond': bond_kernel_densities, 
            'angle': angle_kernel_densities, 
            'torsion': torsion_kernel_densities,
            'non_aromatic_ring': ring_kernel_densities,
        }
        
        with open(self.kernel_densities_filepath, 'wb') as f:
            pickle.dump(kernel_densities, f)
            
        return kernel_densities
            
            
    def silverman_scott_bandwidth_estimation(self, 
                                             values: list[float]) -> float:
        n = len(values)
        std = np.std(values)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        bandwidth = 1.06 * np.min([std, iqr / 1.34]) * n ** (-1/5)
        return bandwidth
    
    
    def taylor_von_mises_bandwidth_estimation(self,
                                                values: list[float],
                                                bandwidth_min_value: float = 10.0,
                                                bandwidth_max_value: float = 300.0,
                                                kappa_max_value: float = 200.0) -> float:
        n = len(values)
        C = np.sum(np.cos(values))
        S = np.sum(np.sin(values))
        R_dash = np.sqrt((C ** 2) + (S ** 2)) / n
        if R_dash < 0.53:
            kappa = 2 * R_dash + (R_dash ** 3) + 5 * (R_dash ** 5) / 6
        elif R_dash < 0.85:
            kappa = -0.4 + 1.39 * R_dash + 0.43 / (1 - R_dash)
        elif R_dash < 1.0:
            kappa = 1 / ((R_dash ** 3) - 4 * (R_dash ** 2) + 3 * R_dash)
            if kappa > kappa_max_value:
                return bandwidth_max_value
        else:
            return bandwidth_max_value
        
        # kappa = np.min([kappa, 100]) # high kappa values leads to iv(0, kappa) = np.inf by overflow
        
        num = 3 * n * np.square(kappa) * iv(2, 2 * kappa)
        den = (4 * np.power(np.pi, 1/2) * np.square(iv(0, kappa)))
        bandwidth = np.power(num / den, 2 / 5)
        bandwidth = np.min([bandwidth, bandwidth_max_value])
        bandwidth = np.max([bandwidth, bandwidth_min_value])
        
        return bandwidth
        
        
    def get_max_likelihood(self,
                           estimator: KernelDensity,
                           values: list[float],
                           geometry: str,
                           n_samples: int = 1000,
                           ) -> float:
        
        def get_neg_log_likelihood(value: np.ndarray):
            log_likelihood = estimator.score_samples(value.reshape(-1, 1))
            return -log_likelihood
        
        # values = mixture.means_.reshape(-1)
        if geometry == 'bond':
            samples = np.linspace(0.5, 3.5, n_samples)
        elif geometry == 'angle':
            samples = np.linspace(0, 180, n_samples)
        else:
            samples = np.linspace(-np.pi, np.pi, n_samples)
        log_likelihoods = estimator.score_samples(np.array(samples).reshape(-1, 1))
        likelihood_argmax = np.argmax(log_likelihoods)
        max_likelihood_sample = samples[likelihood_argmax]
        
        if geometry == 'bond':
            bounds = [(np.min(values), np.max(values))]
        elif geometry == 'angle':
            bounds = [(np.min(values), 180)]
        else:
            bounds = [(-np.pi, np.pi)]
        
        result: OptimizeResult = minimize(fun=get_neg_log_likelihood, 
                                            x0=max_likelihood_sample, 
                                            method='Nelder-Mead',
                                            bounds=bounds)
        min_neg_log_likelihood = result.fun
        max_log_likelihood = -min_neg_log_likelihood
        max_likelihood = np.exp(max_log_likelihood)
        
        return max_likelihood
              
              
    def get_mol_bond_lengths(self,
                             mol: Mol) -> Values:
        bond_values = defaultdict(list)
        for bond in mol.GetBonds():
            bond_pattern = self.geometry_extractor.get_bond_pattern(bond)
            for conf in mol.GetConformers():
                bond_length = self.geometry_extractor.get_bond_length(conf, bond)
                bond_values[bond_pattern].append(bond_length)
        return bond_values
    
    
    def get_mol_angle_values(self,
                             mol: Mol) -> Values:
        angle_values = defaultdict(list)
        angles_atom_ids = self.geometry_extractor.get_angles_atom_ids(mol)
        for begin_atom_idx, second_atom_idx, end_atom_idx in angles_atom_ids :
            
            angle_pattern = self.geometry_extractor.get_angle_pattern(mol, 
                                                                    begin_atom_idx, 
                                                                    second_atom_idx, 
                                                                    end_atom_idx)
                
            for conf in mol.GetConformers():
                angle_value = self.geometry_extractor.get_angle_value(conf,
                                                                        begin_atom_idx,
                                                                        second_atom_idx,
                                                                        end_atom_idx)
                angle_values[angle_pattern].append(angle_value)
                
        return angle_values
        
                    
    def get_mol_torsion_values(self,
                               mol: Mol,
                               ) -> Values:
        
        torsion_values = defaultdict(list)
        torsion_atom_ids = self.geometry_extractor.get_torsions_atom_ids(mol)
        for begin_atom_idx, second_atom_idx, third_atom_idx, end_atom_idx in torsion_atom_ids :
            
            torsion_pattern = self.geometry_extractor.get_torsion_pattern(mol,
                                                                        begin_atom_idx,
                                                                        second_atom_idx,
                                                                        third_atom_idx,
                                                                        end_atom_idx)
                
            for conf in mol.GetConformers():
                torsion_value = self.geometry_extractor.get_torsion_value(conf,
                                                                            begin_atom_idx,
                                                                            second_atom_idx,
                                                                            third_atom_idx,
                                                                            end_atom_idx)
                if not np.isnan(torsion_value):
                    torsion_values[torsion_pattern].append(torsion_value)
                        
        return torsion_values
    

    def get_mol_non_aromatic_ring_values(self, mol: Mol) -> Values:
        ring_values = defaultdict(list)
        ring_atom_ids = self.geometry_extractor.get_non_aromatic_rings_atom_ids(mol)

        for curr_ring_atom_ids in ring_atom_ids:
            try:
                ring_pattern = self.geometry_extractor.get_non_aromatic_ring_pattern(mol, curr_ring_atom_ids)
            except Chem.KekulizeException:
                # Known issue: fails on molecules with aromatic (?) N+ in a non-aromatic ring
                continue
            except Chem.AtomValenceException:
                # Not very known issue: apparently caused by O2+ atoms
                continue

            for conf in mol.GetConformers():
                ring_value = self.geometry_extractor.get_max_distance_to_plane(conf, curr_ring_atom_ids)
                ring_values[ring_pattern].append(ring_value)
                
        return ring_values
        
    
    def geometry_is_valid(
        self, 
        geometry_pattern: GeometryPattern, 
        value: float, 
        geometry: str = 'bond',
        new_pattern_is_valid: bool = True,
        generalize: bool = True,
    ) -> tuple[float, bool]:

        assert geometry in ['bond', 'angle', 'torsion', 'non_aromatic_ring']
        
        if geometry == 'bond':
            kds = self.kernel_densities['bond']
        elif geometry == 'angle':
            kds = self.kernel_densities['angle']
        elif geometry == 'torsion':
            kds = self.kernel_densities['torsion']
            value = np.radians(value)
        elif geometry == 'non_aromatic_ring':
            kds = self.kernel_densities['non_aromatic_ring']
        else:
            raise RuntimeError()
        
        q_value = np.nan
        new_pattern = False
        if geometry_pattern in kds:
            kd = kds[geometry_pattern]
            kernel_density = kd.kernel_density
            log_likelihood = kernel_density.score_samples(np.array(value).reshape(-1, 1))
            likelihood = np.exp(log_likelihood)
            q_value = likelihood.item() / kd.max_likelihood
            
        elif generalize:
            if geometry == 'non_aromatic_ring':
                if new_pattern_is_valid:
                    new_pattern = True
                return q_value, new_pattern

            generalized_pattern = geometry_pattern.generalize() 
            logging.debug(f'Trying to generalize pattern (outer) : {geometry_pattern.to_string()} to {generalized_pattern.to_string()}')
            if generalized_pattern in kds:
                kd = kds[generalized_pattern]
                kernel_density = kd.kernel_density
                log_likelihood = kernel_density.score_samples(np.array(value).reshape(-1, 1))
                likelihood = np.exp(log_likelihood)
                q_value = likelihood.item() / kd.max_likelihood
            else:
                if geometry == 'bond':
                    if new_pattern_is_valid:
                        new_pattern = True
                else:
                    generalized_pattern = geometry_pattern.generalize(inner_neighbors=True)
                    logging.debug(f'Trying to generalize pattern (outer + inner) : {geometry_pattern.to_string()} to {generalized_pattern.to_string()}')
                    if generalized_pattern in kds:
                        kd = kds[generalized_pattern]
                        kernel_density = kd.kernel_density
                        log_likelihood = kernel_density.score_samples(np.array(value).reshape(-1, 1))
                        likelihood = np.exp(log_likelihood)
                        q_value = likelihood.item() / kd.max_likelihood
                    else:
                        if new_pattern_is_valid:
                            new_pattern = True
        
        elif new_pattern_is_valid:
            new_pattern = True
                
        return q_value, new_pattern