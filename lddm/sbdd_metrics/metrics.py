import subprocess
import multiprocessing
import signal
from queue import Empty
import time
import tempfile
import pickle
import json
import yaml
import logging
import contextlib
from contextlib import contextmanager
from abc import abstractmethod
from collections.abc import Collection as abcCollection
from collections import defaultdict, deque, namedtuple
from pathlib import Path
from typing import Union, Dict, Collection, Set, Optional, Callable, Any, List
from unittest.mock import patch
from io import StringIO
from tqdm import tqdm

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, KekulizeException, AtomKekulizeException, AllChem
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem.rdForceFieldHelpers import UFFGetMoleculeForceField, MMFFGetMoleculeForceField, MMFFGetMoleculeProperties
from Bio import PDB
import numpy as np
import pandas as pd
from fcd import get_fcd, canonical_smiles, load_ref_model, get_predictions
from posebusters import PoseBusters
from posebusters.modules.distance_geometry import _get_bond_atom_indices, _get_angle_atom_indices
from scipy.spatial.distance import jensenshannon, cdist
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from useful_rdkit_utils import REOS, RingSystemLookup, get_min_ring_frequency, RingSystemFinder

from .interactions import INTERACTION_LIST, prepare_ligand_plf, read_protein, profile_detailed, filter_profile
from .sascorer import calculateScore
from .posebusters_loc import PoseBustersLocal
from .validity_3d.reference_geometry import ReferenceGeometry
from .validity_3d.sdf_source import SDFSource
from .validity_3d.validity3d import Validity3D
from .hbond_utils import WATER_VDW_RADIUS, HBOND_DISTANCE, CircleIn3D, find_blocked_segment, full_circle_covered, hbond_vectors, get_hbond_donors, get_hbond_acceptors

BOND_SYMBOLS = {
    Chem.rdchem.BondType.SINGLE: '-',
    Chem.rdchem.BondType.DOUBLE: '=',
    Chem.rdchem.BondType.TRIPLE: '#',
    Chem.rdchem.BondType.AROMATIC: ':',
}

ALL_EVALUATORS = [
    'representation',
    'mol_props',
    'posebusters',
    'geometry',
    'energy',
    'validity3d',
    'interactions',
    'gnina',
    'medchem',
    'clashes',
    'ring_count',
    'chembl_ring_systems',
    'reos',
    'strain',
    'ff_relaxation',
    'fingerprint_novelty',
    'uncertainty',
]

LOCAL_EVALUATORS = [
    'posebusters_local',
    'validity3d_local',
    'interactions_local',
    'clashes_local',
    'chirality_local',
    'uncertainty_local',
]

PB_MODES = ["dock", "redock", "mol", "gen", "mol_fast"]

ESOLDescriptor = namedtuple("ESOLDescriptor", "mw logp rotors ap")

def is_nan(value):
    return value is None or pd.isna(value) or np.isnan(value)

@contextmanager
def suppress_logging():
    root_logger = logging.getLogger()
    original_level = root_logger.level
    original_handlers = root_logger.handlers[:]
    try:
        root_logger.handlers = []
        logging.disable(logging.CRITICAL)
        yield
    finally:
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
        logging.disable(logging.NOTSET)

def safe_run(func: Callable[..., Any], args: tuple = (), kwargs: dict = None,
             timeout_duration: float = 30) -> Any:
    """
    Safely run a function with timeout using multiprocessing.
    """
    kwargs = kwargs or {}

    def _run(func, queue, args, kwargs):
        try:
            result = func(*args, **kwargs)
            queue.put(result)
        except Exception as e:
            queue.put(e)

    result = None
    queue = multiprocessing.Queue()
    process = None

    try:
        process = multiprocessing.Process(
            target=_run,
            args=(func, queue, args, kwargs)
        )
        process.start()
        
        # Wait for result with timeout
        start_time = time.monotonic()
        while True:
            if not process.is_alive():
                break
                
            if time.monotonic() - start_time > timeout_duration:
                raise TimeoutError(f"Function {func.__name__} timed out after {timeout_duration} seconds")
                
            try:
                # Check queue with a short timeout to avoid busy waiting
                result = queue.get(timeout=0.1)
                if isinstance(result, Exception):
                    raise result
                break
            except Empty:
                continue
            
    except TimeoutError as e:
        logging.warning(f'Timeout while evaluating {func.__name__}: {str(e)}')

    finally:
        # Safe cleanup with timeout
        if process is not None:
            if process.is_alive():
                process.terminate()
                # Give it a short time to terminate gracefully
                process.join(timeout=0.1)
                if process.is_alive():
                    # Force kill if still alive
                    process.kill()
                    process.join(timeout=0.1)
    return result


class AbstractEvaluator:
    ID = None
    DTYPES = {}

    def __call__(self, molecule: Union[str, Path, Chem.Mol], protein: Union[str, Path] = None, **kwargs):
        """
        Args:
            molecule (Union[str, Path, Chem.Mol]): input molecule
            protein (str): target protein
        
        Returns:
            metrics (dict): dictionary of metrics
        """
        RDLogger.DisableLog('rdApp.*')
        self.check_format(molecule, protein)
        start = time.time()
        results = self.evaluate(molecule, protein, **kwargs)
        end = time.time()
        results['time'] = end - start
        return self.add_id(results)

    @classmethod
    def add_id(cls, results):
        if cls.ID is not None:
            return {f'{cls.ID}.{key}': value for key, value in results.items()}
        else:
            return results
        
    def warn(self, message):
        logging.warning(f"[{type(self).__name__}] {message}")

    @abstractmethod
    def evaluate(self, molecule: Union[str, Path, Chem.Mol], protein: Union[str, Path], **kwargs) -> Dict[str, Union[int, float, str]]:
        raise NotImplementedError
    
    @staticmethod
    def check_format(molecule, protein):
        assert isinstance(molecule, (str, Path, Chem.Mol)), 'Supported molecule types: str, Path, Chem.Mol'
        assert protein is None or isinstance(protein, (str, Path)), 'Supported protein types: str'
        if isinstance(molecule, (str, Path)):
            supp = Chem.SDMolSupplier(str(molecule), sanitize=False)
            assert len(supp) == 1, 'Only one molecule per file is supported'

    def load_molecule(self, molecule):
        if isinstance(molecule, (str, Path)):
            return Chem.SDMolSupplier(str(molecule), sanitize=False)[0]
        try:
            return Chem.RemoveAllHs(molecule, sanitize=False)  # remove Hs and create a copy to avoid overriding properties of the input molecule
        except RuntimeError as e:
            self.warn(f"Couldn't remove hydrogens ({e})")
            return Chem.Mol(molecule)
    
    def save_molecule(self, molecule, sdf_path):
        if isinstance(molecule, (str, Path)):
            return molecule
        
        with Chem.SDWriter(str(sdf_path)) as w:
            try:
                w.write(molecule)
            except (RuntimeError, ValueError) as e:
                if isinstance(e, (KekulizeException, AtomKekulizeException)):
                    w.SetKekulize(False)
                    w.write(molecule)
                    w.SetKekulize(True)
                else:
                    w.write(Chem.Mol())
                    self.warn("Error when saving the molecule")
        
        return sdf_path
    
    @property
    def dtypes(self):
        return self.add_id({
            'time': float,
            **self.DTYPES,
        })


class RepresentationEvaluator(AbstractEvaluator):
    ID = 'representation'
    DTYPES = {'smiles': str}

    def evaluate(self, molecule, protein=None, **kwargs):
        molecule = self.load_molecule(molecule)
        try:
            smiles = Chem.MolToSmiles(molecule)
        except Exception:
            smiles = None

        return {'smiles': smiles}


class MolPropertyEvaluator(AbstractEvaluator):
    ID = 'mol_props'
    DTYPES = {'*': float}

    def evaluate(self, molecule, protein=None, **kwargs):
        molecule = self.load_molecule(molecule)
        return {k: v for k, v in molecule.GetPropsAsDict().items() if isinstance(v, float)}


def _posebusters_config(pb_conf):
    if pb_conf == 'mol_fast':
        config = PoseBusters(config='mol').config
        # Keep the ligand checks, without generating an energy reference ensemble.
        config['modules'] = [m for m in config['modules'] if m['function'] != 'energy_ratio']
        return config
    if isinstance(pb_conf, str) and pb_conf not in PB_MODES:
        pb_conf = Path(pb_conf)
    if isinstance(pb_conf, Path):
        with pb_conf.open() as handle:
            return yaml.safe_load(handle)
    return pb_conf


class PoseBustersEvaluator(AbstractEvaluator):
    ID = 'posebusters'
    DTYPES = {'*': bool}

    def __init__(self, pb_conf: Optional[Union[str, Path]] = 'dock'):
        self.fast_geometry = pb_conf == 'mol_fast'
        self.ligand_only = pb_conf in ('mol', 'mol_fast')
        pb_conf = _posebusters_config(pb_conf)
        self.posebusters = PoseBusters(config=pb_conf)

    @patch('rdkit.RDLogger.EnableLog', lambda x: None)
    @patch('rdkit.RDLogger.DisableLog', lambda x: None)
    def evaluate(self, molecule, protein=None, **kwargs):
        self.posebusters.results.clear()
        with suppress_logging() if logging.root.level >= logging.INFO else contextlib.nullcontext():
            try:
                inputs = {'mol_pred': molecule, 'mol_cond': None if self.ligand_only else protein}
                result = (self.posebusters.bust(**inputs) if self.fast_geometry else
                          safe_run(self.posebusters.bust, kwargs=inputs, timeout_duration=30))
            except (RuntimeError, ValueError) as e:
                self.warn(e)
                result = None

        if result is None:
            return dict()
        
        with pd.option_context("future.no_silent_downcasting", True):
            result = dict(result.fillna(False).iloc[0])
        result['all'] = all([bool(value) if not is_nan(value) else False for value in result.values()])
        return result
    

class GeometryEvaluator(AbstractEvaluator):
    ID = 'geometry'
    DTYPES = {'*': list}

    def evaluate(self, molecule, protein=None, **kwargs):
        mol = self.load_molecule(molecule)
        try:
            _mol = Chem.Mol(mol)
            Chem.SanitizeMol(_mol)
            data = self.get_distances_and_angles(_mol)
        except Exception:
            data = {}
        return data

    @staticmethod
    def angle_repr(mol, triplet):
        i = mol.GetAtomWithIdx(triplet[0]).GetSymbol()
        j = mol.GetAtomWithIdx(triplet[1]).GetSymbol()
        k = mol.GetAtomWithIdx(triplet[2]).GetSymbol()
        ij = BOND_SYMBOLS[mol.GetBondBetweenAtoms(triplet[0], triplet[1]).GetBondType()]
        jk = BOND_SYMBOLS[mol.GetBondBetweenAtoms(triplet[1], triplet[2]).GetBondType()]

        # Unified (sorted) representation
        if i < k:
            return f'{i}{ij}{j}{jk}{k}'
        elif i > j:
            return f'{k}{jk}{j}{ij}{i}'
        elif ij <= jk:
            return f'{i}{ij}{j}{jk}{k}'
        else:
            return f'{k}{jk}{j}{ij}{i}'
    
    @staticmethod
    def bond_repr(mol, pair):
        i = mol.GetAtomWithIdx(pair[0]).GetSymbol()
        j = mol.GetAtomWithIdx(pair[1]).GetSymbol()
        ij = BOND_SYMBOLS[mol.GetBondBetweenAtoms(pair[0], pair[1]).GetBondType()]
        # Unified (sorted) representation
        return f'{i}{ij}{j}' if i <= j else f'{j}{ij}{i}'

    @staticmethod
    def get_bond_distances(mol, bonds):
        i, j = np.array(bonds).T
        x = mol.GetConformer().GetPositions()
        xi = x[i]
        xj = x[j]
        bond_distances = np.linalg.norm(xi - xj, axis=1)
        return bond_distances

    @staticmethod
    def get_angle_values(mol, triplets):
        i, j, k = np.array(triplets).T
        x = mol.GetConformer().GetPositions()
        xi = x[i]
        xj = x[j]
        xk = x[k]
        vji = xi - xj
        vjk = xk - xj
        angles = np.arccos((vji * vjk).sum(axis=1) / (np.linalg.norm(vji, axis=1) * np.linalg.norm(vjk, axis=1)))
        return np.degrees(angles)

    @staticmethod
    def get_distances_and_angles(mol):
        data = defaultdict(list)
        bonds = _get_bond_atom_indices(mol)
        distances = GeometryEvaluator.get_bond_distances(mol, bonds)
        for b, d in zip(bonds, distances):
            data[GeometryEvaluator.bond_repr(mol, b)].append(d)

        triplets = _get_angle_atom_indices(bonds)
        angles = GeometryEvaluator.get_angle_values(mol, triplets)
        for t, a in zip(triplets, angles):
            data[GeometryEvaluator.angle_repr(mol, t)].append(a)

        return data
    

class EnergyEvaluator(AbstractEvaluator):
    ID = 'energy'
    DTYPES = {'energy': float}

    def evaluate(self, molecule, protein=None, **kwargs):
        molecule = self.load_molecule(molecule)
        try:
            energy = self.get_energy(molecule)
        except Exception:
            energy = None
        return {'energy': energy}
    
    @staticmethod
    def get_energy(mol, conf_id=-1):
        mol = Chem.AddHs(mol, addCoords=True)
        uff = UFFGetMoleculeForceField(mol, confId=conf_id)
        e_uff = uff.CalcEnergy()
        return e_uff


class Validity3DEvaluator(AbstractEvaluator):
    MINIMUM_PATTERN_VALUES = 50
    CONFIG = {
        'minimum_pattern_values': 50,
        'tfd_threshold': 0.2,
        'q_value_threshold': 0.001,
        'steric_clash_safety_ratio': 0.75,
        'maximum_ring_plane_distance': 0.1,
        'consider_hydrogens': False,
        'include_torsions_in_validity3D': False,
        'add_minimized_docking_scores': True,
        'overwrite_results': True,
        'generalize': False,
    }

    ID = 'validity3d'
    DTYPES = {
        'all': bool,
        'global_*': bool,
        'local_*': float,
        'count_*': int,
        'atoms_*': list,
    }

    def __init__(self, reference_ligands_path: Path, limit: int = None):
        
        tfd_threshold = self.CONFIG['tfd_threshold']
        q_value_threshold = self.CONFIG['q_value_threshold']
        steric_clash_safety_ratio = self.CONFIG['steric_clash_safety_ratio']
        maximum_ring_plane_distance = self.CONFIG['maximum_ring_plane_distance']
        include_torsions_in_validity3D = self.CONFIG['include_torsions_in_validity3D']
        consider_hydrogens =self.CONFIG['consider_hydrogens']

        name = reference_ligands_path.name.replace('.sdf', '')
        self.reference_geometry = ReferenceGeometry(
            source=SDFSource(ligands_path=str(reference_ligands_path), name=name, limit=limit),
            root=str(reference_ligands_path.parent), 
            minimum_pattern_values=self.MINIMUM_PATTERN_VALUES,
        )

        self.validity3d = Validity3D(
            reference_geometry=self.reference_geometry,
            q_value_threshold=q_value_threshold,
            steric_clash_safety_ratio=steric_clash_safety_ratio,
            maximum_ring_plane_distance=maximum_ring_plane_distance,
            include_torsions=include_torsions_in_validity3D,
            consider_hydrogens=consider_hydrogens,
            generalize=self.CONFIG['generalize'],
        )
        

    @patch('rdkit.RDLogger.EnableLog', lambda x: None)
    @patch('rdkit.RDLogger.DisableLog', lambda x: None)
    def evaluate(self, molecule, protein=None, **kwargs):
        try:
            _molecule = self.load_molecule(molecule)
            Chem.SanitizeMol(_molecule)
            if not _molecule.GetNumConformers() or any(
                not np.isfinite(conf.GetPositions()).all() for conf in _molecule.GetConformers()
            ):
                raise ValueError('Validity3D requires finite conformer coordinates')
        except Exception:
            return {
                'all': False,
                'global_bonds': None,
                'global_angles': None,
                'global_torsions': None,
                'global_rings': None,
                'global_non_aromatic_rings': None,
                'local_bonds': None,
                'local_angles': None,
                'local_torsions': None,
                'local_rings': None,
                'local_non_aromatic_rings': None,
                'count_valid_bonds': None,
                'count_valid_angles': None,
                'count_valid_torsions': None,
                'count_valid_rings': None,
                'count_valid_non_aromatic_rings': None,
                'count_invalid_bonds': None,
                'count_invalid_angles': None,
                'count_invalid_torsions': None,
                'count_invalid_rings': None,
                'count_invalid_non_aromatic_rings': None,
                'atoms_invalid_bonds': [],
                'atoms_invalid_angles': [],
                'atoms_invalid_torsions': [],
                'atoms_invalid_rings': [],
                'atoms_invalid_non_aromatic_rings': [],
            }
        
        validities, new_patterns, clashes = self.validity3d.evaluate(_molecule, analyze_torsions=True, analyze_clashes=False)
        results = {
            'global_bonds': True, 
            'global_angles': True,
            'global_torsions': True,
            'global_rings': True,
            'global_non_aromatic_rings': True,
            'local_bonds': None,
            'local_angles': None,
            'local_torsions': None,
            'local_rings': None,
            'local_non_aromatic_rings': None,
            'atoms_invalid_bonds': [],
            'atoms_invalid_angles': [],
            'atoms_invalid_torsions': [],
            'atoms_invalid_rings': [],
            'atoms_invalid_non_aromatic_rings': [],
        }
        counts = {
            'count_valid_bonds': 0,
            'count_valid_angles': 0,
            'count_valid_torsions': 0,
            'count_valid_rings': 0,
            'count_valid_non_aromatic_rings': 0,
            'count_invalid_bonds': 0,
            'count_invalid_angles': 0,
            'count_invalid_torsions': 0,
            'count_invalid_rings': 0,
            'count_invalid_non_aromatic_rings': 0,
        }
        for geometry in ('bond', 'angle', 'torsion', 'non_aromatic_ring'):
            counts[f'count_unknown_{geometry}s'] = sum(kind == geometry for kind, _ in new_patterns)
        for x in validities:
            geometry = x['geometry_type']
            if x['valid']:
                counts[f'count_valid_{geometry}s'] += 1
            else:
                counts[f'count_invalid_{geometry}s'] += 1
                results[f'global_{geometry}s'] = False
                results[f'atoms_invalid_{geometry}s'].append(x['atoms'])

        for geometry in ['bonds', 'angles', 'torsions', 'non_aromatic_rings']:
            total = counts[f'count_valid_{geometry}'] + counts[f'count_invalid_{geometry}']
            if total > 0:
                results[f'local_{geometry}'] = counts[f'count_valid_{geometry}'] / total

        global_all = (
            results['global_bonds'] & 
            results['global_angles'] & 
            results['global_torsions'] & 
            results['global_non_aromatic_rings'] 
        )
        return {
            'all': global_all,
            **results,
            **counts,
        }
    
class InteractionsEvaluator(AbstractEvaluator):
    ID = 'interactions'
    DTYPES = {
        'NonHydrophobic': int,
        '*B*': int,
        'unsatisfied_*': int,
        'full_table': pd.DataFrame
    }
    DTYPES.update({f'{key}': int for key in INTERACTION_LIST})

    def __init__(self, reduce='reduce', guess_waters=True, keep_explicit_waters=False, protein=None, **kwargs):
        self.reduce = reduce
        self.guess_waters = guess_waters
        self.protein_plf = read_protein(str(protein), reduce_exec=str(self.reduce)) if protein is not None else None
        self.keep_explicit_waters = keep_explicit_waters
        if self.keep_explicit_waters and self.guess_waters:
            self.warn("Options guess_waters and keep_explicit_waters are mutually exclusive. Setting guess_waters=False.")
            self.guess_waters = False

    def update_protein(self, protein: Union[Path, str]):
        if protein is None:
            self.protein_plf = None
            return
        if isinstance(protein, (str, Path)):
            protein = Path(protein)
        self.protein_plf = read_protein(str(protein), reduce_exec=str(self.reduce))
    
    @property
    def default_profile(self):
        default_profile = {i: 0 for i in INTERACTION_LIST}
        return default_profile

    @staticmethod
    def _get_coords_and_radii(protein, ligand, cutoff=None):

        protein_no_h = Chem.RemoveAllHs(protein, sanitize=False)
        ligand_no_h = Chem.RemoveAllHs(ligand, sanitize=False)

        protein_coords = protein_no_h.GetConformer().GetPositions()
        ligand_coords = ligand_no_h.GetConformer().GetPositions()

        protein_radii = np.array([Chem.GetPeriodicTable().GetRvdw(a.GetSymbol()) for a in protein_no_h.GetAtoms()])
        ligand_radii = np.array([Chem.GetPeriodicTable().GetRvdw(a.GetSymbol()) for a in ligand_no_h.GetAtoms()])

        # only keep potentially relevant (nearby) atoms
        if cutoff is not None:
            dists = np.linalg.norm(protein_coords.reshape(-1, 1, 3) - ligand_coords.reshape(1, -1, 3), axis=-1)
            mask = dists.min(axis=1) <= cutoff
            protein_coords = protein_coords[mask]
            protein_radii = protein_radii[mask]

        atom_coords = np.concatenate([protein_coords, ligand_coords], axis=0)
        atom_radii = np.concatenate([protein_radii, ligand_radii], axis=0)
        return {"coords": atom_coords, "radii": atom_radii}
    
    @staticmethod
    def enough_space_for_water(ligand, atom_idx, context, acc_don, water_radius=WATER_VDW_RADIUS, hbond_distance=HBOND_DISTANCE):
        assert acc_don in {"acceptor", "donor"}
        vects, vec_type = hbond_vectors(ligand, atom_idx, acc_don, length=hbond_distance)
        if vects is None:
            # Unsupported donor/acceptor geometry cannot justify water satisfaction.
            return False

        atom_coords = context["coords"]
        atom_radii = context["radii"]

        if vec_type == 'linear':
            # check if molecule with radius water_radius located (hbond_distance + water_radius) away along the vector clashes with any atom
            def is_clashing(vec):
                # predicted_direction = vec[1] - vec[0]
                # predicted_direction = np.array([predicted_direction.x, predicted_direction.y, predicted_direction.z])
                predicted_location = np.array([vec[1].x, vec[1].y, vec[1].z])
                dists = np.linalg.norm(predicted_location[None, :] - atom_coords, axis=-1)
                return np.any(dists <= atom_radii + water_radius)

            return not np.all([is_clashing(v) for v in vects])

        elif vec_type == 'cone':
            # check if water could be placed at at least one location on the cone without clashing with any atoms
            assert len(vects) == 1
            predicted_direction = vects[0][1] - vects[0][0]
            predicted_direction = np.array([predicted_direction.x, predicted_direction.y, predicted_direction.z])
            circle = CircleIn3D(
                center=np.array([vects[0][1].x, vects[0][1].y, vects[0][1].z]), 
                radius=hbond_distance * np.sin(np.pi / 3),  # theoretical value at a 60 degree angle
                normal=predicted_direction,
            )
            left, right = find_blocked_segment(circle, donut_radius=water_radius, sphere_centers=atom_coords, sphere_radii=atom_radii)
            segments = [(l, r) for l, r in zip(left, right)]
            return not full_circle_covered(segments)

        else:
            raise NotImplementedError(f"vec_type={vec_type}, vects={vects}")
    
    def get_unsatisfied_hbonds(self, molecule, interactions, protein=None):

        potential_donors = set(get_hbond_donors(molecule))
        potential_acceptors = set(get_hbond_acceptors(molecule))

        found_donors = set([int(idx) for row in interactions[interactions.interaction == "HBDonor"].ligand_atoms for idx in row.split(",")])
        found_acceptors = set([int(idx) for row in interactions[interactions.interaction == "HBAcceptor"].ligand_atoms for idx in row.split(",")])

        unsatisfied_donors = potential_donors - found_donors
        unsatisfied_acceptors = potential_acceptors - found_acceptors

        # keep original unsatisfied H-bond statistics for now so that we can compare the results
        # (can be removed in the future)
        res = {
            "unsatisfied_hbond_donors": len(unsatisfied_donors), 
            "unsatisfied_hbond_acceptors": len(unsatisfied_acceptors),
        }

        if self.guess_waters:
            assert protein is not None
            mol_no_h = Chem.RemoveAllHs(molecule, sanitize=True)
            context = self._get_coords_and_radii(protein, mol_no_h, cutoff=10.0)
            water_radius = 0.5  # literature value seems to be too strict
            unsatisfied_donors = [idx for idx in unsatisfied_donors if not self.enough_space_for_water(mol_no_h, idx, context, 'donor', water_radius)]
            unsatisfied_acceptors = [idx for idx in unsatisfied_acceptors if not self.enough_space_for_water(mol_no_h, idx, context, 'acceptor', water_radius)]
            res.update({
                "UHBDonor": len(unsatisfied_donors), 
                "UHBAcceptor": len(unsatisfied_acceptors),
                "UHB": len(unsatisfied_donors) + len(unsatisfied_acceptors),
            })

        return res

    @staticmethod
    def clean_up_interactions(interactions, ligand_plf, protein_plf):

        def not_bb_nitrogen_is_HBA(row):
            if row.interaction == 'HBDonor':  # this means the protein residue is an H-bond acceptor
                assert not ',' in row.protein_orig_atoms, "only one atom should be involved in the acceptor role"
                atom_idx = int(row.protein_orig_atoms)
                atom_name = protein_plf.GetAtomWithIdx(atom_idx).GetPDBResidueInfo().GetName().strip()
                return not (atom_name == 'N')
            return True

        mask = interactions.apply(not_bb_nitrogen_is_HBA, axis=1)
        return interactions[mask]

    def evaluate(self, molecule, protein=None, timeout=60, return_table=False, interaction_list=None,**kwargs):
        if interaction_list is None:
            interaction_list = INTERACTION_LIST
        molecule = self.load_molecule(molecule)
        profile = self.default_profile
        assert self.protein_plf is not None or protein is not None
        try:
            ligand_plf = prepare_ligand_plf(molecule)
            protein_plf = read_protein(str(protein), keep_water=self.keep_explicit_waters, reduce_exec=str(self.reduce)) if self.protein_plf is None else self.protein_plf
            if not (ligand_plf is None or protein_plf is None):
                interactions = safe_run(
                    profile_detailed, 
                    kwargs={'ligand_plf': ligand_plf, 'protein_plf': protein_plf, 'interaction_list': interaction_list},
                    timeout_duration=timeout
                )
                if interactions is None:
                    profile = {i: None for i in INTERACTION_LIST}
                if not interactions.empty:
                    interactions = self.clean_up_interactions(interactions, ligand_plf, protein_plf)
                    profile.update(dict(interactions.interaction.value_counts()))
                    profile.update(self.get_unsatisfied_hbonds(ligand_plf, interactions, protein=protein_plf))
                    profile['HB'] = profile['HBDonor'] + profile['HBAcceptor']
                    profile['XB'] = profile['XBDonor'] + profile['XBAcceptor']
                    profile['SB'] = profile['Anionic'] + profile['Cationic']
                    profile['NonHydrophobic'] = (
                        profile['HB'] + profile['XB'] + profile['SB'] + 
                        profile['PiStacking'] + profile['CationPi'] + profile['PiCation']
                    )
                    if return_table:
                        profile['full_table'] = interactions
        except Exception as e:
            self.warn(f"Error while evaluating interactions: {str(e)}")
            # import traceback
            # traceback.print_exc()
            pass
        return profile


class GninaEvaluator(AbstractEvaluator):
    ID = 'gnina'
    DTYPES = {'*': float}

    def __init__(self, gnina):
        self.gnina = gnina

    @staticmethod
    def randomize_pose(rdmol, random_seed):
        # create copy
        rdmol = Chem.Mol(rdmol)

        # relax structure
        rdmol = Chem.AddHs(rdmol)
        AllChem.EmbedMolecule(rdmol, randomSeed=random_seed)
        AllChem.MMFFOptimizeMolecule(rdmol, maxIters=1000)
        rdmol = Chem.RemoveHs(rdmol)

        # subtract center of mass
        conf = rdmol.GetConformer()
        center_of_mass = AllChem.ComputeCentroid(conf)
        centered_positions = conf.GetPositions() - center_of_mass
        
        # apply random rotation
        rand_rot = Rotation.random(random_state=random_seed).as_matrix()
        rotated_positions = np.dot(centered_positions, rand_rot)

        for atom_idx, pos in enumerate(rotated_positions):
            conf.SetAtomPosition(atom_idx, pos)

        return rdmol

    def evaluate(self, molecule, protein=None, timeout=None, mode='minimize', **kwargs):
        if timeout is None:
            timeout = 30 if mode == 'minimize' else 400
        assert protein is not None, "Protein structure must be provided for GNINA evaluation"
        protein = Path(protein)
        n_atoms = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            molecule = self.load_molecule(molecule)
            n_atoms = molecule.GetNumAtoms()
            tmp_molecule_path = self.save_molecule(molecule, sdf_path=Path(tmpdir, 'molecule.sdf'))
            out_mol = Path(tmpdir, 'gnina_scored.sdf')

            if mode == 'minimize':
                gnina_cmd = [
                    self.gnina,
                    '-r', str(protein),
                    '-l', str(tmp_molecule_path),
                    '-o', str(out_mol),
                    '--minimize',
                    '--seed', str(42),
                    '--no_gpu'
                ]
            elif mode == 'dock':
                # randomizing pose before docking to avoid bias
                molecule_rand = self.randomize_pose(molecule, random_seed=42)
                tmp_molecule_path_rand = self.save_molecule(molecule_rand, sdf_path=Path(tmpdir, 'molecule_rand.sdf'))

                gnina_cmd = [
                    self.gnina,
                    '-r', str(protein),
                    '-l', str(tmp_molecule_path_rand),
                    '-o', str(out_mol),
                    '--autobox_ligand', str(tmp_molecule_path),
                    '--seed', str(42),
                    '--no_gpu'
                ]
            else:
                raise NotImplementedError(f"Mode {mode} is not supported")
            gnina_mol = None
            try:
                subprocess.run(
                    gnina_cmd,
                    timeout=timeout,
                    capture_output=True,
                    check=True  # Raises CalledProcessError on non-zero exit
                )
            except subprocess.TimeoutExpired as e:
                self.warn(f"GNINA docking timed out after {timeout} seconds")
            except subprocess.CalledProcessError as e:
                self.warn(f"GNINA docking failed with exit code {e.returncode}")

            # read out mol
            if out_mol.exists():
                gnina_mol = Chem.SDMolSupplier(str(out_mol), sanitize=False)
                gnina_mol = gnina_mol[0] if len(gnina_mol) > 0 else None

        if mode == 'dock':
            # RMSD has to be computed w.r.t. the original molecule
            gnina_scores = self.read_gnina_results(gnina_mol, in_mol=molecule)
        else:
            gnina_scores = self.read_gnina_results(gnina_mol)

        # Additionally computing ligand efficiency
        gnina_scores['vina_efficiency'] = gnina_scores['vina_score'] / n_atoms \
            if gnina_scores['vina_score'] is not None and n_atoms > 0 else None
        gnina_scores['gnina_efficiency'] = gnina_scores['gnina_score'] / n_atoms \
            if gnina_scores['gnina_score'] is not None and n_atoms > 0 else None
        return gnina_scores

    def read_gnina_results(self, gnina_mol, in_mol=None):
        res = {
            'vina_score': None,
            'gnina_score': None,
            'minimisation_rmsd': None,
            'docking_rmsd': None,
            'cnn_score': None,
        }
        if gnina_mol is not None:
            if gnina_mol.HasProp('minimizedAffinity'):
                res['vina_score'] = gnina_mol.GetDoubleProp('minimizedAffinity')
            if gnina_mol.HasProp('CNNaffinity'):
                res['gnina_score'] = gnina_mol.GetDoubleProp('CNNaffinity')
            if gnina_mol.HasProp('CNNscore'):
                res['cnn_score'] = gnina_mol.GetDoubleProp('CNNscore')
            if gnina_mol.HasProp('minimizedRMSD'):
                res['minimisation_rmsd'] = gnina_mol.GetDoubleProp('minimizedRMSD')
            if in_mol is not None:
                params = Chem.AdjustQueryParameters.NoAdjustments()
                params.makeBondsGeneric = True
                query_generic_bonds = Chem.AdjustQueryProperties(in_mol, params)
        
                rmsd = Chem.rdMolAlign.CalcRMS(query_generic_bonds, gnina_mol)
                res['docking_rmsd'] = rmsd
        return res


class MedChemEvaluator(AbstractEvaluator):
    ID = 'medchem'
    DTYPES = {
        'valid': bool,
        'connected': bool,
        'qed': float,
        'sa': float,
        'logp': float,
        'log_p_normalized': float,
        'lipinski': int,
        'size': int,
        'mw': float,
        'n_rotatable_bonds': int,
        'n_chiral_centers': int,
        'fsp3': float,
        'esol': float,
        'tpsa': float,
    }

    def __init__(self, connectivity_threshold=1.0):
        self.connectivity_threshold = connectivity_threshold

    def evaluate(self, molecule, protein=None, **kwargs):
        molecule = self.load_molecule(molecule)
        valid = self.is_valid(molecule)

        if valid:
            Chem.SanitizeMol(molecule)

        connected = None if not valid else self.is_connected(molecule)
        qed = None if not valid else self.calculate_qed(molecule)
        sa = None if not valid else self.calculate_sa(molecule)
        logp = None if not valid else self.calculate_logp(molecule)
        lipinski = None if not valid else self.calculate_lipinski(molecule)
        n_rotatable_bonds = None if not valid else self.calculate_rotatable_bonds(molecule)
        size, mw = self.calculate_molecule_size(molecule)
        n_chiral_centers = self.calculate_chiral_centers(molecule)
        logp_normalized = (logp / size) if (size > 0 and logp is not None) else None
        fsp3 = self.calculate_fsp3(molecule)
        esol = self.calculate_esol(molecule)
        tpsa = self.calculate_tpsa(molecule)

        return {
            'valid': valid,
            'connected': connected,
            'qed': qed,
            'sa': sa,
            'logp': logp,
            'log_p_normalized': logp_normalized,
            'lipinski': lipinski,
            'size': size,
            'mw': mw,
            'n_rotatable_bonds': n_rotatable_bonds,
            'n_chiral_centers': n_chiral_centers,
            'fsp3': fsp3,
            'esol': esol,
            'tpsa': tpsa,
        }

    @staticmethod
    def is_valid(rdmol):
        if rdmol.GetNumAtoms() < 1:
            return False

        _mol = Chem.Mol(rdmol)
        try:
            Chem.SanitizeMol(_mol)
        except ValueError:
            return False

        return True

    def is_connected(self, rdmol):
        if rdmol.GetNumAtoms() < 1:
            return False

        try:
            mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True)
            largest_frag = max(mol_frags, default=rdmol, key=lambda m: m.GetNumAtoms())
            return largest_frag.GetNumAtoms() / rdmol.GetNumAtoms() >= self.connectivity_threshold
        except Exception:
            return False
    
    @staticmethod
    def calculate_qed(rdmol):
        try:
            return QED.qed(rdmol)
        except Exception:
            return None

    @staticmethod
    def calculate_sa(rdmol):
        try:
            sa = calculateScore(rdmol)
            return sa
        except Exception as e:
            return None

    @staticmethod
    def calculate_logp(rdmol):
        try:
            return Crippen.MolLogP(rdmol)
        except Exception:
            return None

    @staticmethod
    def calculate_lipinski(rdmol):
        try:
            rule_1 = Descriptors.ExactMolWt(rdmol) < 500
            rule_2 = Lipinski.NumHDonors(rdmol) <= 5
            rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
            rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
            rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
            return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
        except Exception:
            return None
        
    @staticmethod
    def calculate_molecule_size(rdmol):
        try:
            return rdmol.GetNumAtoms(), Descriptors.ExactMolWt(rdmol)
        except Exception:
            return None
        
    @staticmethod
    def calculate_rotatable_bonds(rdmol):
        try:
            return Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol)
        except Exception:
            return None

    @staticmethod 
    def calculate_chiral_centers(rdmol):
        try:
            chiral_centers = Chem.FindMolChiralCenters(
                rdmol, 
                includeUnassigned=True, 
                includeCIP=False, 
                useLegacyImplementation=False
            )
            return len(chiral_centers)
        except Exception:
            return None
        
    @staticmethod
    def calculate_fsp3(rdmol):
        try:
            return Lipinski.FractionCSP3(rdmol)
        except Exception:
            return None

    @staticmethod  
    def _calc_esol_descriptors(mol):
        """
        Calcuate mw,logp,rotors and aromatic proportion (ap)
        :param mol: input molecule
        :return: named tuple with descriptor values
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        aromatic_query = Chem.MolFromSmarts("a")
        matches = mol.GetSubstructMatches(aromatic_query)
        ap = len(matches) / mol.GetNumAtoms()
        return ESOLDescriptor(mw=mw, logp=logp, rotors=rotors, ap=ap)
    
    @staticmethod
    def calculate_esol(rdmol):
        """
        Compute predicted solubility.
        Adapted from: https://github.com/PatWalters/solubility/blob/master/esol.py
        which implements John S. Delaney, J. Chem. Inf. Comput. Sci., 2004, 44, 1000 - 1005
        """
        intercept = 0.16
        coef = {"logp": -0.63, "mw": -0.0062, "rotors": 0.066, "ap": -0.74}
        try:
            desc = MedChemEvaluator._calc_esol_descriptors(rdmol)
            esol = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw \
                + coef["rotors"] * desc.rotors + coef["ap"] * desc.ap
        except Exception:
            return None
        return esol

    @staticmethod
    def calculate_tpsa(rdmol):
        try:
            return AllChem.CalcTPSA(rdmol)
        except Exception:
            return None
    

class ClashEvaluator(AbstractEvaluator):
    ID = 'clashes'
    DTYPES = {
        'clash_score_ligands': float,
        'clash_score_pockets': float,
        'clash_score_between': float,
        'passed_clash_score_ligands': bool,
        'passed_clash_score_pockets': bool,
        'passed_clash_score_between': bool,
    }

    def __init__(self, margin=0.75, ignore={'H'}, truncation_cutoff=8.0):
        self.margin = margin
        self.ignore = ignore
        self.truncation_cutoff = truncation_cutoff
    
    @staticmethod
    def load_pocket(
        pdb_file: str,
        ligand: Chem.Mol,
        cutoff: float = 8.0,
    ) -> Chem.Mol:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_file)
        ligand_positions = ligand.GetConformer().GetPositions()

        nearby_residues = []
        for residue in structure.get_residues():
            res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
            is_interacting = cutoff is None or (((res_coords[:, None, :] - ligand_positions[None, :, :]) ** 2).sum(-1) ** 0.5).min() < cutoff
            if is_interacting:
                nearby_residues.append(residue)

        # Build new structure
        new_structure = PDB.Structure.Structure("binding_site")
        new_model = PDB.Model.Model(0)
        new_structure.add(new_model)

        # Group residues by chain to reconstruct chain objects
        chain_residues = {}
        for residue in nearby_residues:
            chain_id = residue.get_parent().id
            chain_residues.setdefault(chain_id, []).append(residue)

        for chain_id, residues in chain_residues.items():
            new_chain = PDB.Chain.Chain(chain_id)
            new_model.add(new_chain)
            for residue in residues:
                new_chain.add(residue.copy())

        io = PDB.PDBIO()
        buffer = StringIO()
        io.set_structure(new_structure)
        io.save(buffer)

        return Chem.MolFromPDBBlock(buffer.getvalue(), removeHs=False)

    def evaluate(self, molecule=None, protein=None, **kwargs):
        result = {
            'passed_clash_score_ligands': None,
            'passed_clash_score_pockets': None,
            'passed_clash_score_between': None,
        }
        if molecule is not None:
            try:
                molecule = self.load_molecule(molecule)
                clash_score = self.clash_score(molecule)
                result['clash_score_ligands'] = clash_score
                result['passed_clash_score_ligands'] = (clash_score == 0)
            except Exception:
                pass

        if protein is not None:
            try:
                if self.truncation_cutoff and molecule is not None:
                    protein = self.load_pocket(protein, molecule, self.truncation_cutoff)
                else:
                    protein = Chem.MolFromPDBFile(str(protein), sanitize=False)                    
                clash_score = self.clash_score(protein)
                result['clash_score_pockets'] = clash_score
                result['passed_clash_score_pockets'] = (clash_score == 0)
            except Exception:
                pass
        
        if molecule is not None and protein is not None:
            try:
                clash_score = self.clash_score(molecule, protein)
                result['clash_score_between'] = clash_score
                result['passed_clash_score_between'] = (clash_score == 0)
            except Exception:
                pass
        
        return result
    
    def clash_score(self, rdmol1, rdmol2=None):
        """
        Computes a clash score as the number of atoms that have at least one
        clash divided by the number of atoms in the molecule.

        INTERMOLECULAR CLASH SCORE
        If rdmol2 is provided, the score is the percentage of atoms in rdmol1
        that have at least one clash with rdmol2.
        We define a clash if two atoms are closer than "margin times the sum of
        their van der Waals radii".

        INTRAMOLECULAR CLASH SCORE
        If rdmol2 is not provided, the score is the percentage of atoms in rdmol1
        that have at least one clash with other atoms in rdmol1.
        In this case, a clash is defined by margin times the atoms' smallest
        covalent radii (among single, double and triple bond radii). This is done
        so that this function is applicable even if no connectivity information is
        available.
        """

        intramolecular = rdmol2 is None
        if intramolecular:
            rdmol2 = rdmol1
        try:
            coord1, radii1 = self.coord_and_radii(rdmol1, intramolecular=intramolecular)
            coord2, radii2 = self.coord_and_radii(rdmol2, intramolecular=intramolecular)
        except (IndexError, AttributeError) as e:
            logging.error(f'[ERROR] ClashEvaluator failed: {e}')
            return None

        dist = cdist(coord1, coord2, metric="euclidean")
        if intramolecular:
            np.fill_diagonal(dist, np.inf)

        clashes = dist < self.margin * (radii1[:, None] + radii2[None, :])
        clashes = np.any(clashes, axis=1)
        return np.mean(clashes)
    
    def coord_and_radii(self, rdmol, intramolecular):
        _periodic_table = Chem.GetPeriodicTable()
        _get_radius = _periodic_table.GetRcovalent if intramolecular else _periodic_table.GetRvdw

        coord = rdmol.GetConformer().GetPositions()
        radii = np.array([_get_radius(a.GetSymbol()) for a in rdmol.GetAtoms()])

        mask = np.array([a.GetSymbol() not in self.ignore for a in rdmol.GetAtoms()])
        coord = coord[mask]
        radii = radii[mask]

        assert coord.shape[0] == radii.shape[0]
        return coord, radii


class RingCountEvaluator(AbstractEvaluator):
    ID = 'ring_count'
    DTYPES = {'*': int}

    def evaluate(self, molecule, protein=None, **kwargs):
        _mol = self.load_molecule(molecule)

        # compute ring info if not yet available
        try:
            _mol.UpdatePropertyCache()
        except ValueError:
            return {}
        Chem.GetSymmSSSR(_mol)

        rings = _mol.GetRingInfo().AtomRings()
        ring_sizes = [len(r) for r in rings]

        ring_counts = defaultdict(int)
        for k in ring_sizes:
            ring_counts[f"num_{k}_rings"] += 1

        return ring_counts


class ChemblRingEvaluator(AbstractEvaluator):
    ID = 'chembl_ring_systems'
    DTYPES = {
        'min_ring_smi': str,
        'min_ring_freq_gt0_': bool,
        'min_ring_freq_gt10_': bool,
        'min_ring_freq_gt100_': bool,
    }

    def __init__(self):
        self.ring_system_lookup = RingSystemLookup.default()  # ChEMBL

    def evaluate(self, molecule, protein=None, **kwargs):

        results = {
            'min_ring_smi': None,
            'min_ring_freq_gt0_': None,
            'min_ring_freq_gt10_': None,
            'min_ring_freq_gt100_': None,
        }

        molecule = self.load_molecule(molecule)

        try:
            Chem.SanitizeMol(molecule)
            freq_list = self.ring_system_lookup.process_mol(molecule)
            freq_list = self.ring_system_lookup.process_mol(molecule)
        except ValueError:
            return results

        min_ring, min_freq = get_min_ring_frequency(freq_list)

        return {
            'min_ring_smi': min_ring,
            'min_ring_freq_gt0_': min_freq > 0,
            'min_ring_freq_gt10_': min_freq > 10,
            'min_ring_freq_gt100_': min_freq > 100,
        }


class REOSEvaluator(AbstractEvaluator):
    # Based on https://practicalcheminformatics.blogspot.com/2024/05/generative-molecular-design-isnt-as.html
    ID = 'reos'
    DTYPES = {'*': bool}

    def __init__(self):
        self.reos = REOS()

    def evaluate(self, molecule, protein=None, **kwargs):
        
        molecule = self.load_molecule(molecule)
        try:
            Chem.SanitizeMol(molecule)
        except ValueError:
            return {rule_set: False for rule_set in self.reos.get_available_rule_sets()}
        
        results = {}
        for rule_set in self.reos.get_available_rule_sets():
            self.reos.set_active_rule_sets([rule_set])
            if rule_set == 'PW':
                self.reos.drop_rule('furans')

            reos_res = self.reos.process_mol(molecule)
            results[rule_set] = reos_res[0] == 'ok'

        results['all'] = all([bool(value) if not is_nan(value) else False for value in results.values()])
        return results


class StarDropAlertsEvaluator(AbstractEvaluator):
    ID = 'stardrop'
    DTYPES = {'*': bool}
    STARDROP_ALERT_SMARTS = [
        '*1[O,S,N]*1',
        '[S,C](=[O,S])[F,Br,Cl,I]',
        '[CX4][Cl,Br,I]',
        '[C,c]S(=O)(=O)O[C,c]',
        '[$([CH]),$(CC)]#CC(=O)[C,c]',
        '[$([CH]),$(CC)]#CC(=O)O[C,c]',
        'n[OH]',
        '[$([CH]),$(CC)]#CS(=O)(=O)[C,c]',
        'C=C(C=O)C=O',
        'n1c([F,Cl,Br,I])cccc1',
        '[CH1](=O)',
        '[O,o][O,o]',
        '[C;!R]=[N;!R]',
        '[N!R]=[N!R]',
        '[#6](=O)[#6](=O)',
        '[S,s][S,s]',
        '[N,n][NH2]',
        'C(=O)N[NH2]',
        '[C,c]=S',
        '[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]=[$([CH2]),$([CH][CX4]),$(C([CX4])[CX4])]',
        'C1(=[O,N])C=CC(=[O,N])C=C1',
        'C1(=[O,N])C(=[O,N])C=CC=C1',
        'a21aa3a(aa1aaaa2)aaaa3',
        'a31a(a2a(aa1)aaaa2)aaaa3',
        'a1aa2a3a(a1)A=AA=A3=AA=A2',
        'c1cc([NH2])ccc1',
        '[Hg,Fe,As,Sb,Zn,Se,Te,B,Si,Na,Ca,Ge,Ag,Mg,K,Ba,Sr,Be,Ti,Mo,Mn,Ru,Pd,Ni,Cu,Au,Cd,Al,Ga,Sn,Rh,Tl,Bi,Nb,Li,Pb,Hf,Ho]',
        'I',
        'OS(=O)(=O)[O-]',
        '[$([N+](=O)[O-]),$(N(=O)=O)]',
        'C(=O)N[OH]',
        'C1NC(=O)NC(=O)1',
        '[SH]',
        '[S-]',
        'c1ccc([Cl,Br,I,F])c([Cl,Br,I,F])c1[Cl,Br,I,F]',
        'c1cc([Cl,Br,I,F])cc([Cl,Br,I,F])c1[Cl,Br,I,F]',
        '[CR1]1[CR1][CR1][CR1][CR1][CR1][CR1]1',
        '[CR1]1[CR1][CR1]cc[CR1][CR1]1',
        '[CR2]1[CR2][CR2][CR2][CR2][CR2][CR2][CR2]1',
        '[CR2]1[CR2][CR2]cc[CR2][CR2][CR2]1',
        '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
        '[CH2R2]1N[CH2R2][CH2R2][CH2R2][CH2R2][CH2R2][CH2R2]1',
        'C#C',
        '[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]@[CR2]@[CR2]@[OR2,NR2]',
        '[$([N+R]),$([n+R]),$([N+]=C)][O-]',
        '[C,c]=N[OH]',
        '[C,c]=NOC=O',
        '[C,c](=O)[CX4,CR0X3,O][C,c](=O)',
        'c1ccc2c(c1)ccc(=O)o2',
        '[O+,o+,S+,s+]',
        'N=C=O',
        '[NX3,NX4][F,Cl,Br,I]',
        'c1ccccc1OC(=O)[#6]',
        '[CR0]=[CR0][CR0]=[CR0]',
        '[C+,c+,C-,c-]',
        'N=[N+]=[N-]',
        'C12C(NC(N1)=O)CSC2',
        'c1c([OH])c([OH,NH2,NH])ccc1',
        'P',
        '[N,O,S]C#N',
        'C=C=O',
        '[Si][F,Cl,Br,I]',
        '[SX2]O',
        '[SiR0,CR0](c1ccccc1)(c2ccccc2)(c3ccccc3)',
        'O1CCCCC1OC2CCC3CCCCC3C2',
        'N=[CR0][N,n,O,S]',
        '[cR2]1[cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2][cR2]1[cR2]2[cR2][cR2][cR2]([Nv3X3,Nv4X4])[cR2][cR2]2',
        'C=[C!r]C#N',
        '[cR2]1[cR2]c([N+0X3R0,nX3R0])c([N+0X3R0,nX3R0])[cR2][cR2]1',
        '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2]c([N+0X3R0,nX3R0])[cR2]1',
        '[cR2]1[cR2]c([N+0X3R0,nX3R0])[cR2][cR2]c1([N+0X3R0,nX3R0])',
        '[OH]c1ccc([OH,NH2,NH])cc1',
        'c1ccccc1OC(=O)O',
        '[SX2H0][N]',
        'c12ccccc1(SC(S)=N2)',
        'c12ccccc1(SC(=S)N2)',
        'c1nnnn1C=O',
        's1c(S)nnc1NC=O',
        'S1C=CSC1=S',
        'C(=O)Onnn',
        'OS(=O)(=O)C(F)(F)F',
        'N#CC[OH]',
        'N#CC(=O)',
        'S(=O)(=O)C#N',
        'N[CH2]C#N',
        'C1(=O)NCC1',
        'S(=O)(=O)[O-,OH]',
        'NC[F,Cl,Br,I]',
        'C=[C!r]O',
        '[NX2+0]=[O+0]',
        '[$([OR0,NR0][OR0,NR0]),$(N(=O)=O)]',
        '(C(=O)O[C,H1]).(C(=O)O[C,H1]).(C(=O)O[C,H1])',
        '[CX2R0][NX3R0]',
        'c1ccccc1[C;!R]=[C;!R]c2ccccc2',
        '[NX3R0,NX4R0,OR0,SX2R0][CX4][NX3R0,NX4R0,OR0,SX2R0]',
        '[s,S,c,C,n,N,o,O]~[n+,N+](~[s,S,c,C,n,N,o,O])(~[s,S,c,C,n,N,o,O])~[s,S,c,C,n,N,o,O]',
        '[s,S,c,C,n,N,o,O]~[nX3+,NX3+](~[s,S,c,C,n,N])~[s,S,c,C,n,N]',
        '[*]=[N+]=[*]',
        '[SX3](=O)[O-,OH]',
        'N#N',
        'F.F.F.F',
        '[R0;D2][R0;D2][R0;D2][R0;D2]',
        '[cR,CR]~C(=O)NC(=O)~[cR,CR]',
        'C=!@CC=[O,S]',
        '[#6,#8,#16][C,c](=O)O[C,c]',
        'c[C;R0](=[O,S])[C,c]',
        'c[SX2][C;!R]',
        'C=C=C',
        'c1nc([F,Cl,Br,I,S])ncc1',
        'c1ncnc([F,Cl,Br,I,S])c1',
        'c1nc(c2c(n1)nc(n2)[F,Cl,Br,I])',
        '[C,c]S(=O)(=O)c1ccc(cc1)F',
        '[15N,13C,18O,2H,34S]'
    ]

    def __init__(self):
        self.alerts = []
        for smarts in self.STARDROP_ALERT_SMARTS:
            alert = Chem.MolFromSmarts(smarts)
            if alert is not None:
                self.alerts.append(alert)

    def is_valid(self, mol):
        for alert in self.alerts:
            if mol.HasSubstructMatch(alert):
                return False
        return True
    
    def count_hydroxyl_groups(self, molecule):
        hydroxyl_group = Chem.MolFromSmarts('[OX2H]')
        num_matches = len(molecule.GetSubstructMatches(hydroxyl_group))
        return num_matches

    def evaluate(self, molecule, protein=None, **kwargs):        
        molecule = self.load_molecule(molecule)
        try:
            Chem.SanitizeMol(molecule)
        except ValueError:
            return {'passed': False}
        return {
            'passed': self.is_valid(molecule),
            'num_hydroxyl_groups': self.count_hydroxyl_groups(molecule),
        }


class FingerprintEvaluator(AbstractEvaluator):
    ID = None
    DTYPES = {
        '*': list,
    }
    def __init__(
            self, 
            fpgen=AllChem.GetMorganGenerator(radius=2,fpSize=2048), 
            use_counts=False, 
            similarity="Tanimoto",
        ):

        self.fpgen = fpgen
        self.fp_fn = self.fpgen.GetSparseCountFingerprint if use_counts else self.fpgen.GetFingerprint
        self.sim_fn = getattr(DataStructs, f"Bulk{similarity}Similarity")

    def get_fingerprint(self, molecule):

        # mol provided as SMILES string
        if isinstance(molecule, str):
            molecule = Chem.MolFromSmiles(molecule)

        return self.fp_fn(molecule)


class FingerprintNoveltyEvaluator(FingerprintEvaluator):
    ID = 'fingerprint_novelty'
    DTYPES = {
        'closest_match_sim': float, 
        'closest_match_id': str,
        'closest_match_smiles': str,
    }

    def __init__(
            self, 
            reference_mols: Union[str, Path, Collection[Union[str, Chem.Mol]]],
            fpgen=AllChem.GetMorganGenerator(radius=2,fpSize=2048), 
            use_counts=False, 
            similarity="Tanimoto",
        ):

        super().__init__(fpgen=fpgen, use_counts=use_counts, similarity=similarity)
        if isinstance(reference_mols, (str, Path)):
            self.data = self.load_reference_fps(reference_mols)
        elif isinstance(reference_mols, abcCollection):
            self.data = self.precompute_reference_fps(reference_mols)
        
    def precompute_reference_fps(self, molecules):
        data = []
        already_included = set()
        for mol in tqdm(molecules, desc="Computing fingerprints"):
            if isinstance(mol, str):
                try:
                    mol = Chem.MolFromSmiles(mol)
                except Exception as e:
                    self.warn(f"Failed to parse SMILES '{mol}': {e}")
                    continue
            try:
                Chem.SanitizeMol(mol)
                fp = self.get_fingerprint(mol)
                binary_fp = fp.ToBinary()
            except Exception as e:
                self.warn(f"Failed to get fingerprint: {e}")
                continue

            if binary_fp in already_included:
                continue
            already_included.add(binary_fp)

            data.append({
                'id': str(len(data)),
                'fp': fp,
                'smiles': Chem.MolToSmiles(mol),
            })
        data = pd.DataFrame(data)
        return data

    @staticmethod
    def load_reference_fps(reference_fp_path):
        with open(reference_fp_path, "rb") as f:
            ref = pickle.load(f)
        return ref

    def evaluate(self, molecule, protein=None, **kwargs):
        molecule = self.load_molecule(molecule)
        try:
            Chem.SanitizeMol(molecule)
            fp = self.get_fingerprint(molecule)
            all_similarities = self.sim_fn(fp, self.data['fp'])
            idxmax = np.argmax(all_similarities)
            closest_match_sim = all_similarities[idxmax]
            closest_id = self.data['id'][idxmax]
            closest_smiles = self.data['smiles'][idxmax]
        except ValueError as e:
            self.warn(e)
            closest_match_sim = None 
            closest_id = None
            closest_smiles = None
        return {
            'closest_match_sim': closest_match_sim,
            'closest_match_id': closest_id,
            'closest_match_smiles': closest_smiles,
        }


class StrainEvaluator(AbstractEvaluator):
    ID = 'strain'
    DTYPES = {'*': float}

    def __init__(self, max_iter=200, ff_type="MMFF"):
        self.max_iter = max_iter
        assert ff_type in {"MMFF", "UFF"}
        self.ff_type = ff_type

    def get_force_field(self, mol, nonbonded=True):
        if self.ff_type == "MMFF":
            mmff_mol_props = MMFFGetMoleculeProperties(mol)
            nonbonded_thresh = 100.0 if nonbonded else 0.0
            ff = MMFFGetMoleculeForceField(mol, mmff_mol_props, nonBondedThresh=nonbonded_thresh)
        elif self.ff_type == "UFF":
            # vdw_thresh = 10.0 if nonbonded else 0.0
            # ff = UFFGetMoleculeForceField(mol, vdwThresh=vdw_thresh)
            if not nonbonded:
                self.warn("Turning off non-bonded terms not implemented for UFF.")
            ff = UFFGetMoleculeForceField(mol)
        else:
            raise NotImplementedError()
        return ff

    @staticmethod
    def get_neighbor(mol, atom_idx: int, exclude: int) -> int:
        focus_atom = mol.GetAtomWithIdx(atom_idx)
        nb_indices = set(a.GetIdx() for a in focus_atom.GetNeighbors())
        nb_indices = nb_indices - set([exclude])  # exclude index
        assert len(nb_indices) >= 1
        return list(nb_indices)[0]  # return one of the remaining indices
    
    @staticmethod
    def get_rotatable_bonds(mol, return_atom_pairs=False):
        # taken from https://github.com/rdkit/rdkit/blob/64061b6ca71121f7c3837393ff0a35f6261596a9/rdkit/Chem/Lipinski.py#L41
        RotatableBondSmarts = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')

        rotatable_bonds = mol.GetSubstructMatches(RotatableBondSmarts)
        if return_atom_pairs:
            return rotatable_bonds

        bond_indices = [mol.GetBondBetweenAtoms(idx1, idx2).GetIdx() for idx1, idx2 in rotatable_bonds]
        return bond_indices
    
    @classmethod
    def get_fixed_torsion_angles(cls, mol):

        # add one torsion constraint for each rotatable bond
        torsion_angle_indices = []
        # for idx2, idx3 in cls.get_rotatable_bonds(mol, return_atom_pairs=True):
        for idx2, idx3 in cls.get_rotatable_bonds(Chem.RemoveHs(mol), return_atom_pairs=True):
            idx1 = cls.get_neighbor(mol, idx2, exclude=idx3)
            idx4 = cls.get_neighbor(mol, idx3, exclude=idx2)
            torsion_angle_indices.append((idx1, idx2, idx3, idx4))

        return torsion_angle_indices

    def add_torsion_constraints(self, force_field, mol):
        """Fix rotatable bonds."""

        # add one torsion constraint for each rotatable bond
        for idx1, idx2, idx3, idx4 in self.get_fixed_torsion_angles(mol):
            
            getattr(force_field, f"{self.ff_type}AddTorsionConstraint")(
                idx1, idx2, idx3, idx4, 
                relative=True, 
                minDihedralDeg=0.0, 
                maxDihedralDeg=0.0, 
                forceConstant=1e6,
            )

        return force_field

    def compute_dE(self, mol, fix_rotatable_bonds=False, include_nonbonded=False):

        # Add hydrogens
        new_mol = Chem.AddHs(mol, addCoords=True)

        if not getattr(rdForceFieldHelpers, f"{self.ff_type}HasAllMoleculeParams")(new_mol):
            raise RuntimeError(f"{self.ff_type} parameters not available for all atoms.")

        ff = self.get_force_field(new_mol, nonbonded=include_nonbonded)
        if fix_rotatable_bonds:
            ff = self.add_torsion_constraints(ff, new_mol)

        ff.Initialize()
        # energy_before = ff.CalcEnergy()
        energy_before = self.get_force_field(new_mol, include_nonbonded).CalcEnergy()
        ff.Minimize(maxIts=self.max_iter)
        # energy_after = ff.CalcEnergy()
        energy_after = self.get_force_field(new_mol, include_nonbonded).CalcEnergy()

        return energy_before - energy_after

    def evaluate(self, molecule, protein=None, **kwargs):
        molecule = self.load_molecule(molecule)

        # Initialize output dictionary
        out = {
            "dE": None,
        }

        try:
            Chem.SanitizeMol(molecule)
            out["dE"] = self.compute_dE(molecule, 
                                        fix_rotatable_bonds=False, 
                                        include_nonbonded=False) 
        except (RuntimeError, ValueError) as e:
            self.warn(e)
        
        # Add normalized values
        _tmp = {}
        for k, v in out.items():
            _tmp[f"{k}_per_heavy_atom"] = v / molecule.GetNumHeavyAtoms() if v is not None else None

        out.update(_tmp)

        return out
    

class ForceFieldEvaluator(AbstractEvaluator):
    ID = 'ff_relaxation'
    DTYPES = {
        "converged": bool,
        "energy_before": float,
        "energy_after": float,
        "rmsd": float,
        "relaxed_molecule": Chem.Mol,
    }

    def __init__(self, max_iter=500, ff_type="UFF"):
        self.max_iter = max_iter
        assert ff_type in {"MMFF", "UFF"}
        self.ff_type = ff_type

    def _trim_protein(self, protein, ligand, thresh=5.0):
        # only keep immediately interacting residues for efficiency
    
        def get_res_id(atom):
            try:
                chain_id = atom.GetPDBResidueInfo().GetChainId()
                resnum = atom.GetPDBResidueInfo().GetResidueNumber()
            except AttributeError as e:
                return None
            return (chain_id, resnum)
    
        residue_distances = {}
            
        # for each residue, compute distance to ligand
        for idx in range(protein.GetNumAtoms()):
            atom_coord = protein.GetConformer().GetAtomPosition(idx)
            res_id = get_res_id(protein.GetAtomWithIdx(idx))
            if res_id is None:
                continue
    
            dists = [
                atom_coord.Distance(ligand.GetConformer().GetAtomPosition(lig_idx))
                for lig_idx in range(ligand.GetNumAtoms())
            ]
            dist_to_ligand = min(dists)
    
            if res_id in residue_distances:
                residue_distances[res_id] = min(residue_distances[res_id], dist_to_ligand)
            else:
                residue_distances[res_id] = dist_to_ligand
    
        # remove atoms
        trimmed_pocket = Chem.EditableMol(protein)
        for idx in reversed(range(protein.GetNumAtoms())):
            res_id = get_res_id(protein.GetAtomWithIdx(idx))
            if res_id is None or residue_distances[res_id] > thresh:
                trimmed_pocket.RemoveAtom(idx)
                
        return trimmed_pocket.GetMol()

    def relax_ligand(self, ligand: Chem.Mol, protein: Chem.Mol = None, fix_atoms: List[int] = None, max_iter: int = 500):

        # Prepare the molecules
        if protein is not None:
            pocket = self._trim_protein(protein, ligand)
            ligand = Chem.AddHs(ligand, addCoords=True)
            pocket = Chem.AddHs(pocket, addCoords=True)
            complex = Chem.CombineMols(ligand, pocket)
        else:
            ligand = Chem.AddHs(ligand, addCoords=True)
            complex = Chem.Mol(ligand)
        Chem.SanitizeMol(complex)
        
        # Set up the force field
        if self.ff_type == "MMFF":
            mmff_mol_props = AllChem.MMFFGetMoleculeProperties(complex)
            ff = AllChem.MMFFGetMoleculeForceField(complex, mmff_mol_props, nonBondedThresh=10.0, ignoreInterfragInteractions=False)
        elif self.ff_type == "UFF":
            ff = AllChem.UFFGetMoleculeForceField(complex, ignoreInterfragInteractions=False)
        else:
            raise NotImplementedError()

        # Add constraints
        if fix_atoms is None:
            fix_atoms = []
        fix_atoms.extend(range(ligand.GetNumAtoms(), complex.GetNumAtoms()))
    
        # Fix protein and, optionally, parts of the ligand
        for idx in fix_atoms:
            atom = complex.GetAtomWithIdx(idx)
            if atom.GetSymbol() != 'H':
                if self.ff_type == "MMFF":
                    ff.MMFFAddPositionConstraint(idx, maxDispl=0.0, forceConstant=1.e4)
                elif self.ff_type == "UFF":
                    ff.UFFAddPositionConstraint(idx, maxDispl=0.0, forceConstant=1.e4)
                else:
                    raise NotImplementedError()

        # Minimize
        ff.Initialize()
        energy_before = ff.CalcEnergy()
        not_converged = ff.Minimize(maxIts=max_iter)
        energy_after = ff.CalcEnergy()
    
        ligand = Chem.GetMolFrags(complex, asMols=True)[0]
        ligand = Chem.RemoveAllHs(ligand)

        info = {
            "converged": not_converged == 0,
            "energy_before": energy_before,
            "energy_after": energy_after,
        }

        return ligand, info

    def evaluate(self, molecule, protein=None, fix_atoms: List[int] = None, **kwargs):
        molecule = self.load_molecule(molecule)

        # Initialize output dictionary
        out = {
            "converged": None,
            "energy_before": None,
            "energy_after": None,
            "rmsd": None,
            "relaxed_molecule": None,
        }

        try:
            if protein is not None:
                protein = Chem.MolFromPDBFile(str(protein), sanitize=False)
                assert protein is not None

            minimized_molecule, info = self.relax_ligand(molecule, protein, max_iter=self.max_iter, fix_atoms=fix_atoms)
            out.update(info)
            out["rmsd"] = AllChem.CalcRMS(molecule, minimized_molecule)
            out["relaxed_molecule"] = minimized_molecule
        except (RuntimeError, ValueError, AssertionError) as e:
            # raise(e)
            self.warn(e)

        return out
    

class FCDEmbeddingEvaluator(AbstractEvaluator):
    ID = 'fcd_embedding'
    DTYPES = {"embedding": np.ndarray}

    def __init__(self, device='cpu'):
        self.model = load_ref_model()
        self.device = device

    def evaluate(self, molecule, protein=None, **kwargs):
        molecule = self.load_molecule(molecule)
        c_smiles = canonical_smiles([Chem.MolToSmiles(molecule)])
        if c_smiles is None:
            return {"embedding": None}
        embedding = get_predictions(self.model, c_smiles, device=self.device)[0]
        return {"embedding": embedding}


class UncertaintyEvaluator(AbstractEvaluator):
    ID = 'uncertainty'
    DTYPES = {
        'mean_uncertainty': float,
        'min_uncertainty': float,
        'max_uncertainty': float,
    }

    def __init__(self):
        super().__init__()

    def evaluate(self, molecule, protein=None, **kwargs):
        molecule = self.load_molecule(molecule)
        # excluding fixed atoms
        uncert_x = [float(atom.GetProp('sigma_x')) 
            for atom in molecule.GetAtoms() 
            if atom.HasProp('sigma_x') and float(atom.GetProp('sigma_x')) > 0
        ]
        entropy_h = [float(atom.GetProp('entropy_h')) 
            for atom in molecule.GetAtoms() 
            if atom.HasProp('entropy_h') and float(atom.GetProp('entropy_h')) > 0
        ]
        if len(uncert_x) == 0 and molecule.HasProp('sigma_x'):
            uncert_x = molecule.GetProp('sigma_x').split(',')
            if len(uncert_x) == molecule.GetNumAtoms():
                uncert_x = [float(u) for u in uncert_x if float(u) > 0] # excluding fixed atoms
            else:
                uncert_x = []
        if len(entropy_h) == 0 and molecule.HasProp('entropy_h'):
            entropy_h = molecule.GetProp('entropy_h').split(',')
            if len(entropy_h) == molecule.GetNumAtoms():
                entropy_h = [float(h) for h in entropy_h if float(h) > 0] # excluding fixed atoms
            else:
                entropy_h = []
        mean_uncert_x = np.mean(uncert_x) if len(uncert_x) > 0 else None
        min_uncert_x = np.min(uncert_x) if len(uncert_x) > 0 else None
        max_uncert_x = np.max(uncert_x) if len(uncert_x) > 0 else None
        mean_entropy_h = np.mean(entropy_h) if len(entropy_h) > 0 else None
        min_entropy_h = np.min(entropy_h) if len(entropy_h) > 0 else None
        max_entropy_h = np.max(entropy_h) if len(entropy_h) > 0 else None
        return {
            'mean_uncertainty': mean_uncert_x, 'min_uncertainty': min_uncert_x, 'max_uncertainty': max_uncert_x,
            'mean_entropy_h': mean_entropy_h, 'min_entropy_h': min_entropy_h, 'max_entropy_h': max_entropy_h
        }


class FullEvaluator(AbstractEvaluator):
    COMPONENTS = [
        RepresentationEvaluator,
        MolPropertyEvaluator,
        PoseBustersEvaluator,
        Validity3DEvaluator,
        MedChemEvaluator,
        ClashEvaluator,
        GninaEvaluator,
        InteractionsEvaluator,
        GeometryEvaluator,
        RingCountEvaluator,
        EnergyEvaluator,
        ChemblRingEvaluator,
        REOSEvaluator,
        StarDropAlertsEvaluator,
        StrainEvaluator,
        ForceFieldEvaluator,
        FingerprintNoveltyEvaluator,
        UncertaintyEvaluator,
    ]
        
    DTYPES = {}
    for evaluator_class in COMPONENTS:
        DTYPES.update(evaluator_class.add_id(evaluator_class.DTYPES))
        DTYPES.update(evaluator_class.add_id({'time': float}))
        DTYPES['time'] = float

    def __init__(
            self,
            pb_conf: Optional[Union[Path, str]] = 'dock',
            gnina: Optional[Union[Path, str]] = None,
            reduce: Optional[Union[Path, str]] = None,
            reference_mols: Optional[Union[Collection[Union[str, Chem.Mol]], Path, str]] = None,
            reference_mols_validity3d: Optional[Union[Path, str]] = None,
            connectivity_threshold: float = 1.0, 
            margin: float = 0.75, 
            ignore: Set[str] = {'H'},
            exclude_evaluators: Collection[str] = [],
            protein: Union[Path, str] = None,
            relaxation_max_iter: int = 500,
    ):
        all_evaluators = [  # (cls, kwargs) pairs
            (RepresentationEvaluator, {}),
            (MolPropertyEvaluator, {}),
            (PoseBustersEvaluator, {"pb_conf": pb_conf}),
            (MedChemEvaluator, {"connectivity_threshold": connectivity_threshold}),
            (ClashEvaluator, {"margin": margin, "ignore": ignore}),
            (GeometryEvaluator, {}),
            (RingCountEvaluator, {}),
            (EnergyEvaluator, {}),
            (ChemblRingEvaluator, {}),
            (REOSEvaluator, {}),
            (StarDropAlertsEvaluator, {}),
            (StrainEvaluator, {}),
            (UncertaintyEvaluator, {}),
            (ForceFieldEvaluator, {"max_iter": relaxation_max_iter}),
        ]
        if gnina is not None:
            all_evaluators.append(
                (GninaEvaluator, {"gnina": gnina})
            )
        else:
            logging.debug(f'Evaluator [{GninaEvaluator.ID}] is not included')
        if reduce is not None:
            all_evaluators.append(
                (InteractionsEvaluator, {"reduce": reduce, "protein": protein})
            )
        else:
            logging.debug(f'Evaluator [{InteractionsEvaluator.ID}] is not included')
        if reference_mols is not None:
            all_evaluators.append(
                (FingerprintNoveltyEvaluator, {"reference_mols": reference_mols})
            )
        else:
            logging.debug(f'Evaluator [{FingerprintNoveltyEvaluator.ID}] is not included')
        if reference_mols_validity3d is not None:
            all_evaluators.append(
                (Validity3DEvaluator, {"reference_ligands_path": Path(reference_mols_validity3d)})
            )
        else:
            logging.debug(f'Evaluator [{Validity3DEvaluator.ID}] is not included')

        self.evaluators = []
        for E, kwargs in all_evaluators:
            if E.ID in exclude_evaluators:
                logging.debug(f'Excluded Evaluator [{E.ID}]')
            else:
                self.evaluators.append(E(**kwargs))

        logging.info('Will use the following evaluators:')
        for e in self.evaluators:
            logging.info(f'- [{e.ID}]')

    def evaluate(self, molecule, protein, verbose=False, **kwargs):
        results = {}
        for evaluator in self.evaluators:
            if verbose:
                logging.debug(f'Evaluating {evaluator.ID}')
            results.update(evaluator(molecule, protein, **kwargs))
        return results
        
    def get_evaluator(self, evaluator_id):
        for evaluator in self.evaluators:
            if evaluator.ID == evaluator_id:
                return evaluator
        raise KeyError(evaluator_id)

    def update_protein(self, protein: Union[Path, str]):
        for evaluator in self.evaluators:
            if hasattr(evaluator, 'update_protein'):
                logging.debug(f'Updating protein for evaluator {evaluator.ID}')
                evaluator.update_protein(protein)

########################################################################################
################################## Per-atom Metrics ####################################
########################################################################################

class AbstractLocalEvaluator(AbstractEvaluator):
    DTYPES = {'*': list}

    def __init__(self, results_to_dict=True):
        self.results_to_dict = results_to_dict
        
    def __call__(self, molecule: Union[str, Path, Chem.Mol], protein: Union[str, Path] = None, **kwargs):
        """
        Args:
            molecule (Union[str, Path, Chem.Mol]): input molecule
            protein (str): target protein
        
        Returns:
            metrics (dict): dictionary of metrics
        """
        RDLogger.DisableLog('rdApp.*')
        self.check_format(molecule, protein)
        results = self.evaluate(molecule, protein, **kwargs)
        if self.results_to_dict:
            mol = self.load_molecule(molecule)
            results = self.transform_results(mol, results)
        return self.add_id(results)
    
    @abstractmethod
    def evaluate(self, molecule: Union[str, Path, Chem.Mol], protein: Union[str, Path], **kwargs) -> Dict[str, List[Union[bool, int, float]]]:
        raise NotImplementedError
       
    @staticmethod
    def transform_results(molecule, results):
        # checks if _InitialIndex is present, then resorts the results
        if any(atm.HasProp('_InitialIndex') for atm in molecule.GetAtoms()):
            atom_idx_to_init_idx = {atom.GetIdx(): int(atom.GetProp('_InitialIndex')) for atom in molecule.GetAtoms()
                                        if atom.HasProp('_InitialIndex')}
        else:
            atom_idx_to_init_idx = {atom.GetIdx(): atom.GetIdx() for atom in molecule.GetAtoms()}
        
        new_results = {}
        for key, value in results.items():
            new_results[key] = {}
            for atm_idx, val in enumerate(value):
                if atm_idx in atom_idx_to_init_idx:
                    new_results[key][atom_idx_to_init_idx[atm_idx]] = val
        return new_results

class LocalInteractionsEvaluator(AbstractLocalEvaluator):
    ID = 'interactions_local'

    UNSATISFIED_HBOND_METRICS = [
        "unsatisfied_hbond_donors",
        "unsatisfied_hbond_acceptors",
        "UHBDonor",
        "UHBAcceptor",
        "UHB",  # unsatisfied H-bonds
    ]

    def __init__(self, reduce='reduce', guess_waters=True, protein=None, interaction_list=None, residue_constraints=None):
        super().__init__(results_to_dict=False)
        self.reduce = reduce
        self.guess_waters = guess_waters
        self.protein_plf = read_protein(str(protein), reduce_exec=str(self.reduce)) if protein is not None else None
        if interaction_list is None:
            interaction_list = INTERACTION_LIST
        self.interaction_list = interaction_list
        # Unsatisfied H-bonds are derived below, not ProLIF interaction classes.
        self.prolif_interactions = [
            name for name in interaction_list if name not in self.UNSATISFIED_HBOND_METRICS
        ]
        if any(name in self.UNSATISFIED_HBOND_METRICS for name in interaction_list):
            self.prolif_interactions = list(dict.fromkeys(
                self.prolif_interactions + ['HBAcceptor', 'HBDonor']
            ))
        self.residue_constraints = None
        if isinstance(residue_constraints, str) or isinstance(residue_constraints, Path):
            with open(residue_constraints, 'r') as f:
                residue_constraints = json.load(f)
            self.residue_constraints = residue_constraints
        elif isinstance(residue_constraints, dict):
            self.residue_constraints = residue_constraints

    def update_protein(self, protein: Union[Path, str]):
        if protein is None:
            self.protein_plf = None
            return
        if isinstance(protein, (Path, str)):
            protein = Path(protein)
        self.protein_plf = read_protein(str(protein), reduce_exec=str(self.reduce))

    @staticmethod
    def prepare_molecule(molecule):
        for idx, atom in enumerate(molecule.GetAtoms()):
            if not atom.HasProp('_InitialIndex'):
                atom.SetProp('_InitialIndex', str(idx))
        molecule = Chem.AddHs(molecule, addCoords=True)
        return molecule
    
    def get_default_profile(self, molecule):
        counts = [0]*len(molecule.GetAtoms())
        cols = self.interaction_list + ['unsatisfied_hbond_donors', 'unsatisfied_hbond_acceptors']
        return {i: counts.copy() for i in cols}
    
    def get_unsatisfied_hbonds(self, molecule, interactions, protein=None):
        potential_donors = set(get_hbond_donors(molecule))
        potential_acceptors = set(get_hbond_acceptors(molecule))

        found_donors = set([int(idx) for row in interactions[interactions.interaction == "HBDonor"].ligand_atoms for idx in row.split(",")])
        found_acceptors = set([int(idx) for row in interactions[interactions.interaction == "HBAcceptor"].ligand_atoms for idx in row.split(",")])

        is_unsatisfied_hbond_donor = [0]*len(molecule.GetAtoms())
        is_unsatisfied_hbond_acceptor = [0]*len(molecule.GetAtoms())
        for idx in potential_donors - found_donors:
            is_unsatisfied_hbond_donor[int(idx)] = 1
        for idx in potential_acceptors - found_acceptors:
            is_unsatisfied_hbond_acceptor[int(idx)] = 1

        res = {
            "unsatisfied_hbond_donors": is_unsatisfied_hbond_donor,
            "unsatisfied_hbond_acceptors": is_unsatisfied_hbond_acceptor,
        }

        if self.guess_waters:
            assert protein is not None
            mol_no_h = Chem.RemoveAllHs(molecule, sanitize=True)
            context = InteractionsEvaluator._get_coords_and_radii(protein, mol_no_h, cutoff=10.0)
            water_radius = 0.5  # literature value seems to be too strict
            unsatisfied_donors = [1 
                if is_unsatisfied_hbond_donor[idx] and not InteractionsEvaluator.enough_space_for_water(
                    mol_no_h, idx, context, 'donor', water_radius)
                else 0 
                for idx in range(len(molecule.GetAtoms()))]
            unsatisfied_acceptors = [1
                if is_unsatisfied_hbond_acceptor[idx] and not InteractionsEvaluator.enough_space_for_water(
                    mol_no_h, idx, context, 'acceptor', water_radius)
                else 0
                for idx in range(len(molecule.GetAtoms()))]
            # uhbs (or operation between unsatisfied_donors and unsatisfied_acceptors)
            uhbs = [1 if unsatisfied_donors[idx] or unsatisfied_acceptors[idx] else 0 for idx in range(len(molecule.GetAtoms()))]
            res.update({
                "UHBDonor": unsatisfied_donors,
                "UHBAcceptor": unsatisfied_acceptors,
                "UHB": uhbs,
            })

        return res
    
    def update_profile(self, profile, interactions, only_assign_first_atom=True):
        for interact_name in self.interaction_list:
            filt_interactions = interactions.loc[interactions.interaction == interact_name]
            for atms in filt_interactions.ligand_atoms:
                atom_list = atms.split(",")
                if only_assign_first_atom:
                    atom_list = [atom_list[0]]  # only assign the first atom
                for i in atom_list:
                    profile[interact_name][int(i)] += 1
        return profile

    def evaluate(self, molecule, protein=None, timeout=60, **kwargs):
        molecule = self.load_molecule(molecule)
        molecule = self.prepare_molecule(molecule)
        profile = self.get_default_profile(molecule)
        assert self.protein_plf is not None or protein is not None
        try:
            ligand_plf = prepare_ligand_plf(molecule)
            protein_plf = read_protein(str(protein), reduce_exec=str(self.reduce)) if self.protein_plf is None else self.protein_plf
            if not (ligand_plf is None or protein_plf is None):
                interactions = safe_run(
                    profile_detailed, 
                    kwargs={'ligand_plf': ligand_plf, 'protein_plf': protein_plf, 'interaction_list': self.prolif_interactions},
                    timeout_duration=timeout
                )
                if interactions is None:
                    profile = self.get_default_profile(molecule)
                elif not interactions.empty:
                    interactions = InteractionsEvaluator.clean_up_interactions(interactions, ligand_plf, protein_plf)
                    interactions = filter_profile(
                        interactions,
                        filter_dict=self.residue_constraints,
                        interaction_list=self.interaction_list
                    )
                    profile = self.update_profile(profile, interactions)
                    profile.update(self.get_unsatisfied_hbonds(ligand_plf, interactions, protein=protein_plf))
        except Exception as e:
            self.warn(f"Error while evaluating interactions: {str(e)}")
            pass
        res = self.transform_results(molecule, profile)
        return res
    

class LocalClashEvaluator(AbstractLocalEvaluator):
    ID = 'clashes_local'

    def __init__(self, margin=0.75, ignore={'H'}):
        super().__init__()
        self.margin = margin
        self.ignore = ignore

    def evaluate(self, molecule=None, protein=None, **kwargs):
        result = {
            'passed_clash_score_ligands': None,
            'passed_clash_score_pockets': None,
            'passed_clash_score_between': None,
        }
        if molecule is not None:
            molecule = self.load_molecule(molecule)
            result['passed_clash_score_ligands'] = self.get_clashes(molecule)

        if protein is not None:
            protein = Chem.MolFromPDBFile(str(protein), sanitize=False)
            result['passed_clash_score_pockets'] = self.get_clashes(protein)
        
        if molecule is not None and protein is not None:
            result['passed_clash_score_between'] = self.get_clashes(molecule, protein)
        
        return result
    
    def get_clashes(self, rdmol1, rdmol2=None):
        """
        Flags all atoms that have at least one clash with another atom.
        """

        intramolecular = rdmol2 is None
        if intramolecular:
            rdmol2 = rdmol1

        coord1, radii1, mask1 = self.coord_and_radii(rdmol1, intramolecular=intramolecular, ignore=self.ignore)
        coord2, radii2, mask2 = self.coord_and_radii(rdmol2, intramolecular=intramolecular, ignore=self.ignore)

        dist = np.sqrt(np.sum((coord1[:, None, :] - coord2[None, :, :]) ** 2, axis=-1))
        if intramolecular:
            np.fill_diagonal(dist, np.inf)

        clashes = dist < self.margin * (radii1[:, None] + radii2[None, :])
        clashes = np.any(clashes, axis=1)
        clashes = clashes & mask1
        return list(~ clashes)
    
    @staticmethod
    def coord_and_radii(rdmol, intramolecular, ignore={'H'}):
        _periodic_table = Chem.GetPeriodicTable()
        _get_radius = _periodic_table.GetRcovalent if intramolecular else _periodic_table.GetRvdw

        coord = rdmol.GetConformer().GetPositions()
        radii = np.array([_get_radius(a.GetSymbol()) for a in rdmol.GetAtoms()])

        mask = np.array([a.GetSymbol() not in ignore for a in rdmol.GetAtoms()])

        assert coord.shape[0] == radii.shape[0]
        return coord, radii, mask


class LocalChiralityEvaluator(AbstractLocalEvaluator):
    ID = 'chirality_local'

    def __init__(self):
        super().__init__()

    def evaluate(self, molecule, protein=None, **kwargs):
        molecule = self.load_molecule(molecule)

        chiral_centers = Chem.FindMolChiralCenters(
            molecule, 
            includeUnassigned=True, 
            includeCIP=False, 
            useLegacyImplementation=False
        )

        chiral_centers_per_atm = [None]*len(molecule.GetAtoms())
        for atom, center in chiral_centers:
            chiral_centers_per_atm[int(atom)] = center
        res = {
            "chiral_centers": chiral_centers_per_atm
        }
        return res
    

class LocalPoseBustersEvaluator(AbstractLocalEvaluator):
    ID = 'posebusters_local'

    def __init__(self, pb_conf: Optional[Union[Path, str]] = 'dock'):
        super().__init__()
        self.fast_geometry = pb_conf == 'mol_fast'
        self.ligand_only = pb_conf in ('mol', 'mol_fast')
        pb_conf = _posebusters_config(pb_conf)
        self.posebusters = PoseBustersLocal(config=pb_conf)

    def evaluate(self, molecule, protein=None, **kwargs):
        mol = self.load_molecule(molecule)
        try:
            Chem.SanitizeMol(mol)
        except (RuntimeError, ValueError):
            return {'all': [False]*mol.GetNumAtoms()}
        if not mol.GetNumConformers() or not np.isfinite(mol.GetConformer().GetPositions()).all():
            return {'all': [False]*mol.GetNumAtoms()}
        self.posebusters.results.clear()
        with suppress_logging() if logging.root.level >= logging.INFO else \
             contextlib.nullcontext():
            inputs = {'mol_pred': mol, 'mol_cond': None if self.ligand_only else protein, 'full_report': True}
            try:
                result = (self.posebusters.bust(**inputs) if self.fast_geometry else
                          safe_run(self.posebusters.bust, kwargs=inputs, timeout_duration=30))
            except (RuntimeError, ValueError) as exc:
                self.warn(exc)
                result = None
            if result is None:
                self.warn(f"PoseBusters evaluation failed.")
                mol = self.load_molecule(molecule)
                return {
                    'all': [False]*len(mol.GetAtoms()),
                }
        
        # filter cols that start with atm_passed
        cols_filt = [col for col in result.columns if col.startswith('atm_passed')]

        mol = self.load_molecule(molecule)

        if not cols_filt:
            return {'all': [False]*mol.GetNumAtoms()}
        formatted_result = {}
        for col in cols_filt:
            formatted_result[col] = [True]*len(mol.GetAtoms())
            for idx in result[col].iloc[0]:
                formatted_result[col][idx] = False

        array = np.array([formatted_result[col] for col in cols_filt])
        formatted_result['all'] = np.all(array, axis=0).tolist()
        return formatted_result


class LocalValidity3DEvaluator(AbstractLocalEvaluator):
    ID = 'validity3d_local'

    def __init__(self, reference_ligands_path: Path, limit: int = None):
        super().__init__()
        tfd_threshold = Validity3DEvaluator.CONFIG['tfd_threshold']
        q_value_threshold = Validity3DEvaluator.CONFIG['q_value_threshold']
        steric_clash_safety_ratio = Validity3DEvaluator.CONFIG['steric_clash_safety_ratio']
        maximum_ring_plane_distance = Validity3DEvaluator.CONFIG['maximum_ring_plane_distance']
        include_torsions_in_validity3D = Validity3DEvaluator.CONFIG['include_torsions_in_validity3D']
        consider_hydrogens =Validity3DEvaluator.CONFIG['consider_hydrogens']

        name = reference_ligands_path.name.replace('.sdf', '')
        self.reference_geometry = ReferenceGeometry(
            source=SDFSource(ligands_path=str(reference_ligands_path), name=name, limit=limit),
            root=str(reference_ligands_path.parent), 
            minimum_pattern_values=Validity3DEvaluator.MINIMUM_PATTERN_VALUES,
        )

        self.validity3d = Validity3D(
            reference_geometry=self.reference_geometry,
            q_value_threshold=q_value_threshold,
            steric_clash_safety_ratio=steric_clash_safety_ratio,
            maximum_ring_plane_distance=maximum_ring_plane_distance,
            include_torsions=include_torsions_in_validity3D,
            consider_hydrogens=consider_hydrogens,
            generalize=Validity3DEvaluator.CONFIG['generalize'],
        )

    @patch('rdkit.RDLogger.EnableLog', lambda x: None)
    @patch('rdkit.RDLogger.DisableLog', lambda x: None)
    def evaluate(self, molecule, protein=None, **kwargs):
        default_return = {
            key: [True]*molecule.GetNumAtoms()
            for key in ('all', 'invalid_bonds', 'invalid_angles', 'invalid_torsions',
                        'invalid_rings', 'invalid_non_aromatic_rings')
        }
        try:
            _molecule = self.load_molecule(molecule)
            Chem.SanitizeMol(_molecule)
            if not _molecule.GetNumConformers() or any(
                not np.isfinite(conf.GetPositions()).all() for conf in _molecule.GetConformers()
            ):
                raise ValueError('Validity3D requires finite conformer coordinates')
        except Exception:
            return {key: [False]*molecule.GetNumAtoms() for key in default_return}
        
        validities, _, clashes = self.validity3d.evaluate(_molecule, analyze_torsions=True, analyze_clashes=False)
        results = default_return.copy()
        for x in validities:
            geometry = x['geometry_type']
            if not x['valid']:
                invalid_atoms = [int(atm) for atm in x['atoms']]
                for atm in invalid_atoms:
                    results['all'][atm] = False
                    results[f'invalid_{geometry}s'][atm] = False
        return results

class LocalUncertaintyEvaluator(AbstractLocalEvaluator):
    ID = 'uncertainty_local'

    def __init__(self):
        super().__init__()

    def evaluate(self, molecule, protein=None, **kwargs):
        molecule = self.load_molecule(molecule)
        # get atom props
        uncert_x = [None]*len(molecule.GetAtoms())
        uncert_x_in_atm_props = False
        for idx, atom in enumerate(molecule.GetAtoms()):
            if atom.HasProp('sigma_x'):
                uncert_x[idx] = float(atom.GetProp('sigma_x'))
                uncert_x_in_atm_props = True
        if molecule.HasProp('sigma_x') and not uncert_x_in_atm_props:
            uncert_x = molecule.GetProp('sigma_x').split(',')
            if len(uncert_x) == molecule.GetNumAtoms():
                uncert_x = [float(u) for u in uncert_x]
            else:
                uncert_x = [None]*len(molecule.GetAtoms())
        uncert_h = [None]*len(molecule.GetAtoms())
        uncert_h_in_atm_props = False
        for idx, atom in enumerate(molecule.GetAtoms()):
            if atom.HasProp('entropy_h'):
                uncert_h[idx] = float(atom.GetProp('entropy_h'))
                uncert_h_in_atm_props = True
        if molecule.HasProp('entropy_h') and not uncert_h_in_atm_props:
            uncert_h = molecule.GetProp('entropy_h').split(',')
            if len(uncert_h) == molecule.GetNumAtoms():
                uncert_h = [float(h) for h in uncert_h]
            else:
                uncert_h = [None]*len(molecule.GetAtoms())
        return {'sigma_x': uncert_x, 'entropy_h': uncert_h}


class FullLocalEvaluator(AbstractLocalEvaluator):
    def __init__(
        self,
        pb_conf: Optional[Union[Path, str]] = 'dock',
        reduce: Optional[Union[Path, str]] = None,
        residue_constraints: Optional[Union[Path, str, dict]] = None,
        interaction_list: Collection[str] = INTERACTION_LIST,
        reference_mols_validity3d: Optional[Union[Path, str]] = None,
        margin=0.75, 
        ignore={'H'},
        exclude_evaluators: Collection[str] = [],
        protein: Union[Path, str] = None,
    ):
        super().__init__(results_to_dict=False)
        all_evaluators = [
            LocalClashEvaluator(margin=margin, ignore=ignore),
            LocalChiralityEvaluator(),
            LocalPoseBustersEvaluator(pb_conf=pb_conf),
            LocalUncertaintyEvaluator(),
        ]
        if reduce is not None:
            all_evaluators.append(LocalInteractionsEvaluator(
                reduce=reduce,
                protein=protein,
                residue_constraints=residue_constraints,
                interaction_list=interaction_list,
            ))
        else:
            logging.debug(f'Evaluator [{LocalInteractionsEvaluator.ID}] is not included')
        if reference_mols_validity3d is not None:
            all_evaluators.append(LocalValidity3DEvaluator(Path(reference_mols_validity3d)))
        else:
            logging.debug(f'Evaluator [{LocalValidity3DEvaluator.ID}] is not included')

        self.evaluators = []
        for e in all_evaluators:
            if e.ID in exclude_evaluators:
                logging.debug(f'Excluded Evaluator [{e.ID}]')
            else:
                self.evaluators.append(e)

        logging.info('Will use the following evaluators:')
        for e in self.evaluators:
            logging.info(f'- [{e.ID}]')

    def evaluate(self, molecule, protein, **kwargs):
        results = {}
        for evaluator in self.evaluators:
            logging.debug(f'Evaluating {evaluator.ID}')
            results.update(evaluator(molecule, protein, **kwargs))
        return results

    def get_evaluator(self, evaluator_id):
        for evaluator in self.evaluators:
            if evaluator.ID == evaluator_id:
                return evaluator
        raise KeyError(evaluator_id)

    def update_protein(self, protein: Union[Path, str]):
        for evaluator in self.evaluators:
            if hasattr(evaluator, 'update_protein'):
                logging.debug(f'Updating protein for evaluator {evaluator.ID}')
                evaluator.update_protein(protein)

########################################################################################
################################# Collection Metrics ###################################
########################################################################################

class AbstractCollectionEvaluator:
    ID = None
    def __init__(self, reference_smiles: Collection[str]):
        self.reference_smiles = set(list(reference_smiles))

    @classmethod
    def add_id(cls, results):
        if cls.ID is not None:
            return {f'{cls.ID}.{key}': value for key, value in results.items()}
        else:
            return results

    def warn(self, message):
        logging.info(f"[{type(self).__name__}] {message}")

    @abstractmethod
    def evaluate(self, molecule: Union[str, Path, Chem.Mol], protein: Union[str, Path]) -> Dict[str, Union[int, float, str]]:
        raise NotImplementedError
        
    def __call__(self, smiles: Collection[str]):
        """
        Args:
            smiles (Collection[smiles]): input list of SMILES
        
        Returns:
            metrics (dict): dictionary of metrics
        """
        if self.ID is not None:
            logging.info(f'Running CollectionEvaluator [{self.ID}]')

        RDLogger.DisableLog('rdApp.*')
        self.check_format(smiles)
        return self.evaluate(smiles)
    
    @staticmethod
    def check_format(smiles):
        assert len(smiles) > 0, 'List of input SMILES cannot be empty'
        assert isinstance(smiles, Collection), 'Only list of SMILES supported'
        assert isinstance(smiles[0], str), 'Only list of SMILES supported'

    def warn(self, message):
        logging.warning(f"[{type(self).__name__}] {message}")
        
    
class UniquenessEvaluator(AbstractCollectionEvaluator):
    ID = 'uniqueness'
    DTYPES = {
        'uniqueness': float,
    }

    def evaluate(self, smiles: Collection[str]):
        uniqueness = len(set(smiles)) / len(smiles)
        return {'uniqueness': uniqueness}
    

class NoveltyEvaluator(AbstractCollectionEvaluator):
    ID = 'novelty'
    DTYPES = {
        'novelty': float,
    }

    def __init__(self, reference_smiles: Collection[str]):
        self.reference_smiles = set(list(reference_smiles))
        assert len(self.reference_smiles) > 0, 'List of reference SMILES cannot be empty'

    def evaluate(self, smiles: Collection[str]):
        smiles = set(smiles)
        novel = [smi for smi in smiles if smi not in self.reference_smiles]
        novelty = len(novel) / len(smiles)
        return {'novelty': novelty}
    

class FCDEvaluator(AbstractCollectionEvaluator):
    ID = 'fcd'
    DTYPES = {
        'fcd': float,
    }

    def __init__(self, reference_smiles: Collection[str]):
        self.reference_smiles = list(reference_smiles)
        assert len(self.reference_smiles) > 0, 'List of refernce SMILES cannot be empty'

    def evaluate(self, smiles: Collection[str]):
        if len(smiles) > len(self.reference_smiles):
            self.warn('Number of reference molecules should be greater than number of input molecules')
            return {'fcd': None}
        
        np.random.seed(42)
        reference_smiles = np.random.choice(self.reference_smiles, len(smiles), replace=False).tolist()
        reference_smiles_canonical = [w for w in canonical_smiles(reference_smiles) if w is not None]
        smiles_canonical = [w for w in canonical_smiles(smiles) if w is not None]
        fcd = get_fcd(reference_smiles_canonical, smiles_canonical)
        return {'fcd': fcd}


class RingDistributionEvaluator(AbstractCollectionEvaluator):
    ID = 'ring_system_distribution'
    DTYPES = {
        '*': float,
    }

    def __init__(self, reference_smiles: Collection[str], jsd_on_k_most_freq: Collection[int] = (10, 100, 1000, 10000)):
        self.ring_system_finder = RingSystemFinder()
        self.ref_ring_dict = self.compute_ring_dict(reference_smiles)
        self.jsd_on_k_most_freq = jsd_on_k_most_freq

    def compute_ring_dict(self, molecules):

        ring_system_dict = defaultdict(int)

        for mol in tqdm(molecules, desc="Computing ring systems"):

            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)

            try:
                ring_system_list = self.ring_system_finder.find_ring_systems(mol, as_mols=True)
            except (ValueError, AttributeError):
                self.warn("Error while computing ring systems; skipping molecule.")
                continue
            
            for ring in ring_system_list:
                inchi_key = Chem.MolToInchiKey(ring)
                ring_system_dict[inchi_key] += 1

        return ring_system_dict

    def precision(self, query_ring_dict):
        query_ring_systems = set(query_ring_dict.keys())
        ref_ring_systems = set(self.ref_ring_dict.keys())
        intersection = ref_ring_systems & query_ring_systems
        return len(intersection) / len(query_ring_systems)

    def recall(self, query_ring_dict):
        query_ring_systems = set(query_ring_dict.keys())
        ref_ring_systems = set(self.ref_ring_dict.keys())
        intersection = ref_ring_systems & query_ring_systems
        return len(intersection) / len(ref_ring_systems)

    def jsd(self, query_ring_dict, k_most_freq=None):

        if k_most_freq is None:
            # example on the union of all ring systems
            sample_space = set(self.ref_ring_dict.keys()) | set(query_ring_dict.keys())
        else:
            # evaluate only on the k most common rings from the reference set
            sorted_rings = [k for k, v in sorted(self.ref_ring_dict.items(), key=lambda item: item[1], reverse=True)]
            sample_space = sorted_rings[:k_most_freq]

        p = np.zeros(len(sample_space))
        q = np.zeros(len(sample_space))

        for i, inchi_key in enumerate(sample_space):
            p[i] = self.ref_ring_dict.get(inchi_key, 0)
            q[i] = query_ring_dict.get(inchi_key, 0)

        # normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        return jensenshannon(p, q)

    def evaluate(self, smiles: Collection[str]):

        query_ring_dict = self.compute_ring_dict(smiles)

        out = {
            "precision": self.precision(query_ring_dict),
            "recall": self.recall(query_ring_dict),
            "jsd": self.jsd(query_ring_dict),
        }

        out.update(
            {f"jsd_{k}_most_freq": self.jsd(query_ring_dict, k_most_freq=k) for k in self.jsd_on_k_most_freq}
        )

        return out


class FullCollectionEvaluator(AbstractCollectionEvaluator):
    COMPONENTS = [
        UniquenessEvaluator,
        NoveltyEvaluator,
        FCDEvaluator,
        RingDistributionEvaluator,
    ]
    DTYPES = {}
    for evaluator_class in COMPONENTS:
        DTYPES.update(evaluator_class.add_id(evaluator_class.DTYPES))
        DTYPES.update(evaluator_class.add_id({'time': float}))
        DTYPES['time'] = float
    
    def __init__(self, reference_smiles: Collection[str], exclude_evaluators: Collection[str] = []):
        all_evaluators = [
            UniquenessEvaluator,
            NoveltyEvaluator,
            FCDEvaluator,
            RingDistributionEvaluator,
        ]

        self.evaluators = []
        for e in all_evaluators:
            if e.ID in exclude_evaluators:
                logging.debug(f'Excluded CollectionEvaluator [{e.ID}]')
            else:
                self.evaluators.append(e(reference_smiles))

        logging.info('Will use the follwoing CollectionEvaluators:')
        for e in self.evaluators:
            logging.info(f'- [{e.ID}]')

    def evaluate(self, smiles):
        results = {}
        for evaluator in self.evaluators:
            results.update(evaluator(smiles))
        return results
