import prody
import prolif as plf
import pandas as pd
import subprocess
from io import StringIO

import numpy as np
from prolif.fingerprint import Fingerprint
from prolif.plotting.complex3d import Complex3D
from prolif.plotting.network import LigNetwork
from prolif.residue import Residue, ResidueId
from prolif.ifp import IFP
from rdkit import Chem
from tqdm import tqdm
from .hbond_utils import get_hbond_acceptors, WATER_RESNAMES


prody.confProDy(verbosity='none')

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

INTERACTION_LIST = [
    'Anionic', 'Cationic', # Salt Bridges ~400 kJ/mol
    'HBAcceptor', 'HBDonor', # Hydrogen bonds ~10 kJ/mol
    'XBAcceptor', 'XBDonor', # Halogen bonds ~5-30 kJ/mol
    'CationPi', 'PiCation', # 5-10 kJ/mol
    'PiStacking', # ~2-10 kJ/mol
    'Hydrophobic', # 1-10 kJ/mol
]

INTERACTION_ALIASES = {
    'Anionic': 'SaltBridge',
    'Cationic': 'SaltBridge',
    'HBAcceptor': 'HBAcceptor',
    'HBDonor': 'HBDonor',
    'XBAcceptor': 'HalogenBond',
    'XBDonor': 'HalogenBond',
    'CationPi': 'CationPi',
    'PiCation': 'PiCation',
    'PiStacking': 'PiStacking',
    'Hydrophobic': 'Hydrophobic',
}

INTERACTION_COLORS = {
    'SaltBridge': '#eba823',
    'HBDonor': '#3d5dfc',
    'HBAcceptor': '#3d5dfc',
    'HalogenBond': '#53f514',
    'CationPi': '#ff0000',
    'PiCation': '#ff0000',
    'PiStacking': '#e359d8',
    'Hydrophobic': '#c9c5c5',
}

INTERACTION_IMPORTANCE = ['SaltBridge', 'HydrogenBond', 'HBAcceptor', 'HBDonor', 'CationPi', 'PiCation', 'PiStacking', 'Hydrophobic']

REDUCE_EXEC = 'reduce'

def remove_residue_by_atomic_number(structure, resnum, chain_id, icode):
    exclude_selection = f'not (chain {chain_id} and resnum {resnum} and icode {icode})'
    structure = structure.select(exclude_selection)
    return structure


def load_and_protonate(protein_path, keep_water=False, verbose=False, reduce_exec=REDUCE_EXEC):
    selection_query = "protein or water" if keep_water else "protein"
    structure = prody.parsePDB(protein_path).select(selection_query)
    hydrogens = structure.select('hydrogen')
    if hydrogens is None or len(hydrogens) < len(set(structure.getResnums())) * 6:  # quick check whether protonation might be incomplete (amino acids usually have > 6 hydrogens)
        if verbose:
            print('Target structure is not protonated. Adding hydrogens...')

        reduce_cmd = f'{str(reduce_exec)} "{protein_path}"'
        reduce_result = subprocess.run(reduce_cmd, shell=True, capture_output=True, text=True)
        if reduce_result.returncode != 0:
            raise RuntimeError('Error during reduce execution:', reduce_result.stderr)

        pdb_content = reduce_result.stdout
        stream = StringIO()
        stream.write(pdb_content)
        stream.seek(0)
        structure = prody.parsePDBStream(stream).select(selection_query)
    
    return structure


def remove_altlocs(structure):
    # Select only one (largest) altloc
    altlocs = set(structure.getAltlocs())
    try:
        best_altloc = max(altlocs, key=lambda a: structure.select(f'altloc "{a}"').numAtoms())
        structure = structure.select(f'altloc "{best_altloc}"')
    except TypeError:
        # Strange thing that happens only once in the beginning sometimes...
        best_altloc = max(altlocs, key=lambda a: structure.select(f'altloc "{a}"').numAtoms())
        structure = structure.select(f'altloc "{best_altloc}"')
    return structure


def read_protein(protein_path, verbose=False, reduce_exec=REDUCE_EXEC, keep_water=False, to_exclude=[]):
    structure = load_and_protonate(protein_path, keep_water=keep_water, verbose=verbose, reduce_exec=reduce_exec)
    structure = remove_altlocs(structure)
    return prepare_protein(structure, to_exclude=to_exclude, verbose=verbose)


def prepare_protein(input_structure, to_exclude=[], verbose=False):
    structure = input_structure.copy()

    # Remove residues with bad atoms
    if verbose and len(to_exclude) > 0:
        print(f'Removing {len(to_exclude)} residues...')
    for resnum, chain_id, icode in to_exclude:
        exclude_selection = f'not (chain {chain_id} and resnum {resnum})'
        structure = structure.select(exclude_selection)

    # Write new PDB content to the stream
    stream = StringIO()
    prody.writePDBStream(stream, structure)
    stream.seek(0)
    
    # Sanitize
    rdprot = Chem.MolFromPDBBlock(stream.read(), sanitize=False, removeHs=False)
    try:
        Chem.SanitizeMol(rdprot)
        plfprot = plf.Molecule(rdprot)
        return plfprot
    
    except Chem.AtomValenceException as e:
        atom_num = int(e.args[0].replace('Explicit valence for atom # ', '').split()[0])
        info = rdprot.GetAtomWithIdx(atom_num).GetPDBResidueInfo()
        resnum = info.GetResidueNumber()
        chain_id = info.GetChainId()
        icode = f'"{info.GetInsertionCode()}"'
        
        to_exclude_next = to_exclude + [(resnum, chain_id, icode)]
        if verbose:
            print(f'[{len(to_exclude_next)}] Removing broken residue with atom={atom_num}, resnum={resnum}, chain_id={chain_id}, icode={icode}')
        return prepare_protein(input_structure, to_exclude=to_exclude_next, verbose=verbose)


def prepare_ligand_plf(mol):
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol, addCoords=True)
    ligand_plf = plf.Molecule.from_rdkit(mol)
    return ligand_plf


def sdf_reader(sdf_path, progress_bar=False):
    supp = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=False)
    for mol in tqdm(supp) if progress_bar else supp:
        yield prepare_ligand_plf(mol)


def find_hbonds_with_water_as_donor(ligand_res: Residue, water_res: Residue, distance_thresholds: tuple[float, float] = (2.1, 3.5)) -> tuple[dict]:
    """
    Waters typically aren't protonated so ProLIF cannot identify them as H-bond 
    donors. Here we estimate this interaction with a simple distance-based 
    criterion. Results are returned in a ProLIF metadata compatible format.
    """
    if not water_res.resid.name in WATER_RESNAMES:
        return ()
    
    oxygen_coord = water_res.GetConformer().GetPositions()
    if len(oxygen_coord) > 1:
        # water has explicit hydrogens or something else went wrong
        return ()
    
    # Get potential H-bond acceptors on the ligand
    potential_acceptors = np.array(get_hbond_acceptors(ligand_res))

    # Purely distance-based criterion
    lig_coord = ligand_res.GetConformer().GetPositions()[potential_acceptors]
    dists = np.sqrt(np.sum((lig_coord - oxygen_coord)**2, axis=-1))
    is_hbond = (dists >= distance_thresholds[0]) & (dists <= distance_thresholds[1])

    res = []
    for idx, dist in zip(potential_acceptors[is_hbond], dists[is_hbond]):
        res.append({
            "indices": {
                "ligand": (idx.item(),),
                "protein": (0,),  # only oxygen so the index must be 0
            },
            "parent_indices": {
                "ligand": (ligand_res.GetAtomWithIdx(idx.item()).GetUnsignedProp("mapindex"),),
                "protein": (ligand_res.GetAtomWithIdx(0).GetUnsignedProp("mapindex"),),
            },
            "distance": dist,
            "DHA_angle": None,
        })
    
    return tuple(res)


def profile_detailed(
        ligand_plf, protein_plf, interaction_list=INTERACTION_LIST, ligand_name='ligand', protein_name='protein', n_jobs=1,
    ):

    fp = Fingerprint(interactions=interaction_list)
    fp.run_from_iterable(lig_iterable=[ligand_plf], prot_mol=protein_plf, progress=False, n_jobs=n_jobs)

    profile = []

    for ligand_residue in ligand_plf.residues:
        for protein_residue in protein_plf.residues:
            metadata = fp.metadata(ligand_plf[ligand_residue], protein_plf[protein_residue])

            # Add water as H-bond donor
            if protein_plf[protein_residue].resid.name in WATER_RESNAMES and not "HBAcceptor" in metadata:
                hbonds_with_water_as_donor = find_hbonds_with_water_as_donor(ligand_plf[ligand_residue], protein_plf[protein_residue])
                if len(hbonds_with_water_as_donor) > 0:
                    metadata["HBAcceptor"] = hbonds_with_water_as_donor
            
            for int_name, int_metadata in metadata.items():
                for int_instance in int_metadata:
                    profile.append({
                        'ligand': ligand_name,
                        'protein': protein_name,
                        'ligand_residue': str(ligand_residue),
                        'protein_residue': str(protein_residue),
                        'interaction': int_name,
                        'alias': INTERACTION_ALIASES[int_name],
                        'ligand_atoms': ','.join(map(str, int_instance['indices']['ligand'])),
                        'protein_atoms': ','.join(map(str, int_instance['indices']['protein'])),
                        'ligand_orig_atoms': ','.join(map(str, int_instance['parent_indices']['ligand'])),
                        'protein_orig_atoms': ','.join(map(str, int_instance['parent_indices']['protein'])),
                        'distance': int_instance['distance'],
                        'plane_angle': int_instance.get('plane_angle', None),
                        'normal_to_centroid_angle': int_instance.get('normal_to_centroid_angle', None),
                        'intersect_distance': int_instance.get('intersect_distance', None),
                        'intersect_radius': int_instance.get('intersect_radius', None),
                        'pi_ring': int_instance.get('pi_ring', None),
                    })

    return pd.DataFrame(profile)

def filter_profile(profile, filter_dict=None, interaction_list=None, ignore_chains=False):
    if interaction_list is None:
        interaction_list = INTERACTION_LIST
    # filter by interaction type
    profile = profile[profile['interaction'].isin(interaction_list)]

    # filter by residue type
    profile['filtered'] = True
    if filter_dict is not None:
        filter_dict_parsed = dict()
        for key, value in filter_dict.items():
            if value == '*':
                filter_dict_parsed[key] = interaction_list
            else:
                filter_dict_parsed[key] = [v for v in value if v in interaction_list]
            if ignore_chains:
                val = filter_dict_parsed.pop(key)
                new_key = key.split('.')[0]
                filter_dict_parsed[new_key] = val
        if ignore_chains:
            profile['protein_residue'] = profile['protein_residue'].apply(lambda x: x.split('.')[0])
        profile['filtered'] = False
        for i,row in profile.iterrows():
            for key, value in filter_dict_parsed.items():
                if row['interaction'] in value and row['protein_residue'] == key:
                    profile.at[i, 'filtered'] = True
                    break
        profile = profile[profile['filtered'] == True]
    profile = profile.drop(columns=['filtered'])
    return profile

def map_orig_atoms_to_new(atoms, mol):
    orig2new = dict()
    for atom in mol.GetAtoms():
        orig2new[atom.GetUnsignedProp("mapindex")] = atom.GetIdx()
    
    atoms = list(map(int, atoms.split(',')))
    new_atoms = ','.join(map(str, [orig2new[atom] for atom in atoms]))
    return new_atoms


def visualize(profile, ligand_plf, protein_plf, ligand_mol=None, plot_3d=True):
    metadata = dict()

    for _, row in profile.iterrows():
        if 'ligand_atoms' not in row:
            row['ligand_atoms'] = map_orig_atoms_to_new(row['ligand_orig_atoms'], ligand_plf)
        if 'protein_atoms' not in row:
            row['protein_atoms'] = map_orig_atoms_to_new(row['protein_orig_atoms'], protein_plf[row['residue']])

        namenum, chain = row['protein_residue'].split('.')
        name = namenum[:3]
        num = int(namenum[3:])
        protres = ResidueId(name=name, number=num, chain=chain)
        key = (ResidueId(name='UNL', number=1, chain=None), protres)

        metadata.setdefault(key, dict())
        interaction = {
            'indices': {
                'ligand': tuple(map(int, row['ligand_atoms'].split(','))),
                'protein': tuple(map(int, row['protein_atoms'].split(','))),
            },
            'parent_indices': {
                'ligand': tuple(map(int, row['ligand_atoms'].split(','))),
                'protein': tuple(map(int, row['protein_atoms'].split(','))),
            },
            'distance': row['distance'],
        }
        metadata[key].setdefault(row['alias'], list()).append(interaction)
    
    ifp = IFP(metadata)
    fp = Fingerprint(interactions=INTERACTION_LIST, vicinity_cutoff=8.0)
    fp.ifp = {0: ifp}

    if plot_3d:
        Complex3D.COLORS.update(INTERACTION_COLORS)
        v = fp.plot_3d(ligand_mol=ligand_plf, protein_mol=protein_plf, frame=0)
    else:
        v = LigNetwork.from_fingerprint(fp, ligand_mol=ligand_plf)
    return v