import random
import socket
import warnings
import logging
import shutil
from copy import deepcopy
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Union, Iterable

import lightning.pytorch as pl
import networkx as nx
import numpy as np
import torch
from networkx.algorithms import isomorphism
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import KekulizeException, AtomKekulizeException

from lddm.scatter import scatter_add, scatter_mean


class Queue():
    def __init__(self, max_len=50):
        self.items = []
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def add(self, item):
        self.items.insert(0, item)
        if len(self) > self.max_len:
            self.items.pop()

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


#####


def sum_except_batch(x, indices):
    if len(x.size()) < 2:
        x = x.unsqueeze(-1)
    return scatter_add(x.sum(list(range(1, len(x.size())))), indices, dim=0)


def remove_mean_batch(x, batch_mask, dim_size=None):
    # Compute center of mass per sample
    mean = scatter_mean(x, batch_mask, dim=0, dim_size=dim_size)
    x = x - mean[batch_mask]
    return x, mean


def assert_mean_zero(x, batch_mask, thresh=1e-2, eps=1e-10):
    largest_value = x.abs().max().item()
    error = scatter_add(x, batch_mask, dim=0).abs().max().item()
    rel_error = error / (largest_value + eps)
    assert rel_error < thresh, f'Mean is not zero, relative_error {rel_error}'


def create_scatter_output_tensor(like, dim_size, dim=0, fill_value=0):
    _dims = [dim_size] + [x for i, x in enumerate(like.size()) if i != dim]
    return torch.full(_dims, fill_value=fill_value, dtype=like.dtype, device=like.device)


def bvm(v, m):
    """
    Batched vector-matrix product of the form out = v @ m
    :param v: (b, n_in)
    :param m: (b, n_in, n_out)
    :return: (b, n_out)
    """
    # return (v.unsqueeze(1) @ m).squeeze()
    return torch.bmm(v.unsqueeze(1), m).squeeze(1)


def get_grad_norm(
        parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
        norm_type: float = 2.0) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.)

    device = parameters[0].grad.device

    total_norm = torch.norm(torch.stack(
        [torch.norm(p.grad.detach(), norm_type).to(device) for p in
         parameters]), norm_type)

    return total_norm


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, 'w') as f:
        f.write(out)


def atomprops_to_molprops(mol):
    """
    Collects all atom-level properties and writes them as comma-separated lists on the molecule level.
    """
    if mol.GetNumAtoms() == 0:
        return mol

    all_props = set()
    for atom in mol.GetAtoms():
        all_props.update(atom.GetPropsAsDict().keys())

    for prop in all_props:
        values = []
        for atom in mol.GetAtoms():
            props = atom.GetPropsAsDict()
            value = props.get(prop, "")
            values.append(str(value))
        mol.SetProp(prop, ",".join(values))

    return mol


def molprops_to_atomprops(mol):
    """
    Reverses `atomprops_to_molprops`: for each molecule-level property that
    looks like a comma-separated list, if its length matches the number of atoms,
    assign each value back to the corresponding atom property.
    """
    num_atoms = mol.GetNumAtoms()
    for key, val in mol.GetPropsAsDict().items():
        parts = val.split(',')
        if len(parts) == num_atoms:
            for atom, v in zip(mol.GetAtoms(), parts):
                if v != "":
                    atom.SetProp(key, v)
    return mol


def add_mol_to_sdwriter(w, mol, catch_errors=True, connected=False, add_atomprops_as_molprops=False, name='ligand'):
    try:
        if mol is None:
            raise ValueError("Mol is None.")
        mol_comp = get_largest_connected_component(mol, name=name) if connected else mol
        if add_atomprops_as_molprops:
            mol_comp = atomprops_to_molprops(mol_comp)
        w.write(mol_comp)

    except (RuntimeError, ValueError) as e:
        if not catch_errors:
            raise e

        if isinstance(e, (KekulizeException, AtomKekulizeException)):
            mol.SetProp('_Name', name)
            w.SetKekulize(False)
            mol_comp = get_largest_connected_component(mol, name=name) if connected else mol
            if add_atomprops_as_molprops:
                mol_comp = atomprops_to_molprops(mol_comp)
            w.write(mol_comp)
            w.SetKekulize(True)
            warnings.warn(f"Mol saved without kekulization.")
        else:
            # write empty mol to preserve the original order
            empty_mol = Chem.Mol()
            empty_mol.SetProp('_Name', name)
            w.write(empty_mol)
            warnings.warn(f"Erroneous mol replaced with empty dummy.")


def write_sdf_file(sdf_path, molecules, catch_errors=True, connected=False, add_atomprops_as_molprops=False):
    with Chem.SDWriter(str(sdf_path)) as w:
        for mol in molecules:
            add_mol_to_sdwriter(w, mol, catch_errors, connected, add_atomprops_as_molprops)


def write_smpls_to_dir(outprefix, mols, pocket_p):
    outprefix = Path(outprefix)
    out_sdf_path = f'{outprefix}_samples.sdf'
    out_pocket_path = f'{outprefix}_pocket.pdb'

    w = Chem.SDWriter(out_sdf_path)
    for idx, mol in enumerate(mols):
        if mol is None:
            continue
        mol.SetProp('ligand_idx', str(idx))
        add_mol_to_sdwriter(w, mol, catch_errors=True, connected=True)
    w.close()
    shutil.copy(pocket_p, out_pocket_path)  # same pocket for all ligands


def fix_aromaticity(mol):
    if mol is None:
        return None
    try:
        newmol = Chem.RemoveHs(mol, sanitize=False)
        for atm in newmol.GetAtoms():
            if not atm.IsInRing() and atm.GetIsAromatic():
                atm.SetIsAromatic(False)
                for bond in atm.GetBonds():
                    bond.SetIsAromatic(False)
        Chem.SetAromaticity(newmol)
        Chem.SanitizeMol(newmol)
    except Exception as e:
        return None
    return newmol


def get_largest_connected_component(mol, name=None, strict=False):
    if mol is None:
        if strict:
            return None
        else:
            return Chem.Mol()
    try:
        frags = Chem.GetMolFrags(mol, asMols=True)
        newmol = max(frags, key=lambda m: m.GetNumAtoms())
    except Exception as e:
        warnings.warn(f"Failed to get largest connected component: {e}")
        if strict:
            return None
        newmol = mol
    if name is not None:
        newmol.SetProp('_Name', name)
    return newmol


def write_chain(filename, rdmol_chain):
    with open(filename, 'w') as f:
        f.write("".join([Chem.MolToXYZBlock(m) for m in rdmol_chain]))


def combine_sdfs(sdf_list, out_file):
    all_content = []
    for sdf in sdf_list:
        with open(sdf, 'r') as f:
            all_content.append(f.read())
    combined_str = '$$$$\n'.join(all_content)
    with open(out_file, 'w') as f:
        f.write(combined_str)


def batch_to_list_legacy(data, batch_mask, keep_order=True):
    if keep_order:  # preserve order of elements within each sample
        data_list = [data[batch_mask == i]
                     for i in torch.unique(batch_mask, sorted=True)]
        return data_list

    # make sure batch_mask is increasing
    idx = torch.argsort(batch_mask)
    batch_mask = batch_mask[idx]
    data = data[idx]

    chunk_sizes = torch.unique(batch_mask, return_counts=True)[1].tolist()
    return torch.split(data, chunk_sizes)


def zero_elem_tensor_like(x: torch.Tensor):
    _dims = [0, *x.size()[1:]]
    return torch.empty(_dims, dtype=x.dtype, layout=x.layout, device=x.device)


def batch_to_list(data, batch_mask, batch_size=None):
    existing_batch_inds = torch.unique(batch_mask, sorted=True)
    batch_size = batch_size or len(existing_batch_inds)

    assert (batch_mask >= 0).all()
    assert (batch_mask < batch_size).all()

    data_list = [
        data[batch_mask == i] if i in existing_batch_inds else zero_elem_tensor_like(data) 
        for i in range(batch_size)
    ]
    return data_list


def batch_to_list_for_indices(indices, batch_mask, offsets=None, batch_size=None):
    # (2, n) -> (n, 2)
    split = batch_to_list(indices.T, batch_mask, batch_size=batch_size)

    # rebase indices at zero & (n, 2) -> (2, n)
    if offsets is None:
        warnings.warn("Trying to infer index offset from smallest element in "
                      "batch. This might be wrong.")
        split = [x.T - x.min() for x in split]
    else:
        # Typically 'offsets' would be accumulate(sizes[:-1], initial=0)
        assert len(offsets) == len(split) or indices.numel() == 0
        split = [x.T - offset for x, offset in zip(split, offsets)]

    return split


def num_nodes_to_batch_mask(n_samples, num_nodes, device):
    assert isinstance(num_nodes, int) or len(num_nodes) == n_samples

    if isinstance(num_nodes, torch.Tensor):
        num_nodes = num_nodes.to(device)

    sample_inds = torch.arange(n_samples, device=device)

    return torch.repeat_interleave(sample_inds, num_nodes)


def rdmol_to_nxgraph(rdmol):
    graph = nx.Graph()
    for atom in rdmol.GetAtoms():
        # Add the atoms as nodes
        graph.add_node(atom.GetIdx(), atom_type=atom.GetAtomicNum())

    # Add the bonds as edges
    for bond in rdmol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    return graph


def calc_rmsd(mol_a, mol_b):
    """ Calculate RMSD of two molecules with unknown atom correspondence. """
    graph_a = rdmol_to_nxgraph(mol_a)
    graph_b = rdmol_to_nxgraph(mol_b)

    gm = isomorphism.GraphMatcher(
        graph_a, graph_b,
        node_match=lambda na, nb: na['atom_type'] == nb['atom_type'])

    isomorphisms = list(gm.isomorphisms_iter())
    if len(isomorphisms) < 1:
        return None

    all_rmsds = []
    for mapping in isomorphisms:
        atom_types_a = [atom.GetAtomicNum() for atom in mol_a.GetAtoms()]
        atom_types_b = [mol_b.GetAtomWithIdx(mapping[i]).GetAtomicNum()
                        for i in range(mol_b.GetNumAtoms())]
        assert atom_types_a == atom_types_b

        conf_a = mol_a.GetConformer()
        coords_a = np.array([conf_a.GetAtomPosition(i)
                             for i in range(mol_a.GetNumAtoms())])
        conf_b = mol_b.GetConformer()
        coords_b = np.array([conf_b.GetAtomPosition(mapping[i])
                             for i in range(mol_b.GetNumAtoms())])

        diff = coords_a - coords_b
        rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        all_rmsds.append(rmsd)

    if len(isomorphisms) > 1:
        print("More than one isomorphism found. Returning minimum RMSD.")

    return min(all_rmsds)


def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger().setLevel(level)

def disable_rdkit_logging():
    # RDLogger.DisableLog('rdApp.*')
    RDLogger.DisableLog('rdApp.info')
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')

class quiet_rdkit:
    """Context manager for temporarily disabling rdkit logging."""
    def __init__(self):
        self.previous_status = None
        self.quiet_status = {
            "rdApp.error": False,
            "rdApp.warning": False,
            "rdApp.debug": False,
            "rdApp.info": False,
        }

    def _get_log_status(self):
        """Get the current log status of RDKit logs."""
        log_status = rdBase.LogStatus()
        log_status = {s.split(":")[0]: s.split(":")[1] for s in log_status.split("\n")}
        log_status = {k: True if v == "enabled" else False for k, v in log_status.items()}
        return log_status

    def _apply_log_status(self, log_status):
        """Apply an RDKit log status."""
        for k, enable in log_status.items():
            if enable:
                rdBase.EnableLog(k)
            else:
                rdBase.DisableLog(k)

    def __enter__(self):
        self.previous_status = self._get_log_status()
        self._apply_log_status(self.quiet_status)

    def __exit__(self, *args, **kwargs):
        self._apply_log_status(self.previous_status)


def set_default(namespace, name, value):
    if not hasattr(namespace, name):
        setattr(namespace, name, value)


def merge_args_and_yaml(args, config):
    config = dict(config or {})
    config.update({key: value for key, value in vars(args).items() if value is not None})
    return dict_to_namespace(config)


def dict_to_namespace(input_dict):
    """ Recursively convert a nested dictionary into a Namespace object. """
    if isinstance(input_dict, dict):
        output_namespace = Namespace()
        output = output_namespace.__dict__
        for key, value in input_dict.items():
            output[key] = dict_to_namespace(value)
        return output_namespace

    elif isinstance(input_dict, Namespace):
        # recurse as Namespace might contain dictionaries
        return dict_to_namespace(input_dict.__dict__)

    else:
        return input_dict


def namespace_to_dict(x):
    """ Recursively convert a nested Namespace object into a dictionary. """

    if isinstance(x, Namespace):
        x = vars(x)
    
    # recurse
    if isinstance(x, dict):
        return {key: namespace_to_dict(value) for key, value in x.items()}
    
    elif isinstance(x, list):
        return [namespace_to_dict(item) for item in x]
    
    else:
        return x


def trace_handler(prof: torch.profiler.profile):
    TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    # Construct the trace file.
    prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")


class DumpProfileCallback(pl.Callback):
    def __init__(self, profile_dir: Path):
        self.profile_dir = profile_dir

    def on_train_batch_end(self, pl_module, *args, **kwargs):
        if pl_module.global_step == 5:
            torch.cuda.memory._dump_snapshot(self.profile_dir / 'cuda_dump.pickle')
            torch.cuda.memory._record_memory_history(enabled=None)


def batch_to_ptr(batch):
    # _, counts = torch.unique(batch, return_counts=True)
    # ptr = torch.cat([torch.tensor([0], device=batch.device), counts.cumsum(dim=0)])
    assert torch.all(batch[1:] - batch[:-1] >= 0), "batch mask not monotonically increasing"
    ptr = torch.where(batch[1:] - batch[:-1] > 0)[0] + 1
    ptr = torch.cat([torch.tensor([0], device=batch.device), ptr, torch.tensor([len(batch)], device=batch.device)])  # add first and one after last
    return ptr


def optimal_remapping(z0_x, x, batch_mask):
    ptr = batch_to_ptr(batch_mask)
    z0_x_permuted = []
    for start, end in zip(ptr[:-1], ptr[1:]):
        z0_x_permuted.append(optimal_remapping_single_batch(z0_x[start:end], x[start:end]))

    return torch.cat(z0_x_permuted, dim=0)


def optimal_remapping_single_batch(z0_x, x):
    import ot as pot
    cost = torch.cdist(z0_x, x, p=2)
    a, b = pot.unif(z0_x.shape[0]), pot.unif(x.shape[0])
    p = pot.emd(a, b, cost.detach().cpu().numpy())
    return z0_x[p.argmax(0)]


def is_one_hot(x, dim=-1):
    is_binary = (x == 0) | (x == 1)
    row_sums = x.sum(dim=dim)
    return is_binary.all() and torch.all(row_sums == 1)


def argsort_within_batch(x, indices, descending=False, dim=-1):
    """
    adapted from 
    https://github.com/pyg-team/pytorch_geometric/blob/ecf40202a4f5aaeb264b7183cc5038fdf14a45ad/torch_geometric/nn/aggr/quantile.py#L88-L93
    """
    # Two sorts: the first one on the value, the second (stable) on the indices:
    x_perm = torch.argsort(x, descending=descending, dim=dim)
    indices = indices.take_along_dim(x_perm, dim=dim)
    index_perm = torch.argsort(indices, dim=dim, stable=True)  # preserve order of equivalent elements
    return x_perm[index_perm]


def combine_args(base_args, override_args):
    assert not isinstance(base_args, dict)
    assert not isinstance(override_args, dict)

    arg_dict = base_args.__dict__
    for key, value in override_args.__dict__.items():
        if key not in arg_dict or arg_dict[key] is None:  # parameter not provided previously
            print(f"Add parameter {key}: {value}")
            arg_dict[key] = value
        elif isinstance(value, Namespace):
            arg_dict[key] = combine_args(arg_dict[key], value)
        else:
            logging.debug(f"Replace parameter {key}: {arg_dict[key]} -> {value}")
            arg_dict[key] = value
    return base_args

class CategoricalDistribution:
    EPS = 1e-10

    def __init__(self, histogram_dict, mapping):
        histogram = np.zeros(len(mapping))
        for k, v in histogram_dict.items():
            histogram[mapping[k]] = v

        # Normalize histogram
        self.p = histogram / histogram.sum()
        self.mapping = deepcopy(mapping)

    @classmethod
    def from_file(cls, histogram_file, mapping):
        histogram_dict = np.load(histogram_file, allow_pickle=True).item()
        return cls(histogram_dict, mapping)

    def kl_divergence(self, other_sample):
        sample_histogram = np.zeros(len(self.mapping))
        for x in other_sample:
            # sample_histogram[self.mapping[x]] += 1
            sample_histogram[x] += 1

        # Normalize
        q = sample_histogram / sample_histogram.sum()

        return -np.sum(self.p * np.log(q / (self.p + self.EPS) + self.EPS))
