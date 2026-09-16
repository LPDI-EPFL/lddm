import random
import warnings
from typing import List, Union
from pathlib import Path
import yaml
import collections

import torch
from torch.utils.data import Dataset, IterableDataset
import torch.distributed as dist

from lddm.constants import INT_TYPE
from lddm.data.data_utils import TensorDict, collate_entity, edge_mask_by_node_mask
from lddm.config.data import DatasetConfig


class ProcessedDataset(Dataset):
    def __init__(self, data_path, stage, transforms=None, catch_errors=False):
        pt_path = Path(data_path, f'{stage}.pt')
        self.transforms = transforms
        self.catch_errors = catch_errors
        self.pt_path = pt_path
        self.data = torch.load(pt_path)
        config = Path(data_path, "config.yaml")
        self.config = DatasetConfig.from_yaml(config) if config.exists() else None
        
    def __len__(self):
        return len(self.data['ligands']['name'])

    def __getitem__(self, idx):
        data = {
            'ligand': {key: val[idx] for key, val in self.data['ligands'].items()},
            'pocket': {key: val[idx] for key, val in self.data['pockets'].items()}
        }
        data = self.set_additional_attributes(data)

        try:
            data = self.apply_transforms(data, self.transforms)
        except (RuntimeError, ValueError) as e:
            if self.catch_errors:
                # Replace bad item with a random one
                warnings.warn(f"{type(e).__name__}('{e}') in data transform. Returning random item instead")
                rand_idx = random.randint(0, len(self) - 1)
                return self[rand_idx]
            else:
                raise e
        return data

    @staticmethod
    def apply_transforms(data, transforms):
        if transforms is not None:
            for transform in transforms:
                data = transform(data)
        return data
    
    @staticmethod
    def set_additional_ligand_attributes(ligand):
        # Add number of nodes for convenience
        ligand['size'] = len(ligand['x'])
        ligand['n_bonds'] = len(ligand['bond_one_hot'])

        # If no fragment detection was run we mark it as one entire fragment
        if 'fragments' not in ligand:
            ligand['fragments'] = torch.zeros(ligand['size'], dtype=INT_TYPE)

        # By default masked modeling is disabled
        if 'known_x' not in ligand:
            ligand['known_x'] = torch.zeros_like(ligand['fragments']).bool()
        if 'known_h' not in ligand:
            ligand['known_h'] = torch.zeros_like(ligand['fragments']).bool()
        if 'known_e' not in ligand:
            ligand['known_e'] = edge_mask_by_node_mask(node_mask=ligand['known_h'], edges=ligand['bonds'])

        # Some datasets may not have the following attributes
        if 'smiles' not in ligand: 
            ligand['smiles'] = ''
        if 'affinity' not in ligand: 
            ligand['affinity'] = 0.0
        if 'name' not in ligand:
            ligand['name'] = ''

        return ligand

    @staticmethod
    def set_additional_pocket_attributes(pocket):
        # Add number of nodes for convenience
        pocket['size'] = len(pocket['x'])
        pocket['n_bonds'] = len(pocket['bond_one_hot'])
        # Some datasets may not have the following attributes
        if 'name' not in pocket: 
            pocket['name'] = ''
        return pocket

    @staticmethod
    def set_additional_attributes(data):
        # Add number of nodes for convenience
        data['ligand'] = ProcessedDataset.set_additional_ligand_attributes(data['ligand'])
        data['pocket'] = ProcessedDataset.set_additional_pocket_attributes(data['pocket'])
        return data

    @staticmethod
    def collate_fn(batch_pairs, ligand_transform=None):
        out = {}
        for entity in ['ligand', 'pocket']:
            batch = [x[entity] for x in batch_pairs]

            if entity == 'ligand' and ligand_transform is not None:
                max_size = max(x['size'].item() for x in batch)
                batch = [ligand_transform(x, max_size=max_size) for x in batch]

            out[entity] = TensorDict(**collate_entity(batch))

        return out


class ClusteredDataset(ProcessedDataset):
    def __init__(self, data_path, stage, transforms=None, catch_errors=False, deterministic=False):
        super().__init__(data_path, stage, transforms, catch_errors)
        self.clusters = list(self.data['clusters'].values())
        self.deterministic = deterministic
        print(f'Dataset {data_path} ({stage}) has {len(self.clusters)} clusters, deterministic={self.deterministic}')

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, cidx):
        cluster_inds = self.clusters[cidx]
        idx = cluster_inds[0] if self.deterministic else random.choice(cluster_inds)
        return super().__getitem__(idx)
    
    def iterate_over_all_data_points(self):
        for i in range(len(self.data['ligands']['name'])):
            yield super().__getitem__(i)
    

class CycleDataset:
    def __init__(self, base_dataset: IterableDataset):
        self.dataset = base_dataset
        self.iterator = iter(base_dataset)
        self.config = self.dataset.config

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataset)
            return next(self.iterator)
    

class MixedDataset(IterableDataset):
    def __init__(self, datasets: List[Union[Dataset, IterableDataset]], 
                 samples_per_epoch: int, weights: List[float] | None = None):
        
        weights = weights or [1] * len(datasets)
        assert len(weights) == len(datasets)

        self.datasets = [CycleDataset(d) if isinstance(d, IterableDataset) else d for d in datasets]
        self.weights = weights
        self.samples_per_epoch = samples_per_epoch

        self.config = self.datasets[0].config
        if len(self.datasets) > 1 and self.config is not None:
            # make sure all datasets have been featurized in the same way
            for d in self.datasets:
                assert d.config.featurization == self.config.featurization, "Cannot mix datasets with different featurization settings"

    def __len__(self):
        return self.samples_per_epoch

    def get_sample(self, dataset):
        if isinstance(dataset, CycleDataset):
            # wrapper around iterable-style dataset
            return dataset.next()

        else:
            # map-style dataset
            idx = random.randrange(len(dataset))
            return dataset[idx]

    def sample_generator(self):
        for i in range(self.samples_per_epoch):
            data_source = random.choices(self.datasets, weights=self.weights)[0]
            yield self.get_sample(data_source)

    def __iter__(self):
        return iter(self.sample_generator())
    

class DynamicBatchIterableDataset(IterableDataset):
    """
    Wrapper for IterableDataset that enables dynamic batching.

    Example::

        dataset = DynamicBatchIterableDataset(
            dataset, max_tokens_per_batch=4096,
        )
        loader = DataLoader(dataset, batch_size=None, collate_fn=collate)
    """
    def __init__(self, base_dataset: IterableDataset, max_tokens_per_batch: int):
        self.base_dataset = base_dataset
        self.max_tokens_per_batch = max_tokens_per_batch

    def _get_num_tokens(self, item: dict) -> int:
        """Retrieve `num_tokens` from a single data item."""
        return item["ligand"]["size"] + item["pocket"]["size"]

    def __iter__(self):
        """
        Ensures batches have the same size across devices to avoid synchronization issues.
        Caveat: the dynamic batch size estimation becomes less efficient and effective with more devices.

        Currently needs to run with num_workers == 0 so that the reduce happens in the main process.
        NOTE: worker-sharding for num_workers > 0 is currently broken anyway when we use iterable datasets.
        """
        is_ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        device = torch.device(f"cuda:{torch.cuda.current_device()}") if is_ddp else None

        pending = collections.deque()  # carry-over samples
        base_iter = iter(self.base_dataset)
        exhausted = False

        while True:
            batch, batch_tokens = [], 0

            # 1) drain carry-over first
            while pending:
                n = self._get_num_tokens(pending[0])
                if batch and batch_tokens + n > self.max_tokens_per_batch:
                    break
                batch.append(pending.popleft())
                batch_tokens += n

            # 2) then pull fresh samples up to the token budget
            while not exhausted and batch_tokens < self.max_tokens_per_batch:
                try:
                    sample = next(base_iter)
                except StopIteration:
                    exhausted = True
                    break
                n = self._get_num_tokens(sample)
                if batch and batch_tokens + n > self.max_tokens_per_batch:
                    pending.appendleft(sample)  # save for next batch
                    break
                batch.append(sample)
                batch_tokens += n

            # 3) align batch size across ranks
            if is_ddp:
                s = torch.tensor([len(batch)], device=device)
                dist.all_reduce(s, op=dist.ReduceOp.MIN)
                min_size = int(s.item())
            else:
                min_size = len(batch)

            if min_size == 0:
                return                          # at least one rank has nothing left

            # 4) push leftovers back at the head of the queue, in order
            for x in reversed(batch[min_size:]):
                pending.appendleft(x)

            yield batch[:min_size]
