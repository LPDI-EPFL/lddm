import random
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

from lddm.constants import INT_TYPE
from lddm.data.data_utils import edge_mask_by_node_mask, Ligand, Residues, extract_substructure
from lddm.scatter import scatter_mean
from lddm.utils import batch_to_ptr


class AppendVirtualNodesInCoM:
    def __init__(self, atom_encoder, bond_encoder, add_min=0, add_max=10):
        self.atom_encoder = atom_encoder
        self.bond_encoder = bond_encoder
        self.vidx = atom_encoder['NOATOM']
        self.bidx = bond_encoder['NOBOND']
        self.add_min = add_min
        self.add_max = add_max

    def __call__(self, ligand):
        device = ligand['x'].device
        n_virt = random.randint(self.add_min, self.add_max)

        # all virtual coordinates in the CoM
        virt_coords = ligand['x'].mean(0, keepdim=True).repeat(n_virt, 1)

        # insert virtual atom column
        virt_one_hot = F.one_hot(torch.ones(n_virt, dtype=torch.int64, device=device) * self.vidx, num_classes=len(self.atom_encoder))
        virt_mask = torch.cat([torch.zeros(ligand['size'], dtype=bool), torch.ones(n_virt, dtype=bool)]).to(device)

        ligand['x'] = torch.cat([ligand['x'], virt_coords])
        ligand['one_hot'] = torch.cat(([ligand['one_hot'], virt_one_hot]))
        ligand['virtual_mask'] = virt_mask
        ligand['size'] = len(ligand['x'])
        ligand['mask'] = torch.zeros(ligand['size'], dtype=INT_TYPE, device=device)

        # Bonds
        new_bonds = torch.triu_indices(ligand['size'], ligand['size'], offset=1, device=device)
        bond_types = torch.ones(ligand['size'], ligand['size'], dtype=INT_TYPE, device=device) * self.bidx
        row, col = ligand['bonds']
        bond_types[row, col] = ligand['bond_one_hot'].argmax(dim=1)
        new_row, new_col = new_bonds
        bond_types = bond_types[new_row, new_col]

        ligand['bonds'] = new_bonds
        ligand['bond_one_hot'] = F.one_hot(bond_types, num_classes=len(self.bond_encoder)).to(ligand['bond_one_hot'].dtype)
        ligand['n_bonds'] = len(ligand['bond_one_hot'])
        ligand['bond_mask'] = torch.zeros(ligand['n_bonds'], dtype=INT_TYPE, device=device)

        # Extending fragments (with virtual type)
        fragments = ligand['fragments']
        max_fragment_id = max(fragments) if fragments.numel() > 0 else -1
        virt_fragment_id = max_fragment_id + 1
        virt_fragments = virt_fragment_id * torch.ones(n_virt, dtype=INT_TYPE, device=device)
        ligand['fragments'] = torch.cat([fragments, virt_fragments])

        # Extending fragment_mask and fragment_edge_mask marking virtual nodes and edges to virtual as "unknown"
        new_known_x = torch.cat([ligand['known_x'], torch.zeros(n_virt, device=device).bool()])
        new_known_h = torch.cat([ligand['known_h'], torch.zeros(n_virt, device=device).bool()])
        new_known_e = edge_mask_by_node_mask(new_known_h, new_bonds)

        ligand['known_x'] = new_known_x
        ligand['known_h'] = new_known_h
        ligand['known_e'] = new_known_e

        return ligand


class Mask:
    REGIMES = ['design', 'docking', 'context']

    def __init__(self, mask_shares):
        self.intervals = {}
        self.enabled_regimes = []
        left = 0
        total = sum(v for k, v in mask_shares.items())
        for k, v in mask_shares.items():
            assert k in self.REGIMES
            if v > 0:
                self.enabled_regimes.append(k)
            
            step = v / total
            self.intervals[k] = (left, left + step)
            left += step
        
        print(f'Mask intervals: {self.intervals}')
        print(f'Enabled regimes: {self.enabled_regimes}')

    def __call__(self, ligand):
        assert torch.all(~ligand['known_h'])
        assert torch.all(~ligand['known_e'])
        
        # Special case for covalent docking - when dataset already has info about what atom should be fixed
        if torch.any(ligand['known_x']):
            assert len(self.enabled_regimes) == 1 and self.enabled_regimes[0] == 'docking'
            ligand['known_h'] = torch.ones_like(ligand['fragments']).bool()
            ligand['known_e'] = edge_mask_by_node_mask(ligand['known_h'], ligand['bonds'])
            return ligand

        fragments = ligand['fragments']
        fragment_num = len(fragments.unique())
        n = len(ligand['x'])
        device = ligand['x'].device

        regime2mask = {
            'design': torch.zeros(n, device=device).bool(),
            'docking': torch.zeros(n, device=device).bool(),
            'context': torch.zeros(n, device=device).bool(),
        }
        for fragment_id, x in enumerate(torch.rand(fragment_num)):
            for regime, mask in regime2mask.items():
                left, right = self.intervals.get(regime, (-1, -1))
                if left <= x < right:
                    mask |= (fragments == fragment_id)
                    break
        
        if torch.all(regime2mask['context']):
            regime2mask['context'] = torch.zeros(n, device=device).bool()
            if 'design' in self.enabled_regimes:
                regime2mask['design'] =  torch.ones(n, device=device).bool()
            elif 'docking' in self.enabled_regimes:
                regime2mask['docking'] =  torch.ones(n, device=device).bool()
            else:
                raise NotImplementedError(self.enabled_regimes)

        ligand['known_x'] = regime2mask['context']
        ligand['known_h'] = regime2mask['context'] | regime2mask['docking']

        if 'virtual_mask' in ligand:
            ligand['known_x'] &= ~ligand['virtual_mask']
            ligand['known_h'] &= ~ligand['virtual_mask']

        ligand['known_e'] = edge_mask_by_node_mask(ligand['known_h'], ligand['bonds'])
        return ligand
    

class DataTransform(ABC):
    """Base class for transformations that are applied to protein-ligand pairs."""
    @abstractmethod
    def __call__(self, data):
        """
        Takes a dictionary with 'ligand' and 'pocket' keys and applies 
        transformations to the values before returning it.
        """
        pass


class AddVirtualNodesToLigand(DataTransform):
    def __init__(self, *args, **kwargs):
        self._func = AppendVirtualNodesInCoM(*args, **kwargs)
    
    def __call__(self, data):
        data['ligand'] = self._func(data['ligand'])
        return data


class MaskLigand(DataTransform):
    def __init__(self, *args, **kwargs):
        self._func = Mask(*args, **kwargs)
    
    def __call__(self, data):
        data['ligand'] = self._func(data['ligand'])
        return data


class CenterData(DataTransform):
    def __call__(self, data):
        ligand = data['ligand']
        pocket = Residues(**data['pocket'])
        ligand['mask'] = torch.zeros(len(ligand['x']), dtype=INT_TYPE, device=ligand['x'].device)
        pocket['mask'] = torch.zeros(len(pocket['x']), dtype=INT_TYPE, device=pocket['x'].device)
        if pocket['x'].numel() > 0:
            center_of_mass = pocket.center()  # removes the CoM of the pocket already
        else:
            # center ligand at zero if the pocket is empty
            center_of_mass = scatter_mean(ligand['x'], ligand['mask'], dim=0)
        ligand['x'] = ligand['x'] - center_of_mass[ligand['mask']]
        return {'ligand': ligand, 'pocket': pocket}
    

class RandomRotation(DataTransform):
    """Apply random rotation."""
    def __call__(self, data):
        ligand = Ligand(**data['ligand'])
        pocket = Residues(**data['pocket'])
        rot_mat = R.random().as_matrix()
        rot_mat = torch.tensor(rot_mat, device=ligand['x'].device, dtype=torch.float32)
        ligand.rigid_transform(rot=rot_mat)
        pocket.rigid_transform(rot=rot_mat)
        return {'ligand': ligand, 'pocket': pocket}
