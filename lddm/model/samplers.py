from abc import abstractmethod
import math

import torch
from torch.distributions.categorical import Categorical

from lddm.data.data_utils import Ligand, Residues, TensorDict


class TimeSteps:
    """
    Inspired by
    https://arxiv.org/abs/2406.07266, section 4 (Sampling Molecules)
    https://github.com/rssrwn/semla-flow/blob/main/semlaflow/models/fm.py#L868
    """
    def __init__(self, num_steps, t_start=0.0, t_end=1.0, strategy='linear'):
        if strategy == 'linear':
            self.time_points = torch.linspace(t_start, t_end, num_steps + 1)
        elif strategy == 'log':
            eps = max(0.01, 1 - t_end)
            self.time_points = (
                1 - torch.logspace(math.log10(eps), math.log10(1 - t_start), num_steps + 1, base=10)
            ).flip(dims=(0,))
            self.time_points[1:] += eps
        else:
            raise NotImplementedError()
        
        self.delta_t = self.time_points[1:] - self.time_points[:-1]
        self.num_steps = num_steps
        self.i = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i >= self.num_steps:
            raise StopIteration 
        t, dt = self.time_points[self.i], self.delta_t[self.i]
        self.i += 1
        return t, dt


class AbstractSampler:
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def sample_zt_given_zs(self, zs_ligand, zs_pocket, s, t, uncertainty=None, known_x=None, known_h=None, known_e=None):
        """Return ligand and protein at time step t"""
        zt_ligand, zt_pocket = None, None
        pred_ligand, pred_pocket = None, None
        return zt_ligand, zt_pocket, pred_ligand, pred_pocket

    def __call__(self, ligand: Ligand, pocket: Residues, timesteps, t_start, t_end=1.0, return_frames=1, project_final=False, known_x=None, known_h=None, known_e=None, step_spacing='linear'):
        """
        Take a version of the ligand and pocket (at any time step t_start) and
        simulate the generative process from t_start to t_end.
        """

        # assert 0 < return_frames <= timesteps
        assert return_frames > 0
        return_frames = min(return_frames, timesteps)
        assert timesteps % return_frames == 0
        assert 0.0 <= t_start < 1.0
        assert 0 < t_end <= 1.0
        assert t_start < t_end

        device = ligand['x'].device
        n_samples = len(pocket['size'])
        # delta_t = (t_end - t_start) / timesteps

        # infer tensor shapes
        n_ligand_atoms = len(ligand['mask'])
        n_ligand_edges = len(ligand['bond_mask'])
        n_residues = len(pocket['mask'])
        x_dim = ligand["x"].size(1)
        atom_nf = ligand["h"].size(1)
        bond_nf = ligand["e"].size(1)
        n_atoms_per_aa = pocket["v"].size(1)

        # Initialize output tensors
        out_ligand = {
            'x': torch.zeros((return_frames, n_ligand_atoms, x_dim), device=device),
            'h': torch.zeros((return_frames, n_ligand_atoms, atom_nf), device=device),
            'e': torch.zeros((return_frames, n_ligand_edges, bond_nf), device=device)
        }
        if self.model.predict_confidence:
            out_ligand['sigma_x'] = torch.zeros((return_frames, n_ligand_atoms), device=device)
            out_ligand['entropy_h'] = torch.zeros((return_frames, n_ligand_atoms), device=device)
        out_pocket = {
            'x': torch.zeros((return_frames, n_residues, x_dim), device=device),  # CA-coord
            'v': torch.zeros((return_frames, n_residues, n_atoms_per_aa, x_dim), device=device)  # difference vectors to all other atoms
        }

        cumulative_uncertainty = {
            'sigma_x_squared': torch.zeros(n_ligand_atoms, device=device),
            'entropy_h': torch.zeros(n_ligand_atoms, device=device)
        } if self.model.predict_confidence else None

        # for i, t in enumerate(torch.linspace(t_start, t_end - delta_t, timesteps)):
        for i, (t, delta_t) in enumerate(TimeSteps(timesteps, t_start=t_start, t_end=t_end, strategy=step_spacing)):
            t_array = torch.full((n_samples, 1), fill_value=t, device=device)

            ligand, pocket, pred_ligand, pred_pocket = self.sample_zt_given_zs(
                ligand, pocket, t_array, t_array + delta_t, cumulative_uncertainty,
                known_x, known_h, known_e,
            )
            if self.model.predict_confidence:
                cumulative_uncertainty['sigma_x_squared'][known_x] = 0.0
                cumulative_uncertainty['entropy_h'][known_h] = 0.0

            # save frame
            if (i + 1) % (timesteps // return_frames) == 0:
                idx = (i + 1) // (timesteps // return_frames)
                idx = idx - 1

                frame_ligand = self.model.project_final_ligand(ligand, pred_ligand, t_array) if project_final and i < timesteps - 1 else ligand
                frame_pocket = pocket  # the pocket is kept fixed during generation

                out_ligand['x'][idx] = frame_ligand['x'].detach()
                out_ligand['h'][idx] = frame_ligand['h'].detach()
                out_ligand['e'][idx] = frame_ligand['e'].detach()
                out_pocket['x'][idx] = frame_pocket['x'].detach()
                out_pocket['v'][idx] = frame_pocket['v'][:, :n_atoms_per_aa, :].detach()
                if self.model.predict_confidence:
                    out_ligand['sigma_x'][idx] = cumulative_uncertainty['sigma_x_squared'].sqrt().detach()
                    out_ligand['entropy_h'][idx] = cumulative_uncertainty['entropy_h'].detach()

        # remove frame dimension if only the final molecule is returned
        out_ligand = {k: v.squeeze(0) for k, v in out_ligand.items()}
        out_pocket = {k: v.squeeze(0) for k, v in out_pocket.items()}

        return out_ligand, out_pocket
        

class ForwardEuler(AbstractSampler):
    def sample_zt_given_zs(
            self, zs_ligand: Ligand, zs_pocket: Residues, s, t, uncertainty=None, known_x=None, known_h=None, known_e=None,
    ):

        sc_transform = self.model.get_sc_transform_fn(zs_ligand['x'], s, zs_ligand['mask'])
        pred_ligand, pred_residues = self.model.dynamics(
            zs_ligand['x'], zs_ligand['h'], zs_ligand['mask'], zs_pocket, s, bonds_ligand=(zs_ligand['bonds'], zs_ligand['e']),
            sc_transform=sc_transform, known_x=known_x, known_h=known_h, known_e=known_e,
        )

        zt_ligand = zs_ligand.deepcopy()
        zt_ligand['x'] = self.model.module_x.sample_zt_given_zs(zs_ligand['x'], pred_ligand['vel'], s, t, zs_ligand['mask'])
        zt_ligand['h'] = self.model.module_h.sample_zt_given_zs(zs_ligand['h'], pred_ligand['logits_h'], s, t, zs_ligand['mask'])
        zt_ligand['e'] = self.model.module_e.sample_zt_given_zs(zs_ligand['e'], pred_ligand['logits_e'], s, t, zs_ligand['bond_mask'])

        # the pocket is kept fixed during generation
        zt_pocket = zs_pocket.deepcopy()

        # Masking
        zt_ligand = zt_ligand.insert_known_variables()

        if self.model.predict_confidence:
            assert uncertainty is not None
            dt = (t - s).view(-1)[zt_ligand['mask']]
            sigma2 = pred_ligand['uncertainty_vel'] if self.model.uncertainty_is_variance else pred_ligand['uncertainty_vel'] ** 2
            uncertainty['sigma_x_squared'] += (dt * sigma2) * (1 - known_x.float())
            uncertainty['entropy_h'] += (dt * Categorical(logits=pred_ligand['logits_h']).entropy()) * (1 - known_h.float())

        return zt_ligand, zt_pocket, pred_ligand, pred_residues


class HeunSampler(AbstractSampler):
    """ https://en.wikipedia.org/wiki/Heun%27s_method """
    def predict(
            self, zs_ligand, zs_pocket, s, known_x=None, known_h=None, known_e=None,
    ):

        sc_transform = self.model.get_sc_transform_fn(zs_ligand['x'], s, zs_ligand['mask'])
        pred_ligand, pred_residues = self.model.dynamics(
            zs_ligand['x'], zs_ligand['h'], zs_ligand['mask'], zs_pocket, s, bonds_ligand=(zs_ligand['bonds'], zs_ligand['e']),
            sc_transform=sc_transform, known_x=known_x, known_h=known_h, known_e=known_e,
        )

        return pred_ligand, pred_residues

    def update(
            self, zs_ligand: Ligand, zs_pocket: Residues, pred_ligand, pred_residues, s, t, 
    ):

        zt_ligand = zs_ligand.deepcopy()
        zt_pocket = zs_pocket.deepcopy()

        zt_ligand['x'] = self.model.module_x.sample_zt_given_zs(zs_ligand['x'], pred_ligand['vel'], s, t, zs_ligand['mask'])
        zt_ligand['h'] = self.model.module_h.sample_zt_given_zs(zs_ligand['h'], pred_ligand['logits_h'], s, t, zs_ligand['mask'])
        zt_ligand['e'] = self.model.module_e.sample_zt_given_zs(zs_ligand['e'], pred_ligand['logits_e'], s, t, zs_ligand['bond_mask'])

        # the pocket is kept fixed during generation

        # Masking
        zt_ligand = zt_ligand.insert_known_variables()

        return zt_ligand, zt_pocket

    def sample_zt_given_zs(
            self, zs_ligand, zs_pocket, s, t, uncertainty=None, known_x=None, known_h=None, known_e=None,
    ):
        # Perform standard Euler step to obtain intermediate state
        pred_ligand_1, pred_residues_1 = self.predict(
            zs_ligand, zs_pocket, s, known_x, known_h, known_e,
        )
        zt_ligand_tmp, zt_pocket_tmp = self.update(
            zs_ligand, zs_pocket, pred_ligand_1, pred_residues_1, s, t
        )

        # Estimate gradient at next time step
        pred_ligand_2, pred_residues_2 = self.predict(
            zt_ligand_tmp, zt_pocket_tmp, t, known_x, known_h, known_e,
        )

        # Average predictions
        expected_keys_ligand = {'vel', 'logits_h', 'logits_e', 'uncertainty_vel'}
        pred_ligand = {k: 0.5 * (pred_ligand_1[k] + pred_ligand_2[k]) for k in set(pred_ligand_2.keys()) & expected_keys_ligand}

        pred_residues = {}

        # Perform final update
        zt_ligand, zt_pocket = self.update(
            zs_ligand, zs_pocket, pred_ligand, pred_residues, s, t
        )

        if self.model.dynamics.self_conditioning:
            # Use 'corrected' prediction instead of the intermediate prediction from the last step
            self.model.dynamics.prev_ligand = TensorDict(**pred_ligand).deepcopy()
            self.model.dynamics.prev_residues = TensorDict(**pred_residues).deepcopy()

        if self.model.predict_confidence:
            assert uncertainty is not None
            dt = (t - s).view(-1)[zt_ligand['mask']]
            sigma2 = pred_ligand['uncertainty_vel'] if self.model.uncertainty_is_variance else pred_ligand['uncertainty_vel'] ** 2
            uncertainty['sigma_x_squared'] += (dt * sigma2) * (1 - known_x.float())
            uncertainty['entropy_h'] += (dt * Categorical(logits=pred_ligand['logits_h']).entropy()) * (1 - known_h.float())

        return zt_ligand, zt_pocket, pred_ligand, pred_residues
