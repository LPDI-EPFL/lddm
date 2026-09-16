import logging
from abc import ABC
from abc import abstractmethod
import math
import torch
import torch.nn.functional as F
from scipy.stats import chi2

from lddm.model.diffusion_utils import diagonalize
from lddm.scatter import scatter_mean, scatter_add
from lddm.utils import is_one_hot, argsort_within_batch


class ICFM(ABC):
    """
    Abstract base class for all Independent-coupling CFM classes.
    Defines a common interface.
    Notation:
    - zt is the intermediate representation at time step t \in [0, 1]
    - zs is the noised representation at time step s < t

    Reference:
    Tong, Alexander, et al. 
    "Improving and generalizing flow-based generative models with minibatch optimal transport." 
    arXiv preprint arXiv:2302.00482 (2023).
    """
    def __init__(self):
        pass

    @abstractmethod
    def sample_zt(self, z0, z1, t, *args, **kwargs):
        pass

    @abstractmethod
    def sample_zt_given_zs(self, *args, **kwargs):
        """ Perform update, typically using an explicit Euler step. """
        pass

    @abstractmethod
    def sample_z0(self, *args, **kwargs):
        """ Prior. """
        pass


    @abstractmethod
    def compute_loss(self, pred, z0, z1, zt, t, *args, **kwargs):
        """ Compute loss per sample. """
        pass

    @abstractmethod
    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, *args, **kwargs):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """
        pass


# ------------------------------------------------------------------------------
# Riemannian ICFM classes
# ------------------------------------------------------------------------------

class RiemannianICFM(ICFM):
    """
    Following:
    Chen, Ricky TQ, and Yaron Lipman.
    "Riemannian flow matching on general geometries."
    arXiv preprint arXiv:2302.03660 (2023).
    """
    def __init__(self, sigma, dim, scale=1, scheduler_args=None, predict_final=False, loss_type=None):
        super().__init__()
        self.sigma = sigma
        self.dim = dim
        self.loss_type = loss_type
        self.predict_final = predict_final

        # If z0 and z1 are normally distributed, the standard deviation of their 
        # difference (same as vector field) is sqrt(sigma0^2 + sigma1^2).
        # This can be used to normalize the target vector field.
        self.scale = scale

        # Scheduler that determines the rate at which the geodesic distance decreases
        scheduler_args = scheduler_args or {}
        scheduler_args["type"] = scheduler_args.get("type", "linear")  # default
        scheduler_args["learn_scaled"] = scheduler_args.get("learn_scaled", False)  # default

        # linear scheduler: kappa(t) = 1-t (default)
        if scheduler_args["type"] == "linear":
            # equivalent to: 1 - kappa(t)
            self.flow_scaling = lambda t: t

            # equivalent to: -1 * d/dt kappa(t)
            self.velocity_scaling = lambda t: torch.ones_like(t)
            logging.debug('Using linear schedule')

        elif scheduler_args["type"] == "piecewise-linear":
            self.start = scheduler_args["start"]
            self.end = scheduler_args["end"]

            # equivalent to: 1 - kappa(t)
            self.flow_scaling = lambda t: torch.clamp((t - self.start) / (self.end - self.start), 0, 1)

            # equivalent to: -1 * d/dt kappa(t)
            self.velocity_scaling = lambda t: (
                torch.ones_like(t) / (self.end - self.start) *
                (t > self.start).float() * (t < self.end).float()
            )
            logging.debug('Using piecewise-linear schedule')

        # exponential scheduler: kappa(t) = exp(-c*t)
        elif scheduler_args["type"] == "exponential":

            self.c = scheduler_args["c"]
            assert self.c > 0

            # equivalent to: 1 - kappa(t)
            self.flow_scaling = lambda t: 1 - torch.exp(-self.c * t)

            # equivalent to: -1 * d/dt kappa(t)
            self.velocity_scaling = lambda t: self.c * torch.exp(-self.c * t)
            logging.debug('Using exponential schedule')

        # polynomial scheduler: kappa(t) = (1-t)^k
        elif scheduler_args["type"] == "polynomial":
            self.k = scheduler_args["k"]
            assert self.k > 0

            # equivalent to: 1 - kappa(t)
            self.flow_scaling = lambda t: 1 - (1 - t)**self.k

            # equivalent to: -1 * d/dt kappa(t)
            self.velocity_scaling = lambda t: self.k * (1 - t)**(self.k - 1)
            logging.debug('Using polynomial schedule')

        else:
            raise NotImplementedError(f"Scheduler {scheduler_args['type']} not implemented.")

        kappa_interval = self.flow_scaling(torch.tensor([0.0, 1.0]))
        if kappa_interval[0] != 0.0 or kappa_interval[1] != 1.0:
            print(f"Scheduler should satisfy kappa(0)=1 and kappa(1)=0. Found "
                  f"interval {kappa_interval.tolist()} instead.")

        # determines whether the scaled vector field is learned or the scheduler
        # is post-multiplied
        self.learn_scaled = scheduler_args["learn_scaled"]
        assert not (self.learn_scaled and self.predict_final), "Options are mutually exclusive."

    def sample_zt(self, z0, z1, t, batch_mask):
        """ expressed in terms of exponential and logarithm maps """

        # apply logarithm map
        zt_tangent = self.flow_scaling(t)[batch_mask] * self.logarithm_map(z0, z1)

        if self.sigma is not None:
            zt_tangent = zt_tangent + self.sigma * torch.randn_like(zt_tangent)

        # apply exponential map
        return self.exponential_map(z0, zt_tangent)
    
    def pred_to_vector_field(self, pred, t, batch_mask, eps=1e-8):

        if self.predict_final:
            # pred = z1 - zt
            vel = pred / torch.clamp(1 - t, min=eps)[batch_mask]
        
        else:
            # pred = (z1 - z0) / self.scale = (z1 - zt) / (1 - t) / self.scale
            vel = self.scale * pred
            
        if not self.learn_scaled:
            vel = self.velocity_scaling(t)[batch_mask] * vel
        
        return vel
    
    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask):
        """ Perform update, typically using an explicit Euler step. """

        step_size = t - s
        vel = self.pred_to_vector_field(pred, s, batch_mask)

        zt_tangent = step_size[batch_mask] * vel

        # exponential map
        return self.exponential_map(zs, zt_tangent)
    
    def get_z1_given_zt_and_pred(self, zt, pred, z0, t, batch_mask):
        """ Make a best guess on the final state z1 given the current state and
        the network prediction. """

        vel = self.pred_to_vector_field(pred, t, batch_mask)

        # Undo velocity scaling to obtain a constant velocity
        vel = vel / torch.clamp(self.velocity_scaling(t), min=1e-3)[batch_mask]

        z1_tangent = (1 - t)[batch_mask] * vel

        # exponential map
        return self.exponential_map(zt, z1_tangent)

    def reduce_loss(self, loss, batch_mask, reduce, batch_size):
        assert reduce in {'mean', 'sum', 'none'}

        if reduce == 'mean':
            loss = scatter_mean(loss / self.dim, batch_mask, dim=0, dim_size=batch_size)
        elif reduce == 'sum':
            loss = scatter_add(loss, batch_mask, dim=0, dim_size=batch_size)

        if 0 < loss.numel() < batch_size:
            # Last k samples are not used in the loss computation 
            # because they were assigned as known
            assert len(loss) == batch_mask.max() + 1
            n_missing = batch_size - len(loss)
            zeros = torch.zeros(n_missing, device=loss.device, dtype=loss.dtype)
            loss = torch.cat([loss, zeros])

        return loss
    
    def target_vector_field(self, zt, t, z1, *, z0=None, batch_mask, eps=1e-8, **kwargs):

        # zt_dot = self.logarithm_map(z0, z1)
        zt_dot = self.logarithm_map(zt, z1) / torch.clamp(1 - t, min=eps)[batch_mask]
        zt_dot = zt_dot / self.scale
        if self.learn_scaled:
            # NOTE: potentially requires output magnitude to vary substantially
            zt_dot = self.velocity_scaling(t)[batch_mask] * zt_dot

        return zt_dot
    
    def compute_loss(self, pred, z0, z1, zt, t, batch_mask, reduce='mean', known_mask=None, batch_size=None):
        """ Compute loss per sample. """

        if self.predict_final:
            # target = z1 - zt
            target = self.logarithm_map(zt, z1)
        else:
            # target = zt_dot = (z1 - zt) / (1 - t)
            target = self.target_vector_field(zt, t, z1, z0=z0, batch_mask=batch_mask)
        
        if self.loss_type == "L1":
            loss = torch.linalg.norm(pred - target, dim=-1)
        else:
            # default flow matching loss
            loss = torch.sum((pred - target) ** 2, dim=-1)

        bs = batch_size or batch_mask.max() + 1
        if known_mask is not None:
            return self.reduce_loss(loss[~known_mask], batch_mask[~known_mask], reduce, bs)
        else:
            return self.reduce_loss(loss, batch_mask, reduce, bs)
        
    @classmethod
    @abstractmethod
    def exponential_map(cls, x, u):
        """
        :param x: point on the manifold
        :param u: point on the tangent space
        """
        pass

    @classmethod
    @abstractmethod
    def logarithm_map(cls, x, y):
        """
        :param x, y: points on the manifold
        """
        pass


class CoordICFM(RiemannianICFM):
    def __init__(self, scheduler_args=None, loss_type=None, prior_scale=1.0, sigma=None, predict_final=False, noise_scale=None, **kwargs):
        super().__init__(sigma=sigma, dim=3, scale=2.7, scheduler_args=scheduler_args, predict_final=predict_final, loss_type=loss_type, **kwargs)

        self.prior_scale = prior_scale  # scales the prior
        self.noise_scale = noise_scale

        # adjust scaling of the vector field
        assert self.scale >= 1.0
        self.scale = math.sqrt(self.prior_scale**2 - 1 + self.scale**2)

        # used for score correction
        self.last_com = None
        
    @classmethod
    def exponential_map(cls, x, u):
        return x + u
    
    @classmethod
    def logarithm_map(cls, x, y):
        return y - x

    def sample_z0(self, com, batch_mask, edges=None, ptr=None):
        """ Prior. """
        n = len(batch_mask)
        z0 = torch.randn((n, self.dim), device=batch_mask.device)
        
        if edges is not None and ptr is not None:
            # Harmonic prior
            D, P = diagonalize(n=n, edges=edges, ptr=ptr)
            z0 = P @ (z0 / torch.sqrt(D)[:, None])

        z0 = self.prior_scale * z0
                
        # Move center of mass
        z0 = z0 + com[batch_mask]

        self.last_com = com  # keep track of the noise mean

        return z0
    
    def sample_zt_given_zs(self, zs, pred, s, t, batch_mask):
        """ Perform update, typically using an explicit Euler step. """
        
        step_size = t - s
        vel = self.pred_to_vector_field(pred, s, batch_mask)

        if self.noise_scale is not None:
            # See Section II.B in: https://arxiv.org/abs/2509.01543
            # zt = beta_t * z0 + alpha_t * z1
            # Here: alpha_t = 1 - kappa_t, beta_t = kappa_t
            kappa_s = 1 - self.flow_scaling(s)
            kappa_dot_s = -1 * self.velocity_scaling(s)

            # NOTE: (only) in the Gaussian case, the score can be expressed directly as a function of the velocity field
            # score = (((1 - kappa_s) / minus_kappa_dot_s)[batch_mask] * vel - zs) / kappa_s[batch_mask]
            score = -1 * (((1 - kappa_s) / kappa_dot_s)[batch_mask] * vel + zs - self.last_com[batch_mask]) / kappa_s[batch_mask]
            # NOTE: the self.last_com term enters the equation when re-deriving the score for the case where z0 is 
            # sampled from a Gaussian with non-zero mean, mu.
            # We can change Eq. (105) in Section B.4 of https://arxiv.org/abs/2409.08861
            # score(z,t) = 1/(beta_t * (\dot{alpha}_t/alpha_t * beta_t - \dot{beta}_t)) * (v(z,t) - \dot{alpha_t}/alpha_t * z) + mu / beta_t

            # NOTE: noise schedule sigma: [0, 1] -> R_{\geq0} can be chosen         
            # sigma_s = self.noise_scale * torch.sqrt((1 - s) / s.clamp(min=0.01))  # from Flow-GRPO (https://arxiv.org/abs/2505.05470)
            # sigma_s = self.noise_scale * torch.ones_like(s)  # constant
            sigma_s = self.noise_scale * (1 - s)  # as FK-Flow, Appendix A.3 (https://arxiv.org/abs/2509.01543)
            # sigma_s = self.noise_scale * s * (1 - s)  # similar to FoldFlow (https://arxiv.org/abs/2310.02391)

            # dx = f * dt + g * sqrt(dt) * randn (with drift vector f and diffusion matrix g)
            drift = vel + (sigma_s**2)[batch_mask] / 2 * score
            diffusion = sigma_s[batch_mask]
            zt_tangent = drift * step_size[batch_mask] + \
                diffusion * torch.sqrt(step_size)[batch_mask] * torch.randn_like(zs)            
        else:
            zt_tangent = step_size[batch_mask] * vel

        # exponential map
        return self.exponential_map(zs, zt_tangent)
