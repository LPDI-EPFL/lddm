
import math
import torch
from torch import nn


def binomial_coefficient(n, k):
    # source: https://discuss.pytorch.org/t/n-choose-k-function/121974
    return ((n + 1).lgamma() - (k + 1).lgamma() - ((n - k) + 1).lgamma()).exp()


def cycle_counts(adj):
    assert (adj.diag() == 0).all()
    assert (adj == adj.T).all()

    A = adj.float()
    d = A.sum(dim=-1)

    # Compute powers
    A2 = A @ A
    A3 = A2 @ A
    A4 = A3 @ A
    A5 = A4 @ A

    x3 = A3.diag() / 2
    x4 = (A4.diag() - d * (d - 1) - A @ d) / 2

    """ New (different from DiGress)
    case where correction is relevant:
    2   o
        |
    1,3 o--o 4
        | /
    0,5 o
    """
    # Triangle count matrix (indicates for each node i how many triangles it shares with node j)
    T = adj * A2
    x5 = (A5.diag() - 2 * T @ d - 4 * d * x3 - 2 * A @ x3 + 10 * x3) / 2

    return torch.stack([x3, x4, x5], dim=-1)


def eigenfeatures(A, batch_mask, k=5):

    # split adjacency matrix
    batch = []
    for i in torch.unique(batch_mask, sorted=True):
        batch_inds = torch.where(batch_mask == i)[0]
        batch.append(A[torch.meshgrid(batch_inds, batch_inds, indexing='ij')])

    eigenfeats = [get_nontrivial_eigenvectors(adj)[:, :k] for adj in batch]
    # if there are less than k non-trivial eigenvectors
    eigenfeats = [torch.cat([
        x, torch.zeros(x.size(0), max(k - x.size(1), 0), device=x.device)], dim=-1)
        for x in eigenfeats]
    return torch.cat(eigenfeats, dim=0)


def get_nontrivial_eigenvectors(A, normalize_l=True, thresh=1e-5, norm_eps=1e-12):
    """
    Compute eigenvectors of the graph Laplacian corresponding to non-zero
    eigenvalues.
    """
    assert (A == A.T).all(), "undirected graph"

    # Compute laplacian
    d = A.sum(-1)
    D = d.diag()
    L = D - A

    if normalize_l:
        D_inv_sqrt = (1 / (d.sqrt() + norm_eps)).diag()
        L = D_inv_sqrt @ L @ D_inv_sqrt

    # Eigendecomposition
    # eigenvalues are sorted in ascending order
    # eigvecs matrix contains eigenvectors as its columns
    eigvals, eigvecs = torch.linalg.eigh(L)

    # index of first non-trivial eigenvector
    try:
        idx = torch.nonzero(eigvals > thresh)[0].item()
    except IndexError:
        # recover if no non-trivial eigenvectors are found
        idx = eigvecs.size(1)

    return eigvecs[:, idx:]


def map_edges(source_edges, target_edges, shape):
            
    target_indices = torch.full(shape, float("nan"), device=target_edges.device)
    target_indices[*target_edges] = torch.arange(target_edges.size(1), device=target_edges.device, dtype=target_indices.dtype)

    target_edge = target_indices[source_edges[0], source_edges[1]]
    source_edge_mask = ~target_edge.isnan()  # indicates whether the source edge is one of the target edges

    return target_edge.int(), source_edge_mask


class RBFEmbedding(nn.Module):
    """
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].

    Adapted from: https://github.com/jingraham/neurips19-graph-protein-design
    """
    def __init__(self, d_min=0., d_max=20., d_count=16):
        super().__init__()
        self.d_min = d_min
        self.d_max = d_max
        self.d_count = d_count
        self.d_sigma = (self.d_max - self.d_min) / self.d_count

    def forward(self, d):
        D_mu = torch.linspace(self.d_min, self.d_max, self.d_count, device=d.device)
        D_mu = D_mu.view([1, -1])
        D_expand = torch.unsqueeze(d, -1)
        RBF = torch.exp(-((D_expand - D_mu) / self.d_sigma) ** 2)
        return RBF


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Adapted from: https://wandb.ai/byyoung3/ml-news/reports/A-Gentle-Introduction-to-Diffusion---Vmlldzo2MzgxNjc3
    """
    def __init__(self, dim, max_period=10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, scheduler, warmup_steps, last_epoch=-1):
        """
        Args:
            optimizer: torch.optim.Optimizer
            scheduler: a torch.optim.lr_scheduler._LRScheduler instance (the main scheduler)
            warmup_steps: number of steps to warm up
            last_epoch: the index of last epoch (default: -1)
        """
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup: scale base LR linearly with step
            return [base_lr * float(self.last_epoch + 1) / float(self.warmup_steps)
                    for base_lr in self.base_lrs]
        else:
            # After warmup, delegate to the main scheduler
            return self.scheduler.get_last_lr()

    def step(self, epoch=None):
        """Step both warmup and main scheduler."""
        if self.last_epoch < self.warmup_steps:
            return super().step(epoch)
        else:
            # step the wrapped scheduler
            if epoch is None:
                self.scheduler.step()
            else:
                self.scheduler.step(epoch - self.warmup_steps)
            self.last_epoch += 1
