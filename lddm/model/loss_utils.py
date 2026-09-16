import torch


class UniformTimestepSampler:
    def __call__(self, num_samples, device='cpu'):
        return torch.rand(num_samples, device=device)
    

class TrapezoidalTimestepSampler:
    def __init__(self, height_left=1.0):
        assert 0 <= height_left <= 2.0
        if height_left == 1.0:
            self.inverse_cdf = lambda x: x
        elif height_left < 1.0:
            term1 = 0.5 * height_left / (1 - height_left)
            self.inverse_cdf = lambda y: -term1 + torch.sqrt(term1**2 + y / (1 - height_left))
        elif height_left > 1.0:
            term1 = 0.5 * height_left / (1 - height_left)
            self.inverse_cdf = lambda y: -term1 - torch.sqrt(term1**2 + y / (1 - height_left))

    def __call__(self, num_samples, device='cpu'):
        uniform_samples = torch.rand(num_samples, device=device) #.unsqueeze(-1)
        return self.inverse_cdf(uniform_samples)


class TimestepWeights:
    def __init__(self, weight_type, a, b):
        if weight_type != 'sigmoid':
            raise NotImplementedError("Only sigmoidal loss weighting is available.")
        # self.weight_fn = lambda t: a * torch.sigmoid((-t + 0.5) * b) + (1 - a / 2)
        self.weight_fn = lambda t: a * torch.sigmoid((t - 0.5) * b) + (1 - a / 2)

    def __call__(self, t_array):
        # normalized t \in [0, 1]
        # return self.weight_fn(1 - t_array)
        return self.weight_fn(t_array)
