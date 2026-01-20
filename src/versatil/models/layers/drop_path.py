import torch
from torch import nn


class DropPath(nn.Module):
    """Stochastic depth as in timm's DropPath, dropping entire samples from the batch.
    Taken from https://arxiv.org/pdf/1603.09382.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = bool(scale_by_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Work with broadcastable shape (N, 1, 1, 1, ...) to drop whole samples
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep and keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor
