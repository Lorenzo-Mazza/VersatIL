"""Kernel functions for Maximum Mean Discrepancy (MMD) computation.

Provides composable kernel modules that can be plugged into MMD-based losses.
"""

import abc
import enum

import torch
from torch import nn


class MMDKernel(nn.Module, abc.ABC):
    """Base class for MMD kernels.

    Provides shared utilities for pairwise distance computation and
    the median heuristic for adaptive bandwidth selection.
    """

    def compute_pairwise_squared_distances(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise squared Euclidean distances between point sets.

        Args:
            x: First point set, shape (N, D).
            y: Second point set, shape (M, D).

        Returns:
            Distance matrix, shape (N, M).
        """
        if x.dim() > 2:
            x = x.view(-1, x.size(-1))
        if y.dim() > 2:
            y = y.view(-1, y.size(-1))
        xx = (x**2).sum(-1, keepdim=True)  # (N, 1)
        yy = (y**2).sum(-1, keepdim=True).t()  # (1, M)
        xy = torch.mm(x, y.t())  # (N, M)
        dist_sq = xx + yy - 2 * xy
        return torch.clamp(dist_sq, min=1e-10)

    def compute_median_squared_distance(self, points: torch.Tensor) -> float:
        """Compute median pairwise squared distance for bandwidth selection.

        Ref: https://torchdrift.org/notebooks/note_on_mmd.html

        Args:
            points: Point set, shape (N, D).

        Returns:
            Median squared distance (scalar).
        """
        points = points.detach()
        if points.dim() > 2:
            points = points.view(-1, points.size(-1))
        device = points.device
        norms = (points**2).sum(-1)
        dist_sq = (
            norms.unsqueeze(0) + norms.unsqueeze(1) - 2 * torch.mm(points, points.t())
        )
        dist_sq = torch.clamp(dist_sq, min=0.0)
        triu_i, triu_j = torch.triu_indices(
            points.shape[0], points.shape[0], offset=1, device=device
        )
        pairwise_dist_sq = dist_sq[triu_i, triu_j]
        if pairwise_dist_sq.numel() == 0:
            return 1.0
        median = torch.median(pairwise_dist_sq)
        if median <= 1e-6:
            median = torch.tensor(1.0, device=device)
        return median.item()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix K(x, y).

        Args:
            x: First point set, shape (N, D).
            y: Second point set, shape (M, D).

        Returns:
            Kernel matrix, shape (N, M).
        """
        raise NotImplementedError


class RBFKernel(MMDKernel):
    """Multi-scale RBF (Gaussian) kernel with median heuristic bandwidth.

    K(x, y) = (1/S) * sum_i exp(-||x-y||^2 / (2 * s_i * median_dist^2))
    where s_i are bandwidth multipliers and median_dist^2 is computed from the data.

    Ref: Gretton et al., "A Kernel Two-Sample Test",  https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

    Args:
        bandwidth_multipliers: Scale factors applied to the median heuristic bandwidth.
    """

    def __init__(
        self,
        bandwidth_multipliers: list[float] | None = None,
    ):
        super().__init__()
        if bandwidth_multipliers is None:
            bandwidth_multipliers = [0.2, 0.5, 1.0, 2.0, 5.0]
        self.bandwidth_multipliers = bandwidth_multipliers

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale RBF kernel matrix.

        Args:
            x: First point set, shape (N, D).
            y: Second point set, shape (M, D).

        Returns:
            Kernel matrix, shape (N, M).
        """
        combined = torch.cat([x, y], dim=0)
        median_dist_sq = self.compute_median_squared_distance(combined)
        dist_sq = self.compute_pairwise_squared_distances(x, y)
        kernel = torch.zeros_like(dist_sq)
        for mult in self.bandwidth_multipliers:
            bandwidth = 2.0 * mult * median_dist_sq
            kernel = kernel + torch.exp(-dist_sq / bandwidth)
        kernel = kernel / len(self.bandwidth_multipliers)
        return kernel


class IMQKernel(MMDKernel):
    """Multi-scale Inverse Multiquadratic kernel with median heuristic bandwidth.

    K(x, y) = (1/S) * sum_i C_i / (C_i + ||x-y||^2)
    where C_i = 2 * s_i * median_dist^2 and s_i are bandwidth multipliers.

    Heavier polynomial tails than RBF — provides gradient signal for outlier
    latent codes that RBF's exponential decay misses, especially early in training.

    Ref: Tolstikhin et al., "Wasserstein Auto-Encoders" (ICLR 2018)

    Args:
        bandwidth_multipliers: Scale factors applied to the median heuristic bandwidth.
    """

    def __init__(
        self,
        bandwidth_multipliers: list[float] | None = None,
    ):
        super().__init__()
        if bandwidth_multipliers is None:
            bandwidth_multipliers = [0.2, 0.5, 1.0, 2.0, 5.0]
        self.bandwidth_multipliers = bandwidth_multipliers

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale IMQ kernel matrix.

        Args:
            x: First point set, shape (N, D).
            y: Second point set, shape (M, D).

        Returns:
            Kernel matrix, shape (N, M).
        """
        combined = torch.cat([x, y], dim=0)
        median_dist_sq = self.compute_median_squared_distance(combined)
        dist_sq = self.compute_pairwise_squared_distances(x, y)
        kernel = torch.zeros_like(dist_sq)
        for mult in self.bandwidth_multipliers:
            c = 2.0 * mult * median_dist_sq
            kernel = kernel + c / (c + dist_sq)
        kernel = kernel / len(self.bandwidth_multipliers)
        return kernel


class KernelType(str, enum.Enum):
    """Kernel type for MMD computation."""

    RBF = "rbf"
    IMQ = "imq"

    def to_kernel(self, bandwidth_multipliers: list[float] | None = None) -> MMDKernel:
        """Instantiate the corresponding kernel.

        Args:
            bandwidth_multipliers: Scale factors for the median heuristic bandwidth.

        Returns:
            Instantiated MMDKernel.
        """
        kernel_classes: dict[KernelType, type[MMDKernel]] = {
            KernelType.RBF: RBFKernel,
            KernelType.IMQ: IMQKernel,
        }
        return kernel_classes[self](bandwidth_multipliers=bandwidth_multipliers)
