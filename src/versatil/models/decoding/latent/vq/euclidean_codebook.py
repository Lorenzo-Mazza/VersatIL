"""Euclidean codebook with EMA updates for vector quantization.

Manages a set of learnable embedding vectors updated via exponential moving
averages of encoder outputs, following van den Oord et al. (2017) and
Tolstikhin et al. (2018). Supports automatic initialization on the first
batch and dead code replacement.
"""

import torch
from torch import nn


class EuclideanCodebook(nn.Module):
    """Codebook of embedding vectors with nearest-neighbor lookup and EMA updates.

    Args:
        num_codes: Number of codebook entries (K).
        code_dim: Dimension of each codebook vector.
        ema_decay: Exponential moving average decay for codebook updates.
            Higher values (e.g. 0.99) produce slower, more stable updates.
        dead_code_threshold: Minimum average cluster size below which a
            code is considered dead and replaced with a random encoder output.
        kmeans_init: Initialize codebook vectors from the first batch (True)
            or from N(0, 1) (False).
    """

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
        kmeans_init: bool = True,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold
        self.kmeans_init = kmeans_init

        if kmeans_init:
            embed = torch.zeros(
                num_codes, code_dim
            )  # (K, D) — placeholder until first batch
        else:
            embed = torch.randn(num_codes, code_dim)  # (K, D) — random init

        self.register_buffer("embed", embed)  # (K, D)
        self.register_buffer("cluster_size", torch.zeros(num_codes))  # (K,)
        self.register_buffer("embed_avg", embed.clone())  # (K, D)
        self.register_buffer(
            "initialized", torch.tensor(not kmeans_init)
        )  # scalar bool

    def _initialize_from_data(self, data: torch.Tensor) -> None:
        """Initialize codebook from the first batch of encoder outputs.

        Picks num_codes random points from the data as initial codebook vectors.

        Args:
            data: Encoder outputs, shape (N, D).
        """
        num_samples = data.shape[0]
        if num_samples < self.num_codes:
            indices = torch.randint(
                0, num_samples, (self.num_codes,), device=data.device
            )  # (K,)
        else:
            indices = torch.randperm(num_samples, device=data.device)[
                : self.num_codes
            ]  # (K,)
        selected = data[indices]  # (K, D)
        self.embed.data.copy_(selected)
        self.embed_avg.data.copy_(selected)
        self.cluster_size.data.fill_(1.0)
        self.initialized.fill_(True)

    def _replace_dead_codes(self, data: torch.Tensor) -> None:
        """Replace codebook entries with low usage by random encoder outputs.

        Args:
            data: Encoder outputs from the current batch, shape (N, D).
        """
        dead_mask = self.cluster_size < self.dead_code_threshold  # (K,) bool
        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return
        num_samples = data.shape[0]
        replace_indices = torch.randint(
            0, num_samples, (num_dead,), device=data.device
        )  # (num_dead,)
        self.embed.data[dead_mask] = data[replace_indices].detach()  # (num_dead, D)
        self.embed_avg.data[dead_mask] = data[replace_indices].detach()  # (num_dead, D)
        self.cluster_size.data[dead_mask] = 1.0

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize encoder outputs to nearest codebook entries.

        During training, updates codebook via EMA and replaces dead codes.

        Args:
            z_e: Encoder outputs, shape (B, D).

        Returns:
            Tuple of:
                quantized: Nearest codebook vectors, shape (B, D).
                indices: Codebook indices for each input, shape (B,).
        """
        if self.kmeans_init and not self.initialized:
            self._initialize_from_data(z_e.detach())

        # z_e: (B, D), self.embed: (K, D) -> dist: (B, K)
        dist = torch.cdist(z_e, self.embed)
        indices = dist.argmin(dim=-1)  # (B,)
        quantized = self.embed[indices]  # (B, D)

        if self.training:
            # One-hot assignment matrix: (B, K)
            one_hot = torch.zeros(z_e.shape[0], self.num_codes, device=z_e.device)
            one_hot.scatter_(
                1, indices.unsqueeze(1), 1.0
            )  # indices: (B,) -> (B, 1) for scatter

            # Cluster statistics for EMA
            new_cluster_size = one_hot.sum(dim=0)  # (K,)
            new_embed_sum = one_hot.T @ z_e.detach()  # (K, B) @ (B, D) -> (K, D)

            # EMA update: running_avg = decay * old + (1 - decay) * new
            self.cluster_size.data.mul_(self.ema_decay).add_(
                new_cluster_size, alpha=1.0 - self.ema_decay
            )  # (K,)
            self.embed_avg.data.mul_(self.ema_decay).add_(
                new_embed_sum, alpha=1.0 - self.ema_decay
            )  # (K, D)

            # Normalize to get updated embeddings: embed = embed_avg / cluster_size
            smoothed_cluster_size = (
                self.cluster_size + 1e-5
            )  # (K,) — avoid division by zero
            self.embed.data.copy_(
                self.embed_avg
                / smoothed_cluster_size.unsqueeze(1)  # (K, 1) broadcast over D
            )  # (K, D)

            self._replace_dead_codes(z_e.detach())

        return quantized, indices
