"""Binary mapper for discrete latent sampling with gradient pass-through.

Based on "The Free Transformer" (Fleuret, 2025) - arXiv:2510.17558
Section 3.2: https://arxiv.org/abs/2510.17558

The binary mapper converts continuous features to discrete one-hot latent codes
using independent Bernoulli sampling per bit, with a proper straight-through gradient estimator
that computes the full soft distribution over all 2^H codes.
"""

import torch
import torch.nn as nn


class BinaryMapper(nn.Module):
    """Binary mapper for discrete latent sampling with gradient pass-through.

    Converts continuous logits to discrete one-hot latent codes using independent
    Bernoulli sampling per bit, with the paper's exact straight-through estimator.

    For gradients, computes the full soft distribution G_t over all 2^H codes:
    G_{t,d} = ∏_{h=0}^{H-1} [σ(L_{t,h})^{b_h} · (1-σ(L_{t,h}))^{1-b_h}]

    Uses straight-through: Y_t + G_t - detach(G_t)

    Args:
        latent_bits: Number of bits in the latent code (H in paper, typically 16)
        embedding_dimension: Input feature dimension
    """

    def __init__(
        self,
        latent_bits: int = 16,
        embedding_dimension: int = 256,
    ):
        """Initialize binary mapper.

        Args:
            latent_bits: Number of bits for the latent code (default: 16, giving 2^16 = 65536 codes)
            embedding_dimension: Dimension of input features

        Raises:
            ValueError: If dimensions are not positive.
        """
        super().__init__()
        if latent_bits <= 0:
            raise ValueError(f"latent_bits must be positive, got {latent_bits}.")
        if embedding_dimension <= 0:
            raise ValueError(
                f"embedding_dimension must be positive, got {embedding_dimension}."
            )
        self.latent_bits = latent_bits
        self.latent_dim = 1 << latent_bits  # 2^H dimensional one-hot vectors
        self.embedding_dimension = embedding_dimension
        self.logit_projection = nn.Linear(embedding_dimension, latent_bits)

        # Precompute bit patterns for all possible codes (0 to 2^H - 1)
        # Shape: (2^H, H) where each row is the binary representation of an index d (from 0 to 2^H - 1),
        # as an H-length vector of 0s and 1s (bits b_0 to b_{H-1}).
        all_indices = torch.arange(self.latent_dim)
        bit_patterns = torch.zeros(self.latent_dim, latent_bits)
        for h in range(latent_bits):
            bit_patterns[:, h] = (all_indices // (2**h)) % 2
        self.register_buffer("bit_patterns", bit_patterns)  # (2^H, H)

    def _compute_soft_distribution(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute soft distribution G_t over all 2^H codes.

        G_{t,d} = ∏_{h=0}^{H-1} [σ(L_{t,h})^{b_h} · (1-σ(L_{t,h}))^{1-b_h}]

        Args:
            logits: Bit logits (B, T, H)

        Returns:
            Soft distribution over codes (B, T, 2^H)
        """
        *batch_dims, H = logits.shape  # (B*T, H)
        if self.latent_bits != H:
            raise ValueError(
                f"Logits last dimension {H} does not match latent_bits {self.latent_bits}"
            )
        # Compute probabilities for each bit
        probs = torch.sigmoid(logits)  # (..., H)
        log_probs = torch.log(probs.clamp(min=1e-8))  # log σ(L_h) (..., H)
        log_one_minus_probs = torch.log(
            (1 - probs).clamp(min=1e-8)
        )  # log (1-σ(L_h)) (..., H)
        # Expand dimensions for broadcasting and vectorized computation in parallel
        bit_patterns_expanded = self.bit_patterns.reshape(
            *([1] * len(batch_dims)), self.latent_dim, H
        ).expand(*batch_dims, self.latent_dim, H)  # (..., 2^H, H)
        # Compute probability for each code d
        # For each bit h: σ(L_h)^{b_h} · (1-σ(L_h))^{1-b_h}
        # = σ(L_h) if b_h=1, else (1-σ(L_h))
        log_prob_if_one = log_probs.unsqueeze(-2)  # (..., 1, H)
        log_prob_if_zero = log_one_minus_probs.unsqueeze(-2)  # (..., 1, H)
        # Select probability based on bit pattern
        # bit_patterns: (2^H, H) with values 0 or 1
        log_probs_per_code = torch.where(
            (bit_patterns_expanded == 1).to(torch.bool),
            log_prob_if_one,  # broadcasts to (..., 2^H, H)
            log_prob_if_zero,
        )  # (..., 2^H, H)
        # Product over bits = sum of log probabilities
        log_soft_dist = log_probs_per_code.sum(dim=-1)  # (..., 2^H)
        # Numerical stability: softmax over log (prevents underflow/overflow)
        soft_dist = torch.softmax(log_soft_dist, dim=-1)  # (..., 2^H)
        return soft_dist

    def forward(
        self,
        features: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map features to discrete latent codes with proper gradient flow.

        Args:
            features: Input features (B, T, embedding_dimension) or (B, embedding_dimension)
            deterministic: If True, use hard threshold instead of sampling

        Returns:
            Tuple of:
                - one_hot: One-hot latent codes (B, T, latent_dim) or (B, latent_dim)
                - logits: Raw logits before sampling (B, T, latent_bits) or (B, latent_bits)
        """
        # Project to per-bit logits
        logits = self.logit_projection(features)  # (..., H)

        if deterministic:
            # Deterministic mode: use hard threshold at 0.5
            bits = (torch.sigmoid(logits) > 0.5).float()
        else:
            # Sample each bit independently: P(B_{t,h}=1) = sigmoid(L_{t,h})
            probs = torch.sigmoid(logits)
            bits = torch.bernoulli(probs)  # (..., H)

        # Convert sampled bits to index (hard one-hot)
        powers_of_two = 2 ** torch.arange(
            self.latent_bits, dtype=torch.long, device=bits.device
        )
        hard_indices = (bits.to(torch.long) * powers_of_two).sum(
            dim=-1, keepdim=True
        )  # (..., 1)

        # Create hard one-hot Y_t
        y_hard = torch.zeros(
            *hard_indices.shape[:-1],
            self.latent_dim,
            device=features.device,
            dtype=logits.dtype,
        )
        y_hard.scatter_(-1, hard_indices, 1.0)  # (..., 2^H)

        # Compute soft distribution G_t
        g_soft = self._compute_soft_distribution(logits)  # (..., 2^H)

        # Straight-through estimator trick: Y_t + G_t - detach(G_t)
        # Forward: hard one-hot Y_t
        # Backward: gradients from soft distribution G_t
        one_hot = y_hard + g_soft - g_soft.detach()
        return one_hot, logits
