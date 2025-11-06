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
        """
        super().__init__()
        self.latent_bits = latent_bits
        self.latent_dim = 2**latent_bits  # 2^H dimensional one-hot vectors
        self.embedding_dimension = embedding_dimension

        # Learned projection to latent logits
        self.logit_projection = nn.Linear(embedding_dimension, latent_bits)

        # Precompute bit patterns for all possible codes (0 to 2^H - 1)
        # Shape: (2^H, H) where each row is the binary representation of an index
        all_indices = torch.arange(self.latent_dim)
        bit_patterns = torch.zeros(self.latent_dim, latent_bits)
        for h in range(latent_bits):
            bit_patterns[:, h] = (all_indices // (2**h)) % 2
        self.register_buffer("bit_patterns", bit_patterns)  # (2^H, H)

    def _compute_soft_distribution(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute soft distribution G_t over all 2^H codes.

        G_{t,d} = ∏_{h=0}^{H-1} [σ(L_{t,h})^{b_h} · (1-σ(L_{t,h}))^{1-b_h}]

        Args:
            logits: Bit logits (..., H)

        Returns:
            Soft distribution over codes (..., 2^H)
        """
        # Compute probabilities for each bit
        probs = torch.sigmoid(logits)  # (..., H)

        # Expand dimensions for broadcasting
        # probs: (..., 1, H)
        # bit_patterns: (2^H, H)
        probs_expanded = probs.unsqueeze(-2)  # (..., 1, H)
        bit_patterns = self.bit_patterns  # (2^H, H)

        # Compute probability for each code d
        # For each bit h: σ(L_h)^{b_h} · (1-σ(L_h))^{1-b_h}
        # = σ(L_h) if b_h=1, else (1-σ(L_h))
        prob_if_one = probs_expanded  # (..., 1, H)
        prob_if_zero = 1 - probs_expanded  # (..., 1, H)

        # Select probability based on bit pattern
        # bit_patterns: (2^H, H) with values 0 or 1
        log_probs = torch.where(
            bit_patterns.unsqueeze(0) == 1,  # (1, 2^H, H)
            torch.log(prob_if_one + 1e-8),  # (..., 1, H)
            torch.log(prob_if_zero + 1e-8),  # (..., 1, H)
        )  # (..., 2^H, H)

        # Product over bits = sum of log probabilities
        log_soft_dist = log_probs.sum(dim=-1)  # (..., 2^H)
        soft_dist = torch.exp(log_soft_dist)  # (..., 2^H)

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
        hard_indices = (bits * powers_of_two).sum(dim=-1, keepdim=True).long()  # (..., 1)

        # Create hard one-hot Y_t
        y_hard = torch.zeros(*hard_indices.shape[:-1], self.latent_dim, device=features.device)
        y_hard.scatter_(-1, hard_indices, 1.0)  # (..., 2^H)

        # Compute soft distribution G_t
        g_soft = self._compute_soft_distribution(logits)  # (..., 2^H)

        # Straight-through estimator: Y_t + G_t - detach(G_t)
        # Forward: hard one-hot Y_t
        # Backward: gradients from soft distribution G_t
        one_hot = y_hard + g_soft - g_soft.detach()
        if self.training:
            print(f"Mean probs: {probs.mean().item()}")
        return one_hot, logits