"""Single-layer vector quantization with straight-through gradient estimator.

Wraps an EuclideanCodebook with optional input/output projections.
Gradients flow through the quantization step via the straight-through
estimator: forward uses the discrete codebook lookup, backward pretends
it was an identity. Loss computation is handled externally by the
metrics module (VQCommitmentLoss).
"""

import torch
from torch import nn

from versatil.models.decoding.latent.vq.euclidean_codebook import EuclideanCodebook


class VectorQuantize(nn.Module):
    """Single-layer vector quantizer with straight-through gradients.

    Args:
        input_dim: Dimension of the input vectors from the encoder.
        code_dim: Dimension of each codebook vector. If different from
            input_dim, linear projections are added.
        num_codes: Number of codebook entries (K).
        ema_decay: EMA decay for codebook updates.
        dead_code_threshold: Cluster size below which a code is replaced.
        kmeans_init: Initialize codebook from the first batch.
    """

    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        num_codes: int,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
        kmeans_init: bool = True,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}.")
        if code_dim <= 0:
            raise ValueError(f"code_dim must be positive, got {code_dim}.")
        if num_codes <= 0:
            raise ValueError(f"num_codes must be positive, got {num_codes}.")
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.num_codes = num_codes

        needs_projection = input_dim != code_dim
        self.project_in = (
            nn.Linear(input_dim, code_dim, bias=False)
            if needs_projection
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(code_dim, input_dim, bias=False)
            if needs_projection
            else nn.Identity()
        )

        self.codebook = EuclideanCodebook(
            num_codes=num_codes,
            code_dim=code_dim,
            ema_decay=ema_decay,
            dead_code_threshold=dead_code_threshold,
            kmeans_init=kmeans_init,
        )

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize input via nearest codebook lookup with straight-through gradient.

        Args:
            z_e: Encoder output, shape (B, input_dim).

        Returns:
            Tuple of:
                z_q: Quantized output with straight-through gradient,
                    shape (B, input_dim).
                indices: Codebook indices, shape (B,).
                z_e_projected: Pre-quantization encoder output in code space,
                    shape (B, code_dim). Carries gradient; used as the
                    commitment-loss target for the encoder.
                z_q_code: Hard-quantized codebook vector in code space,
                    shape (B, code_dim), detached from the computation
                    graph. Paired with z_e_projected for per-layer
                    commitment loss.
        """
        z_e_projected = self.project_in(z_e)  # (B, code_dim)

        quantized, indices = self.codebook(z_e_projected)  # (B, code_dim), (B,)

        # Straight-through estimator: forward uses quantized, backward uses z_e_projected
        z_q = z_e_projected + (quantized - z_e_projected).detach()  # (B, code_dim)

        z_q = self.project_out(z_q)  # (B, input_dim)

        return z_q, indices, z_e_projected, quantized.detach()
