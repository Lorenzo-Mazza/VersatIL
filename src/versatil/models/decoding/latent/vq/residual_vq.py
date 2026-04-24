"""Residual vector quantization with cascading codebook layers.

Each layer quantizes the residual left by previous layers, producing a
hierarchical discrete representation. Coarse layers capture the dominant
structure, fine layers capture residual detail.
"""

import torch
from torch import nn

from versatil.models.decoding.latent.vq.vector_quantize import VectorQuantize


class ResidualVQ(nn.Module):
    """Multi-layer residual vector quantizer.

    Cascades N VectorQuantize layers. Each layer quantizes the residual
    from the previous layer. The final quantized output is the sum of
    all layers' contributions.

    Args:
        input_dim: Dimension of the input vectors.
        code_dim: Dimension of each codebook vector per layer.
        num_codes: Number of codebook entries per layer (K).
        num_layers: Number of residual VQ layers.
        ema_decay: EMA decay for codebook updates in each layer.
        dead_code_threshold: Dead code replacement threshold per layer.
        kmeans_init: Initialize each layer's codebook from data.
    """

    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        num_codes: int,
        num_layers: int = 1,
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
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                VectorQuantize(
                    input_dim=input_dim,
                    code_dim=code_dim,
                    num_codes=num_codes,
                    ema_decay=ema_decay,
                    dead_code_threshold=dead_code_threshold,
                    kmeans_init=kmeans_init,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Quantize input through cascading residual layers.

        Args:
            z_e: Encoder output, shape (B, input_dim).

        Returns:
            Tuple of:
                z_q: Sum of all layers' quantized outputs, shape (B, input_dim).
                all_indices: List of per-layer codebook indices, each shape (B,).
                z_e_per_layer: Per-layer pre-quantization encoder outputs in
                    code space, stacked along dim 0, shape (L, B, code_dim).
                    Carries gradient; used as commitment-loss target.
                z_q_per_layer: Per-layer hard-quantized codebook vectors in
                    code space, stacked along dim 0, shape (L, B, code_dim).
                    Detached; paired with z_e_per_layer for per-layer
                    commitment loss.
        """
        residual = z_e  # (B, input_dim)
        z_q_hard_total = torch.zeros_like(z_e)  # (B, input_dim)
        z_q_straight_through_total = torch.zeros_like(z_e)  # (B, input_dim)
        all_indices = []
        all_z_e_projected = []
        all_z_q_code = []

        for layer in self.layers:
            z_q_layer, indices, z_e_projected, z_q_code = layer(
                residual
            )  # (B, input_dim), (B,), (B, code_dim), (B, code_dim)
            residual = (
                residual - z_q_layer.detach()
            )  # (B, input_dim) — detach to stop gradient across layers
            z_q_hard_total = z_q_hard_total + z_q_layer.detach()  # (B, input_dim)
            z_q_straight_through_total = (
                z_q_straight_through_total + z_q_layer
            )  # (B, input_dim)
            all_indices.append(indices)
            all_z_e_projected.append(z_e_projected)
            all_z_q_code.append(z_q_code)

        z_e_per_layer = torch.stack(all_z_e_projected, dim=0)  # (L, B, code_dim)
        z_q_per_layer = torch.stack(all_z_q_code, dim=0)  # (L, B, code_dim)
        # Keep the hard RVQ sum in the forward pass, but average the per-layer
        # straight-through paths so an L-layer RVQ does not multiply encoder
        # gradients by L when the projections are identities.
        z_q_total = (
            z_q_hard_total
            + (z_q_straight_through_total - z_q_straight_through_total.detach())
            / self.num_layers
        )

        return z_q_total, all_indices, z_e_per_layer, z_q_per_layer

    def decode_from_indices(self, all_indices: list[torch.Tensor]) -> torch.Tensor:
        """Reconstruct quantized output from codebook indices.

        Used by the prior at inference to convert predicted indices into
        the quantized embedding that the decoder expects.

        Args:
            all_indices: List of per-layer codebook indices, each shape (B,).

        Returns:
            Reconstructed quantized output, shape (B, input_dim).
        """
        z_q_total = torch.zeros(
            all_indices[0].shape[0],
            self.input_dim,
            device=all_indices[0].device,
        )  # (B, input_dim)

        for layer, indices in zip(self.layers, all_indices, strict=True):
            codebook_vectors = layer.codebook.embed[indices]  # (B, code_dim)
            z_q_total = z_q_total + layer.project_out(
                codebook_vectors
            )  # (B, input_dim)

        return z_q_total
