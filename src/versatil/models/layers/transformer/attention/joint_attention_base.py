"""Base class for dual-stream attention mechanisms.

Provides shared K/V concatenation, GQA expansion, dual SDPA, mask
building, and tensor reshaping used by both full and precomputed
dual-stream attention variants.

References:
    Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    https://arxiv.org/abs/2403.03206
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointAttentionBase(nn.Module):
    """Shared dual-stream attention computation.

    Subclasses handle projection (full vs precomputed primary) and call
    ``_joint_sdpa`` with the projected Q/K/V tensors.

    Shape notation:
        B: batch size
        S: primary sequence length
        T: secondary sequence length
        H: number of query heads
        KV_H: number of key/value heads
        D_head: per-head dimension
    """

    def __init__(
        self,
        number_of_heads: int,
        number_of_key_value_heads: int,
        head_dimension: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if number_of_heads <= 0:
            raise ValueError(
                f"number_of_heads must be positive, got {number_of_heads}."
            )
        if number_of_key_value_heads <= 0:
            raise ValueError(
                "number_of_key_value_heads must be positive, "
                f"got {number_of_key_value_heads}."
            )
        if number_of_heads % number_of_key_value_heads != 0:
            raise ValueError(
                f"number_of_heads ({number_of_heads}) must be divisible by "
                f"number_of_key_value_heads ({number_of_key_value_heads})."
            )
        self.number_of_heads = number_of_heads
        self.number_of_key_value_heads = number_of_key_value_heads
        self.head_dimension = head_dimension
        self.group_size = number_of_heads // number_of_key_value_heads
        self.dropout = dropout

    def _reshape_for_query(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape projected query tensor to (B, H, S, D_head)."""
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        ).transpose(1, 2)

    def _reshape_for_key_value(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape projected key/value tensor to (B, KV_H, S, D_head)."""
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            self.number_of_key_value_heads,
            self.head_dimension,
        ).transpose(1, 2)

    def _joint_sdpa(
        self,
        query_primary: torch.Tensor,
        key_primary: torch.Tensor,
        value_primary: torch.Tensor,
        query_secondary: torch.Tensor,
        key_secondary: torch.Tensor,
        value_secondary: torch.Tensor,
        sequence_length_primary: int,
        sequence_length_secondary: int,
        attention_mask_primary: torch.Tensor | None = None,
        attention_mask_secondary: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate K/V, GQA expand, run dual SDPA, reshape outputs.

        Args:
            query_primary: Primary queries (B, H, S, D_head).
            key_primary: Primary keys (B, KV_H, S, D_head).
            value_primary: Primary values (B, KV_H, S, D_head).
            query_secondary: Secondary queries (B, H, T, D_head).
            key_secondary: Secondary keys (B, KV_H, T, D_head).
            value_secondary: Secondary values (B, KV_H, T, D_head).
            sequence_length_primary: S.
            sequence_length_secondary: T.
            attention_mask_primary: Per-stream padding mask (B, S), True = masked.
            attention_mask_secondary: Per-stream padding mask (B, T), True = masked.
            joint_attention_mask: Pre-built joint mask (B, 1, S+T, S+T).

        Returns:
            Tuple of raw attention outputs (B, S, H*D_head) and (B, T, H*D_head)
            before output projection.
        """
        key_joint = torch.cat(
            [key_primary, key_secondary], dim=2
        )  # (B, KV_H, S+T, D_head)
        value_joint = torch.cat(
            [value_primary, value_secondary], dim=2
        )  # (B, KV_H, S+T, D_head)
        if self.group_size > 1:
            key_joint = torch.repeat_interleave(
                key_joint, self.group_size, dim=1
            )  # (B, H, S+T, D_head)
            value_joint = torch.repeat_interleave(
                value_joint, self.group_size, dim=1
            )  # (B, H, S+T, D_head)

        if joint_attention_mask is not None:
            sdpa_mask_primary = ~joint_attention_mask[:, :, :sequence_length_primary, :]
            sdpa_mask_secondary = ~joint_attention_mask[
                :, :, sequence_length_primary:, :
            ]
        else:
            resolved_mask = self._build_joint_attention_mask(
                mask_primary=attention_mask_primary,
                mask_secondary=attention_mask_secondary,
                sequence_length_primary=sequence_length_primary,
                sequence_length_secondary=sequence_length_secondary,
                device=query_primary.device,
            )
            # Broadcast mask (B, 1, 1, S+T) — works for both streams
            sdpa_mask_primary = ~resolved_mask if resolved_mask is not None else None
            sdpa_mask_secondary = sdpa_mask_primary

        attention_output_primary = F.scaled_dot_product_attention(
            query=query_primary,
            key=key_joint,
            value=value_joint,
            attn_mask=sdpa_mask_primary,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, S, D_head)
        attention_output_secondary = F.scaled_dot_product_attention(
            query=query_secondary,
            key=key_joint,
            value=value_joint,
            attn_mask=sdpa_mask_secondary,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, T, D_head)

        batch_size = query_primary.shape[0]
        query_dimension = self.number_of_heads * self.head_dimension
        attention_output_primary = (
            attention_output_primary.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length_primary, query_dimension)
        )  # (B, S, H*D_head)
        attention_output_secondary = (
            attention_output_secondary.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length_secondary, query_dimension)
        )  # (B, T, H*D_head)
        return attention_output_primary, attention_output_secondary

    @staticmethod
    def _build_joint_attention_mask(
        mask_primary: torch.Tensor | None,
        mask_secondary: torch.Tensor | None,
        sequence_length_primary: int,
        sequence_length_secondary: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build combined attention mask for joint key-value sequence.

        Args:
            mask_primary: Padding mask for primary stream (B, S), True = masked.
            mask_secondary: Padding mask for secondary stream (B, T), True = masked.
            sequence_length_primary: Length of primary sequence.
            sequence_length_secondary: Length of secondary sequence.
            device: Device for the mask tensor.

        Returns:
            Joint attention mask (B, 1, 1, S+T), or None if no masks provided.
        """
        if mask_primary is None and mask_secondary is None:
            return None
        batch_size = (
            mask_primary.shape[0]
            if mask_primary is not None
            else mask_secondary.shape[0]
        )
        if mask_primary is None:
            mask_primary = torch.zeros(
                batch_size, sequence_length_primary, dtype=torch.bool, device=device
            )
        if mask_secondary is None:
            mask_secondary = torch.zeros(
                batch_size, sequence_length_secondary, dtype=torch.bool, device=device
            )
        expected_shapes = {
            "primary": (mask_primary, sequence_length_primary),
            "secondary": (mask_secondary, sequence_length_secondary),
        }
        for stream, (mask, expected_length) in expected_shapes.items():
            if mask.shape != (batch_size, expected_length):
                raise ValueError(
                    f"Joint attention {stream} mask has shape "
                    f"{tuple(mask.shape)}, expected "
                    f"({batch_size}, {expected_length}); a mismatched mask "
                    "would silently mask the wrong positions."
                )
        joint_mask = torch.cat([mask_primary, mask_secondary], dim=1)  # (B, S+T)
        return joint_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S+T)
