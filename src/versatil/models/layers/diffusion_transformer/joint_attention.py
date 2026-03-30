"""Joint Attention mechanism for MMDiT architectures.

Implements dual-stream attention where two sequences (observations and actions)
compute attention jointly by concatenating their key-value pairs.

References:
    Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    https://arxiv.org/abs/2403.03206
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from versatil.models.layers.diffusion_transformer.query_key_norm import QueryKeyNorm
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.positional_encoding import (
    apply_rope_positional_encoding,
)


class JointAttention(nn.Module):
    """Joint attention for dual-stream processing.

    Each stream has independent Q, K, V projections. Keys and values from both
    streams are concatenated, allowing each stream to attend to both itself
    and the other stream.

    Shape notation:
        B: batch size
        S: primary (observation) sequence length
        T: secondary (action) sequence length
        D_p: primary embedding dimension
        D_s: secondary embedding dimension
        H: number of query heads
        KV_H: number of key/value heads
        D_head: per-head dimension
    """

    def __init__(
        self,
        primary_embedding_dimension: int,
        number_of_heads: int,
        secondary_embedding_dimension: int | None = None,
        number_of_key_value_heads: int | None = None,
        head_dimension: int | None = None,
        dropout: float = 0.0,
        precomputed_primary_stream: bool = False,
        use_query_key_norm: bool = True,
        normalization_epsilon: float = 1e-6,
        bias: bool = True,
    ):
        """Initialize JointAttention.

        Args:
            primary_embedding_dimension: Hidden dimension for the primary (observation) stream.
            number_of_heads: Number of query attention heads for both streams.
            secondary_embedding_dimension: Hidden dimension for the secondary (action) stream.
                Defaults to ``primary_embedding_dimension`` for symmetric streams.
            number_of_key_value_heads: Number of key/value heads for GQA.
                Defaults to ``number_of_heads`` (standard multi-head attention).
            head_dimension: Per-head dimension. Defaults to
                ``primary_embedding_dimension // number_of_heads``. Override for
                architectures where hidden_size != num_heads * head_dim.
            dropout: Dropout rate for attention weights.
            precomputed_primary_stream: When ``True``, skips creating Q/K/V
                projection layers for the primary stream. Expects
                ``precomputed_observation`` at forward time instead.
            use_query_key_norm: Whether to apply QK-normalization.
            normalization_epsilon: Epsilon for normalization layers.
            bias: Whether to use bias in projections.
        """
        super().__init__()
        secondary_embedding_dimension = (
            secondary_embedding_dimension or primary_embedding_dimension
        )
        number_of_key_value_heads = number_of_key_value_heads or number_of_heads
        head_dimension = (
            head_dimension or primary_embedding_dimension // number_of_heads
        )
        self.primary_embedding_dimension = primary_embedding_dimension
        self.secondary_embedding_dimension = secondary_embedding_dimension
        self.number_of_heads = number_of_heads
        self.number_of_key_value_heads = number_of_key_value_heads
        self.head_dimension = head_dimension
        self.group_size = number_of_heads // number_of_key_value_heads
        self.dropout = dropout
        self.precomputed_primary_stream = precomputed_primary_stream
        self.use_query_key_norm = use_query_key_norm
        query_dimension = number_of_heads * head_dimension
        key_value_dimension = number_of_key_value_heads * head_dimension
        if not precomputed_primary_stream:
            self.query_projection_observation = nn.Linear(
                primary_embedding_dimension, query_dimension, bias=bias
            )
            self.key_projection_observation = nn.Linear(
                primary_embedding_dimension, key_value_dimension, bias=bias
            )
            self.value_projection_observation = nn.Linear(
                primary_embedding_dimension, key_value_dimension, bias=bias
            )
        if not precomputed_primary_stream:
            self.output_projection_observation = nn.Linear(
                query_dimension, primary_embedding_dimension, bias=bias
            )
        self.query_projection_action = nn.Linear(
            secondary_embedding_dimension, query_dimension, bias=bias
        )
        self.key_projection_action = nn.Linear(
            secondary_embedding_dimension, key_value_dimension, bias=bias
        )
        self.value_projection_action = nn.Linear(
            secondary_embedding_dimension, key_value_dimension, bias=bias
        )
        self.output_projection_action = nn.Linear(
            query_dimension, secondary_embedding_dimension, bias=bias
        )
        if not precomputed_primary_stream:
            self.output_projection_observation.SQUARE_ROOT_WEIGHT = True
        self.output_projection_action.SQUARE_ROOT_WEIGHT = True

        if use_query_key_norm:
            if not precomputed_primary_stream:
                self.query_key_norm_observation = QueryKeyNorm(
                    head_dimension, epsilon=normalization_epsilon
                )
            self.query_key_norm_action = QueryKeyNorm(
                head_dimension, epsilon=normalization_epsilon
            )

    def _reshape_for_query(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape projected query tensor to (B, num_heads, S, head_dimension)."""
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size, sequence_length, self.number_of_heads, self.head_dimension
        ).transpose(1, 2)

    def _reshape_for_key_value(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape projected key/value tensor to (B, num_kv_heads, S, head_dimension)."""
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            self.number_of_key_value_heads,
            self.head_dimension,
        ).transpose(1, 2)

    def forward(
        self,
        hidden_states_observation: torch.Tensor,
        hidden_states_action: torch.Tensor,
        attention_mask_observation: torch.Tensor | None = None,
        attention_mask_action: torch.Tensor | None = None,
        joint_attention_mask: torch.Tensor | None = None,
        precomputed_observation: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        positional_encoding_observation: RotaryPositionalEncoding | None = None,
        positional_encoding_action: RotaryPositionalEncoding | None = None,
        precomputed_action_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute joint attention for both streams.

        Args:
            hidden_states_observation: Primary stream tokens (B, S, D_p).
            hidden_states_action: Secondary stream tokens (B, T, D_s).
            attention_mask_observation: Per-stream padding mask for primary (B, S), True = masked.
            attention_mask_action: Per-stream padding mask for secondary (B, T), True = masked.
            joint_attention_mask: Pre-built joint mask (B, 1, S+T, S+T). When provided,
                used directly instead of building from per-stream masks.
            precomputed_observation: Pre-projected primary (Q, K, V) tuple, each shaped
                (B, H/KV_H, S, D_head). Skips primary-side projections when provided.
            positional_encoding_observation: Optional RoPE for primary stream.
            positional_encoding_action: Optional RoPE for secondary stream.
            precomputed_action_rope: Pre-computed (cos, sin) for the action stream's
                positions. When provided, applied via half-rotation instead of using
                ``positional_encoding_action``. For VLA decoders that share the VLM's
                rotary embedding.

        Returns:
            Tuple of (primary_output, secondary_output) with shapes (B, S, D_p) and (B, T, D_s).
        """
        if precomputed_observation is not None:
            query_observation, key_observation, value_observation = (
                precomputed_observation
            )
        elif self.precomputed_primary_stream:
            raise ValueError(
                "precomputed_observation required when precomputed_primary_stream=True"
            )
        else:
            query_observation = self._reshape_for_query(
                self.query_projection_observation(hidden_states_observation)
            )  # (B, H, S, D_head)
            key_observation = self._reshape_for_key_value(
                self.key_projection_observation(hidden_states_observation)
            )  # (B, KV_H, S, D_head)
            value_observation = self._reshape_for_key_value(
                self.value_projection_observation(hidden_states_observation)
            )  # (B, KV_H, S, D_head)
        query_action = self._reshape_for_query(
            self.query_projection_action(hidden_states_action)
        )  # (B, H, T, D_head)
        key_action = self._reshape_for_key_value(
            self.key_projection_action(hidden_states_action)
        )  # (B, KV_H, T, D_head)
        value_action = self._reshape_for_key_value(
            self.value_projection_action(hidden_states_action)
        )  # (B, KV_H, T, D_head)

        if positional_encoding_observation is not None:
            query_observation, key_observation = apply_rope_positional_encoding(
                queries=query_observation,
                keys=key_observation,
                positional_encoding=positional_encoding_observation,
                cache_position=0,
            )

        if precomputed_action_rope is not None:
            cos_action, sin_action = precomputed_action_rope
            query_action = RotaryPositionalEncoding.apply_rotation_half(
                query_action, sin_action, cos_action
            )
            key_action = RotaryPositionalEncoding.apply_rotation_half(
                key_action, sin_action, cos_action
            )
        elif positional_encoding_action is not None:
            query_action, key_action = apply_rope_positional_encoding(
                queries=query_action,
                keys=key_action,
                positional_encoding=positional_encoding_action,
                cache_position=0,
            )

        if self.use_query_key_norm:
            if not self.precomputed_primary_stream:
                query_observation, key_observation = self.query_key_norm_observation(
                    query_observation, key_observation
                )
            query_action, key_action = self.query_key_norm_action(
                query_action, key_action
            )

        key_joint = torch.cat(
            [key_observation, key_action], dim=2
        )  # (B, KV_H, S+T, D_head)
        value_joint = torch.cat(
            [value_observation, value_action], dim=2
        )  # (B, KV_H, S+T, D_head)
        if self.group_size > 1:
            key_joint = torch.repeat_interleave(
                key_joint, self.group_size, dim=1
            )  # (B, H, S+T, D_head)
            value_joint = torch.repeat_interleave(
                value_joint, self.group_size, dim=1
            )  # (B, H, S+T, D_head)
        sequence_length_observation = hidden_states_observation.shape[1]
        sequence_length_action = hidden_states_action.shape[1]
        if joint_attention_mask is not None:
            # Per-query mask (B, 1, S+T, S+T) — slice per stream
            sdpa_mask_observation = ~joint_attention_mask[
                :, :, :sequence_length_observation, :
            ]
            sdpa_mask_action = ~joint_attention_mask[
                :, :, sequence_length_observation:, :
            ]
        else:
            resolved_mask = self._build_joint_attention_mask(
                mask_observation=attention_mask_observation,
                mask_action=attention_mask_action,
                sequence_length_observation=sequence_length_observation,
                sequence_length_action=sequence_length_action,
                device=hidden_states_observation.device,
            )
            # Broadcast mask (B, 1, 1, S+T) — works for both streams
            sdpa_mask_observation = (
                ~resolved_mask if resolved_mask is not None else None
            )
            sdpa_mask_action = sdpa_mask_observation
        attention_output_observation = F.scaled_dot_product_attention(
            query=query_observation,
            key=key_joint,
            value=value_joint,
            attn_mask=sdpa_mask_observation,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, S, D_head)
        attention_output_action = F.scaled_dot_product_attention(
            query=query_action,
            key=key_joint,
            value=value_joint,
            attn_mask=sdpa_mask_action,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, T, D_head)
        batch_size = hidden_states_observation.shape[0]
        query_dimension = self.number_of_heads * self.head_dimension
        attention_output_observation = (
            attention_output_observation.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length_observation, query_dimension)
        )  # (B, S, H * D_head)
        attention_output_action = (
            attention_output_action.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length_action, query_dimension)
        )  # (B, T, H * D_head)
        if self.precomputed_primary_stream:
            output_observation = attention_output_observation
        else:
            output_observation = self.output_projection_observation(
                attention_output_observation
            )
        output_action = self.output_projection_action(attention_output_action)
        return output_observation, output_action

    def _build_joint_attention_mask(
        self,
        mask_observation: torch.Tensor | None,
        mask_action: torch.Tensor | None,
        sequence_length_observation: int,
        sequence_length_action: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Build combined attention mask for joint key-value sequence.

        Args:
            mask_observation: Padding mask for observations (B, S) where True = masked.
            mask_action: Padding mask for actions (B, T) where True = masked.
            sequence_length_observation: Length of observation sequence.
            sequence_length_action: Length of action sequence.
            device: Device for the mask tensor.

        Returns:
            Joint attention mask (B, 1, 1, S+T) which broadcasts over the keys, or None if no masks provided.
        """
        if mask_observation is None and mask_action is None:
            return None
        batch_size = (
            mask_observation.shape[0]
            if mask_observation is not None
            else mask_action.shape[0]
        )
        if mask_observation is None:
            mask_observation = torch.zeros(
                batch_size, sequence_length_observation, dtype=torch.bool, device=device
            )
        if mask_action is None:
            mask_action = torch.zeros(
                batch_size, sequence_length_action, dtype=torch.bool, device=device
            )
        joint_mask = torch.cat([mask_observation, mask_action], dim=1)  # (B, S+T)
        return joint_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S+T)
