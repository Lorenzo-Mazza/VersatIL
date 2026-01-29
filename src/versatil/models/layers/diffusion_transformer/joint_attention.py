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
        S: observation sequence length
        T: action sequence length
        D: embedding dimension
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        dropout: float = 0.0,
        use_query_key_norm: bool = True,
        normalization_epsilon: float = 1e-6,
        bias: bool = True,
    ):
        """Initialize JointAttention.

        Args:
            embedding_dimension: Hidden dimension for both streams.
            number_of_heads: Number of attention heads.
            dropout: Dropout rate for attention weights.
            use_query_key_norm: Whether to apply QK-normalization.
            normalization_epsilon: Epsilon for normalization layers.
            bias: Whether to use bias in projections.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = embedding_dimension // number_of_heads
        self.dropout = dropout
        self.use_query_key_norm = use_query_key_norm
        self.query_projection_observation = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.key_projection_observation = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.value_projection_observation = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.output_projection_observation = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.query_projection_action = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.key_projection_action = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.value_projection_action = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.output_projection_action = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.output_projection_observation.SQUARE_ROOT_WEIGHT = True
        self.output_projection_action.SQUARE_ROOT_WEIGHT = True

        if use_query_key_norm:
            self.query_key_norm_observation = QueryKeyNorm(self.head_dimension, epsilon=normalization_epsilon)
            self.query_key_norm_action = QueryKeyNorm(self.head_dimension, epsilon=normalization_epsilon)

    def _reshape_for_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention.

        Args:
            tensor: Input tensor (B, S, D).

        Returns:
            Reshaped tensor (B, num_heads, S, head_dimension).
        """
        batch_size, sequence_length, _ = tensor.shape
        tensor = tensor.view(batch_size, sequence_length, self.number_of_heads, self.head_dimension)
        return tensor.transpose(1, 2)

    def _reshape_from_attention(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape tensor from multi-head attention back to sequence.

        Args:
            tensor: Input tensor (B, num_heads, S, head_dimension).

        Returns:
            Reshaped tensor (B, S, D).
        """
        batch_size, _, sequence_length, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, sequence_length, self.embedding_dimension)

    def forward(
        self,
        hidden_states_observation: torch.Tensor,
        hidden_states_action: torch.Tensor,
        attention_mask_observation: torch.Tensor | None = None,
        attention_mask_action: torch.Tensor | None = None,
        positional_encoding_observation: RotaryPositionalEncoding | None = None,
        positional_encoding_action: RotaryPositionalEncoding | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute joint attention for both streams.

        Args:
            hidden_states_observation: Observation tokens (B, S, D).
            hidden_states_action: Action tokens (B, T, D).
            attention_mask_observation: Padding mask for observations (B, S) where True = masked.
            attention_mask_action: Padding mask for actions (B, T) where True = masked.
            positional_encoding_observation: Optional RoPE for observation stream.
            positional_encoding_action: Optional RoPE for action stream.

        Returns:
            Tuple of (observation_output, action_output) with same shapes as inputs.
        """
        query_observation = self._reshape_for_attention(self.query_projection_observation(hidden_states_observation)) # (B, num_heads, S, head_dimension)
        key_observation = self._reshape_for_attention(self.key_projection_observation(hidden_states_observation)) # (B, num_heads, S, head_dimension)
        value_observation = self._reshape_for_attention(self.value_projection_observation(hidden_states_observation)) # (B, num_heads, S, head_dimension)
        query_action = self._reshape_for_attention(self.query_projection_action(hidden_states_action)) # (B, num_heads, T, head_dimension)
        key_action = self._reshape_for_attention(self.key_projection_action(hidden_states_action)) # (B, num_heads, T, head_dimension)
        value_action = self._reshape_for_attention(self.value_projection_action(hidden_states_action)) # (B, num_heads, T, head_dimension)

        if positional_encoding_observation is not None:
            query_observation, key_observation = apply_rope_positional_encoding(
                queries=query_observation,
                keys=key_observation,
                positional_encoding=positional_encoding_observation,
                cache_position=0,
            )

        if positional_encoding_action is not None:
            query_action, key_action = apply_rope_positional_encoding(
                queries=query_action,
                keys=key_action,
                positional_encoding=positional_encoding_action,
                cache_position=0,
            )

        if self.use_query_key_norm:
            query_observation, key_observation = self.query_key_norm_observation(query_observation, key_observation)
            query_action, key_action = self.query_key_norm_action(query_action, key_action)

        key_joint = torch.cat([key_observation, key_action], dim=2)  # (B, num_heads, S+T, head_dimension)
        value_joint = torch.cat([value_observation, value_action], dim=2)  # (B, num_heads, S+T, head_dimension)
        sequence_length_observation = hidden_states_observation.shape[1]
        sequence_length_action = hidden_states_action.shape[1]
        joint_attention_mask = self._build_joint_attention_mask(
            mask_observation=attention_mask_observation,
            mask_action=attention_mask_action,
            sequence_length_observation=sequence_length_observation,
            sequence_length_action=sequence_length_action,
            device=hidden_states_observation.device,
        )
        attention_output_observation = F.scaled_dot_product_attention(
            query=query_observation,
            key=key_joint,
            value=value_joint,
            attn_mask=~joint_attention_mask if joint_attention_mask is not None else None,
            dropout_p=self.dropout if self.training else 0.0,
        ) # (B, num_heads, S, head_dimension)
        attention_output_action = F.scaled_dot_product_attention(
            query=query_action,
            key=key_joint,
            value=value_joint,
            attn_mask=~joint_attention_mask if joint_attention_mask is not None else None,
            dropout_p=self.dropout if self.training else 0.0,
        ) # (B, num_heads, T, head_dimension)
        attention_output_observation = self._reshape_from_attention(attention_output_observation) # (B, S, D)
        attention_output_action = self._reshape_from_attention(attention_output_action) # (B, T, D)
        output_observation = self.output_projection_observation(attention_output_observation)
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
        batch_size = mask_observation.shape[0] if mask_observation is not None else mask_action.shape[0]
        if mask_observation is None:
            mask_observation = torch.zeros(batch_size, sequence_length_observation, dtype=torch.bool, device=device)
        if mask_action is None:
            mask_action = torch.zeros(batch_size, sequence_length_action, dtype=torch.bool, device=device)
        joint_mask = torch.cat([mask_observation, mask_action], dim=1) # (B, S+T)
        return joint_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, S+T)