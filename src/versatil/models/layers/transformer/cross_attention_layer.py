"""Cross-attention layer accepting external key/value tensors."""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.swiglu import SwiGLU
from versatil.models.layers.transformer.attention import CachedAttention


class CrossAttentionLayer(nn.Module):
    """Cross-attention to external key/value states, followed by feedforward.

    Computes queries from the input hidden states via ``CachedAttention``,
    then attends to externally provided keys and values. Does NOT include
    self-attention — use ``TransformerDecoderLayer`` when self-attention
    is also needed.

    Shape notation:
        B: batch size
        S: query sequence length
        P: key/value sequence length
        H: number of query heads
        KV_H: number of key/value heads
        D_head: per-head dimension
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int | None = None,
        head_dimension: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.GROUPED_QUERY.value,
        bias: bool = True,
    ):
        """Initialize cross-attention layer.

        Args:
            embedding_dimension: Hidden dimension of the query stream.
            number_of_heads: Number of query attention heads.
            number_of_key_value_heads: Number of key/value heads for GQA.
            head_dimension: Per-head dimension override.
            feedforward_dimension: FFN hidden dimension. Defaults to 4 * embedding_dimension.
            dropout: Dropout rate.
            activation: Activation function for FFN.
            normalization_type: Normalization layer type.
            attention_type: Attention type.
            bias: Whether to use bias in projections.
        """
        super().__init__()
        if feedforward_dimension is None:
            feedforward_dimension = 4 * embedding_dimension
        self.cross_attention = CachedAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            head_dimension=head_dimension,
            attention_type=attention_type,
            bias=bias,
        )
        self.pre_attention_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
        )
        self.pre_feedforward_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
        )
        if activation == ActivationFunction.SWIGLU.value:
            self.feedforward = nn.Sequential(
                SwiGLU(embedding_dimension, feedforward_dimension, bias=bias),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )
        else:
            activation_function = ActivationFunction(activation).to_torch_activation()()
            self.feedforward = nn.Sequential(
                nn.Linear(embedding_dimension, feedforward_dimension, bias=bias),
                activation_function,
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=bias),
            )
        self.feedforward[-1].SQUARE_ROOT_WEIGHT = True
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        precomputed_query_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Cross-attend to external key/value states.

        Args:
            hidden_states: Query tokens (B, S, D).
            keys: Key states (B, KV_H, P, D_head).
            values: Value states (B, KV_H, P, D_head).
            attention_mask: Optional mask (B, 1, S, P).
            precomputed_query_rope: Pre-computed (cos, sin) for query positions.
                Applied to queries via half-rotation after projection.

        Returns:
            Updated hidden states (B, S, D).
        """
        normalized = self.pre_attention_normalization(hidden_states)
        queries, _, _ = self.cross_attention.compute_query_key_value(
            normalized,
            normalized,
            normalized,
        )
        if precomputed_query_rope is not None:
            cos, sin = precomputed_query_rope
            queries = RotaryPositionalEncoding.apply_rotation_half(queries, sin, cos)
        attention_output = self.cross_attention.compute_attention(
            queries=queries,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        residual = hidden_states
        hidden_states = residual + self.dropout(
            self.feedforward(self.pre_feedforward_normalization(hidden_states))
        )
        return hidden_states
