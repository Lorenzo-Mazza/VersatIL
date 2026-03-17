"""Conditional bidirectional transformer decoder with latent conditioning."""

import math

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.normalization.rms_norm import RMSNorm
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.rotary import (
    RotaryPositionalEncoding,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)
from versatil.models.layers.transformer.conditional_decoder_layer import (
    ConditionalTransformerDecoderLayer,
)
from versatil.models.layers.transformer.positional_encoding import (
    create_positional_encoding,
)


class ConditionalBidirectionalDecoder(nn.Module):
    """Bidirectional transformer decoder with conditional modulation.

    This decoder extends the standard bidirectional decoder to accept a conditioning
    vector that modulates each transformer layer via AdaLN.

    The conditioning is applied at each layer after normalization, enabling the model
    to adapt its representations based on the conditioning signal throughout the network.
    """

    def __init__(
        self,
        number_of_layers: int,
        embedding_dimension: int,
        condition_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.GROUPED_QUERY.value,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 2048,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
        modulation_init_strategy: str = "identity",
    ):
        """Initialize conditional bidirectional decoder.

        Args:
            number_of_layers: Number of decoder layers
            embedding_dimension: Model embedding dimension
            condition_dimension: Dimension of conditioning vector (e.g., latent dim)
            number_of_heads: Number of attention heads
            number_of_key_value_heads: Number of K/V heads (for GQA)
            feedforward_dimension: FFN hidden dimension
            dropout: Dropout probability for residual connections
            attention_dropout: Dropout probability for attention weights
            activation: Activation function (use ActivationFunction enum values)
            normalization_type: Type of normalization (use NormalizationType enum values)
            attention_type: Type of attention (use AttentionType enum values)
            positional_encoding_type: Type of positional encoding (or None)
            maximum_sequence_length: Maximum sequence length for positional encoding
            bias: Whether to use bias in linear layers
            normalization_epsilon: Epsilon for normalization layers
            initializer_range: Standard deviation for weight initialization
            modulation_init_strategy: Initialization for modulation layers
        """
        super().__init__()
        self.number_of_layers = number_of_layers
        self.embedding_dimension = embedding_dimension
        self.condition_dimension = condition_dimension
        self.maximum_sequence_length = maximum_sequence_length
        self.initializer_range = initializer_range
        self.number_of_heads = number_of_heads
        if attention_type == AttentionType.GROUPED_QUERY.value:
            if number_of_key_value_heads is None:
                raise ValueError("number_of_key_value_heads required for GQA")
            self.number_of_key_value_heads = number_of_key_value_heads
        else:
            self.number_of_key_value_heads = number_of_heads

        self.head_dimension = embedding_dimension // number_of_heads
        self.positional_encoding = None
        if positional_encoding_type is not None:
            self.positional_encoding = create_positional_encoding(
                encoding_type=positional_encoding_type,
                embedding_dimension=embedding_dimension,
                maximum_length=maximum_sequence_length,
                num_heads=number_of_heads,
            )
        self.layers = nn.ModuleList(
            [
                ConditionalTransformerDecoderLayer(
                    embedding_dimension=embedding_dimension,
                    condition_dimension=condition_dimension,
                    number_of_heads=number_of_heads,
                    number_of_key_value_heads=number_of_key_value_heads,
                    feedforward_dimension=feedforward_dimension,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation=activation,
                    normalization_type=normalization_type,
                    attention_type=attention_type,
                    use_cross_attention=True,
                    bias=bias,
                    normalization_epsilon=normalization_epsilon,
                    modulation_init_strategy=modulation_init_strategy,
                )
                for _ in range(number_of_layers)
            ]
        )
        self.final_normalization = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with GPT-2 style initialization."""
        if hasattr(module, "SQUARE_ROOT_WEIGHT"):
            std = self.initializer_range / math.sqrt(3 * self.number_of_layers)
        else:
            std = self.initializer_range
        if isinstance(module, nn.Linear):
            if hasattr(module, "_is_modulation_layer") and module._is_modulation_layer:
                return
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm, AdaNorm)):
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, "weight") and module.weight is not None:
                module.weight.data.fill_(1.0)

    @staticmethod
    def _expand_padding_mask(
        padding_mask: torch.Tensor,
        query_length: int,
    ) -> torch.Tensor:
        """Expand 2D padding mask to 4D attention mask.

        Args:
            padding_mask: (B, key_length) where True means masked/padded
            query_length: Length of query sequence

        Returns:
            Attention mask (B, 1, query_length, key_length) where True means masked
        """
        return padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, query_length, -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        condition: torch.Tensor,
        encoded_features: torch.Tensor,
        query_padding_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through conditional bidirectional decoder.

        Args:
            hidden_states: Query embeddings (B, query_length, D)
            condition: Conditioning vector (B, condition_dim) - e.g., latent from VAE
            encoded_features: Encoder features to cross-attend to (B, memory_length, D)
            query_padding_mask: Optional padding mask for queries (B, query_length)
            memory_padding_mask: Optional padding mask for memory (B, memory_length)

        Returns:
            Output hidden states (B, query_length, D)
        """
        query_length = hidden_states.shape[1]
        self_attention_mask = None
        if query_padding_mask is not None:
            self_attention_mask = self._expand_padding_mask(
                query_padding_mask, query_length
            )
        cross_attention_mask = None
        if memory_padding_mask is not None:
            cross_attention_mask = self._expand_padding_mask(
                memory_padding_mask, query_length
            )
        if isinstance(
            self.positional_encoding,
            (SinusoidalPositionalEncoding1D, LearnedPositionalEncoding1D),
        ):
            hidden_states = hidden_states + self.positional_encoding(hidden_states)
        rope_pe = (
            self.positional_encoding
            if isinstance(self.positional_encoding, RotaryPositionalEncoding)
            else None
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                condition=condition,
                encoded_features=encoded_features,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
                positional_encoding=rope_pe,
            )
        hidden_states = self.final_normalization(hidden_states)
        return hidden_states
