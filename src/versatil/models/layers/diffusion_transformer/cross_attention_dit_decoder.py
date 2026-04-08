"""DiT decoder with cross-attention for conditioning (PixArt-style).

Uses adaptive normalization (AdaNorm+gating) for self-attention and FFN,
but plain normalization for cross-attention to encoder features.
"""

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
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)
from versatil.models.layers.transformer.decoder_layer import TransformerDecoderLayer
from versatil.models.layers.transformer.positional_encoding import (
    create_positional_encoding,
)


class CrossConditioningDecoder(nn.Module):
    """DiT decoder with cross-attention for conditioning (PixArt-style).

    Each layer uses adaptive normalization for self-attention and feedforward,
    but plain normalization for cross-attention to encoder features.
    """

    def __init__(
        self,
        number_of_layers: int,
        embedding_dimension: int,
        timestep_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 256,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        use_gating: bool = True,
        initializer_range: float = 0.02,
        cross_attention_normalization_type: str | None = None,
    ):
        """Initialize the cross-conditioning decoder.

        Args:
            number_of_layers: Number of decoder layers.
            embedding_dimension: Hidden dimension of the transformer.
            timestep_dimension: Dimension of the conditioning vector.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of K/V heads (for GQA).
            feedforward_dimension: FFN hidden dimension.
            dropout: Dropout rate.
            attention_dropout: Dropout rate for attention weights.
            activation: Activation function name (use ActivationFunction enum values).
            normalization_type: Normalization type for self-attention and FFN.
            attention_type: Type of attention (use AttentionType enum values).
            positional_encoding_type: Type of positional encoding (or None).
            maximum_sequence_length: Maximum sequence length for positional encoding.
            bias: Whether to use bias in linear layers.
            normalization_epsilon: Epsilon for normalization layers.
            use_gating: Whether to use gating in AdaNorm (AdaLN-Zero style).
            initializer_range: Standard deviation for weight initialization.
            cross_attention_normalization_type: Normalization type for cross-attention.
                Defaults to ``normalization_type`` when None.
        """
        super().__init__()
        self.number_of_layers = number_of_layers
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.maximum_sequence_length = maximum_sequence_length
        self.initializer_range = initializer_range
        if attention_type == AttentionType.GROUPED_QUERY.value:
            if number_of_key_value_heads is None:
                raise ValueError("number_of_key_value_heads required for GQA")
            self.number_of_key_value_heads = number_of_key_value_heads
        else:
            self.number_of_key_value_heads = number_of_heads
        self.head_dimension = embedding_dimension // number_of_heads
        cross_attention_normalization_type = (
            cross_attention_normalization_type or normalization_type
        )
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
                TransformerDecoderLayer(
                    embedding_dimension=embedding_dimension,
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
                    autoregressive=False,
                    conditioning_dimension=timestep_dimension,
                    use_gating=use_gating,
                    cross_attention_normalization_type=cross_attention_normalization_type,
                    cross_attention_conditioning_dimension=None,
                )
                for _ in range(number_of_layers)
            ]
        )
        self.final_normalization = create_normalization_layer(
            normalization_type=cross_attention_normalization_type,
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
        key_length: int,
    ) -> torch.Tensor:
        """Expand 2D padding mask to 4D attention mask.

        Args:
            padding_mask: (B, key_length) where True means masked/padded.
            query_length: Length of the query sequence.
            key_length: Length of the key sequence.

        Returns:
            Attention mask (B, 1, query_length, key_length) where True means masked.
        """
        return (
            padding_mask.unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, -1, query_length, key_length)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_padding_mask: torch.Tensor | None = None,
        encoder_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            hidden_states: Input tokens (B, T, D).
            conditioning_embedding: Timestep conditioning vector (B, D).
            encoder_hidden_states: Encoder output for cross-attention (B, S, D).
            decoder_padding_mask: Optional padding mask for decoder (B, T).
            encoder_padding_mask: Optional padding mask for encoder (B, S).

        Returns:
            Output hidden states (B, T, D).
        """
        decoder_sequence_length = hidden_states.shape[1]
        encoder_sequence_length = encoder_hidden_states.shape[1]
        self_attention_mask = None
        if decoder_padding_mask is not None:
            self_attention_mask = self._expand_padding_mask(
                decoder_padding_mask, decoder_sequence_length, decoder_sequence_length
            )
        cross_attention_mask = None
        if encoder_padding_mask is not None:
            cross_attention_mask = self._expand_padding_mask(
                encoder_padding_mask, decoder_sequence_length, encoder_sequence_length
            )
        if isinstance(
            self.positional_encoding,
            (SinusoidalPositionalEncoding1D, LearnedPositionalEncoding1D),
        ):
            hidden_states = hidden_states + self.positional_encoding(hidden_states)

        rotary_positional_encoding = (
            self.positional_encoding
            if isinstance(self.positional_encoding, RotaryPositionalEncoding)
            else None
        )
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states=hidden_states,
                encoded_features=encoder_hidden_states,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
                positional_encoding=rotary_positional_encoding,
                conditioning=conditioning_embedding,
            )
        hidden_states = self.final_normalization(hidden_states)
        return hidden_states
