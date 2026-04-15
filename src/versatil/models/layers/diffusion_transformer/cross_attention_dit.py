"""Cross-Attention 1D Diffusion Transformer (PixArt style).

DiT that conditions via cross-attention to observation tokens.

Shape notation:
    B: batch size
    S: observation sequence length (from external embeddings)
    T: action sequence length
    D: embedding dimension


References:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt.py#L25
    https://arxiv.org/pdf/2310.00426
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/pixart_transformer_2d.py
"""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.diffusion_transformer.final_prediction_layer import (
    FinalPredictionLayer,
)
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.positional_encoding.base import (
    DenominatorMode,
    OrderingMode,
    PositionSource,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)
from versatil.models.layers.transformer.cache.conditioning import ConditioningCache
from versatil.models.layers.transformer.conditional_bidirectional_decoder import (
    ConditionalBidirectionalDecoder,
)


class CrossAttentionDiT(nn.Module):
    """DiT that conditions via cross-attention (PixArt style)."""

    def __init__(
        self,
        number_of_layers: int,
        embedding_dimension: int,
        number_of_heads: int,
        output_dimension: int | None = None,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 2048,
        timestep_embedding_dimension: int = 256,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        use_gating: bool = True,
        initializer_range: float = 0.02,
    ):
        """Initialize CrossAttentionDiT.

        Args:
            number_of_layers: Number of decoder layers.
            embedding_dimension: Hidden dimension of the transformer.
            number_of_heads: Number of attention heads.
            output_dimension: Output dimension (defaults to embedding_dimension).
            number_of_key_value_heads: Number of K/V heads (for GQA).
            feedforward_dimension: Feedforward network hidden dimension.
            dropout: Dropout rate.
            attention_dropout: Dropout rate for attention.
            activation: Activation function.
            normalization_type: Type of normalization.
            attention_type: Type of attention.
            positional_encoding_type: Type of positional encoding for decoder.
            maximum_sequence_length: Maximum decoder sequence length.
            timestep_embedding_dimension: Dimension for timestep sinusoidal embedding.
            bias: Whether to use bias in linear layers.
            normalization_epsilon: Epsilon for normalization layers.
            use_gating: Whether to use gating in decoder AdaNorm (AdaLN-Zero style).
            initializer_range: Standard deviation for weight initialization.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.initializer_range = initializer_range

        if feedforward_dimension is None:
            feedforward_dimension = 4 * embedding_dimension

        self.timestep_embedding_network = SinusoidalPositionalEncoding1D(
            embedding_dimension=timestep_embedding_dimension,
            denominator_mode=DenominatorMode.HALF_MINUS_ONE.value,
            ordering_mode=OrderingMode.CAT_COS_SIN.value,
            position_source=PositionSource.SCALAR.value,
            precompute_encodings=False,
            temperature=10000.0,
            learnable_frequencies=False,
            mlp_activation=nn.SiLU,
            mlp_hidden_dimensions=[embedding_dimension, embedding_dimension],
        )

        self.decoder = ConditionalBidirectionalDecoder(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            positional_encoding_type=positional_encoding_type,
            maximum_sequence_length=maximum_sequence_length,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            use_gating=use_gating,
            initializer_range=initializer_range,
            condition_final_normalization=False,
        )

        self.output_dimension = output_dimension or embedding_dimension
        self.prediction_layer = FinalPredictionLayer(
            self.embedding_dimension, self.output_dimension
        )

    def precompute_conditioning_kv(
        self,
        encoder_hidden_states: torch.Tensor,
    ) -> ConditioningCache:
        """Precompute decoder conditioning K/V for forward pass reuse."""
        return self.decoder.precompute_conditioning_kv(
            encoded_features=encoder_hidden_states,
        )

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        conditioning_cache: ConditioningCache | None = None,
        encoder_padding_mask: torch.Tensor | None = None,
        decoder_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the transformer.

        Args:
            decoder_hidden_states: Noisy action tokens (B, T, D).
            timesteps: Diffusion timesteps (B,).
            encoder_hidden_states: External observation embeddings (B, S, D).
            conditioning_cache: Precomputed K/V for reuse across denoising steps.
                When provided, encoder_hidden_states is not needed.
            encoder_padding_mask: Padding mask for observations (B, S).
            decoder_padding_mask: Padding mask for actions (B, T).

        Returns:
            Predicted actions of shape (B, T, output_dimension).
        """
        timestep_embedding = self.timestep_embedding_network(timesteps)
        decoder_output = self.decoder(
            hidden_states=decoder_hidden_states,
            condition=timestep_embedding,
            encoded_features=encoder_hidden_states,
            conditioning_cache=conditioning_cache,
            query_padding_mask=decoder_padding_mask,
            memory_padding_mask=encoder_padding_mask,
        )
        return self.prediction_layer(decoder_output, timestep_embedding)
