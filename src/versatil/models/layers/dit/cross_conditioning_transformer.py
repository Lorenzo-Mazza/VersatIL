"""
Diffusion Transformer architecture from the PixArt paper.

This architecture incorporates cross-attention modules into standard Diffusion Transformer (DiT)
to inject conditioning information. Differently from the original DiT, the conditioning information
is added via cross-attention layers rather than averaging and summing it to the timestep embedding.
This allows for more fine-grained control over how the decoder attends to the conditioning information.

References:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt.py#L25
    https://arxiv.org/pdf/2310.00426
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/pixart_transformer_2d.py
"""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.dit.cross_conditioning_decoder import (
    CrossConditioningDecoder,
)
from versatil.models.layers.dit.final_prediction_layer import FinalPredictionLayer
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.positional_encoding.base import (
    DenominatorMode,
    OrderingMode,
    PositionSource,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)
from versatil.models.layers.transformer.encoder import TransformerEncoder


class CrossConditioningDiffusionTransformer(nn.Module):
    """Encoder-decoder transformer with cross-attention conditioning (PixArt-style).

    Unlike the standard DiffusionTransformer which averages encoder outputs and adds them
    to the timestep embedding, this architecture uses cross-attention to allow the decoder
    to attend to all encoder hidden states. This enables more fine-grained conditioning.

    Architecture:
        - Encoder: Bidirectional transformer (processes observation tokens)
        - Decoder: DiT decoder with cross-attention (generates action tokens)
        - Timestep conditioning: AdaNorm in self-attention and FFN blocks
        - Encoder conditioning: Cross-attention with standard LayerNorm
    """

    def __init__(
        self,
        number_of_encoder_layers: int,
        number_of_decoder_layers: int,
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
        maximum_decoder_length: int = 256,
        timestep_embedding_dimension: int = 256,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        use_gating: bool = True,
        initializer_range: float = 0.02,
    ):
        """Initialize the CrossConditioningDiffusionTransformer.

        Args:
            number_of_encoder_layers: Number of encoder layers.
            number_of_decoder_layers: Number of decoder layers.
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
            positional_encoding_type: Type of positional encoding for encoder.
            maximum_sequence_length: Maximum encoder sequence length.
            maximum_decoder_length: Maximum decoder sequence length.
            timestep_embedding_dimension: Dimension for timestep sinusoidal embedding.
            bias: Whether to use bias in linear layers.
            normalization_epsilon: Epsilon for normalization layers.
            use_gating: Whether to use gating in decoder AdaNorm (AdaLN-Zero style).
            initializer_range: Standard deviation for weight initialization.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_encoder_layers = number_of_encoder_layers
        self.number_of_decoder_layers = number_of_decoder_layers
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

        self.encoder = TransformerEncoder(
            number_of_layers=number_of_encoder_layers,
            embedding_dimension=embedding_dimension,
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
            initializer_range=initializer_range,
        )

        self.decoder = CrossConditioningDecoder(
            number_of_layers=number_of_decoder_layers,
            embedding_dimension=embedding_dimension,
            timestep_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            positional_encoding_type=positional_encoding_type,
            maximum_sequence_length=maximum_decoder_length,
            bias=bias,
            normalization_epsilon=normalization_epsilon,
            use_gating=use_gating,
            initializer_range=initializer_range,
        )

        self.output_dimension = output_dimension or embedding_dimension
        self.epsilon_network = FinalPredictionLayer(
            self.embedding_dimension, self.output_dimension
        )

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_padding_mask: torch.Tensor | None = None,
        decoder_padding_mask: torch.Tensor | None = None,
        encoder_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the transformer.

        Args:
            decoder_hidden_states: Decoder input tokens (B, T, D).
            timesteps: Diffusion timesteps (B,).
            encoder_hidden_states: Encoder input tokens (B, S, D).
            encoder_padding_mask: Padding mask for encoder (B, S).
            decoder_padding_mask: Padding mask for decoder (B, T).
            encoder_cache: Precomputed encoder outputs (B, S, D) for inference caching.

        Returns:
            Tuple of (encoder_outputs, decoder_output):
                - encoder_outputs: Full encoder hidden states (B, S, D) for caching.
                - decoder_output: Decoder output tokens (B, T, output_dimension).
        """
        if encoder_cache is None:
            encoder_outputs = self.forward_encoder(
                encoder_hidden_states, encoder_padding_mask
            )
        else:
            encoder_outputs = encoder_cache

        decoder_output = self.forward_decoder(
            decoder_hidden_states,
            timesteps,
            encoder_outputs,
            decoder_padding_mask,
            encoder_padding_mask,
        )
        return encoder_outputs, decoder_output

    def forward_encoder(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode input tokens.

        Args:
            hidden_states: Input tokens (B, S, D).
            padding_mask: Padding mask (B, S) where True means masked.

        Returns:
            Encoder hidden states (B, S, D) for cross-attention.
        """
        return self.encoder(
            hidden_states=hidden_states,
            padding_mask=padding_mask,
        )

    def forward_decoder(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        decoder_padding_mask: torch.Tensor | None = None,
        encoder_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode with timestep and encoder conditioning.

        Args:
            hidden_states: Decoder input tokens (B, T, D).
            timesteps: Timesteps (B,).
            encoder_hidden_states: Encoder outputs for cross-attention (B, S, D).
            decoder_padding_mask: Padding mask for decoder (B, T).
            encoder_padding_mask: Padding mask for encoder (B, S).

        Returns:
            Decoder output tokens (B, T, output_dimension).
        """
        timestep_embedding = self.timestep_embedding_network(timesteps)

        decoder_output = self.decoder(
            hidden_states=hidden_states,
            conditioning_embedding=timestep_embedding,
            encoder_hidden_states=encoder_hidden_states,
            decoder_padding_mask=decoder_padding_mask,
            encoder_padding_mask=encoder_padding_mask,
        )
        return self.epsilon_network(decoder_output, timestep_embedding)