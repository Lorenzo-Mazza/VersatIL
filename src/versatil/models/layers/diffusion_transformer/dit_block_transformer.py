""" 'DiT Block' architecture inspired by the original implementation by Peebles and Xie, adapted for robotics in
 "The Ingredients for Robotic Diffusion Transformers" by Dasari et al.

 The implementation by Dasari had several bugs/differences from the original DiT paper, which have been corrected here.

 The architecture consists of an encoder-decoder transformer where the encoder processes pooled observation features,
    and the decoder generates action tokens conditioned on both the timestep embedding and the encoder output mean.

References:
    https://arxiv.org/html/2410.10088v1
    https://github.com/SudeepDasari/dit-policy/blob/main/data4robotics/models/diffusion.py#L282
    https://arxiv.org/abs/2212.09748
    https://github.com/facebookresearch/DiT/blob/main/models.py
"""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.diffusion_transformer.dit_decoder import (
    DiffusionTransformerDecoder,
)
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
from versatil.models.layers.transformer.encoder import TransformerEncoder


class DiTBlock(nn.Module):
    """DiT-Block paper network architecture.

    The encoder processes input tokens bidirectionally and pools output to a single vector.
    The decoder generates output tokens conditioned on (timestep + pooled encoder output) via AdaLN.

    Shape notation:
        B: batch size
        S: encoder sequence length
        T: decoder sequence length
        D: embedding dimension
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
        """Initialize the DiffusionTransformer.

        Args:
            number_of_encoder_layers: Number of encoder layers.
            number_of_decoder_layers: Number of decoder layers.
            embedding_dimension: Hidden dimension of the transformer.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of Key/Values heads (for Group Query Attention).
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
            use_gating: Whether to use gating in decoder AdaNorm (often referred to as AdaLNZeroNorm).
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
        self.decoder = DiffusionTransformerDecoder(
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
            decoder_hidden_states: Decoder input tokens (batch size (B), decoder sequence length (T), embedding dimension (D)).
            timesteps: Diffusion timesteps (B,).
            encoder_hidden_states: Encoder input tokens (B, encoder sequence length (S), D).
            encoder_padding_mask: Padding mask for encoder (B, S).
            decoder_padding_mask: Padding mask for decoder (B, T).
            encoder_cache: Precomputed encoder output mean (B, D) for inference.

        Returns:
            Tuple of (encoder_output_mean, decoder_output):
                - encoder_output_mean: Mean of encoder outputs (B, D) for caching.
                - decoder_output: Decoder output tokens (B, T, self.output_dimension).
        """
        if encoder_cache is None:
            encoder_output_mean = self.forward_encoder(
                encoder_hidden_states, encoder_padding_mask
            )
        else:
            encoder_output_mean = encoder_cache

        decoder_output = self.forward_decoder(
            decoder_hidden_states,
            timesteps,
            encoder_output_mean,
            decoder_padding_mask,
        )
        return encoder_output_mean, decoder_output

    def forward_encoder(
        self,
        hidden_states: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode input tokens.

        Args:
            hidden_states: Input tokens (batch size (B), encoder sequence length (S), embedding dimension (D)).
            padding_mask: Padding mask (B, S) where True means masked.

        Returns:
            Mean of encoder outputs (B, D) for conditioning the decoder.
        """
        encoder_output = self.encoder(
            hidden_states=hidden_states,
            padding_mask=padding_mask,
        )
        encoder_output_mean = encoder_output.mean(
            dim=1
        )  # Mean over sequence length, shape (B, D)
        return encoder_output_mean

    def forward_decoder(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_output_mean: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode with timestep conditioning.

        Args:
            hidden_states: Decoder input tokens (batch size (B), decoder sequence length (T), embedding dimension (D)).
            timesteps: Timesteps (B,).
            encoder_output_mean: Mean encoder output (B, D).
            padding_mask: Padding mask (B, T).

        Returns:
            Decoder output tokens (B, T, self.output_dimension).
        """
        timestep_embedding = self.timestep_embedding_network(timesteps)  # (B, D)
        combined_conditioning = timestep_embedding + encoder_output_mean
        decoder_output = self.decoder(
            hidden_states=hidden_states,
            conditioning_embedding=combined_conditioning,
            padding_mask=padding_mask,
        )
        return self.epsilon_network(decoder_output, combined_conditioning)
