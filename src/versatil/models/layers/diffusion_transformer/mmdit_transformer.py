"""MMDiT (Multimodal Diffusion Transformer) implementation.

References:
    Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    https://arxiv.org/abs/2403.03206
"""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
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
from versatil.models.layers.transformer.dual_stream_decoder import (
    DualStreamBidirectionalDecoder,
)


class MMDiTTransformer(nn.Module):
    """MMDiT transformer for diffusion-based action generation.


    Components:
    - Timestep embedding network
    - MMDiT decoder with joint attention between observations and action tokens
    - Final prediction layer for action output

    Shape notation:
        B: batch size
        S: observation sequence length
        T: action sequence length
        D: embedding dimension
    """

    def __init__(
        self,
        number_of_layers: int,
        embedding_dimension: int,
        number_of_heads: int,
        output_dimension: int | None = None,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        positional_encoding_type: str | None = None,
        maximum_sequence_length: int = 2048,
        maximum_decoder_length: int = 256,
        timestep_embedding_dimension: int = 256,
        use_query_key_norm: bool = True,
        use_gating: bool = True,
        bias: bool = True,
        normalization_epsilon: float = 1e-6,
        initializer_range: float = 0.02,
    ):
        """Initialize MMDiT Transformer.

        Args:
            number_of_layers: Number of MMDiT layers.
            embedding_dimension: Hidden dimension of the transformer.
            number_of_heads: Number of attention heads.
            output_dimension: Output dimension for action predictions.
            feedforward_dimension: FFN hidden dimension.
            dropout: Dropout rate.
            attention_dropout: Dropout rate for attention.
            activation: Activation function.
            normalization_type: Type of normalization.
            positional_encoding_type: Type of positional encoding.
            maximum_sequence_length: Maximum observation sequence length.
            maximum_decoder_length: Maximum action sequence length.
            timestep_embedding_dimension: Dimension for timestep sinusoidal embedding.
            use_query_key_norm: Whether to use QK-normalization in MMDiT layers.
            use_gating: Whether to use gating in AdaNorm (AdaLN-Zero).
            bias: Whether to use bias in linear layers.
            normalization_epsilon: Epsilon for normalization layers.
            initializer_range: Standard deviation for weight initialization.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
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
        self.decoder = DualStreamBidirectionalDecoder(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            conditioning_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            normalization_epsilon=normalization_epsilon,
            use_query_key_norm=use_query_key_norm,
            use_gating=use_gating,
            positional_encoding_type=positional_encoding_type,
            maximum_sequence_length_observation=maximum_sequence_length,
            maximum_sequence_length_action=maximum_decoder_length,
            bias=bias,
            initializer_range=initializer_range,
        )
        self.output_dimension = output_dimension or embedding_dimension
        self.prediction_layer = FinalPredictionLayer(
            embedding_dimension, self.output_dimension
        )

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_padding_mask: torch.Tensor | None = None,
        decoder_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the MMDiT transformer.

        Args:
            decoder_hidden_states: Noisy action tokens (B, T, D).
            timesteps: Diffusion timesteps (B,).
            encoder_hidden_states: Observation tokens (B, S, D).
            encoder_padding_mask: Padding mask for observations (B, S).
            decoder_padding_mask: Padding mask for actions (B, T).

        Returns:
            Predicted actions of shape (B, T, output_dimension).
        """
        timestep_embedding = self.timestep_embedding_network(timesteps)
        observation_output, action_output = self.decoder(
            hidden_states_observation=encoder_hidden_states,
            hidden_states_action=decoder_hidden_states,
            conditioning=timestep_embedding,
            attention_mask_observation=encoder_padding_mask,
            attention_mask_action=decoder_padding_mask,
        )
        return self.prediction_layer(action_output, timestep_embedding)
