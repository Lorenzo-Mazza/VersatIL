"""MMDiT Decoder that stacks MMDiTLayer blocks.

Provides positional encoding and final normalization for both observation
and action streams processed through joint attention layers.

References:
    Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
    https://arxiv.org/abs/2403.03206
"""

import math

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.diffusion_transformer.mmdit_layer import MMDiTLayer
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.normalization.factory import create_normalization_layer
from versatil.models.layers.normalization.rms_norm import RMSNorm
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding
from versatil.models.layers.transformer.positional_encoding import (
    create_positional_encoding,
)


class MMDiTDecoder(nn.Module):
    """Multimodal Diffusion Transformer decoder.

    Stacks multiple MMDiTLayer blocks with optional positional encodings
    and final normalization for both streams.

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
        conditioning_dimension: int,
        number_of_heads: int,
        feedforward_dimension: int | None = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        normalization_epsilon: float = 1e-6,
        use_query_key_norm: bool = True,
        use_gating: bool = True,
        positional_encoding_type: str | None = None,
        maximum_sequence_length_observation: int = 1024,
        maximum_sequence_length_action: int = 256,
        bias: bool = True,
        initializer_range: float = 0.02,
    ):
        """Initialize MMDiT decoder.

        Args:
            number_of_layers: Number of MMDiT layers.
            embedding_dimension: Hidden dimension for both streams.
            conditioning_dimension: Dimension of conditioning vector.
            number_of_heads: Number of attention heads.
            feedforward_dimension: FFN hidden dimension.
            dropout: Dropout rate for residual connections.
            attention_dropout: Dropout rate for attention weights.
            activation: Activation function for FFN.
            normalization_type: Type of normalization layer.
            normalization_epsilon: Epsilon for normalization layers.
            use_query_key_norm: Whether to apply QK-normalization.
            positional_encoding_type: Type of positional encoding (sinusoidal, learned, rope).
            maximum_sequence_length_observation: Max observation sequence length.
            maximum_sequence_length_action: Max action sequence length.
            bias: Whether to use bias in linear layers.
            initializer_range: Standard deviation for weight initialization.
        """
        super().__init__()
        self.number_of_layers = number_of_layers
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.initializer_range = initializer_range
        self.positional_encoding_observation = None
        self.positional_encoding_action = None
        if positional_encoding_type is not None:
            self.positional_encoding_observation = create_positional_encoding(
                encoding_type=positional_encoding_type,
                embedding_dimension=embedding_dimension,
                maximum_length=maximum_sequence_length_observation,
                num_heads=number_of_heads,
            )
            self.positional_encoding_action = create_positional_encoding(
                encoding_type=positional_encoding_type,
                embedding_dimension=embedding_dimension,
                maximum_length=maximum_sequence_length_action,
                num_heads=number_of_heads,
            )
        self.layers = nn.ModuleList(
            [
                MMDiTLayer(
                    embedding_dimension=embedding_dimension,
                    conditioning_dimension=conditioning_dimension,
                    number_of_heads=number_of_heads,
                    feedforward_dimension=feedforward_dimension,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    activation=activation,
                    normalization_type=normalization_type,
                    normalization_epsilon=normalization_epsilon,
                    use_query_key_norm=use_query_key_norm,
                    use_gating=use_gating,
                    bias=bias,
                )
                for _ in range(number_of_layers)
            ]
        )
        self.final_normalization_observation = create_normalization_layer(
            normalization_type=normalization_type,
            dimension=embedding_dimension,
            epsilon=normalization_epsilon,
        )
        self.final_normalization_action = create_normalization_layer(
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

    def forward(
        self,
        hidden_states_observation: torch.Tensor,
        hidden_states_action: torch.Tensor,
        conditioning: torch.Tensor,
        attention_mask_observation: torch.Tensor | None = None,
        attention_mask_action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through MMDiT decoder.

        Args:
            hidden_states_observation: Observation tokens (B, S, D).
            hidden_states_action: Action tokens (B, T, D).
            conditioning: Conditioning vector (B, D).
            attention_mask_observation: Padding mask for observations (B, S).
            attention_mask_action: Padding mask for actions (B, T).

        Returns:
            Tuple of (observation_output, action_output) with same shapes.
        """
        rope_observation = None
        rope_action = None
        if self.positional_encoding_observation is not None:
            if isinstance(
                self.positional_encoding_observation, RotaryPositionalEncoding
            ):
                rope_observation = self.positional_encoding_observation
            else:
                hidden_states_observation = (
                    hidden_states_observation
                    + self.positional_encoding_observation(hidden_states_observation)
                )
        if self.positional_encoding_action is not None:
            if isinstance(self.positional_encoding_action, RotaryPositionalEncoding):
                rope_action = self.positional_encoding_action
            else:
                hidden_states_action = (
                    hidden_states_action
                    + self.positional_encoding_action(hidden_states_action)
                )
        for layer in self.layers:
            hidden_states_observation, hidden_states_action = layer(
                hidden_states_observation=hidden_states_observation,
                hidden_states_action=hidden_states_action,
                conditioning=conditioning,
                attention_mask_observation=attention_mask_observation,
                attention_mask_action=attention_mask_action,
                positional_encoding_observation=rope_observation,
                positional_encoding_action=rope_action,
            )
        hidden_states_observation = self.final_normalization_observation(
            hidden_states_observation
        )
        hidden_states_action = self.final_normalization_action(hidden_states_action)
        return hidden_states_observation, hidden_states_action
