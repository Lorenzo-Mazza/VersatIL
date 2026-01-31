"""DiT Block action transformer decoder with pooled conditioning.

Uses DiTBlock Policy architecture, a diffusion transformer with an encoder that pools encoder output to a single conditioning vector.
Supports encoder caching for inference optimization.
"""

import logging
from typing import Optional

import torch
from torch import nn

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.constants import FeatureType
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.base import DecoderInput, ActionDecoder
from versatil.models.layers import MLP
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.diffusion_transformer import DiTBlock
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
)
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder


class DiTBlockActionTransformer(ActionDecoder):
    """Diffusion action transformer decoder using DiTBlock with pooled conditioning.

    This architecture:
    - Processes observation tokens through encoder with mean pooling
    - Conditions decoder via the sum of pooled vector + timestep embedding (AdaLN)
    - Caches pooled encoder output during inference

    """

    def __init__(
        self,
        input_keys: list[str],
        action_space: ActionSpace,
        action_heads: dict[str, ActionHead],
        observation_space: ObservationSpace,
        observation_horizon: int,
        prediction_horizon: int,
        device: str,
        max_sequence_length: int = 1024,
        embedding_dimension: int = 512,
        timestep_embedding_dimension: int = 256,
        number_of_heads: int = 8,
        number_of_key_value_heads: int | None = None,
        number_of_encoder_layers: int = 6,
        number_of_decoder_layers: int = 6,
        feedforward_dimension: int = 2048,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        positional_encoding_type: str | None = PositionalEncodingType.ROPE.value,
        use_gating: bool = True,
    ):
        """Initialize DiTBlock action decoder.

        Args:
            input_keys: List of feature keys expected from encoder pipeline.
            action_space: Action space configuration.
            action_heads: Dictionary of action head modules.
            observation_space: Observation space configuration.
            observation_horizon: Number of observation timesteps (for history).
            prediction_horizon: Number of actions to predict (horizon).
            device: Device to run the model on.
            max_sequence_length: Maximum sequence length for input tokens.
            embedding_dimension: Transformer hidden dimension.
            timestep_embedding_dimension: Diffusion timestep embedding dimension.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of K/V heads for GQA.
            number_of_encoder_layers: Number of transformer encoder layers.
            number_of_decoder_layers: Number of transformer decoder layers.
            feedforward_dimension: Feedforward network dimension.
            activation: Activation function name.
            normalization_type: Normalization type name.
            attention_type: Attention type name (gqa, mha).
            dropout_rate: Dropout probability for residual connections.
            attention_dropout: Dropout probability for attention weights.
            positional_encoding_type: Type of positional encoding.
            use_gating: Whether to use gating in AdaLN-Zero layers.
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.timestep_embedding_dimension = timestep_embedding_dimension
        self.number_of_heads = number_of_heads
        self.number_of_key_value_heads = number_of_key_value_heads or number_of_heads
        self.number_of_encoder_layers = number_of_encoder_layers
        self.number_of_decoder_layers = number_of_decoder_layers
        self.feedforward_dimension = feedforward_dimension or (4 * embedding_dimension)
        self.activation = activation
        self.normalization_type = normalization_type
        self.attention_type = attention_type
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.positional_encoding_type = positional_encoding_type
        self.use_gating = use_gating

        decoder_input = DecoderInput(
            keys=input_keys,
            raises_for_types=[FeatureType.SPATIAL.value],
            requires_actions=True,
        )
        for k, head in action_heads.items():
            if len(head.blocks) > 0:
                logging.warning(
                    f"Action heads are ignored by DiTBlockActionTransformer, but one was provided for action '{k}'. Skipping."
                )
                action_heads[k].blocks = nn.ModuleList()

        super().__init__(
            decoder_input=decoder_input,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            device=device,
        )
        self._build_transformer_components()

    def _build_transformer_components(self):
        """Build DiTBlock transformer and input processing layers."""
        image_positional_encoding = SinusoidalPositionalEncoding2D(
            embedding_dimension=self.embedding_dimension, normalize=True
        )
        temporal_positional_encoding = None
        if self.observation_horizon > 1:
            temporal_positional_encoding = LearnedPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension
            )
        self.input_builder = TransformerInputBuilder(
            embedding_dim=self.embedding_dimension,
            has_time_dim=self.observation_horizon > 1,
            spatial_positional_encoding_layer=image_positional_encoding,
            flat_positional_encoding_layer=SinusoidalPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension
            ),
            temporal_positional_encoding_layer=temporal_positional_encoding,
        )
        self.transformer = DiTBlock(
            number_of_encoder_layers=self.number_of_encoder_layers,
            number_of_decoder_layers=self.number_of_decoder_layers,
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.number_of_heads,
            number_of_key_value_heads=self.number_of_key_value_heads,
            feedforward_dimension=self.feedforward_dimension,
            dropout=self.dropout_rate,
            attention_dropout=self.attention_dropout,
            activation=self.activation,
            normalization_type=self.normalization_type,
            attention_type=self.attention_type,
            positional_encoding_type=self.positional_encoding_type,
            maximum_sequence_length=self.max_sequence_length,
            maximum_decoder_length=self.prediction_horizon,
            timestep_embedding_dimension=self.timestep_embedding_dimension,
            use_gating=self.use_gating,
        )
        self.noisy_input_projection = MLP(
            input_dim=self.action_space.get_total_action_dim(),
            output_dim=self.embedding_dimension,
            hidden_dims=[self.embedding_dimension, self.embedding_dimension],
            activation_function=ActivationFunction(
                self.activation
            ).to_torch_activation(),
            dropout=self.dropout_rate,
        )
        self._encoder_cache: Optional[torch.Tensor] = None
        self.to(self.device)

    def _prepare_observation_tokens(
        self, features: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Prepare observation features as token sequence.

        Args:
            features: Dictionary of encoded features from the encoding pipeline.

        Returns:
            Tuple of (observation_tokens, positional_encodings, observation_padding_mask).

        Raises:
            ValueError: If no valid observation features are provided.
        """
        (
            observation_tokens,
            positional_encodings,
            observation_padding_mask,
        ) = self.input_builder(features)
        if observation_tokens is None:
            raise ValueError(
                "No valid observation features provided to DiTBlockActionTransformer"
            )
        return observation_tokens, positional_encodings, observation_padding_mask

    def reset_encoder_cache(self):
        """Reset the encoder cache used for inference optimization."""
        self._encoder_cache = None

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through DiTBlock transformer.

        Args:
            features: Dictionary of encoded features plus timestep.
            actions: Dictionary of noise-injected actions.

        Returns:
            Dictionary containing denoised predictions for each action head.

        Raises:
            ValueError: If timesteps or actions are missing.
        """
        if actions is None:
            raise ValueError(
                "DiTBlockActionTransformer requires 'actions' parameter. "
                "The algorithm should provide noisy actions during forward pass."
            )
        if DecoderOutputKey.TIMESTEP.value not in features:
            raise ValueError(
                f"Missing '{DecoderOutputKey.TIMESTEP.value}' in features dict. "
                "The algorithm should inject timesteps into features."
            )
        timesteps = features.pop(DecoderOutputKey.TIMESTEP.value)
        if len(timesteps.shape) == 2:
            timesteps = timesteps.squeeze(-1)  # (B, 1) -> (B,)
        (
            observation_tokens,
            observation_positional_encodings,
            observation_padding_mask,
        ) = self._prepare_observation_tokens(features)
        if observation_positional_encodings is not None:
            observation_tokens = observation_tokens + observation_positional_encodings
        action_tensors = []
        for action_key in sorted(actions.keys()):
            action_tensors.append(actions[action_key])
        noisy_actions = torch.cat(action_tensors, dim=-1)  # (B, T, D_action)
        noisy_embedding = self.noisy_input_projection(noisy_actions)  # (B, T, D)
        if self.training:
            self._encoder_cache = None

        encoder_cache, noise_predictions = self.transformer(
            decoder_hidden_states=noisy_embedding,
            timesteps=timesteps,
            encoder_hidden_states=observation_tokens,
            encoder_padding_mask=observation_padding_mask,
            decoder_padding_mask=None,
            encoder_cache=self._encoder_cache if not self.training else None,
        )  # (B, S, D), (B, T, D)
        if not self.training:
            self._encoder_cache = encoder_cache
        outputs = {}
        start_index = 0
        for action_key in sorted(actions.keys()):
            head = self.action_heads[action_key]
            end_index = start_index + head.output_dim
            action_slice = noise_predictions[..., start_index:end_index]
            outputs[action_key] = action_slice
            start_index = end_index
        return outputs
