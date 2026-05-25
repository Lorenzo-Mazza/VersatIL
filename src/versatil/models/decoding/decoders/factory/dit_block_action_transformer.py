"""DiT Block action transformer decoder with pooled conditioning.

Uses DiTBlock Policy architecture, a diffusion transformer with an encoder that pools encoder output to a single conditioning vector.
Supports encoder caching for inference optimization.
"""

import torch

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead, ConditionalActionHead
from versatil.models.decoding.constants import ActionHeadLayout
from versatil.models.decoding.decoders.base import DecoderInput
from versatil.models.decoding.decoders.parallel_transformer import (
    BaseParallelTransformerDecoder,
)
from versatil.models.decoding.decoders.timestep_conditioning import (
    extract_timestep_conditioning,
    filter_timestep_feature,
)
from versatil.models.feature_meta import FeatureType
from versatil.models.layers import MLP
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.diffusion_transformer.dit_block_transformer import DiTBlock
from versatil.models.layers.normalization.constants import NormalizationType


class DiTBlockActionTransformer(BaseParallelTransformerDecoder):
    """Diffusion action transformer decoder using DiTBlock with pooled conditioning.

    This architecture:
    - Processes observation tokens through encoder with mean pooling
    - Conditions decoder via the sum of pooled vector + timestep embedding (AdaLN)
    - Caches pooled encoder output during inference

    """

    action_head_layout: ActionHeadLayout = ActionHeadLayout.JOINT

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
    ) -> None:
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
        self.max_sequence_length = max_sequence_length
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
        super().__init__(
            decoder_input=decoder_input,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            device=device,
            embedding_dimension=embedding_dimension,
        )
        self._validate_conditional_action_head()
        self._build_transformer_components()

    def _conditional_action_head(self) -> ConditionalActionHead:
        """Return the configured timestep-conditioned action head."""
        action_head = self._single_action_head()
        if isinstance(action_head, ConditionalActionHead):
            return action_head
        raise ValueError(
            f"{type(self).__name__} requires a ConditionalActionHead because "
            "DiT decoder hidden states are projected with timestep conditioning."
        )

    def _validate_conditional_action_head(self) -> None:
        """Validate the conditional action-head dimensions."""
        action_head = self._conditional_action_head()
        if action_head.input_dim != self.embedding_dimension:
            raise ValueError(
                f"{type(self).__name__} action head input_dim must equal "
                f"embedding_dimension {self.embedding_dimension}, got "
                f"{action_head.input_dim}."
            )
        if action_head.condition_dim != self.embedding_dimension:
            raise ValueError(
                f"{type(self).__name__} action head condition_dim must equal "
                f"embedding_dimension {self.embedding_dimension}, got "
                f"{action_head.condition_dim}."
            )

    def _build_transformer_components(self) -> None:
        """Build DiTBlock transformer and input processing layers."""
        self.input_builder = self._build_parallel_input_sequence_builder(
            flat_positional_encoding_type=PositionalEncodingType.SINUSOIDAL.value,
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
        self._encoder_cache: torch.Tensor | None = None
        self._caching_enabled: bool = False
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

    def enable_encoder_cache(self) -> None:
        """Enable encoder caching for multi-step inference loops.

        Called by algorithms at the start of predict() to enable caching
        across denoising steps within a single sample.
        """
        self._encoder_cache = None
        self._caching_enabled = True

    def disable_encoder_cache(self) -> None:
        """Disable encoder caching and clear cached values."""
        self._encoder_cache = None
        self._caching_enabled = False

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
        noisy_actions = self.action_space.concatenate_action_tensors(
            actions=actions,
            prediction_horizon=self.prediction_horizon,
            owner_name=self.__class__.__name__,
        )
        timesteps = extract_timestep_conditioning(
            features=features,
            batch_size=noisy_actions.shape[0],
            action_device=noisy_actions.device,
        )
        observation_features = filter_timestep_feature(features=features)
        (
            observation_tokens,
            observation_positional_encodings,
            observation_padding_mask,
        ) = self._prepare_observation_tokens(observation_features)
        if observation_positional_encodings is not None:
            observation_tokens = observation_tokens + observation_positional_encodings
        noisy_embedding = self.noisy_input_projection(noisy_actions)  # (B, T, D)
        encoder_cache, action_hidden, action_conditioning = (
            self.transformer.forward_features(
                decoder_hidden_states=noisy_embedding,
                timesteps=timesteps,
                encoder_hidden_states=observation_tokens,
                encoder_padding_mask=observation_padding_mask,
                decoder_padding_mask=None,
                encoder_cache=self._encoder_cache if self._caching_enabled else None,
            )
        )  # (B, D), (B, T, D), (B, D)
        if self._caching_enabled:
            self._encoder_cache = encoder_cache
        noise_predictions = self._conditional_action_head()(
            action_hidden,
            action_conditioning,
        )
        return self.action_space.split_action_tensor(
            action_tensor=noise_predictions,
            owner_name=self.__class__.__name__,
        )
