"""Latent Action Transformer (LACT) architecture for action decoding.

LACT is an Action Transformer with latent-conditioned decoding via AdaLN or FiLM.
"""

import torch
from torch import nn

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.constants import FeatureType, LATENT_KEY
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, ConditioningType
from versatil.models.layers.modulation.conditional_modulation import ConditionalModulation
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding2D,
)
from versatil.models.layers.transformer.conditional_bidirectional_decoder import (
    ConditionalBidirectionalDecoder,
)
from versatil.models.layers.transformer_input_builder import TransformerInputBuilder


class LACT(ActionDecoder):
    """Latent ACtion Transformer for generative action decoding.

    Forward pass steps:
        Build observation tokens from spatial/flat features
        Condition learnable queries with latent via AdaLN/FiLM
        Decode actions using conditional transformer with latent modulation at each layer
        Apply action heads to produce predictions
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
        latent_dimension: int,
        embedding_dimension: int = 256,
        number_of_heads: int = 8,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        number_of_layers: int = 6,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        conditioning_type: str = ConditioningType.ADALN.value,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        condition_queries: bool = True,
        modulation_init_strategy: str = "identity",
    ):
        """Initialize LACT decoder.

        Args:
            input_keys: List of feature keys expected from encoder pipeline
            action_space: Action space configuration
            action_heads: Dictionary of action prediction heads
            observation_space: Observation space configuration
            observation_horizon: Number of observation timesteps
            prediction_horizon: Number of actions to predict
            device: Device to run the model on
            latent_dimension: Dimension of latent conditioning vector
            embedding_dimension: Transformer hidden dimension
            number_of_heads: Number of attention heads
            number_of_key_value_heads: Number of K/V heads for GQA (None for MHA)
            feedforward_dimension: FFN hidden dimension (default: 4 * embedding_dimension)
            number_of_layers: Number of conditional transformer decoder layers
            activation: Activation function name
            normalization_type: Type of normalization layer
            attention_type: Type of attention mechanism
            conditioning_type: Type of conditioning - "adaln" or "film"
            dropout_rate: Dropout probability for residual connections
            attention_dropout: Dropout probability for attention weights
            condition_queries: Whether to condition learnable queries before transformer
            modulation_init_strategy: Initialization for modulation layers
        """
        decoder_input = DecoderInput(
            keys=input_keys,
            required_types=[FeatureType.SPATIAL.value],
            requires_actions=False,
            conditioning_key=LATENT_KEY,
            conditioning_required=[LATENT_KEY],
        )
        super().__init__(
            decoder_input=decoder_input,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            device=device,
        )
        self.embedding_dimension = embedding_dimension
        self.latent_dimension = latent_dimension
        self.number_of_layers = number_of_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.feedforward_dimension = feedforward_dimension
        self.number_of_heads = number_of_heads
        self.number_of_key_value_heads = number_of_key_value_heads
        self.normalization_type = normalization_type
        self.attention_type = attention_type
        self.attention_dropout = attention_dropout
        self.conditioning_type = conditioning_type
        self.condition_queries = condition_queries
        self.modulation_init_strategy = modulation_init_strategy
        self._build_components()
        self.to(self.device)

    def _build_components(self):
        """Build LACT components."""
        image_positional_encoding = SinusoidalPositionalEncoding2D(
            embedding_dimension=self.embedding_dimension, normalize=True
        )
        temporal_positional_encoding = None
        if self.observation_horizon > 1:
            temporal_positional_encoding = LearnedPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension
            )
        self.input_sequence_builder = TransformerInputBuilder(
            embedding_dim=self.embedding_dimension,
            has_time_dim=self.observation_horizon > 1,
            spatial_positional_encoding_layer=image_positional_encoding,
            flat_positional_encoding_layer=LearnedPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension,
            ),
            temporal_positional_encoding_layer=temporal_positional_encoding,
            exclude_keys=[LATENT_KEY],  # Don't include latent as observation token
        )
        self.learnable_query = nn.Embedding(
            self.prediction_horizon, self.embedding_dimension
        )
        if self.condition_queries:
            self.query_modulation = ConditionalModulation(
                condition_dim=self.latent_dimension,
                feature_dim=self.embedding_dimension,
                use_shift=True,
                init_strategy=self.modulation_init_strategy,
            )
        else:
            self.query_modulation = None

        self.action_decoder = ConditionalBidirectionalDecoder(
            number_of_layers=self.number_of_layers,
            embedding_dimension=self.embedding_dimension,
            condition_dimension=self.latent_dimension,
            number_of_heads=self.number_of_heads,
            number_of_key_value_heads=self.number_of_key_value_heads,
            feedforward_dimension=self.feedforward_dimension,
            dropout=self.dropout_rate,
            attention_dropout=self.attention_dropout,
            activation=self.activation,
            normalization_type=self.normalization_type,
            attention_type=self.attention_type,
            conditioning_type=self.conditioning_type,
            modulation_init_strategy=self.modulation_init_strategy,
        )

    def _apply_action_heads(
        self, action_embeddings: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Apply prediction heads to action embeddings.

        Args:
            action_embeddings: Action embeddings (B, horizon, embedding_dimension)

        Returns:
            Dictionary of predicted actions
        """
        predictions = {}
        for action_key, head in self.action_heads.items():
            predictions[action_key] = head(action_embeddings)
        return predictions

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of LACT architecture.

        Args:
            features: Dictionary of encoded features from EncodingPipeline.
                Must contain LATENT_KEY with shape (B, latent_dimension).
            actions: Not used, present for API compatibility.

        Returns:
            Dictionary containing action head predictions (e.g. position, orientation, gripper)

        Raises:
            ValueError: If LATENT_KEY is not present in features
        """
        if LATENT_KEY not in features:
            raise ValueError(
                f"LACT requires '{LATENT_KEY}' in features. "
                f"Make sure to use a variational algorithm that provides latent embeddings. "
                f"Available features: {list(features.keys())}"
            )
        latent = features[LATENT_KEY]  # (B, latent_dim)
        obs_tokens, obs_pos_encodings, obs_padding_mask = self.input_sequence_builder(
            features
        )
        obs_tokens = obs_tokens + obs_pos_encodings
        batch_size = obs_tokens.shape[0]
        query = self.learnable_query.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (B, pred_horizon, embedding_dim)
        if self.query_modulation is not None:
            query = self.query_modulation(query, latent)
        action_embeddings = self.action_decoder(
            hidden_states=query,
            condition=latent,
            encoded_features=obs_tokens,
            query_padding_mask=None,
            memory_padding_mask=obs_padding_mask,
        )
        predictions = self._apply_action_heads(action_embeddings)
        return predictions