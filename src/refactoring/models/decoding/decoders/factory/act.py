"""Action Chunking Transformer (ACT) architecture for action decoding.

Reference: https://arxiv.org/abs/2304.13705
"""
import logging

import torch
from torch import nn

from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.models.decoding.action_heads import ActionHead
from refactoring.models.decoding.constants import FeatureType, LATENT_KEY, LOGVAR_KEY, MU_KEY
from refactoring.models.decoding.decoders.base import ActionDecoder, DecoderInput
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.detr_transformer import Transformer
from refactoring.models.layers.feature_projection import (
    FeatureProjection,
    SpatialFeatureConcatenator,
)
from refactoring.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding2D,
)


class ACT(ActionDecoder):
    """Action Chunking Transformer network for action decoding.

    This architecture:
    - Encodes multi-camera images into spatial features
    - Optionally accepts a latent embedding from the algorithm layer (e.g., from VAE)
    - Decodes actions using a transformer with learnable queries
    - Supports multiple action types: position, orientation, gripper

    Note: Latent action encoding is handled at the Algorithm level,
    not within this decoder. The decoder expects latent embeddings to be passed
    via the features dictionary with key LATENT_KEY.
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
            embedding_dimension: int = 256,
            number_of_heads: int = 8,
            feedforward_dimension: int = 512,
            number_of_encoder_layers: int = 6,
            number_of_decoder_layers: int = 6,
            activation: str = ActivationFunction.RELU.value,
            dropout_rate: float = 0.1,
            normalize_before: bool = False,
    ):
        """Initialize ACT-style decoder.

        Args:
            input_keys: List of feature keys expected from encoder pipeline
            action_space: Action space configuration
            observation_space: Observation space configuration
            observation_horizon: Number of observation timesteps
            prediction_horizon: Number of actions to predict
            device: Device to run the model on
            embedding_dimension: Transformer hidden dimension
            number_of_heads: Number of attention heads
            feedforward_dimension: Feedforward network dimension
            number_of_encoder_layers: Number of transformer encoder layers
            number_of_decoder_layers: Number of transformer decoder layers
            activation: Activation function name
            dropout_rate: Dropout probability
            normalize_before: Use pre-normalization

        Warns:
            If observation history is provided, since ACT only uses the most recent timestep.

        """
        decoder_input = DecoderInput(
            keys=input_keys,
            required_types=[FeatureType.SPATIAL.value],
            requires_actions=True
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
        if self.has_history:
            logging.warning("ACT does not support observation history; using only the most recent timestep.")

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.feedforward_dimension = feedforward_dimension
        self.number_of_encoder_layers = number_of_encoder_layers
        self.number_of_decoder_layers = number_of_decoder_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.normalize_before = normalize_before

        # Feature projection utilities for handling dimension mismatches
        self.spatial_feature_concatenator = SpatialFeatureConcatenator(
            target_channels=self.embedding_dimension,
            concat_dim=3,  # Concatenate along width for multi-camera
            warn_on_projection=True,
        )
        self.flat_feature_projection = FeatureProjection(
            embedding_dim=self.embedding_dimension,
            warn_on_projection=False, # Don't warn for latent features.
            raise_on_mismatch=False,
        )

        self._build_transformer_components()

        # Move all components to device (including LazyLinear layers created above)
        self.to(self.device)



    def _build_transformer_components(self):
        """Build core transformer encoder-decoder and positional encodings."""
        # Positional encoding for image features
        self.image_positional_encoding = SinusoidalPositionalEncoding2D(
            embedding_dimension=self.embedding_dimension,
            normalize=True
        )
        self.action_decoder = Transformer(
            embedding_dimension=self.embedding_dimension,
            number_of_heads=self.number_of_heads,
            number_of_encoder_layers=self.number_of_encoder_layers,
            number_of_decoder_layers=self.number_of_decoder_layers,
            activation=self.activation,
            dropout=self.dropout_rate,
            normalize_before=self.normalize_before,
            feedforward_dimension=self.feedforward_dimension,
        )
        # Learnable queries for action prediction (DETR-style)
        self.learnable_query = nn.Embedding(self.prediction_horizon, self.embedding_dimension)


    def _prepare_flat_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Extract and project flat features (proprioceptive, language, etc.).

        Uses the FeatureProjection utility to handle features with different
        dimensions. If mismatches are detected, warnings will be issued.

        Args:
            features: Dictionary of encoded features

        Returns:
            Concatenated flat features (B, total_embedding_dimension) or None if no flat features
        """
        flat_features_dict = {}
        for key, feature in features.items():
            if len(feature.shape) == 3:
                if self.has_history:
                    feature = feature[:, -1]  # Use most recent timestep
                elif feature.shape[1] == 1:
                    feature = feature.squeeze(1)  # Squeeze temporal dimension when T=1
                else:
                    raise ValueError(f"Feature {key} has temporal dimension T={feature.shape[1]}, but ACT expects single-frame observation")
            if len(feature.shape)==2:
                flat_features_dict[key] = feature
        if len(flat_features_dict) == 0:
            return None
        return self.flat_feature_projection.project_and_concatenate(
            flat_features_dict,
            concatenation_dimension=-1,
        )

    def _prepare_image_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Collect and concatenate spatial features from encoder pipeline.

        Uses the SpatialFeatureConcatenator to handle features with different
        channel dimensions. If mismatches are detected, warnings will be issued
        suggesting the user configure a SpatialProjectionFusion module in the
        encoding pipeline.

        Args:
            features: Dictionary of encoded features

        Returns:
            Concatenated spatial features (B, embedding_dimension, H, W_total)
            or (B, embedding_dimension, H, W_total)

        Raises:
            ValueError: If no spatial features are found
        """
        spatial_features_dict = {}

        for key, feature in sorted(features.items()):
            if len(feature.shape) == 5:
                if self.has_history:
                    feature = feature[:, -1]  # Use most recent timestep
                elif feature.shape[1] == 1:
                    feature = feature.squeeze(1)  # Squeeze temporal dimension when T=1
                else:
                    raise ValueError(f"Feature {key} has temporal dimension T={feature.shape[1]}, but ACT expects single-frame observation")
            if len(feature.shape) == 4:  # Spatial features (B, C, H, W)
                spatial_features_dict[key] = feature
        if len(spatial_features_dict) == 0:
            raise ValueError(
                f"No spatial features found. Available keys: {list(features.keys())}"
            )
        elif len(spatial_features_dict) == 1:
            # Single spatial feature - still need to project if channel dimension mismatches
            feature_name, feature_tensor = list(spatial_features_dict.items())[0]
            return self.spatial_feature_concatenator({feature_name: feature_tensor})  # type: ignore[no-any-return]
        else:
            # Multiple spatial features - concatenate with automatic projection
            return self.spatial_feature_concatenator(spatial_features_dict)  # type: ignore[no-any-return]

    def _prepare_encoder_input(
            self,
            spatial_features: torch.Tensor,
            latent_embedding: torch.Tensor | None,
            batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare transformer encoder input from spatial features and optional latent.

        Args:
            spatial_features: Concatenated spatial features (B, C, H, W)
            latent_embedding: Optional VAE latent embedding (B, embedding_dimension)
            batch_size: Batch size

        Returns:
            Tuple of (encoder_input, positional_encoding)
            - encoder_input: (seq_len, B, embedding_dimension)
            - positional_encoding: (seq_len, B, embedding_dimension)
        """
        _, _, height, width = spatial_features.shape
        positional_encoding = self.image_positional_encoding(
            torch.zeros(1, 1, height, width, device=self.device)
        )[0] # (embedding_dimension, H, W)
        # Flatten spatial features: (B, C, H, W) -> (H*W, B, C)
        flattened_features = spatial_features.flatten(2).permute(2, 0, 1)
        # Flatten positional encoding and repeat for batch
        positional_encoding_flat = (
            positional_encoding.flatten(1)
            .permute(1, 0)
            .unsqueeze(1)
            .repeat(1, batch_size, 1)
        )
        # Optionally prepend token for latent variable z if using VAE
        if latent_embedding is not None:
            latent_token = latent_embedding.unsqueeze(0)  # (1, B, embedding_dimension)
            encoder_input = torch.cat([latent_token, flattened_features], dim=0)

            # Create positional encoding for latent token (zeros)
            latent_pos = torch.zeros(
                1, batch_size, self.embedding_dimension, device=self.device
            )
            positional_encoding_with_latent = torch.cat(
                [latent_pos, positional_encoding_flat], dim=0
            )
            return encoder_input, positional_encoding_with_latent
        else:
            # No VAE: use spatial features directly
            return flattened_features, positional_encoding_flat

    def _decode_actions(
            self,
            encoder_input: torch.Tensor,
            positional_encoding: torch.Tensor,
            batch_size: int,
    ) -> torch.Tensor:
        """Run transformer decoder to predict action sequence.

        Args:
            encoder_input: Encoder memory (seq_len, B, embedding_dimension)
            positional_encoding: Positional encodings (seq_len, B, embedding_dimension)
            batch_size: Batch size

        Returns:
            Action embeddings (B, prediction_horizon, embedding_dimension)
        """
        # Repeat learnable queries per batch element
        queries = self.learnable_query.weight.unsqueeze(1).repeat(1, batch_size, 1) # (horizon, B, embedding_dimension)
        decoder_output = self.action_decoder(
            source=encoder_input,
            target=queries,
            source_positional_encoding=positional_encoding
        )  # (1, horizon, B, embedding_dimension)
        # Take first element to get (horizon, B, embedding_dimension)
        decoder_output = decoder_output[0]
        # Transpose to (B, horizon, embedding_dimension)
        return decoder_output.permute(1, 0, 2)  # type: ignore[no-any-return]


    def _apply_action_heads(self, action_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply modular prediction heads to action embeddings.

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
            actions: dict[str, torch.Tensor] | None = None
    ) -> dict[str, torch.Tensor]:
        """Forward pass of ACT architecture.

        Args:
            features: Dictionary of encoded features from EncodingPipeline
                Expected to contain spatial features (B, C, H, W) or (B, T, C, H, W)
                May optionally contain LATENT_KEY from algorithm's latent encoder
            actions: Optional ground-truth actions for training (passed to transformer)

        Returns:
            Dictionary containing:
                - Action head predictions (e.g. position, orientation, gripper)
                - Preserves any latent-related keys (mu, logvar) from features

        Note:
            If LATENT_KEY is present in features, it will be used as the latent embedding
            token prepended to the transformer encoder input.
        """
        # Determine batch size
        batch_size = list(features.values())[0].shape[0]

        # Extract latent embedding if provided by algorithm
        latent_embedding = features.get(LATENT_KEY, None)

        # Prepare spatial and flat features (excluding latent-related keys)
        observation_features = {
            k: v for k, v in features.items()
            if k not in {LATENT_KEY, MU_KEY, LOGVAR_KEY}
        }
        spatial_features = self._prepare_image_features(observation_features)

        encoder_input, positional_encoding = self._prepare_encoder_input(
            spatial_features, latent_embedding, batch_size
        )
        action_embeddings = self._decode_actions(
            encoder_input, positional_encoding, batch_size
        )
        predictions = self._apply_action_heads(action_embeddings)

        # Preserve latent-related outputs from algorithm (e.g., mu, logvar for loss computation)
        for key in [MU_KEY, LOGVAR_KEY]:
            if key in features:
                predictions[key] = features[key]

        return predictions
