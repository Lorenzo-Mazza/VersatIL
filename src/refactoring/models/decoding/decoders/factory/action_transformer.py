import torch
from torch import nn

from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.models.decoding.action_heads import ActionHead
from refactoring.models.decoding.constants import FeatureType
from refactoring.models.decoding.decoders import ActionDecoder, DecoderInput
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.feature_projection import FeatureProjection
from refactoring.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)


class ActionTransformer(ActionDecoder):
    """Vanilla action transformer for action decoding.

    This architecture:
    - Receives a list of flat features from the encoding/fusion block.
    - Uses fixed positional encodings.
    - Decodes the action chunks using a standard transformer decoder from torch.nn.
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
            number_of_decoder_layers: int = 6,
            activation: str = ActivationFunction.RELU.value,
            dropout_rate: float = 0.1,
            normalize_before: bool = False,
    ):
        decoder_input = DecoderInput(
            keys=input_keys,
            requires_actions=False,
            required_types=[FeatureType.FLAT.value]
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
        self.prediction_horizon = prediction_horizon
        self.embedding_projection = nn.LazyLinear(self.embedding_dimension)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dimension,
                                                        nhead=number_of_heads,
                                                        batch_first=True,
                                                        activation=activation,
                                                        dropout=dropout_rate,
                                                        norm_first=normalize_before,
                                                        dim_feedforward=feedforward_dimension
                                                        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=number_of_decoder_layers)
        self.flat_feature_projection = FeatureProjection(
            embedding_dim=self.embedding_dimension,
            warn_on_projection=True,
            raise_on_mismatch=False,
        )
        self.fixed_positional_encoding_input = SinusoidalPositionalEncoding1D(embedding_dimension=embedding_dimension,
                                                                              mlp_hidden_dimensions=None)
        self.fixed_positional_encoding = SinusoidalPositionalEncoding1D(embedding_dimension=embedding_dimension,
                                                                              mlp_hidden_dimensions=None)


    def _prepare_flat_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Extract and project flat features (proprioceptive, language, latent, etc.).

        Uses the FeatureProjection utility to handle features with different
        dimensions. If mismatches are detected, warnings will be issued.

        Args:
            features: Dictionary of encoded features

        Returns:
            Concatenated flat features (B, total_embedding_dimension) or None if no flat features

        Raises:
            ValueError: If no flat features are found
        """
        flat_features_dict = {}
        for key, feature in features.items():
            if self.has_history and len(feature.shape) == 3:
                batch_size, temporal_length, embedding_size = feature.shape
                feature = feature.reshape(batch_size * temporal_length, embedding_size)
            if len(feature.shape)==2:
                flat_features_dict[key] = feature
        if len(flat_features_dict) == 0:
            raise ValueError("No flat features found. Action Transformer requires at least 1 flat feature as input.")
        return self.flat_feature_projection.project_and_concatenate(
            flat_features_dict,
            concatenation_dimension=-1,
        )


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
        """Forward pass of the transformer decoder architecture.

        Args:
            features: Dictionary of encoded features from EncodingPipeline
                Expected to contain flat features (B, embedding_dimension) or (B, T, embedding_dimension)
            actions: Not used here

        Returns:
            Dictionary containing:
                - Action head predictions (e.g. position, orientation, gripper)
        """
        for key in features:
            if (features[key].shape == 4 and not self.has_history) or (features[key].shape == 5 and  self.has_history):
                raise ValueError("Action transformer decoder does not support spatial features."
                                 " Please flatten your features before passing them to the decoder.")

        flat_feature_vector = self._prepare_flat_features(features=features)
        flat_feature_vector = self.embedding_projection(flat_feature_vector)  # Shape: (B*T_obs if has_history else B, embedding_dimension)
        # Reshape to 3D (B, S, embedding_dimension) where S = observation_horizon or 1
        if self.has_history:
            batch_size = flat_feature_vector.size(0) // self.observation_horizon
            flat_feature_vector = flat_feature_vector.reshape(batch_size, self.observation_horizon,
                                                              self.embedding_dimension)  # Shape: (B, T_obs, embedding_dimension)
        else:
            batch_size = flat_feature_vector.size(0)
            flat_feature_vector = flat_feature_vector.unsqueeze(1)  # Shape: (B, 1, embedding_dimension)
        flat_feature_vector = flat_feature_vector.permute(1, 0, 2)  # Shape: (S, B, embedding_dimension)
        flat_feature_vector = self.fixed_positional_encoding_input(flat_feature_vector)  # Shape: (S, B, embedding_dimension)
        flat_feature_vector = flat_feature_vector.permute(1, 0, 2)
        query = self.fixed_positional_encoding(torch.zeros(self.prediction_horizon, batch_size, self.embedding_dimension, dtype=torch.float32).to(self.device))
        query = query.permute(1, 0, 2) # Shape: (B, S, embedding_dimension)
        action_embeddings = self.transformer_decoder(tgt=query, memory=flat_feature_vector)
        predictions = self._apply_action_heads(action_embeddings)
        return predictions








