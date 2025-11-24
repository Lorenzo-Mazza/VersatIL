import torch
from torch import nn

from refactoring.data.task import ActionSpace, ObservationSpace
from refactoring.models.decoding.action_heads import ActionHead
from refactoring.models.decoding.constants import FeatureType
from refactoring.models.decoding.decoders import ActionDecoder, DecoderInput
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.positional_encoding.learned import LearnedPositionalEncoding1D
from refactoring.models.layers.positional_encoding.sinusoidal import (
 SinusoidalPositionalEncoding2D,
)
from refactoring.models.layers.transformer_input_builder import TransformerInputBuilder


class ActionTransformer(ActionDecoder):
    """Vanilla action transformer for action decoding.

    This architecture:
    - Receives observation features from the encoding/fusion block and tokenizes them into a list of tokens.
    - Uses 2D fixed positional encodings for image tokens, 1D learnable pe for sequential or flat features.
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
            activation: str = ActivationFunction.GELU.value,
            dropout_rate: float = 0.1,
            normalize_before: bool = False,
    ):
        decoder_input = DecoderInput(
            keys=input_keys,
            required_types=[FeatureType.SPATIAL.value],
            requires_actions=False
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
        self.number_of_decoder_layers = number_of_decoder_layers
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dimension,
                                                        nhead=number_of_heads,
                                                        batch_first=True,
                                                        activation=activation,
                                                        dropout=dropout_rate,
                                                        norm_first=normalize_before,
                                                        dim_feedforward=feedforward_dimension
                                                        )
        self._build_transformer_components()
        self.to(self.device)


    def _build_transformer_components(self):
        """Build core transformer encoder-decoder and positional encodings."""
        image_positional_encoding = SinusoidalPositionalEncoding2D(
            embedding_dimension=self.embedding_dimension,
            normalize=True
        )
        temporal_positional_encoding = None
        if self.observation_horizon > 1:
            temporal_positional_encoding = LearnedPositionalEncoding1D(embedding_dimension=self.embedding_dimension)
        # This layer transforms input features into a sequence of token embeddings + positional encodings
        self.input_sequence_builder = TransformerInputBuilder(
            embedding_dim=self.embedding_dimension,
            has_time_dim=self.observation_horizon > 1,
            spatial_positional_encoding_layer=image_positional_encoding,
            flat_positional_encoding_layer=LearnedPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension,
            ),
            temporal_positional_encoding_layer=temporal_positional_encoding,
        )
        self.learnable_query = nn.Embedding(self.prediction_horizon, self.embedding_dimension) # (pred_horizon, emb)
        self.action_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.number_of_decoder_layers)



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
        obs_tokens, obs_pos_encodings, obs_padding_mask = self.input_sequence_builder(features) # (B, obs_token_len, embedding_dimension)
        batch_size = obs_tokens.shape[0]
        query_positional_encoding = self.learnable_query.weight.unsqueeze(0).repeat(batch_size, 1, 1) # (B, pred_horizon, embedding_dimension)
        query = torch.zeros_like(query_positional_encoding).to(self.device)
        query += query_positional_encoding
        action_embeddings = self.action_decoder(tgt=query, memory=obs_tokens, memory_key_padding_mask=obs_padding_mask)
        predictions = self._apply_action_heads(action_embeddings)
        return predictions








