"""Decoder-only transformer for parallel action chunk prediction, with cross-attention to encoding features."""

import torch

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.decoders import DecoderInput
from versatil.models.decoding.decoders.parallel_transformer import (
    BaseParallelTransformerDecoder,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType, PositionalEncodingType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.transformer.bidirectional_decoder import (
    BidirectionalDecoder,
)


class ActionTransformer(BaseParallelTransformerDecoder):
    """Bidirectional Transformer decoder which decodes action chunks with cross-attention to  observation tokens."""

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
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        number_of_layers: int = 6,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        positional_encoding_type: str | None = PositionalEncodingType.ROPE.value,
    ) -> None:
        decoder_input = DecoderInput(
            keys=input_keys,
            required_types=[],
            requires_actions=False,
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
        self.number_of_layers = number_of_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.feedforward_dimension = feedforward_dimension
        self.number_of_heads = number_of_heads
        self.number_of_key_value_heads = number_of_key_value_heads
        self.normalization_type = normalization_type
        self.attention_type = attention_type
        self.attention_dropout = attention_dropout
        self.positional_encoding_type = positional_encoding_type
        self._build_transformer_components()
        self.to(self.device)

    def _build_transformer_components(self) -> None:
        """Build core transformer encoder-decoder and positional encodings."""
        self.input_sequence_builder = self._build_parallel_input_sequence_builder()
        self.learnable_query = (
            self._build_parallel_query_embedding()
        )  # (prediction_horizon, embedding_dimension)
        self.action_decoder = BidirectionalDecoder(
            number_of_layers=self.number_of_layers,
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
        )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
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
        obs_tokens, obs_padding_mask = self._build_parallel_observation_tokens(
            input_sequence_builder=self.input_sequence_builder,
            features=features,
            add_positional_encodings=True,
        )  # (B, observation_token_count, embedding_dimension), (B, observation_token_count)
        batch_size = obs_tokens.shape[0]
        query = self._expand_parallel_query_embedding(
            query_embedding=self.learnable_query,
            batch_size=batch_size,
        )  # (B, prediction_horizon, embedding_dimension)
        action_embeddings = self.action_decoder(
            hidden_states=query,
            encoded_features=obs_tokens,
            query_padding_mask=None,
            memory_padding_mask=obs_padding_mask,
        )  # (B, prediction_horizon, embedding_dimension)
        predictions = self._apply_action_heads(action_embeddings)
        return predictions
