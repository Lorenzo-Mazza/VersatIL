"""Shared helpers for parallel transformer action decoders."""

import torch
from torch import nn

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import BaseActionHead
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
from versatil.models.layers.constants import PositionalEncodingType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
)


class BaseParallelTransformerDecoder(ActionDecoder):
    """Base class for transformer decoders that predict action chunks."""

    def __init__(
        self,
        *,
        decoder_input: DecoderInput,
        action_space: ActionSpace,
        action_heads: dict[str, BaseActionHead],
        observation_space: ObservationSpace,
        prediction_horizon: int,
        observation_horizon: int,
        device: str,
        embedding_dimension: int,
    ) -> None:
        """Initialize shared parallel transformer decoder state."""
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

    def _build_parallel_input_sequence_builder(
        self,
        exclude_keys: list[str] | None = None,
        flat_positional_encoding_type: str = PositionalEncodingType.LEARNED.value,
    ) -> TransformerInputBuilder:
        """Create the standard observation-token builder."""
        temporal_positional_encoding = None
        if self.observation_horizon > 1:
            temporal_positional_encoding = LearnedPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension,
            )
        flat_positional_encoding = self._build_flat_positional_encoding(
            flat_positional_encoding_type=flat_positional_encoding_type,
        )
        return TransformerInputBuilder(
            embedding_dimension=self.embedding_dimension,
            spatial_positional_encoding_layer=SinusoidalPositionalEncoding2D(
                embedding_dimension=self.embedding_dimension,
                normalize=True,
            ),
            flat_positional_encoding_layer=flat_positional_encoding,
            temporal_positional_encoding_layer=temporal_positional_encoding,
            exclude_keys=exclude_keys,
        )

    def _build_flat_positional_encoding(
        self,
        flat_positional_encoding_type: str,
    ) -> LearnedPositionalEncoding1D | SinusoidalPositionalEncoding1D:
        """Create the flat-feature positional encoding layer."""
        match flat_positional_encoding_type:
            case PositionalEncodingType.LEARNED.value:
                return LearnedPositionalEncoding1D(
                    embedding_dimension=self.embedding_dimension,
                )
            case PositionalEncodingType.SINUSOIDAL.value:
                return SinusoidalPositionalEncoding1D(
                    embedding_dimension=self.embedding_dimension,
                )
            case _:
                raise ValueError(
                    "flat_positional_encoding_type must be one of "
                    f"{PositionalEncodingType.LEARNED.value!r} or "
                    f"{PositionalEncodingType.SINUSOIDAL.value!r}, got "
                    f"{flat_positional_encoding_type!r}."
                )

    def _build_parallel_query_embedding(self) -> nn.Embedding:
        """Create learned query embeddings for parallel action prediction."""
        return nn.Embedding(self.prediction_horizon, self.embedding_dimension)

    @staticmethod
    def _expand_parallel_query_embedding(
        query_embedding: nn.Embedding,
        batch_size: int,
    ) -> torch.Tensor:
        """Expand learned query embeddings to the current batch."""
        return query_embedding.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (B, prediction_horizon, embedding_dimension)

    @staticmethod
    def _expand_parallel_query_tensor(
        query: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Expand a learned query tensor to the current batch."""
        return query.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (B, query_length, embedding_dimension)

    def _build_parallel_observation_tokens(
        self,
        input_sequence_builder: TransformerInputBuilder,
        features: dict[str, torch.Tensor],
        add_positional_encodings: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build observation tokens and optional padding mask."""
        observation_tokens, positional_encodings, padding_mask = input_sequence_builder(
            features
        )
        # tokens/positions: (B, observation_token_count, embedding_dimension)
        # padding_mask: (B, observation_token_count)
        if add_positional_encodings and positional_encodings is not None:
            observation_tokens = observation_tokens + positional_encodings
        return observation_tokens, padding_mask

    def _apply_action_heads(
        self,
        action_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Apply component action heads to action embeddings."""
        return {
            action_key: action_head(action_embeddings)
            for action_key, action_head in self.action_heads.items()
        }
