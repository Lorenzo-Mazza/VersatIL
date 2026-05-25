"""Action Chunking Transformer (ACT) architecture for action decoding.

Reference: https://arxiv.org/abs/2304.13705
"""

import torch

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.decoders.base import DecoderInput
from versatil.models.decoding.decoders.parallel_transformer import (
    BaseParallelTransformerDecoder,
)
from versatil.models.feature_meta import FeatureType
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.detr_transformer import Transformer


class ACT(BaseParallelTransformerDecoder):
    """Action Chunking Transformer network for action decoding.

    This architecture:
    - Encodes multi-camera images into spatial features
    - Optionally accepts a latent embedding from the algorithm layer (e.g., from VAE)
    - Convert flat and spatial features into a sequence of token embeddings with shared embedding dimension
    - Decodes actions in parallel using a DETR-style non-autoregressive transformer with learnable queries
    - Supports multiple action heads: position, orientation, gripper

    Note: Latent action encoding is handled at the Algorithm level,
    not within this decoder. The decoder expects latent embeddings to be passed
    via the features dictionary with key LatentKey.POSTERIOR_LATENT.
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
    ) -> None:
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

        """
        decoder_input = DecoderInput(
            keys=input_keys,
            required_types=[FeatureType.SPATIAL.value],
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
        self.number_of_heads = number_of_heads
        self.feedforward_dimension = feedforward_dimension
        self.number_of_encoder_layers = number_of_encoder_layers
        self.number_of_decoder_layers = number_of_decoder_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.normalize_before = normalize_before
        self._build_transformer_components()
        self.to(self.device)

    def _build_transformer_components(self) -> None:
        """Build core transformer encoder-decoder and positional encodings."""
        self.input_sequence_builder = self._build_parallel_input_sequence_builder()
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
        self.learnable_query = (
            self._build_parallel_query_embedding()
        )  # (prediction_horizon, embedding_dimension)

    def _decode_actions(
        self,
        input_tokens: torch.Tensor,
        positional_encodings: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run DETR non-causal transformer encoder-decoder to predict chunks of action embeddings in parallel.

        Args:
            input_tokens: Input tokens to the action decoder, shape (B, obs_sequence_len, embedding_dimension)
            positional_encodings: Positional encodings, shape (B, obs_sequence_len, embedding_dimension)
            padding_mask: Optional padding mask for encoder tokens, shape (B, obs_sequence_len), where True indicates padding tokens.

        Returns:
            Action embeddings (B, prediction_horizon, embedding_dimension)
        """
        batch_size = input_tokens.shape[0]
        query_positional_encoding = self._expand_parallel_query_embedding(
            query_embedding=self.learnable_query,
            batch_size=batch_size,
        )  # (B, prediction_horizon, embedding_dimension)
        target = torch.zeros_like(query_positional_encoding)  # (B, H, D)
        decoder_outputs = self.action_decoder(
            source=input_tokens,
            target=target,
            source_positional_encoding=positional_encodings,
            source_key_padding_mask=padding_mask,
            target_positional_encoding=query_positional_encoding,
        )
        return decoder_outputs[-1]  # (B, prediction_horizon, embedding_dimension)

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of ACT architecture.

        Args:
            features: Dictionary of encoded features from EncodingPipeline
            actions: Not used, present for API compatibility.

        Returns:
            Dictionary containing action head predictions (e.g. position, orientation, gripper)

        Note:
            If LatentKey.POSTERIOR_LATENT is present in features, it will be used as an extra token embedding for the transformer cross-attention.
        """
        # This creates a sequence of input tokens and positional encodings in the format ACT expects
        input_tokens, pos_encodings, padding_mask = self.input_sequence_builder(
            features
        )  # (B, observation_token_count, embedding_dimension)
        action_embeddings = self._decode_actions(
            input_tokens=input_tokens,
            positional_encodings=pos_encodings,
            padding_mask=padding_mask,
        )  # (B, prediction_horizon, embedding_dimension)
        predictions = self._apply_action_heads(action_embeddings)
        return predictions
