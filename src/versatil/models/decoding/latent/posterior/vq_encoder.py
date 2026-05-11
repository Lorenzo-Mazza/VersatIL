"""Posterior encoder with vector quantization for discrete latent variable models.

Uses the same transformer backbone as VAETransformerEncoder to produce a
continuous embedding from actions and observations, then quantizes it
via ResidualVQ to produce a discrete latent code. The quantized embedding
is passed to the decoder via the VariationalAlgorithm. Commitment loss
inputs (continuous z and quantized z) are stored in the output dict for
external loss computation in the metrics module.
"""

import logging

import torch
from torch import nn

from versatil.data.constants import SampleKey
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey
from versatil.models.decoding.latent.posterior.base_posterior import (
    PosteriorLatentEncoder,
)
from versatil.models.decoding.latent.vq.residual_vq import ResidualVQ
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import AttentionType
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
)
from versatil.models.layers.transformer.encoder import TransformerEncoder


class VQPosteriorEncoder(PosteriorLatentEncoder):
    """Transformer posterior encoder with residual vector quantization.

    Encodes actions and observations into a continuous embedding via a
    transformer encoder with a CLS token, then quantizes the embedding
    through a ResidualVQ bottleneck. The quantized output is the latent
    z passed to the decoder. The continuous pre-quantization embedding
    and codebook indices are stored in the output dict for commitment
    loss computation and prior training.

    Args:
        latent_dimension: Dimension of each codebook vector and the
            latent space passed to the decoder.
        num_codes: Number of codebook entries per residual layer (K).
        num_residual_layers: Number of cascading VQ layers.
        ema_decay: EMA decay for codebook updates.
        dead_code_threshold: Cluster size below which codes are replaced.
        embedding_dimension: Transformer hidden dimension.
        prediction_horizon: Number of action timesteps.
        observation_horizon: Number of observation timesteps.
        device: Device string.
        number_of_heads: Number of attention heads.
        feedforward_dimension: Feedforward network dimension.
        number_of_encoder_layers: Number of transformer encoder layers.
        activation: Activation function name.
        dropout_rate: Dropout probability.
        attention_type: Attention mechanism type (use AttentionType enum values).
        exclude_keys: Observation keys to exclude from encoding.
    """

    def __init__(
        self,
        latent_dimension: int,
        num_codes: int,
        num_residual_layers: int,
        embedding_dimension: int,
        prediction_horizon: int,
        observation_horizon: int,
        device: str,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
        number_of_heads: int = 4,
        feedforward_dimension: int = 128,
        number_of_encoder_layers: int = 1,
        activation: str = ActivationFunction.SWIGLU.value,
        dropout_rate: float = 0.0,
        attention_dropout: float = 0.0,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        positional_encoding_type: str | None = None,
        exclude_keys: list[str] | None = None,
    ):
        super().__init__(
            latent_dimension=latent_dimension,
            device=device,
        )
        self.code_dim = latent_dimension
        self.num_codes = num_codes
        self.num_residual_layers = num_residual_layers
        self.exclude_keys = exclude_keys if exclude_keys is not None else []
        self.embedding_dimension = embedding_dimension
        self.prediction_horizon = prediction_horizon
        self.observation_horizon = observation_horizon

        self.transformer_encoder = TransformerEncoder(
            number_of_layers=number_of_encoder_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            activation=activation,
            dropout=dropout_rate,
            attention_dropout=attention_dropout,
            normalization_type=normalization_type,
            attention_type=attention_type,
            positional_encoding_type=positional_encoding_type,
        )

        temporal_positional_encoding = None
        if observation_horizon > 1:
            temporal_positional_encoding = LearnedPositionalEncoding1D(
                embedding_dimension=embedding_dimension
            )

        self.input_sequence_builder = TransformerInputBuilder(
            embedding_dim=embedding_dimension,
            has_time_dim=observation_horizon > 1,
            spatial_positional_encoding_layer=SinusoidalPositionalEncoding2D(
                embedding_dimension=embedding_dimension, normalize=True
            ),
            flat_positional_encoding_layer=SinusoidalPositionalEncoding1D(
                embedding_dimension=embedding_dimension,
                maximum_length=1000,
            ),
            temporal_positional_encoding_layer=temporal_positional_encoding,
        )

        self.cls_token = nn.Embedding(1, embedding_dimension)  # (1, emb_dim)
        self.latent_projection = nn.Linear(embedding_dimension, self.code_dim)

        self.residual_vq = ResidualVQ(
            input_dim=self.code_dim,
            code_dim=self.code_dim,
            num_codes=num_codes,
            num_layers=num_residual_layers,
            ema_decay=ema_decay,
            dead_code_threshold=dead_code_threshold,
            kmeans_init=True,
        )

        self.to(device)

    def get_auxiliary_output_keys(self) -> set[str]:
        """VQ posterior outputs quantized latent, codebook indices, and continuous z."""
        return {
            LatentKey.POSTERIOR_LATENT.value,
            LatentKey.VQ_INDICES.value,
            LatentKey.VQ_Z_CONTINUOUS.value,
            LatentKey.VQ_QUANTIZED.value,
        }

    def encode(
        self,
        actions: dict[str, torch.Tensor],
        observations: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode actions into a vector-quantized latent.

        Args:
            actions: Dictionary of action tensors,
                shape (B, prediction_horizon, action_dim) per component.
            observations: Optional observation features for conditioning.

        Returns:
            Dictionary containing:
                - LatentKey.POSTERIOR_LATENT: Quantized latent z (B, code_dim),
                    with straight-through gradient for decoder training.
                - LatentKey.VQ_INDICES: Per-layer codebook indices,
                    list of (B,) tensors, for prior training.
                - LatentKey.VQ_Z_CONTINUOUS: Per-layer pre-quantization
                    encoder outputs in code space (L, B, code_dim). Carries
                    gradient; paired with VQ_QUANTIZED for commitment loss.
                - LatentKey.VQ_QUANTIZED: Per-layer hard-quantized codebook
                    vectors in code space (L, B, code_dim), detached. Used
                    with VQ_Z_CONTINUOUS for per-layer commitment loss.
        """
        if observations is not None:
            input_observations = {
                k: v for k, v in observations.items() if k not in self.exclude_keys
            }
        else:
            input_observations = {}

        action_feature_keys = []
        for action_key, action_tensor in actions.items():
            if action_key == SampleKey.IS_PAD_ACTION.value:
                continue
            input_observations[action_key] = action_tensor.to(
                self.cls_token.weight.device
            ).float()
            action_feature_keys.append(action_key)

        batch_size = list(input_observations.values())[0].size(0)
        is_pad = actions.get(SampleKey.IS_PAD_ACTION.value)
        if is_pad is None:
            logging.warning("No padding key found in actions; assuming no padding.")
            is_pad = torch.zeros(
                batch_size,
                self.prediction_horizon,
                dtype=torch.bool,
                device=self.cls_token.weight.device,
            )
            input_observations[SampleKey.IS_PAD_ACTION.value] = is_pad
        else:
            is_pad = is_pad.to(device=self.cls_token.weight.device, dtype=torch.bool)
            input_observations[SampleKey.IS_PAD_ACTION.value] = is_pad

        for action_key in action_feature_keys:
            input_observations[
                f"{action_key}_{EncoderOutputKeys.PADDING_MASK.value}"
            ] = is_pad

        cls_embedding = self.cls_token.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (B, 1, emb_dim)
        input_observations[DecoderOutputKey.CLASS_TOKEN.value] = cls_embedding

        input_tokens, pos_encodings, padding_mask = self.input_sequence_builder(
            input_observations
        )  # (B, seq_len, emb_dim)

        hidden_states = input_tokens + pos_encodings
        encoder_output = self.transformer_encoder(
            hidden_states=hidden_states,
            padding_mask=padding_mask,
        )[:, -1, :]  # (B, emb_dim) — CLS token at last position

        z_continuous = self.latent_projection(encoder_output)  # (B, code_dim)

        z_q, all_indices, z_e_per_layer, z_q_per_layer = self.residual_vq(
            z_continuous
        )  # (B, code_dim), list[(B,)], (L, B, code_dim), (L, B, code_dim)

        return {
            LatentKey.POSTERIOR_LATENT.value: z_q,
            LatentKey.VQ_INDICES.value: all_indices,
            LatentKey.VQ_Z_CONTINUOUS.value: z_e_per_layer,
            LatentKey.VQ_QUANTIZED.value: z_q_per_layer,
        }
