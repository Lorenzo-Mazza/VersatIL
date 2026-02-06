"""
This is the transformer posterior encoder used in the original Action-Chunking Transformer paper.
It takes as input a chunk of actions plus observation tokens and uses a transformer encoder with a CLS token to produce
a latent embedding (split into mean and log variance), which is then reparameterized to produce a latent sample.
"""
import logging

import torch
from torch import nn

from versatil.data.constants import SampleKey
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey
from versatil.models.decoding.latent.posterior.base_posterior import (
    PosteriorLatentEncoder,
)
from versatil.models.decoding.latent.reparametrize import reparametrize
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.detr_transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
)
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder


class VAETransformerEncoder(PosteriorLatentEncoder):
    """Transformer-based posterior encoder for encoding actions into latent space.

    Args:
        embedding_dimension: Transformer hidden dimension
        number_of_heads: Number of attention heads
        feedforward_dimension: Feedforward network dimension
        number_of_encoder_layers: Number of transformer encoder layers
        activation: Activation function name
        dropout_rate: Dropout probability
        normalize_before: Use pre-normalization
        latent_dimension: Dimension of VAE latent space (z)
        use_proprioceptive: Whether to condition on proprioceptive observations
        prediction_horizon: Number of action timesteps
        device: Device to place encoder on
    """

    def __init__(
        self,
        embedding_dimension: int,
        latent_dimension: int,
        prediction_horizon: int,
        observation_horizon: int,
        device: str,
        number_of_heads: int = 8,
        feedforward_dimension: int = 512,
        number_of_encoder_layers: int = 4,
        activation: str = ActivationFunction.SWIGLU.value,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        use_proprioceptive: bool = False,
        exclude_keys: list[str] = None,
        min_logvar: float | None = None,
        deterministic: bool = False,
    ):
        """Initialize VAE latent action encoder.

        Args:
            embedding_dimension: Dimension of the output embedding
            latent_dimension: Dimension of VAE latent space, i.e. the dimension of the output z
            prediction_horizon: Number of action timesteps
            observation_horizon: Number of observation timesteps
            device: Device to place encoder on
            number_of_heads: Number of attention heads
            feedforward_dimension: Feedforward network dimension
            number_of_encoder_layers: Number of transformer encoder layers
            activation: Activation function name
            dropout_rate: Dropout probability
            normalize_before: Use pre-normalization
            use_proprioceptive: Whether to condition on proprioceptive observations
            exclude_keys: List of keys to exclude from encoding
            min_logvar: Minimum log variance for avoiding variance collapse
            deterministic: If True, output deterministic embeddings without reparameterization.
                Use with MMD or OT regularizers instead of KL divergence.

        """
        super().__init__(
            latent_dimension=latent_dimension,
            device=device,
        )
        self.exclude_keys = exclude_keys if exclude_keys is not None else []
        self.min_logvar = min_logvar
        self.deterministic = deterministic
        self.embedding_dimension = embedding_dimension
        self.use_proprioceptive = use_proprioceptive
        self.prediction_horizon = prediction_horizon
        self.observation_horizon = observation_horizon
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.feedforward_dimension = feedforward_dimension
        self.number_of_encoder_layers = number_of_encoder_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.normalize_before = normalize_before
        self.vae_latent_dimension = latent_dimension
        self.use_proprioceptive = use_proprioceptive
        self.prediction_horizon = prediction_horizon
        self.observation_horizon = observation_horizon
        self.device = device
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                embedding_dimension=self.embedding_dimension,
                number_of_heads=self.number_of_heads,
                feedforward_dimension=self.feedforward_dimension,
                activation=self.activation,
                dropout=self.dropout_rate,
                normalize_before=self.normalize_before,
            ),
            number_of_layers=self.number_of_encoder_layers,
            normalization=nn.LayerNorm(self.embedding_dimension)
            if self.normalize_before
            else None,
        )
        temporal_positional_encoding = None
        if self.observation_horizon > 1:
            temporal_positional_encoding = LearnedPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension
            )

        self.input_sequence_builder = TransformerInputBuilder(
            embedding_dim=self.embedding_dimension,
            has_time_dim=self.observation_horizon > 1,
            spatial_positional_encoding_layer=SinusoidalPositionalEncoding2D(
                embedding_dimension=self.embedding_dimension, normalize=True
            ),
            flat_positional_encoding_layer=SinusoidalPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension,
                maximum_length=1000,
            ),
            temporal_positional_encoding_layer=temporal_positional_encoding,
        )
        self.cls_token = nn.Embedding(1, self.embedding_dimension)  # CLS input token
        projection_dim = (
            self.vae_latent_dimension # mu
            if self.deterministic
            else self.vae_latent_dimension * 2  # mu and logvar
        )
        self.latent_projection = nn.Linear(
            self.embedding_dimension,
            projection_dim,
        )
        self.to(device)

    def encode(
        self,
        actions: dict[str, torch.Tensor],
        observations: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        """Encode actions into latent space using VAE.

        Args:
            actions: Dictionary of action tensors
                Shape: (B, prediction_horizon, action_dim) for each component
            observations: Optional observation features to condition encoding

        Note:
            Image observations are automatically excluded from encoding, plus any additional custom key.

        Returns:
            Dictionary containing:
                - LatentKey.POSTERIOR_LATENT: Latent embedding z (B, vae_latent_dimension)
                - LatentKey.POSTERIOR_MU: Latent distribution mean (B, vae_latent_dimension)
                - LatentKey.POSTERIOR_LOGVAR: Latent distribution log variance (B, vae_latent_dimension)
                - STATE_FEATURE_KEYS: Input observations used for encoding (dict or None)
        """
        if observations is not None:
            input_observations = {
                k: v for k, v in observations.items() if not (k in self.exclude_keys)
            }
        else:
            input_observations = {}

        for action in actions:
            input_observations[action] = (
                actions[action].to(self.cls_token.weight.device).float()
            )

        batch_size = list(input_observations.values())[0].size(0)
        is_pad = input_observations.get(SampleKey.IS_PAD_ACTION.value)
        if is_pad is None:
            logging.warning("No padding key found in actions; assuming no padding.")
            is_pad = torch.zeros(
                batch_size,
                self.prediction_horizon,
                dtype=torch.bool,
                device=self.cls_token.weight.device,
            )
            input_observations[SampleKey.IS_PAD_ACTION.value] = is_pad
        cls_embedding = self.cls_token.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (B, 1, emb_dim)
        input_observations[DecoderOutputKey.CLASS_TOKEN.value] = cls_embedding
        input_tokens, pos_encodings, padding_mask = self.input_sequence_builder(
            input_observations
        )  # (B, seq_len, embedding_dimension), CLS token at the end
        encoder_output = self.transformer_encoder(
            input_tokens,
            positional_encoding=pos_encodings,
            source_key_padding_mask=padding_mask,
        )[
            :, -1, :
        ]  # (B, CLS_TOKEN only, embedding_dim)
        latent_stats = self.latent_projection(encoder_output)
        if self.deterministic:
            z = latent_stats  # (B, latent_dim)
            return {
                LatentKey.POSTERIOR_LATENT.value: z,
                LatentKey.POSTERIOR_MU.value: z,
            }
        mu, logvar = latent_stats.chunk(2, dim=1)  # Each (B, latent_dim)
        if self.min_logvar is not None:
            logvar = torch.clamp(logvar, min=self.min_logvar)
        z = reparametrize(mu, logvar)  # (B, latent_dim)
        return {
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.POSTERIOR_MU.value: mu,
            LatentKey.POSTERIOR_LOGVAR.value: logvar,
        }
