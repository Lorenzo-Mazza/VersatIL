"""Transformer Encoder used as the learnable parameterized prior for Variational Inference."""

import torch
from torch import nn

from versatil.models.decoding.constants import DecoderOutputKey, LatentKey
from versatil.models.decoding.latent import PriorLatentEncoder
from versatil.models.decoding.latent.reparametrize import reparametrize
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
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


class PriorTransformerEncoder(PriorLatentEncoder):
    """Transformer encoder network to model the conditional prior p_psi(z|s).

    Handles input projection, transformer encoding with CLS token, and latent sampling and reparametrization.
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
        exclude_keys: list[str] | None = None,
        learn_variance: bool = True,
        min_logvar: float | None = None,
        deterministic: bool = False,
    ):
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
        self.number_of_heads = number_of_heads
        self.feedforward_dimension = feedforward_dimension
        self.number_of_encoder_layers = number_of_encoder_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.normalize_before = normalize_before
        self.learn_variance = learn_variance
        self.encoder = TransformerEncoder(
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
            temporal_positional_encoding_layer=temporal_positional_encoding,
            flat_positional_encoding_layer=SinusoidalPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension,
                maximum_length=1000,
            ),
        )
        self.cls_token = nn.Embedding(1, self.embedding_dimension)  # CLS input token
        if self.deterministic:
            output_dim = self.latent_dimension
        elif self.learn_variance:
            output_dim = self.latent_dimension * 2
        else:
            output_dim = self.latent_dimension
        self.latent_projection = nn.Linear(
            self.embedding_dimension,
            output_dim,
        )
        self.to(device)

    def forward(
        self,
        observations: dict[str, torch.Tensor],
        target_latents: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode observation features to latent space z embedding using Variational Inference.

        Args:
            observations: Dictionary of observation features used as the input.
            target_latents: Target latent from posterior q(z|a,s). Not used here.

        Returns:
            Dictionary of tensors (z, mu, logvar) with shape (B, latent_dim) for each.
        """
        input_observations = {
            k: v for k, v in observations.items() if k not in self.exclude_keys
        }

        batch_size = list(input_observations.values())[0].size(0)
        cls_embedding = self.cls_token.weight.unsqueeze(0).repeat(
            batch_size, 1, 1
        )  # (B, 1, emb_dim)
        input_observations[DecoderOutputKey.CLASS_TOKEN.value] = cls_embedding
        input_tokens, pos_encodings, padding_mask = self.input_sequence_builder(
            input_observations
        )  # (B, seq_len, embedding_dimension)
        # input_tokens contains the CLS token at the end of the sequence
        encoder_output = self.encoder(
            input_tokens,
            positional_encoding=pos_encodings,
            source_key_padding_mask=padding_mask,
        )[:, -1, :]  # (B, CLS_TOKEN only, embedding_dim)
        latent_stats = self.latent_projection(encoder_output)
        if self.deterministic:
            z = latent_stats  # (B, latent_dim)
            return {
                LatentKey.PRIOR_LATENT.value: z,
                LatentKey.PRIOR_MU.value: z,
            }
        if self.learn_variance:
            mu, logvar = latent_stats.chunk(2, dim=1)  # Each (B, latent_dim)
        else:
            mu = latent_stats  # (B, latent_dim)
            logvar = torch.zeros_like(mu)  # Fixed logvar = 0.0 (std = 1.0)
        if self.min_logvar is not None:
            logvar = torch.clamp(logvar, min=self.min_logvar)
        z = reparametrize(mu, logvar)  # (B, latent_dim)
        return {
            LatentKey.PRIOR_MU.value: mu,
            LatentKey.PRIOR_LOGVAR.value: logvar,
            LatentKey.PRIOR_LATENT.value: z,
        }

    def sample_prior(
        self,
        batch_size: int,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sample latent variable from learned prior p(z|s)."""
        return self.forward(
            target_latents=None,
            observations=observations,
        )[LatentKey.PRIOR_LATENT.value]  # Return only z
