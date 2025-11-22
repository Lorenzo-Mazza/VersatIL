import logging

import torch
from torch import nn

from refactoring.data.constants import IS_PAD_ACTION_KEY
from refactoring.models.decoding.constants import CLASS_TOKEN_KEY
from refactoring.models.layers.detr_transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from refactoring.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)
from refactoring.models.layers.transformer_input_builder import TransformerInputBuilder


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Reparametrization trick for VAE: sample from N(mu, var).

    Args:
        mu: Mean of the latent distribution, shape (batch, latent_dim).
        logvar: Log variance of the latent distribution, shape (batch, latent_dim).

    Returns:
        Sampled latent vector, shape (batch, latent_dim).
    """
    std = (logvar / 2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


class VAE(nn.Module):
    """VAE transformer architecture for encoding inputs into latent space.

    Handles input projection, transformer encoding with CLS token, and latent sampling and reparametrization.
    """
    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        feedforward_dimension: int,
        number_of_encoder_layers: int,
        activation: str,
        dropout_rate: float,
        normalize_before: bool,
        latent_dimension: int,
        use_proprioceptive: bool,
        prediction_horizon: int,
        observation_horizon: int,
        device: str,
    ):
        super().__init__()
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
        self.vae_encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                embedding_dimension=self.embedding_dimension,
                number_of_heads=self.number_of_heads,
                feedforward_dimension=self.feedforward_dimension,
                activation=self.activation,
                dropout=self.dropout_rate,
                normalize_before=self.normalize_before,
            ),
            number_of_layers=self.number_of_encoder_layers,
            normalization=nn.LayerNorm(self.embedding_dimension) if self.normalize_before else None
        )
        self.input_sequence_builder = TransformerInputBuilder(
            embedding_dim=self.embedding_dimension,
            has_time_dim=self.observation_horizon > 1,
            spatial_positional_encoding_layer=None,
            flat_positional_encoding_layer=SinusoidalPositionalEncoding1D(
                embedding_dimension=self.embedding_dimension,
                maximum_length=1000,
            ),
        )
        self.cls_token = nn.Embedding(1, self.embedding_dimension) # CLS input token
        self.latent_stats_projection = nn.Linear(
            self.embedding_dimension,
            self.vae_latent_dimension * 2  # Latent gaussian distribution parameters: mu and logvar
        )
        self.to(device)


    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        observations: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode action chunk to latent space z embedding using Variational Inference.

        Args:
            inputs: Dictionary containing:
             - ground-truth action chunk tensor of shape (B, prediction_horizon, total_input_dim)
             - optional padding mask tensor of shape (B, prediction_horizon) with boolean values
            observations: Optional dictionary of state features for conditional encoding with variable shapes

        Returns:
            Dictionary of tensors (z, mu, logvar) with shape (B, latent_dim) for each.
        """
        if observations is None:
            observations = {}
        for action in inputs:
            observations[action] = inputs[action].to(self.cls_token.weight.device).float()

        batch_size = list(inputs.values())[0].size(0)
        is_pad = observations.get(IS_PAD_ACTION_KEY)
        if is_pad is None:
            logging.warning("No padding key found in actions; assuming no padding.")
            is_pad = torch.zeros(
                batch_size,
                self.prediction_horizon,
                dtype=torch.bool,
                device=self.cls_token.weight.device
            )
            observations[IS_PAD_ACTION_KEY] = is_pad
        cls_embedding = self.cls_token.weight.unsqueeze(0).repeat(batch_size, 1, 1) # (B, 1, emb_dim)
        observations[CLASS_TOKEN_KEY] = cls_embedding
        input_tokens, pos_encodings, padding_mask = self.input_sequence_builder(observations) # (B, seq_len, embedding_dimension)
        encoder_output = self.vae_encoder(
            input_tokens,
            positional_encoding=pos_encodings,
            source_key_padding_mask=padding_mask
        )[:, -1, :]  # (B, CLS_TOKEN only, embedding_dim)
        latent_stats = self.latent_stats_projection(encoder_output) # (B, latent_dim * 2)
        mu, logvar = latent_stats.chunk(2, dim=1) # Each (B, latent_dim)
        z = reparametrize(mu, logvar) # Sample using reparametrization trick (B, latent_dim)
        return z, mu, logvar
