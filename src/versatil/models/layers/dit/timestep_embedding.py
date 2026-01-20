"""Timestep embedding network for diffusion processes."""

import numpy as np
import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.positional_encoding.base import (
    DenominatorMode,
    OrderingMode,
    PositionSource,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)


class TimestepEmbeddingNetwork(nn.Module):
    """Embeds timesteps using sinusoidal encodings followed by an MLP."""

    def __init__(
        self,
        timestep_embedding_dim: int,
        output_dim: int,
        learnable_frequencies: bool = False,
    ) -> None:
        """Initialize the TimestepEmbeddingNetwork."""
        assert timestep_embedding_dim % 2 == 0, "timestep_embedding_dim must be even!"
        super().__init__()

        # Use SinusoidalPositionalEncoding1D for timestep embedding
        # Configured for scalar input (timestep value)
        self.sinusoidal_encoding = SinusoidalPositionalEncoding1D(
            embedding_dimension=timestep_embedding_dim,
            denominator_mode=DenominatorMode.HALF_MINUS_ONE.value,
            ordering_mode=OrderingMode.CAT_COS_SIN.value,
            position_source=PositionSource.SCALAR.value,
            precompute_encodings=False,
            temperature=10000.0,
        )

        # MLP to process sinusoidal embeddings
        self.output_network = nn.Sequential(
            nn.Linear(timestep_embedding_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed the timesteps.

        Args:
            timesteps: 1D tensor of timesteps (batch_size,).

        Returns:
            Embedded timesteps of shape (batch_size, output_dim).
        """
        assert len(timesteps.shape) == 1, "Assumes 1D input timestep array."

        # SinusoidalPositionalEncoding1D expects (B,) for scalar source
        # Returns (B, embedding_dim)
        sinusoidal_embeddings = self.sinusoidal_encoding(timesteps)

        # Process through MLP
        return self.output_network(sinusoidal_embeddings)

