"""Shared fixtures for latent variable model tests."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.models.decoding.latent.posterior.vq_encoder import VQPosteriorEncoder
from versatil.models.decoding.latent.vq.residual_vq import ResidualVQ


@pytest.fixture
def residual_vq_factory() -> Callable[..., ResidualVQ]:

    def factory(
        input_dimension: int = 8,
        code_dim: int = 8,
        num_codes: int = 4,
        num_layers: int = 1,
    ) -> ResidualVQ:
        return ResidualVQ(
            input_dimension=input_dimension,
            code_dim=code_dim,
            num_codes=num_codes,
            num_layers=num_layers,
            kmeans_init=False,
        )

    return factory


@pytest.fixture
def target_latents_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:

    def factory(
        batch_size: int = 4,
        latent_dim: int = 8,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, latent_dim)).astype(np.float32)
        )

    return factory


@pytest.fixture
def mock_vq_posterior_factory(
    residual_vq_factory: Callable[..., ResidualVQ],
) -> Callable[..., MagicMock]:

    def factory(
        code_dim: int = 8,
        num_codes: int = 4,
        num_layers: int = 1,
    ) -> MagicMock:
        mock = MagicMock(spec=VQPosteriorEncoder)
        mock.residual_vq = residual_vq_factory(
            input_dimension=code_dim,
            code_dim=code_dim,
            num_codes=num_codes,
            num_layers=num_layers,
        )
        return mock

    return factory
