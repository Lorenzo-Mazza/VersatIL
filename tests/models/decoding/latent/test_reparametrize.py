"""Tests for versatil.models.decoding.latent.reparametrize module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.models.decoding.latent.reparametrize import reparametrize


@pytest.fixture
def mu_logvar_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Factory for mu and logvar tensors with configurable shape."""

    def factory(
        batch_size: int = 2,
        latent_dim: int = 8,
        mu_value: float | None = None,
        logvar_value: float | None = None,
        requires_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mu_value is not None:
            mu = torch.full(
                (batch_size, latent_dim),
                mu_value,
                dtype=torch.float32,
            )
        else:
            mu = torch.from_numpy(
                rng.standard_normal((batch_size, latent_dim)).astype(np.float32)
            )
        if logvar_value is not None:
            logvar = torch.full(
                (batch_size, latent_dim),
                logvar_value,
                dtype=torch.float32,
            )
        else:
            logvar = torch.from_numpy(
                rng.standard_normal((batch_size, latent_dim)).astype(np.float32)
            )
        mu.requires_grad_(requires_grad)
        logvar.requires_grad_(requires_grad)
        return mu, logvar

    return factory


class TestReparametrize:
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("latent_dim", [8, 32])
    def test_output_shape_matches_mu_shape(
        self,
        mu_logvar_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        batch_size: int,
        latent_dim: int,
    ):
        mu, logvar = mu_logvar_factory(
            batch_size=batch_size,
            latent_dim=latent_dim,
        )
        result = reparametrize(mu=mu, logvar=logvar)
        assert result.shape == (batch_size, latent_dim)

    def test_very_negative_logvar_samples_near_mu(
        self,
        mu_logvar_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        # When logvar=-40, std=exp(-20) ≈ 2e-9, so z ≈ mu consistently
        torch.manual_seed(42)
        mu, logvar = mu_logvar_factory(
            batch_size=4,
            latent_dim=16,
            mu_value=3.0,
            logvar_value=-40.0,
        )
        result = reparametrize(mu=mu, logvar=logvar)
        assert torch.allclose(result, mu, atol=1e-4)

    def test_large_negative_logvar_returns_near_mu(
        self,
        mu_logvar_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        # When logvar=-40, std=exp(-20) ≈ 2e-9, so z ≈ mu
        torch.manual_seed(42)
        mu, logvar = mu_logvar_factory(
            batch_size=4,
            latent_dim=16,
            mu_value=5.0,
            logvar_value=-40.0,
        )
        result = reparametrize(mu=mu, logvar=logvar)
        assert torch.allclose(result, mu, atol=1e-6)

    def test_gradient_flows_through_mu_and_logvar(
        self,
        mu_logvar_factory: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    ):
        mu, logvar = mu_logvar_factory(
            batch_size=2,
            latent_dim=8,
            requires_grad=True,
        )
        result = reparametrize(mu=mu, logvar=logvar)
        loss = result.sum()
        loss.backward()
        assert mu.grad is not None
        assert logvar.grad is not None
        assert mu.grad.shape == mu.shape
        assert logvar.grad.shape == logvar.shape
