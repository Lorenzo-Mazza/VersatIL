"""Tests for versatil.models.decoding.latent.prior.uniform_codebook_prior module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent.prior.uniform_codebook_prior import (
    UniformCodebookPrior,
)


@pytest.fixture
def uniform_prior_factory() -> Callable[..., UniformCodebookPrior]:

    def factory(
        latent_dimension: int = 8,
        num_codes: int = 4,
        num_residual_layers: int = 1,
    ) -> UniformCodebookPrior:
        return UniformCodebookPrior(
            latent_dimension=latent_dimension,
            num_codes=num_codes,
            num_residual_layers=num_residual_layers,
            device="cpu",
        )

    return factory


class TestUniformCodebookPriorInit:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "latent_dimension, num_codes, num_residual_layers",
        [(8, 4, 1), (16, 2, 2), (32, 16, 3)],
    )
    def test_stores_configuration(
        self,
        latent_dimension: int,
        num_codes: int,
        num_residual_layers: int,
    ) -> None:
        prior = UniformCodebookPrior(
            latent_dimension=latent_dimension,
            num_codes=num_codes,
            num_residual_layers=num_residual_layers,
            device="cpu",
        )
        assert prior.code_dim == latent_dimension
        assert prior.num_codes == num_codes
        assert prior.num_residual_layers == num_residual_layers
        assert prior.latent_dimension == latent_dimension

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "latent_dimension, num_codes, num_residual_layers, expected_message",
        [
            (0, 4, 1, "latent_dimension must be positive, got 0."),
            (8, 0, 1, "num_codes must be positive, got 0."),
            (8, 4, 0, "num_residual_layers must be positive, got 0."),
        ],
    )
    def test_rejects_invalid_configuration(
        self,
        latent_dimension: int,
        num_codes: int,
        num_residual_layers: int,
        expected_message: str,
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            UniformCodebookPrior(
                latent_dimension=latent_dimension,
                num_codes=num_codes,
                num_residual_layers=num_residual_layers,
                device="cpu",
            )


class TestUniformCodebookPriorGetAuxiliaryOutputKeys:
    @pytest.mark.unit
    def test_returns_latent_and_prior_indices_only(
        self,
        uniform_prior_factory: Callable[..., UniformCodebookPrior],
    ) -> None:
        prior = uniform_prior_factory(latent_dimension=8)
        keys = prior.get_auxiliary_output_keys()
        assert keys == {
            LatentKey.PRIOR_LATENT.value,
            LatentKey.VQ_PRIOR_INDICES.value,
        }

    @pytest.mark.unit
    def test_does_not_contain_gaussian_logits_or_posterior_keys(
        self,
        uniform_prior_factory: Callable[..., UniformCodebookPrior],
    ) -> None:
        prior = uniform_prior_factory(latent_dimension=8)
        keys = prior.get_auxiliary_output_keys()
        assert LatentKey.PRIOR_MU.value not in keys
        assert LatentKey.PRIOR_LOGVAR.value not in keys
        assert LatentKey.PRIOR_CODE_LOGITS.value not in keys
        assert LatentKey.VQ_INDICES.value not in keys


class TestUniformCodebookPriorWirePosterior:
    @pytest.mark.unit
    def test_sets_residual_vq(
        self,
        uniform_prior_factory: Callable[..., UniformCodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
    ) -> None:
        prior = uniform_prior_factory(latent_dimension=8)
        mock_posterior = mock_vq_posterior_factory(code_dim=8)
        prior.wire_posterior(mock_posterior)
        assert prior.residual_vq is mock_posterior.residual_vq

    @pytest.mark.unit
    def test_shared_residual_vq_is_not_registered_as_prior_child(
        self,
        uniform_prior_factory: Callable[..., UniformCodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
    ) -> None:
        prior = uniform_prior_factory(latent_dimension=8)
        mock_posterior = mock_vq_posterior_factory(code_dim=8)
        prior.wire_posterior(mock_posterior)

        assert prior.residual_vq is mock_posterior.residual_vq
        module_names = {name for name, _ in prior.named_modules()}
        assert "residual_vq" not in module_names
        assert all(not name.startswith("residual_vq.") for name in module_names)
        prior.eval()
        assert mock_posterior.residual_vq.training is True

    @pytest.mark.unit
    def test_raises_on_code_dim_mismatch(
        self,
        uniform_prior_factory: Callable[..., UniformCodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
    ) -> None:
        prior = uniform_prior_factory(latent_dimension=8)
        mock_posterior = mock_vq_posterior_factory(code_dim=16)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "ResidualVQ code_dim (16) does not match "
                "UniformCodebookPrior code_dim (8)"
            ),
        ):
            prior.wire_posterior(mock_posterior)

    @pytest.mark.unit
    def test_raises_on_num_codes_mismatch(
        self,
        uniform_prior_factory: Callable[..., UniformCodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
    ) -> None:
        prior = uniform_prior_factory(latent_dimension=8, num_codes=4)
        mock_posterior = mock_vq_posterior_factory(code_dim=8, num_codes=8)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "ResidualVQ num_codes (8) does not match "
                "UniformCodebookPrior num_codes (4)"
            ),
        ):
            prior.wire_posterior(mock_posterior)

    @pytest.mark.unit
    def test_raises_on_num_layers_mismatch(
        self,
        uniform_prior_factory: Callable[..., UniformCodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
    ) -> None:
        prior = uniform_prior_factory(latent_dimension=8, num_residual_layers=2)
        mock_posterior = mock_vq_posterior_factory(code_dim=8, num_layers=3)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "ResidualVQ num_layers (3) does not match "
                "UniformCodebookPrior num_residual_layers (2)"
            ),
        ):
            prior.wire_posterior(mock_posterior)

    @pytest.mark.unit
    def test_raises_on_missing_residual_vq_attribute(
        self,
        uniform_prior_factory: Callable[..., UniformCodebookPrior],
    ) -> None:
        prior = uniform_prior_factory(latent_dimension=8)
        mock_posterior = MagicMock(spec=[])
        with pytest.raises(
            AttributeError,
            match=re.escape(
                "Posterior MagicMock does not expose a "
                "residual_vq attribute required by UniformCodebookPrior."
            ),
        ):
            prior.wire_posterior(mock_posterior)


class TestUniformCodebookPriorForward:
    @pytest.mark.unit
    def test_raises_without_residual_vq(
        self,
        uniform_prior_factory: Callable[..., UniformCodebookPrior],
        target_latents_factory: Callable[..., torch.Tensor],
    ) -> None:
        prior = uniform_prior_factory(latent_dimension=8)
        target = target_latents_factory(batch_size=4, latent_dim=8)
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "UniformCodebookPrior.residual_vq is not set or has been "
                "garbage-collected. Call wire_posterior() before forward(), "
                "and keep the posterior alive."
            ),
        ):
            prior.forward(target_latents=target, observations={})


class TestUniformCodebookPriorForwardIntegration:
    @pytest.mark.integration
    @pytest.mark.parametrize("latent_dimension", [4, 16])
    @pytest.mark.parametrize("num_residual_layers", [1, 2])
    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_output_shapes(
        self,
        uniform_prior_factory: Callable[..., UniformCodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
        target_latents_factory: Callable[..., torch.Tensor],
        latent_dimension: int,
        num_residual_layers: int,
        batch_size: int,
    ) -> None:
        prior = uniform_prior_factory(
            latent_dimension=latent_dimension,
            num_residual_layers=num_residual_layers,
        )
        prior.wire_posterior(
            mock_vq_posterior_factory(
                code_dim=latent_dimension, num_layers=num_residual_layers
            )
        )
        target = target_latents_factory(
            batch_size=batch_size, latent_dim=latent_dimension
        )

        result = prior.forward(target_latents=target, observations={})

        assert result[LatentKey.PRIOR_LATENT.value].shape == (
            batch_size,
            latent_dimension,
        )
        assert LatentKey.VQ_INDICES.value not in result
        assert len(result[LatentKey.VQ_PRIOR_INDICES.value]) == num_residual_layers
        for indices in result[LatentKey.VQ_PRIOR_INDICES.value]:
            assert indices.shape == (batch_size,)
            assert indices.max().item() < prior.num_codes

    @pytest.mark.integration
    def test_sample_prior_returns_correct_shape(
        self,
        uniform_prior_factory: Callable[..., UniformCodebookPrior],
        mock_vq_posterior_factory: Callable[..., MagicMock],
    ) -> None:
        batch_size = 6
        latent_dimension = 8
        prior = uniform_prior_factory(latent_dimension=latent_dimension)
        prior.wire_posterior(mock_vq_posterior_factory(code_dim=latent_dimension))

        z = prior.sample_prior(batch_size=batch_size)

        assert z.shape == (batch_size, latent_dimension)

    @pytest.mark.integration
    def test_sample_prior_raises_without_residual_vq(
        self,
        uniform_prior_factory: Callable[..., UniformCodebookPrior],
    ) -> None:
        prior = uniform_prior_factory(latent_dimension=8)
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "UniformCodebookPrior.residual_vq is not set or has been "
                "garbage-collected. Call wire_posterior() before sample_prior(), "
                "and keep the posterior alive."
            ),
        ):
            prior.sample_prior(batch_size=4)
