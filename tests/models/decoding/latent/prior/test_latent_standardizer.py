"""Tests for versatil.models.decoding.latent.prior.latent_standardizer module."""

import re
from collections.abc import Callable

import pytest
import torch

from versatil.models.decoding.latent.prior.latent_standardizer import LatentStandardizer


@pytest.fixture
def latent_standardizer_factory() -> Callable[..., LatentStandardizer]:
    """Factory for latent standardizers."""

    def factory(
        latent_dimension: int = 3,
        enabled: bool = True,
        eps: float = 1e-6,
        require_fitted: bool = False,
    ) -> LatentStandardizer:
        return LatentStandardizer(
            latent_dimension=latent_dimension,
            enabled=enabled,
            eps=eps,
            require_fitted=require_fitted,
        )

    return factory


class TestLatentStandardizerTransforms:
    @pytest.mark.unit
    def test_unfitted_standardizer_is_identity_by_default(
        self,
        latent_standardizer_factory: Callable[..., LatentStandardizer],
    ) -> None:
        standardizer = latent_standardizer_factory(latent_dimension=3)
        latents = torch.tensor([[1.0, 2.0, 3.0]])

        torch.testing.assert_close(standardizer.standardize(latents), latents)
        torch.testing.assert_close(standardizer.unstandardize(latents), latents)

    @pytest.mark.unit
    def test_disabled_standardizer_stays_identity_after_fit(
        self,
        latent_standardizer_factory: Callable[..., LatentStandardizer],
    ) -> None:
        standardizer = latent_standardizer_factory(latent_dimension=3, enabled=False)
        standardizer.set_stats(
            mean=torch.tensor([1.0, 2.0, 3.0]),
            std=torch.tensor([2.0, 4.0, 5.0]),
        )
        latents = torch.tensor([[3.0, 10.0, 18.0]])

        torch.testing.assert_close(standardizer.standardize(latents), latents)
        torch.testing.assert_close(standardizer.unstandardize(latents), latents)

    @pytest.mark.unit
    def test_set_stats_standardizes_and_unstandardizes(
        self,
        latent_standardizer_factory: Callable[..., LatentStandardizer],
    ) -> None:
        standardizer = latent_standardizer_factory(latent_dimension=3)
        standardizer.set_stats(
            mean=torch.tensor([1.0, 2.0, 3.0]),
            std=torch.tensor([2.0, 4.0, 5.0]),
        )
        latents = torch.tensor([[3.0, 10.0, 18.0]])

        standardized = standardizer.standardize(latents)
        reconstructed = standardizer.unstandardize(standardized)

        torch.testing.assert_close(
            standardized,
            torch.tensor([[1.0, 2.0, 3.0]]),
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(reconstructed, latents, rtol=1e-5, atol=1e-5)

    @pytest.mark.unit
    def test_fit_computes_population_statistics(
        self,
        latent_standardizer_factory: Callable[..., LatentStandardizer],
    ) -> None:
        standardizer = latent_standardizer_factory(latent_dimension=2)
        latents = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )

        standardizer.fit(latents)

        torch.testing.assert_close(standardizer.mean, torch.tensor([4.0, 5.0]))
        torch.testing.assert_close(
            standardizer.std,
            torch.tensor([2.236068, 2.236068]),
            rtol=1e-5,
            atol=1e-5,
        )
        assert standardizer.is_fitted.item() is True

    @pytest.mark.unit
    def test_require_fitted_raises_when_stats_missing(
        self,
        latent_standardizer_factory: Callable[..., LatentStandardizer],
    ) -> None:
        standardizer = latent_standardizer_factory(
            latent_dimension=3,
            enabled=True,
            require_fitted=True,
        )
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "LatentStandardizer requires fitted latent statistics before use."
            ),
        ):
            standardizer.standardize(torch.zeros(1, 3))


class TestLatentStandardizerValidation:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "mean, std, expected_message",
        [
            pytest.param(
                torch.zeros(2),
                torch.ones(3),
                "Latent stats must have shape (3,), got mean=(2,), std=(3,).",
                id="mean-wrong-shape",
            ),
            pytest.param(
                torch.zeros(3),
                torch.ones(2),
                "Latent stats must have shape (3,), got mean=(3,), std=(2,).",
                id="std-wrong-shape",
            ),
            pytest.param(
                torch.tensor([0.0, float("inf"), 0.0]),
                torch.ones(3),
                "Latent stats must be finite.",
                id="mean-non-finite",
            ),
            pytest.param(
                torch.zeros(3),
                torch.tensor([1.0, 0.0, 1.0]),
                "Latent std must be strictly positive.",
                id="std-zero",
            ),
        ],
    )
    def test_set_stats_rejects_invalid_statistics(
        self,
        latent_standardizer_factory: Callable[..., LatentStandardizer],
        mean: torch.Tensor,
        std: torch.Tensor,
        expected_message: str,
    ) -> None:
        standardizer = latent_standardizer_factory(latent_dimension=3)

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            standardizer.set_stats(mean=mean, std=std)

    @pytest.mark.unit
    def test_fit_rejects_wrong_trailing_dimension(
        self,
        latent_standardizer_factory: Callable[..., LatentStandardizer],
    ) -> None:
        standardizer = latent_standardizer_factory(latent_dimension=3)

        with pytest.raises(
            ValueError,
            match="Latents must have trailing dimension 3, got 2.",
        ):
            standardizer.fit(torch.zeros(4, 2))


class TestLatentStandardizerCheckpointing:
    @pytest.mark.unit
    def test_state_dict_reload_preserves_stats(
        self,
        latent_standardizer_factory: Callable[..., LatentStandardizer],
    ) -> None:
        source = latent_standardizer_factory(latent_dimension=3)
        source.set_stats(
            mean=torch.tensor([1.0, 2.0, 3.0]),
            std=torch.tensor([2.0, 4.0, 5.0]),
        )
        target = latent_standardizer_factory(latent_dimension=3)

        target.load_state_dict(source.state_dict())

        torch.testing.assert_close(target.mean, source.mean)
        torch.testing.assert_close(target.std, source.std)
        assert target.is_fitted.item() is True
