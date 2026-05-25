"""Tests for versatil.metrics.ot_loss module (integration, real geomloss)."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.metrics.ot_loss import LatentOptimalTransportLoss, OptimalTransportLoss
from versatil.models.decoding.constants import LatentKey

pytest.importorskip("geomloss")


@pytest.fixture
def action_trajectory_factory(
    rng,
) -> Callable[..., torch.Tensor]:
    def factory(
        batch_size: int = 4,
        horizon: int = 8,
        action_dimension: int = 3,
    ) -> torch.Tensor:
        data = rng.standard_normal((batch_size, horizon, action_dimension)).astype(
            np.float32
        )
        return torch.from_numpy(data)

    return factory


@pytest.fixture
def ot_loss_factory() -> Callable[..., OptimalTransportLoss]:
    def factory(
        action_keys: list[str] | None = None,
        weight: float = 1.0,
        p: int = 2,
        blur_fraction: float = 0.1,
        reach_multiplier: float | None = None,
        expected_std: float = 1.0,
        time_scale: float = 1.0,
    ) -> OptimalTransportLoss:
        return OptimalTransportLoss(
            action_keys=action_keys if action_keys is not None else ["position"],
            weight=weight,
            p=p,
            blur_fraction=blur_fraction,
            reach_multiplier=reach_multiplier,
            expected_std=expected_std,
            time_scale=time_scale,
        )

    return factory


@pytest.fixture
def latent_ot_loss_factory() -> Callable[..., LatentOptimalTransportLoss]:
    def factory(
        weight: float = 1.0,
        p: int = 2,
        blur_fraction: float = 0.1,
        reach_multiplier: float | None = None,
    ) -> LatentOptimalTransportLoss:
        return LatentOptimalTransportLoss(
            weight=weight,
            p=p,
            blur_fraction=blur_fraction,
            reach_multiplier=reach_multiplier,
        )

    return factory


@pytest.mark.integration
class TestOptimalTransportLossNumerics:
    def test_identical_predictions_and_targets_give_near_zero_loss(
        self,
        ot_loss_factory: Callable[..., OptimalTransportLoss],
        action_trajectory_factory: Callable[..., torch.Tensor],
    ) -> None:
        loss_fn = ot_loss_factory()
        data = action_trajectory_factory()
        predictions = {"position": data}
        targets = {"position": data.clone()}

        result = loss_fn.forward(predictions=predictions, targets=targets, is_pad=None)

        assert result.total_loss.abs().item() < 1e-3

    def test_loss_decreases_as_predictions_approach_targets(
        self,
        ot_loss_factory: Callable[..., OptimalTransportLoss],
        action_trajectory_factory: Callable[..., torch.Tensor],
    ) -> None:
        loss_fn = ot_loss_factory()
        targets_data = action_trajectory_factory()
        noise = action_trajectory_factory()

        losses = []
        for scale in [2.0, 1.0, 0.5, 0.1]:
            predictions = {"position": targets_data + scale * noise}
            result = loss_fn.forward(
                predictions=predictions,
                targets={"position": targets_data},
                is_pad=None,
            )
            losses.append(result.total_loss.item())

        for i in range(len(losses) - 1):
            assert losses[i] > losses[i + 1]

    def test_gradient_flows_to_predictions(
        self,
        ot_loss_factory: Callable[..., OptimalTransportLoss],
        action_trajectory_factory: Callable[..., torch.Tensor],
    ) -> None:
        loss_fn = ot_loss_factory()
        predictions_data = action_trajectory_factory().requires_grad_(True)
        targets_data = action_trajectory_factory()

        result = loss_fn.forward(
            predictions={"position": predictions_data},
            targets={"position": targets_data},
            is_pad=None,
        )
        result.total_loss.backward()

        assert torch.isfinite(predictions_data.grad).all()
        assert predictions_data.grad.abs().sum().item() > 0

    @pytest.mark.parametrize("blur_fraction", [0.01, 0.05, 0.1, 0.5, 1.0])
    def test_loss_stays_finite_and_non_negative_across_blur_fraction(
        self,
        ot_loss_factory: Callable[..., OptimalTransportLoss],
        action_trajectory_factory: Callable[..., torch.Tensor],
        blur_fraction: float,
    ) -> None:
        loss_fn = ot_loss_factory(blur_fraction=blur_fraction)
        predictions_data = action_trajectory_factory()
        targets_data = action_trajectory_factory()

        result = loss_fn.forward(
            predictions={"position": predictions_data},
            targets={"position": targets_data},
            is_pad=None,
        )

        assert torch.isfinite(result.total_loss).item()
        assert result.total_loss.item() >= -1e-3

    def test_time_scale_zero_is_permutation_invariant(
        self,
        ot_loss_factory: Callable[..., OptimalTransportLoss],
        action_trajectory_factory: Callable[..., torch.Tensor],
    ) -> None:
        loss_fn = ot_loss_factory(time_scale=0.0)
        horizon = 8
        predictions_data = action_trajectory_factory(horizon=horizon)
        targets_data = action_trajectory_factory(horizon=horizon)
        perm = torch.randperm(horizon)
        shuffled_predictions = predictions_data[:, perm, :]

        loss_original = loss_fn.forward(
            predictions={"position": predictions_data},
            targets={"position": targets_data},
            is_pad=None,
        ).total_loss.item()
        loss_shuffled = loss_fn.forward(
            predictions={"position": shuffled_predictions},
            targets={"position": targets_data},
            is_pad=None,
        ).total_loss.item()

        assert abs(loss_original - loss_shuffled) < 1e-4

    def test_large_time_scale_penalizes_temporal_reversal(
        self,
        ot_loss_factory: Callable[..., OptimalTransportLoss],
        action_trajectory_factory: Callable[..., torch.Tensor],
    ) -> None:
        loss_fn = ot_loss_factory(time_scale=10.0)
        predictions_data = action_trajectory_factory()
        targets_data = action_trajectory_factory()
        reversed_predictions = predictions_data.flip(dims=[1])

        loss_original = loss_fn.forward(
            predictions={"position": predictions_data},
            targets={"position": targets_data},
            is_pad=None,
        ).total_loss.item()
        loss_reversed = loss_fn.forward(
            predictions={"position": reversed_predictions},
            targets={"position": targets_data},
            is_pad=None,
        ).total_loss.item()

        assert abs(loss_original - loss_reversed) > 1e-3

    def test_all_padded_batch_does_not_produce_nan(
        self,
        ot_loss_factory: Callable[..., OptimalTransportLoss],
        action_trajectory_factory: Callable[..., torch.Tensor],
    ) -> None:
        loss_fn = ot_loss_factory()
        batch_size, horizon = 2, 4
        predictions_data = action_trajectory_factory(
            batch_size=batch_size, horizon=horizon
        )
        targets_data = action_trajectory_factory(batch_size=batch_size, horizon=horizon)
        is_pad = torch.ones(batch_size, horizon, dtype=torch.bool)

        result = loss_fn.forward(
            predictions={"position": predictions_data},
            targets={"position": targets_data},
            is_pad=is_pad,
        )

        assert torch.isfinite(result.total_loss).item()


@pytest.mark.integration
class TestLatentOptimalTransportLossNumerics:
    def test_identical_latents_give_near_zero_loss(
        self,
        latent_ot_loss_factory: Callable[..., LatentOptimalTransportLoss],
        latent_sample_factory: Callable[..., torch.Tensor],
    ) -> None:
        loss_fn = latent_ot_loss_factory()
        latent = latent_sample_factory()
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: latent,
            LatentKey.PRIOR_LATENT.value: latent.clone(),
        }

        result = loss_fn.forward(predictions=predictions, targets={}, is_pad=None)

        assert result.total_loss.abs().item() < 1e-3

    def test_distant_latents_give_larger_loss_than_close_latents(
        self,
        latent_ot_loss_factory: Callable[..., LatentOptimalTransportLoss],
        latent_sample_factory: Callable[..., torch.Tensor],
    ) -> None:
        loss_fn = latent_ot_loss_factory()
        batch_size, latent_dimension = 32, 8
        prior = latent_sample_factory(
            batch_size=batch_size, latent_dimension=latent_dimension
        )
        close_posterior = prior + 0.1 * latent_sample_factory(
            batch_size=batch_size, latent_dimension=latent_dimension
        )
        far_posterior = prior + 5.0 * latent_sample_factory(
            batch_size=batch_size, latent_dimension=latent_dimension
        )

        loss_close = loss_fn.forward(
            predictions={
                LatentKey.POSTERIOR_LATENT.value: close_posterior,
                LatentKey.PRIOR_LATENT.value: prior,
            },
            targets={},
            is_pad=None,
        ).total_loss.item()
        loss_far = loss_fn.forward(
            predictions={
                LatentKey.POSTERIOR_LATENT.value: far_posterior,
                LatentKey.PRIOR_LATENT.value: prior,
            },
            targets={},
            is_pad=None,
        ).total_loss.item()

        assert loss_far > loss_close

    def test_unbalanced_differs_from_balanced(
        self,
        latent_ot_loss_factory: Callable[..., LatentOptimalTransportLoss],
        latent_sample_factory: Callable[..., torch.Tensor],
    ) -> None:
        batch_size, latent_dimension = 32, 8
        prior = latent_sample_factory(
            batch_size=batch_size, latent_dimension=latent_dimension
        )
        posterior_base = latent_sample_factory(
            batch_size=batch_size - 1, latent_dimension=latent_dimension
        )
        outlier = 20.0 * latent_sample_factory(
            batch_size=1, latent_dimension=latent_dimension
        )
        posterior_with_outlier = torch.cat([posterior_base, outlier], dim=0)
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: posterior_with_outlier,
            LatentKey.PRIOR_LATENT.value: prior,
        }

        balanced = latent_ot_loss_factory(reach_multiplier=None)
        unbalanced = latent_ot_loss_factory(reach_multiplier=1.0)

        loss_balanced = balanced.forward(
            predictions=predictions, targets={}, is_pad=None
        ).total_loss.item()
        loss_unbalanced = unbalanced.forward(
            predictions=predictions, targets={}, is_pad=None
        ).total_loss.item()

        # With an extreme outlier, balanced OT is forced to transport mass
        # to the distant point at high cost; unbalanced OT can drop it at
        # bounded KL cost, yielding a strictly smaller loss.
        assert loss_unbalanced < loss_balanced

    def test_gradient_flows_to_posterior(
        self,
        latent_ot_loss_factory: Callable[..., LatentOptimalTransportLoss],
        latent_sample_factory: Callable[..., torch.Tensor],
    ) -> None:
        loss_fn = latent_ot_loss_factory()
        posterior = latent_sample_factory().requires_grad_(True)
        prior = latent_sample_factory()

        result = loss_fn.forward(
            predictions={
                LatentKey.POSTERIOR_LATENT.value: posterior,
                LatentKey.PRIOR_LATENT.value: prior,
            },
            targets={},
            is_pad=None,
        )
        result.total_loss.backward()

        assert torch.isfinite(posterior.grad).all()
        assert posterior.grad.abs().sum().item() > 0
