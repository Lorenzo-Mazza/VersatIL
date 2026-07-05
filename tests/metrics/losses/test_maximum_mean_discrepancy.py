"""Tests for versatil.metrics.losses.maximum_mean_discrepancy module."""

import re
from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pytest
import torch

from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.metrics.kernels import KernelType
from versatil.metrics.losses.maximum_mean_discrepancy import (
    BinaryMaximumMeanDiscrepancyLoss,
    ConditionalMaximumMeanDiscrepancyLoss,
    MaximumMeanDiscrepancyLoss,
)
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey


@pytest.mark.unit
class TestMaximumMeanDiscrepancyLossGetRequiredKeys:
    def test_includes_posterior_latent(self):
        loss = MaximumMeanDiscrepancyLoss()
        assert LatentKey.POSTERIOR_LATENT.value in loss.get_required_keys()

    def test_uses_configured_prior_target_key(self):
        loss = MaximumMeanDiscrepancyLoss(prior_target_key=LatentKey.POSTERIOR_MU.value)
        assert LatentKey.POSTERIOR_MU.value in loss.get_required_keys()
        assert LatentKey.POSTERIOR_LATENT.value not in loss.get_required_keys()

    def test_includes_prior_when_not_fixed_gaussian(self):
        loss = MaximumMeanDiscrepancyLoss(use_fixed_gaussian_as_prior=False)
        assert LatentKey.PRIOR_LATENT.value in loss.get_required_keys()

    def test_excludes_prior_when_fixed_gaussian(self):
        loss = MaximumMeanDiscrepancyLoss(use_fixed_gaussian_as_prior=True)
        assert LatentKey.PRIOR_LATENT.value not in loss.get_required_keys()


@pytest.mark.unit
class TestMaximumMeanDiscrepancyLossForward:
    def test_identical_samples_produce_small_mmd(self, rng):
        z = torch.from_numpy(rng.standard_normal((32, 8)).astype(np.float32))
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_LATENT.value: z.clone(),
        }
        loss = MaximumMeanDiscrepancyLoss(weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0, abs=0.01)

    def test_different_samples_produce_larger_mmd(self, rng):
        z_posterior = (
            torch.from_numpy(rng.standard_normal((32, 8)).astype(np.float32)) + 5.0
        )
        z_prior = torch.from_numpy(rng.standard_normal((32, 8)).astype(np.float32))
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: z_posterior,
            LatentKey.PRIOR_LATENT.value: z_prior,
        }
        loss = MaximumMeanDiscrepancyLoss(weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() > 0.01

    def test_uses_configured_prior_target_key_for_matching(self, rng):
        posterior_latent = (
            torch.from_numpy(rng.standard_normal((32, 8)).astype(np.float32)) + 5.0
        )
        posterior_mu = torch.from_numpy(rng.standard_normal((32, 8)).astype(np.float32))
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: posterior_latent,
            LatentKey.POSTERIOR_MU.value: posterior_mu,
            LatentKey.PRIOR_LATENT.value: posterior_mu.clone(),
        }
        loss = MaximumMeanDiscrepancyLoss(
            weight=1.0,
            prior_target_key=LatentKey.POSTERIOR_MU.value,
        )
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0, abs=0.01)
        torch.testing.assert_close(
            output.metadata[MetadataKey.POSTERIOR_Z.value],
            posterior_latent,
        )
        torch.testing.assert_close(
            output.metadata[MetadataKey.POSTERIOR_MU.value],
            posterior_mu,
        )

    def test_prior_regularization_penalizes_non_standard_prior(self, rng):
        z_posterior = torch.from_numpy(rng.standard_normal((32, 8)).astype(np.float32))
        z_prior = (
            torch.from_numpy(rng.standard_normal((32, 8)).astype(np.float32)) + 5.0
        )
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: z_posterior,
            LatentKey.PRIOR_LATENT.value: z_prior,
        }
        loss_no_reg = MaximumMeanDiscrepancyLoss(
            weight=1.0, prior_regularization_weight=0.0
        )
        loss_with_reg = MaximumMeanDiscrepancyLoss(
            weight=1.0, prior_regularization_weight=1.0
        )
        output_no_reg = loss_no_reg(predictions, {})
        output_with_reg = loss_with_reg(predictions, {})
        assert output_with_reg.total_loss.item() >= output_no_reg.total_loss.item()
        assert (
            MetricKey.HYPERPRIOR_MMD_REGULARIZATION.value
            in output_with_reg.component_losses
        )

    def test_raises_when_prior_missing_and_not_fixed(self):
        loss = MaximumMeanDiscrepancyLoss(use_fixed_gaussian_as_prior=False)
        predictions = {LatentKey.POSTERIOR_LATENT.value: torch.zeros(4, 8)}
        with pytest.raises(
            ValueError,
            match="for MaximumMeanDiscrepancyLoss",
        ):
            loss(predictions, {})

    @pytest.mark.parametrize(
        "kernel_type", [KernelType.RBF.value, KernelType.IMQ.value]
    )
    def test_accepts_different_kernel_types(self, rng, kernel_type):
        z = torch.from_numpy(rng.standard_normal((16, 4)).astype(np.float32))
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_LATENT.value: z.clone(),
        }
        loss = MaximumMeanDiscrepancyLoss(kernel_type=kernel_type)
        output = loss(predictions, {})
        assert output.total_loss.item() >= 0.0

    def test_stores_use_median_heuristic(self):
        loss = MaximumMeanDiscrepancyLoss(use_median_heuristic=False)
        assert loss.kernel.use_median_heuristic is False

    def test_stores_use_median_heuristic_default_true(self):
        loss = MaximumMeanDiscrepancyLoss()
        assert loss.kernel.use_median_heuristic is True

    def test_forwards_bandwidth_multipliers_to_kernel(self):
        multipliers = [2.0, 16.0]
        loss = MaximumMeanDiscrepancyLoss(bandwidth_multipliers=multipliers)
        assert loss.kernel.bandwidth_multipliers == multipliers

    def test_fixed_bandwidth_produces_valid_loss(self, rng: np.random.Generator):
        latent_dim = 4
        z_post = torch.from_numpy(
            rng.standard_normal((16, latent_dim)).astype(np.float32)
        )
        z_prior = torch.from_numpy(
            rng.standard_normal((16, latent_dim)).astype(np.float32)
        )
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: z_post,
            LatentKey.PRIOR_LATENT.value: z_prior,
        }
        loss = MaximumMeanDiscrepancyLoss(
            kernel_type=KernelType.IMQ.value,
            bandwidth_multipliers=[2.0 * latent_dim],
            use_median_heuristic=False,
        )
        output = loss(predictions, {})
        assert output.total_loss.item() >= 0.0


@pytest.mark.unit
class TestConditionalMaximumMeanDiscrepancyLossGetRequiredKeys:
    def test_returns_posterior_prior_and_condition_keys(self):
        loss = ConditionalMaximumMeanDiscrepancyLoss()

        assert loss.get_required_keys() == {
            LatentKey.POSTERIOR_LATENT.value,
            LatentKey.PRIOR_LATENT.value,
            LatentKey.PRIOR_CONDITION.value,
        }

    def test_uses_configured_keys(self):
        loss = ConditionalMaximumMeanDiscrepancyLoss(
            prior_target_key=LatentKey.POSTERIOR_MU.value,
            condition_key="custom_condition",
        )

        assert loss.get_required_keys() == {
            LatentKey.POSTERIOR_MU.value,
            LatentKey.PRIOR_LATENT.value,
            "custom_condition",
        }


@pytest.mark.unit
class TestConditionalMaximumMeanDiscrepancyLossForward:
    def test_identical_conditioned_samples_produce_small_mmd(self, rng):
        z = torch.from_numpy(rng.standard_normal((32, 4)).astype(np.float32))
        condition = torch.from_numpy(rng.standard_normal((32, 3)).astype(np.float32))
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_LATENT.value: z.clone(),
            LatentKey.PRIOR_CONDITION.value: condition,
        }
        loss = ConditionalMaximumMeanDiscrepancyLoss(weight=1.0)

        output = loss(predictions, {})

        assert output.total_loss.item() == pytest.approx(0.0, abs=0.01)

    def test_detects_state_conditional_latent_swap(self):
        condition = torch.tensor([[-1.0], [1.0]])
        posterior_latents = torch.tensor([[-1.0], [1.0]])
        prior_latents = torch.tensor([[1.0], [-1.0]])
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: posterior_latents,
            LatentKey.PRIOR_LATENT.value: prior_latents,
            LatentKey.PRIOR_CONDITION.value: condition,
        }
        loss = ConditionalMaximumMeanDiscrepancyLoss(
            weight=1.0,
            state_weight=1.0,
            bandwidth_multipliers=[1.0],
            use_median_heuristic=False,
        )

        output = loss(predictions, {})

        assert output.total_loss.item() > 0.01

    def test_uses_separate_condition_and_latent_kernels(self, rng):
        posterior_latents = torch.from_numpy(
            rng.standard_normal((8, 4)).astype(np.float32)
        )
        prior_latents = torch.from_numpy(rng.standard_normal((8, 4)).astype(np.float32))
        condition = torch.from_numpy(rng.standard_normal((8, 3)).astype(np.float32))
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: posterior_latents,
            LatentKey.PRIOR_LATENT.value: prior_latents,
            LatentKey.PRIOR_CONDITION.value: condition,
        }
        loss = ConditionalMaximumMeanDiscrepancyLoss()

        with (
            patch.object(
                loss.condition_kernel,
                "forward",
                wraps=loss.condition_kernel.forward,
            ) as condition_kernel_spy,
            patch.object(
                loss.latent_kernel,
                "forward",
                wraps=loss.latent_kernel.forward,
            ) as latent_kernel_spy,
        ):
            loss(predictions, {})

        assert condition_kernel_spy.call_count == 1
        assert latent_kernel_spy.call_count == 3

    def test_uses_configured_prior_target_key_for_matching(self, rng):
        posterior_latent = torch.from_numpy(
            rng.standard_normal((16, 4)).astype(np.float32)
        )
        posterior_mu = torch.from_numpy(rng.standard_normal((16, 4)).astype(np.float32))
        condition = torch.from_numpy(rng.standard_normal((16, 3)).astype(np.float32))
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: posterior_latent,
            LatentKey.POSTERIOR_MU.value: posterior_mu,
            LatentKey.PRIOR_LATENT.value: posterior_mu.clone(),
            LatentKey.PRIOR_CONDITION.value: condition,
        }
        loss = ConditionalMaximumMeanDiscrepancyLoss(
            weight=1.0,
            prior_target_key=LatentKey.POSTERIOR_MU.value,
        )

        output = loss(predictions, {})

        assert output.total_loss.item() == pytest.approx(0.0, abs=0.01)
        torch.testing.assert_close(
            output.metadata[MetadataKey.POSTERIOR_Z.value],
            posterior_latent,
        )
        torch.testing.assert_close(
            output.metadata[MetadataKey.POSTERIOR_MU.value],
            posterior_mu,
        )

    def test_includes_prior_condition_metadata(self, rng):
        posterior_latent = torch.from_numpy(
            rng.standard_normal((8, 4)).astype(np.float32)
        )
        prior_latent = torch.from_numpy(rng.standard_normal((8, 4)).astype(np.float32))
        condition = torch.from_numpy(rng.standard_normal((8, 3)).astype(np.float32))
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: posterior_latent,
            LatentKey.PRIOR_LATENT.value: prior_latent,
            LatentKey.PRIOR_CONDITION.value: condition,
        }
        loss = ConditionalMaximumMeanDiscrepancyLoss()

        output = loss(predictions, {})

        torch.testing.assert_close(
            output.metadata[MetadataKey.PRIOR_CONDITION.value],
            condition,
        )

    def test_rejects_negative_state_weight(self):
        with pytest.raises(
            ValueError,
            match=re.escape("state_weight must be non-negative, got -1.0."),
        ):
            ConditionalMaximumMeanDiscrepancyLoss(state_weight=-1.0)

    def test_raises_on_missing_keys(self):
        loss = ConditionalMaximumMeanDiscrepancyLoss()
        predictions = {LatentKey.POSTERIOR_LATENT.value: torch.zeros(4, 8)}

        with pytest.raises(
            ValueError,
            match="for ConditionalMaximumMeanDiscrepancyLoss",
        ):
            loss(predictions, {})

    def test_rejects_batch_size_mismatch(self):
        loss = ConditionalMaximumMeanDiscrepancyLoss()
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: torch.zeros(4, 8),
            LatentKey.PRIOR_LATENT.value: torch.zeros(4, 8),
            LatentKey.PRIOR_CONDITION.value: torch.zeros(3, 2),
        }

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Latent and condition samples must have the same batch size"
            ),
        ):
            loss(predictions, {})


@pytest.mark.unit
class TestMaximumMeanDiscrepancyLossSharedBandwidth:
    def test_all_three_kernel_calls_use_same_bandwidth(
        self, latent_sample_factory: Callable[..., torch.Tensor]
    ):
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: latent_sample_factory(),
            LatentKey.PRIOR_LATENT.value: latent_sample_factory(),
        }
        loss = MaximumMeanDiscrepancyLoss(weight=1.0, use_median_heuristic=True)

        with patch.object(
            loss.kernel, "forward", wraps=loss.kernel.forward
        ) as kernel_spy:
            loss(predictions, {})

        assert kernel_spy.call_count == 3
        bandwidths = [call.kwargs["bandwidth"] for call in kernel_spy.call_args_list]
        assert bandwidths[0] is not None
        assert bandwidths[0] == bandwidths[1] == bandwidths[2]

    def test_shared_bandwidth_equals_resolved_from_combined_samples(
        self, latent_sample_factory: Callable[..., torch.Tensor]
    ):
        z_posterior = latent_sample_factory()
        z_prior = latent_sample_factory()
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: z_posterior,
            LatentKey.PRIOR_LATENT.value: z_prior,
        }
        loss = MaximumMeanDiscrepancyLoss(weight=1.0, use_median_heuristic=True)
        expected_bandwidth = loss.kernel.resolve_base_bandwidth(
            torch.cat([z_posterior, z_prior], dim=0)
        )

        with patch.object(
            loss.kernel, "forward", wraps=loss.kernel.forward
        ) as kernel_spy:
            loss(predictions, {})

        used_bandwidth = kernel_spy.call_args_list[0].kwargs["bandwidth"]
        assert used_bandwidth == pytest.approx(expected_bandwidth, rel=1e-6)

    def test_prior_regularization_uses_own_shared_bandwidth(
        self, latent_sample_factory: Callable[..., torch.Tensor]
    ):
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: latent_sample_factory(),
            LatentKey.PRIOR_LATENT.value: latent_sample_factory(),
        }
        loss = MaximumMeanDiscrepancyLoss(
            weight=1.0,
            prior_regularization_weight=1.0,
            use_median_heuristic=True,
        )

        with patch.object(
            loss.kernel, "forward", wraps=loss.kernel.forward
        ) as kernel_spy:
            loss(predictions, {})

        assert kernel_spy.call_count == 6
        main_bandwidths = [
            call.kwargs["bandwidth"] for call in kernel_spy.call_args_list[:3]
        ]
        regularization_bandwidths = [
            call.kwargs["bandwidth"] for call in kernel_spy.call_args_list[3:]
        ]
        assert main_bandwidths[0] == main_bandwidths[1] == main_bandwidths[2]
        assert (
            regularization_bandwidths[0]
            == regularization_bandwidths[1]
            == regularization_bandwidths[2]
        )
        assert main_bandwidths[0] != regularization_bandwidths[0]


@pytest.mark.unit
class TestBinaryMaximumMeanDiscrepancyLossForward:
    def test_raises_on_missing_key(self):
        loss = BinaryMaximumMeanDiscrepancyLoss()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain '{DecoderOutputKey.BINARY_LOGITS.value}' for BinaryMaximumMeanDiscrepancyLoss."
            ),
        ):
            loss({}, {})

    def test_mmd_is_non_negative(self, rng):
        logits = torch.from_numpy(rng.standard_normal((8, 4, 16)).astype(np.float32))
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        loss = BinaryMaximumMeanDiscrepancyLoss(weight=1.0)
        output = loss(predictions, {})
        # MMD can be slightly negative due to clamping, but result should be >= 0
        assert output.total_loss.item() >= -0.1

    def test_uniform_logits_produce_smaller_mmd_than_extreme(self, rng):
        # Uniform logits: sigmoid(0) = 0.5 => matches Bernoulli(0.5) prior
        uniform_logits = torch.zeros(32, 4, 16)
        # Extreme logits: sigmoid(10) close to 1 => deviates from prior
        extreme_logits = 10.0 * torch.ones(32, 4, 16)
        loss = BinaryMaximumMeanDiscrepancyLoss(weight=1.0)

        output_uniform = loss(
            {DecoderOutputKey.BINARY_LOGITS.value: uniform_logits}, {}
        )
        output_extreme = loss(
            {DecoderOutputKey.BINARY_LOGITS.value: extreme_logits}, {}
        )

        # MMD for uniform should be much smaller (samples match prior)
        assert output_uniform.total_loss.item() < output_extreme.total_loss.item()

    def test_metadata_contains_posterior_z(self, rng):
        logits = torch.from_numpy(rng.standard_normal((8, 4, 16)).astype(np.float32))
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        loss = BinaryMaximumMeanDiscrepancyLoss(weight=1.0)
        output = loss(predictions, {})
        assert MetadataKey.POSTERIOR_Z.value in output.metadata
        # Posterior z should have same shape as logits
        assert output.metadata[MetadataKey.POSTERIOR_Z.value].shape == logits.shape

    @pytest.mark.parametrize("weight", [1.0, 3.0, 0.5])
    def test_weight_scales_total_loss_relative_to_component(self, rng, weight):
        logits = torch.from_numpy(rng.standard_normal((8, 4, 16)).astype(np.float32))
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        loss = BinaryMaximumMeanDiscrepancyLoss(weight=weight)
        output = loss(predictions, {})
        # total_loss = weight * mmd_component
        mmd_component = output.component_losses[MetricKey.BINARY_MMD_LOSS.value].item()
        assert output.total_loss.item() == pytest.approx(
            weight * mmd_component, rel=1e-4
        )

    def test_all_three_kernel_calls_use_same_bandwidth(self, rng: np.random.Generator):
        logits = torch.from_numpy(rng.standard_normal((16, 4, 8)).astype(np.float32))
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        loss = BinaryMaximumMeanDiscrepancyLoss(weight=1.0)

        with patch.object(
            loss.kernel, "forward", wraps=loss.kernel.forward
        ) as kernel_spy:
            loss(predictions, {})

        assert kernel_spy.call_count == 3
        bandwidths = [call.kwargs["bandwidth"] for call in kernel_spy.call_args_list]
        assert bandwidths[0] is not None
        assert bandwidths[0] == bandwidths[1] == bandwidths[2]


class TestMMDBranches:
    def test_missing_prior_latent_raises_without_fixed_gaussian(
        self, rng: np.random.Generator
    ):
        loss = MaximumMeanDiscrepancyLoss(use_fixed_gaussian_as_prior=False)
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: torch.from_numpy(
                rng.standard_normal((4, 8)).astype(np.float32)
            ),
            LatentKey.PRIOR_LATENT.value: None,
        }
        with pytest.raises(ValueError, match="Prior latent is required"):
            loss(predictions, {})

    @pytest.mark.parametrize("kernel_type", ["rbf", "imq"])
    @pytest.mark.parametrize("use_median_heuristic", [True, False])
    def test_kernel_variants_produce_finite_loss(
        self, kernel_type, use_median_heuristic, rng: np.random.Generator
    ):
        loss = MaximumMeanDiscrepancyLoss(
            use_fixed_gaussian_as_prior=True,
            kernel_type=kernel_type,
            use_median_heuristic=use_median_heuristic,
        )
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: torch.from_numpy(
                rng.standard_normal((6, 8)).astype(np.float32)
            )
        }
        output = loss(predictions, {})
        assert output.total_loss.isfinite()
