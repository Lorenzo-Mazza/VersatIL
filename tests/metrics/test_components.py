"""Tests for versatil.metrics.components module."""

import math
import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from versatil.configs.experiment import ExperimentConfig
from versatil.data.constants import BinaryGripperRange, GripperType, SampleKey
from versatil.data.metadata import (
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
)
from versatil.metrics.base import BaseLoss
from versatil.metrics.components import (
    ActionTokenLoss,
    BinaryKLDivergenceLoss,
    BinaryMaximumMeanDiscrepancyLoss,
    GaussianEntropyLoss,
    GaussianMixtureNLLoss,
    GripperLoss,
    GripperMixtureNLLoss,
    KLDivergenceLoss,
    MaximumMeanDiscrepancyLoss,
    MetadataPassthrough,
    MoELoss,
    PhaseClassificationLoss,
    PosteriorGeometryLoss,
    PriorDenoisingLoss,
    RegressionLoss,
    TrajectoryLengthLoss,
    TrajectorySmoothness,
    VICLatentLoss,
    VQCommitmentLoss,
    VQPriorCrossEntropyLoss,
)
from versatil.metrics.composite import CompositeLoss
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.metrics.kernels import KernelType
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey
from versatil.training.callbacks.expert_usage import ExpertUsageCallback


@pytest.fixture
def binary_gripper_metadata_factory():
    def factory(
        gripper_type: str = GripperType.BINARY.value,
        binary_gripper_range: str = BinaryGripperRange.ZERO_ONE.value,
    ) -> dict[str, GripperActionMetadata]:
        return {
            "gripper": GripperActionMetadata(
                gripper_type=gripper_type,
                raw_data_column_keys=["gripper_state"],
                storage_dimension=1,
                prediction_dimension=1,
                needs_normalization=False,
                dtype="int32",
                binary_gripper_range=binary_gripper_range,
            )
        }

    return factory


@pytest.fixture
def continuous_gripper_metadata_factory():
    def factory() -> dict[str, GripperActionMetadata]:
        return {
            "gripper": GripperActionMetadata(
                gripper_type=GripperType.CONTINUOUS.value,
                raw_data_column_keys=["gripper_state"],
                storage_dimension=1,
                prediction_dimension=1,
                needs_normalization=True,
                dtype="float32",
            )
        }

    return factory


@pytest.mark.unit
class TestRegressionLossInit:
    def test_stores_action_keys(self):
        loss = RegressionLoss(action_keys=["position", "orientation"])
        assert loss.action_keys == ["position", "orientation"]

    @pytest.mark.parametrize(
        "mse_weight, l1_weight, huber_weight",
        [
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.3, 0.2),
        ],
    )
    def test_stores_loss_weights(self, mse_weight, l1_weight, huber_weight):
        loss = RegressionLoss(
            action_keys=["position"],
            mse_weight=mse_weight,
            l1_weight=l1_weight,
            huber_weight=huber_weight,
        )
        assert loss.mse_weight == mse_weight
        assert loss.l1_weight == l1_weight
        assert loss.huber_weight == huber_weight


@pytest.mark.unit
class TestRegressionLossGetRequiredKeys:
    def test_returns_action_keys_as_set(self):
        loss = RegressionLoss(action_keys=["position", "orientation"])
        assert loss.get_required_keys() == {"position", "orientation"}


@pytest.mark.unit
class TestRegressionLossForward:
    def test_mse_only_computes_correct_value(self, action_tensor_factory):
        predictions = {"position": torch.tensor([[[1.0, 2.0]]])}
        targets = {"position": torch.tensor([[[3.0, 4.0]]])}
        loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        output = loss(predictions, targets)
        # MSE = ((1-3)^2 + (2-4)^2) / 2 = (4 + 4) / 2 = 4.0
        expected_mse = F.mse_loss(predictions["position"], targets["position"])
        assert output.total_loss.item() == pytest.approx(expected_mse.item())

    def test_l1_only_computes_correct_value(self):
        predictions = {"position": torch.tensor([[[1.0, 2.0]]])}
        targets = {"position": torch.tensor([[[3.0, 5.0]]])}
        loss = RegressionLoss(
            action_keys=["position"],
            mse_weight=0.0,
            l1_weight=1.0,
        )
        output = loss(predictions, targets)
        # L1 = (|1-3| + |2-5|) / 2 = (2 + 3) / 2 = 2.5
        expected = F.l1_loss(predictions["position"], targets["position"])
        assert output.total_loss.item() == pytest.approx(expected.item())

    def test_huber_only_computes_correct_value(self):
        predictions = {"position": torch.tensor([[[0.0]]])}
        targets = {"position": torch.tensor([[[0.3]]])}
        delta = 1.0
        loss = RegressionLoss(
            action_keys=["position"],
            mse_weight=0.0,
            huber_weight=1.0,
            huber_delta=delta,
        )
        output = loss(predictions, targets)
        expected = F.huber_loss(
            predictions["position"], targets["position"], delta=delta
        )
        assert output.total_loss.item() == pytest.approx(expected.item())

    def test_per_key_weights_scale_loss(self):
        predictions = {
            "position": torch.ones(1, 1, 3),
            "orientation": torch.ones(1, 1, 1),
        }
        targets = {
            "position": torch.zeros(1, 1, 3),
            "orientation": torch.zeros(1, 1, 1),
        }
        loss = RegressionLoss(
            action_keys=["position", "orientation"],
            mse_weight=1.0,
            per_key_weights={"position": 2.0, "orientation": 0.5},
        )
        output = loss(predictions, targets)
        # position MSE = 1.0, orientation MSE = 1.0
        # total = 1.0 * 2.0 * 1.0 + 1.0 * 0.5 * 1.0 = 2.5
        assert output.total_loss.item() == pytest.approx(2.5)

    def test_component_losses_are_keyed_correctly(self):
        predictions = {"position": torch.ones(1, 1, 3)}
        targets = {"position": torch.zeros(1, 1, 3)}
        loss = RegressionLoss(
            action_keys=["position"],
            mse_weight=1.0,
            l1_weight=1.0,
        )
        output = loss(predictions, targets)
        assert f"position_{MetricKey.MSE_LOSS.value}" in output.component_losses
        assert f"position_{MetricKey.L1_LOSS.value}" in output.component_losses

    def test_raises_on_missing_key(self):
        predictions = {"wrong_key": torch.ones(1, 1, 3)}
        targets = {"position": torch.zeros(1, 1, 3)}
        loss = RegressionLoss(action_keys=["position"])
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Predictions and targets must contain key 'position' for RegressionLoss."
            ),
        ):
            loss(predictions, targets)

    def test_padding_mask_excludes_padded_positions(self):
        batch_size, horizon, action_dim = 1, 4, 2
        predictions = {"position": torch.ones(batch_size, horizon, action_dim)}
        targets = {"position": torch.zeros(batch_size, horizon, action_dim)}
        is_pad = torch.tensor([[False, False, True, True]])
        loss_no_pad = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        loss_with_pad = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        output_no_pad = loss_no_pad(predictions, targets)
        output_with_pad = loss_with_pad(predictions, targets, is_pad=is_pad)
        # Padded positions contribute nothing; with half valid, the per-position
        # loss is the same (all ones vs zeros = MSE 1.0 per element).
        # reduce_loss_with_padding divides by pad_mask.sum() (number of valid positions).
        # masked_loss.sum() = 1 * 2 * 2 = 4; pad_mask.sum() = 2 (after unsqueeze)
        # result = 4 / 2 = 2.0
        # Without padding: mean over all elements = 1.0
        # The key behavioral check: padded outputs don't affect the loss
        assert output_with_pad.total_loss.item() != output_no_pad.total_loss.item()

    def test_handles_long_dtype_targets(self):
        predictions = {"gripper": torch.tensor([[[0.8]]])}
        targets = {"gripper": torch.tensor([[[1]]], dtype=torch.long)}
        loss = RegressionLoss(action_keys=["gripper"], mse_weight=1.0)
        output = loss(predictions, targets)
        expected = F.mse_loss(predictions["gripper"], targets["gripper"].float())
        assert output.total_loss.item() == pytest.approx(expected.item())


@pytest.mark.unit
class TestGripperLossInit:
    def test_binary_gripper_stores_type(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        loss = GripperLoss(key="gripper", actions_metadata=metadata)
        assert loss.gripper_type == GripperType.BINARY.value
        assert loss.binary_gripper_range == BinaryGripperRange.ZERO_ONE.value

    def test_continuous_gripper_stores_type(self, continuous_gripper_metadata_factory):
        metadata = continuous_gripper_metadata_factory()
        loss = GripperLoss(key="gripper", actions_metadata=metadata)
        assert loss.gripper_type == GripperType.CONTINUOUS.value

    def test_raises_on_missing_key(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        with pytest.raises(
            ValueError,
            match=re.escape("wrong_key is not available to the action space"),
        ):
            GripperLoss(key="wrong_key", actions_metadata=metadata)

    def test_on_the_fly_metadata_extracts_gripper_type(self):
        source = GripperObservationMetadata(
            raw_data_column_keys=["gripper"],
            dimension=1,
            dtype="int32",
            needs_normalization=False,
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.MINUS_ONE_ONE.value,
        )
        on_the_fly = OnTheFlyActionMetadata(source_metadata=source)
        loss = GripperLoss(key="gripper", actions_metadata={"gripper": on_the_fly})
        assert loss.gripper_type == GripperType.BINARY.value
        assert loss.binary_gripper_range == BinaryGripperRange.MINUS_ONE_ONE.value


@pytest.mark.unit
class TestGripperLossRequiresActionSpaceTargets:
    def test_true_when_bce_weight_positive(self, binary_gripper_metadata_factory):
        loss = GripperLoss(
            key="gripper",
            actions_metadata=binary_gripper_metadata_factory(),
            bce_weight=0.05,
        )
        assert loss.requires_action_space_targets is True

    def test_false_when_bce_weight_zero(self, binary_gripper_metadata_factory):
        loss = GripperLoss(
            key="gripper",
            actions_metadata=binary_gripper_metadata_factory(),
            bce_weight=0.0,
            mse_weight=1.0,
        )
        assert loss.requires_action_space_targets is False


class TestGripperLossGetRequiredKeys:
    def test_returns_gripper_key(self, binary_gripper_metadata_factory):
        loss = GripperLoss(
            key="gripper", actions_metadata=binary_gripper_metadata_factory()
        )
        assert loss.get_required_keys() == {"gripper"}


@pytest.mark.unit
class TestGripperLossForward:
    def test_binary_gripper_computes_bce(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        loss = GripperLoss(key="gripper", actions_metadata=metadata, bce_weight=1.0)
        predictions = {"gripper": torch.tensor([[[0.0]]])}
        targets = {"gripper": torch.tensor([[[1.0]]])}
        output = loss(predictions, targets)
        expected_bce = F.binary_cross_entropy_with_logits(
            predictions["gripper"],
            targets["gripper"].float(),
        )
        assert output.total_loss.item() == pytest.approx(expected_bce.item())
        assert MetricKey.GRIPPER_BCE.value in output.component_losses

    def test_binary_gripper_minus_one_one_range_normalizes_targets(
        self, binary_gripper_metadata_factory
    ):
        metadata = binary_gripper_metadata_factory(
            binary_gripper_range=BinaryGripperRange.MINUS_ONE_ONE.value
        )
        loss = GripperLoss(key="gripper", actions_metadata=metadata, bce_weight=1.0)
        predictions = {"gripper": torch.tensor([[[0.0]]])}
        targets = {"gripper": torch.tensor([[[-1.0]]])}
        output = loss(predictions, targets)
        # -1 should be mapped to 0.0: (-1 + 1) / 2 = 0
        expected_bce = F.binary_cross_entropy_with_logits(
            torch.tensor([[[0.0]]]),
            torch.tensor([[[0.0]]]),
        )
        assert output.total_loss.item() == pytest.approx(expected_bce.item())

    def test_continuous_gripper_computes_mse(self, continuous_gripper_metadata_factory):
        metadata = continuous_gripper_metadata_factory()
        loss = GripperLoss(key="gripper", actions_metadata=metadata, mse_weight=1.0)
        predictions = {"gripper": torch.tensor([[[0.5]]])}
        targets = {"gripper": torch.tensor([[[1.0]]])}
        output = loss(predictions, targets)
        expected_mse = F.mse_loss(predictions["gripper"], targets["gripper"])
        assert output.total_loss.item() == pytest.approx(expected_mse.item())
        assert MetricKey.GRIPPER_MSE.value in output.component_losses

    def test_raises_on_missing_key(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        loss = GripperLoss(key="gripper", actions_metadata=metadata)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Predictions and targets must contain key 'gripper' for GripperLoss."
            ),
        ):
            loss({"wrong": torch.tensor(1.0)}, {"wrong": torch.tensor(1.0)})


@pytest.mark.unit
class TestGaussianEntropyLossInit:
    def test_raises_if_key_does_not_contain_logvar(self):
        with pytest.raises(
            ValueError,
            match=re.escape("GaussianEntropyLoss expects a logvar key, got 'mu'."),
        ):
            GaussianEntropyLoss(key="mu")

    def test_stores_parameters(self):
        loss = GaussianEntropyLoss(
            key=LatentKey.PRIOR_LOGVAR.value,
            weight=0.05,
            logvar_min=-3.0,
            logvar_max=3.0,
            bound_weight=2.0,
        )
        assert loss.weight == 0.05
        assert loss.logvar_min == -3.0
        assert loss.logvar_max == 3.0
        assert loss.bound_weight == 2.0


@pytest.mark.unit
class TestGaussianEntropyLossForward:
    def test_entropy_computation_is_correct(self):
        latent_dim = 4
        logvar = torch.zeros(2, latent_dim)  # logvar=0 => variance=1
        # H(N(mu, sigma^2)) = 0.5 * sum_d (1 + log(2*pi) + logvar_d)
        # With logvar=0: per-sample entropy = 0.5 * latent_dim * (1 + log(2*pi))
        expected_entropy_per_sample = 0.5 * latent_dim * (1 + math.log(2 * math.pi))
        expected_mean_entropy = expected_entropy_per_sample
        loss = GaussianEntropyLoss(
            key=LatentKey.PRIOR_LOGVAR.value,
            weight=1.0,
            bound_weight=0.0,
        )
        predictions = {LatentKey.PRIOR_LOGVAR.value: logvar}
        output = loss(predictions, {})
        # total_loss = -weight * entropy_mean + 0.0 * bound
        assert output.total_loss.item() == pytest.approx(-expected_mean_entropy)

    def test_bound_violation_penalizes_out_of_range_logvar(self):
        logvar_max = 2.0
        logvar = torch.tensor([[3.0, 3.0]])  # exceeds logvar_max by 1.0
        loss = GaussianEntropyLoss(
            key=LatentKey.PRIOR_LOGVAR.value,
            weight=0.0,  # disable entropy contribution
            logvar_max=logvar_max,
            logvar_min=-4.0,
            bound_weight=1.0,
        )
        output = loss({LatentKey.PRIOR_LOGVAR.value: logvar}, {})
        # bound_violation = relu(3 - 2)^2.mean() + relu(-4 - 3)^2.mean() = 1.0 + 0 = 1.0
        assert output.total_loss.item() == pytest.approx(1.0)

    def test_raises_on_missing_key(self):
        loss = GaussianEntropyLoss(key=LatentKey.PRIOR_LOGVAR.value)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain '{LatentKey.PRIOR_LOGVAR.value}' for GaussianEntropyLoss."
            ),
        ):
            loss({}, {})

    def test_component_loss_reports_entropy(self):
        loss = GaussianEntropyLoss(key=LatentKey.PRIOR_LOGVAR.value, weight=1.0)
        logvar = torch.zeros(2, 4)
        output = loss({LatentKey.PRIOR_LOGVAR.value: logvar}, {})
        component_key = f"{LatentKey.PRIOR_LOGVAR.value}_{MetricKey.ENTROPY.value}"
        assert component_key in output.component_losses


@pytest.mark.unit
class TestKLDivergenceLossGetRequiredKeys:
    def test_returns_latent_keys(self):
        loss = KLDivergenceLoss()
        keys = loss.get_required_keys()
        assert LatentKey.POSTERIOR_LATENT.value in keys
        assert LatentKey.PRIOR_LATENT.value in keys
        assert LatentKey.POSTERIOR_MU.value in keys
        assert LatentKey.POSTERIOR_LOGVAR.value in keys


@pytest.mark.unit
class TestKLDivergenceLossForwardClosedForm:
    def test_kl_is_zero_for_identical_distributions(self):
        batch_size, latent_dim = 4, 8
        mu = torch.zeros(batch_size, latent_dim)
        logvar = torch.zeros(batch_size, latent_dim)
        z = torch.zeros(batch_size, latent_dim)
        predictions = {
            LatentKey.POSTERIOR_MU.value: mu,
            LatentKey.POSTERIOR_LOGVAR.value: logvar,
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_MU.value: mu.clone(),
            LatentKey.PRIOR_LOGVAR.value: logvar.clone(),
            LatentKey.PRIOR_LATENT.value: z.clone(),
        }
        loss = KLDivergenceLoss(weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_kl_is_positive_for_different_distributions(self, rng):
        batch_size, latent_dim = 4, 8
        mu_post = torch.from_numpy(
            rng.standard_normal((batch_size, latent_dim)).astype(np.float32)
        )
        logvar_post = torch.from_numpy(
            rng.standard_normal((batch_size, latent_dim)).astype(np.float32)
        )
        mu_prior = torch.zeros(batch_size, latent_dim)
        logvar_prior = torch.zeros(batch_size, latent_dim)
        z = mu_post + torch.exp(0.5 * logvar_post)
        predictions = {
            LatentKey.POSTERIOR_MU.value: mu_post,
            LatentKey.POSTERIOR_LOGVAR.value: logvar_post,
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_MU.value: mu_prior,
            LatentKey.PRIOR_LOGVAR.value: logvar_prior,
            LatentKey.PRIOR_LATENT.value: torch.zeros(batch_size, latent_dim),
        }
        loss = KLDivergenceLoss(weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() > 0

    def test_weight_scales_kl(self):
        batch_size, latent_dim = 2, 4
        mu_post = torch.ones(batch_size, latent_dim)
        logvar_post = torch.zeros(batch_size, latent_dim)
        mu_prior = torch.zeros(batch_size, latent_dim)
        logvar_prior = torch.zeros(batch_size, latent_dim)
        z = mu_post.clone()
        predictions = {
            LatentKey.POSTERIOR_MU.value: mu_post,
            LatentKey.POSTERIOR_LOGVAR.value: logvar_post,
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_MU.value: mu_prior,
            LatentKey.PRIOR_LOGVAR.value: logvar_prior,
            LatentKey.PRIOR_LATENT.value: torch.zeros_like(z),
        }
        loss_w1 = KLDivergenceLoss(weight=1.0)
        loss_w5 = KLDivergenceLoss(weight=5.0)
        output_w1 = loss_w1(predictions, {})
        output_w5 = loss_w5(predictions, {})
        assert output_w5.total_loss.item() == pytest.approx(
            5.0 * output_w1.total_loss.item(), rel=1e-5
        )

    def test_prior_regularization_adds_kl_to_standard_gaussian(self):
        batch_size, latent_dim = 4, 8
        # Prior = N(2, 1) so KL(prior || N(0,I)) > 0
        mu_post = torch.zeros(batch_size, latent_dim)
        logvar_post = torch.zeros(batch_size, latent_dim)
        mu_prior = 2.0 * torch.ones(batch_size, latent_dim)
        logvar_prior = torch.zeros(batch_size, latent_dim)
        z = torch.zeros(batch_size, latent_dim)
        predictions = {
            LatentKey.POSTERIOR_MU.value: mu_post,
            LatentKey.POSTERIOR_LOGVAR.value: logvar_post,
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_MU.value: mu_prior,
            LatentKey.PRIOR_LOGVAR.value: logvar_prior,
            LatentKey.PRIOR_LATENT.value: z.clone(),
        }
        loss_no_reg = KLDivergenceLoss(weight=1.0, prior_regularization_weight=0.0)
        loss_with_reg = KLDivergenceLoss(weight=1.0, prior_regularization_weight=1.0)
        output_no_reg = loss_no_reg(predictions, {})
        output_with_reg = loss_with_reg(predictions, {})
        assert output_with_reg.total_loss.item() > output_no_reg.total_loss.item()
        assert (
            MetricKey.HYPERPRIOR_KL_REGULARIZATION.value
            in output_with_reg.component_losses
        )

    def test_metadata_includes_latent_variables(self):
        batch_size, latent_dim = 2, 4
        mu = torch.zeros(batch_size, latent_dim)
        logvar = torch.zeros(batch_size, latent_dim)
        z = torch.zeros(batch_size, latent_dim)
        predictions = {
            LatentKey.POSTERIOR_MU.value: mu,
            LatentKey.POSTERIOR_LOGVAR.value: logvar,
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_MU.value: mu.clone(),
            LatentKey.PRIOR_LOGVAR.value: logvar.clone(),
            LatentKey.PRIOR_LATENT.value: z.clone(),
        }
        loss = KLDivergenceLoss(weight=1.0)
        output = loss(predictions, {})
        assert MetadataKey.POSTERIOR_Z.value in output.metadata
        assert MetadataKey.POSTERIOR_MU.value in output.metadata
        assert MetadataKey.POSTERIOR_LOGVAR.value in output.metadata
        assert MetadataKey.PRIOR_Z.value in output.metadata
        assert MetadataKey.PRIOR_MU.value in output.metadata
        assert MetadataKey.PRIOR_LOGVAR.value in output.metadata


@pytest.mark.unit
class TestKLDivergenceLossForwardWithLogProb:
    def test_uses_log_prob_path_when_prior_log_prob_present(self):
        batch_size, latent_dim = 4, 8
        mu = torch.zeros(batch_size, latent_dim)
        logvar = torch.zeros(batch_size, latent_dim)
        z = torch.zeros(batch_size, latent_dim)
        # log_prob of z under N(0,I) prior
        prior_log_prob = (
            torch.distributions.Normal(torch.zeros(latent_dim), torch.ones(latent_dim))
            .log_prob(z)
            .sum(dim=-1)
        )
        predictions = {
            LatentKey.POSTERIOR_MU.value: mu,
            LatentKey.POSTERIOR_LOGVAR.value: logvar,
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_LOG_PROB.value: prior_log_prob,
        }
        loss = KLDivergenceLoss(weight=1.0)
        output = loss(predictions, {})
        # KL = E_q[log q - log p], both N(0,I), evaluated at z=0 => KL ≈ 0
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-4)


@pytest.mark.unit
class TestBinaryKLDivergenceLossGetRequiredKeys:
    def test_returns_binary_logits_key(self):
        loss = BinaryKLDivergenceLoss()
        assert loss.get_required_keys() == {DecoderOutputKey.BINARY_LOGITS.value}


@pytest.mark.unit
class TestBinaryKLDivergenceLossForward:
    def test_uniform_logits_produce_near_zero_kl(self):
        # sigmoid(0) = 0.5 => Bernoulli(0.5) matches uniform prior
        logits = torch.zeros(4, 3, 8)  # (B, T, H)
        loss = BinaryKLDivergenceLoss(weight=1.0, entropy_weight=0.0, free_bits=0.0)
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        output = loss(predictions, {})
        assert output.component_losses[
            MetricKey.RAW_KL_DIVERGENCE.value
        ].item() == pytest.approx(0.0, abs=1e-5)

    def test_extreme_logits_produce_positive_kl(self):
        logits = 10.0 * torch.ones(4, 3, 8)  # sigmoid(10) ≈ 1
        loss = BinaryKLDivergenceLoss(weight=1.0, entropy_weight=0.0, free_bits=0.0)
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        output = loss(predictions, {})
        assert output.component_losses[MetricKey.RAW_KL_DIVERGENCE.value].item() > 0.1

    def test_free_bits_clamps_kl(self):
        logits = torch.zeros(4, 3, 8)  # KL ≈ 0
        free_bits = 1.0
        loss = BinaryKLDivergenceLoss(
            weight=1.0, entropy_weight=0.0, free_bits=free_bits
        )
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        output = loss(predictions, {})
        assert output.component_losses[
            MetricKey.CLAMPED_KL_DIVERGENCE.value
        ].item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("free_bits", [0.5, 1.0, 2.0])
    def test_free_bits_reduces_effective_kl_below_threshold(self, free_bits):
        # Extreme logits: sigmoid(5) close to 1 => large per-bit KL
        logits = 5.0 * torch.ones(4, 3, 8)  # (B, T, H)
        loss_no_free = BinaryKLDivergenceLoss(
            weight=1.0, entropy_weight=0.0, free_bits=0.0
        )
        loss_with_free = BinaryKLDivergenceLoss(
            weight=1.0, entropy_weight=0.0, free_bits=free_bits
        )
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        output_no_free = loss_no_free(predictions, {})
        output_with_free = loss_with_free(predictions, {})

        raw_kl = output_no_free.component_losses[
            MetricKey.RAW_KL_DIVERGENCE.value
        ].item()
        clamped_kl = output_with_free.component_losses[
            MetricKey.CLAMPED_KL_DIVERGENCE.value
        ].item()
        # Raw KL should exceed free_bits for extreme logits
        assert raw_kl > free_bits
        # Clamped KL = mean(max(0, kl_per_token - free_bits)) < raw_kl
        assert clamped_kl < raw_kl
        # Mathematically: clamped = raw - free_bits (since all tokens have same KL)
        assert clamped_kl == pytest.approx(raw_kl - free_bits, abs=1e-4)

    def test_raises_on_missing_key(self):
        loss = BinaryKLDivergenceLoss()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain key '{DecoderOutputKey.BINARY_LOGITS.value}' for BinaryKLDivergenceLoss."
            ),
        ):
            loss({}, {})

    def test_latent_code_usage_tracked_when_available(self):
        logits = torch.zeros(2, 3, 4)  # (B, T, H)
        latent_codes = torch.zeros(2, 3, 16)  # (B, T, 2^H)
        latent_codes[:, :, 0] = 1.0  # all use code 0
        predictions = {
            DecoderOutputKey.BINARY_LOGITS.value: logits,
            DecoderOutputKey.LATENT_CODES.value: latent_codes,
        }
        loss = BinaryKLDivergenceLoss(weight=1.0, latent_bits=4)
        output = loss(predictions, {})
        assert MetricKey.LATENT_CODE_USAGE.value in output.component_losses


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


@pytest.mark.unit
class TestTrajectoryLengthLossForward:
    def test_identical_trajectories_produce_zero_loss(self):
        trajectory = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        predictions = {"position": trajectory}
        targets = {"position": trajectory.clone()}
        loss = TrajectoryLengthLoss(action_key="position", weight=1.0)
        output = loss(predictions, targets)
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_length_trajectories_produce_positive_loss(self):
        pred = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
        target = torch.tensor([[[0.0, 0.0], [3.0, 0.0], [6.0, 0.0]]])
        predictions = {"position": pred}
        targets = {"position": target}
        loss = TrajectoryLengthLoss(action_key="position", weight=1.0)
        output = loss(predictions, targets)
        # pred length = 1 + 1 = 2, target length = 3 + 3 = 6
        # Per step norms: pred = [1, 1], target = [3, 3]
        # Mean of norms: pred = 1.0, target = 3.0
        # (1.0 - 3.0)^2 = 4.0
        assert output.total_loss.item() == pytest.approx(4.0)
        assert MetricKey.LENGTH_LOSS.value in output.component_losses

    def test_raises_on_missing_key(self):
        loss = TrajectoryLengthLoss(action_key="position")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Predictions and targets must contain key 'position' for TrajectoryLengthLoss."
            ),
        ):
            loss({"wrong": torch.zeros(1)}, {"wrong": torch.zeros(1)})

    def test_get_required_keys(self):
        loss = TrajectoryLengthLoss(action_key="position")
        assert loss.get_required_keys() == {"position"}


@pytest.mark.unit
class TestTrajectorySmoothnessForward:
    def test_linear_trajectory_has_zero_smoothness(self):
        # Constant velocity => zero acceleration
        trajectory = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
        predictions = {"position": trajectory}
        loss = TrajectorySmoothness(action_key="position", weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_non_linear_trajectory_has_positive_smoothness(self):
        # Accelerating trajectory
        trajectory = torch.tensor([[[0.0], [1.0], [4.0], [9.0]]])
        predictions = {"position": trajectory}
        loss = TrajectorySmoothness(action_key="position", weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() > 0.0

    def test_too_short_trajectory_returns_zero(self):
        trajectory = torch.tensor([[[0.0], [1.0]]])  # Only 2 timesteps
        predictions = {"position": trajectory}
        loss = TrajectorySmoothness(action_key="position", weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0)

    def test_get_required_keys_returns_empty_set(self):
        loss = TrajectorySmoothness(action_key="position")
        assert loss.get_required_keys() == set()

    def test_raises_on_missing_key(self):
        loss = TrajectorySmoothness(action_key="position")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Predictions must contain key 'position' for TrajectorySmoothness loss."
            ),
        ):
            loss({"wrong": torch.zeros(1)}, {})


@pytest.mark.unit
class TestPhaseClassificationLossGetRequiredKeys:
    def test_returns_phase_key(self):
        loss = PhaseClassificationLoss(key="phase_label")
        assert loss.get_required_keys() == {"phase_label"}


@pytest.mark.unit
class TestPhaseClassificationLossForward:
    def test_perfect_predictions_produce_low_cross_entropy(self):
        batch_size, horizon, num_phases = 2, 3, 4
        labels = torch.zeros(batch_size, horizon, dtype=torch.long)
        logits = torch.zeros(batch_size, horizon, num_phases)
        logits[:, :, 0] = 100.0  # strong signal for class 0
        predictions = {"phase_label": logits}
        targets = {"phase_label": labels}
        loss = PhaseClassificationLoss(
            key="phase_label",
            cross_entropy_weight=1.0,
            entropy_weight=0.0,
            label_smoothing=0.0,
        )
        output = loss(predictions, targets)
        assert output.total_loss.item() < 0.01

    def test_random_predictions_produce_higher_loss(self, rng):
        batch_size, horizon, num_phases = 4, 5, 3
        logits_data = rng.standard_normal((batch_size, horizon, num_phases)).astype(
            np.float32
        )
        logits = torch.from_numpy(logits_data)
        labels = torch.zeros(batch_size, horizon, dtype=torch.long)
        predictions = {"phase_label": logits}
        targets = {"phase_label": labels}
        loss = PhaseClassificationLoss(
            key="phase_label",
            cross_entropy_weight=1.0,
            entropy_weight=0.0,
            label_smoothing=0.0,
        )
        output = loss(predictions, targets)
        assert output.total_loss.item() > 0.1

    def test_entropy_regularization_subtracts_from_loss(self, rng):
        batch_size, horizon, num_phases = 4, 5, 3
        logits_data = rng.standard_normal((batch_size, horizon, num_phases)).astype(
            np.float32
        )
        logits = torch.from_numpy(logits_data)
        labels = torch.zeros(batch_size, horizon, dtype=torch.long)
        predictions = {"phase_label": logits}
        targets = {"phase_label": labels}
        loss_no_entropy = PhaseClassificationLoss(
            key="phase_label", cross_entropy_weight=1.0, entropy_weight=0.0
        )
        loss_with_entropy = PhaseClassificationLoss(
            key="phase_label", cross_entropy_weight=1.0, entropy_weight=1.0
        )
        output_no = loss_no_entropy(predictions, targets)
        output_with = loss_with_entropy(predictions, targets)
        # Entropy term is subtracted, so loss_with < loss_no
        assert output_with.total_loss.item() < output_no.total_loss.item()

    def test_squeezed_trailing_dim_labels(self):
        batch_size, horizon, num_phases = 2, 3, 4
        logits = torch.zeros(batch_size, horizon, num_phases)
        logits[:, :, 0] = 100.0
        labels = torch.zeros(batch_size, horizon, 1, dtype=torch.long)  # (B, T, 1)
        loss = PhaseClassificationLoss(
            key="phase_label",
            cross_entropy_weight=1.0,
            entropy_weight=0.0,
            label_smoothing=0.0,
        )
        output = loss({"phase_label": logits}, {"phase_label": labels})
        assert output.total_loss.item() < 0.01

    def test_metadata_includes_logits_and_labels(self):
        logits = torch.zeros(2, 3, 4)
        labels = torch.zeros(2, 3, dtype=torch.long)
        loss = PhaseClassificationLoss(key="phase_label")
        output = loss({"phase_label": logits}, {"phase_label": labels})
        assert MetadataKey.PHASE_LOGITS.value in output.metadata
        assert MetadataKey.PHASE_LABEL.value in output.metadata

    def test_raises_on_missing_key(self):
        loss = PhaseClassificationLoss(key="phase_label")
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Predictions and targets must contain key 'phase_label' for PhaseClassificationLoss."
            ),
        ):
            loss({"wrong": torch.zeros(1)}, {"wrong": torch.zeros(1)})


@pytest.mark.unit
class TestActionTokenLossGetRequiredKeys:
    def test_returns_action_logits_key(self):
        loss = ActionTokenLoss()
        assert loss.get_required_keys() == {DecoderOutputKey.ACTION_LOGITS.value}


@pytest.mark.unit
class TestActionTokenLossForward:
    def test_perfect_predictions_produce_zero_loss(self):
        vocab_size = 10
        batch_size, horizon = 2, 3
        target_tokens = torch.zeros(batch_size, horizon, dtype=torch.long)
        logits = torch.zeros(batch_size, horizon, vocab_size)
        logits[:, :, 0] = 100.0
        predictions = {DecoderOutputKey.ACTION_LOGITS.value: logits}
        targets = {SampleKey.TOKENIZED_ACTIONS.value: target_tokens}
        loss = ActionTokenLoss(label_smoothing=0.0)
        output = loss(predictions, targets)
        assert output.total_loss.item() < 0.01
        assert output.component_losses[
            MetricKey.TOKEN_ACCURACY.value
        ].item() == pytest.approx(1.0)

    def test_random_predictions_have_low_accuracy(self, rng):
        vocab_size = 100
        batch_size, horizon = 4, 10
        logits_data = rng.standard_normal((batch_size, horizon, vocab_size)).astype(
            np.float32
        )
        logits = torch.from_numpy(logits_data)
        target_tokens = torch.zeros(batch_size, horizon, dtype=torch.long)
        predictions = {DecoderOutputKey.ACTION_LOGITS.value: logits}
        targets = {SampleKey.TOKENIZED_ACTIONS.value: target_tokens}
        loss = ActionTokenLoss(label_smoothing=0.0)
        output = loss(predictions, targets)
        assert output.component_losses[MetricKey.TOKEN_ACCURACY.value].item() < 0.5

    def test_perplexity_is_exp_of_cross_entropy(self, rng):
        vocab_size = 10
        batch_size, horizon = 2, 4
        logits_data = rng.standard_normal((batch_size, horizon, vocab_size)).astype(
            np.float32
        )
        logits = torch.from_numpy(logits_data)
        target_tokens = torch.zeros(batch_size, horizon, dtype=torch.long)
        predictions = {DecoderOutputKey.ACTION_LOGITS.value: logits}
        targets = {SampleKey.TOKENIZED_ACTIONS.value: target_tokens}
        loss = ActionTokenLoss(label_smoothing=0.0)
        output = loss(predictions, targets)
        ce = output.component_losses[MetricKey.ACTION_TOKEN_CROSS_ENTROPY.value].item()
        perplexity = output.component_losses[MetricKey.PERPLEXITY.value].item()
        assert perplexity == pytest.approx(math.exp(ce), rel=1e-4)

    def test_raises_on_missing_logits(self):
        loss = ActionTokenLoss()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain keys '{DecoderOutputKey.ACTION_LOGITS.value}' for ActionTokenLoss."
            ),
        ):
            loss({}, {SampleKey.TOKENIZED_ACTIONS.value: torch.zeros(1)})


@pytest.mark.unit
class TestPriorDenoisingLossGetRequiredKeys:
    def test_returns_prediction_and_target_keys(self):
        loss = PriorDenoisingLoss()
        keys = loss.get_required_keys()
        assert LatentKey.PRIOR_PREDICTION.value in keys
        assert LatentKey.PRIOR_TARGET.value in keys


@pytest.mark.unit
class TestPriorDenoisingLossForward:
    def test_identical_prediction_and_target_produce_zero_loss(self):
        z = torch.ones(4, 8)
        predictions = {
            LatentKey.PRIOR_PREDICTION.value: z,
            LatentKey.PRIOR_TARGET.value: z.clone(),
        }
        loss = PriorDenoisingLoss(weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_prediction_and_target_produce_positive_loss(self, rng):
        pred = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        target = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        predictions = {
            LatentKey.PRIOR_PREDICTION.value: pred,
            LatentKey.PRIOR_TARGET.value: target,
        }
        loss = PriorDenoisingLoss(weight=1.0)
        output = loss(predictions, {})
        expected = F.mse_loss(pred, target)
        assert output.total_loss.item() == pytest.approx(expected.item())

    def test_logs_target_scale_normalized_prior_metrics(self):
        pred = torch.zeros(2, 2)
        target = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
        predictions = {
            LatentKey.PRIOR_PREDICTION.value: pred,
            LatentKey.PRIOR_TARGET.value: target,
        }
        loss = PriorDenoisingLoss(weight=1.0)
        output = loss(predictions, {})

        prior_mse = F.mse_loss(pred, target)
        target_var = target.var(unbiased=False)
        target_std = torch.sqrt(target_var + 1e-8)
        assert output.component_losses[
            MetricKey.PRIOR_DENOISING_TARGET_STD.value
        ].item() == pytest.approx(target_std.item())
        assert output.component_losses[
            MetricKey.PRIOR_DENOISING_NORMALIZED_MSE.value
        ].item() == pytest.approx((prior_mse / (target_var + 1e-8)).item())
        assert output.component_losses[
            MetricKey.PRIOR_DENOISING_NORMALIZED_RMSE.value
        ].item() == pytest.approx((torch.sqrt(prior_mse) / target_std).item())

    def test_normalized_prior_metrics_are_finite_for_constant_target(self):
        pred = torch.zeros(2, 2)
        target = torch.ones(2, 2)
        predictions = {
            LatentKey.PRIOR_PREDICTION.value: pred,
            LatentKey.PRIOR_TARGET.value: target,
        }
        loss = PriorDenoisingLoss(weight=1.0)
        output = loss(predictions, {})

        assert torch.isfinite(
            output.component_losses[MetricKey.PRIOR_DENOISING_NORMALIZED_MSE.value]
        )
        assert torch.isfinite(
            output.component_losses[MetricKey.PRIOR_DENOISING_NORMALIZED_RMSE.value]
        )

    def test_weight_scales_loss(self, rng):
        pred = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        target = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        predictions = {
            LatentKey.PRIOR_PREDICTION.value: pred,
            LatentKey.PRIOR_TARGET.value: target,
        }
        loss_w1 = PriorDenoisingLoss(weight=1.0)
        loss_w3 = PriorDenoisingLoss(weight=3.0)
        output_w1 = loss_w1(predictions, {})
        output_w3 = loss_w3(predictions, {})
        assert output_w3.total_loss.item() == pytest.approx(
            3.0 * output_w1.total_loss.item(), rel=1e-5
        )

    def test_raises_on_missing_prediction(self):
        loss = PriorDenoisingLoss()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain '{LatentKey.PRIOR_PREDICTION.value}' for PriorDenoisingLoss."
            ),
        ):
            loss({LatentKey.PRIOR_TARGET.value: torch.zeros(1)}, {})

    def test_raises_on_missing_target(self):
        loss = PriorDenoisingLoss()
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain '{LatentKey.PRIOR_TARGET.value}' for PriorDenoisingLoss."
            ),
        ):
            loss({LatentKey.PRIOR_PREDICTION.value: torch.zeros(1)}, {})

    def test_metadata_includes_latent_variables_when_available(self, rng):
        pred = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        target = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        z_posterior = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        mu_posterior = torch.from_numpy(rng.standard_normal((4, 8)).astype(np.float32))
        predictions = {
            LatentKey.PRIOR_PREDICTION.value: pred,
            LatentKey.PRIOR_TARGET.value: target,
            LatentKey.POSTERIOR_LATENT.value: z_posterior,
            LatentKey.POSTERIOR_MU.value: mu_posterior,
        }
        loss = PriorDenoisingLoss(weight=1.0)
        output = loss(predictions, {})
        assert MetadataKey.POSTERIOR_Z.value in output.metadata
        assert MetadataKey.POSTERIOR_MU.value in output.metadata


@pytest.mark.unit
class TestGaussianMixtureNLLossGetRequiredKeys:
    def test_returns_action_keys(self):
        loss = GaussianMixtureNLLoss(action_keys=["position", "orientation"])
        assert loss.get_required_keys() == {"position", "orientation"}


@pytest.mark.unit
class TestGaussianMixtureNLLossForward:
    def test_perfect_match_produces_low_nll(self):
        batch_size, horizon, num_experts, action_dim = 2, 3, 2, 2
        # Target matches expert 0's mean exactly
        target = torch.ones(batch_size, horizon, action_dim)
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[:, :, 0] = 1.0  # Expert 0 = perfect match
        logvars = -2.0 * torch.ones(
            batch_size, horizon, num_experts, action_dim
        )  # small variance
        routing_weights = torch.tensor([[0.99, 0.01]])  # heavily weight expert 0
        predictions = {
            "position_mean": means,
            "position_logvar": logvars,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": target}
        loss = GaussianMixtureNLLoss(action_keys=["position"], learned_variance=True)
        output = loss(predictions, targets)
        assert output.total_loss.item() < 5.0

    def test_fixed_variance_mode(self):
        batch_size, horizon, num_experts, action_dim = 2, 3, 2, 2
        target = torch.zeros(batch_size, horizon, action_dim)
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[:, :, 0] = 0.0  # Expert 0 matches target
        routing_weights = torch.tensor([[0.5, 0.5]])
        predictions = {
            "position": means,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": target}
        loss = GaussianMixtureNLLoss(
            action_keys=["position"],
            learned_variance=False,
            sigmas={"position": 1.0},
        )
        output = loss(predictions, targets)
        assert output.total_loss.isfinite()

    def test_fixed_variance_reads_mean_key_from_gaussian_head(self):
        batch_size, horizon, num_experts, action_dim = 2, 3, 2, 2
        target = torch.zeros(batch_size, horizon, action_dim)
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[:, :, 0] = 0.0
        routing_weights = torch.tensor([[0.5, 0.5]])
        predictions = {
            "position_mean": means,
            "position_logvar": torch.zeros_like(means),
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": target}
        loss = GaussianMixtureNLLoss(
            action_keys=["position"],
            learned_variance=False,
            sigmas={"position": 1.0},
        )
        output = loss(predictions, targets)
        assert output.total_loss.isfinite()

    def test_weight_scales_output(self, rng):
        batch_size, horizon, num_experts, action_dim = 2, 3, 2, 2
        target_data = rng.standard_normal((batch_size, horizon, action_dim)).astype(
            np.float32
        )
        target = torch.from_numpy(target_data)
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        routing_weights = torch.ones(1, num_experts) / num_experts
        predictions = {
            "position": means,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": target}
        loss_w1 = GaussianMixtureNLLoss(
            action_keys=["position"], weight=1.0, learned_variance=False
        )
        loss_w2 = GaussianMixtureNLLoss(
            action_keys=["position"], weight=2.0, learned_variance=False
        )
        output_w1 = loss_w1(predictions, targets)
        output_w2 = loss_w2(predictions, targets)
        assert output_w2.total_loss.item() == pytest.approx(
            2.0 * output_w1.total_loss.item(), rel=1e-4
        )

    def test_per_trajectory_routing_uses_joint_logsumexp(self):
        batch_size, horizon, num_experts, action_dim = 1, 3, 2, 2
        target = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[0, :, 0, :] = target[0]
        means[0, :, 1, :] = -target[0]
        logvars = torch.zeros(batch_size, horizon, num_experts, action_dim)
        routing_weights = torch.tensor([[0.5, 0.5]])
        predictions = {
            "position_mean": means,
            "position_logvar": logvars,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        loss = GaussianMixtureNLLoss(action_keys=["position"], learned_variance=True)
        actual = loss(predictions, {"position": target}).total_loss
        log_norm = -0.5 * action_dim * math.log(2 * math.pi)
        log_pi = math.log(0.5)
        sum_sq_match = float((target**2).sum().item())
        log_traj_match = horizon * log_norm - 0.5 * 0.0
        log_traj_other = horizon * log_norm - 0.5 * 4.0 * sum_sq_match
        expected = (
            -math.log(
                math.exp(log_pi + log_traj_match) + math.exp(log_pi + log_traj_other)
            )
            / horizon
        )
        assert actual.item() == pytest.approx(expected, rel=1e-4)

    def test_per_step_routing_uses_per_step_logsumexp(self):
        batch_size, horizon, num_experts, action_dim = 1, 3, 2, 2
        target = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[0, :, 0, :] = target[0]
        means[0, :, 1, :] = -target[0]
        logvars = torch.zeros(batch_size, horizon, num_experts, action_dim)
        routing_weights = torch.full((batch_size, horizon, num_experts), 0.5)
        predictions = {
            "position_mean": means,
            "position_logvar": logvars,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        loss = GaussianMixtureNLLoss(action_keys=["position"], learned_variance=True)
        actual = loss(predictions, {"position": target}).total_loss
        log_norm = -0.5 * action_dim * math.log(2 * math.pi)
        log_pi = math.log(0.5)
        per_step_nll = 0.0
        for step in range(horizon):
            sq_match = float((target[0, step] ** 2).sum().item())
            log_match = log_norm - 0.5 * 0.0
            log_other = log_norm - 0.5 * 4.0 * sq_match
            log_step_mix = math.log(
                math.exp(log_pi + log_match) + math.exp(log_pi + log_other)
            )
            per_step_nll -= log_step_mix
        expected = per_step_nll / horizon
        assert actual.item() == pytest.approx(expected, rel=1e-4)

    @pytest.mark.parametrize("per_step_routing", [False, True])
    def test_multi_key_routing_uses_joint_component_assignment(
        self, per_step_routing: bool
    ):
        batch_size, horizon, num_experts, action_dim = 1, 1, 2, 1
        target = torch.zeros(batch_size, horizon, action_dim)
        position_means = 10.0 * torch.ones(batch_size, horizon, num_experts, action_dim)
        orientation_means = 10.0 * torch.ones(
            batch_size, horizon, num_experts, action_dim
        )
        position_means[:, :, 0, :] = 0.0
        orientation_means[:, :, 1, :] = 0.0
        logvars = torch.zeros(batch_size, horizon, num_experts, action_dim)
        if per_step_routing:
            routing_weights = torch.tensor([[[0.5, 0.5]]])
            log_routing_weights = torch.log(routing_weights[:, 0, :])
        else:
            routing_weights = torch.tensor([[0.5, 0.5]])
            log_routing_weights = torch.log(routing_weights)
        predictions = {
            "position_mean": position_means,
            "position_logvar": logvars,
            "orientation_mean": orientation_means,
            "orientation_logvar": logvars,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        loss = GaussianMixtureNLLoss(
            action_keys=["position", "orientation"], learned_variance=True
        )
        actual = loss(
            predictions=predictions,
            targets={"position": target, "orientation": target},
        )
        log_normalization = -0.5 * action_dim * math.log(2 * math.pi)
        joint_log_component = torch.tensor(
            [[2 * log_normalization - 50.0, 2 * log_normalization - 50.0]]
        )
        expected = -torch.logsumexp(
            log_routing_weights + joint_log_component, dim=-1
        ).mean()
        assert actual.total_loss.item() == pytest.approx(expected.item(), rel=1e-4)
        assert actual.component_losses[
            MetricKey.GAUSSIAN_MIXTURE_NLL.value
        ].item() == pytest.approx(expected.item(), rel=1e-4)

    def test_padding_excludes_padded_timesteps_from_trajectory_sum(self):
        batch_size, horizon, num_experts, action_dim = 1, 4, 2, 2
        target = torch.zeros(batch_size, horizon, action_dim)
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        means[0, 3, :, :] = 100.0
        logvars = torch.zeros(batch_size, horizon, num_experts, action_dim)
        routing_weights = torch.tensor([[0.5, 0.5]])
        predictions = {
            "position_mean": means,
            "position_logvar": logvars,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        loss = GaussianMixtureNLLoss(action_keys=["position"], learned_variance=True)
        is_pad = torch.tensor([[False, False, False, True]])
        finite = loss(predictions, {"position": target}, is_pad=is_pad).total_loss
        no_pad_target = target[:, :3].clone()
        no_pad_means = means[:, :3].clone()
        no_pad_logvars = logvars[:, :3].clone()
        reference_predictions = {
            "position_mean": no_pad_means,
            "position_logvar": no_pad_logvars,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        reference = loss(reference_predictions, {"position": no_pad_target}).total_loss
        assert finite.item() == pytest.approx(reference.item(), rel=1e-5)

    def test_trajectory_loss_separates_opposite_modes(self):
        torch.manual_seed(0)
        batch_size, horizon, num_experts, action_dim = 256, 60, 2, 2
        timesteps = torch.linspace(0, 2 * math.pi, horizon)
        cos = torch.cos(timesteps)
        sin = torch.sin(timesteps)
        mode_a = torch.stack([cos, sin], dim=-1)
        mode_b = torch.stack([cos, -sin], dim=-1)
        targets = torch.stack(
            [mode_a if i < batch_size // 2 else mode_b for i in range(batch_size)]
        )
        targets = targets + 0.01 * torch.randn_like(targets)
        means = torch.nn.Parameter(torch.randn(num_experts, horizon, action_dim) * 0.3)
        logvars = torch.nn.Parameter(torch.zeros(num_experts, horizon, action_dim))
        pi_logits = torch.nn.Parameter(torch.zeros(num_experts))
        optimizer = torch.optim.Adam([means, logvars, pi_logits], lr=1e-2)
        loss = GaussianMixtureNLLoss(action_keys=["position"], learned_variance=True)
        for _ in range(800):
            stacked_means = (
                means.unsqueeze(0).expand(batch_size, -1, -1, -1).permute(0, 2, 1, 3)
            )
            stacked_logvars = (
                logvars.unsqueeze(0).expand(batch_size, -1, -1, -1).permute(0, 2, 1, 3)
            )
            routing_weights = (
                torch.softmax(pi_logits, dim=0).unsqueeze(0).expand(batch_size, -1)
            )
            output = loss(
                {
                    "position_mean": stacked_means,
                    "position_logvar": stacked_logvars,
                    DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
                },
                {"position": targets},
            )
            optimizer.zero_grad()
            output.total_loss.backward()
            optimizer.step()
        with torch.no_grad():
            mode_a_reference = mode_a
            mode_b_reference = mode_b
            distances_to_a = [
                (means[k] - mode_a_reference).pow(2).mean().sqrt().item()
                for k in range(num_experts)
            ]
            distances_to_b = [
                (means[k] - mode_b_reference).pow(2).mean().sqrt().item()
                for k in range(num_experts)
            ]
        assignment_one = distances_to_a[0] + distances_to_b[1]
        assignment_two = distances_to_a[1] + distances_to_b[0]
        best_assignment = min(assignment_one, assignment_two)
        assert best_assignment < 0.05


@pytest.mark.unit
class TestGripperMixtureNLLossInit:
    def test_binary_gripper_stores_type(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        loss = GripperMixtureNLLoss(key="gripper", actions_metadata=metadata)
        assert loss.gripper_type == GripperType.BINARY.value

    def test_raises_on_missing_key(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        with pytest.raises(
            ValueError,
            match=re.escape("wrong is not available to the action space"),
        ):
            GripperMixtureNLLoss(key="wrong", actions_metadata=metadata)


@pytest.mark.unit
class TestGripperMixtureNLLossForward:
    def test_binary_gripper_produces_finite_loss(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        loss = GripperMixtureNLLoss(
            key="gripper", actions_metadata=metadata, weight=1.0
        )
        batch_size, horizon, num_experts = 2, 3, 2
        expert_logits = torch.zeros(batch_size, horizon, num_experts)
        routing_weights = torch.tensor([[0.5, 0.5]])
        targets = {"gripper": torch.ones(batch_size, horizon)}
        predictions = {
            "gripper": expert_logits,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        output = loss(predictions, targets)
        assert output.total_loss.isfinite()
        assert MetricKey.GRIPPER_NLL.value in output.component_losses

    def test_continuous_gripper_fixed_variance(
        self, continuous_gripper_metadata_factory
    ):
        metadata = continuous_gripper_metadata_factory()
        loss = GripperMixtureNLLoss(
            key="gripper",
            actions_metadata=metadata,
            weight=1.0,
            learned_variance=False,
            sigma=0.5,
        )
        batch_size, horizon, num_experts = 2, 3, 2
        means = torch.zeros(batch_size, horizon, num_experts, 1)
        routing_weights = torch.tensor([[0.5, 0.5]])
        targets = {"gripper": torch.zeros(batch_size, horizon, 1)}
        predictions = {
            "gripper": means,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        output = loss(predictions, targets)
        assert output.total_loss.isfinite()

    def test_raises_on_missing_target_key(self, binary_gripper_metadata_factory):
        metadata = binary_gripper_metadata_factory()
        loss = GripperMixtureNLLoss(key="gripper", actions_metadata=metadata)
        with pytest.raises(
            ValueError,
            match=re.escape("Targets must contain 'gripper' for GripperMixtureNLLoss."),
        ):
            loss(
                {DecoderOutputKey.ROUTING_WEIGHTS.value: torch.zeros(1)},
                {"wrong": torch.zeros(1)},
            )


@pytest.mark.unit
class TestMoELossGetRequiredKeys:
    def test_union_of_base_and_routing_keys(self):
        base_loss = RegressionLoss(action_keys=["position"])
        moe_loss = MoELoss(base_loss=base_loss)
        keys = moe_loss.get_required_keys()
        assert "position" in keys
        assert DecoderOutputKey.ROUTING_WEIGHTS.value in keys


class TestMoELossGetCallbacks:
    def test_returns_expert_usage_callback(self):
        base_loss = RegressionLoss(action_keys=["position"])
        moe_loss = MoELoss(base_loss=base_loss)
        experiment_config = MagicMock(spec=ExperimentConfig)
        callbacks = moe_loss.get_callbacks(experiment_config=experiment_config)
        assert len(callbacks) == 1
        assert isinstance(callbacks[0], ExpertUsageCallback)
        assert callbacks[0].log_every_n_epochs == 1


@pytest.mark.unit
class TestMoELossForward:
    def test_passes_through_base_loss(self):
        predictions = {
            "position": torch.ones(2, 3, 2),
            DecoderOutputKey.ROUTING_WEIGHTS.value: torch.tensor([[0.5, 0.5]]),
        }
        targets = {"position": torch.zeros(2, 3, 2)}
        base_loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        moe_loss = MoELoss(base_loss=base_loss, entropy_weight=0.0)
        output = moe_loss(predictions, targets)
        # Base loss MSE = 1.0
        expected_mse = F.mse_loss(predictions["position"], targets["position"])
        assert output.total_loss.item() == pytest.approx(expected_mse.item())
        assert MetadataKey.EXPERT_USAGE.value in output.metadata

    def test_entropy_regularization_reduces_loss(self, rng):
        batch_size, horizon, action_dim, num_experts = 2, 3, 2, 3
        predictions = {
            "position": torch.ones(batch_size, horizon, action_dim),
            DecoderOutputKey.ROUTING_WEIGHTS.value: torch.softmax(
                torch.from_numpy(
                    rng.standard_normal((batch_size, num_experts)).astype(np.float32)
                ),
                dim=-1,
            ),
        }
        targets = {"position": torch.zeros(batch_size, horizon, action_dim)}
        base_loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        moe_no_entropy = MoELoss(base_loss=base_loss, entropy_weight=0.0)
        moe_with_entropy = MoELoss(base_loss=base_loss, entropy_weight=1.0)
        output_no = moe_no_entropy(predictions, targets)
        output_with = moe_with_entropy(predictions, targets)
        # Entropy is subtracted, so total should be lower
        assert output_with.total_loss.item() < output_no.total_loss.item()
        assert MetricKey.EXPERTS_ENTROPY.value in output_with.component_losses

    def test_adds_weighted_mean_predictions_for_gmm(self):
        batch_size, horizon, num_experts, action_dim = 2, 3, 2, 2
        means = torch.ones(batch_size, horizon, num_experts, action_dim)
        means[:, :, 0] = 0.0  # expert 0 = 0
        means[:, :, 1] = 2.0  # expert 1 = 2
        routing_weights = torch.tensor([[0.5, 0.5]])
        predictions = {
            f"position_{DecoderOutputKey.MEAN.value}": means,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": torch.ones(batch_size, horizon, action_dim)}
        base_loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        moe_loss = MoELoss(base_loss=base_loss, entropy_weight=0.0)
        output = moe_loss(predictions, targets)
        # Weighted mean = 0.5 * 0 + 0.5 * 2 = 1.0 = target => MSE ≈ 0
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_load_balance_minimum_at_uniform_per_trajectory_routing(self):
        batch_size, num_experts = 8, 4
        # Each example argmaxes to a different expert in round-robin → uniform usage
        routing_weights = torch.full((batch_size, num_experts), 1.0 / num_experts)
        argmax_targets = torch.arange(batch_size) % num_experts
        for b, k in enumerate(argmax_targets):
            routing_weights[b, k] += 1e-3  # break ties toward round-robin assignment
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        predictions = {
            "position": torch.zeros(batch_size, 3, 2),
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": torch.zeros(batch_size, 3, 2)}
        base_loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        moe_loss = MoELoss(
            base_loss=base_loss, entropy_weight=0.0, load_balance_weight=1.0
        )
        output = moe_loss(predictions, targets)
        load_balance = output.component_losses[
            MetricKey.EXPERTS_LOAD_BALANCE.value
        ].item()
        # f_k = 1/K, P_k ≈ 1/K, so K * sum(f_k * P_k) ≈ K * K * (1/K)² = 1.0
        assert load_balance == pytest.approx(1.0, rel=1e-3)

    def test_load_balance_penalises_collapsed_routing(self):
        batch_size, num_experts = 8, 4
        # All examples route exclusively to expert 0
        routing_weights = torch.zeros(batch_size, num_experts)
        routing_weights[:, 0] = 1.0
        predictions = {
            "position": torch.zeros(batch_size, 3, 2),
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": torch.zeros(batch_size, 3, 2)}
        base_loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        moe_loss = MoELoss(
            base_loss=base_loss, entropy_weight=0.0, load_balance_weight=1.0
        )
        output = moe_loss(predictions, targets)
        load_balance = output.component_losses[
            MetricKey.EXPERTS_LOAD_BALANCE.value
        ].item()
        # f_0 = 1, P_0 = 1, all others = 0 → K * sum = K * 1 = num_experts
        assert load_balance == pytest.approx(float(num_experts), rel=1e-5)

    def test_load_balance_handles_per_step_routing_with_padding(self):
        batch_size, horizon, num_experts = 2, 4, 4
        # Per-step routing collapsed at every valid timestep to expert 0;
        # padded timestep would (incorrectly) look balanced but must be ignored.
        routing_weights = torch.zeros(batch_size, horizon, num_experts)
        routing_weights[:, :3, 0] = 1.0
        routing_weights[:, 3, :] = 1.0 / num_experts
        is_pad = torch.tensor(
            [[False, False, False, True], [False, False, False, True]]
        )
        predictions = {
            "position": torch.zeros(batch_size, horizon, 2),
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": torch.zeros(batch_size, horizon, 2)}
        base_loss = RegressionLoss(action_keys=["position"], mse_weight=1.0)
        moe_loss = MoELoss(
            base_loss=base_loss, entropy_weight=0.0, load_balance_weight=1.0
        )
        output = moe_loss(predictions, targets, is_pad=is_pad)
        load_balance = output.component_losses[
            MetricKey.EXPERTS_LOAD_BALANCE.value
        ].item()
        # Only valid positions count → all route to expert 0 → load_balance == K
        assert load_balance == pytest.approx(float(num_experts), rel=1e-5)


@pytest.mark.unit
class TestMetadataPassthroughGetRequiredKeys:
    def test_returns_target_keys(self):
        loss = MetadataPassthrough(
            keys_mapping={"phase_label": "phase_label", "extra": "extra_meta"}
        )
        assert loss.get_required_keys() == {"phase_label", "extra"}


@pytest.mark.unit
class TestMetadataPassthroughForward:
    def test_extracts_targets_into_metadata(self):
        predictions = {"dummy": torch.tensor([1.0])}
        phase_labels = torch.tensor([[0, 1, 2]])
        targets = {"phase_label": phase_labels}
        loss = MetadataPassthrough(keys_mapping={"phase_label": "phase_meta"})
        output = loss(predictions, targets)
        assert torch.equal(output.metadata["phase_meta"], phase_labels)
        assert output.total_loss.item() == pytest.approx(0.0)

    def test_missing_target_key_is_silently_skipped(self):
        predictions = {"dummy": torch.tensor([1.0])}
        targets = {}
        loss = MetadataPassthrough(keys_mapping={"missing_key": "meta"})
        output = loss(predictions, targets)
        assert "meta" not in output.metadata
        assert output.total_loss.item() == pytest.approx(0.0)


@pytest.mark.unit
class TestVICLatentLossGetRequiredKeys:
    def test_returns_key(self):
        loss = VICLatentLoss(key=LatentKey.POSTERIOR_MU.value)
        assert loss.get_required_keys() == {LatentKey.POSTERIOR_MU.value}


@pytest.mark.unit
class TestVICLatentLossForward:
    def test_zero_variance_produces_positive_variance_loss(self):
        # All vectors identical => std = 0, which is below gamma
        latent = torch.ones(8, 4)
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = VICLatentLoss(
            key=LatentKey.POSTERIOR_MU.value,
            covariance_weight=0.0,
            variance_weight=1.0,
            gamma=0.3,
        )
        output = loss(predictions, {})
        assert output.total_loss.item() > 0.0
        assert MetricKey.VARIANCE_LOSS.value in output.component_losses

    def test_correlated_dimensions_produce_positive_covariance_loss(self):
        batch_size = 32
        # Two dimensions perfectly correlated: x2 = x1
        x1 = torch.linspace(-1, 1, batch_size).unsqueeze(1)
        latent = torch.cat([x1, x1], dim=1)
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = VICLatentLoss(
            key=LatentKey.POSTERIOR_MU.value,
            covariance_weight=1.0,
            variance_weight=0.0,
        )
        output = loss(predictions, {})
        assert output.total_loss.item() > 0.0
        assert MetricKey.COVARIANCE_LOSS.value in output.component_losses

    def test_independent_high_variance_produces_low_loss(self, rng):
        batch_size, latent_dim = 64, 4
        # Independent, high-variance data
        data = rng.standard_normal((batch_size, latent_dim)).astype(np.float32) * 2.0
        latent = torch.from_numpy(data)
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = VICLatentLoss(
            key=LatentKey.POSTERIOR_MU.value,
            covariance_weight=1.0,
            variance_weight=1.0,
            gamma=0.3,
        )
        output = loss(predictions, {})
        # With independent, high-variance data, both losses should be low
        assert output.component_losses[MetricKey.VARIANCE_LOSS.value].item() < 0.5

    def test_raises_on_missing_key(self):
        loss = VICLatentLoss(key=LatentKey.POSTERIOR_MU.value)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain '{LatentKey.POSTERIOR_MU.value}' for VICLatentLoss."
            ),
        ):
            loss({}, {})


@pytest.mark.unit
class TestPosteriorGeometryLossGetRequiredKeys:
    def test_returns_key(self):
        loss = PosteriorGeometryLoss(key=LatentKey.POSTERIOR_MU.value)
        assert loss.get_required_keys() == {LatentKey.POSTERIOR_MU.value}


@pytest.mark.unit
class TestPosteriorGeometryLossInitialization:
    @pytest.mark.parametrize(
        "target_std, max_std, eps, expectation",
        [
            (1.0, 2.0, 1e-6, does_not_raise()),
            (
                0.0,
                2.0,
                1e-6,
                pytest.raises(
                    ValueError,
                    match=re.escape("target_std must be positive, got 0.0."),
                ),
            ),
            (
                1.0,
                0.0,
                1e-6,
                pytest.raises(
                    ValueError,
                    match=re.escape("max_std must be positive, got 0.0."),
                ),
            ),
            (
                1.0,
                2.0,
                0.0,
                pytest.raises(
                    ValueError,
                    match=re.escape("eps must be positive, got 0.0."),
                ),
            ),
        ],
    )
    def test_validates_positive_scale_configuration(
        self,
        target_std: float,
        max_std: float,
        eps: float,
        expectation: AbstractContextManager,
    ):
        with expectation:
            PosteriorGeometryLoss(
                target_std=target_std,
                max_std=max_std,
                eps=eps,
            )


@pytest.mark.unit
class TestPosteriorGeometryLossForward:
    def test_centered_unit_std_independent_latents_have_low_loss(self):
        latent = torch.tensor(
            [
                [-1.0, -1.0],
                [-1.0, 1.0],
                [1.0, -1.0],
                [1.0, 1.0],
            ]
        )
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = PosteriorGeometryLoss(
            mean_weight=1.0,
            std_weight=1.0,
            target_std=1.0,
            max_std_weight=1.0,
            max_std=2.0,
            covariance_weight=1.0,
            eps=1e-6,
        )
        output = loss(predictions, {})
        assert output.total_loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_nonzero_mean_produces_mean_loss(self):
        latent = torch.ones(8, 2)
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = PosteriorGeometryLoss(
            mean_weight=1.0,
            std_weight=0.0,
            max_std_weight=0.0,
            covariance_weight=0.0,
        )
        output = loss(predictions, {})
        assert (
            output.component_losses[MetricKey.POSTERIOR_GEOMETRY_MEAN_LOSS.value].item()
            > 0.0
        )

    def test_large_std_produces_target_and_max_std_losses(self):
        latent = torch.tensor(
            [
                [-3.0, -3.0],
                [-3.0, 3.0],
                [3.0, -3.0],
                [3.0, 3.0],
            ]
        )
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = PosteriorGeometryLoss(
            mean_weight=0.0,
            std_weight=1.0,
            target_std=1.0,
            max_std_weight=1.0,
            max_std=2.0,
            covariance_weight=0.0,
        )
        output = loss(predictions, {})
        assert (
            output.component_losses[MetricKey.POSTERIOR_GEOMETRY_STD_LOSS.value].item()
            > 0.0
        )
        assert (
            output.component_losses[
                MetricKey.POSTERIOR_GEOMETRY_MAX_STD_LOSS.value
            ].item()
            > 0.0
        )

    def test_correlated_latents_produce_covariance_loss(self):
        x = torch.linspace(-1.0, 1.0, 8).unsqueeze(1)
        latent = torch.cat([x, x], dim=1)
        predictions = {LatentKey.POSTERIOR_MU.value: latent}
        loss = PosteriorGeometryLoss(
            mean_weight=0.0,
            std_weight=0.0,
            max_std_weight=0.0,
            covariance_weight=1.0,
        )
        output = loss(predictions, {})
        assert (
            output.component_losses[
                MetricKey.POSTERIOR_GEOMETRY_COVARIANCE_LOSS.value
            ].item()
            > 0.0
        )

    def test_raises_on_missing_key(self):
        loss = PosteriorGeometryLoss(key=LatentKey.POSTERIOR_MU.value)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain '{LatentKey.POSTERIOR_MU.value}' "
                "for PosteriorGeometryLoss."
            ),
        ):
            loss({}, {})

    def test_raises_on_non_matrix_latents(self):
        loss = PosteriorGeometryLoss(key=LatentKey.POSTERIOR_MU.value)
        predictions = {
            LatentKey.POSTERIOR_MU.value: torch.zeros(2, 3, 4),
        }
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"PosteriorGeometryLoss expects '{LatentKey.POSTERIOR_MU.value}' "
                "with shape (batch_size, latent_dimension), got (2, 3, 4)."
            ),
        ):
            loss(predictions, {})


class TestVQCommitmentLoss:
    @pytest.fixture
    def vq_predictions_factory(
        self, rng: np.random.Generator
    ) -> Callable[..., dict[str, torch.Tensor]]:

        def factory(
            batch_size: int = 8,
            code_dim: int = 16,
            num_codes: int = 4,
            num_layers: int = 1,
        ) -> dict[str, torch.Tensor]:
            z_continuous = torch.from_numpy(
                rng.standard_normal((num_layers, batch_size, code_dim)).astype(
                    np.float32
                )
            )
            z_quantized = torch.from_numpy(
                rng.standard_normal((num_layers, batch_size, code_dim)).astype(
                    np.float32
                )
            )
            all_indices = [
                torch.from_numpy(
                    rng.integers(0, num_codes, size=(batch_size,)).astype(np.int64)
                )
                for _ in range(num_layers)
            ]
            return {
                LatentKey.VQ_Z_CONTINUOUS.value: z_continuous,
                LatentKey.VQ_QUANTIZED.value: z_quantized,
                LatentKey.VQ_INDICES.value: all_indices,
            }

        return factory

    @pytest.mark.unit
    @pytest.mark.parametrize("weight", [0.5, 1.0, 10.0])
    def test_stores_weight(self, weight: float) -> None:
        loss = VQCommitmentLoss(num_codes=4, num_residual_layers=1, weight=weight)
        assert loss.weight == weight

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "num_codes, num_residual_layers",
        [(4, 1), (16, 2), (8, 4)],
    )
    def test_stores_codebook_dimensions(
        self, num_codes: int, num_residual_layers: int
    ) -> None:
        loss = VQCommitmentLoss(
            num_codes=num_codes,
            num_residual_layers=num_residual_layers,
        )
        assert loss.num_codes == num_codes
        assert loss.num_residual_layers == num_residual_layers

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "num_codes, num_residual_layers, expectation",
        [
            (4, 1, does_not_raise()),
            (
                0,
                1,
                pytest.raises(
                    ValueError,
                    match=re.escape("num_codes must be positive, got 0."),
                ),
            ),
            (
                -1,
                1,
                pytest.raises(
                    ValueError,
                    match=re.escape("num_codes must be positive, got -1."),
                ),
            ),
            (
                4,
                0,
                pytest.raises(
                    ValueError,
                    match=re.escape("num_residual_layers must be positive, got 0."),
                ),
            ),
            (
                4,
                -2,
                pytest.raises(
                    ValueError,
                    match=re.escape("num_residual_layers must be positive, got -2."),
                ),
            ),
        ],
    )
    def test_rejects_invalid_codebook_dimensions(
        self,
        num_codes: int,
        num_residual_layers: int,
        expectation: AbstractContextManager,
    ) -> None:
        with expectation:
            VQCommitmentLoss(
                num_codes=num_codes,
                num_residual_layers=num_residual_layers,
            )

    @pytest.mark.unit
    def test_raises_on_missing_keys(self) -> None:
        loss = VQCommitmentLoss(num_codes=4, num_residual_layers=1, weight=1.0)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain {loss.get_required_keys()} for VQCommitmentLoss."
            ),
        ):
            loss.forward(predictions={}, targets={})

    @pytest.mark.unit
    @pytest.mark.parametrize("code_dim", [4, 16, 64])
    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_returns_nonnegative_scalar(
        self,
        vq_predictions_factory: Callable[..., dict[str, torch.Tensor]],
        code_dim: int,
        batch_size: int,
        num_layers: int,
    ) -> None:
        loss = VQCommitmentLoss(num_codes=4, num_residual_layers=num_layers, weight=1.0)
        predictions = vq_predictions_factory(
            batch_size=batch_size, code_dim=code_dim, num_layers=num_layers
        )
        result = loss.forward(predictions=predictions, targets={})
        assert result.total_loss.dim() == 0
        assert result.total_loss.item() >= 0.0

    @pytest.mark.unit
    @pytest.mark.parametrize("code_dim", [4, 16])
    def test_weight_scales_total_loss(
        self,
        vq_predictions_factory: Callable[..., dict[str, torch.Tensor]],
        code_dim: int,
    ) -> None:
        predictions = vq_predictions_factory(batch_size=8, code_dim=code_dim)
        result_w1 = VQCommitmentLoss(
            num_codes=4, num_residual_layers=1, weight=1.0
        ).forward(predictions=predictions, targets={})
        result_w5 = VQCommitmentLoss(
            num_codes=4, num_residual_layers=1, weight=5.0
        ).forward(predictions=predictions, targets={})
        assert torch.isclose(
            result_w5.total_loss, result_w1.total_loss * 5.0, rtol=1e-5
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("num_layers", [1, 3])
    def test_zero_loss_when_continuous_equals_quantized(
        self, rng: np.random.Generator, num_layers: int
    ) -> None:
        z = torch.from_numpy(
            rng.standard_normal((num_layers, 8, 16)).astype(np.float32)
        )
        predictions = {
            LatentKey.VQ_Z_CONTINUOUS.value: z,
            LatentKey.VQ_QUANTIZED.value: z.clone(),
        }
        result = VQCommitmentLoss(
            num_codes=4, num_residual_layers=num_layers, weight=1.0
        ).forward(predictions=predictions, targets={})
        assert torch.isclose(result.total_loss, torch.tensor(0.0), atol=1e-6)

    @pytest.mark.unit
    def test_averages_commitment_across_layers(self, rng: np.random.Generator) -> None:
        num_layers = 3
        batch_size = 8
        code_dim = 4
        z_continuous = torch.from_numpy(
            rng.standard_normal((num_layers, batch_size, code_dim)).astype(np.float32)
        )
        z_quantized = torch.from_numpy(
            rng.standard_normal((num_layers, batch_size, code_dim)).astype(np.float32)
        )
        predictions = {
            LatentKey.VQ_Z_CONTINUOUS.value: z_continuous,
            LatentKey.VQ_QUANTIZED.value: z_quantized,
        }
        total = (
            VQCommitmentLoss(num_codes=4, num_residual_layers=num_layers, weight=1.0)
            .forward(predictions=predictions, targets={})
            .total_loss
        )

        per_layer_mse = torch.stack(
            [
                F.mse_loss(z_continuous[layer_index], z_quantized[layer_index])
                for layer_index in range(num_layers)
            ]
        )
        assert torch.isclose(total, per_layer_mse.mean(), atol=1e-6)

    @pytest.mark.unit
    def test_component_losses_contains_commitment_key(
        self,
        vq_predictions_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        result = VQCommitmentLoss(
            num_codes=4, num_residual_layers=1, weight=1.0
        ).forward(
            predictions=vq_predictions_factory(batch_size=8, code_dim=16), targets={}
        )
        assert MetricKey.VQ_COMMITMENT_LOSS.value in result.component_losses

    @pytest.mark.unit
    def test_codebook_usage_in_metadata(
        self,
        vq_predictions_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        result = VQCommitmentLoss(
            num_codes=4, num_residual_layers=1, weight=1.0
        ).forward(
            predictions=vq_predictions_factory(batch_size=8, code_dim=16, num_codes=4),
            targets={},
        )
        assert MetricKey.VQ_CODEBOOK_USAGE.value in result.metadata

    @pytest.mark.unit
    def test_codebook_usage_uses_k_times_l_denominator(self) -> None:
        num_codes = 8
        num_layers = 2
        batch_size = 16
        z = torch.zeros((num_layers, batch_size, 4))
        # Layer 0: all 8 distinct codes used (0..7 twice). Layer 1: only 4
        # distinct codes used. Total distinct = 8 + 4 = 12; capacity = K*L = 16.
        layer_0 = torch.arange(batch_size) % num_codes
        layer_1 = torch.arange(batch_size) % 4
        predictions = {
            LatentKey.VQ_Z_CONTINUOUS.value: z,
            LatentKey.VQ_QUANTIZED.value: z.clone(),
            LatentKey.VQ_INDICES.value: [layer_0, layer_1],
        }
        result = VQCommitmentLoss(
            num_codes=num_codes, num_residual_layers=num_layers, weight=1.0
        ).forward(predictions=predictions, targets={})
        expected_usage = (num_codes + 4) / (num_codes * num_layers)
        assert result.metadata[MetricKey.VQ_CODEBOOK_USAGE.value] == pytest.approx(
            expected_usage
        )

    @pytest.mark.unit
    def test_codebook_usage_absent_when_indices_not_provided(
        self, rng: np.random.Generator
    ) -> None:
        z = torch.from_numpy(rng.standard_normal((1, 8, 16)).astype(np.float32))
        predictions = {
            LatentKey.VQ_Z_CONTINUOUS.value: z,
            LatentKey.VQ_QUANTIZED.value: z.clone(),
        }
        result = VQCommitmentLoss(
            num_codes=4, num_residual_layers=1, weight=1.0
        ).forward(predictions=predictions, targets={})
        assert MetricKey.VQ_CODEBOOK_USAGE.value not in result.metadata


class TestVQPriorCrossEntropyLoss:
    @pytest.fixture
    def prior_ce_predictions_factory(
        self, rng: np.random.Generator
    ) -> Callable[..., dict[str, torch.Tensor]]:

        def factory(
            batch_size: int = 8,
            num_codes: int = 4,
            num_layers: int = 1,
        ) -> dict[str, torch.Tensor]:
            all_logits = [
                torch.from_numpy(
                    rng.standard_normal((batch_size, num_codes)).astype(np.float32)
                )
                for _ in range(num_layers)
            ]
            all_indices = [
                torch.from_numpy(
                    rng.integers(0, num_codes, size=(batch_size,)).astype(np.int64)
                )
                for _ in range(num_layers)
            ]
            return {
                LatentKey.PRIOR_CODE_LOGITS.value: all_logits,
                LatentKey.VQ_INDICES.value: all_indices,
            }

        return factory

    @pytest.mark.unit
    @pytest.mark.parametrize("weight", [0.5, 1.0, 10.0])
    def test_stores_weight(self, weight: float) -> None:
        loss = VQPriorCrossEntropyLoss(weight=weight)
        assert loss.weight == weight

    @pytest.mark.unit
    def test_raises_on_missing_keys(self) -> None:
        loss = VQPriorCrossEntropyLoss(weight=1.0)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Predictions must contain {loss.get_required_keys()} for VQPriorCrossEntropyLoss."
            ),
        ):
            loss.forward(predictions={}, targets={})

    @pytest.mark.unit
    def test_raises_when_prior_logits_are_empty(self) -> None:
        predictions = {
            LatentKey.PRIOR_CODE_LOGITS.value: [],
            LatentKey.VQ_INDICES.value: [],
        }
        with pytest.raises(
            ValueError,
            match=re.escape("VQPriorCrossEntropyLoss received no prior logits."),
        ):
            VQPriorCrossEntropyLoss(weight=1.0).forward(
                predictions=predictions, targets={}
            )

    @pytest.mark.unit
    def test_raises_on_layer_count_mismatch(self) -> None:
        predictions = {
            LatentKey.PRIOR_CODE_LOGITS.value: [torch.zeros(4, 8)],
            LatentKey.VQ_INDICES.value: [
                torch.zeros(4, dtype=torch.long),
                torch.zeros(4, dtype=torch.long),
            ],
        }
        with pytest.raises(
            ValueError,
            match=re.escape(
                "VQPriorCrossEntropyLoss expected the same number of prior logit "
                "layers and posterior index layers, got 1 and 2."
            ),
        ):
            VQPriorCrossEntropyLoss(weight=1.0).forward(
                predictions=predictions, targets={}
            )

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "prior_logits, posterior_indices, expected_message",
        [
            (
                torch.zeros(4, 2, 8),
                torch.zeros(4, dtype=torch.long),
                "Prior logits for VQ layer 0 must have shape (B, K), got (4, 2, 8).",
            ),
            (
                torch.zeros(4, 8),
                torch.zeros(4, 1, dtype=torch.long),
                "Posterior indices for VQ layer 0 must have shape (B,), got (4, 1).",
            ),
            (
                torch.zeros(4, 8),
                torch.zeros(3, dtype=torch.long),
                "Prior logits and posterior indices for VQ layer 0 must have the "
                "same batch size, got 4 and 3.",
            ),
        ],
    )
    def test_raises_on_invalid_layer_shapes(
        self,
        prior_logits: torch.Tensor,
        posterior_indices: torch.Tensor,
        expected_message: str,
    ) -> None:
        predictions = {
            LatentKey.PRIOR_CODE_LOGITS.value: [prior_logits],
            LatentKey.VQ_INDICES.value: [posterior_indices],
        }
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            VQPriorCrossEntropyLoss(weight=1.0).forward(
                predictions=predictions, targets={}
            )

    @pytest.mark.unit
    @pytest.mark.parametrize("num_codes", [2, 4, 16])
    @pytest.mark.parametrize("num_layers", [1, 2])
    @pytest.mark.parametrize("batch_size", [1, 8])
    def test_returns_nonnegative_scalar(
        self,
        prior_ce_predictions_factory: Callable[..., dict[str, torch.Tensor]],
        num_codes: int,
        num_layers: int,
        batch_size: int,
    ) -> None:
        loss = VQPriorCrossEntropyLoss(weight=1.0)
        predictions = prior_ce_predictions_factory(
            batch_size=batch_size, num_codes=num_codes, num_layers=num_layers
        )
        result = loss.forward(predictions=predictions, targets={})
        assert result.total_loss.dim() == 0
        assert result.total_loss.item() >= 0.0

    @pytest.mark.unit
    def test_weight_scales_total_loss(
        self,
        prior_ce_predictions_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        predictions = prior_ce_predictions_factory(batch_size=8, num_codes=4)
        result_w1 = VQPriorCrossEntropyLoss(weight=1.0).forward(
            predictions=predictions, targets={}
        )
        result_w5 = VQPriorCrossEntropyLoss(weight=5.0).forward(
            predictions=predictions, targets={}
        )
        assert torch.isclose(
            result_w5.total_loss, result_w1.total_loss * 5.0, rtol=1e-5
        )

    @pytest.mark.unit
    def test_perfect_prior_gives_low_loss(self, rng: np.random.Generator) -> None:
        batch_size = 8
        num_codes = 4
        indices = torch.from_numpy(
            rng.integers(0, num_codes, size=(batch_size,)).astype(np.int64)
        )  # (B,)
        logits = torch.full((batch_size, num_codes), -10.0)  # (B, K)
        for i in range(batch_size):
            logits[i, indices[i]] = 10.0
        predictions = {
            LatentKey.PRIOR_CODE_LOGITS.value: [logits],
            LatentKey.VQ_INDICES.value: [indices],
        }
        result = VQPriorCrossEntropyLoss(weight=1.0).forward(
            predictions=predictions, targets={}
        )
        assert result.total_loss.item() < 0.01

    @pytest.mark.unit
    def test_uniform_prior_gives_log_k_loss(self) -> None:
        batch_size = 64
        num_codes = 4
        logits = torch.zeros(batch_size, num_codes)  # uniform logits
        indices = torch.zeros(batch_size, dtype=torch.long)
        predictions = {
            LatentKey.PRIOR_CODE_LOGITS.value: [logits],
            LatentKey.VQ_INDICES.value: [indices],
        }
        result = VQPriorCrossEntropyLoss(weight=1.0).forward(
            predictions=predictions, targets={}
        )
        expected_ce = torch.log(torch.tensor(float(num_codes)))
        assert torch.isclose(result.total_loss, expected_ce, atol=0.01)

    @pytest.mark.unit
    def test_component_losses_contains_ce_key(
        self,
        prior_ce_predictions_factory: Callable[..., dict[str, torch.Tensor]],
    ) -> None:
        result = VQPriorCrossEntropyLoss(weight=1.0).forward(
            predictions=prior_ce_predictions_factory(batch_size=8, num_codes=4),
            targets={},
        )
        assert MetricKey.VQ_PRIOR_CROSS_ENTROPY.value in result.component_losses


@pytest.fixture
def leaf_weight_spec_factory(
    binary_gripper_metadata_factory: Callable[..., dict],
) -> Callable[..., dict[str, Any]]:
    """Factory fixture: one `(loss, initial, set_to, partial, expected_after_partial)` per name."""

    def factory(name: str) -> dict[str, Any]:
        match name:
            case "regression":
                return {
                    "loss": RegressionLoss(
                        action_keys=["a"],
                        mse_weight=1.0,
                        l1_weight=0.5,
                        huber_weight=0.25,
                    ),
                    "initial_weights": {
                        "mse_weight": 1.0,
                        "l1_weight": 0.5,
                        "huber_weight": 0.25,
                    },
                    "set_to": {
                        "mse_weight": 0.2,
                        "l1_weight": 0.3,
                        "huber_weight": 0.4,
                    },
                    "partial_update": {"l1_weight": 0.1},
                    "expected_after_partial": {
                        "mse_weight": 1.0,
                        "l1_weight": 0.1,
                        "huber_weight": 0.25,
                    },
                }
            case "gripper":
                return {
                    "loss": GripperLoss(
                        key="gripper",
                        actions_metadata=binary_gripper_metadata_factory(),
                        bce_weight=0.01,
                        mse_weight=0.0,
                    ),
                    "initial_weights": {"bce_weight": 0.01, "mse_weight": 0.0},
                    "set_to": {"bce_weight": 0.2, "mse_weight": 0.3},
                    "partial_update": {"mse_weight": 0.8},
                    "expected_after_partial": {"bce_weight": 0.01, "mse_weight": 0.8},
                }
            case "kl_divergence":
                return {
                    "loss": KLDivergenceLoss(
                        weight=10.0,
                        prior_entropy_weight=0.1,
                        prior_regularization_weight=0.2,
                    ),
                    "initial_weights": {
                        "weight": 10.0,
                        "prior_entropy_weight": 0.1,
                        "prior_regularization_weight": 0.2,
                    },
                    "set_to": {
                        "weight": 1.0,
                        "prior_entropy_weight": 0.0,
                        "prior_regularization_weight": 0.9,
                    },
                    "partial_update": {"prior_regularization_weight": 0.9},
                    "expected_after_partial": {
                        "weight": 10.0,
                        "prior_entropy_weight": 0.1,
                        "prior_regularization_weight": 0.9,
                    },
                }
            case "binary_kl_divergence":
                return {
                    "loss": BinaryKLDivergenceLoss(weight=5.0, entropy_weight=0.001),
                    "initial_weights": {"weight": 5.0, "entropy_weight": 0.001},
                    "set_to": {"weight": 1.0, "entropy_weight": 0.0},
                    "partial_update": {"entropy_weight": 0.5},
                    "expected_after_partial": {"weight": 5.0, "entropy_weight": 0.5},
                }
            case "gaussian_entropy":
                return {
                    "loss": GaussianEntropyLoss(weight=0.02, bound_weight=0.5),
                    "initial_weights": {"weight": 0.02, "bound_weight": 0.5},
                    "set_to": {"weight": 1.0, "bound_weight": 2.0},
                    "partial_update": {"bound_weight": 0.0},
                    "expected_after_partial": {"weight": 0.02, "bound_weight": 0.0},
                }
            case "maximum_mean_discrepancy":
                return {
                    "loss": MaximumMeanDiscrepancyLoss(
                        weight=1.0, prior_regularization_weight=0.2
                    ),
                    "initial_weights": {
                        "weight": 1.0,
                        "prior_regularization_weight": 0.2,
                    },
                    "set_to": {"weight": 3.0, "prior_regularization_weight": 0.05},
                    "partial_update": {"prior_regularization_weight": 0.9},
                    "expected_after_partial": {
                        "weight": 1.0,
                        "prior_regularization_weight": 0.9,
                    },
                }
            case "phase_classification":
                return {
                    "loss": PhaseClassificationLoss(
                        key="phase",
                        cross_entropy_weight=0.1,
                        entropy_weight=0.05,
                    ),
                    "initial_weights": {
                        "cross_entropy_weight": 0.1,
                        "entropy_weight": 0.05,
                    },
                    "set_to": {"cross_entropy_weight": 0.3, "entropy_weight": 0.0},
                    "partial_update": {"entropy_weight": 0.2},
                    "expected_after_partial": {
                        "cross_entropy_weight": 0.1,
                        "entropy_weight": 0.2,
                    },
                }
            case "vic_latent":
                return {
                    "loss": VICLatentLoss(covariance_weight=3.0, variance_weight=10.0),
                    "initial_weights": {
                        "covariance_weight": 3.0,
                        "variance_weight": 10.0,
                    },
                    "set_to": {"covariance_weight": 1.0, "variance_weight": 2.0},
                    "partial_update": {"variance_weight": 5.0},
                    "expected_after_partial": {
                        "covariance_weight": 3.0,
                        "variance_weight": 5.0,
                    },
                }
            case "posterior_geometry":
                return {
                    "loss": PosteriorGeometryLoss(
                        mean_weight=0.1,
                        std_weight=0.2,
                        max_std_weight=0.3,
                        covariance_weight=0.4,
                    ),
                    "initial_weights": {
                        "mean_weight": 0.1,
                        "std_weight": 0.2,
                        "max_std_weight": 0.3,
                        "covariance_weight": 0.4,
                    },
                    "set_to": {
                        "mean_weight": 0.5,
                        "std_weight": 0.6,
                        "max_std_weight": 0.7,
                        "covariance_weight": 0.8,
                    },
                    "partial_update": {"max_std_weight": 0.9},
                    "expected_after_partial": {
                        "mean_weight": 0.1,
                        "std_weight": 0.2,
                        "max_std_weight": 0.9,
                        "covariance_weight": 0.4,
                    },
                }
            case "prior_denoising":
                return {
                    "loss": PriorDenoisingLoss(weight=0.03),
                    "initial_weights": {"weight": 0.03},
                    "set_to": {"weight": 0.5},
                    "partial_update": {"weight": 0.9},
                    "expected_after_partial": {"weight": 0.9},
                }
        raise ValueError(f"Unknown leaf_weight_spec_factory name: {name}")

    return factory


@pytest.mark.unit
class TestLossWeightsAPI:
    @pytest.mark.parametrize(
        "leaf_name",
        [
            "regression",
            "gripper",
            "kl_divergence",
            "binary_kl_divergence",
            "gaussian_entropy",
            "maximum_mean_discrepancy",
            "phase_classification",
            "vic_latent",
            "posterior_geometry",
            "prior_denoising",
        ],
    )
    def test_weights_returns_initial_values(
        self,
        leaf_weight_spec_factory: Callable[..., dict[str, Any]],
        leaf_name: str,
    ) -> None:
        spec = leaf_weight_spec_factory(leaf_name)
        assert spec["loss"].weights == spec["initial_weights"]

    @pytest.mark.parametrize(
        "leaf_name",
        [
            "regression",
            "gripper",
            "kl_divergence",
            "binary_kl_divergence",
            "gaussian_entropy",
            "maximum_mean_discrepancy",
            "phase_classification",
            "vic_latent",
            "posterior_geometry",
            "prior_denoising",
        ],
    )
    def test_set_weights_replaces_full_tree(
        self,
        leaf_weight_spec_factory: Callable[..., dict[str, Any]],
        leaf_name: str,
    ) -> None:
        spec = leaf_weight_spec_factory(leaf_name)
        loss = spec["loss"]
        loss.set_weights(spec["set_to"])
        assert loss.weights == spec["set_to"]

    @pytest.mark.parametrize(
        "leaf_name",
        [
            "regression",
            "gripper",
            "kl_divergence",
            "binary_kl_divergence",
            "gaussian_entropy",
            "maximum_mean_discrepancy",
            "phase_classification",
            "vic_latent",
            "posterior_geometry",
            "prior_denoising",
        ],
    )
    def test_update_weights_applies_partial_override(
        self,
        leaf_weight_spec_factory: Callable[..., dict[str, Any]],
        leaf_name: str,
    ) -> None:
        spec = leaf_weight_spec_factory(leaf_name)
        loss = spec["loss"]
        loss.update_weights(spec["partial_update"])
        assert loss.weights == spec["expected_after_partial"]

    def test_action_token_loss_weight_scales_forward_output(self) -> None:
        batch_size, horizon, vocab_size = 2, 4, 5
        torch.manual_seed(0)
        logits = torch.randn(batch_size, horizon, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size, horizon))
        predictions = {DecoderOutputKey.ACTION_LOGITS.value: logits}
        target_dict = {SampleKey.TOKENIZED_ACTIONS.value: targets}

        output_unit = ActionTokenLoss(weight=1.0, label_smoothing=0.0)(
            predictions, target_dict
        )
        output_half = ActionTokenLoss(weight=0.5, label_smoothing=0.0)(
            predictions, target_dict
        )
        assert output_half.total_loss.item() == pytest.approx(
            output_unit.total_loss.item() * 0.5, rel=1e-5
        )

    def test_moe_loss_weights_includes_base_loss_subtree(self) -> None:
        base = RegressionLoss(
            action_keys=["a"], mse_weight=1.0, l1_weight=0.5, huber_weight=0.25
        )
        loss = MoELoss(base_loss=base, entropy_weight=0.01, load_balance_weight=0.2)
        assert loss.weights == {
            "entropy_weight": 0.01,
            "load_balance_weight": 0.2,
            "base_loss": {
                "mse_weight": 1.0,
                "l1_weight": 0.5,
                "huber_weight": 0.25,
            },
        }

    def test_moe_loss_set_weights_delegates_to_base_loss(self) -> None:
        base = RegressionLoss(
            action_keys=["a"], mse_weight=1.0, l1_weight=0.5, huber_weight=0.25
        )
        loss = MoELoss(base_loss=base, entropy_weight=0.01, load_balance_weight=0.2)
        loss.set_weights(
            {
                "entropy_weight": 0.05,
                "load_balance_weight": 0.3,
                "base_loss": {
                    "mse_weight": 0.0,
                    "l1_weight": 1.0,
                    "huber_weight": 0.0,
                },
            }
        )
        assert loss.entropy_weight == pytest.approx(0.05)
        assert loss.load_balance_weight == pytest.approx(0.3)
        assert base.mse_weight == pytest.approx(0.0)
        assert base.l1_weight == pytest.approx(1.0)

    def test_moe_loss_update_weights_targets_nested_leaf(self) -> None:
        base = RegressionLoss(
            action_keys=["a"], mse_weight=1.0, l1_weight=0.5, huber_weight=0.25
        )
        loss = MoELoss(base_loss=base, entropy_weight=0.01, load_balance_weight=0.2)
        loss.update_weights({"base_loss": {"l1_weight": 0.9}})
        assert base.l1_weight == pytest.approx(0.9)
        assert base.mse_weight == pytest.approx(1.0)
        assert loss.entropy_weight == pytest.approx(0.01)


@pytest.fixture(
    params=[
        "prior_denoising",
        "trajectory_length",
        "action_token",
        "vq_commitment",
        "vq_prior_ce",
        "regression",
        "gripper",
        "phase_classification",
        "kl_divergence",
        "binary_kl_divergence",
        "gaussian_entropy",
        "maximum_mean_discrepancy",
        "vic_latent",
        "posterior_geometry",
    ]
)
def leaf_loss_case(
    request: pytest.FixtureRequest,
    binary_gripper_metadata_factory: Callable[..., dict],
) -> tuple[BaseLoss, set[str]]:
    """Factory fixture: one ``(leaf_loss, expected_weight_keys)`` pair per param id."""
    match request.param:
        case "prior_denoising":
            return PriorDenoisingLoss(weight=0.5), {"weight"}
        case "trajectory_length":
            return TrajectoryLengthLoss(action_key="action"), {"weight"}
        case "action_token":
            return ActionTokenLoss(), {"weight"}
        case "vq_commitment":
            return (
                VQCommitmentLoss(num_codes=4, num_residual_layers=1),
                {"weight"},
            )
        case "vq_prior_ce":
            return VQPriorCrossEntropyLoss(), {"weight"}
        case "regression":
            return (
                RegressionLoss(action_keys=["action"]),
                {"mse_weight", "l1_weight", "huber_weight"},
            )
        case "gripper":
            return (
                GripperLoss(
                    key="gripper",
                    actions_metadata=binary_gripper_metadata_factory(),
                ),
                {"bce_weight", "mse_weight"},
            )
        case "phase_classification":
            return (
                PhaseClassificationLoss(key="phase"),
                {"cross_entropy_weight", "entropy_weight"},
            )
        case "kl_divergence":
            return (
                KLDivergenceLoss(),
                {"weight", "prior_entropy_weight", "prior_regularization_weight"},
            )
        case "binary_kl_divergence":
            return BinaryKLDivergenceLoss(), {"weight", "entropy_weight"}
        case "gaussian_entropy":
            return GaussianEntropyLoss(), {"weight", "bound_weight"}
        case "maximum_mean_discrepancy":
            return (
                MaximumMeanDiscrepancyLoss(),
                {"weight", "prior_regularization_weight"},
            )
        case "vic_latent":
            return VICLatentLoss(), {"covariance_weight", "variance_weight"}
        case "posterior_geometry":
            return (
                PosteriorGeometryLoss(),
                {
                    "mean_weight",
                    "std_weight",
                    "max_std_weight",
                    "covariance_weight",
                },
            )
    raise ValueError(f"Unknown leaf_loss_case param: {request.param}")


@pytest.mark.unit
class TestSetWeightsValidation:
    def test_leaf_set_weights_rejects_missing_key(
        self, leaf_loss_case: tuple[BaseLoss, set[str]]
    ) -> None:
        loss, expected_keys = leaf_loss_case
        missing_key = sorted(expected_keys)[0]
        partial = {key: 0.1 for key in expected_keys if key != missing_key}
        with pytest.raises(
            KeyError,
            match=(
                f"{type(loss).__name__}.set_weights: "
                rf"missing=\['{missing_key}'\]"
            ),
        ):
            loss.set_weights(partial)

    def test_leaf_set_weights_rejects_extra_key(
        self, leaf_loss_case: tuple[BaseLoss, set[str]]
    ) -> None:
        loss, expected_keys = leaf_loss_case
        extra = dict.fromkeys(expected_keys, 0.1)
        extra["bogus_weight"] = 0.1
        with pytest.raises(
            KeyError,
            match=(
                f"{type(loss).__name__}.set_weights: "
                r"missing=\[\], extra=\['bogus_weight'\]"
            ),
        ):
            loss.set_weights(extra)

    def test_composite_set_weights_rejects_missing_child(self) -> None:
        composite = CompositeLoss(
            loss_modules={
                "a": PriorDenoisingLoss(weight=0.1),
                "b": PriorDenoisingLoss(weight=0.2),
            }
        )
        with pytest.raises(
            KeyError,
            match=r"CompositeLoss\.set_weights: missing=\['b'\]",
        ):
            composite.set_weights({"a": {"weight": 0.5}})

    def test_composite_set_weights_rejects_extra_child(self) -> None:
        composite = CompositeLoss(loss_modules={"a": PriorDenoisingLoss(weight=0.1)})
        with pytest.raises(
            KeyError,
            match=r"CompositeLoss\.set_weights: missing=\[\], extra=\['bogus'\]",
        ):
            composite.set_weights({"a": {"weight": 0.5}, "bogus": {"weight": 0.5}})

    def test_moe_set_weights_rejects_missing_base_loss(
        self,
        binary_gripper_metadata_factory: Callable[..., dict],
    ) -> None:
        inner = PriorDenoisingLoss(weight=0.1)
        moe = MoELoss(base_loss=inner)
        with pytest.raises(
            KeyError,
            match=r"MoELoss\.set_weights: missing=\['base_loss'\]",
        ):
            moe.set_weights({"entropy_weight": 0.0, "load_balance_weight": 0.0})

    def test_moe_set_weights_rejects_extra_key(self) -> None:
        inner = PriorDenoisingLoss(weight=0.1)
        moe = MoELoss(base_loss=inner)
        with pytest.raises(
            KeyError,
            match=r"MoELoss\.set_weights: missing=\[\], extra=\['bogus'\]",
        ):
            moe.set_weights(
                {
                    "entropy_weight": 0.0,
                    "load_balance_weight": 0.0,
                    "base_loss": {"weight": 0.3},
                    "bogus": 1.0,
                }
            )

    def test_update_weights_rejects_unknown_nested_key(self) -> None:
        composite = CompositeLoss(
            loss_modules={
                "regression": RegressionLoss(action_keys=["action"]),
            }
        )
        with pytest.raises(KeyError, match="Unknown weight key 'bogus'"):
            composite.update_weights({"regression": {"bogus": 0.1}})

    def test_update_weights_rejects_dict_for_scalar_leaf(self) -> None:
        composite = CompositeLoss(
            loss_modules={"denoising": PriorDenoisingLoss(weight=0.5)}
        )
        with pytest.raises(
            TypeError,
            match="Weight override for 'weight' expects a scalar",
        ):
            composite.update_weights({"denoising": {"weight": {"nested": 0.1}}})

    def test_update_weights_rejects_scalar_for_dict_subtree(self) -> None:
        composite = CompositeLoss(
            loss_modules={
                "regression": RegressionLoss(action_keys=["action"]),
            }
        )
        with pytest.raises(
            TypeError,
            match="Weight override for 'regression' expects a dict subtree",
        ):
            composite.update_weights({"regression": 0.5})
