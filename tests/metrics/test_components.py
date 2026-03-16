"""Tests for versatil.metrics.components module."""
import math
import re

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from versatil.data.constants import BinaryGripperRange, GripperType, SampleKey
from versatil.data.metadata import (
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
)
from versatil.metrics.base import LossOutput
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
    PriorDenoisingLoss,
    RegressionLoss,
    TrajectoryLengthLoss,
    TrajectorySmoothness,
    VICLatentLoss,
)
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.metrics.kernels import KernelType
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey


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

    @pytest.mark.parametrize("mse_weight, l1_weight, huber_weight", [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.3, 0.2),
    ])
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
        expected = F.huber_loss(predictions["position"], targets["position"], delta=delta)
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
class TestGripperLossGetRequiredKeys:
    def test_returns_gripper_key(self, binary_gripper_metadata_factory):
        loss = GripperLoss(key="gripper", actions_metadata=binary_gripper_metadata_factory())
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
        mu_post = torch.from_numpy(rng.standard_normal((batch_size, latent_dim)).astype(np.float32))
        logvar_post = torch.from_numpy(rng.standard_normal((batch_size, latent_dim)).astype(np.float32))
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
        assert MetricKey.HYPERPRIOR_KL_REGULARIZATION.value in output_with_reg.component_losses

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
        prior_log_prob = torch.distributions.Normal(
            torch.zeros(latent_dim), torch.ones(latent_dim)
        ).log_prob(z).sum(dim=-1)
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
        assert output.component_losses[MetricKey.RAW_KL_DIVERGENCE.value].item() == pytest.approx(
            0.0, abs=1e-5
        )

    def test_extreme_logits_produce_positive_kl(self):
        logits = 10.0 * torch.ones(4, 3, 8)  # sigmoid(10) ≈ 1
        loss = BinaryKLDivergenceLoss(weight=1.0, entropy_weight=0.0, free_bits=0.0)
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        output = loss(predictions, {})
        assert output.component_losses[MetricKey.RAW_KL_DIVERGENCE.value].item() > 0.1

    def test_free_bits_clamps_kl(self):
        logits = torch.zeros(4, 3, 8)  # KL ≈ 0
        free_bits = 1.0
        loss = BinaryKLDivergenceLoss(weight=1.0, entropy_weight=0.0, free_bits=free_bits)
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        output = loss(predictions, {})
        assert output.component_losses[MetricKey.CLAMPED_KL_DIVERGENCE.value].item() == pytest.approx(
            0.0, abs=1e-5
        )

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

        raw_kl = output_no_free.component_losses[MetricKey.RAW_KL_DIVERGENCE.value].item()
        clamped_kl = output_with_free.component_losses[MetricKey.CLAMPED_KL_DIVERGENCE.value].item()
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
        z_posterior = torch.from_numpy(rng.standard_normal((32, 8)).astype(np.float32)) + 5.0
        z_prior = torch.from_numpy(rng.standard_normal((32, 8)).astype(np.float32))
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: z_posterior,
            LatentKey.PRIOR_LATENT.value: z_prior,
        }
        loss = MaximumMeanDiscrepancyLoss(weight=1.0)
        output = loss(predictions, {})
        assert output.total_loss.item() > 0.01

    def test_prior_regularization_penalizes_non_standard_prior(self, rng):
        z_posterior = torch.from_numpy(rng.standard_normal((32, 8)).astype(np.float32))
        z_prior = torch.from_numpy(rng.standard_normal((32, 8)).astype(np.float32)) + 5.0
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: z_posterior,
            LatentKey.PRIOR_LATENT.value: z_prior,
        }
        loss_no_reg = MaximumMeanDiscrepancyLoss(weight=1.0, prior_regularization_weight=0.0)
        loss_with_reg = MaximumMeanDiscrepancyLoss(weight=1.0, prior_regularization_weight=1.0)
        output_no_reg = loss_no_reg(predictions, {})
        output_with_reg = loss_with_reg(predictions, {})
        assert output_with_reg.total_loss.item() >= output_no_reg.total_loss.item()
        assert MetricKey.HYPERPRIOR_MMD_REGULARIZATION.value in output_with_reg.component_losses

    def test_raises_when_prior_missing_and_not_fixed(self):
        loss = MaximumMeanDiscrepancyLoss(use_fixed_gaussian_as_prior=False)
        predictions = {LatentKey.POSTERIOR_LATENT.value: torch.zeros(4, 8)}
        with pytest.raises(
            ValueError,
            match="for MaximumMeanDiscrepancyLoss",
        ):
            loss(predictions, {})

    @pytest.mark.parametrize("kernel_type", [KernelType.RBF.value, KernelType.IMQ.value])
    def test_accepts_different_kernel_types(self, rng, kernel_type):
        z = torch.from_numpy(rng.standard_normal((16, 4)).astype(np.float32))
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: z,
            LatentKey.PRIOR_LATENT.value: z.clone(),
        }
        loss = MaximumMeanDiscrepancyLoss(kernel_type=kernel_type)
        output = loss(predictions, {})
        assert output.total_loss.item() >= 0.0


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
        logits = torch.from_numpy(
            rng.standard_normal((8, 4, 16)).astype(np.float32)
        )
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        loss = BinaryMaximumMeanDiscrepancyLoss(weight=1.0)
        output = loss(predictions, {})
        assert MetadataKey.POSTERIOR_Z.value in output.metadata
        # Posterior z should have same shape as logits
        assert output.metadata[MetadataKey.POSTERIOR_Z.value].shape == logits.shape

    @pytest.mark.parametrize("weight", [1.0, 3.0, 0.5])
    def test_weight_scales_total_loss_relative_to_component(self, rng, weight):
        logits = torch.from_numpy(
            rng.standard_normal((8, 4, 16)).astype(np.float32)
        )
        predictions = {DecoderOutputKey.BINARY_LOGITS.value: logits}
        loss = BinaryMaximumMeanDiscrepancyLoss(weight=weight)
        output = loss(predictions, {})
        # total_loss = weight * mmd_component
        mmd_component = output.component_losses[MetricKey.BINARY_MMD_LOSS.value].item()
        assert output.total_loss.item() == pytest.approx(
            weight * mmd_component, rel=1e-4
        )


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
        logits_data = rng.standard_normal((batch_size, horizon, num_phases)).astype(np.float32)
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
        logits_data = rng.standard_normal((batch_size, horizon, num_phases)).astype(np.float32)
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
            key="phase_label", cross_entropy_weight=1.0, entropy_weight=0.0, label_smoothing=0.0,
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
        assert output.component_losses[MetricKey.TOKEN_ACCURACY.value].item() == pytest.approx(1.0)

    def test_random_predictions_have_low_accuracy(self, rng):
        vocab_size = 100
        batch_size, horizon = 4, 10
        logits_data = rng.standard_normal((batch_size, horizon, vocab_size)).astype(np.float32)
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
        logits_data = rng.standard_normal((batch_size, horizon, vocab_size)).astype(np.float32)
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
        logvars = -2.0 * torch.ones(batch_size, horizon, num_experts, action_dim)  # small variance
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

    def test_weight_scales_output(self, rng):
        batch_size, horizon, num_experts, action_dim = 2, 3, 2, 2
        target_data = rng.standard_normal((batch_size, horizon, action_dim)).astype(np.float32)
        target = torch.from_numpy(target_data)
        means = torch.zeros(batch_size, horizon, num_experts, action_dim)
        routing_weights = torch.ones(1, num_experts) / num_experts
        predictions = {
            "position": means,
            DecoderOutputKey.ROUTING_WEIGHTS.value: routing_weights,
        }
        targets = {"position": target}
        loss_w1 = GaussianMixtureNLLoss(action_keys=["position"], weight=1.0, learned_variance=False)
        loss_w2 = GaussianMixtureNLLoss(action_keys=["position"], weight=2.0, learned_variance=False)
        output_w1 = loss_w1(predictions, targets)
        output_w2 = loss_w2(predictions, targets)
        assert output_w2.total_loss.item() == pytest.approx(
            2.0 * output_w1.total_loss.item(), rel=1e-4
        )


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
        loss = GripperMixtureNLLoss(key="gripper", actions_metadata=metadata, weight=1.0)
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

    def test_continuous_gripper_fixed_variance(self, continuous_gripper_metadata_factory):
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
                torch.from_numpy(rng.standard_normal((batch_size, num_experts)).astype(np.float32)),
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


@pytest.mark.unit
class TestMetadataPassthroughGetRequiredKeys:
    def test_returns_target_keys(self):
        loss = MetadataPassthrough(keys_mapping={"phase_label": "phase_label", "extra": "extra_meta"})
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
