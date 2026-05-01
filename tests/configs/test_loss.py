"""Tests for versatil.configs.loss module."""

import pytest
from hydra.utils import instantiate
from omegaconf import MISSING

from versatil.configs.loss import (
    ActionTokenLossConfig,
    BaseLossConfig,
    BinaryKLDivergenceLossConfig,
    BinaryMaximumMeanDiscrepancyLossConfig,
    CompositeLossConfig,
    ConditionalMaximumMeanDiscrepancyLossConfig,
    GaussianEntropyLossConfig,
    GaussianMixtureNLLossConfig,
    GripperLossConfig,
    GripperMixtureNLLossConfig,
    KLDivergenceLossConfig,
    LatentOptimalTransportLossConfig,
    MaximumMeanDiscrepancyLossConfig,
    MoELossConfig,
    OptimalTransportLossConfig,
    PhaseClassificationLossConfig,
    PosteriorGeometryLossConfig,
    PriorDenoisingLossConfig,
    RegressionLossConfig,
    RelaxedConditionalLatentOptimalTransportLossConfig,
    TrajectoryLengthLossConfig,
    TrajectorySmoothnessConfig,
    VICLatentLossConfig,
)
from versatil.metrics.kernels import KernelType


@pytest.mark.unit
def test_base_loss_config_target_defaults_to_missing():
    config = BaseLossConfig()
    assert config._target_ == MISSING


@pytest.mark.unit
class TestRegressionLossConfig:
    def test_target_points_to_regression_loss(self):
        config = RegressionLossConfig(action_keys=["position"])
        assert config._target_ == "versatil.metrics.components.RegressionLoss"

    @pytest.mark.parametrize("mse_weight", [1.0, 0.5])
    @pytest.mark.parametrize("l1_weight", [0.0, 0.3])
    @pytest.mark.parametrize("huber_weight", [0.0, 0.2])
    def test_stores_loss_weights(self, mse_weight, l1_weight, huber_weight):
        config = RegressionLossConfig(
            action_keys=["position"],
            mse_weight=mse_weight,
            l1_weight=l1_weight,
            huber_weight=huber_weight,
        )
        assert config.mse_weight == mse_weight
        assert config.l1_weight == l1_weight
        assert config.huber_weight == huber_weight

    def test_action_keys_required(self):
        config = RegressionLossConfig()
        assert config.action_keys == MISSING

    @pytest.mark.parametrize("per_key_weights", [None, {"position": 1.0}])
    def test_stores_per_key_weights(self, per_key_weights):
        config = RegressionLossConfig(
            action_keys=["position"],
            per_key_weights=per_key_weights,
        )
        assert config.per_key_weights == per_key_weights

    def test_inherits_from_base_loss_config(self):
        config = RegressionLossConfig(action_keys=["position"])
        assert isinstance(config, BaseLossConfig)


@pytest.mark.unit
class TestGripperLossConfig:
    def test_target_points_to_gripper_loss(self):
        config = GripperLossConfig(key="gripper")
        assert config._target_ == "versatil.metrics.components.GripperLoss"

    @pytest.mark.parametrize("bce_weight", [1.0, 0.5])
    @pytest.mark.parametrize("mse_weight", [1.0, 0.0])
    def test_stores_loss_weights(self, bce_weight, mse_weight):
        config = GripperLossConfig(
            key="gripper", bce_weight=bce_weight, mse_weight=mse_weight
        )
        assert config.bce_weight == bce_weight
        assert config.mse_weight == mse_weight


@pytest.mark.unit
class TestKLDivergenceLossConfig:
    def test_target_points_to_kl_loss(self):
        config = KLDivergenceLossConfig()
        assert config._target_ == "versatil.metrics.components.KLDivergenceLoss"

    @pytest.mark.parametrize("weight", [0.0001, 0.01])
    @pytest.mark.parametrize("prior_regularization_weight", [0.0, 0.1])
    def test_stores_configuration(self, weight, prior_regularization_weight):
        config = KLDivergenceLossConfig(
            weight=weight,
            prior_regularization_weight=prior_regularization_weight,
        )
        assert config.weight == weight
        assert config.prior_regularization_weight == prior_regularization_weight


@pytest.mark.unit
class TestMaximumMeanDiscrepancyLossConfig:
    def test_target_points_to_mmd_loss(self):
        config = MaximumMeanDiscrepancyLossConfig()
        assert (
            config._target_ == "versatil.metrics.components.MaximumMeanDiscrepancyLoss"
        )

    @pytest.mark.parametrize("use_median_heuristic", [True, False])
    @pytest.mark.parametrize(
        "kernel_type", [KernelType.RBF.value, KernelType.IMQ.value]
    )
    def test_stores_configuration(
        self,
        kernel_type: str,
        use_median_heuristic: bool,
    ):
        bandwidth_multipliers = [1.0, 2.0]
        prior_target_key = "mu"
        config = MaximumMeanDiscrepancyLossConfig(
            kernel_type=kernel_type,
            bandwidth_multipliers=bandwidth_multipliers,
            use_median_heuristic=use_median_heuristic,
            prior_target_key=prior_target_key,
        )
        assert config.kernel_type == kernel_type
        assert config.bandwidth_multipliers == bandwidth_multipliers
        assert config.use_median_heuristic is use_median_heuristic
        assert config.prior_target_key == prior_target_key


@pytest.mark.unit
class TestConditionalMaximumMeanDiscrepancyLossConfig:
    def test_target_points_to_conditional_mmd_loss(self):
        config = ConditionalMaximumMeanDiscrepancyLossConfig()
        assert (
            config._target_
            == "versatil.metrics.components.ConditionalMaximumMeanDiscrepancyLoss"
        )

    @pytest.mark.parametrize("normalize_condition", [True, False])
    @pytest.mark.parametrize(
        "kernel_type", [KernelType.RBF.value, KernelType.IMQ.value]
    )
    def test_stores_configuration(
        self,
        kernel_type: str,
        normalize_condition: bool,
    ):
        bandwidth_multipliers = [1.0, 2.0]
        config = ConditionalMaximumMeanDiscrepancyLossConfig(
            state_weight=2.0,
            kernel_type=kernel_type,
            bandwidth_multipliers=bandwidth_multipliers,
            condition_kernel_type=KernelType.RBF.value,
            condition_bandwidth_multipliers=[0.5, 1.0],
            condition_use_median_heuristic=False,
            prior_target_key="mu",
            condition_key="prior_condition",
            normalize_condition=normalize_condition,
        )
        assert config.state_weight == 2.0
        assert config.kernel_type == kernel_type
        assert config.bandwidth_multipliers == bandwidth_multipliers
        assert config.condition_kernel_type == KernelType.RBF.value
        assert config.condition_bandwidth_multipliers == [0.5, 1.0]
        assert config.condition_use_median_heuristic is False
        assert config.prior_target_key == "mu"
        assert config.condition_key == "prior_condition"
        assert config.normalize_condition is normalize_condition


@pytest.mark.unit
class TestCompositeLossConfig:
    def test_target_points_to_composite_loss(self):
        config = CompositeLossConfig()
        assert config._target_ == "versatil.metrics.composite.CompositeLoss"

    def test_loss_modules_default_to_empty_dict(self):
        config = CompositeLossConfig()
        assert config.loss_modules == {}

    def test_weights_default_to_none(self):
        config = CompositeLossConfig()
        assert config.weights is None


@pytest.mark.unit
class TestPhaseClassificationLossConfig:
    def test_target_points_to_phase_classification_loss(self):
        config = PhaseClassificationLossConfig(key="phase")
        assert config._target_ == "versatil.metrics.components.PhaseClassificationLoss"

    @pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
    def test_stores_label_smoothing(self, label_smoothing):
        config = PhaseClassificationLossConfig(
            key="phase", label_smoothing=label_smoothing
        )
        assert config.label_smoothing == label_smoothing


@pytest.mark.unit
class TestTrajectoryLengthLossConfig:
    def test_target_points_to_trajectory_length_loss(self):
        config = TrajectoryLengthLossConfig(action_key="position")
        assert config._target_ == "versatil.metrics.components.TrajectoryLengthLoss"

    def test_action_key_required(self):
        config = TrajectoryLengthLossConfig()
        assert config.action_key == MISSING


@pytest.mark.unit
class TestTrajectorySmoothnessConfig:
    def test_target_points_to_trajectory_smoothness(self):
        config = TrajectorySmoothnessConfig(action_key="position")
        assert config._target_ == "versatil.metrics.components.TrajectorySmoothness"


@pytest.mark.unit
class TestBinaryKLDivergenceLossConfig:
    def test_target_points_to_binary_kl_loss(self):
        config = BinaryKLDivergenceLossConfig(latent_bits=16)
        assert config._target_ == "versatil.metrics.components.BinaryKLDivergenceLoss"

    @pytest.mark.parametrize("free_bits", [0.0, 0.1])
    def test_stores_free_bits(self, free_bits):
        config = BinaryKLDivergenceLossConfig(latent_bits=16, free_bits=free_bits)
        assert config.free_bits == free_bits


@pytest.mark.unit
class TestGaussianEntropyLossConfig:
    def test_target_points_to_entropy_loss(self):
        config = GaussianEntropyLossConfig(key="latent")
        assert config._target_ == "versatil.metrics.components.GaussianEntropyLoss"


@pytest.mark.unit
class TestMoELossConfig:
    def test_target_points_to_moe_loss(self):
        config = MoELossConfig()
        assert config._target_ == "versatil.metrics.components.MoELoss"

    def test_base_loss_required(self):
        config = MoELossConfig()
        assert config.base_loss == MISSING


@pytest.mark.unit
class TestGaussianMixtureNLLossConfig:
    def test_target_points_to_gmm_nll_loss(self):
        config = GaussianMixtureNLLossConfig(action_keys=["position"])
        assert config._target_ == "versatil.metrics.components.GaussianMixtureNLLoss"

    @pytest.mark.parametrize("learned_variance", [True, False])
    def test_stores_learned_variance(self, learned_variance):
        config = GaussianMixtureNLLossConfig(
            action_keys=["position"], learned_variance=learned_variance
        )
        assert config.learned_variance == learned_variance


@pytest.mark.unit
class TestGripperMixtureNLLossConfig:
    def test_target_points_to_gripper_mixture_nll(self):
        config = GripperMixtureNLLossConfig(key="gripper")
        assert config._target_ == "versatil.metrics.components.GripperMixtureNLLoss"


@pytest.mark.unit
class TestVICLatentLossConfig:
    def test_target_points_to_vic_latent_loss(self):
        config = VICLatentLossConfig()
        assert config._target_ == "versatil.metrics.components.VICLatentLoss"

    @pytest.mark.parametrize("covariance_weight", [3.0, 1.0])
    @pytest.mark.parametrize("variance_weight", [10.0, 5.0])
    def test_stores_weights(self, covariance_weight, variance_weight):
        config = VICLatentLossConfig(
            covariance_weight=covariance_weight,
            variance_weight=variance_weight,
        )
        assert config.covariance_weight == covariance_weight
        assert config.variance_weight == variance_weight


@pytest.mark.unit
class TestPosteriorGeometryLossConfig:
    def test_target_points_to_posterior_geometry_loss(self):
        config = PosteriorGeometryLossConfig()
        assert config._target_ == "versatil.metrics.components.PosteriorGeometryLoss"

    @pytest.mark.parametrize("mean_weight", [0.0, 0.1])
    @pytest.mark.parametrize("std_weight", [0.0, 0.2])
    @pytest.mark.parametrize("max_std_weight", [0.0, 0.3])
    @pytest.mark.parametrize("covariance_weight", [0.0, 0.4])
    def test_stores_weights(
        self,
        mean_weight,
        std_weight,
        max_std_weight,
        covariance_weight,
    ):
        config = PosteriorGeometryLossConfig(
            mean_weight=mean_weight,
            std_weight=std_weight,
            max_std_weight=max_std_weight,
            covariance_weight=covariance_weight,
        )
        assert config.mean_weight == mean_weight
        assert config.std_weight == std_weight
        assert config.max_std_weight == max_std_weight
        assert config.covariance_weight == covariance_weight


@pytest.mark.unit
class TestOptimalTransportLossConfig:
    def test_target_points_to_ot_loss(self):
        config = OptimalTransportLossConfig(action_keys=["position"])
        assert config._target_ == "versatil.metrics.ot_loss.OptimalTransportLoss"

    @pytest.mark.parametrize("p", [1, 2])
    def test_stores_distance_norm(self, p):
        config = OptimalTransportLossConfig(action_keys=["position"], p=p)
        assert config.p == p


@pytest.mark.unit
class TestLatentOptimalTransportLossConfig:
    def test_target_points_to_latent_ot_loss(self):
        config = LatentOptimalTransportLossConfig()
        assert config._target_ == "versatil.metrics.ot_loss.LatentOptimalTransportLoss"

    def test_stores_prior_target_key(self):
        prior_target_key = "mu"
        config = LatentOptimalTransportLossConfig(prior_target_key=prior_target_key)
        assert config.prior_target_key == prior_target_key


@pytest.mark.unit
class TestRelaxedConditionalLatentOptimalTransportLossConfig:
    def test_target_points_to_relaxed_conditional_latent_ot_loss(self):
        config = RelaxedConditionalLatentOptimalTransportLossConfig()
        assert (
            config._target_
            == "versatil.metrics.ot_loss.RelaxedConditionalLatentOptimalTransportLoss"
        )

    @pytest.mark.parametrize("normalize_condition", [True, False])
    def test_stores_configuration(self, normalize_condition):
        config = RelaxedConditionalLatentOptimalTransportLossConfig(
            prior_target_key="mu",
            condition_key="prior_condition",
            state_weight=2.0,
            normalize_condition=normalize_condition,
        )
        assert config.prior_target_key == "mu"
        assert config.condition_key == "prior_condition"
        assert config.state_weight == 2.0
        assert config.normalize_condition is normalize_condition


@pytest.mark.unit
class TestPriorDenoisingLossConfig:
    def test_target_points_to_prior_denoising_loss(self):
        config = PriorDenoisingLossConfig()
        assert config._target_ == "versatil.metrics.components.PriorDenoisingLoss"


@pytest.mark.unit
class TestBinaryMaximumMeanDiscrepancyLossConfig:
    def test_target_points_to_binary_mmd_loss(self):
        config = BinaryMaximumMeanDiscrepancyLossConfig()
        assert (
            config._target_
            == "versatil.metrics.components.BinaryMaximumMeanDiscrepancyLoss"
        )


@pytest.mark.unit
class TestActionTokenLossConfig:
    def test_target_points_to_action_token_loss(self):
        config = ActionTokenLossConfig()
        assert config._target_ == "versatil.metrics.components.ActionTokenLoss"

    @pytest.mark.parametrize("label_smoothing", [0.0, 0.2, 0.5])
    def test_stores_label_smoothing(self, label_smoothing):
        config = ActionTokenLossConfig(label_smoothing=label_smoothing)
        assert config.label_smoothing == label_smoothing

    @pytest.mark.parametrize("weight", [0.5, 1.0, 2.0])
    def test_stores_weight(self, weight):
        config = ActionTokenLossConfig(weight=weight)
        assert config.weight == weight

    def test_weight_defaults_to_one(self):
        config = ActionTokenLossConfig()
        assert config.weight == 1.0

    def test_inherits_from_base_loss_config(self):
        config = ActionTokenLossConfig()
        assert isinstance(config, BaseLossConfig)


@pytest.mark.unit
class TestLossInstantiation:
    def test_regression_loss_instantiates(self):
        config = RegressionLossConfig(action_keys=["position_action"])
        instance = instantiate(config)
        assert instance.mse_weight == 1.0
        assert instance.action_keys == ["position_action"]

    def test_kl_divergence_loss_instantiates(self):
        config = KLDivergenceLossConfig(weight=0.001)
        instance = instantiate(config)
        assert instance.weight == 0.001

    def test_mmd_loss_instantiates(self):
        config = MaximumMeanDiscrepancyLossConfig(weight=2.0)
        instance = instantiate(config)
        assert instance.weight == 2.0

    def test_composite_loss_instantiates_with_nested_sub_losses(self):
        config = CompositeLossConfig(
            loss_modules={
                "regression": RegressionLossConfig(action_keys=["position_action"]),
                "kl": KLDivergenceLossConfig(weight=0.001),
            },
            weights={"regression": 1.0, "kl": 0.001},
        )
        instance = instantiate(config)
        assert "regression" in instance.loss_modules
        assert "kl" in instance.loss_modules

    def test_trajectory_length_loss_instantiates(self):
        config = TrajectoryLengthLossConfig(weight=0.1, action_key="position_action")
        instance = instantiate(config)
        assert instance.weight == 0.1

    def test_phase_classification_loss_instantiates(self):
        config = PhaseClassificationLossConfig(
            key="phase_label", cross_entropy_weight=1.0
        )
        instance = instantiate(config)
        assert instance.key == "phase_label"

    def test_action_token_loss_instantiates(self):
        config = ActionTokenLossConfig(label_smoothing=0.1)
        instance = instantiate(config)
        assert instance.label_smoothing == 0.1
