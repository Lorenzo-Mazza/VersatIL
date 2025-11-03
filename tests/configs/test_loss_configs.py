import inspect

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from refactoring.configs.loss import (
    ActionReconstructionLossConfig,
    CompositeLossConfig,
    GripperLossConfig,
    KLDivergenceLossConfig,
    PhaseActionLossConfig,
    PhaseClassificationLossConfig,
    RegressionLossConfig,
    TrajectoryLengthLossConfig,
    TrajectorySmoothnessConfig,
)
from refactoring.data.constants import GripperType
from refactoring.metrics import (
    ActionReconstructionLoss,
    CompositeLoss,
    GripperLoss,
    KLDivergenceLoss,
    PhaseActionLoss,
    PhaseClassificationLoss,
    RegressionLoss,
    TrajectoryLengthLoss,
    TrajectorySmoothness,
)


@pytest.mark.unit
class TestRegressionLossConfig:
    def test_config_has_correct_target(self):
        config = RegressionLossConfig()
        assert config._target_ == "refactoring.metrics.RegressionLoss"

    def test_config_instantiates_correctly(self):
        config = RegressionLossConfig()
        loss = instantiate(config)
        assert isinstance(loss, RegressionLoss)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(RegressionLoss.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        config = RegressionLossConfig()
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {"_target_"}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"

    def test_default_weights(self):
        config = RegressionLossConfig()
        assert config.mse_weight == 1.0
        assert config.l1_weight == 0.0
        assert config.huber_weight == 0.0


@pytest.mark.unit
class TestGripperLossConfig:
    def test_config_has_correct_target(self):
        config = GripperLossConfig()
        assert config._target_ == "refactoring.metrics.GripperLoss"

    def test_config_instantiates_correctly(self):
        config = GripperLossConfig()
        loss = instantiate(config)
        assert isinstance(loss, GripperLoss)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(GripperLoss.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        config = GripperLossConfig()
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {"_target_"}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"

    def test_default_gripper_type(self):
        config = GripperLossConfig()
        assert config.gripper_type == GripperType.BINARY.value


@pytest.mark.unit
class TestKLDivergenceLossConfig:
    def test_config_has_correct_target(self):
        config = KLDivergenceLossConfig()
        assert config._target_ == "refactoring.metrics.KLDivergenceLoss"

    def test_config_instantiates_correctly(self):
        config = KLDivergenceLossConfig()
        loss = instantiate(config)
        assert isinstance(loss, KLDivergenceLoss)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(KLDivergenceLoss.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        config = KLDivergenceLossConfig()
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {"_target_"}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"

    def test_default_weight(self):
        config = KLDivergenceLossConfig()
        assert config.weight == 0.0001


@pytest.mark.unit
class TestTrajectoryLengthLossConfig:
    def test_config_has_correct_target(self):
        config = TrajectoryLengthLossConfig()
        assert config._target_ == "refactoring.metrics.TrajectoryLengthLoss"

    def test_config_instantiates_correctly(self):
        config = TrajectoryLengthLossConfig()
        loss = instantiate(config)
        assert isinstance(loss, TrajectoryLengthLoss)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(TrajectoryLengthLoss.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        config = TrajectoryLengthLossConfig()
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {"_target_"}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"


@pytest.mark.unit
class TestTrajectorySmoothnessConfig:
    def test_config_has_correct_target(self):
        config = TrajectorySmoothnessConfig()
        assert config._target_ == "refactoring.metrics.TrajectorySmoothness"

    def test_config_instantiates_correctly(self):
        config = TrajectorySmoothnessConfig()
        loss = instantiate(config)
        assert isinstance(loss, TrajectorySmoothness)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(TrajectorySmoothness.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        config = TrajectorySmoothnessConfig()
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {"_target_"}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"


@pytest.mark.unit
class TestPhaseClassificationLossConfig:
    def test_config_has_correct_target(self):
        config = PhaseClassificationLossConfig()
        assert config._target_ == "refactoring.metrics.PhaseClassificationLoss"

    def test_config_instantiates_correctly(self):
        config = PhaseClassificationLossConfig()
        loss = instantiate(config)
        assert isinstance(loss, PhaseClassificationLoss)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(PhaseClassificationLoss.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        config = PhaseClassificationLossConfig()
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {"_target_"}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"


@pytest.mark.unit
class TestActionReconstructionLossConfig:
    def test_config_has_correct_target(self):
        config = ActionReconstructionLossConfig()
        assert config._target_ == "refactoring.metrics.ActionReconstructionLoss"

    def test_config_instantiates_correctly(self):
        config = ActionReconstructionLossConfig()
        loss = instantiate(config)
        assert isinstance(loss, ActionReconstructionLoss)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(ActionReconstructionLoss.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        config = ActionReconstructionLossConfig()
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {"_target_"}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"

    def test_default_use_vae(self):
        config = ActionReconstructionLossConfig()
        assert config.use_vae is False


@pytest.mark.unit
class TestPhaseActionLossConfig:
    def test_config_has_correct_target(self):
        config = PhaseActionLossConfig()
        assert config._target_ == "refactoring.metrics.PhaseActionLoss"

    def test_config_instantiates_correctly(self):
        config = PhaseActionLossConfig()
        loss = instantiate(config)
        assert isinstance(loss, PhaseActionLoss)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(PhaseActionLoss.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        config = PhaseActionLossConfig()
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {"_target_"}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"

    def test_default_use_vae(self):
        config = PhaseActionLossConfig()
        assert config.use_vae is False


@pytest.mark.unit
class TestCompositeLossConfig:
    def test_config_has_correct_target(self):
        config = CompositeLossConfig()
        assert config._target_ == "refactoring.metrics.CompositeLoss"

    def test_config_instantiates_correctly(self):
        config = CompositeLossConfig()
        loss = instantiate(config)
        assert isinstance(loss, CompositeLoss)

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(CompositeLoss.__init__)
        params = set(sig.parameters.keys()) - {"self"}
        config = CompositeLossConfig()
        config_dict = OmegaConf.structured(config)
        config_keys = set(config_dict.keys()) - {"_target_"}
        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"

    def test_default_loss_modules_empty(self):
        config = CompositeLossConfig()
        assert config.loss_modules == {}
