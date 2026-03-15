"""Tests for versatil.configs.training module."""
import dataclasses

import pytest

from versatil.configs.training import (
    AdamConfig,
    AdamWConfig,
    OptimizerConfig,
    ParameterGroupConfig,
    SGDConfig,
    TrainingConfig,
)


@pytest.mark.unit
class TestParameterGroupConfig:

    @pytest.mark.parametrize("name", ["backbone", "decoder"])
    @pytest.mark.parametrize("learning_rate", [1e-5, 1e-3])
    def test_stores_configuration(self, name, learning_rate):
        config = ParameterGroupConfig(name=name, lr=learning_rate)
        assert config.name == name
        assert config.lr == learning_rate

    @pytest.mark.parametrize("weight_decay", [None, 1e-4])
    def test_stores_optional_weight_decay(self, weight_decay):
        config = ParameterGroupConfig(
            name="group", lr=1e-4, weight_decay=weight_decay
        )
        assert config.weight_decay == weight_decay

    @pytest.mark.parametrize("params_pattern", [None, "backbone.*"])
    def test_stores_optional_params_pattern(self, params_pattern):
        config = ParameterGroupConfig(
            name="group", lr=1e-4, params_pattern=params_pattern
        )
        assert config.params_pattern == params_pattern


@pytest.mark.unit
class TestOptimizerConfig:

    def test_target_class_is_required(self):
        # target_class uses dataclasses.MISSING, making it a required positional arg
        with pytest.raises(TypeError, match="target_class"):
            OptimizerConfig()

    @pytest.mark.parametrize("learning_rate", [1e-4, 1e-3])
    def test_stores_learning_rate(self, learning_rate):
        config = OptimizerConfig(target_class="torch.optim.AdamW", lr=learning_rate)
        assert config.lr == learning_rate

    def test_param_groups_default_to_empty_list(self):
        config = OptimizerConfig(target_class="torch.optim.AdamW")
        assert config.param_groups == []

    def test_stores_param_groups(self):
        groups = [
            ParameterGroupConfig(name="backbone", lr=1e-5),
            ParameterGroupConfig(name="decoder", lr=1e-4),
        ]
        config = OptimizerConfig(
            target_class="torch.optim.AdamW", param_groups=groups
        )
        assert len(config.param_groups) == 2
        assert config.param_groups[0].name == "backbone"
        assert config.param_groups[1].name == "decoder"


@pytest.mark.unit
class TestAdamWConfig:

    def test_target_class_points_to_adamw(self):
        config = AdamWConfig()
        assert config.target_class == "torch.optim.AdamW"

    @pytest.mark.parametrize("learning_rate", [1e-4, 1e-3])
    @pytest.mark.parametrize("weight_decay", [1e-4, 0.0])
    @pytest.mark.parametrize("betas", [(0.9, 0.999), (0.95, 0.999)])
    def test_stores_configuration(self, learning_rate, weight_decay, betas):
        config = AdamWConfig(
            lr=learning_rate, weight_decay=weight_decay, betas=betas
        )
        assert config.lr == learning_rate
        assert config.weight_decay == weight_decay
        assert config.betas == betas

    def test_inherits_from_optimizer_config(self):
        config = AdamWConfig()
        assert isinstance(config, OptimizerConfig)


@pytest.mark.unit
class TestAdamConfig:

    def test_target_class_points_to_adam(self):
        config = AdamConfig()
        assert config.target_class == "torch.optim.Adam"

    @pytest.mark.parametrize("learning_rate", [1e-4, 5e-4])
    @pytest.mark.parametrize("weight_decay", [0.0, 1e-5])
    def test_stores_configuration(self, learning_rate, weight_decay):
        config = AdamConfig(lr=learning_rate, weight_decay=weight_decay)
        assert config.lr == learning_rate
        assert config.weight_decay == weight_decay

    def test_inherits_from_optimizer_config(self):
        config = AdamConfig()
        assert isinstance(config, OptimizerConfig)


@pytest.mark.unit
class TestSGDConfig:

    def test_target_class_points_to_sgd(self):
        config = SGDConfig()
        assert config.target_class == "torch.optim.SGD"

    @pytest.mark.parametrize("learning_rate", [1e-2, 1e-1])
    @pytest.mark.parametrize("momentum", [0.0, 0.9])
    @pytest.mark.parametrize("nesterov", [True, False])
    def test_stores_configuration(self, learning_rate, momentum, nesterov):
        config = SGDConfig(
            lr=learning_rate, momentum=momentum, nesterov=nesterov
        )
        assert config.lr == learning_rate
        assert config.momentum == momentum
        assert config.nesterov == nesterov

    def test_inherits_from_optimizer_config(self):
        config = SGDConfig()
        assert isinstance(config, OptimizerConfig)


@pytest.mark.unit
class TestTrainingConfig:

    @pytest.mark.parametrize("num_epochs", [50, 200])
    @pytest.mark.parametrize("use_ema", [True, False])
    @pytest.mark.parametrize("clip_gradient_norm", [True, False])
    def test_stores_configuration(self, num_epochs, use_ema, clip_gradient_norm):
        config = TrainingConfig(
            num_epochs=num_epochs,
            use_ema=use_ema,
            clip_gradient_norm=clip_gradient_norm,
        )
        assert config.num_epochs == num_epochs
        assert config.use_ema == use_ema
        assert config.clip_gradient_norm == clip_gradient_norm

    def test_default_optimizer_is_adamw(self):
        config = TrainingConfig()
        assert isinstance(config.optimizer, AdamWConfig)
        assert config.optimizer.target_class == "torch.optim.AdamW"

    @pytest.mark.parametrize("lr_schedule", [None, "cosine", "linear"])
    def test_stores_lr_schedule(self, lr_schedule):
        config = TrainingConfig(lr_schedule=lr_schedule)
        assert config.lr_schedule == lr_schedule

    @pytest.mark.parametrize("swa_lrs", [None, 1e-5])
    def test_stores_swa_configuration(self, swa_lrs):
        config = TrainingConfig(swa_lrs=swa_lrs)
        assert config.swa_lrs == swa_lrs

    def test_has_all_expected_fields(self):
        field_names = {f.name for f in dataclasses.fields(TrainingConfig)}
        expected = {
            "num_epochs",
            "gradient_accumulate_every",
            "optimizer",
            "clip_gradient_norm",
            "clip_max_norm",
            "lr_schedule",
            "lr_warmup_steps",
            "use_ema",
            "ema_power",
            "swa_lrs",
            "swa_epoch_start",
            "swa_annealing_epochs",
            "tune_lr",
            "early_stopping_patience",
            "reduce_lr_on_plateau",
            "reduce_lr_patience",
            "reduce_lr_cooldown",
        }
        assert expected == field_names
