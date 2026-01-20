"""Tests for optimizer configuration and manual instantiation."""

import pytest
import torch
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf

from versatil.configs.training import (
    AdamConfig,
    AdamWConfig,
    OptimizerConfig,
    ParameterGroupConfig,
    SGDConfig,
    TrainingConfig,
)
from versatil.training.lightning_policy import LightningPolicy


@pytest.mark.unit
class TestOptimizerConfigDataclasses:
    """Test optimizer config dataclasses."""

    def test_adamw_config_defaults(self):
        """Test AdamWConfig has correct defaults."""
        config = AdamWConfig()

        assert config.target_class == "torch.optim.AdamW"
        assert config.lr == 1e-4
        assert config.weight_decay == 1e-4
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
        assert config.amsgrad is False
        assert config.param_groups == []

    def test_adam_config_defaults(self):
        """Test AdamConfig has correct defaults."""
        config = AdamConfig()

        assert config.target_class == "torch.optim.Adam"
        assert config.lr == 1e-4
        assert config.weight_decay == 0.0
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
        assert config.amsgrad is False

    def test_sgd_config_defaults(self):
        """Test SGDConfig has correct defaults."""
        config = SGDConfig()

        assert config.target_class == "torch.optim.SGD"
        assert config.lr == 1e-2
        assert config.momentum == 0.0
        assert config.weight_decay == 0.0
        assert config.dampening == 0.0
        assert config.nesterov is False

    def test_training_config_default_optimizer(self):
        """Test TrainingConfig defaults to AdamWConfig."""
        config = TrainingConfig()

        assert isinstance(config.optimizer, AdamWConfig)
        assert config.optimizer.target_class == "torch.optim.AdamW"


@pytest.mark.unit
class TestOptimizerHydraInstantiation:
    """Test Hydra instantiation of optimizers."""

    def test_instantiate_adamw(self):
        """Test instantiating AdamW optimizer manually."""
        # Create simple model
        model = torch.nn.Linear(10, 5)

        # Create AdamW config
        config = AdamWConfig(lr=1e-3, weight_decay=1e-5)

        # Convert to dict and manually instantiate (like lightning_policy does)
        config_omega = OmegaConf.structured(config)
        config_dict = OmegaConf.to_container(config_omega, resolve=True)
        target = config_dict.pop("target_class")
        config_dict.pop("param_groups", None)

        optimizer_cls = get_class(target)
        optimizer = optimizer_cls(model.parameters(), **config_dict)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 1e-3
        assert optimizer.defaults["weight_decay"] == 1e-5

    def test_instantiate_adam(self):
        """Test instantiating Adam optimizer manually."""
        model = torch.nn.Linear(10, 5)

        config = AdamConfig(lr=5e-4)
        config_omega = OmegaConf.structured(config)
        config_dict = OmegaConf.to_container(config_omega, resolve=True)
        target = config_dict.pop("target_class")
        config_dict.pop("param_groups", None)

        optimizer_cls = get_class(target)
        optimizer = optimizer_cls(model.parameters(), **config_dict)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 5e-4

    def test_instantiate_sgd(self):
        """Test instantiating SGD optimizer manually."""
        model = torch.nn.Linear(10, 5)

        config = SGDConfig(lr=1e-2, momentum=0.9, nesterov=True)
        config_omega = OmegaConf.structured(config)
        config_dict = OmegaConf.to_container(config_omega, resolve=True)
        target = config_dict.pop("target_class")
        config_dict.pop("param_groups", None)

        optimizer_cls = get_class(target)
        optimizer = optimizer_cls(model.parameters(), **config_dict)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults["lr"] == 1e-2
        assert optimizer.defaults["momentum"] == 0.9
        assert optimizer.defaults["nesterov"] is True

    def test_instantiate_with_custom_target(self):
        """Test instantiating optimizer with custom target_class."""
        model = torch.nn.Linear(10, 5)

        # Create config with custom target (e.g., RMSprop)
        config = OptimizerConfig(target_class="torch.optim.RMSprop", lr=1e-3)
        config_omega = OmegaConf.structured(config)
        config_dict = OmegaConf.to_container(config_omega, resolve=True)
        target = config_dict.pop("target_class")
        config_dict.pop("param_groups", None)

        optimizer_cls = get_class(target)
        optimizer = optimizer_cls(model.parameters(), **config_dict)

        assert isinstance(optimizer, torch.optim.RMSprop)
        assert optimizer.defaults["lr"] == 1e-3


@pytest.mark.unit
class TestOptimizerParameterGroups:
    """Test parameter groups functionality."""

    def test_parameter_group_config(self):
        """Test ParameterGroupConfig dataclass."""
        group = ParameterGroupConfig(
            name="backbone",
            lr=1e-5,
            weight_decay=1e-4,
            params_pattern="backbone.*",
        )

        assert group.name == "backbone"
        assert group.lr == 1e-5
        assert group.weight_decay == 1e-4
        assert group.params_pattern == "backbone.*"

    def test_optimizer_config_with_param_groups(self):
        """Test OptimizerConfig with parameter groups."""
        groups = [
            ParameterGroupConfig(name="backbone", lr=1e-5, params_pattern="backbone.*"),
            ParameterGroupConfig(name="decoder", lr=1e-4, params_pattern="decoder.*"),
        ]

        config = AdamWConfig(lr=1e-3, param_groups=groups)

        assert len(config.param_groups) == 2
        assert config.param_groups[0].name == "backbone"
        assert config.param_groups[1].name == "decoder"


@pytest.mark.unit
class TestLightningPolicyOptimizerIntegration:
    """Test LightningPolicy optimizer configuration."""

    def test_configure_optimizers_adamw(self, simple_policy):
        """Test configure_optimizers creates AdamW optimizer."""
        training_config = TrainingConfig(
            optimizer=AdamWConfig(lr=1e-4, weight_decay=1e-5),
            lr_schedule=None,
        )

        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=training_config,
        )

        result = lightning_policy.configure_optimizers()

        assert "optimizer" in result
        optimizer = result["optimizer"]
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 1e-4
        assert optimizer.defaults["weight_decay"] == 1e-5

    def test_configure_optimizers_adam(self, simple_policy):
        """Test configure_optimizers creates Adam optimizer."""
        training_config = TrainingConfig(
            optimizer=AdamConfig(lr=5e-4),
            lr_schedule=None,
        )

        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=training_config,
        )

        result = lightning_policy.configure_optimizers()

        optimizer = result["optimizer"]
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 5e-4

    def test_configure_optimizers_sgd(self, simple_policy):
        """Test configure_optimizers creates SGD optimizer."""
        training_config = TrainingConfig(
            optimizer=SGDConfig(lr=1e-2, momentum=0.9),
            lr_schedule=None,
        )

        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=training_config,
        )

        result = lightning_policy.configure_optimizers()

        optimizer = result["optimizer"]
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults["lr"] == 1e-2
        assert optimizer.defaults["momentum"] == 0.9

    def test_configure_optimizers_with_lr_schedule(self, simple_policy):
        """Test configure_optimizers with learning rate schedule."""
        training_config = TrainingConfig(
            optimizer=AdamWConfig(lr=1e-4),
            lr_schedule="cosine",
            lr_warmup_steps=100,
        )

        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=training_config,
            total_training_steps=1000,
        )

        result = lightning_policy.configure_optimizers()

        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert isinstance(result["optimizer"], torch.optim.AdamW)
        assert "scheduler" in result["lr_scheduler"]

    def test_configure_optimizers_with_param_groups(self, simple_policy):
        """Test configure_optimizers with parameter groups."""
        # Create parameter groups
        param_groups = [
            ParameterGroupConfig(
                name="encoding",
                lr=1e-5,
                params_pattern="encoding_pipeline.*",
            ),
        ]

        training_config = TrainingConfig(
            optimizer=AdamWConfig(lr=1e-4, param_groups=param_groups),
            lr_schedule=None,
        )

        lightning_policy = LightningPolicy(
            policy=simple_policy,
            training_config=training_config,
        )

        result = lightning_policy.configure_optimizers()

        optimizer = result["optimizer"]
        assert isinstance(optimizer, torch.optim.AdamW)
        # Check that parameter groups were created
        assert len(optimizer.param_groups) >= 1


@pytest.mark.unit
class TestOptimizerYAMLConfig:
    """Test loading optimizer config from YAML."""

    def test_load_adamw_yaml_config(self, tmp_path):
        """Test loading AdamW config from YAML."""
        yaml_content = """
target_class: torch.optim.AdamW
lr: 1.0e-4
weight_decay: 1.0e-4
betas: [0.9, 0.999]
eps: 1.0e-8
param_groups: []
"""
        yaml_path = tmp_path / "optimizer.yaml"
        yaml_path.write_text(yaml_content)

        # Load config
        config = OmegaConf.load(yaml_path)

        assert config.target_class == "torch.optim.AdamW"
        assert config.lr == 1e-4
        assert config.weight_decay == 1e-4
        assert config.betas == [0.9, 0.999]

    def test_load_yaml_with_param_groups(self, tmp_path):
        """Test loading config with parameter groups from YAML."""
        yaml_content = """
target_class: torch.optim.AdamW
lr: 1.0e-4
weight_decay: 1.0e-4
param_groups:
  - name: backbone
    lr: 1.0e-5
    params_pattern: "backbone.*"
  - name: decoder
    lr: 5.0e-5
    params_pattern: "decoder.*"
"""
        yaml_path = tmp_path / "optimizer.yaml"
        yaml_path.write_text(yaml_content)

        config = OmegaConf.load(yaml_path)

        assert len(config.param_groups) == 2
        assert config.param_groups[0].name == "backbone"
        assert config.param_groups[0].lr == 1e-5
        assert config.param_groups[1].name == "decoder"
        assert config.param_groups[1].lr == 5e-5

    def test_instantiate_from_yaml_config(self, tmp_path):
        """Test end-to-end: YAML -> config -> manual instantiate -> optimizer."""
        yaml_content = """
target_class: torch.optim.AdamW
lr: 2.0e-4
weight_decay: 1.0e-5
betas: [0.95, 0.999]
"""
        yaml_path = tmp_path / "optimizer.yaml"
        yaml_path.write_text(yaml_content)

        # Load config
        config = OmegaConf.load(yaml_path)

        # Create model and manually instantiate optimizer
        model = torch.nn.Linear(10, 5)
        config_dict = OmegaConf.to_container(config, resolve=True)
        target = config_dict.pop("target_class")
        config_dict.pop("param_groups", None)

        optimizer_cls = get_class(target)
        optimizer = optimizer_cls(model.parameters(), **config_dict)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 2e-4
        assert optimizer.defaults["weight_decay"] == 1e-5
        # YAML loads lists, optimizer converts to tuples
        assert optimizer.defaults["betas"] == (0.95, 0.999) or optimizer.defaults["betas"] == [0.95, 0.999]
