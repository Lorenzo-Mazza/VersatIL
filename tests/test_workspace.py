"""Tests for versatil.workspace module."""

import os
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from versatil.configs.experiment import ExperimentConfig
from versatil.configs.training import AdamWConfig, TrainingConfig
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.metrics import MoELoss
from versatil.models.decoding.algorithm import VariationalAlgorithm
from versatil.models.decoding.decoders.factory.free_action_transformer import (
    FreeActionTransformer,
)
from versatil.models.decoding.decoders.factory.phase_act import PhaseACT
from versatil.models.policy import Policy
from versatil.training.callbacks import (
    ConfusionMatrixCallback,
    EMACallback,
    ExpertUsageCallback,
    GradientNormCallback,
    LatentVisualizationCallback,
    ReduceLROnPlateauCallback,
    ResumableEarlyStopping,
)
from versatil.training.lightning_policy import LightningPolicy
from versatil.workspace import Workspace


@pytest.fixture
def experiment_config_factory() -> Callable[..., MagicMock]:
    """Factory for mock ExperimentConfig instances."""

    def factory(
        name: str = "test_experiment",
        seed: int = 42,
        checkpoint_folder: str = "/tmp/test_checkpoints",
        resume_from: str | None = None,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        device: str = "cpu",
        distributed: bool = False,
        precision: str = "32",
        float32_matmul_precision: str | None = None,
        checkpoint_every: int = 10,
        val_every: int = 1,
        plot_every: int = 200,
        validate_loss_keys: bool = True,
    ) -> MagicMock:
        config = MagicMock(spec=ExperimentConfig)
        config.name = name
        config.seed = seed
        config.checkpoint_folder = checkpoint_folder
        config.resume_from = resume_from
        config.use_wandb = use_wandb
        config.wandb_project = wandb_project
        config.wandb_entity = wandb_entity
        config.device = device
        config.distributed = distributed
        config.precision = precision
        config.float32_matmul_precision = float32_matmul_precision
        config.checkpoint_every = checkpoint_every
        config.val_every = val_every
        config.plot_every = plot_every
        config.validate_loss_keys = validate_loss_keys
        return config

    return factory


@pytest.fixture
def mock_training_config_factory() -> Callable[..., MagicMock]:
    """Factory for mock TrainingConfig instances."""

    def factory(
        num_epochs: int = 10,
        gradient_accumulate_every: int = 1,
        clip_gradient_norm: bool = False,
        clip_max_norm: float = 0.1,
        use_ema: bool = False,
        ema_power: float = 0.75,
        swa_lrs: float | None = None,
        swa_epoch_start: float = 0.5,
        swa_annealing_epochs: int = 10,
        tune_lr: bool = False,
        early_stopping_patience: int = 10,
        reduce_lr_on_plateau: bool = False,
        reduce_lr_patience: int = 10,
        reduce_lr_cooldown: int = 10,
        compile: bool = False,
        compile_mode: str = "default",
        lr: float = 1e-4,
    ) -> MagicMock:
        config = MagicMock(spec=TrainingConfig)
        config.num_epochs = num_epochs
        config.gradient_accumulate_every = gradient_accumulate_every
        config.clip_gradient_norm = clip_gradient_norm
        config.clip_max_norm = clip_max_norm
        config.use_ema = use_ema
        config.ema_power = ema_power
        config.swa_lrs = swa_lrs
        config.swa_epoch_start = swa_epoch_start
        config.swa_annealing_epochs = swa_annealing_epochs
        config.tune_lr = tune_lr
        config.early_stopping_patience = early_stopping_patience
        config.reduce_lr_on_plateau = reduce_lr_on_plateau
        config.reduce_lr_patience = reduce_lr_patience
        config.reduce_lr_cooldown = reduce_lr_cooldown
        config.compile = compile
        config.compile_mode = compile_mode
        config.optimizer = MagicMock(spec=AdamWConfig)
        config.optimizer.lr = lr
        return config

    return factory


@pytest.fixture
def mock_workspace_policy_factory() -> Callable[..., MagicMock]:
    """Factory for mock Policy instances with configurable decoder and algorithm."""

    def factory(
        decoder_type: str | None = None,
        algorithm_type: str | None = None,
        has_moe_loss: bool = False,
    ) -> MagicMock:
        policy = MagicMock(spec=Policy)
        policy.set_normalizer = MagicMock()
        policy.set_tokenizer = MagicMock()
        policy.set_denoising_thresholds = MagicMock()
        policy.set_gripper_class_weights = MagicMock()
        policy.predict_action = MagicMock(return_value={"action": torch.zeros(2, 7)})

        if decoder_type == "phase_act":
            policy.decoder = MagicMock(spec=PhaseACT)
        elif decoder_type == "free_action_transformer":
            policy.decoder = MagicMock(spec=FreeActionTransformer)
        else:
            policy.decoder = MagicMock()

        if algorithm_type == "variational":
            policy.algorithm = MagicMock(spec=VariationalAlgorithm)
        else:
            policy.algorithm = MagicMock()

        loss_module = MagicMock()
        if has_moe_loss:
            moe_loss = MagicMock(spec=MoELoss)
            loss_module.loss_modules = {"moe": moe_loss}
        else:
            plain_loss = MagicMock()
            loss_module.loss_modules = {"regression": plain_loss}
        policy.loss_module = loss_module

        return policy

    return factory


@pytest.fixture
def main_config_factory(
    experiment_config_factory: Callable[..., MagicMock],
    mock_training_config_factory: Callable[..., MagicMock],
) -> Callable[..., MagicMock]:
    """Factory for mock MainConfig instances combining experiment and training configs."""

    def factory(
        experiment_kwargs: dict | None = None,
        training_kwargs: dict | None = None,
        policy: MagicMock | None = None,
    ) -> MagicMock:
        experiment_kwargs = experiment_kwargs or {}
        training_kwargs = training_kwargs or {}

        config = MagicMock()
        config.experiment = experiment_config_factory(**experiment_kwargs)
        config.training = mock_training_config_factory(**training_kwargs)
        config.policy = policy or MagicMock()
        return config

    return factory


@pytest.fixture
def original_yaml_config_factory() -> Callable[..., OmegaConf]:
    """Factory for OmegaConf YAML config objects."""

    def factory(name: str = "test_experiment") -> OmegaConf:
        yaml_config = OmegaConf.create(
            {
                "experiment": {"name": name},
                "training": {"optimizer": {"lr": 1e-4}},
            }
        )
        return yaml_config

    return factory


@pytest.fixture
def workspace_factory(
    main_config_factory: Callable[..., MagicMock],
    original_yaml_config_factory: Callable[..., OmegaConf],
    tmp_path: Path,
) -> Callable[..., Workspace]:
    """Factory for Workspace instances with mocked Hydra config."""

    def factory(
        experiment_kwargs: dict | None = None,
        training_kwargs: dict | None = None,
        policy: MagicMock | None = None,
        config_name: str = "test_config",
    ) -> Workspace:
        experiment_kwargs = experiment_kwargs or {}
        experiment_kwargs.setdefault("checkpoint_folder", str(tmp_path))

        config = main_config_factory(
            experiment_kwargs=experiment_kwargs,
            training_kwargs=training_kwargs,
            policy=policy,
        )
        yaml_config = original_yaml_config_factory(
            name=experiment_kwargs.get("name", "test_experiment"),
        )

        mock_hydra_cfg = MagicMock()
        mock_hydra_cfg.job.config_name = config_name

        with patch("versatil.workspace.HydraConfig.get", return_value=mock_hydra_cfg):
            workspace = Workspace(
                config=config,
                original_yaml_config=yaml_config,
            )
        return workspace

    return factory


@pytest.mark.unit
class TestWorkspaceInitialization:
    def test_experiment_name_combines_config_name_and_experiment_name(
        self, workspace_factory
    ):
        workspace = workspace_factory(
            experiment_kwargs={"name": "my_run"},
            config_name="my_config",
        )

        assert workspace.exp_name == "my_config/my_run"

    def test_experiment_name_defaults_to_experiment_when_config_name_empty(
        self, workspace_factory
    ):
        workspace = workspace_factory(
            experiment_kwargs={"name": "my_run"},
            config_name="",
        )

        # When config_name is empty string, the condition `if hydra_cfg.job.config_name`
        # is False, so it defaults to "experiment"
        assert workspace.exp_name == "experiment/my_run"

    def test_output_directory_is_created(self, workspace_factory, tmp_path):
        workspace = workspace_factory(
            experiment_kwargs={"name": "output_test"},
            config_name="cfg",
        )

        expected_dir = tmp_path / "cfg" / "output_test"
        assert workspace.output_dir == expected_dir
        assert expected_dir.exists()

    def test_initial_state_is_none(self, workspace_factory):
        workspace = workspace_factory()

        assert workspace.policy is None
        assert workspace.lightning_policy is None
        assert workspace.trainer is None
        assert workspace.train_loader is None
        assert workspace.val_loader is None
        assert workspace.normalizer is None
        assert workspace.tokenizer is None
        assert workspace.logger is None
        assert workspace.gripper_class_weights is None

    def test_seed_is_set_during_init(self, workspace_factory):
        seed = 123
        with (
            patch("versatil.workspace.torch.manual_seed") as mock_manual_seed,
            patch("versatil.workspace.np.random.seed") as mock_np_seed,
        ):
            workspace_factory(experiment_kwargs={"seed": seed})

            mock_manual_seed.assert_called_once_with(seed)
            mock_np_seed.assert_called_once_with(seed)

    def test_config_saved_during_init(self, workspace_factory, tmp_path):
        workspace_factory(
            experiment_kwargs={"name": "save_test"},
            config_name="cfg",
        )

        config_path = tmp_path / "cfg" / "save_test" / "config.yaml"
        assert config_path.exists()

    def test_updates_experiment_name_on_both_configs(self, workspace_factory):
        workspace = workspace_factory(
            experiment_kwargs={"name": "updated_name"},
            config_name="base_config",
        )

        expected_name = "base_config/updated_name"
        assert workspace.config.experiment.name == expected_name


@pytest.mark.unit
class TestSaveConfig:
    def test_saves_resolved_yaml_to_output_directory(self, workspace_factory, tmp_path):
        workspace = workspace_factory(
            experiment_kwargs={"name": "save_cfg"},
            config_name="base",
        )

        config_path = workspace.output_dir / "config.yaml"
        assert config_path.exists()

        loaded = OmegaConf.load(config_path)
        assert "experiment" in loaded


@pytest.mark.unit
class TestSetSeed:
    def test_cuda_seed_set_when_cuda_available(self, workspace_factory):
        # torch.manual_seed internally calls torch.cuda.manual_seed_all once,
        # and workspace._set_seed calls it again explicitly when CUDA is available,
        # resulting in two calls total.
        with (
            patch("versatil.workspace.torch.cuda.is_available", return_value=True),
            patch("versatil.workspace.torch.cuda.manual_seed_all") as mock_cuda_seed,
        ):
            workspace_factory(experiment_kwargs={"seed": 99})

            assert mock_cuda_seed.call_count == 2
            mock_cuda_seed.assert_called_with(99)

    def test_cuda_seed_not_set_when_cuda_unavailable(self, workspace_factory):
        # torch.manual_seed internally calls torch.cuda.manual_seed_all once
        # even when CUDA is unavailable. The workspace skips the explicit call
        # but the internal call from torch.manual_seed still happens.
        with (
            patch("versatil.workspace.torch.cuda.is_available", return_value=False),
            patch("versatil.workspace.torch.cuda.manual_seed_all") as mock_cuda_seed,
        ):
            workspace_factory(experiment_kwargs={"seed": 99})

            # Only the internal call from torch.manual_seed, not the explicit one
            assert mock_cuda_seed.call_count == 1


@pytest.mark.unit
class TestCreateLogger:
    def test_returns_none_when_wandb_disabled(self, workspace_factory):
        workspace = workspace_factory(
            experiment_kwargs={"use_wandb": False},
        )

        result = workspace._create_logger()

        assert result is None

    def test_returns_none_when_api_key_missing(self, workspace_factory):
        workspace = workspace_factory(
            experiment_kwargs={"use_wandb": True},
        )

        with patch.dict(os.environ, {}, clear=True):
            result = workspace._create_logger()

        assert result is None
        # Should also disable wandb in config
        assert workspace.config.experiment.use_wandb is False

    @patch("versatil.workspace.wandb")
    @patch("versatil.workspace.WandbLogger")
    def test_creates_wandb_logger_when_enabled_and_key_set(
        self, mock_wandb_logger_cls, mock_wandb, workspace_factory
    ):
        workspace = workspace_factory(
            experiment_kwargs={
                "use_wandb": True,
                "wandb_project": "test_project",
                "wandb_entity": "test_entity",
            },
        )

        mock_logger_instance = MagicMock(spec=WandbLogger)
        mock_wandb_logger_cls.return_value = mock_logger_instance

        with patch.dict(os.environ, {"WANDB_API_KEY": "fake_key"}):
            result = workspace._create_logger()

        assert result == mock_logger_instance
        mock_wandb_logger_cls.assert_called_once_with(
            project="test_project",
            entity="test_entity",
            name=workspace.exp_name,
            save_dir=workspace.output_dir,
            log_model=False,
        )

    @patch("versatil.workspace.wandb")
    @patch("versatil.workspace.WandbLogger")
    def test_logs_hyperparams_and_defines_metrics(
        self, mock_wandb_logger_cls, mock_wandb, workspace_factory
    ):
        workspace = workspace_factory(
            experiment_kwargs={"use_wandb": True},
        )

        mock_logger_instance = MagicMock(spec=WandbLogger)
        mock_wandb_logger_cls.return_value = mock_logger_instance

        with patch.dict(os.environ, {"WANDB_API_KEY": "fake_key"}):
            workspace._create_logger()

        mock_logger_instance.log_hyperparams.assert_called_once()
        mock_wandb.define_metric.assert_any_call("epoch")
        mock_wandb.define_metric.assert_any_call("*", step_metric="epoch")


@pytest.mark.unit
class TestCreateStrategy:
    def test_returns_auto_when_not_distributed(self, workspace_factory):
        workspace = workspace_factory(
            experiment_kwargs={"distributed": False},
        )

        result = workspace._create_strategy()

        assert result == "auto"

    def test_returns_ddp_strategy_when_distributed(self, workspace_factory):
        workspace = workspace_factory(
            experiment_kwargs={"distributed": True},
        )

        result = workspace._create_strategy()

        assert isinstance(result, DDPStrategy)

    def test_ddp_strategy_disables_find_unused_parameters(self, workspace_factory):
        workspace = workspace_factory(
            experiment_kwargs={"distributed": True},
        )

        strategy = workspace._create_strategy()

        assert strategy._ddp_kwargs["find_unused_parameters"] is False

    def test_ddp_strategy_enables_gradient_as_bucket_view(self, workspace_factory):
        workspace = workspace_factory(
            experiment_kwargs={"distributed": True},
        )

        strategy = workspace._create_strategy()

        assert strategy._ddp_kwargs["gradient_as_bucket_view"] is True


@pytest.mark.unit
class TestCreateCallbacks:
    def test_always_includes_checkpoint_gradient_norm_and_lr_monitor(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        callback_types = [type(cb) for cb in callbacks]
        assert ModelCheckpoint in callback_types
        assert GradientNormCallback in callback_types
        assert LearningRateMonitor in callback_types

    def test_ema_callback_added_when_use_ema_enabled(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={"use_ema": True, "ema_power": 0.9},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        ema_callbacks = [cb for cb in callbacks if isinstance(cb, EMACallback)]
        assert len(ema_callbacks) == 1
        assert ema_callbacks[0].power == 0.9

    def test_ema_callback_not_added_when_use_ema_disabled(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={"use_ema": False},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        ema_callbacks = [cb for cb in callbacks if isinstance(cb, EMACallback)]
        assert len(ema_callbacks) == 0

    def test_early_stopping_added_when_validation_loader_present(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={"early_stopping_patience": 5},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = MagicMock()

        callbacks = workspace._create_callbacks()

        early_stop_callbacks = [
            cb for cb in callbacks if isinstance(cb, ResumableEarlyStopping)
        ]
        assert len(early_stop_callbacks) == 1
        assert early_stop_callbacks[0].patience == 5

    def test_early_stopping_not_added_without_validation_loader(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        early_stop_callbacks = [
            cb for cb in callbacks if isinstance(cb, ResumableEarlyStopping)
        ]
        assert len(early_stop_callbacks) == 0

    def test_checkpoint_monitors_val_loss_when_validation_present(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = MagicMock()

        callbacks = workspace._create_callbacks()

        best_checkpoint = [
            cb
            for cb in callbacks
            if isinstance(cb, ModelCheckpoint) and "best" in (cb.filename or "")
        ]
        assert len(best_checkpoint) == 1
        assert best_checkpoint[0].monitor == "val_loss"

    def test_checkpoint_monitors_train_loss_when_no_validation(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        best_checkpoint = [
            cb
            for cb in callbacks
            if isinstance(cb, ModelCheckpoint) and "best" in (cb.filename or "")
        ]
        assert len(best_checkpoint) == 1
        assert best_checkpoint[0].monitor == "train_loss_epoch"

    def test_swa_callback_added_when_swa_lrs_set(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={
                "swa_lrs": 0.001,
                "swa_epoch_start": 0.5,
                "num_epochs": 100,
                "swa_annealing_epochs": 5,
            },
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        swa_callbacks = [
            cb for cb in callbacks if isinstance(cb, StochasticWeightAveraging)
        ]
        assert len(swa_callbacks) == 1

    def test_swa_callback_not_added_when_swa_lrs_none(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={"swa_lrs": None},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        swa_callbacks = [
            cb for cb in callbacks if isinstance(cb, StochasticWeightAveraging)
        ]
        assert len(swa_callbacks) == 0

    def test_swa_epoch_start_computed_as_fraction_of_total_epochs(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={
                "swa_lrs": 0.001,
                "swa_epoch_start": 0.8,
                "num_epochs": 100,
                "swa_annealing_epochs": 5,
            },
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        with patch("versatil.workspace.StochasticWeightAveraging") as mock_swa_cls:
            mock_swa_cls.return_value = MagicMock(spec=StochasticWeightAveraging)
            workspace._create_callbacks()

            mock_swa_cls.assert_called_once_with(
                swa_lrs=0.001,
                swa_epoch_start=80,  # int(0.8 * 100)
                annealing_epochs=5,
            )

    def test_confusion_matrix_callback_added_for_phase_act_decoder(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory(decoder_type="phase_act")
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        cm_callbacks = [
            cb for cb in callbacks if isinstance(cb, ConfusionMatrixCallback)
        ]
        assert len(cm_callbacks) == 1

    def test_confusion_matrix_callback_not_added_for_regular_decoder(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        cm_callbacks = [
            cb for cb in callbacks if isinstance(cb, ConfusionMatrixCallback)
        ]
        assert len(cm_callbacks) == 0

    def test_latent_visualization_added_for_variational_algorithm(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory(algorithm_type="variational")
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        latent_callbacks = [
            cb for cb in callbacks if isinstance(cb, LatentVisualizationCallback)
        ]
        assert len(latent_callbacks) == 1

    def test_latent_visualization_added_for_free_action_transformer_decoder(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory(decoder_type="free_action_transformer")
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        latent_callbacks = [
            cb for cb in callbacks if isinstance(cb, LatentVisualizationCallback)
        ]
        assert len(latent_callbacks) == 1

    def test_latent_visualization_not_added_for_regular_algorithm_and_decoder(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        latent_callbacks = [
            cb for cb in callbacks if isinstance(cb, LatentVisualizationCallback)
        ]
        assert len(latent_callbacks) == 0

    def test_reduce_lr_callback_added_when_enabled(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={
                "reduce_lr_on_plateau": True,
                "reduce_lr_patience": 5,
                "reduce_lr_cooldown": 3,
            },
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = MagicMock()

        callbacks = workspace._create_callbacks()

        reduce_lr_callbacks = [
            cb for cb in callbacks if isinstance(cb, ReduceLROnPlateauCallback)
        ]
        assert len(reduce_lr_callbacks) == 1

    def test_reduce_lr_monitors_val_loss_when_validation_present(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={
                "reduce_lr_on_plateau": True,
                "reduce_lr_patience": 5,
                "reduce_lr_cooldown": 3,
            },
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = MagicMock()

        with patch("versatil.workspace.ReduceLROnPlateauCallback") as mock_reduce_cls:
            mock_reduce_cls.return_value = MagicMock(spec=ReduceLROnPlateauCallback)
            workspace._create_callbacks()

            mock_reduce_cls.assert_called_once_with(
                monitor="val_loss",
                patience=5,
                cooldown=3,
            )

    def test_reduce_lr_monitors_train_loss_when_no_validation(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={
                "reduce_lr_on_plateau": True,
                "reduce_lr_patience": 5,
                "reduce_lr_cooldown": 3,
            },
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        with patch("versatil.workspace.ReduceLROnPlateauCallback") as mock_reduce_cls:
            mock_reduce_cls.return_value = MagicMock(spec=ReduceLROnPlateauCallback)
            workspace._create_callbacks()

            mock_reduce_cls.assert_called_once_with(
                monitor="train_loss",
                patience=5,
                cooldown=3,
            )

    def test_reduce_lr_not_added_when_disabled(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={"reduce_lr_on_plateau": False},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        reduce_lr_callbacks = [
            cb for cb in callbacks if isinstance(cb, ReduceLROnPlateauCallback)
        ]
        assert len(reduce_lr_callbacks) == 0

    def test_expert_usage_callback_added_when_moe_loss_present(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory(has_moe_loss=True)
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        expert_callbacks = [
            cb for cb in callbacks if isinstance(cb, ExpertUsageCallback)
        ]
        assert len(expert_callbacks) == 1

    def test_expert_usage_callback_not_added_without_moe_loss(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory(has_moe_loss=False)
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        expert_callbacks = [
            cb for cb in callbacks if isinstance(cb, ExpertUsageCallback)
        ]
        assert len(expert_callbacks) == 0

    def test_latest_checkpoint_saves_on_train_epoch_end_without_validation(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = None

        callbacks = workspace._create_callbacks()

        latest_checkpoints = [
            cb
            for cb in callbacks
            if isinstance(cb, ModelCheckpoint) and "latest" in (cb.filename or "")
        ]
        assert len(latest_checkpoints) == 1
        assert latest_checkpoints[0]._save_on_train_epoch_end is True

    def test_latest_checkpoint_does_not_save_on_train_epoch_end_with_validation(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.val_loader = MagicMock()

        callbacks = workspace._create_callbacks()

        latest_checkpoints = [
            cb
            for cb in callbacks
            if isinstance(cb, ModelCheckpoint) and "latest" in (cb.filename or "")
        ]
        assert len(latest_checkpoints) == 1
        assert latest_checkpoints[0]._save_on_train_epoch_end is False


@pytest.mark.unit
class TestSetupTrainer:
    def test_trainer_uses_gpu_accelerator_for_cuda_device(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            experiment_kwargs={"device": "cuda"},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        with patch("versatil.workspace.pl.Trainer") as mock_trainer_cls:
            workspace._setup_trainer()

            call_kwargs = mock_trainer_cls.call_args[1]
            assert call_kwargs["accelerator"] == "gpu"

    def test_trainer_uses_cpu_accelerator_for_cpu_device(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            experiment_kwargs={"device": "cpu"},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        with patch("versatil.workspace.pl.Trainer") as mock_trainer_cls:
            workspace._setup_trainer()

            call_kwargs = mock_trainer_cls.call_args[1]
            assert call_kwargs["accelerator"] == "cpu"

    def test_trainer_uses_auto_devices_when_distributed(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            experiment_kwargs={"distributed": True},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        with patch("versatil.workspace.pl.Trainer") as mock_trainer_cls:
            workspace._setup_trainer()

            call_kwargs = mock_trainer_cls.call_args[1]
            assert call_kwargs["devices"] == "auto"

    def test_trainer_uses_single_device_when_not_distributed(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            experiment_kwargs={"distributed": False},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        with patch("versatil.workspace.pl.Trainer") as mock_trainer_cls:
            workspace._setup_trainer()

            call_kwargs = mock_trainer_cls.call_args[1]
            assert call_kwargs["devices"] == 1

    def test_gradient_clipping_passed_when_enabled(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={
                "clip_gradient_norm": True,
                "clip_max_norm": 0.5,
            },
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        with patch("versatil.workspace.pl.Trainer") as mock_trainer_cls:
            workspace._setup_trainer()

            call_kwargs = mock_trainer_cls.call_args[1]
            assert call_kwargs["gradient_clip_val"] == 0.5

    def test_gradient_clipping_none_when_disabled(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={"clip_gradient_norm": False},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        with patch("versatil.workspace.pl.Trainer") as mock_trainer_cls:
            workspace._setup_trainer()

            call_kwargs = mock_trainer_cls.call_args[1]
            assert call_kwargs["gradient_clip_val"] is None

    def test_float32_matmul_precision_set_when_configured(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            experiment_kwargs={"float32_matmul_precision": "high"},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        with (
            patch("versatil.workspace.pl.Trainer"),
            patch(
                "versatil.workspace.torch.set_float32_matmul_precision"
            ) as mock_set_precision,
        ):
            workspace._setup_trainer()

            mock_set_precision.assert_called_once_with("high")

    def test_float32_matmul_precision_not_set_when_none(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            experiment_kwargs={"float32_matmul_precision": None},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        with (
            patch("versatil.workspace.pl.Trainer"),
            patch(
                "versatil.workspace.torch.set_float32_matmul_precision"
            ) as mock_set_precision,
        ):
            workspace._setup_trainer()

            mock_set_precision.assert_not_called()

    def test_val_check_interval_zero_when_no_validation(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            experiment_kwargs={"val_every": 5},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = None

        with patch("versatil.workspace.pl.Trainer") as mock_trainer_cls:
            workspace._setup_trainer()

            call_kwargs = mock_trainer_cls.call_args[1]
            assert call_kwargs["check_val_every_n_epoch"] == 0
            assert call_kwargs["limit_val_batches"] == 0

    def test_val_check_interval_uses_config_when_validation_present(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            experiment_kwargs={"val_every": 5},
            policy=policy,
        )
        workspace.policy = policy
        workspace.val_loader = MagicMock()

        with patch("versatil.workspace.pl.Trainer") as mock_trainer_cls:
            workspace._setup_trainer()

            call_kwargs = mock_trainer_cls.call_args[1]
            assert call_kwargs["check_val_every_n_epoch"] == 5
            assert call_kwargs["limit_val_batches"] == 1.0


@pytest.mark.unit
class TestSetupPolicy:
    def test_sets_normalizer_tokenizer_and_thresholds_on_policy(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)
        workspace.policy = None  # Reset to simulate pre-setup state

        mock_normalizer = MagicMock(spec=LinearNormalizer)
        workspace.normalizer = mock_normalizer
        workspace.tokenizer = None
        workspace.denoising_thresholds = {"key": 0.5}
        workspace.gripper_class_weights = None

        mock_train_loader = MagicMock()
        mock_train_loader.__len__ = MagicMock(return_value=100)
        workspace.train_loader = mock_train_loader

        workspace.config.policy = policy

        with patch.object(workspace, "_initialize_lazy_modules"):
            workspace._setup_policy()

        policy.set_normalizer.assert_called_once_with(mock_normalizer)
        policy.set_tokenizer.assert_called_once_with(None)
        policy.set_denoising_thresholds.assert_called_once_with({"key": 0.5})
        policy.set_gripper_class_weights.assert_called_once_with(None)

    def test_computes_total_training_steps_correctly(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={
                "num_epochs": 10,
                "gradient_accumulate_every": 2,
            },
            policy=policy,
        )
        workspace.policy = None

        workspace.normalizer = MagicMock(spec=LinearNormalizer)
        workspace.tokenizer = None
        workspace.denoising_thresholds = {}
        workspace.gripper_class_weights = None

        mock_train_loader = MagicMock()
        mock_train_loader.__len__ = MagicMock(return_value=100)
        workspace.train_loader = mock_train_loader

        workspace.config.policy = policy

        with (
            patch.object(workspace, "_initialize_lazy_modules"),
            patch("versatil.workspace.LightningPolicy") as mock_lightning_cls,
        ):
            mock_lightning_cls.return_value = MagicMock(spec=LightningPolicy)
            workspace._setup_policy()

            # steps_per_epoch = 100 // 2 = 50
            # total = 50 * 10 = 500
            mock_lightning_cls.assert_called_once_with(
                policy=policy,
                training_config=workspace.config.training,
                total_training_steps=500,
            )


@pytest.mark.unit
class TestInitializeLazyModules:
    def test_raises_runtime_error_when_train_loader_is_none(self, workspace_factory):
        workspace = workspace_factory()
        workspace.train_loader = None

        with pytest.raises(
            RuntimeError,
            match="Train loader is not initialized or empty, cannot initialize lazy modules",
        ):
            workspace._initialize_lazy_modules()

    def test_raises_runtime_error_when_train_loader_is_empty(self, workspace_factory):
        workspace = workspace_factory()
        mock_loader = MagicMock()
        mock_loader.__len__ = MagicMock(return_value=0)
        workspace.train_loader = mock_loader

        with pytest.raises(
            RuntimeError,
            match="Train loader is not initialized or empty, cannot initialize lazy modules",
        ):
            workspace._initialize_lazy_modules()

    def test_performs_forward_pass_in_train_mode(self, workspace_factory):
        workspace = workspace_factory(experiment_kwargs={"device": "cpu"})

        mock_batch = {"obs": torch.zeros(2, 3)}
        mock_loader = MagicMock()
        mock_loader.__len__ = MagicMock(return_value=10)
        mock_loader.__iter__ = MagicMock(return_value=iter([mock_batch]))
        workspace.train_loader = mock_loader

        mock_lightning_policy = MagicMock(spec=LightningPolicy)
        workspace.lightning_policy = mock_lightning_policy

        with patch("versatil.workspace.to_device", return_value=mock_batch):
            workspace._initialize_lazy_modules()

        mock_lightning_policy.eval.assert_not_called()
        mock_lightning_policy.training_step.assert_called_once_with(mock_batch, 0)
        mock_lightning_policy.train.assert_called_once()


@pytest.mark.unit
class TestLoadCheckpoint:
    def test_raises_runtime_error_when_policy_is_none(self, workspace_factory):
        workspace = workspace_factory()
        workspace.policy = None
        workspace.lightning_policy = MagicMock()

        with (
            patch("versatil.workspace.torch.load", return_value={"state_dict": {}}),
            pytest.raises(
                RuntimeError,
                match="Policy must be initialized before loading checkpoint",
            ),
        ):
            workspace.load_checkpoint("/fake/path.ckpt")

    def test_raises_runtime_error_when_lightning_policy_is_none(
        self, workspace_factory
    ):
        workspace = workspace_factory()
        workspace.policy = MagicMock()
        workspace.lightning_policy = None

        with (
            patch("versatil.workspace.torch.load", return_value={"state_dict": {}}),
            pytest.raises(
                RuntimeError,
                match="LightningPolicy must be initialized before loading checkpoint",
            ),
        ):
            workspace.load_checkpoint("/fake/path.ckpt")

    def test_raises_value_error_for_unrecognized_checkpoint_format(
        self, workspace_factory
    ):
        workspace = workspace_factory()
        workspace.policy = MagicMock()
        workspace.lightning_policy = MagicMock()

        with (
            patch("versatil.workspace.torch.load", return_value={"weights": {}}),
            pytest.raises(
                ValueError,
                match="Checkpoint format not recognized",
            ),
        ):
            workspace.load_checkpoint("/fake/path.ckpt")

    def test_loads_state_dict_into_lightning_policy(self, workspace_factory):
        workspace = workspace_factory()
        workspace.policy = MagicMock()
        workspace.lightning_policy = MagicMock()

        mock_state_dict = {"layer.weight": torch.zeros(3, 3)}

        with patch(
            "versatil.workspace.torch.load",
            return_value={"state_dict": mock_state_dict},
        ):
            workspace.load_checkpoint("/fake/path.ckpt")

        workspace.lightning_policy.load_state_dict.assert_called_once_with(
            mock_state_dict
        )

    def test_loads_tokenizer_from_pretrained_when_path_exists(
        self, workspace_factory, tmp_path
    ):
        workspace = workspace_factory()
        workspace.policy = MagicMock()
        workspace.lightning_policy = MagicMock()

        tokenizer_dir = workspace.output_dir / "tokenizer"
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        mock_tokenizer = MagicMock()
        with (
            patch(
                "versatil.workspace.torch.load",
                return_value={"state_dict": {}},
            ),
            patch(
                "versatil.workspace.Tokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ) as mock_from_pretrained,
        ):
            workspace.load_checkpoint("/fake/path.ckpt")

        mock_from_pretrained.assert_called_once()
        workspace.policy.set_tokenizer.assert_called_once_with(mock_tokenizer)
        assert workspace.tokenizer == mock_tokenizer

    def test_sets_tokenizer_to_none_when_path_does_not_exist(self, workspace_factory):
        workspace = workspace_factory()
        workspace.policy = MagicMock()
        workspace.lightning_policy = MagicMock()

        with patch(
            "versatil.workspace.torch.load",
            return_value={"state_dict": {}},
        ):
            workspace.load_checkpoint("/fake/path.ckpt")

        assert workspace.tokenizer is None

    def test_loads_checkpoint_with_correct_map_location(self, workspace_factory):
        workspace = workspace_factory(experiment_kwargs={"device": "cpu"})
        workspace.policy = MagicMock()
        workspace.lightning_policy = MagicMock()

        with patch(
            "versatil.workspace.torch.load",
            return_value={"state_dict": {}},
        ) as mock_load:
            workspace.load_checkpoint("/fake/path.ckpt")

            mock_load.assert_called_once_with(
                "/fake/path.ckpt",
                map_location="cpu",
                weights_only=False,
            )


@pytest.mark.unit
class TestPredict:
    def test_raises_runtime_error_when_lightning_policy_is_none(
        self, workspace_factory
    ):
        workspace = workspace_factory()
        workspace.lightning_policy = None

        with pytest.raises(
            RuntimeError,
            match="Policy not initialized. Call run\\(\\) first.",
        ):
            workspace.predict({"obs": torch.zeros(2, 3)})

    def test_uses_ema_model_when_ema_enabled_and_callback_available(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={"use_ema": True},
            policy=policy,
        )
        workspace.policy = policy
        workspace.lightning_policy = MagicMock(spec=LightningPolicy)

        mock_ema_model = MagicMock()
        mock_ema_model.predict_action = MagicMock(
            return_value={"action": torch.ones(2, 7)}
        )

        ema_callback = MagicMock(spec=EMACallback)
        ema_callback.ema_model = mock_ema_model

        workspace.trainer = MagicMock()
        workspace.trainer.callbacks = [ema_callback]

        obs_dict = {"obs": torch.zeros(2, 3)}
        workspace.predict(obs_dict)

        mock_ema_model.eval.assert_called_once()
        mock_ema_model.predict_action.assert_called_once_with(obs_dict)

    def test_uses_original_policy_when_ema_disabled(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={"use_ema": False},
            policy=policy,
        )
        workspace.policy = policy
        workspace.lightning_policy = MagicMock(spec=LightningPolicy)
        workspace.trainer = None

        obs_dict = {"obs": torch.zeros(2, 3)}
        workspace.predict(obs_dict)

        policy.eval.assert_called_once()
        policy.predict_action.assert_called_once_with(obs_dict)

    def test_uses_original_policy_when_ema_model_is_none(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            training_kwargs={"use_ema": True},
            policy=policy,
        )
        workspace.policy = policy
        workspace.lightning_policy = MagicMock(spec=LightningPolicy)

        ema_callback = MagicMock(spec=EMACallback)
        ema_callback.ema_model = None

        workspace.trainer = MagicMock()
        workspace.trainer.callbacks = [ema_callback]

        obs_dict = {"obs": torch.zeros(2, 3)}
        workspace.predict(obs_dict)

        policy.eval.assert_called_once()
        policy.predict_action.assert_called_once_with(obs_dict)

    def test_runs_in_no_grad_context(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)
        workspace.policy = policy
        workspace.lightning_policy = MagicMock(spec=LightningPolicy)
        workspace.trainer = None

        obs_dict = {"obs": torch.zeros(2, 3)}

        with patch("versatil.workspace.torch.no_grad") as mock_no_grad:
            mock_context = MagicMock()
            mock_no_grad.return_value = mock_context
            workspace.predict(obs_dict)

            mock_no_grad.assert_called_once()


@pytest.mark.unit
class TestRun:
    def test_run_orchestrates_full_workflow(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)

        with (
            patch.object(workspace, "_create_logger", return_value=None) as mock_logger,
            patch.object(workspace, "_setup_data") as mock_data,
            patch.object(workspace, "_setup_policy") as mock_policy_setup,
            patch.object(workspace, "_setup_trainer") as mock_trainer_setup,
            patch.object(workspace, "_tune_hyperparameters") as mock_tune,
        ):
            mock_lightning_policy = MagicMock()
            workspace.lightning_policy = mock_lightning_policy
            workspace.train_loader = MagicMock()
            workspace.val_loader = MagicMock()

            mock_trainer = MagicMock()
            workspace.trainer = mock_trainer

            workspace.run()

            mock_logger.assert_called_once()
            mock_data.assert_called_once()
            mock_policy_setup.assert_called_once()
            mock_trainer_setup.assert_called_once()
            mock_tune.assert_called_once()
            mock_trainer.fit.assert_called_once_with(
                model=mock_lightning_policy,
                ckpt_path=None,
            )

    def test_run_passes_resume_checkpoint_when_path_exists(
        self, workspace_factory, mock_workspace_policy_factory, tmp_path
    ):
        checkpoint_path = tmp_path / "checkpoint.ckpt"
        checkpoint_path.touch()

        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            experiment_kwargs={"resume_from": str(checkpoint_path)},
            policy=policy,
        )

        with (
            patch.object(workspace, "_create_logger", return_value=None),
            patch.object(workspace, "_setup_data"),
            patch.object(workspace, "_setup_policy"),
            patch.object(workspace, "_setup_trainer"),
            patch.object(workspace, "_tune_hyperparameters"),
        ):
            workspace.lightning_policy = MagicMock()
            workspace.train_loader = MagicMock()
            workspace.val_loader = MagicMock()

            mock_trainer = MagicMock()
            workspace.trainer = mock_trainer

            workspace.run()

            mock_trainer.fit.assert_called_once_with(
                model=workspace.lightning_policy,
                ckpt_path=str(checkpoint_path),
            )

    def test_run_starts_from_scratch_when_resume_path_does_not_exist(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(
            experiment_kwargs={"resume_from": "/nonexistent/checkpoint.ckpt"},
            policy=policy,
        )

        with (
            patch.object(workspace, "_create_logger", return_value=None),
            patch.object(workspace, "_setup_data"),
            patch.object(workspace, "_setup_policy"),
            patch.object(workspace, "_setup_trainer"),
            patch.object(workspace, "_tune_hyperparameters"),
        ):
            workspace.lightning_policy = MagicMock()
            workspace.train_loader = MagicMock()
            workspace.val_loader = MagicMock()

            mock_trainer = MagicMock()
            workspace.trainer = mock_trainer

            workspace.run()

            mock_trainer.fit.assert_called_once_with(
                model=workspace.lightning_policy,
                ckpt_path=None,
            )

    def test_run_assigns_dataloaders_to_lightning_policy(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)

        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()

        with (
            patch.object(workspace, "_create_logger", return_value=None),
            patch.object(workspace, "_setup_data"),
            patch.object(workspace, "_setup_policy"),
            patch.object(workspace, "_setup_trainer"),
            patch.object(workspace, "_tune_hyperparameters"),
        ):
            mock_lightning = MagicMock()
            workspace.lightning_policy = mock_lightning
            workspace.train_loader = mock_train_loader
            workspace.val_loader = mock_val_loader
            workspace.trainer = MagicMock()

            workspace.run()

            assert mock_lightning._train_dataloader == mock_train_loader
            assert mock_lightning._val_dataloader == mock_val_loader


@pytest.mark.unit
class TestTuneHyperparameters:
    def test_skips_tuning_when_tune_lr_disabled(self, workspace_factory):
        workspace = workspace_factory(training_kwargs={"tune_lr": False})

        with patch("versatil.workspace.Tuner") as mock_tuner_cls:
            workspace._tune_hyperparameters()

            mock_tuner_cls.assert_not_called()

    def test_skips_tuning_when_distributed(self, workspace_factory):
        workspace = workspace_factory(
            experiment_kwargs={"distributed": True},
            training_kwargs={"tune_lr": True},
        )
        workspace.trainer = MagicMock()
        workspace.lightning_policy = MagicMock()

        with patch("versatil.workspace.Tuner") as mock_tuner_cls:
            workspace._tune_hyperparameters()

            mock_tuner_cls.assert_not_called()

    def test_removes_swa_callbacks_during_tuning(self, workspace_factory):
        workspace = workspace_factory(training_kwargs={"tune_lr": True})
        workspace.lightning_policy = MagicMock()

        swa_callback = MagicMock(spec=StochasticWeightAveraging)
        other_callback = MagicMock()
        workspace.trainer = MagicMock()
        workspace.trainer.callbacks = [swa_callback, other_callback]

        mock_tuner = MagicMock()
        mock_lr_result = MagicMock()
        mock_lr_result.suggestion.return_value = 0.001
        mock_tuner.lr_find.return_value = mock_lr_result

        with patch("versatil.workspace.Tuner", return_value=mock_tuner):
            workspace._tune_hyperparameters()

        # After tuning, original callbacks (including SWA) should be restored
        assert swa_callback in workspace.trainer.callbacks
        assert other_callback in workspace.trainer.callbacks

    def test_updates_config_with_suggested_lr(self, workspace_factory):
        workspace = workspace_factory(training_kwargs={"tune_lr": True})
        workspace.lightning_policy = MagicMock()
        workspace.trainer = MagicMock()
        workspace.trainer.callbacks = []

        mock_tuner = MagicMock()
        mock_lr_result = MagicMock()
        mock_lr_result.suggestion.return_value = 0.0042
        mock_tuner.lr_find.return_value = mock_lr_result

        with (
            patch("versatil.workspace.Tuner", return_value=mock_tuner),
            patch.object(workspace, "save_config"),
        ):
            workspace._tune_hyperparameters()

        assert workspace.config.training.optimizer.lr == 0.0042

    def test_saves_config_after_tuning(self, workspace_factory):
        workspace = workspace_factory(training_kwargs={"tune_lr": True})
        workspace.lightning_policy = MagicMock()
        workspace.trainer = MagicMock()
        workspace.trainer.callbacks = []

        mock_tuner = MagicMock()
        mock_lr_result = MagicMock()
        mock_lr_result.suggestion.return_value = 0.001
        mock_tuner.lr_find.return_value = mock_lr_result

        with (
            patch("versatil.workspace.Tuner", return_value=mock_tuner),
            patch.object(workspace, "save_config") as mock_save,
        ):
            workspace._tune_hyperparameters()

            mock_save.assert_called_once()


@pytest.mark.unit
class TestSetupData:
    @patch("versatil.workspace.get_dataloaders")
    @patch("versatil.workspace.plt")
    def test_stores_normalizer_and_tokenizer(
        self, mock_plt, mock_get_dataloaders, workspace_factory
    ):
        workspace = workspace_factory()
        workspace.logger = None

        mock_train_loader = MagicMock()
        mock_train_loader.dataset.__len__ = MagicMock(return_value=100)
        mock_train_loader.dataset.action_processor.plot_action_magnitude_distribution.return_value = None
        mock_train_loader.dataset.action_processor.denoising_thresholds = {}

        mock_val_loader = MagicMock()
        mock_val_loader.dataset.__len__ = MagicMock(return_value=20)

        mock_normalizer = MagicMock(spec=LinearNormalizer)

        mock_get_dataloaders.return_value = (
            mock_train_loader,
            mock_val_loader,
            mock_normalizer,
            None,  # tokenizer
            None,  # gripper_class_weights
        )

        workspace._setup_data()

        assert workspace.train_loader == mock_train_loader
        assert workspace.val_loader == mock_val_loader
        assert workspace.normalizer == mock_normalizer
        assert workspace.tokenizer is None

    @patch("versatil.workspace.get_dataloaders")
    @patch("versatil.workspace.plt")
    def test_stores_gripper_class_weights_on_device(
        self, mock_plt, mock_get_dataloaders, workspace_factory
    ):
        workspace = workspace_factory(experiment_kwargs={"device": "cpu"})
        workspace.logger = None

        mock_train_loader = MagicMock()
        mock_train_loader.dataset.__len__ = MagicMock(return_value=100)
        mock_train_loader.dataset.action_processor.plot_action_magnitude_distribution.return_value = None
        mock_train_loader.dataset.action_processor.denoising_thresholds = {}

        mock_get_dataloaders.return_value = (
            mock_train_loader,
            None,  # val_loader
            MagicMock(spec=LinearNormalizer),
            None,  # tokenizer
            2.5,  # gripper_class_weights
        )

        workspace._setup_data()

        assert workspace.gripper_class_weights is not None
        assert workspace.gripper_class_weights.device.type == "cpu"

    @patch("versatil.workspace.get_dataloaders")
    @patch("versatil.workspace.plt")
    def test_saves_tokenizer_when_provided(
        self, mock_plt, mock_get_dataloaders, workspace_factory
    ):
        workspace = workspace_factory()
        workspace.logger = None

        mock_tokenizer = MagicMock()
        mock_train_loader = MagicMock()
        mock_train_loader.dataset.__len__ = MagicMock(return_value=100)
        mock_train_loader.dataset.action_processor.plot_action_magnitude_distribution.return_value = None
        mock_train_loader.dataset.action_processor.denoising_thresholds = {}

        mock_get_dataloaders.return_value = (
            mock_train_loader,
            None,
            MagicMock(spec=LinearNormalizer),
            mock_tokenizer,
            None,
        )

        workspace._setup_data()

        expected_tokenizer_path = workspace.output_dir / "tokenizer"
        mock_tokenizer.save_pretrained.assert_called_once_with(expected_tokenizer_path)

    @patch("versatil.workspace.get_dataloaders")
    @patch("versatil.workspace.wandb")
    @patch("versatil.workspace.plt")
    def test_logs_action_distribution_plot_to_wandb_when_logger_present(
        self, mock_plt, mock_wandb, mock_get_dataloaders, workspace_factory
    ):
        workspace = workspace_factory()
        workspace.logger = MagicMock()

        mock_fig = MagicMock()
        mock_train_loader = MagicMock()
        mock_train_loader.dataset.__len__ = MagicMock(return_value=100)
        mock_train_loader.dataset.action_processor.plot_action_magnitude_distribution.return_value = mock_fig
        mock_train_loader.dataset.action_processor.denoising_thresholds = {}

        mock_get_dataloaders.return_value = (
            mock_train_loader,
            None,
            MagicMock(spec=LinearNormalizer),
            None,
            None,
        )

        workspace._setup_data()

        mock_fig.savefig.assert_called_once()
        workspace.logger.experiment.log.assert_called_once()
        mock_plt.close.assert_called_once_with(mock_fig)

    @patch("versatil.workspace.get_dataloaders")
    @patch("versatil.workspace.plt")
    def test_stores_denoising_thresholds(
        self, mock_plt, mock_get_dataloaders, workspace_factory
    ):
        workspace = workspace_factory()
        workspace.logger = None

        expected_thresholds = {"position": 0.01, "orientation": 0.005}
        mock_train_loader = MagicMock()
        mock_train_loader.dataset.__len__ = MagicMock(return_value=100)
        mock_train_loader.dataset.action_processor.plot_action_magnitude_distribution.return_value = None
        mock_train_loader.dataset.action_processor.denoising_thresholds = (
            expected_thresholds
        )

        mock_get_dataloaders.return_value = (
            mock_train_loader,
            None,
            MagicMock(spec=LinearNormalizer),
            None,
            None,
        )

        workspace._setup_data()

        assert workspace.denoising_thresholds == expected_thresholds


@pytest.mark.unit
class TestWorkspaceStateGuards:
    def test_run_raises_when_trainer_is_none(
        self, workspace_factory, mock_workspace_policy_factory
    ):
        policy = mock_workspace_policy_factory()
        workspace = workspace_factory(policy=policy)

        with (
            patch.object(workspace, "_create_logger", return_value=None),
            patch.object(workspace, "_setup_data"),
            patch.object(workspace, "_setup_policy"),
            patch.object(workspace, "_setup_trainer"),
            patch.object(workspace, "_tune_hyperparameters"),
        ):
            workspace.lightning_policy = MagicMock()
            workspace.train_loader = MagicMock()
            workspace.val_loader = MagicMock()
            workspace.trainer = None

            with pytest.raises(
                RuntimeError,
                match="Trainer should be initialized before training",
            ):
                workspace.run()

    def test_predict_raises_when_policy_is_none(self, workspace_factory):
        workspace = workspace_factory()
        workspace.lightning_policy = MagicMock()
        workspace.policy = None
        workspace.trainer = None

        with pytest.raises(
            RuntimeError,
            match="Policy must be initialized",
        ):
            workspace.predict({"obs": torch.zeros(2, 3)})

    def test_tune_hyperparameters_raises_when_trainer_is_none(self, workspace_factory):
        workspace = workspace_factory(training_kwargs={"tune_lr": True})
        workspace.trainer = None
        workspace.lightning_policy = MagicMock()

        with pytest.raises(
            RuntimeError,
            match="Trainer must be initialized before tuning",
        ):
            workspace._tune_hyperparameters()

    def test_tune_hyperparameters_raises_when_lightning_policy_is_none(
        self, workspace_factory
    ):
        workspace = workspace_factory(training_kwargs={"tune_lr": True})
        workspace.trainer = MagicMock()
        workspace.lightning_policy = None

        with pytest.raises(
            RuntimeError,
            match="Lightning policy must be initialized before tuning",
        ):
            workspace._tune_hyperparameters()
