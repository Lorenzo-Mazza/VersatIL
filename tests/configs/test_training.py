"""Tests for versatil.configs.training module."""

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from versatil.configs.training import (
    AdamConfig,
    AdamWConfig,
    OptimizerConfig,
    ParameterGroupConfig,
    SGDConfig,
    TrainingConfig,
    TrainingStageConfig,
)
from versatil.training.constants import OPTIMIZER_UNMATCHED_GROUPS_NAME
from versatil.training.stage import TrainingStage


@pytest.mark.unit
class TestParameterGroupConfig:
    @pytest.mark.parametrize("name", ["backbone", "decoder"])
    @pytest.mark.parametrize("learning_rate", [1e-5, 1e-3])
    @pytest.mark.parametrize("weight_decay", [None, 1e-4])
    @pytest.mark.parametrize("params_pattern", [None, r"backbone\..*"])
    def test_stores_configuration(
        self,
        name: str,
        learning_rate: float,
        weight_decay: float | None,
        params_pattern: str | None,
    ) -> None:
        config = ParameterGroupConfig(
            name=name,
            lr=learning_rate,
            weight_decay=weight_decay,
            params_pattern=params_pattern,
        )
        assert config.name == name
        assert config.lr == learning_rate
        assert config.weight_decay == weight_decay
        assert config.params_pattern == params_pattern

    def test_defaults(self) -> None:
        config = ParameterGroupConfig(name="group", lr=1e-4)
        assert config.weight_decay is None
        assert config.params_pattern is None

    def test_rejects_reserved_default_name(self) -> None:
        with pytest.raises(
            ValueError,
            match=(
                f"'{OPTIMIZER_UNMATCHED_GROUPS_NAME}' is reserved for unmatched "
                "parameters."
            ),
        ):
            ParameterGroupConfig(name=OPTIMIZER_UNMATCHED_GROUPS_NAME, lr=1e-4)


@pytest.mark.unit
class TestOptimizerConfig:
    @pytest.mark.parametrize("learning_rate", [1e-4, 1e-3])
    @pytest.mark.parametrize("param_group_count", [0, 2])
    def test_stores_configuration(
        self,
        learning_rate: float,
        param_group_count: int,
    ) -> None:
        param_groups = [
            ParameterGroupConfig(name=f"group{index}", lr=1e-5)
            for index in range(param_group_count)
        ]
        config = OptimizerConfig(
            target_class="torch.optim.AdamW",
            lr=learning_rate,
            param_groups=param_groups,
        )
        assert config.target_class == "torch.optim.AdamW"
        assert config.lr == learning_rate
        assert config.param_groups == param_groups

    def test_defaults(self) -> None:
        config = OptimizerConfig(target_class="torch.optim.AdamW")
        assert config.lr == 1e-4
        assert config.param_groups == []

    def test_duplicate_param_group_names_raise(self) -> None:
        groups = [
            ParameterGroupConfig(name="decoder", lr=1e-5),
            ParameterGroupConfig(name="decoder", lr=1e-4),
        ]
        with pytest.raises(
            ValueError,
            match=r"Optimizer parameter group names must be unique: \['decoder'\]\.",
        ):
            OptimizerConfig(target_class="torch.optim.AdamW", param_groups=groups)

    def test_rejects_reserved_param_group_name(self) -> None:
        with pytest.raises(
            ValueError,
            match=(
                f"'{OPTIMIZER_UNMATCHED_GROUPS_NAME}' is reserved for unmatched "
                "parameters."
            ),
        ):
            ParameterGroupConfig(name=OPTIMIZER_UNMATCHED_GROUPS_NAME, lr=1e-4)


@pytest.mark.unit
class TestAdamWConfig:
    @pytest.mark.parametrize("learning_rate", [1e-4, 1e-3])
    @pytest.mark.parametrize("weight_decay", [0.0, 1e-4])
    @pytest.mark.parametrize("betas", [(0.9, 0.999), (0.95, 0.999)])
    @pytest.mark.parametrize("eps", [1e-8, 1e-6])
    @pytest.mark.parametrize("amsgrad", [False, True])
    def test_stores_configuration(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        amsgrad: bool,
    ) -> None:
        config = AdamWConfig(
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            amsgrad=amsgrad,
        )
        assert config.target_class == "torch.optim.AdamW"
        assert config.lr == learning_rate
        assert config.weight_decay == weight_decay
        assert config.betas == betas
        assert config.eps == eps
        assert config.amsgrad == amsgrad

    def test_defaults(self) -> None:
        config = AdamWConfig()
        assert config.target_class == "torch.optim.AdamW"
        assert config.lr == 1e-4
        assert config.weight_decay == 1e-4
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
        assert config.amsgrad is False


@pytest.mark.unit
class TestAdamConfig:
    @pytest.mark.parametrize("learning_rate", [1e-4, 5e-4])
    @pytest.mark.parametrize("weight_decay", [0.0, 1e-5])
    @pytest.mark.parametrize("betas", [(0.9, 0.999), (0.95, 0.999)])
    @pytest.mark.parametrize("amsgrad", [False, True])
    def test_stores_configuration(
        self,
        learning_rate: float,
        weight_decay: float,
        betas: tuple[float, float],
        amsgrad: bool,
    ) -> None:
        config = AdamConfig(
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            amsgrad=amsgrad,
        )
        assert config.target_class == "torch.optim.Adam"
        assert config.lr == learning_rate
        assert config.weight_decay == weight_decay
        assert config.betas == betas
        assert config.amsgrad == amsgrad

    def test_defaults(self) -> None:
        config = AdamConfig()
        assert config.target_class == "torch.optim.Adam"
        assert config.lr == 1e-4
        assert config.weight_decay == 0.0
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8
        assert config.amsgrad is False


@pytest.mark.unit
class TestSGDConfig:
    @pytest.mark.parametrize("learning_rate", [1e-2, 1e-1])
    @pytest.mark.parametrize("momentum", [0.0, 0.9])
    @pytest.mark.parametrize("weight_decay", [0.0, 1e-4])
    @pytest.mark.parametrize("dampening", [0.0, 0.1])
    @pytest.mark.parametrize("nesterov", [False, True])
    def test_stores_configuration(
        self,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        dampening: float,
        nesterov: bool,
    ) -> None:
        config = SGDConfig(
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
        assert config.target_class == "torch.optim.SGD"
        assert config.lr == learning_rate
        assert config.momentum == momentum
        assert config.weight_decay == weight_decay
        assert config.dampening == dampening
        assert config.nesterov == nesterov

    def test_defaults(self) -> None:
        config = SGDConfig()
        assert config.target_class == "torch.optim.SGD"
        assert config.lr == 1e-2
        assert config.momentum == 0.0
        assert config.weight_decay == 0.0
        assert config.dampening == 0.0
        assert config.nesterov is False


@pytest.mark.unit
class TestTrainingStageConfig:
    @pytest.mark.parametrize("name", ["vae", "prior"])
    @pytest.mark.parametrize("start_epoch", [0, 200])
    @pytest.mark.parametrize("end_epoch", [None, 1000])
    @pytest.mark.parametrize("eval_frozen_modules", [True, False])
    def test_stores_configuration(
        self,
        name: str,
        start_epoch: int,
        end_epoch: int | None,
        eval_frozen_modules: bool,
    ) -> None:
        config = TrainingStageConfig(
            name=name,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            trainable_groups=["prior"],
            frozen_groups=["decoder"],
            group_lrs={"prior": 2e-4},
            group_weight_decays={"prior": 2e-2},
            loss_weights={"denoising_prior": {"weight": 0.03}},
            eval_frozen_modules=eval_frozen_modules,
        )
        assert config.name == name
        assert config.start_epoch == start_epoch
        assert config.end_epoch == end_epoch
        assert config.trainable_groups == ["prior"]
        assert config.frozen_groups == ["decoder"]
        assert config.group_lrs == {"prior": 2e-4}
        assert config.group_weight_decays == {"prior": 2e-2}
        assert config.loss_weights == {"denoising_prior": {"weight": 0.03}}
        assert config.eval_frozen_modules == eval_frozen_modules

    def test_defaults(self) -> None:
        config = TrainingStageConfig(name="vae", start_epoch=0)
        assert config.end_epoch is None
        assert config.trainable_groups == []
        assert config.frozen_groups == []
        assert config.group_lrs == {}
        assert config.group_weight_decays == {}
        assert config.loss_weights == {}
        assert config.eval_frozen_modules is True

    def test_instantiate_builds_training_stage(self) -> None:
        config = TrainingStageConfig(
            name="prior",
            start_epoch=200,
            end_epoch=1000,
            trainable_groups=["prior"],
            frozen_groups=["decoder"],
            group_lrs={"prior": 2e-4},
            group_weight_decays={"prior": 2e-2},
            loss_weights={"denoising_prior": {"weight": 0.03}},
            eval_frozen_modules=False,
        )
        stage = instantiate(OmegaConf.structured(config))
        assert isinstance(stage, TrainingStage)
        assert stage.name == "prior"
        assert stage.start_epoch == 200
        assert stage.end_epoch == 1000
        assert stage.trainable_groups == ["prior"]
        assert stage.frozen_groups == ["decoder"]
        assert stage.group_lrs == {"prior": 2e-4}
        assert stage.group_weight_decays == {"prior": 2e-2}
        assert stage.loss_weights == {"denoising_prior": {"weight": 0.03}}
        assert stage.eval_frozen_modules is False


@pytest.mark.unit
class TestTrainingConfig:
    @pytest.mark.parametrize("num_epochs", [50, 200])
    @pytest.mark.parametrize("use_ema", [True, False])
    @pytest.mark.parametrize("clip_gradient_norm", [True, False])
    @pytest.mark.parametrize("clip_max_norm", [0.1, 5.0])
    @pytest.mark.parametrize(
        "lr_schedule, lr_scheduler_kwargs",
        [
            (None, {}),
            ("cosine", {}),
            ("cosine_with_min_lr", {"min_lr": 2.5e-6}),
        ],
    )
    @pytest.mark.parametrize("swa_lrs", [None, 1e-5])
    def test_stores_configuration(
        self,
        num_epochs: int,
        use_ema: bool,
        clip_gradient_norm: bool,
        clip_max_norm: float,
        lr_schedule: str | None,
        lr_scheduler_kwargs: dict[str, float],
        swa_lrs: float | None,
    ) -> None:
        stages = [
            TrainingStageConfig(name="vae", start_epoch=0),
            TrainingStageConfig(name="prior", start_epoch=200),
        ]
        config = TrainingConfig(
            num_epochs=num_epochs,
            use_ema=use_ema,
            clip_gradient_norm=clip_gradient_norm,
            clip_max_norm=clip_max_norm,
            lr_schedule=lr_schedule,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            swa_lrs=swa_lrs,
            stages=stages,
        )
        assert config.num_epochs == num_epochs
        assert config.use_ema == use_ema
        assert config.clip_gradient_norm == clip_gradient_norm
        assert config.clip_max_norm == clip_max_norm
        assert config.lr_schedule == lr_schedule
        assert config.lr_scheduler_kwargs == lr_scheduler_kwargs
        assert config.swa_lrs == swa_lrs
        assert config.stages == stages

    def test_defaults(self) -> None:
        config = TrainingConfig()
        assert config.num_epochs == 100
        assert config.gradient_accumulate_every == 1
        assert isinstance(config.optimizer, AdamWConfig)
        assert config.clip_gradient_norm is False
        assert config.clip_max_norm == 0.1
        assert config.lr_schedule is None
        assert config.lr_warmup_steps == 5000
        assert config.lr_scheduler_kwargs == {}
        assert config.use_ema is True
        assert config.ema_power == 0.75
        assert config.swa_lrs is None
        assert config.swa_epoch_start == 0.5
        assert config.swa_annealing_epochs == 10
        assert config.compile is False
        assert config.tune_lr is False
        assert config.early_stopping_patience == 10
        assert config.reduce_lr_on_plateau is False
        assert config.reduce_lr_patience == 10
        assert config.reduce_lr_cooldown == 10
        assert config.stages == []

    def test_stages_do_not_support_reduce_lr_on_plateau(self) -> None:
        stages = [TrainingStageConfig(name="stage", start_epoch=0)]
        with pytest.raises(
            ValueError,
            match="training.stages does not support reduce_lr_on_plateau.",
        ):
            TrainingConfig(stages=stages, reduce_lr_on_plateau=True)
