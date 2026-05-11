"""Tests for versatil.training.callbacks.training_stage module."""

import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, TensorDataset

from versatil.configs.training import (
    AdamConfig,
    ParameterGroupConfig,
    TrainingConfig,
)
from versatil.metrics.base import LossOutput, ScalarWeightedLoss
from versatil.metrics.components import PriorDenoisingLoss, RegressionLoss
from versatil.metrics.composite import CompositeLoss
from versatil.models.policy import Policy
from versatil.training.callbacks.training_stage import TrainingStageCallback
from versatil.training.constants import OPTIMIZER_UNMATCHED_GROUPS_NAME
from versatil.training.lightning_policy import LightningPolicy
from versatil.training.stage import TrainingStage

torch.serialization.add_safe_globals([TrainingConfig, AdamConfig, ParameterGroupConfig])


def _stage(**fields: Any) -> TrainingStage:
    """Build a ``TrainingStage`` from keyword fields for use in callback tests."""
    return TrainingStage(**fields)


@pytest.fixture
def optimizer_with_groups_factory() -> Callable[..., torch.optim.Optimizer]:
    def factory(group_count: int = 2) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            [
                {
                    "name": f"g{index}",
                    "params": [torch.nn.Parameter(torch.zeros(1))],
                    "lr": 1e-3,
                }
                for index in range(group_count)
            ]
        )

    return factory


@pytest.fixture
def trainer_with_scheduler_configs_factory() -> Callable[..., MagicMock]:
    def factory(scheduler_configs: Any) -> MagicMock:
        trainer = MagicMock()
        trainer.lr_scheduler_configs = scheduler_configs
        return trainer

    return factory


@pytest.fixture
def staged_optimizer_factory() -> Callable[..., torch.optim.Optimizer]:
    """Build grouped SGD optimizer keyed to the staged policy module tree."""

    def factory(policy: Policy) -> torch.optim.Optimizer:
        grouped_parameters: dict[str, list[torch.nn.Parameter]] = {
            OPTIMIZER_UNMATCHED_GROUPS_NAME: [],
            "posterior": [],
            "prior": [],
            "decoder": [],
        }
        for name, parameter in policy.named_parameters():
            if name.startswith("algorithm.posterior_encoder."):
                grouped_parameters["posterior"].append(parameter)
            elif name.startswith("algorithm.prior"):
                grouped_parameters["prior"].append(parameter)
            elif name.startswith("decoder."):
                grouped_parameters["decoder"].append(parameter)
            else:
                grouped_parameters[OPTIMIZER_UNMATCHED_GROUPS_NAME].append(parameter)

        return torch.optim.SGD(
            [
                {
                    "name": OPTIMIZER_UNMATCHED_GROUPS_NAME,
                    "params": grouped_parameters[OPTIMIZER_UNMATCHED_GROUPS_NAME],
                    "lr": 1e-3,
                    "weight_decay": 1e-2,
                },
                {
                    "name": "posterior",
                    "params": grouped_parameters["posterior"],
                    "lr": 2e-3,
                    "weight_decay": 2e-2,
                },
                {
                    "name": "prior",
                    "params": grouped_parameters["prior"],
                    "lr": 3e-3,
                    "weight_decay": 3e-2,
                },
                {
                    "name": "decoder",
                    "params": grouped_parameters["decoder"],
                    "lr": 4e-3,
                    "weight_decay": 4e-2,
                },
            ]
        )

    return factory


@pytest.fixture
def staged_trainer_factory() -> Callable[..., MagicMock]:
    """Build a trainer mock with optimizer and optional scheduler state."""

    def factory(
        optimizer: torch.optim.Optimizer,
        current_epoch: int = 0,
        scheduler: LRScheduler | None = None,
    ) -> MagicMock:
        trainer = MagicMock()
        trainer.current_epoch = current_epoch
        trainer.optimizers = [optimizer]
        trainer.lr_scheduler_configs = []
        if scheduler is not None:
            scheduler_config = MagicMock()
            scheduler_config.scheduler = scheduler
            trainer.lr_scheduler_configs = [scheduler_config]
        return trainer

    return factory


@pytest.mark.unit
class TestTrainingStageCallback:
    def test_requires_non_empty_stages(self) -> None:
        with pytest.raises(ValueError, match="requires a non-empty stage list"):
            TrainingStageCallback(stages=[])

    def test_applies_stage_on_train_start(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        policy.train()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="vae",
                    start_epoch=0,
                    frozen_groups=["prior"],
                )
            ]
        )

        callback.on_train_start(trainer, pl_module)

        for name, parameter in policy.named_parameters():
            assert parameter.requires_grad is not name.startswith("algorithm.prior")
        assert policy.algorithm.prior.training is False
        assert policy.algorithm.posterior_encoder.training is True
        assert policy.decoder.training is True
        assert policy.encoding_pipeline.training is True

    def test_base_config_applies_before_and_after_sparse_stage(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        policy.train()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer, current_epoch=0)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="prior_window",
                    start_epoch=1,
                    end_epoch=2,
                    frozen_groups=["prior"],
                    group_lrs={"prior": 5e-2},
                )
            ]
        )
        prior_group = next(
            group for group in optimizer.param_groups if group["name"] == "prior"
        )

        callback.on_train_start(trainer, pl_module)
        assert all(parameter.requires_grad for parameter in policy.parameters())
        assert prior_group["lr"] == pytest.approx(3e-3)

        trainer.current_epoch = 1
        callback.on_train_epoch_start(trainer, pl_module)
        assert all(
            not parameter.requires_grad
            for name, parameter in policy.named_parameters()
            if name.startswith("algorithm.prior")
        )
        assert prior_group["lr"] == pytest.approx(5e-2)

        trainer.current_epoch = 2
        callback.on_train_epoch_start(trainer, pl_module)
        assert all(parameter.requires_grad for parameter in policy.parameters())
        assert policy.algorithm.prior.training is True
        assert prior_group["lr"] == pytest.approx(3e-3)

    def test_conflicting_groups_in_trainable_and_frozen_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match=(
                "Training stage 'conflict' lists groups in both trainable_groups "
                r"and frozen_groups: \['decoder'\]"
            ),
        ):
            TrainingStageCallback(
                stages=[
                    _stage(
                        name="conflict",
                        start_epoch=0,
                        trainable_groups=["prior", "decoder"],
                        frozen_groups=["decoder"],
                    )
                ]
            )

    def test_base_frozen_param_stays_frozen_when_group_not_mentioned(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        for parameter in policy.encoding_pipeline.parameters():
            parameter.requires_grad_(False)
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="freeze_prior",
                    start_epoch=0,
                    frozen_groups=["prior"],
                )
            ]
        )

        callback.on_train_start(trainer, pl_module)

        for parameter in policy.encoding_pipeline.parameters():
            assert parameter.requires_grad is False
        for parameter in policy.algorithm.prior.parameters():
            assert parameter.requires_grad is False
        for parameter in policy.algorithm.posterior_encoder.parameters():
            assert parameter.requires_grad is True
        for parameter in policy.decoder.parameters():
            assert parameter.requires_grad is True

    def test_base_frozen_param_thaws_when_group_in_trainable_groups(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        for parameter in policy.encoding_pipeline.parameters():
            parameter.requires_grad_(False)
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="thaw_encoder",
                    start_epoch=0,
                    trainable_groups=[OPTIMIZER_UNMATCHED_GROUPS_NAME],
                )
            ]
        )

        callback.on_train_start(trainer, pl_module)

        for parameter in policy.encoding_pipeline.parameters():
            assert parameter.requires_grad is True

    def test_base_frozen_param_stays_frozen_when_group_in_frozen_groups(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        for parameter in policy.encoding_pipeline.parameters():
            parameter.requires_grad_(False)
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="also_freeze_encoder",
                    start_epoch=0,
                    frozen_groups=[OPTIMIZER_UNMATCHED_GROUPS_NAME],
                )
            ]
        )

        callback.on_train_start(trainer, pl_module)

        for parameter in policy.encoding_pipeline.parameters():
            assert parameter.requires_grad is False

    def test_base_trainable_param_stays_trainable_when_group_not_mentioned(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="freeze_prior_only",
                    start_epoch=0,
                    frozen_groups=["prior"],
                )
            ]
        )

        callback.on_train_start(trainer, pl_module)

        for parameter in policy.algorithm.posterior_encoder.parameters():
            assert parameter.requires_grad is True
        for parameter in policy.decoder.parameters():
            assert parameter.requires_grad is True

    def test_base_trainable_param_stays_trainable_when_group_in_trainable_groups(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="explicit_trainable_prior",
                    start_epoch=0,
                    trainable_groups=["prior"],
                )
            ]
        )

        callback.on_train_start(trainer, pl_module)

        for parameter in policy.algorithm.prior.parameters():
            assert parameter.requires_grad is True

    def test_base_trainable_param_is_frozen_when_group_in_frozen_groups(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="freeze_decoder",
                    start_epoch=0,
                    frozen_groups=["decoder"],
                )
            ]
        )

        callback.on_train_start(trainer, pl_module)

        for parameter in policy.decoder.parameters():
            assert parameter.requires_grad is False

    def test_frozen_batch_norm_enters_eval_mode(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        policy.algorithm.prior_batch_norm = torch.nn.BatchNorm1d(2)
        policy.train()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="vae",
                    start_epoch=0,
                    frozen_groups=["prior"],
                )
            ]
        )

        callback.on_train_start(trainer, pl_module)

        assert policy.algorithm.prior_batch_norm.training is False

    def test_stage_snapshot_resets_lr_weight_decay_and_loss_weight(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        denoising_loss = PriorDenoisingLoss(weight=0.03)
        policy.loss_module = CompositeLoss({"denoising_prior": denoising_loss})
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="prior_high",
                    start_epoch=0,
                    group_lrs={"prior": 5e-2},
                    group_weight_decays={"prior": 5e-1},
                    loss_weights={"denoising_prior": {"weight": 0.0}},
                ),
                _stage(name="base", start_epoch=2),
            ]
        )

        callback.on_train_start(trainer, pl_module)
        prior_group = next(
            group for group in optimizer.param_groups if group["name"] == "prior"
        )
        assert prior_group["lr"] == pytest.approx(5e-2)
        assert prior_group["weight_decay"] == pytest.approx(5e-1)
        assert denoising_loss.weight == pytest.approx(0.0)

        trainer.current_epoch = 2
        callback.on_train_epoch_start(trainer, pl_module)

        assert prior_group["lr"] == pytest.approx(3e-3)
        assert prior_group["weight_decay"] == pytest.approx(3e-2)
        assert denoising_loss.weight == pytest.approx(0.03)

    def test_scheduler_base_learning_rates_update_with_staged_learning_rate(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 0.1
        )
        trainer = staged_trainer_factory(optimizer, scheduler=scheduler)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="prior",
                    start_epoch=0,
                    group_lrs={"prior": 5e-2},
                )
            ],
            learning_rate_schedule_active=True,
        )

        callback.on_train_start(trainer, pl_module)

        expected_base_rates = [1e-3, 2e-3, 5e-2, 4e-3]
        expected_actual_rates = [rate * 0.1 for rate in expected_base_rates]
        assert scheduler.base_lrs == pytest.approx(expected_base_rates)
        assert scheduler._last_lr == pytest.approx(expected_actual_rates)
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
            expected_actual_rates
        )

    def test_adam_momentum_survives_freeze_and_unfreeze_cycle(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        prior_parameter = next(policy.algorithm.prior.parameters())
        grouped = {
            OPTIMIZER_UNMATCHED_GROUPS_NAME: [],
            "posterior": [],
            "prior": [prior_parameter],
            "decoder": [],
        }
        for name, parameter in policy.named_parameters():
            if parameter is prior_parameter:
                continue
            if name.startswith("algorithm.posterior_encoder."):
                grouped["posterior"].append(parameter)
            elif name.startswith("decoder."):
                grouped["decoder"].append(parameter)
            else:
                grouped[OPTIMIZER_UNMATCHED_GROUPS_NAME].append(parameter)
        optimizer = torch.optim.Adam(
            [
                {
                    "name": OPTIMIZER_UNMATCHED_GROUPS_NAME,
                    "params": grouped[OPTIMIZER_UNMATCHED_GROUPS_NAME],
                    "lr": 1e-3,
                },
                {"name": "posterior", "params": grouped["posterior"], "lr": 1e-3},
                {"name": "prior", "params": grouped["prior"], "lr": 1e-3},
                {"name": "decoder", "params": grouped["decoder"], "lr": 1e-3},
            ]
        )
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="train_prior",
                    start_epoch=0,
                    trainable_groups=["prior"],
                ),
                _stage(
                    name="freeze_prior",
                    start_epoch=1,
                    frozen_groups=["prior"],
                ),
                _stage(
                    name="train_prior_again",
                    start_epoch=2,
                    trainable_groups=["prior"],
                ),
            ]
        )

        callback.on_train_start(trainer, pl_module)
        assert prior_parameter.requires_grad is True
        prior_parameter.grad = torch.ones_like(prior_parameter)
        optimizer.step()
        first_exp_avg = optimizer.state[prior_parameter]["exp_avg"].clone()

        trainer.current_epoch = 1
        callback.on_train_epoch_start(trainer, pl_module)
        assert prior_parameter.requires_grad is False

        trainer.current_epoch = 2
        callback.on_train_epoch_start(trainer, pl_module)
        assert prior_parameter.requires_grad is True

        torch.testing.assert_close(
            optimizer.state[prior_parameter]["exp_avg"], first_exp_avg
        )

    def test_resume_applies_correct_stage_on_train_start(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer, current_epoch=7)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="early",
                    start_epoch=0,
                    group_lrs={"prior": 1e-2},
                ),
                _stage(
                    name="late",
                    start_epoch=5,
                    group_lrs={"prior": 1e-4},
                ),
            ]
        )

        callback.on_train_start(trainer, pl_module)

        prior_group = next(
            group for group in optimizer.param_groups if group["name"] == "prior"
        )
        assert prior_group["lr"] == pytest.approx(1e-4)

    def test_base_regime_updates_scheduler_last_and_base_learning_rates(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 0.1
        )
        trainer = staged_trainer_factory(optimizer, scheduler=scheduler)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="ramp",
                    start_epoch=0,
                    end_epoch=2,
                    group_lrs={"prior": 5e-2},
                )
            ],
            learning_rate_schedule_active=True,
        )

        callback.on_train_start(trainer, pl_module)
        stage_base_rates = [1e-3, 2e-3, 5e-2, 4e-3]
        assert scheduler.base_lrs == pytest.approx(stage_base_rates)
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
            [rate * 0.1 for rate in stage_base_rates]
        )

        trainer.current_epoch = 5
        callback.on_train_epoch_start(trainer, pl_module)

        expected_base_rates = [1e-3, 2e-3, 3e-3, 4e-3]
        expected_actual_rates = [rate * 0.1 for rate in expected_base_rates]
        assert scheduler.base_lrs == pytest.approx(expected_base_rates)
        assert scheduler._last_lr == pytest.approx(expected_actual_rates)
        assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
            expected_actual_rates
        )

    def test_stage_override_then_restore_sub_loss_weight(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        denoising_loss = PriorDenoisingLoss(weight=0.8)
        policy.loss_module = CompositeLoss(
            loss_modules={"denoising_prior": denoising_loss}
        )
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="override",
                    start_epoch=0,
                    loss_weights={"denoising_prior": {"weight": 0.03}},
                ),
                _stage(name="rest", start_epoch=2),
            ]
        )

        callback.on_train_start(trainer, pl_module)

        assert denoising_loss.weight == pytest.approx(0.03)
        assert policy.loss_module.weights == {"denoising_prior": {"weight": 0.03}}

        trainer.current_epoch = 2
        callback.on_train_epoch_start(trainer, pl_module)

        assert denoising_loss.weight == pytest.approx(0.8)
        assert policy.loss_module.weights == {"denoising_prior": {"weight": 0.8}}

    def test_stage_override_with_wrong_leaf_shape_raises(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        policy.loss_module = CompositeLoss(
            {"regression_loss": RegressionLoss(action_keys=["action"])}
        )
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="stage",
                    start_epoch=0,
                    loss_weights={"regression_loss": 0.0},
                )
            ]
        )

        with pytest.raises(
            TypeError,
            match=(
                "Weight override for 'regression_loss' expects a dict subtree, "
                "got float."
            ),
        ):
            callback.on_train_start(trainer, pl_module)

    def test_missing_default_optimizer_group_raises(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = torch.optim.SGD(
            [
                {
                    "name": "prior",
                    "params": list(policy.algorithm.prior.parameters()),
                    "lr": 1e-3,
                }
            ]
        )
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="stage",
                    start_epoch=0,
                    trainable_groups=["prior"],
                )
            ]
        )

        with pytest.raises(
            ValueError,
            match=(
                "training.stages requires an optimizer parameter group named "
                f"'{OPTIMIZER_UNMATCHED_GROUPS_NAME}'."
            ),
        ):
            callback.on_train_start(trainer, pl_module)

    def test_unknown_staged_loss_name_raises(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        policy.loss_module = CompositeLoss(
            {"denoising_prior": PriorDenoisingLoss(weight=1.0)}
        )
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="typo",
                    start_epoch=0,
                    loss_weights={"denoising_proir": {"weight": 0.1}},
                )
            ]
        )

        with pytest.raises(KeyError, match="Unknown weight key 'denoising_proir'"):
            callback.on_train_start(trainer, pl_module)

    def test_active_lr_schedule_without_scheduler_raises(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="stage",
                    start_epoch=0,
                    group_lrs={"prior": 5e-2},
                )
            ],
            learning_rate_schedule_active=True,
        )

        with pytest.raises(
            ValueError,
            match=(
                "training.stages uses group_lrs with an active lr_schedule, "
                "but no Lightning scheduler was found."
            ),
        ):
            callback.on_train_start(trainer, pl_module)

    def test_eval_frozen_modules_false_keeps_policy_in_train_mode(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        policy.algorithm.prior_batch_norm = torch.nn.BatchNorm1d(2)
        policy.train()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="vae",
                    start_epoch=0,
                    frozen_groups=["prior"],
                    eval_frozen_modules=False,
                )
            ]
        )

        callback.on_train_start(trainer, pl_module)

        assert policy.algorithm.prior_batch_norm.training is True
        assert policy.algorithm.prior.training is True

    def test_same_stage_restores_modes_without_duplicate_logs(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="vae",
                    start_epoch=0,
                    frozen_groups=["decoder"],
                )
            ]
        )

        callback.on_train_start(trainer, pl_module)
        assert policy.decoder.training is False

        policy.decoder.train()
        pl_module.log.reset_mock()
        callback.on_train_epoch_start(trainer, pl_module)

        assert policy.decoder.training is False
        pl_module.log.assert_not_called()

    def test_requires_lightning_policy_type_raises(self) -> None:
        callback = TrainingStageCallback(
            stages=[_stage(name="stage", start_epoch=0, frozen_groups=["prior"])]
        )
        plain_module = pl.LightningModule()
        trainer = MagicMock()

        with pytest.raises(
            TypeError,
            match=(
                "TrainingStageCallback requires a LightningPolicy module, "
                "got LightningModule."
            ),
        ):
            callback.on_train_start(trainer, plain_module)

    def test_non_string_optimizer_group_name_raises(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = torch.optim.SGD(
            [
                {
                    "name": 42,
                    "params": list(policy.parameters()),
                    "lr": 1e-3,
                }
            ]
        )
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[_stage(name="stage", start_epoch=0, frozen_groups=["prior"])]
        )

        with pytest.raises(
            ValueError,
            match=(
                "training.stages requires every optimizer parameter group to "
                "have a string 'name'."
            ),
        ):
            callback.on_train_start(trainer, pl_module)

    def test_duplicate_optimizer_group_names_raise(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        prior_params = list(policy.algorithm.prior.parameters())
        posterior_params = list(policy.algorithm.posterior_encoder.parameters())
        optimizer = torch.optim.SGD(
            [
                {"name": "prior", "params": prior_params[:1], "lr": 1e-3},
                {"name": "prior", "params": posterior_params[:1], "lr": 2e-3},
                {
                    "name": OPTIMIZER_UNMATCHED_GROUPS_NAME,
                    "params": prior_params[1:] + posterior_params[1:],
                    "lr": 1e-3,
                },
            ]
        )
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[_stage(name="stage", start_epoch=0, frozen_groups=["prior"])]
        )

        with pytest.raises(
            ValueError,
            match=r"Optimizer parameter group names must be unique: \['prior'\]",
        ):
            callback.on_train_start(trainer, pl_module)

    def test_unregistered_policy_parameter_raises(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        optimizer = staged_optimizer_factory(policy)
        policy.stray_head = torch.nn.Linear(2, 2)
        pl_module = pl_module_with_policy_factory(policy=policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[_stage(name="stage", start_epoch=0, frozen_groups=["prior"])]
        )

        with pytest.raises(
            ValueError,
            match=(
                "training.stages found a policy parameter that is not present "
                "in any optimizer parameter group."
            ),
        ):
            callback.on_train_start(trainer, pl_module)

    def test_scheduler_with_mismatched_base_learning_rates_raises(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        scheduler = MagicMock(spec=LRScheduler)
        scheduler.base_lrs = [1e-3]
        trainer = staged_trainer_factory(optimizer, scheduler=scheduler)
        callback = TrainingStageCallback(
            stages=[
                _stage(
                    name="stage",
                    start_epoch=0,
                    group_lrs={"prior": 5e-2},
                )
            ],
            learning_rate_schedule_active=True,
        )

        with pytest.raises(
            ValueError,
            match=(
                "training.stages with group_lrs requires the scheduler to expose "
                "base_lrs with one entry per optimizer parameter group."
            ),
        ):
            callback.on_train_start(trainer, pl_module)

    def test_parameterless_module_is_skipped_by_mode_sync(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        policy.activation = torch.nn.ReLU()
        policy.train()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[_stage(name="vae", start_epoch=0, frozen_groups=["prior"])]
        )
        policy.activation.eval()

        callback.on_train_start(trainer, pl_module)

        assert policy.activation.training is False

    def test_mixed_trainability_module_keeps_current_mode(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        for parameter in policy.algorithm.posterior_encoder.parameters():
            parameter.requires_grad_(False)
        policy.train()
        policy.algorithm.eval()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[_stage(name="vae", start_epoch=0, frozen_groups=["prior"])]
        )

        callback.on_train_start(trainer, pl_module)

        assert policy.algorithm.training is False

    def test_ensure_initialized_is_idempotent(
        self,
        pl_module_with_policy_factory: Callable[..., MagicMock],
        staged_policy_factory: Callable[..., Policy],
        staged_optimizer_factory: Callable[..., torch.optim.Optimizer],
        staged_trainer_factory: Callable[..., MagicMock],
    ) -> None:
        policy = staged_policy_factory()
        pl_module = pl_module_with_policy_factory(policy=policy)
        optimizer = staged_optimizer_factory(policy)
        trainer = staged_trainer_factory(optimizer)
        callback = TrainingStageCallback(
            stages=[_stage(name="stage", start_epoch=0, frozen_groups=["prior"])]
        )

        callback.on_train_start(trainer, pl_module)
        cached_group_learning_rates = dict(callback._base_group_learning_rates)
        cached_trainability = dict(callback._base_parameter_trainability)
        for group in optimizer.param_groups:
            group["lr"] *= 10.0

        callback.on_train_epoch_start(trainer, pl_module)

        assert callback._base_group_learning_rates == cached_group_learning_rates
        assert callback._base_parameter_trainability == cached_trainability

    def test_training_stage_instances_preserved_without_normalization(self) -> None:
        stage_a = _stage(name="vae", start_epoch=0, frozen_groups=["prior"])
        stage_b = _stage(name="prior", start_epoch=1)
        callback = TrainingStageCallback(stages=[stage_a, stage_b])

        assert callback.stages[0] is stage_a
        assert callback.stages[1] is stage_b


class _ScalarMSELoss(ScalarWeightedLoss):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = float(weight)

    def get_required_keys(self) -> set[str]:
        return set()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        prediction = predictions["value"]
        target = targets["target"]
        loss_value = (prediction - target).pow(2).mean() * self.weight
        return LossOutput(
            total_loss=loss_value, component_losses={"mse": loss_value.detach()}
        )


class _StageStateRecorder(Callback):
    """Capture post-stage-application state at each ``on_train_epoch_start``."""

    def __init__(self) -> None:
        super().__init__()
        self.history: list[dict] = []

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        optimizer = trainer.optimizers[0]
        loss_module = pl_module.policy.loss_module
        self.history.append(
            {
                "epoch": int(trainer.current_epoch),
                "requires_grad": {
                    name: parameter.requires_grad
                    for name, parameter in pl_module.policy.named_parameters()
                },
                "learning_rates": {
                    group["name"]: float(group["lr"])
                    for group in optimizer.param_groups
                },
                "loss_weights": copy.deepcopy(loss_module.weights),
            }
        )


def _grouped_optimizer_config(
    base_learning_rate: float,
    prior_learning_rate: float | None = None,
) -> AdamConfig:
    """Build a real ``AdamConfig`` whose ``param_groups`` regexes bind to the
    real ``nn.Linear`` submodules attached to the staged integration policy."""
    return AdamConfig(
        lr=base_learning_rate,
        weight_decay=0.0,
        param_groups=[
            ParameterGroupConfig(
                name="posterior",
                lr=base_learning_rate,
                params_pattern=r"^posterior_encoder\.",
            ),
            ParameterGroupConfig(
                name="prior",
                lr=(
                    prior_learning_rate
                    if prior_learning_rate is not None
                    else base_learning_rate
                ),
                params_pattern=r"^prior\.",
            ),
            ParameterGroupConfig(
                name="decoder",
                lr=base_learning_rate,
                params_pattern=r"^decoder\.",
            ),
        ],
    )


def _staged_submodules() -> dict[str, torch.nn.Module]:
    return {
        "posterior_encoder": torch.nn.Linear(2, 2),
        "prior": torch.nn.Linear(2, 2),
        "decoder": torch.nn.Linear(2, 2),
    }


def _dummy_training_step(
    self: LightningPolicy,
    batch: tuple[torch.Tensor, torch.Tensor],
    batch_idx: int,
) -> torch.Tensor:
    """Gradient-bearing loss on every policy parameter, bypassing forward."""
    loss = sum(parameter.sum() for _, parameter in self.policy.named_parameters())
    self.log("train_loss", loss, on_step=False, on_epoch=True)
    return loss


def _dummy_dataloader() -> DataLoader:
    """Minimal dataloader — training_step is patched, payload is ignored."""
    inputs = torch.zeros(4, 2)
    targets = torch.zeros(4, 2)
    return DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)


@pytest.mark.integration
class TestTrainingStageCallbackIntegration:
    @pytest.mark.parametrize("use_composite_loss", [True, False])
    @pytest.mark.parametrize("use_scheduler", [True, False])
    @pytest.mark.parametrize("include_gap", [True, False])
    def test_fit_applies_stages_across_epochs(
        self,
        tmp_path: Path,
        real_policy_factory: Callable[..., Policy],
        lightning_policy_factory: Callable[..., LightningPolicy],
        training_config_factory: Callable[..., TrainingConfig],
        use_composite_loss: bool,
        use_scheduler: bool,
        include_gap: bool,
    ) -> None:
        torch.manual_seed(0)
        loss_name = "mse" if use_composite_loss else "loss"
        sub_loss = _ScalarMSELoss(weight=0.8)
        loss_module = (
            CompositeLoss(loss_modules={"mse": sub_loss}, weights={"mse": 1.0})
            if use_composite_loss
            else sub_loss
        )
        policy = real_policy_factory(
            loss=loss_module,
            submodules=_staged_submodules(),
        )
        training_config = training_config_factory(
            optimizer=_grouped_optimizer_config(
                base_learning_rate=1e-3,
                prior_learning_rate=5e-3,
            ),
            lr_schedule="constant" if use_scheduler else None,
            lr_warmup_steps=0,
        )
        lightning_module = lightning_policy_factory(
            policy=policy,
            training_config=training_config,
            total_training_steps=100,
        )

        late_stage_start = 4 if include_gap else 2
        warmup_override = (
            {loss_name: {"weight": 0.5}} if use_composite_loss else {"weight": 0.5}
        )
        late_override = (
            {loss_name: {"weight": 0.1}} if use_composite_loss else {"weight": 0.1}
        )
        stages = [
            _stage(
                name="warmup_posterior",
                start_epoch=0,
                end_epoch=2,
                trainable_groups=["posterior", "decoder"],
                frozen_groups=["prior", OPTIMIZER_UNMATCHED_GROUPS_NAME],
                group_lrs={"posterior": 1e-2},
                loss_weights=warmup_override,
            ),
            _stage(
                name="tune_prior",
                start_epoch=late_stage_start,
                trainable_groups=["prior"],
                frozen_groups=[
                    "posterior",
                    "decoder",
                    OPTIMIZER_UNMATCHED_GROUPS_NAME,
                ],
                group_lrs={"prior": 1e-4},
                loss_weights=late_override,
            ),
        ]
        callback = TrainingStageCallback(
            stages=stages, learning_rate_schedule_active=use_scheduler
        )
        recorder = _StageStateRecorder()
        trainer = pl.Trainer(
            max_epochs=late_stage_start + 1,
            accelerator="cpu",
            devices=1,
            callbacks=[callback, recorder],
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            default_root_dir=str(tmp_path),
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        with patch.object(LightningPolicy, "training_step", _dummy_training_step):
            trainer.fit(model=lightning_module, train_dataloaders=_dummy_dataloader())

        def leaf_weight(snapshot_loss_weights: dict) -> float:
            if use_composite_loss:
                return float(snapshot_loss_weights[loss_name]["weight"])
            return float(snapshot_loss_weights["weight"])

        warmup_snapshot = recorder.history[0]
        assert warmup_snapshot["epoch"] == 0
        assert warmup_snapshot["requires_grad"]["prior.weight"] is False
        assert warmup_snapshot["requires_grad"]["posterior_encoder.weight"] is True
        assert warmup_snapshot["requires_grad"]["decoder.weight"] is True
        assert warmup_snapshot["learning_rates"]["posterior"] == pytest.approx(1e-2)
        assert warmup_snapshot["learning_rates"]["prior"] == pytest.approx(5e-3)
        assert leaf_weight(warmup_snapshot["loss_weights"]) == pytest.approx(0.5)

        if include_gap:
            gap_snapshot = recorder.history[2]
            assert gap_snapshot["epoch"] == 2
            assert gap_snapshot["learning_rates"]["posterior"] == pytest.approx(1e-3)
            assert gap_snapshot["learning_rates"]["prior"] == pytest.approx(5e-3)
            assert leaf_weight(gap_snapshot["loss_weights"]) == pytest.approx(0.8)

        late_snapshot = recorder.history[-1]
        assert late_snapshot["epoch"] == late_stage_start
        assert late_snapshot["requires_grad"]["prior.weight"] is True
        assert late_snapshot["requires_grad"]["posterior_encoder.weight"] is False
        assert late_snapshot["requires_grad"]["decoder.weight"] is False
        assert late_snapshot["learning_rates"]["prior"] == pytest.approx(1e-4)
        assert leaf_weight(late_snapshot["loss_weights"]) == pytest.approx(0.1)

    def test_fit_resume_from_checkpoint_enters_correct_stage(
        self,
        tmp_path: Path,
        real_policy_factory: Callable[..., Policy],
        lightning_policy_factory: Callable[..., LightningPolicy],
        training_config_factory: Callable[..., TrainingConfig],
    ) -> None:
        torch.manual_seed(0)

        def build_components() -> tuple[
            LightningPolicy, list[TrainingStage], _StageStateRecorder
        ]:
            sub_loss = _ScalarMSELoss(weight=0.8)
            loss_module = CompositeLoss(loss_modules={"mse": sub_loss})
            policy = real_policy_factory(
                loss=loss_module,
                submodules=_staged_submodules(),
            )
            training_config = training_config_factory(
                optimizer=_grouped_optimizer_config(base_learning_rate=1e-3),
            )
            lightning_module = lightning_policy_factory(
                policy=policy,
                training_config=training_config,
            )
            stages = [
                _stage(name="early", start_epoch=0, group_lrs={"prior": 1e-2}),
                _stage(name="late", start_epoch=3, group_lrs={"prior": 1e-5}),
            ]
            return lightning_module, stages, _StageStateRecorder()

        first_module, stages, first_recorder = build_components()
        first_callback = TrainingStageCallback(stages=stages)
        checkpoint_path = tmp_path / "checkpoint.ckpt"
        first_trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            callbacks=[first_callback, first_recorder],
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            default_root_dir=str(tmp_path),
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        with patch.object(LightningPolicy, "training_step", _dummy_training_step):
            first_trainer.fit(model=first_module, train_dataloaders=_dummy_dataloader())
        first_trainer.save_checkpoint(str(checkpoint_path))

        assert first_recorder.history[0]["learning_rates"]["prior"] == pytest.approx(
            1e-2
        )

        second_module, _, second_recorder = build_components()
        second_callback = TrainingStageCallback(stages=stages)
        second_trainer = pl.Trainer(
            max_epochs=4,
            accelerator="cpu",
            devices=1,
            callbacks=[second_callback, second_recorder],
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=False,
            logger=False,
            default_root_dir=str(tmp_path),
            num_sanity_val_steps=0,
            limit_val_batches=0,
        )
        with patch.object(LightningPolicy, "training_step", _dummy_training_step):
            second_trainer.fit(
                model=second_module,
                train_dataloaders=_dummy_dataloader(),
                ckpt_path=str(checkpoint_path),
            )

        resumed_epochs = [snapshot["epoch"] for snapshot in second_recorder.history]
        assert 3 in resumed_epochs
        late_snapshot = next(
            snapshot for snapshot in second_recorder.history if snapshot["epoch"] == 3
        )
        assert late_snapshot["learning_rates"]["prior"] == pytest.approx(1e-5)


@pytest.mark.unit
class TestSchedulerHelpers:
    def test_scheduler_current_learning_rates_from_last_lr(
        self, optimizer_with_groups_factory: Callable[..., torch.optim.Optimizer]
    ) -> None:
        optimizer = optimizer_with_groups_factory(group_count=2)
        scheduler = MagicMock()
        scheduler._last_lr = [0.1, 0.2]
        result = TrainingStageCallback._scheduler_current_learning_rates(
            scheduler=scheduler, optimizer=optimizer
        )
        assert result == [0.1, 0.2]

    def test_scheduler_current_learning_rates_falls_back_to_get_last_lr(
        self, optimizer_with_groups_factory: Callable[..., torch.optim.Optimizer]
    ) -> None:
        optimizer = optimizer_with_groups_factory(group_count=2)
        scheduler = MagicMock()
        scheduler._last_lr = None
        scheduler.get_last_lr = MagicMock(return_value=[0.3, 0.4])
        result = TrainingStageCallback._scheduler_current_learning_rates(
            scheduler=scheduler, optimizer=optimizer
        )
        assert result == [0.3, 0.4]
        scheduler.get_last_lr.assert_called_once()

    def test_scheduler_current_learning_rates_raises_when_unavailable(
        self, optimizer_with_groups_factory: Callable[..., torch.optim.Optimizer]
    ) -> None:
        optimizer = optimizer_with_groups_factory(group_count=2)
        scheduler = MagicMock(spec=[])
        with pytest.raises(
            ValueError,
            match=(
                "training.stages with group_lrs requires the scheduler to expose "
                "current learning rates via _last_lr or get_last_lr()."
            ),
        ):
            TrainingStageCallback._scheduler_current_learning_rates(
                scheduler=scheduler, optimizer=optimizer
            )

    def test_scale_learning_rates_zero_base_yields_zero(self) -> None:
        scaled = TrainingStageCallback._scale_learning_rates(
            base_learning_rates=[0.0, 1e-3],
            current_learning_rates=[0.5, 5e-4],
            new_base_learning_rates=[1e-2, 2e-3],
        )
        assert scaled == [0.0, pytest.approx(2e-3 * (5e-4 / 1e-3))]

    def test_get_learning_rate_schedulers_returns_empty_for_invalid_container(
        self,
        trainer_with_scheduler_configs_factory: Callable[..., MagicMock],
    ) -> None:
        trainer = trainer_with_scheduler_configs_factory("not a list or tuple")
        assert (
            TrainingStageCallback._get_learning_rate_schedulers(trainer=trainer) == []
        )

    def test_get_learning_rate_schedulers_mapping_fallback(
        self,
        trainer_with_scheduler_configs_factory: Callable[..., MagicMock],
    ) -> None:
        scheduler = MagicMock()
        trainer = trainer_with_scheduler_configs_factory([{"scheduler": scheduler}])
        assert TrainingStageCallback._get_learning_rate_schedulers(trainer=trainer) == [
            scheduler
        ]

    def test_get_learning_rate_schedulers_skips_none(
        self,
        trainer_with_scheduler_configs_factory: Callable[..., MagicMock],
    ) -> None:
        trainer = trainer_with_scheduler_configs_factory([{"scheduler": None}])
        assert (
            TrainingStageCallback._get_learning_rate_schedulers(trainer=trainer) == []
        )
