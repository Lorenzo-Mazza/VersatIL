"""Shared fixtures for versatil.training tests."""

from collections.abc import Callable
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

from versatil.configs.training import (
    AdamWConfig,
    OptimizerConfig,
    TrainingConfig,
)
from versatil.models.policy import Policy


@pytest.fixture
def training_config_factory() -> Callable[..., TrainingConfig]:
    def factory(
        num_epochs: int = 10,
        optimizer: OptimizerConfig | None = None,
        lr_schedule: str | None = None,
        lr_warmup_steps: int = 100,
        use_ema: bool = False,
        clip_gradient_norm: bool = False,
        clip_max_norm: float = 0.1,
    ) -> TrainingConfig:
        if optimizer is None:
            optimizer = AdamWConfig(lr=1e-4)
        return TrainingConfig(
            num_epochs=num_epochs,
            optimizer=optimizer,
            lr_schedule=lr_schedule,
            lr_warmup_steps=lr_warmup_steps,
            use_ema=use_ema,
            clip_gradient_norm=clip_gradient_norm,
            clip_max_norm=clip_max_norm,
        )

    return factory


@pytest.fixture
def mock_policy_factory(rng: np.random.Generator) -> Callable[..., Mock]:
    def factory(
        named_parameters: list[tuple[str, torch.nn.Parameter]] | None = None,
    ) -> Mock:
        mock = MagicMock(spec=Policy)

        if named_parameters is None:
            weight_data = torch.from_numpy(
                rng.standard_normal((8, 4)).astype(np.float32)
            )
            bias_data = torch.from_numpy(rng.standard_normal((8,)).astype(np.float32))
            weight = torch.nn.Parameter(weight_data)
            bias = torch.nn.Parameter(bias_data)
            named_parameters = [("layer.weight", weight), ("layer.bias", bias)]

        all_params = [param for _, param in named_parameters]
        mock.parameters.return_value = iter(all_params)
        mock.named_parameters.return_value = iter(named_parameters)

        # Support modules() for EMA traversal
        mock_module = MagicMock()
        mock_module.parameters.return_value = iter(all_params)
        mock.modules.return_value = iter([mock_module])

        return mock

    return factory


@pytest.fixture
def mock_trainer_factory() -> Callable[..., Mock]:
    def factory(
        current_epoch: int = 0,
        global_step: int = 0,
        estimated_stepping_batches: int = 1000,
        callback_metrics: dict[str, torch.Tensor] | None = None,
        logger: Mock | None = "default",
        optimizers: list[torch.optim.Optimizer] | None = None,
    ) -> Mock:
        trainer = MagicMock(spec="pl.Trainer")
        trainer.current_epoch = current_epoch
        trainer.global_step = global_step
        trainer.estimated_stepping_batches = estimated_stepping_batches
        trainer.callback_metrics = (
            callback_metrics if callback_metrics is not None else {}
        )
        if logger == "default":
            trainer.logger = MagicMock()
        else:
            trainer.logger = logger
        trainer.optimizers = optimizers if optimizers is not None else []
        return trainer

    return factory
