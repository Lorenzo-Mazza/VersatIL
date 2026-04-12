"""Shared fixtures for versatil.training tests."""

from collections.abc import Callable
from unittest.mock import MagicMock, Mock

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from versatil.configs.training import (
    AdamWConfig,
    OptimizerConfig,
    TrainingConfig,
)


@pytest.fixture
def training_config_factory() -> Callable[..., TrainingConfig]:
    def factory(
        num_epochs: int = 10,
        optimizer: OptimizerConfig | None = None,
        lr_schedule: str | None = None,
        lr_warmup_steps: int = 100,
        lr_scheduler_kwargs: dict[str, float] | None = None,
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
            lr_scheduler_kwargs=lr_scheduler_kwargs or {},
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


class _RealLightningModule(pl.LightningModule):
    """Minimal real LightningModule for integration tests that need trainer.fit()."""

    def __init__(self, policy: torch.nn.Module, learning_rate: float = 0.01):
        super().__init__()
        self.policy = policy
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self.policy(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate)


@pytest.fixture
def real_lightning_module_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[pl.LightningModule, DataLoader]]:
    """Factory that returns a real LightningModule + DataLoader for integration tests."""

    def factory(
        input_dimension: int = 4,
        output_dimension: int = 4,
        num_samples: int = 16,
        batch_size: int = 4,
        learning_rate: float = 0.01,
    ) -> tuple[pl.LightningModule, DataLoader]:
        policy = torch.nn.Linear(input_dimension, output_dimension)
        x = torch.from_numpy(
            rng.standard_normal((num_samples, input_dimension)).astype(np.float32)
        )
        y = torch.from_numpy(
            rng.standard_normal((num_samples, output_dimension)).astype(np.float32)
        )
        dataloader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
        module = _RealLightningModule(policy=policy, learning_rate=learning_rate)
        return module, dataloader

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
