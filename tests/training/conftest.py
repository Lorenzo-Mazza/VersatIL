"""Shared fixtures for versatil.training tests."""

from collections.abc import Callable
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from versatil.configs.training import (
    AdamWConfig,
    OptimizerConfig,
    TrainingConfig,
)
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.metrics.base import BaseLoss
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.encoding.pipeline import EncodingPipeline
from versatil.models.policy import Policy
from versatil.training.lightning_policy import LightningPolicy


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
def real_policy_factory() -> Callable[..., Policy]:
    """Build a real ``Policy`` with ``MagicMock(spec=...)`` heavy deps.

    ``MagicMock(spec=nn.Module)`` passes ``isinstance(..., nn.Module)``, so
    ``Policy.__init__`` registers the mocks inside ``Policy._modules`` via
    ``nn.Module.__setattr__``. Lightning's ``state_dict`` /
    ``load_state_dict`` walk ``_modules`` recursively and trip over the mocks
    (they have no ``_modules`` of their own). To keep the mocks accessible as
    attributes but hidden from the submodule walk, they're moved from
    ``_modules`` into ``__dict__`` via ``object.__setattr__`` after init.

    Pass ``submodules`` as ``{attribute_name: nn.Module}`` to register real
    parameter-bearing children onto the returned policy (overwriting the
    mocked decoder etc.) so ``policy.named_parameters()`` yields real tensors
    usable by the optimizer and the stage callback.
    """

    def factory(
        loss: torch.nn.Module | None = None,
        submodules: dict[str, torch.nn.Module] | None = None,
        prediction_horizon: int = 4,
        observation_horizon: int = 1,
    ) -> Policy:
        encoding_pipeline = MagicMock(spec=EncodingPipeline)
        encoding_pipeline.encoders = {}
        encoding_pipeline.conditional_encoders = {}
        policy = Policy(
            encoding_pipeline=encoding_pipeline,
            algorithm=MagicMock(spec=DecodingAlgorithm),
            decoder=MagicMock(
                spec=ActionDecoder,
                decoder_input=DecoderInput(keys=[]),
            ),
            observation_space=MagicMock(
                spec=ObservationSpace,
                observations_metadata={},
            ),
            action_space=MagicMock(spec=ActionSpace),
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            loss=loss if loss is not None else MagicMock(spec=BaseLoss),
            device="cpu",
        )
        overridden_attributes = set(submodules or {})
        for attribute_name in ("encoding_pipeline", "algorithm", "decoder"):
            if attribute_name in overridden_attributes:
                continue
            mock = policy._modules.pop(attribute_name)
            object.__setattr__(policy, attribute_name, mock)
        for attribute_name, module in (submodules or {}).items():
            setattr(policy, attribute_name, module)
        return policy

    return factory


@pytest.fixture
def lightning_policy_factory(
    mock_policy_factory: Callable,
    training_config_factory: Callable,
) -> Callable[..., LightningPolicy]:
    def factory(
        policy: Mock | None = None,
        training_config: TrainingConfig | None = None,
        total_training_steps: int | None = None,
    ) -> LightningPolicy:
        if policy is None:
            policy = mock_policy_factory()
        if training_config is None:
            training_config = training_config_factory()
        return LightningPolicy(
            policy=policy,
            training_config=training_config,
            total_training_steps=total_training_steps,
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

        all_params = tuple(param for _, param in named_parameters)
        mock.parameters.side_effect = lambda: iter(all_params)
        mock.named_parameters.side_effect = lambda: iter(named_parameters)

        # Support modules() for EMA traversal
        mock_module = MagicMock()
        mock_module.parameters.side_effect = lambda: iter(all_params)
        mock.modules.side_effect = lambda: iter([mock_module])

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
        max_epochs: int = 100,
        estimated_stepping_batches: int = 1000,
        callback_metrics: dict[str, torch.Tensor] | None = None,
        logger: Mock | None = "default",
        optimizers: list[torch.optim.Optimizer] | None = None,
    ) -> Mock:
        trainer = MagicMock(spec="pl.Trainer")
        trainer.current_epoch = current_epoch
        trainer.global_step = global_step
        trainer.max_epochs = max_epochs
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
