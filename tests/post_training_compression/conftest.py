"""Shared fixtures for post-training compression tests."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.configs.post_training_compression import PreparationConfig
from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.post_training_compression.compressor import PostTrainingCompressor
from versatil.post_training_compression.pruning.base import BasePruner
from versatil.quantization.strategies import PT2EStrategy, QuantizeApiStrategy


@pytest.fixture
def spatial_input_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for spatial input tensors (B, C, H, W)."""

    def factory(
        batch_size: int = 2,
        channels: int = 3,
        height: int = 8,
        width: int = 8,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, channels, height, width)).astype(
                np.float32
            )
        )

    return factory


@pytest.fixture
def pruning_model_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Module]:
    """Factory for a model with Conv2d and Linear layers."""

    def factory(
        input_channels: int = 3,
        hidden_channels: int = 32,
        linear_features: int = 64,
        output_features: int = 4,
    ) -> nn.Module:
        model = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_channels,
                out_features=linear_features,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=linear_features,
                out_features=output_features,
            ),
        )
        with torch.no_grad():
            for parameter in model.parameters():
                data = rng.standard_normal(parameter.shape).astype(np.float32)
                parameter.copy_(torch.from_numpy(data))
        return model

    return factory


@pytest.fixture
def mock_policy_factory() -> Callable[..., MagicMock]:
    """Factory for mock nn.Module policies with configurable submodules."""

    def factory(
        submodule_paths: dict[str, MagicMock] | None = None,
    ) -> MagicMock:
        policy = MagicMock(spec=nn.Module)
        if submodule_paths:
            policy.get_submodule.side_effect = lambda path: submodule_paths[path]
        return policy

    return factory


@pytest.fixture
def mock_pruner_factory() -> Callable[..., MagicMock]:
    """Factory for mock BasePruner instances with configurable sparsity."""

    def factory(
        total_parameters: int = 100,
        zero_parameters: int = 50,
    ) -> MagicMock:
        pruner = MagicMock(spec=BasePruner)
        pruner.prune.return_value = (total_parameters, zero_parameters)
        return pruner

    return factory


@pytest.fixture
def compressor_factory() -> Callable[..., PostTrainingCompressor]:
    """Factory for PostTrainingCompressor with sensible defaults."""

    def factory(
        checkpoint_path: str = "/tmp/ckpt",
        modules: list[CompressionTarget] | None = None,
        preparation: PreparationConfig | None = None,
        output_directory: str | None = None,
        generate_report: bool = False,
        calibration_steps: int = 32,
        pruning: list[BasePruner] | None = None,
        quantization: PT2EStrategy | QuantizeApiStrategy | None = None,
    ) -> PostTrainingCompressor:
        return PostTrainingCompressor(
            checkpoint_path=checkpoint_path,
            modules=modules or [],
            preparation=preparation or PreparationConfig(),
            output_directory=output_directory,
            generate_report=generate_report,
            calibration_steps=calibration_steps,
            pruning=pruning,
            quantization=quantization,
        )

    return factory


def verify_reload_fidelity(
    original_model: nn.Module,
    reloaded_model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
) -> bool:
    """Verify exact numerical match between original and reloaded model.

    Args:
        original_model: The model before save.
        reloaded_model: The model after loading from disk.
        example_inputs: Inputs to run through both models.

    Returns:
        True if all output tensors match exactly.
    """
    with torch.no_grad():
        original_outputs = original_model(*example_inputs)
        reloaded_outputs = reloaded_model(*example_inputs)
    if len(original_outputs) != len(reloaded_outputs):
        return False
    return all(
        torch.equal(original, reloaded)
        for original, reloaded in zip(original_outputs, reloaded_outputs)
    )
