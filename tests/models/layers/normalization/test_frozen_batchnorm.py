"""Tests for versatil.models.layers.normalization.frozen_batchnorm module."""
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

from versatil.models.layers.normalization.frozen_batchnorm import FrozenBatchNorm2d


@pytest.fixture
def frozen_batchnorm_factory() -> Callable[..., FrozenBatchNorm2d]:
    """Factory for FrozenBatchNorm2d instances with configurable dimension."""
    def factory(
        dimension: int = 16,
    ) -> FrozenBatchNorm2d:
        return FrozenBatchNorm2d(dimension=dimension)
    return factory


@pytest.fixture
def image_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for 4D image tensors (B, C, H, W)."""
    def factory(
        batch_size: int = 2,
        channels: int = 16,
        height: int = 8,
        width: int = 8,
    ) -> torch.Tensor:
        shape = (batch_size, channels, height, width)
        return torch.from_numpy(
            rng.standard_normal(shape).astype(np.float32)
        )
    return factory


class TestFrozenBatchNorm2dInitialization:

    @pytest.mark.parametrize("dimension", [8, 32])
    def test_registers_buffers_with_correct_values(
        self,
        dimension: int,
    ):
        norm = FrozenBatchNorm2d(dimension=dimension)
        assert torch.allclose(norm.weight, torch.ones(dimension))
        assert torch.allclose(norm.bias, torch.zeros(dimension))
        assert torch.allclose(norm.running_mean, torch.zeros(dimension))
        assert torch.allclose(norm.running_var, torch.ones(dimension))

    def test_buffers_are_not_parameters(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
    ):
        norm = frozen_batchnorm_factory(dimension=16)
        parameter_names = [name for name, _ in norm.named_parameters()]
        assert len(parameter_names) == 0
        buffer_names = [name for name, _ in norm.named_buffers()]
        assert "weight" in buffer_names
        assert "bias" in buffer_names
        assert "running_mean" in buffer_names
        assert "running_var" in buffer_names

    def test_inherits_from_nn_module(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
    ):
        norm = frozen_batchnorm_factory(dimension=16)
        assert isinstance(norm, nn.Module)


class TestFrozenBatchNorm2dForward:

    @pytest.mark.parametrize("batch_size, channels, height, width", [
        (2, 16, 8, 8),
        (1, 32, 4, 4),
    ])
    def test_output_shape_matches_input(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
        image_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
        channels: int,
        height: int,
        width: int,
    ):
        norm = frozen_batchnorm_factory(dimension=channels)
        tensor = image_tensor_factory(
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
        )
        output = norm(tensor)
        assert output.shape == (batch_size, channels, height, width)

    def test_default_stats_act_like_identity(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
        image_tensor_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        norm = frozen_batchnorm_factory(dimension=channels)
        tensor = image_tensor_factory(channels=channels)
        output = norm(tensor)
        # weight=1, bias=0, running_var=1, running_mean=0
        # scale = 1 * (1 + 1e-5).rsqrt() ≈ 1, bias = 0 - 0*scale = 0
        assert torch.allclose(output, tensor, atol=1e-4)

    def test_parameters_do_not_change_in_train_mode(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
        image_tensor_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        norm = frozen_batchnorm_factory(dimension=channels)
        norm.train()
        weight_before = norm.weight.clone()
        bias_before = norm.bias.clone()
        running_mean_before = norm.running_mean.clone()
        running_var_before = norm.running_var.clone()
        tensor = image_tensor_factory(channels=channels)
        norm(tensor)
        assert torch.equal(norm.weight, weight_before)
        assert torch.equal(norm.bias, bias_before)
        assert torch.equal(norm.running_mean, running_mean_before)
        assert torch.equal(norm.running_var, running_var_before)


class TestFrozenBatchNorm2dLoadState:

    def test_load_from_state_dict_removes_num_batches_tracked(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
    ):
        dimension = 16
        norm = frozen_batchnorm_factory(dimension=dimension)
        state_dict = {
            "weight": torch.ones(dimension) * 2.0,
            "bias": torch.ones(dimension) * 0.5,
            "running_mean": torch.ones(dimension) * 0.1,
            "running_var": torch.ones(dimension) * 0.9,
            "num_batches_tracked": torch.tensor(100),
        }
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        norm._load_from_state_dict(
            state_dict=state_dict,
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        assert "num_batches_tracked" not in state_dict
        assert torch.allclose(norm.weight, torch.ones(dimension) * 2.0)
        assert torch.allclose(norm.bias, torch.ones(dimension) * 0.5)

    def test_load_from_state_dict_works_without_num_batches_tracked(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
    ):
        dimension = 16
        norm = frozen_batchnorm_factory(dimension=dimension)
        state_dict = {
            "weight": torch.ones(dimension) * 3.0,
            "bias": torch.zeros(dimension),
            "running_mean": torch.zeros(dimension),
            "running_var": torch.ones(dimension),
        }
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        norm._load_from_state_dict(
            state_dict=state_dict,
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        assert torch.allclose(norm.weight, torch.ones(dimension) * 3.0)
