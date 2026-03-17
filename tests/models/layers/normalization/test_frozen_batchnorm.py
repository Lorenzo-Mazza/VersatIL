"""Tests for versatil.models.layers.normalization.frozen_batchnorm module."""

from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.normalization.frozen_batchnorm import FrozenBatchNorm2d


@pytest.fixture
def frozen_batchnorm_factory() -> Callable[..., FrozenBatchNorm2d]:
    """Factory for FrozenBatchNorm2d instances with configurable dimension."""

    def factory(
        dimension: int = 16,
    ) -> FrozenBatchNorm2d:
        return FrozenBatchNorm2d(dimension=dimension)

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

    def test_buffers_are_not_learnable_parameters(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
    ):
        norm = frozen_batchnorm_factory(dimension=16)
        assert len(list(norm.parameters())) == 0
        buffer_names = [name for name, _ in norm.named_buffers()]
        assert "weight" in buffer_names
        assert "bias" in buffer_names
        assert "running_mean" in buffer_names
        assert "running_var" in buffer_names


class TestFrozenBatchNorm2dForward:
    def test_default_stats_act_like_identity(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        norm = frozen_batchnorm_factory(dimension=channels)
        tensor = nchw_tensor_factory(channels=channels)
        output = norm(tensor)
        # weight=1, bias=0, running_var=1, running_mean=0 → output ≈ input
        assert torch.allclose(output, tensor, atol=1e-4)

    def test_custom_stats_apply_correct_normalization(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        norm = frozen_batchnorm_factory(dimension=channels)
        norm.running_mean.fill_(2.0)
        norm.running_var.fill_(4.0)
        tensor = nchw_tensor_factory(channels=channels)
        output = norm(tensor)
        # scale = weight * (running_var + eps).rsqrt() = 1.0 * (4.0 + 1e-5)^(-0.5)
        # bias = bias - running_mean * scale = 0 - 2.0 * scale
        eps = 1e-5
        scale = 1.0 * (4.0 + eps) ** (-0.5)
        bias = 0.0 - 2.0 * scale
        expected = tensor * scale + bias
        assert torch.allclose(output, expected, atol=1e-5)

    def test_custom_weight_and_bias_affect_output(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        norm = frozen_batchnorm_factory(dimension=channels)
        norm.weight.fill_(2.0)
        norm.bias.fill_(1.0)
        tensor = nchw_tensor_factory(channels=channels)
        output = norm(tensor)
        eps = 1e-5
        scale = 2.0 * (1.0 + eps) ** (-0.5)
        bias = 1.0 - 0.0 * scale
        expected = tensor * scale + bias
        assert torch.allclose(output, expected, atol=1e-5)

    def test_buffers_do_not_change_in_train_mode(
        self,
        frozen_batchnorm_factory: Callable[..., FrozenBatchNorm2d],
        nchw_tensor_factory: Callable[..., torch.Tensor],
    ):
        channels = 16
        norm = frozen_batchnorm_factory(dimension=channels)
        norm.train()
        weight_before = norm.weight.clone()
        bias_before = norm.bias.clone()
        running_mean_before = norm.running_mean.clone()
        running_var_before = norm.running_var.clone()
        tensor = nchw_tensor_factory(channels=channels)
        norm(tensor)
        assert torch.equal(norm.weight, weight_before)
        assert torch.equal(norm.bias, bias_before)
        assert torch.equal(norm.running_mean, running_mean_before)
        assert torch.equal(norm.running_var, running_var_before)


class TestFrozenBatchNorm2dLoadState:
    def test_load_state_dict_strips_num_batches_tracked_and_loads_values(
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
        norm._load_from_state_dict(
            state_dict=state_dict,
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        assert "num_batches_tracked" not in state_dict
        assert torch.allclose(norm.weight, torch.ones(dimension) * 2.0)
        assert torch.allclose(norm.bias, torch.ones(dimension) * 0.5)

    def test_load_state_dict_succeeds_without_num_batches_tracked(
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
        norm._load_from_state_dict(
            state_dict=state_dict,
            prefix="",
            local_metadata={},
            strict=True,
            missing_keys=[],
            unexpected_keys=[],
            error_msgs=[],
        )
        assert torch.allclose(norm.weight, torch.ones(dimension) * 3.0)
