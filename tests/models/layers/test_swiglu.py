"""Tests for versatil.models.layers.swiglu module."""
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.swiglu import SwiGLU


@pytest.fixture
def swiglu_factory() -> Callable[..., SwiGLU]:
    """Factory for SwiGLU instances with configurable parameters."""
    def factory(
        input_dim: int = 32,
        hidden_dim: int = 64,
        bias: bool = False,
    ) -> SwiGLU:
        return SwiGLU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bias=bias,
        )
    return factory


@pytest.fixture
def input_tensor_2d_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for 2D input tensors (batch_size, input_dim)."""
    def factory(
        batch_size: int = 4,
        input_dim: int = 32,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, input_dim)).astype(np.float32)
        )
    return factory


@pytest.fixture
def input_tensor_3d_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for 3D input tensors (batch_size, sequence_length, input_dim)."""
    def factory(
        batch_size: int = 2,
        sequence_length: int = 5,
        input_dim: int = 32,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, sequence_length, input_dim)).astype(
                np.float32
            )
        )
    return factory


class TestSwiGLUInitialization:

    @pytest.mark.parametrize("input_dim", [16, 64])
    @pytest.mark.parametrize("hidden_dim", [32, 128])
    def test_stores_configuration(
        self,
        swiglu_factory: Callable[..., SwiGLU],
        input_dim: int,
        hidden_dim: int,
    ):
        module = swiglu_factory(input_dim=input_dim, hidden_dim=hidden_dim)
        assert module.gate_proj.in_features == input_dim
        assert module.gate_proj.out_features == hidden_dim
        assert module.value_proj.in_features == input_dim
        assert module.value_proj.out_features == hidden_dim

    def test_inherits_from_nn_module(
        self,
        swiglu_factory: Callable[..., SwiGLU],
    ):
        module = swiglu_factory(input_dim=32, hidden_dim=64)
        assert isinstance(module, nn.Module)

    def test_creates_gate_proj_as_linear(
        self,
        swiglu_factory: Callable[..., SwiGLU],
    ):
        module = swiglu_factory(input_dim=32, hidden_dim=64)
        assert isinstance(module.gate_proj, nn.Linear)

    def test_creates_value_proj_as_linear(
        self,
        swiglu_factory: Callable[..., SwiGLU],
    ):
        module = swiglu_factory(input_dim=32, hidden_dim=64)
        assert isinstance(module.value_proj, nn.Linear)

    @pytest.mark.parametrize("bias", [True, False])
    def test_bias_parameter(
        self,
        swiglu_factory: Callable[..., SwiGLU],
        bias: bool,
    ):
        module = swiglu_factory(input_dim=32, hidden_dim=64, bias=bias)
        has_gate_bias = module.gate_proj.bias is not None
        has_value_bias = module.value_proj.bias is not None
        assert has_gate_bias == bias
        assert has_value_bias == bias


class TestSwiGLUForward:

    @pytest.mark.parametrize("input_dim, hidden_dim", [
        (16, 32),
        (64, 128),
    ])
    def test_output_shape_2d(
        self,
        swiglu_factory: Callable[..., SwiGLU],
        input_tensor_2d_factory: Callable[..., torch.Tensor],
        input_dim: int,
        hidden_dim: int,
    ):
        module = swiglu_factory(input_dim=input_dim, hidden_dim=hidden_dim)
        tensor = input_tensor_2d_factory(batch_size=4, input_dim=input_dim)
        output = module(tensor)
        assert output.shape == (4, hidden_dim)

    @pytest.mark.parametrize("input_dim, hidden_dim", [
        (16, 32),
        (64, 128),
    ])
    def test_output_shape_3d(
        self,
        swiglu_factory: Callable[..., SwiGLU],
        input_tensor_3d_factory: Callable[..., torch.Tensor],
        input_dim: int,
        hidden_dim: int,
    ):
        module = swiglu_factory(input_dim=input_dim, hidden_dim=hidden_dim)
        tensor = input_tensor_3d_factory(
            batch_size=2, sequence_length=5, input_dim=input_dim
        )
        output = module(tensor)
        assert output.shape == (2, 5, hidden_dim)

    def test_output_is_differentiable(
        self,
        swiglu_factory: Callable[..., SwiGLU],
        input_tensor_2d_factory: Callable[..., torch.Tensor],
    ):
        module = swiglu_factory(input_dim=32, hidden_dim=64)
        tensor = input_tensor_2d_factory(batch_size=4, input_dim=32)
        tensor.requires_grad_(True)
        output = module(tensor)
        loss = output.sum()
        loss.backward()
        assert tensor.grad is not None
        assert tensor.grad.shape == tensor.shape

    @pytest.mark.parametrize("bias", [True, False])
    def test_forward_with_bias_configuration(
        self,
        swiglu_factory: Callable[..., SwiGLU],
        input_tensor_2d_factory: Callable[..., torch.Tensor],
        bias: bool,
    ):
        module = swiglu_factory(input_dim=32, hidden_dim=64, bias=bias)
        tensor = input_tensor_2d_factory(batch_size=4, input_dim=32)
        output = module(tensor)
        assert output.shape == (4, 64)
        assert output.dtype == torch.float32
