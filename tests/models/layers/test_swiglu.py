"""Tests for versatil.models.layers.swiglu module."""

from collections.abc import Callable

import pytest
import torch

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

    def test_is_registered_as_nn_module(
        self,
        swiglu_factory: Callable[..., SwiGLU],
    ):
        module = swiglu_factory(input_dim=32, hidden_dim=64)
        # Functional: parameters are discoverable via Module API
        param_names = [name for name, _ in module.named_parameters()]
        assert "gate_proj.weight" in param_names
        assert "value_proj.weight" in param_names

    @pytest.mark.parametrize("bias", [True, False])
    def test_bias_affects_parameter_count(
        self,
        swiglu_factory: Callable[..., SwiGLU],
        bias: bool,
    ):
        module = swiglu_factory(input_dim=32, hidden_dim=64, bias=bias)
        param_names = {name for name, _ in module.named_parameters()}
        if bias:
            assert "gate_proj.bias" in param_names
            assert "value_proj.bias" in param_names
        else:
            assert "gate_proj.bias" not in param_names
            assert "value_proj.bias" not in param_names


class TestSwiGLUForward:
    @pytest.mark.parametrize(
        "input_dim, hidden_dim",
        [
            (16, 32),
            (64, 128),
        ],
    )
    def test_output_shape_2d(
        self,
        swiglu_factory: Callable[..., SwiGLU],
        flat_tensor_factory: Callable[..., torch.Tensor],
        input_dim: int,
        hidden_dim: int,
    ):
        module = swiglu_factory(input_dim=input_dim, hidden_dim=hidden_dim)
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=input_dim)
        output = module(tensor)
        assert output.shape == (4, hidden_dim)

    @pytest.mark.parametrize(
        "input_dim, hidden_dim",
        [
            (16, 32),
            (64, 128),
        ],
    )
    def test_output_shape_3d(
        self,
        swiglu_factory: Callable[..., SwiGLU],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        input_dim: int,
        hidden_dim: int,
    ):
        module = swiglu_factory(input_dim=input_dim, hidden_dim=hidden_dim)
        tensor = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=input_dim
        )
        output = module(tensor)
        assert output.shape == (2, 5, hidden_dim)

    def test_output_is_differentiable(
        self,
        swiglu_factory: Callable[..., SwiGLU],
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = swiglu_factory(input_dim=32, hidden_dim=64)
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=32)
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
        flat_tensor_factory: Callable[..., torch.Tensor],
        bias: bool,
    ):
        module = swiglu_factory(input_dim=32, hidden_dim=64, bias=bias)
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=32)
        output = module(tensor)
        assert output.shape == (4, 64)
        assert output.dtype == torch.float32
