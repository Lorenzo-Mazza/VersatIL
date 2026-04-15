"""Tests for versatil.models.layers.gated_linear_unit module."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from versatil.models.layers.gated_linear_unit import GatedLinearUnit, GeGLU, SwiGLU

GATED_VARIANTS = [SwiGLU, GeGLU]
GATE_ACTIVATIONS = {SwiGLU: nn.SiLU, GeGLU: nn.GELU}


@pytest.fixture(params=GATED_VARIANTS, ids=lambda cls: cls.__name__)
def glu_factory(request: pytest.FixtureRequest) -> Callable[..., GatedLinearUnit]:
    """Factory parametrized over all GatedLinearUnit variants."""
    variant_class = request.param

    def factory(
        input_dim: int = 32,
        hidden_dim: int = 64,
        bias: bool = False,
    ) -> GatedLinearUnit:
        return variant_class(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            bias=bias,
        )

    return factory


class TestGatedLinearUnitInitialization:
    @pytest.mark.parametrize("input_dim", [16, 64])
    @pytest.mark.parametrize("hidden_dim", [32, 128])
    def test_stores_configuration(
        self,
        glu_factory: Callable[..., GatedLinearUnit],
        input_dim: int,
        hidden_dim: int,
    ):
        module = glu_factory(input_dim=input_dim, hidden_dim=hidden_dim)
        assert module.gate_proj.in_features == input_dim
        assert module.gate_proj.out_features == hidden_dim
        assert module.value_proj.in_features == input_dim
        assert module.value_proj.out_features == hidden_dim

    def test_is_registered_as_nn_module(
        self,
        glu_factory: Callable[..., GatedLinearUnit],
    ):
        module = glu_factory(input_dim=32, hidden_dim=64)
        param_names = [name for name, _ in module.named_parameters()]
        assert "gate_proj.weight" in param_names
        assert "value_proj.weight" in param_names

    @pytest.mark.parametrize("bias", [True, False])
    def test_bias_affects_parameter_count(
        self,
        glu_factory: Callable[..., GatedLinearUnit],
        bias: bool,
    ):
        module = glu_factory(input_dim=32, hidden_dim=64, bias=bias)
        param_names = {name for name, _ in module.named_parameters()}
        if bias:
            assert "gate_proj.bias" in param_names
            assert "value_proj.bias" in param_names
        else:
            assert "gate_proj.bias" not in param_names
            assert "value_proj.bias" not in param_names


class TestGatedLinearUnitForward:
    @pytest.mark.parametrize(
        "input_dim, hidden_dim",
        [(16, 32), (64, 128)],
    )
    def test_output_shape_2d(
        self,
        glu_factory: Callable[..., GatedLinearUnit],
        flat_tensor_factory: Callable[..., torch.Tensor],
        input_dim: int,
        hidden_dim: int,
    ):
        module = glu_factory(input_dim=input_dim, hidden_dim=hidden_dim)
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=input_dim)
        output = module(tensor)
        assert output.shape == (4, hidden_dim)

    @pytest.mark.parametrize(
        "input_dim, hidden_dim",
        [(16, 32), (64, 128)],
    )
    def test_output_shape_3d(
        self,
        glu_factory: Callable[..., GatedLinearUnit],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        input_dim: int,
        hidden_dim: int,
    ):
        module = glu_factory(input_dim=input_dim, hidden_dim=hidden_dim)
        tensor = sequence_tensor_factory(
            batch_size=2, sequence_length=5, embedding_dimension=input_dim
        )
        output = module(tensor)
        assert output.shape == (2, 5, hidden_dim)

    def test_output_is_differentiable(
        self,
        glu_factory: Callable[..., GatedLinearUnit],
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = glu_factory(input_dim=32, hidden_dim=64)
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
        glu_factory: Callable[..., GatedLinearUnit],
        flat_tensor_factory: Callable[..., torch.Tensor],
        bias: bool,
    ):
        module = glu_factory(input_dim=32, hidden_dim=64, bias=bias)
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=32)
        output = module(tensor)
        assert output.shape == (4, 64)
        assert output.dtype == torch.float32


class TestGateActivationVariants:
    @pytest.mark.parametrize(
        "variant_class, expected_activation",
        list(GATE_ACTIVATIONS.items()),
        ids=lambda cls: cls.__name__,
    )
    def test_gate_activation_type(
        self,
        variant_class: type[GatedLinearUnit],
        expected_activation: type[nn.Module],
    ):
        module = variant_class(input_dim=32, hidden_dim=64)
        assert type(module.gate_activation) is expected_activation

    def test_different_gate_activations_produce_different_outputs(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        swiglu = SwiGLU(input_dim=32, hidden_dim=64)
        geglu = GeGLU(input_dim=32, hidden_dim=64)
        geglu.gate_proj.weight.data.copy_(swiglu.gate_proj.weight.data)
        geglu.value_proj.weight.data.copy_(swiglu.value_proj.weight.data)
        tensor = flat_tensor_factory(batch_size=4, feature_dimension=32)
        assert not torch.allclose(swiglu(tensor), geglu(tensor))

    def test_both_variants_are_gated_linear_units(self):
        assert issubclass(SwiGLU, GatedLinearUnit)
        assert issubclass(GeGLU, GatedLinearUnit)
