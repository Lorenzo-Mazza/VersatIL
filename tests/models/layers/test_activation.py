"""Tests for versatil.models.layers.activation module."""

import enum

import pytest
from torch import nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.swiglu import SwiGLU

EXPECTED_MEMBERS = {
    "RELU": "relu",
    "GELU": "gelu",
    "SILU": "silu",
    "SWIGLU": "swiglu",
    "SIGMOID": "sigmoid",
    "TANH": "tanh",
    "LEAKY_RELU": "leaky_relu",
    "LINEAR": "linear",
    "MISH": "mish",
}

ACTIVATION_TO_TORCH_TYPE = {
    ActivationFunction.RELU: nn.ReLU,
    ActivationFunction.GELU: nn.GELU,
    ActivationFunction.SILU: nn.SiLU,
    ActivationFunction.SWIGLU: SwiGLU,
    ActivationFunction.SIGMOID: nn.Sigmoid,
    ActivationFunction.TANH: nn.Tanh,
    ActivationFunction.LEAKY_RELU: nn.LeakyReLU,
    ActivationFunction.LINEAR: nn.Identity,
    ActivationFunction.MISH: nn.Mish,
}


class TestActivationFunctionEnum:
    def test_is_str_enum(self):
        assert issubclass(ActivationFunction, str)
        assert issubclass(ActivationFunction, enum.Enum)

    def test_has_all_expected_members(self):
        member_names = {member.name for member in ActivationFunction}
        assert member_names == set(EXPECTED_MEMBERS.keys())

    @pytest.mark.parametrize("name, value", list(EXPECTED_MEMBERS.items()))
    def test_member_values(self, name: str, value: str):
        member = ActivationFunction[name]
        assert member.value == value

    def test_member_count(self):
        assert len(ActivationFunction) == len(EXPECTED_MEMBERS)

    @pytest.mark.parametrize("member", list(ActivationFunction))
    def test_members_are_strings(self, member: ActivationFunction):
        assert isinstance(member, str)
        assert isinstance(member.value, str)


class TestToTorchActivation:
    @pytest.mark.parametrize(
        "activation, expected_type",
        list(ACTIVATION_TO_TORCH_TYPE.items()),
        ids=[member.value for member in ACTIVATION_TO_TORCH_TYPE],
    )
    def test_returns_correct_torch_module_type(
        self,
        activation: ActivationFunction,
        expected_type: type[nn.Module],
    ):
        result = activation.to_torch_activation()
        assert result is expected_type

    @pytest.mark.parametrize("activation", list(ActivationFunction))
    def test_returned_type_is_nn_module_subclass(
        self,
        activation: ActivationFunction,
    ):
        result = activation.to_torch_activation()
        assert issubclass(result, nn.Module)
