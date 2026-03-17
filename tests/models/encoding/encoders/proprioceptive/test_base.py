"""Tests for versatil.models.encoding.encoders.proprioceptive.base module."""

from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pytest
import torch

from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.encoding.encoders.proprioceptive.base import ProprioceptiveEncoder
from versatil.models.layers.activation import ActivationFunction


@pytest.fixture
def proprioceptive_input_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for proprioceptive input tensors."""

    def factory(
        keys: list[str] | None = None,
        batch_size: int = 4,
        input_dimension: int = 7,
        time_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        if keys is None:
            keys = ["proprio_robot_frame"]
        result = {}
        for key in keys:
            if time_steps is not None:
                shape = (batch_size, time_steps, input_dimension)
            else:
                shape = (batch_size, input_dimension)
            result[key] = torch.from_numpy(
                rng.standard_normal(shape).astype(np.float32)
            )
        return result

    return factory


@pytest.fixture
def proprioceptive_encoder_factory() -> Callable[..., ProprioceptiveEncoder]:
    """Factory for ProprioceptiveEncoder instances."""

    def factory(
        input_keys: str | list[str] = "proprio_robot_frame",
        output_dimension: int = 64,
        hidden_dimensions: list[int] | None = None,
        activation: str = ActivationFunction.RELU.value,
        dropout: float = 0.0,
        pretrained: bool = False,
        frozen: bool = False,
    ) -> ProprioceptiveEncoder:
        return ProprioceptiveEncoder(
            input_keys=input_keys,
            output_dim=output_dimension,
            hidden_dims=hidden_dimensions,
            activation=activation,
            dropout=dropout,
            pretrained=pretrained,
            frozen=frozen,
        )

    return factory


class TestProprioceptiveEncoderInitialization:
    @pytest.mark.parametrize(
        "input_keys, expected_keys",
        [
            ("proprio_robot_frame", ["proprio_robot_frame"]),
            (
                ["proprio_robot_frame", "gripper_state_obs"],
                ["proprio_robot_frame", "gripper_state_obs"],
            ),
        ],
    )
    @pytest.mark.parametrize("output_dimension", [64, 128])
    @pytest.mark.parametrize("hidden_dimensions", [None, [128]])
    @pytest.mark.parametrize("dropout", [0.0, 0.5])
    @pytest.mark.parametrize(
        "activation, expected_class",
        [
            (ActivationFunction.RELU.value, torch.nn.ReLU),
            (ActivationFunction.GELU.value, torch.nn.GELU),
        ],
    )
    def test_stores_configuration(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
        input_keys: str | list[str],
        expected_keys: list[str],
        output_dimension: int,
        hidden_dimensions: list[int] | None,
        dropout: float,
        activation: str,
        expected_class: type,
    ):
        encoder = proprioceptive_encoder_factory(
            input_keys=input_keys,
            output_dimension=output_dimension,
            hidden_dimensions=hidden_dimensions,
            dropout=dropout,
            activation=activation,
        )
        assert encoder.output_dim == output_dimension
        assert encoder.hidden_dims == hidden_dimensions
        assert encoder.dropout == dropout
        assert encoder.activation_fn == expected_class
        assert encoder.input_specification.keys == expected_keys
        assert encoder.network is None

    def test_has_encoder_interface(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
    ):
        encoder = proprioceptive_encoder_factory()
        assert hasattr(encoder, "forward")
        assert hasattr(encoder, "get_output_specification")
        assert hasattr(encoder, "input_specification")


class TestProprioceptiveEncoderBuildNetwork:
    @pytest.mark.parametrize(
        "frozen, expected_requires_grad",
        [
            (False, True),
            (True, False),
        ],
    )
    def test_parameter_grad_matches_frozen_flag(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
        frozen: bool,
        expected_requires_grad: bool,
    ):
        encoder = proprioceptive_encoder_factory(
            output_dimension=64,
            hidden_dimensions=[128],
            frozen=frozen,
        )
        encoder._build_network(input_dim=7)
        for parameter in encoder.network.parameters():
            assert parameter.requires_grad is expected_requires_grad


class TestProprioceptiveEncoderForward:
    @pytest.mark.parametrize(
        "time_steps, expected_ndim",
        [
            (None, 2),
            (3, 3),
        ],
    )
    def test_output_shape_with_and_without_time(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
        proprioceptive_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int | None,
        expected_ndim: int,
    ):
        batch_size = 4
        output_dimension = 64
        encoder = proprioceptive_encoder_factory(output_dimension=output_dimension)
        inputs = proprioceptive_input_factory(
            batch_size=batch_size,
            input_dimension=7,
            time_steps=time_steps,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.PROPRIOCEPTIVE.value]
        assert features.ndim == expected_ndim
        assert features.shape[0] == batch_size
        assert features.shape[-1] == output_dimension
        if time_steps is not None:
            assert features.shape[1] == time_steps

    def test_lazily_builds_network_on_first_forward(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
        proprioceptive_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = proprioceptive_encoder_factory(output_dimension=64)
        assert encoder.network is None
        inputs = proprioceptive_input_factory(input_dimension=7)
        encoder(inputs)
        assert encoder.network is not None

    def test_moves_network_to_input_device(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
        proprioceptive_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = proprioceptive_encoder_factory(output_dimension=64)
        inputs = proprioceptive_input_factory(input_dimension=7)
        encoder(inputs)
        input_device = next(iter(inputs.values())).device
        for parameter in encoder.network.parameters():
            assert parameter.device.type == input_device.type

    def test_concatenates_multiple_input_keys(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
        proprioceptive_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        keys = ["proprio_robot_frame", "gripper_state_obs"]
        batch_size = 4
        output_dimension = 64
        encoder = proprioceptive_encoder_factory(
            input_keys=keys,
            output_dimension=output_dimension,
        )
        inputs = proprioceptive_input_factory(
            keys=keys,
            batch_size=batch_size,
            input_dimension=7,
        )
        output = encoder(inputs)
        features = output[EncoderOutputKeys.PROPRIOCEPTIVE.value]
        assert features.shape == (batch_size, output_dimension)

    @pytest.mark.parametrize("hidden_dimensions", [None, [128], [256, 128]])
    def test_forward_with_varying_hidden_layers(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
        proprioceptive_input_factory: Callable[..., dict[str, torch.Tensor]],
        hidden_dimensions: list[int] | None,
    ):
        output_dimension = 32
        encoder = proprioceptive_encoder_factory(
            output_dimension=output_dimension,
            hidden_dimensions=hidden_dimensions,
        )
        inputs = proprioceptive_input_factory(input_dimension=7)
        output = encoder(inputs)
        features = output[EncoderOutputKeys.PROPRIOCEPTIVE.value]
        assert features.shape[-1] == output_dimension

    def test_raises_runtime_error_when_build_network_fails(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
        proprioceptive_input_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        encoder = proprioceptive_encoder_factory(output_dimension=64)
        inputs = proprioceptive_input_factory(input_dimension=7)
        with (
            patch.object(
                ProprioceptiveEncoder,
                "_build_network",
            ),
            pytest.raises(RuntimeError, match="Network should be built"),
        ):
            encoder(inputs)


class TestProprioceptiveEncoderGetOutputSpecification:
    def test_features_and_dimensions_match_output_dimension(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
    ):
        encoder = proprioceptive_encoder_factory(output_dimension=128)
        specification = encoder.get_output_specification()
        assert specification.features == [EncoderOutputKeys.PROPRIOCEPTIVE.value]
        assert specification.dimensions == {
            EncoderOutputKeys.PROPRIOCEPTIVE.value: 128,
        }


class TestProprioceptiveEncoderGetOutputDims:
    def test_returns_proprioceptive_key_with_output_dimension(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
    ):
        encoder = proprioceptive_encoder_factory(output_dimension=128)
        output_dims = encoder.get_output_dims()
        assert output_dims == {EncoderOutputKeys.PROPRIOCEPTIVE.value: 128}
