"""Tests for versatil.models.encoding.encoders.proprioceptive.base module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.metadata import BaseMetadata, CameraMetadata
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
        time_steps: int = 1,
    ) -> dict[str, torch.Tensor]:
        if keys is None:
            keys = ["proprio_robot_frame"]
        result = {}
        for key in keys:
            shape = (batch_size, time_steps, input_dimension)
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
        model_dtype: str | None = None,
    ) -> ProprioceptiveEncoder:
        return ProprioceptiveEncoder(
            input_keys=input_keys,
            output_dim=output_dimension,
            hidden_dimensions=hidden_dimensions,
            activation=activation,
            dropout=dropout,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
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
        assert encoder.hidden_dimensions == hidden_dimensions
        assert encoder.dropout == dropout
        assert encoder.activation_fn == expected_class
        assert encoder.input_specification.keys == expected_keys
        assert encoder.network is None

    def test_has_encoder_interface(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
    ):
        encoder = proprioceptive_encoder_factory()
        spec = encoder.get_output_specification()
        feature_keys = [m.key for m in spec]
        assert feature_keys == [EncoderOutputKeys.PROPRIOCEPTIVE.value]


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
        encoder._build_network(input_dimension=7)
        for parameter in encoder.network.parameters():
            assert parameter.requires_grad is expected_requires_grad


class TestProprioceptiveEncoderForward:
    @pytest.mark.parametrize("time_steps", [1, 3])
    def test_output_shape_with_temporal_dimension(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
        proprioceptive_input_factory: Callable[..., dict[str, torch.Tensor]],
        time_steps: int,
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
        assert features.shape == (batch_size, time_steps, output_dimension)

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
        assert features.shape == (batch_size, 1, output_dimension)

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
            pytest.raises(
                RuntimeError,
                match="Network should be built by _build_network",
            ),
        ):
            encoder(inputs)


class TestProprioceptiveEncoderValidateInputMetadata:
    @pytest.mark.parametrize(
        "metadata, expected_error",
        [
            (
                CameraMetadata(
                    camera_key="left",
                    dtype="uint8",
                    channels=3,
                    image_height=224,
                    image_width=224,
                ),
                "ProprioceptiveEncoder cannot process image data for 'proprio_robot_frame'. "
                "Got CameraMetadata, expected proprioceptive state input.",
            ),
            (
                MagicMock(spec=BaseMetadata),
                None,
            ),
        ],
    )
    def test_validates_non_camera_metadata(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
        metadata,
        expected_error: str | None,
    ):
        encoder = proprioceptive_encoder_factory()
        result = encoder.validate_input_metadata(
            key="proprio_robot_frame", metadata=metadata
        )
        assert result == expected_error


class TestProprioceptiveEncoderGetOutputSpecification:
    def test_features_and_dimensions_match_output_dimension(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
    ):
        encoder = proprioceptive_encoder_factory(output_dimension=128)
        specification = encoder.get_output_specification()
        feature_keys = [m.key for m in specification]
        assert feature_keys == [EncoderOutputKeys.PROPRIOCEPTIVE.value]
        assert next(
            m for m in specification if m.key == EncoderOutputKeys.PROPRIOCEPTIVE.value
        ).dimension == (128,)


class TestProprioceptiveEncoderGetOutputDims:
    def test_returns_proprioceptive_key_with_output_dimension(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
    ):
        encoder = proprioceptive_encoder_factory(output_dimension=128)
        output_dims = encoder.get_output_dims()
        assert output_dims == {EncoderOutputKeys.PROPRIOCEPTIVE.value: 128}


class TestProprioceptiveEncoderModelDtype:
    @pytest.mark.unit
    def test_apply_model_dtype_called_in_build_network(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
    ):
        encoder = proprioceptive_encoder_factory(
            hidden_dimensions=[32],
            output_dimension=8,
        )
        with patch.object(ProprioceptiveEncoder, "_apply_model_dtype") as mock_apply:
            encoder._build_network(input_dimension=7)
        mock_apply.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "model_dtype, frozen, expected_dtype",
        [
            (None, False, torch.float32),
            ("32", False, torch.float32),
            ("bf16-mixed", True, torch.bfloat16),
            ("bf16-mixed", False, torch.float32),
        ],
    )
    def test_deferred_mlp_build_respects_model_dtype(
        self,
        proprioceptive_encoder_factory: Callable[..., ProprioceptiveEncoder],
        proprioceptive_input_factory: Callable[..., dict[str, torch.Tensor]],
        model_dtype: str | None,
        frozen: bool,
        expected_dtype: torch.dtype,
    ):
        encoder = proprioceptive_encoder_factory(
            hidden_dimensions=[32, 16],
            output_dimension=8,
            frozen=frozen,
            model_dtype=model_dtype,
        )
        assert encoder.network is None
        inputs = proprioceptive_input_factory(
            keys=["proprio_robot_frame"], input_dimension=7
        )
        inputs["proprio_robot_frame"] = inputs["proprio_robot_frame"].to(expected_dtype)
        encoder(inputs)
        assert encoder.network is not None
        for parameter in encoder.parameters():
            assert parameter.dtype == expected_dtype
