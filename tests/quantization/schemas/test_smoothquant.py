"""Tests for versatil.quantization.schemas.smoothquant module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn
from torchao.core.config import AOBaseConfig
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
    quantize_,
)
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep
from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
    Int4PackingFormat,
)

from versatil.quantization.schemas.smoothquant import SmoothQuantSchema

SMOOTHQUANT_SCHEMA_MODULE = "versatil.quantization.schemas.smoothquant"


@pytest.fixture
def variable_length_linear_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[nn.Module, list[torch.Tensor]]]:
    def factory(
        input_shapes: list[tuple[int, ...]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[nn.Module, list[torch.Tensor]]:
        model = nn.Sequential(
            nn.Linear(in_features=32, out_features=16, bias=False)
        ).eval()
        weight = torch.from_numpy(
            rng.standard_normal((16, 32)).astype(np.float32)
        )  # (output_channels, input_channels)
        with torch.no_grad():
            model[0].weight.copy_(weight)  # (output_channels, input_channels)
        model.to(device=device, dtype=dtype)
        inputs = [
            torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(
                device=device, dtype=dtype
            )  # (..., input_channels)
            for shape in input_shapes
        ]
        return model, inputs

    return factory


@pytest.mark.integration
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "input_shapes, device, deployment_device, weight_bits, dtype",
    [
        pytest.param(
            [(2, 3, 32), (2, 1, 32), (2, 32)],
            torch.device("cpu"),
            torch.device("cpu"),
            8,
            torch.float32,
            id="cpu",
        ),
        pytest.param(
            [(2, 16, 32), (2, 24, 32), (32, 32)],
            torch.device("cuda"),
            torch.device("cuda"),
            8,
            torch.float32,
            marks=pytest.mark.requires_gpu,
            id="cuda",
        ),
        pytest.param(
            [(2, 16, 32), (2, 24, 32), (32, 32)],
            torch.device("cuda"),
            torch.device("cpu"),
            8,
            torch.float32,
            marks=pytest.mark.requires_gpu,
            id="cuda_to_cpu",
        ),
        pytest.param(
            [(2, 3, 32), (2, 1, 32), (2, 32)],
            torch.device("cuda"),
            torch.device("cuda"),
            4,
            torch.bfloat16,
            marks=pytest.mark.requires_gpu,
            id="int4_cuda",
        ),
        pytest.param(
            [(2, 3, 32), (2, 1, 32), (2, 32)],
            torch.device("cpu"),
            torch.device("cuda"),
            4,
            torch.bfloat16,
            marks=pytest.mark.requires_gpu,
            id="int4_cpu_to_cuda",
        ),
    ],
)
def test_variable_length_calibration_preserves_channel_maxima_and_smoothing_factors(
    variable_length_linear_factory: Callable[..., tuple[nn.Module, list[torch.Tensor]]],
    alpha: float,
    input_shapes: list[tuple[int, ...]],
    device: torch.device,
    deployment_device: torch.device,
    weight_bits: int,
    dtype: torch.dtype,
) -> None:
    model, inputs = variable_length_linear_factory(
        input_shapes=input_shapes, device=device, dtype=dtype
    )
    schema = SmoothQuantSchema(
        base_config=(
            Int4WeightOnlyConfig(
                version=2,
                group_size=32,
                int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
            )
            if weight_bits == 4
            else Int8DynamicActivationInt8WeightConfig(
                version=2, weight_only_decode=False
            )
        ),
        alpha=alpha,
    )
    flat_inputs = [
        activation.reshape(-1, 32) for activation in inputs
    ]  # (..., channels) -> (tokens, channels)
    activation_maxima = (
        torch.cat(flat_inputs, dim=0).abs().amax(dim=0)
    )  # (total_tokens, channels) -> (channels,)
    weight_maxima = (
        model[0].weight.detach().abs().amax(dim=0)
    )  # (output_channels, channels) -> (channels,)
    epsilon = torch.finfo(torch.float32).eps
    expected_scale = (weight_maxima + epsilon).pow(1 - alpha) / (
        activation_maxima + epsilon
    ).pow(alpha)  # (channels,)
    quantize_(model=model, config=schema.preparation_config(is_qat=False))
    with torch.no_grad():
        for activation in inputs:
            model(activation)  # (..., input_channels) -> (..., output_channels)
            assert model[0].obs.x_abs_max.numel() == 32
    stored_maxima = model[0].obs.x_abs_max.to(device=device)  # (channels,)
    torch.testing.assert_close(stored_maxima, activation_maxima)
    assert model[0].obs.calibration_count == len(inputs)
    schema.validate_calibration(model=model, module_names={"0"})
    model.to(device=deployment_device)
    schema.move_calibration_state(
        model=model, module_names={"0"}, device=deployment_device
    )
    quantize_(model=model, config=schema.conversion_config(is_qat=False))
    torch.testing.assert_close(
        model[0].weight.act_pre_scale, expected_scale.to(device=deployment_device)
    )
    with torch.no_grad():
        for activation in inputs:
            output = model(
                activation.to(device=deployment_device)
            )  # (..., input_channels) -> (..., output_channels)
            assert output.shape == (*activation.shape[:-1], 16)
            assert torch.isfinite(output).all()
            assert output.device.type == deployment_device.type


@pytest.mark.unit
class TestSmoothQuantSchema:
    @pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda:1")])
    def test_observer_uses_destination_device_and_current_weight_without_moving_statistics(
        self,
        smoothquant_base_config_factory: Callable[..., MagicMock],
        observed_model_factory: Callable[..., MagicMock],
        device: torch.device,
    ) -> None:
        schema = SmoothQuantSchema(
            base_config=smoothquant_base_config_factory(
                version=2, weight_only_decode=False
            ),
            alpha=0.5,
        )
        model = observed_model_factory(prepared=True, observed=True)
        layer = model.get_submodule.return_value
        layer.weight = MagicMock(spec=torch.Tensor)
        layer.weight.device = device
        layer.obs.weight = MagicMock(spec=torch.Tensor)
        layer.obs.device = torch.device("cuda:0")
        statistics = MagicMock(spec=torch.Tensor)
        statistics.device = torch.device("cpu")
        layer.obs.x_abs_max = statistics

        schema.move_calibration_state(
            model=model, module_names={"decoder.projection"}, device=device
        )

        model.get_submodule.assert_called_once_with("decoder.projection")
        assert layer.obs.device == device
        assert layer.obs.weight is layer.weight
        assert layer.obs.x_abs_max is statistics
        assert layer.obs.x_abs_max.device.type == "cpu"
        statistics.to.assert_not_called()
        assert layer.obs.calibration_count == 2

    @pytest.mark.parametrize("alpha", [-0.1, 0.0, 0.5, 1.0, 1.1, float("nan")])
    def test_smoothing_exponent_must_be_in_closed_unit_interval(
        self, smoothquant_base_config_factory: Callable[..., MagicMock], alpha: float
    ) -> None:
        expectation = (
            does_not_raise()
            if 0.0 <= alpha <= 1.0
            else pytest.raises(
                ValueError,
                match=re.escape(
                    f"SmoothQuant alpha must be between 0 and 1, got {alpha}."
                ),
            )
        )
        with expectation:
            schema = SmoothQuantSchema(
                base_config=smoothquant_base_config_factory(
                    version=2, weight_only_decode=False
                ),
                alpha=alpha,
            )
            assert schema.parameters == {"alpha": str(alpha)}
            assert schema.needs_calibration is True

    @pytest.mark.parametrize(
        "config_type, version, weight_only_decode, packing_format, supported",
        [
            (Int8DynamicActivationInt8WeightConfig, 2, False, None, True),
            (Int8DynamicActivationInt8WeightConfig, 1, False, None, False),
            (Int8DynamicActivationInt8WeightConfig, 2, True, None, False),
            (Int4WeightOnlyConfig, 2, False, Int4PackingFormat.TILE_PACKED_TO_4D, True),
            (
                Int4WeightOnlyConfig,
                1,
                False,
                Int4PackingFormat.TILE_PACKED_TO_4D,
                False,
            ),
            (Int4WeightOnlyConfig, 2, False, Int4PackingFormat.PLAIN, False),
            (Int4WeightOnlyConfig, 2, False, Int4PackingFormat.PLAIN_INT32, False),
            (Int4WeightOnlyConfig, 2, False, Int4PackingFormat.PRESHUFFLED, False),
            (Int8WeightOnlyConfig, 2, False, None, False),
        ],
    )
    def test_accepts_supported_formats_before_initializing_base_schema(
        self,
        schema_base_config_factory: Callable[..., MagicMock],
        config_type: type[AOBaseConfig],
        version: int,
        weight_only_decode: bool,
        packing_format: Int4PackingFormat | None,
        supported: bool,
    ) -> None:
        config = schema_base_config_factory(config_type=config_type)
        config.version = version
        config.weight_only_decode = weight_only_decode
        config.int4_packing_format = packing_format
        expectation = (
            does_not_raise()
            if supported
            else pytest.raises(
                ValueError,
                match=re.escape(
                    "SmoothQuantSchema requires version=2 with either "
                    "Int8DynamicActivationInt8WeightConfig(weight_only_decode=False) or "
                    "Int4WeightOnlyConfig(int4_packing_format='tile_packed_to_4d')."
                ),
            )
        )
        with (
            patch(
                f"{SMOOTHQUANT_SCHEMA_MODULE}.QuantizationSchema.__init__",
                return_value=None,
            ) as initialize,
            expectation,
        ):
            SmoothQuantSchema(base_config=config, alpha=0.5)
        if supported:
            initialize.assert_called_once_with(base_config=config)
        else:
            initialize.assert_not_called()

    @pytest.mark.parametrize(
        "step", [QuantizationStep.PREPARE, QuantizationStep.CONVERT]
    )
    @pytest.mark.parametrize("is_qat", [False, True])
    def test_produces_matching_ptq_steps_and_rejects_qat(
        self,
        smoothquant_base_config_factory: Callable[..., MagicMock],
        step: QuantizationStep,
        is_qat: bool,
    ) -> None:
        config = smoothquant_base_config_factory(version=2, weight_only_decode=False)
        schema = SmoothQuantSchema(base_config=config, alpha=0.75)
        expectation = (
            pytest.raises(
                ValueError,
                match=re.escape(
                    "SmoothQuantSchema requires is_qat=False for its supported "
                    "post-training preparation and conversion."
                ),
            )
            if is_qat
            else does_not_raise()
        )
        with (
            patch(
                f"{SMOOTHQUANT_SCHEMA_MODULE}.SmoothQuantConfig"
            ) as smoothquant_config,
            expectation,
        ):
            result = (
                schema.preparation_config(is_qat=is_qat)
                if step == QuantizationStep.PREPARE
                else schema.conversion_config(is_qat=is_qat)
            )
            assert result is smoothquant_config.return_value
        if is_qat:
            smoothquant_config.assert_not_called()
        else:
            smoothquant_config.assert_called_once_with(
                base_config=config, step=step, alpha=0.75, use_running_absmax=True
            )

    @pytest.mark.parametrize(
        "prepared, observed", [(False, False), (True, False), (True, True)]
    )
    def test_requires_prepared_layers_with_observed_inputs(
        self,
        smoothquant_base_config_factory: Callable[..., MagicMock],
        observed_model_factory: Callable[..., MagicMock],
        prepared: bool,
        observed: bool,
    ) -> None:
        schema = SmoothQuantSchema(
            base_config=smoothquant_base_config_factory(
                version=2, weight_only_decode=False
            ),
            alpha=0.5,
        )
        model = observed_model_factory(prepared=prepared, observed=observed)
        message = (
            "SmoothQuant layer 'decoder.projection' was not prepared."
            if not prepared
            else "SmoothQuant layer 'decoder.projection' received no calibration inputs. "
            "Use observations that execute this layer or exclude it from the target."
        )
        expectation = (
            does_not_raise()
            if prepared and observed
            else pytest.raises(ValueError, match=re.escape(message))
        )
        with expectation:
            schema.validate_calibration(
                model=model, module_names={"decoder.projection"}
            )
        model.get_submodule.assert_called_once_with("decoder.projection")
        if observed:
            assert model.get_submodule.return_value.obs.calibration_count == 2
