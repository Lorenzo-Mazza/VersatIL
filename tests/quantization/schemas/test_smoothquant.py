"""Tests for versatil.quantization.schemas.smoothquant module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep

from versatil.quantization.schemas.smoothquant import SmoothQuantSchema

SMOOTHQUANT_SCHEMA_MODULE = "versatil.quantization.schemas.smoothquant"


@pytest.fixture
def variable_length_linear_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[nn.Module, list[torch.Tensor]]]:
    def factory(
        input_shapes: list[tuple[int, ...]],
        device: torch.device,
    ) -> tuple[nn.Module, list[torch.Tensor]]:
        model = nn.Sequential(
            nn.Linear(in_features=32, out_features=16, bias=False)
        ).eval()
        weight = torch.from_numpy(
            rng.standard_normal((16, 32)).astype(np.float32)
        )  # (output_channels, input_channels)
        with torch.no_grad():
            model[0].weight.copy_(weight)  # (output_channels, input_channels)
        model.to(device=device)
        inputs = [
            torch.from_numpy(rng.standard_normal(shape).astype(np.float32)).to(
                device=device
            )  # (..., input_channels)
            for shape in input_shapes
        ]
        return model, inputs

    return factory


@pytest.mark.integration
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
@pytest.mark.parametrize(
    "input_shapes,device",
    [
        pytest.param([(2, 3, 32), (2, 1, 32), (2, 32)], torch.device("cpu"), id="cpu"),
        pytest.param(
            [(2, 16, 32), (2, 24, 32), (32, 32)],
            torch.device("cuda"),
            marks=pytest.mark.requires_gpu,
            id="cuda",
        ),
    ],
)
def test_variable_length_calibration_preserves_channel_maxima_and_smoothing_factors(
    variable_length_linear_factory: Callable[..., tuple[nn.Module, list[torch.Tensor]]],
    alpha: float,
    input_shapes: list[tuple[int, ...]],
    device: torch.device,
) -> None:
    model, inputs = variable_length_linear_factory(
        input_shapes=input_shapes, device=device
    )
    schema = SmoothQuantSchema(
        base_config=Int8DynamicActivationInt8WeightConfig(version=2), alpha=alpha
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
    quantize_(model=model, config=schema.conversion_config(is_qat=False))
    torch.testing.assert_close(model[0].weight.act_pre_scale, expected_scale)
    with torch.no_grad():
        for activation in inputs:
            output = model(
                activation
            )  # (..., input_channels) -> (..., output_channels)
            assert output.shape == (*activation.shape[:-1], 16)
            assert torch.isfinite(output).all()


@pytest.mark.unit
class TestSmoothQuantSchema:
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

    @pytest.mark.parametrize("version, weight_only_decode", [(1, False), (2, True)])
    def test_rejects_base_configurations_without_the_supported_w8a8_format(
        self,
        smoothquant_base_config_factory: Callable[..., MagicMock],
        version: int,
        weight_only_decode: bool,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "SmoothQuantSchema requires Int8DynamicActivationInt8WeightConfig "
                "with version=2 and weight_only_decode=False."
            ),
        ):
            SmoothQuantSchema(
                base_config=smoothquant_base_config_factory(
                    version=version, weight_only_decode=weight_only_decode
                ),
                alpha=0.5,
            )

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
