"""Tests for versatil.quantization.schemas.base module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torchao.core.config import AOBaseConfig
from torchao.prototype.smoothquant.api import SmoothQuantConfig
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
)
from torchao.quantization.granularity import PerGroup
from torchao.quantization.qat import QATConfig
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
    Int4PackingFormat,
)

from versatil.quantization.schemas.base import QuantizationSchema

BASE_SCHEMA_MODULE = "versatil.quantization.schemas.base"


@pytest.fixture
def validation_schema_factory() -> Callable[..., MagicMock]:
    def factory(
        config_type: type[AOBaseConfig],
        group_size: int | None,
        version: int = 2,
        packing: Int4PackingFormat = Int4PackingFormat.TILE_PACKED_TO_4D,
        activation_mapping: MappingType = MappingType.ASYMMETRIC,
    ) -> MagicMock:
        config = MagicMock(spec=config_type)
        config.group_size = group_size
        config.version = version
        config.int4_packing_format = packing
        config.act_mapping_type = activation_mapping
        schema = MagicMock(spec=QuantizationSchema)
        schema.base_config = config
        schema.weight_group_size = group_size
        return schema

    return factory


@pytest.fixture
def weight_property_model_factory() -> Callable[..., MagicMock]:
    def factory(device: str, dtype: torch.dtype) -> MagicMock:
        layer = MagicMock(spec=nn.Linear)
        layer.weight = MagicMock(spec=torch.Tensor)
        layer.weight.device = torch.device(device)
        layer.weight.dtype = dtype
        model = MagicMock(spec=nn.Module)
        model.get_submodule.return_value = layer
        return model

    return factory


@pytest.fixture
def grouped_schema_factory() -> Callable[..., MagicMock]:
    def factory(
        group_size: int | None,
        weight_granularity: int | None,
        granularity: int | None,
    ) -> MagicMock:
        config = MagicMock(spec=AOBaseConfig)
        config.group_size = group_size
        for name, value in (
            ("weight_granularity", weight_granularity),
            ("granularity", granularity),
        ):
            group = MagicMock(spec=PerGroup) if value is not None else None
            if group is not None:
                group.group_size = value
            setattr(config, name, group)
        schema = MagicMock(spec=QuantizationSchema)
        schema.base_config = config
        return schema

    return factory


@pytest.mark.unit
def test_base_schema_requires_preparation_conversion_and_calibration_implementations(
    schema_base_config_factory: Callable[..., MagicMock],
) -> None:
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Can't instantiate abstract class QuantizationSchema without an "
            "implementation for abstract methods 'conversion_config', "
            "'needs_calibration', 'preparation_config', 'validate_calibration'"
        ),
    ):
        QuantizationSchema(
            base_config=schema_base_config_factory(config_type=AOBaseConfig)
        )


@pytest.mark.unit
@pytest.mark.parametrize("config_type", [SmoothQuantConfig, QATConfig])
def test_base_configuration_rejects_preparation_and_conversion_wrappers(
    validation_schema_factory: Callable[..., MagicMock],
    schema_base_config_factory: Callable[..., MagicMock],
    config_type: type[AOBaseConfig],
) -> None:
    schema = validation_schema_factory(config_type=AOBaseConfig, group_size=None)
    config = schema_base_config_factory(config_type=config_type)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Use a base quantization config here. Configure SmoothQuant with "
            "SmoothQuantSchema, or QAT with the workflow's is_qat=True setting."
        ),
    ):
        QuantizationSchema.__init__(schema, base_config=config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "group_size, weight_granularity, granularity, expected",
    [
        (64, 32, 128, 64),
        (None, 32, 128, 32),
        (None, None, 128, 128),
        (None, None, None, None),
    ],
)
def test_weight_group_size_resolves_configuration_and_granularity_fields(
    grouped_schema_factory: Callable[..., MagicMock],
    group_size: int | None,
    weight_granularity: int | None,
    granularity: int | None,
    expected: int | None,
) -> None:
    schema = grouped_schema_factory(
        group_size=group_size,
        weight_granularity=weight_granularity,
        granularity=granularity,
    )

    assert QuantizationSchema.weight_group_size.fget(schema) == expected


@pytest.mark.unit
class TestConfigurationValidation:
    @pytest.mark.parametrize("group_size", [0, -32])
    def test_nonpositive_group_size_fails_before_weight_inspection(
        self,
        validation_schema_factory: Callable[..., MagicMock],
        weight_property_model_factory: Callable[..., MagicMock],
        group_size: int,
    ) -> None:
        schema = validation_schema_factory(
            config_type=Int4WeightOnlyConfig, group_size=group_size
        )
        model = weight_property_model_factory(device="cpu", dtype=torch.float32)

        with pytest.raises(
            ValueError,
            match=re.escape(f"Target 'decoder' has invalid group_size {group_size}."),
        ):
            QuantizationSchema.validate_configuration(
                schema,
                model=model,
                module_names={"decoder.projection"},
                label="decoder",
            )

        model.get_submodule.assert_not_called()

    @pytest.mark.parametrize(
        "version, group_size, expected_error",
        [
            (
                1,
                32,
                "Target 'decoder': Int4WeightOnlyConfig requires version=2 in TorchAO 0.18.",
            ),
            (2, 16, "Target 'decoder': INT4 group_size must be 32, 64, 128 or 256."),
            (2, 32, None),
            (2, 64, None),
            (2, 128, None),
            (2, 256, None),
        ],
    )
    def test_int4_numerical_requirements_apply_during_qat_preparation(
        self,
        validation_schema_factory: Callable[..., MagicMock],
        weight_property_model_factory: Callable[..., MagicMock],
        version: int,
        group_size: int,
        expected_error: str | None,
    ) -> None:
        schema = validation_schema_factory(
            config_type=Int4WeightOnlyConfig, group_size=group_size, version=version
        )
        model = weight_property_model_factory(device="cpu", dtype=torch.float32)
        expectation = (
            does_not_raise()
            if expected_error is None
            else pytest.raises(ValueError, match=re.escape(expected_error))
        )

        with expectation:
            QuantizationSchema.validate_configuration(
                schema,
                model=model,
                module_names={"decoder.projection"},
                label="decoder",
                activation_dtype=torch.float32,
                for_conversion=False,
            )

        model.get_submodule.assert_not_called()

    @pytest.mark.parametrize(
        "packing, device, weight_dtype, activation_dtype, expected_error",
        [
            (
                Int4PackingFormat.PLAIN,
                "cpu",
                torch.bfloat16,
                None,
                "Module 'decoder.projection': Int4PackingFormat.PLAIN INT4 requires CUDA weights at conversion.",
            ),
            (Int4PackingFormat.PLAIN, "cuda", torch.float32, torch.bfloat16, None),
            (
                Int4PackingFormat.TILE_PACKED_TO_4D,
                "cpu",
                torch.bfloat16,
                torch.bfloat16,
                "Module 'decoder.projection': Int4PackingFormat.TILE_PACKED_TO_4D INT4 requires CUDA weights at conversion.",
            ),
            (
                Int4PackingFormat.TILE_PACKED_TO_4D,
                "cuda",
                torch.float32,
                torch.bfloat16,
                "Module 'decoder.projection': tile-packed INT4 requires bfloat16 weights, got torch.float32.",
            ),
            (
                Int4PackingFormat.TILE_PACKED_TO_4D,
                "cuda",
                torch.bfloat16,
                torch.float32,
                "Target 'decoder': Int4PackingFormat.TILE_PACKED_TO_4D INT4 requires bfloat16 activations, got torch.float32.",
            ),
            (
                Int4PackingFormat.TILE_PACKED_TO_4D,
                "cuda",
                torch.bfloat16,
                torch.bfloat16,
                None,
            ),
            (Int4PackingFormat.TILE_PACKED_TO_4D, "cuda", torch.bfloat16, None, None),
        ],
    )
    def test_int4_conversion_checks_weights_and_known_activation_dtype(
        self,
        validation_schema_factory: Callable[..., MagicMock],
        weight_property_model_factory: Callable[..., MagicMock],
        packing: Int4PackingFormat,
        device: str,
        weight_dtype: torch.dtype,
        activation_dtype: torch.dtype | None,
        expected_error: str | None,
    ) -> None:
        schema = validation_schema_factory(
            config_type=Int4WeightOnlyConfig, group_size=32, packing=packing
        )
        model = weight_property_model_factory(device=device, dtype=weight_dtype)
        expectation = (
            does_not_raise()
            if expected_error is None
            else pytest.raises(ValueError, match=re.escape(expected_error))
        )

        with expectation:
            QuantizationSchema.validate_configuration(
                schema,
                model=model,
                module_names={"decoder.projection"},
                label="decoder",
                activation_dtype=activation_dtype,
                for_conversion=True,
            )

        model.get_submodule.assert_called_once_with("decoder.projection")

    @pytest.mark.parametrize(
        "version, activation_mapping, valid",
        [
            (1, MappingType.ASYMMETRIC, False),
            (2, MappingType.SYMMETRIC, False),
            (2, MappingType.ASYMMETRIC, True),
        ],
    )
    def test_dynamic_intx_requires_supported_version_and_activation_mapping(
        self,
        validation_schema_factory: Callable[..., MagicMock],
        weight_property_model_factory: Callable[..., MagicMock],
        version: int,
        activation_mapping: MappingType,
        valid: bool,
    ) -> None:
        schema = validation_schema_factory(
            config_type=Int8DynamicActivationIntxWeightConfig,
            group_size=32,
            version=version,
            activation_mapping=activation_mapping,
        )
        model = weight_property_model_factory(device="cpu", dtype=torch.float32)
        expectation = (
            does_not_raise()
            if valid
            else pytest.raises(
                ValueError,
                match=re.escape(
                    "Target 'decoder': Int8DynamicActivationIntxWeightConfig requires version=2 and asymmetric activations in TorchAO 0.18."
                ),
            )
        )

        with expectation:
            QuantizationSchema.validate_configuration(
                schema,
                model=model,
                module_names={"decoder.projection"},
                label="decoder",
                for_conversion=True,
            )

        model.get_submodule.assert_not_called()

    def test_unknown_configuration_logs_the_configuration_class(
        self,
        validation_schema_factory: Callable[..., MagicMock],
        weight_property_model_factory: Callable[..., MagicMock],
    ) -> None:
        schema = validation_schema_factory(config_type=AOBaseConfig, group_size=None)
        model = weight_property_model_factory(device="cpu", dtype=torch.float32)
        config_type = type(schema.base_config)

        with patch(f"{BASE_SCHEMA_MODULE}.logger.warning") as warning:
            QuantizationSchema.validate_configuration(
                schema,
                model=model,
                module_names={"decoder.projection"},
                label="decoder",
            )

        warning.assert_called_once_with(
            f"Target 'decoder': configuration {config_type.__module__}.{config_type.__qualname__} is unverified."
        )
        model.get_submodule.assert_not_called()
