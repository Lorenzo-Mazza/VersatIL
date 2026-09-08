"""Tests for versatil.quantization.schemas.direct module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
from torchao.core.config import AOBaseConfig
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep

from versatil.quantization.schemas.direct import DirectQuantizationSchema

DIRECT_SCHEMA_MODULE = "versatil.quantization.schemas.direct"


@pytest.mark.unit
class TestDirectQuantizationSchema:
    def test_direct_ptq_reuses_base_config_without_preparation(
        self, schema_base_config_factory: Callable[..., MagicMock]
    ) -> None:
        config = schema_base_config_factory(config_type=AOBaseConfig)
        schema = DirectQuantizationSchema(base_config=config)
        assert schema.preparation_config(is_qat=False) is None
        assert schema.conversion_config(is_qat=False) is config
        assert schema.needs_calibration is False
        assert schema.parameters == {}

    def test_calibration_validation_accepts_unobserved_direct_conversion(
        self,
        schema_base_config_factory: Callable[..., MagicMock],
        observed_model_factory: Callable[..., MagicMock],
    ) -> None:
        schema = DirectQuantizationSchema(
            base_config=schema_base_config_factory(config_type=AOBaseConfig)
        )
        model = observed_model_factory(prepared=False, observed=False)

        schema.validate_calibration(model=model, module_names={"decoder.projection"})

        model.get_submodule.assert_not_called()

    @pytest.mark.parametrize(
        "step", [QuantizationStep.PREPARE, QuantizationStep.CONVERT]
    )
    def test_qat_uses_matching_base_config_for_both_steps(
        self,
        schema_base_config_factory: Callable[..., MagicMock],
        step: QuantizationStep,
    ) -> None:
        config = schema_base_config_factory(config_type=AOBaseConfig)
        schema = DirectQuantizationSchema(base_config=config)
        with patch(f"{DIRECT_SCHEMA_MODULE}.QATConfig") as qat_config:
            result = (
                schema.preparation_config(is_qat=True)
                if step == QuantizationStep.PREPARE
                else schema.conversion_config(is_qat=True)
            )
        qat_config.assert_called_once_with(base_config=config, step=step.value)
        assert result is qat_config.return_value
