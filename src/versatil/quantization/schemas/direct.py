"""Schema for direct TorchAO weight conversion and quantization-aware training."""

import torch.nn as nn
from torchao.core.config import AOBaseConfig
from torchao.quantization.qat import QATConfig
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep

from versatil.quantization.schemas.base import QuantizationSchema


class DirectQuantizationSchema(QuantizationSchema):
    """Configure PTQ conversion from weights, or preparation and conversion for QAT.

    Note:
        For PTQ, this schema returns ``base_config`` as the conversion configuration.
        The workflow passes it to ``quantize_()``, which computes quantization
        parameters from the selected layers' weights and converts those weights
        in one call. For QAT, the schema wraps ``base_config`` in ``QATConfig``:
        preparation inserts layers that simulate quantization during training,
        and conversion replaces the trained layers with their quantized equivalents.
    """

    @property
    def needs_calibration(self) -> bool:
        """Return False for direct PTQ and QAT checkpoint conversion."""
        return False

    def preparation_config(self, is_qat: bool) -> AOBaseConfig | None:
        """Return the configuration for inserting fake-quantization modules.

        Args:
            is_qat: Whether the workflow prepares weights for quantization-aware
                training and subsequent checkpoint restoration.

        Returns:
            A matching QAT preparation configuration, or None for direct PTQ.
        """
        if is_qat:
            return QATConfig(
                base_config=self.base_config, step=QuantizationStep.PREPARE.value
            )
        return None

    def conversion_config(self, is_qat: bool) -> AOBaseConfig:
        """Return the configuration that produces the quantized weights.

        Args:
            is_qat: Whether to convert modules prepared for QAT with this schema.

        Returns:
            The base PTQ configuration or its matching QAT conversion configuration.
        """
        if is_qat:
            return QATConfig(
                base_config=self.base_config, step=QuantizationStep.CONVERT.value
            )
        return self.base_config

    def validate_calibration(self, model: nn.Module, module_names: set[str]) -> None:
        """Return immediately for weight-based PTQ and prepared QAT conversion.

        Args:
            model: Model containing the selected layers.
            module_names: Fully qualified layer names selected for conversion.
        """
