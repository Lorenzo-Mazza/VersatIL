"""Base schema for TorchAO quantization settings and requirements."""

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torchao.core.config import AOBaseConfig
from torchao.prototype.smoothquant.api import SmoothQuantConfig
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8DynamicActivationIntxWeightConfig,
    Int8WeightOnlyConfig,
    IntxWeightOnlyConfig,
)
from torchao.quantization.granularity import PerGroup
from torchao.quantization.qat import QATConfig
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
    Int4PackingFormat,
)

logger = logging.getLogger(__name__)


class QuantizationSchema(ABC):
    """Define preparation and conversion configurations for TorchAO quantization.

    Note:
        Preparation adds observers that collect activation statistics, or layers
        that simulate quantization during training. Conversion creates quantized
        weights using the selected TorchAO configuration. The workflow executes
        these operations; the schema supplies their configurations and checks
        weight properties and any required calibration statistics.
    """

    def __init__(self, base_config: AOBaseConfig) -> None:
        """Store the configuration that determines the converted tensor format.

        Args:
            base_config: TorchAO weight and activation quantization settings.

        Raises:
            ValueError: If a SmoothQuantConfig or QATConfig is passed as the base
                configuration. These require the corresponding schema or QAT mode.
        """
        if isinstance(base_config, (SmoothQuantConfig, QATConfig)):
            raise ValueError(
                "Use a base quantization config here. Configure SmoothQuant with "
                "SmoothQuantSchema, or QAT with the workflow's is_qat=True setting."
            )
        self.base_config = base_config

    @property
    @abstractmethod
    def needs_calibration(self) -> bool:
        """Return whether conversion needs statistics from representative inputs."""

    @property
    def parameters(self) -> dict[str, str]:
        """Return schema settings recorded separately from the base configuration."""
        return {}

    @property
    def weight_group_size(self) -> int | None:
        """Return the configured number of input-channel weights sharing a scale.

        Returns:
            Group size from the configuration or its granularity, or None when
            the configuration uses another weight granularity.
        """
        group_size = getattr(self.base_config, "group_size", None)
        if isinstance(group_size, int):
            return group_size
        for attribute_name in ("weight_granularity", "granularity"):
            granularity = getattr(self.base_config, attribute_name, None)
            if isinstance(granularity, PerGroup):
                return granularity.group_size
        return None

    def validate_configuration(
        self,
        model: nn.Module,
        module_names: set[str],
        label: str,
        activation_dtype: torch.dtype | None = None,
        for_conversion: bool = True,
    ) -> None:
        """Check TorchAO settings and selected weights against format requirements.

        Args:
            model: Model containing the selected layers.
            module_names: Fully qualified names resolved by the module target.
            label: Target path or readable root label for diagnostics.
            activation_dtype: Known input dtype for the converted linear layers.
            for_conversion: Apply conversion-device and inference-dtype checks.
                QAT preparation uses the numerical configuration checks.

        Raises:
            ValueError: If a numerical setting, weight device or dtype conflicts
                with the selected TorchAO representation.

        Note:
            Additional configurations produce a log warning. Their supported
            operations require export and runtime validation.
        """
        config = self.base_config
        group_size = self.weight_group_size
        if group_size is not None and group_size <= 0:
            raise ValueError(f"Target '{label}' has invalid group_size {group_size}.")
        if not isinstance(
            config,
            (
                Int4WeightOnlyConfig,
                Int8WeightOnlyConfig,
                Int8DynamicActivationInt8WeightConfig,
                Int8DynamicActivationIntxWeightConfig,
                IntxWeightOnlyConfig,
            ),
        ):
            config_name = f"{type(config).__module__}.{type(config).__qualname__}"
            logger.warning(
                f"Target '{label}': configuration {config_name} is unverified."
            )
        if isinstance(config, Int4WeightOnlyConfig):
            if config.version != 2:
                raise ValueError(
                    f"Target '{label}': Int4WeightOnlyConfig requires version=2 in TorchAO 0.18."
                )
            if config.group_size not in (32, 64, 128, 256):
                raise ValueError(
                    f"Target '{label}': INT4 group_size must be 32, 64, 128 or 256."
                )
            if for_conversion and config.int4_packing_format in (
                Int4PackingFormat.PLAIN,
                Int4PackingFormat.TILE_PACKED_TO_4D,
            ):
                for name in sorted(module_names):
                    weight = model.get_submodule(name).weight
                    if weight.device.type != "cuda":
                        raise ValueError(
                            f"Module '{name}': {config.int4_packing_format} INT4 requires CUDA weights at conversion."
                        )
                    if (
                        config.int4_packing_format
                        == Int4PackingFormat.TILE_PACKED_TO_4D
                        and weight.dtype != torch.bfloat16
                    ):
                        raise ValueError(
                            f"Module '{name}': tile-packed INT4 requires bfloat16 weights, got {weight.dtype}."
                        )
                if activation_dtype is not None and activation_dtype != torch.bfloat16:
                    raise ValueError(
                        f"Target '{label}': {config.int4_packing_format} INT4 requires bfloat16 activations, got {activation_dtype}."
                    )
        if isinstance(config, Int8DynamicActivationIntxWeightConfig) and (
            config.version != 2 or config.act_mapping_type != MappingType.ASYMMETRIC
        ):
            raise ValueError(
                f"Target '{label}': Int8DynamicActivationIntxWeightConfig requires version=2 and asymmetric activations in TorchAO 0.18."
            )

    @abstractmethod
    def preparation_config(self, is_qat: bool) -> AOBaseConfig | None:
        """Return the configuration used to insert observers or fake quantization.

        Args:
            is_qat: Whether the workflow requests quantization-aware training.

        Returns:
            TorchAO preparation configuration, or None for direct PTQ conversion.
        """

    @abstractmethod
    def conversion_config(self, is_qat: bool) -> AOBaseConfig:
        """Return the configuration used to convert selected layers.

        Args:
            is_qat: Whether the workflow converts quantization-aware trained layers.

        Returns:
            TorchAO configuration for the requested conversion mode.
        """

    @abstractmethod
    def validate_calibration(self, model: nn.Module, module_names: set[str]) -> None:
        """Check the calibration state required for conversion.

        Args:
            model: Model containing the selected layers after preparation.
            module_names: Fully qualified layer names selected for conversion.

        Raises:
            ValueError: If a selected layer lacks required calibration state.
        """
