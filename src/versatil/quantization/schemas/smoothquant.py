"""Schema for SmoothQuant calibration, rescaling and weight conversion."""

import torch
import torch.nn as nn
from torchao.core.config import AOBaseConfig
from torchao.prototype.smoothquant.api import SmoothQuantConfig
from torchao.prototype.smoothquant.core import SmoothQuantObservedLinear
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
)
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep
from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
    Int4PackingFormat,
)

from versatil.quantization.schemas.base import QuantizationSchema


class SmoothQuantSchema(QuantizationSchema):
    """Supply SmoothQuant configurations and validate observed linear inputs.

    Note:
        Smoothing rescales weights using calibrated input-channel maxima. Inference
        must apply the reciprocal scales to inputs to preserve the linear operation,
        so the converted weight format must store and apply those scales. Supported
        formats are dynamic W8A8 and tile-packed INT4 with BF16 activations.
    """

    def __init__(self, base_config: AOBaseConfig, alpha: float = 0.5) -> None:
        """Configure smoothing and the final quantization representation.

        Args:
            base_config: Version 2 Int8DynamicActivationInt8WeightConfig with
                weight_only_decode=False, or version 2 Int4WeightOnlyConfig with
                TILE_PACKED_TO_4D packing.
            alpha: Exponent between zero and one controlling how activation and
                weight channel maxima determine the smoothing factors.

        Raises:
            ValueError: If alpha is outside its range or the base configuration
                uses an unsupported weight format.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"SmoothQuant alpha must be between 0 and 1, got {alpha}.")
        dynamic_int8 = (
            isinstance(base_config, Int8DynamicActivationInt8WeightConfig)
            and base_config.version == 2
            and not base_config.weight_only_decode
        )
        tile_packed_int4 = (
            isinstance(base_config, Int4WeightOnlyConfig)
            and base_config.version == 2
            and base_config.int4_packing_format == Int4PackingFormat.TILE_PACKED_TO_4D
        )
        if not (dynamic_int8 or tile_packed_int4):
            raise ValueError(
                "SmoothQuantSchema requires version=2 with either "
                "Int8DynamicActivationInt8WeightConfig(weight_only_decode=False) or "
                "Int4WeightOnlyConfig(int4_packing_format='tile_packed_to_4d')."
            )
        super().__init__(base_config=base_config)
        self.alpha = alpha

    @property
    def needs_calibration(self) -> bool:
        """Return True because smoothing factors depend on observed activations."""
        return True

    @property
    def parameters(self) -> dict[str, str]:
        """Return the configured smoothing exponent for compression metadata."""
        return {"alpha": str(self.alpha)}

    def preparation_config(self, is_qat: bool) -> AOBaseConfig:
        """Return the configuration that inserts SmoothQuant observers.

        Args:
            is_qat: Whether the workflow requests QAT preparation.

        Returns:
            TorchAO's SmoothQuant preparation configuration.

        Raises:
            ValueError: If QAT is requested; this schema implements the PTQ
                lifecycle provided by TorchAO's SmoothQuant prototype.
        """
        self._validate_mode(is_qat=is_qat)
        return SmoothQuantConfig(
            base_config=self.base_config,
            step=QuantizationStep.PREPARE,
            alpha=self.alpha,
            use_running_absmax=True,
        )

    def conversion_config(self, is_qat: bool) -> AOBaseConfig:
        """Return the configuration that smooths and quantizes observed linears.

        Args:
            is_qat: Whether the workflow requests QAT conversion.

        Returns:
            TorchAO's SmoothQuant conversion configuration.

        Raises:
            ValueError: If QAT is requested for this PTQ schema.
        """
        self._validate_mode(is_qat=is_qat)
        return SmoothQuantConfig(
            base_config=self.base_config,
            step=QuantizationStep.CONVERT,
            alpha=self.alpha,
            use_running_absmax=True,
        )

    def validate_calibration(self, model: nn.Module, module_names: set[str]) -> None:
        """Check that each selected linear layer received calibration inputs.

        Args:
            model: Model containing the prepared SmoothQuant linears.
            module_names: Layer names selected before observers were inserted.

        Raises:
            ValueError: If a layer was not prepared or was never executed during
                calibration.

        Note:
            The workflow converts weights after every target passes this check.
            TorchAO's running observer reduces each input to channel maxima during
            calibration. Inputs with different token lengths contribute to the same
            channel statistics.
        """
        for name in sorted(module_names):
            module = model.get_submodule(name)
            if not isinstance(module, SmoothQuantObservedLinear):
                raise ValueError(f"SmoothQuant layer '{name}' was not prepared.")
            if module.obs.calibration_count == 0:
                raise ValueError(
                    f"SmoothQuant layer '{name}' received no calibration inputs. "
                    "Use observations that execute this layer or exclude it from the target."
                )

    def move_calibration_state(
        self, model: nn.Module, module_names: set[str], device: torch.device
    ) -> None:
        """Align SmoothQuant's observer device and weight reference with conversion.

        Args:
            model: Prepared policy after its tensors move to the destination.
            module_names: Fully qualified names of observed linear layers.
            device: Device for smoothing and final weight packing.

        Note:
            TorchAO keeps running statistics on CPU and copies them to the observer's
            stored device when calculating smoothing factors. That device attribute
            requires an explicit update after ``model.to()``.
        """
        for name in module_names:
            module = model.get_submodule(name)
            module.obs.device = device
            module.obs.weight = module.weight

    @staticmethod
    def _validate_mode(is_qat: bool) -> None:
        """Require the supported post-training conversion mode."""
        if is_qat:
            raise ValueError(
                "SmoothQuantSchema requires is_qat=False for its supported "
                "post-training preparation and conversion."
            )
