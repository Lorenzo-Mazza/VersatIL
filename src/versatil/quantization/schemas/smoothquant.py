"""Schema for SmoothQuant calibration, rescaling and weight conversion."""

import torch.nn as nn
from torchao.core.config import AOBaseConfig
from torchao.prototype.smoothquant.api import SmoothQuantConfig
from torchao.prototype.smoothquant.core import SmoothQuantObservedLinear
from torchao.quantization import Int8DynamicActivationInt8WeightConfig
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep

from versatil.quantization.schemas.base import QuantizationSchema


class SmoothQuantSchema(QuantizationSchema):
    """Supply SmoothQuant configurations and validate observed linear inputs.

    Note:
        The configurations select INT8 weights and dynamic INT8 activations.
        TorchAO's observers collect one absolute maximum per linear input channel.
        Its conversion rescales the weights and stores the corresponding activation
        scaling factors in the Int8Tensor representation.
    """

    def __init__(self, base_config: AOBaseConfig, alpha: float = 0.5) -> None:
        """Configure smoothing and the final quantization representation.

        Args:
            base_config: Int8DynamicActivationInt8WeightConfig with version 2,
                whose weight tensor supports activation channel scaling.
            alpha: Exponent between zero and one controlling how activation and
                weight channel maxima determine the smoothing factors.

        Raises:
            ValueError: If alpha is outside its range or the base configuration
                is outside this schema's supported dynamic W8A8 path.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"SmoothQuant alpha must be between 0 and 1, got {alpha}.")
        if not isinstance(base_config, Int8DynamicActivationInt8WeightConfig) or (
            base_config.version != 2 or base_config.weight_only_decode
        ):
            raise ValueError(
                "SmoothQuantSchema requires Int8DynamicActivationInt8WeightConfig "
                "with version=2 and weight_only_decode=False."
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

    @staticmethod
    def _validate_mode(is_qat: bool) -> None:
        """Require the supported post-training conversion mode."""
        if is_qat:
            raise ValueError(
                "SmoothQuantSchema requires is_qat=False for its supported "
                "post-training preparation and conversion."
            )
