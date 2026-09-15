"""Data recorded for quantized targets and their selected layers."""

from dataclasses import dataclass, field

from versatil.quantization.constants import QuantizationModuleType


@dataclass
class QuantizedLayerMetadata:
    """Record a selected layer's properties before weight conversion.

    Attributes:
        name: Fully qualified layer name within the policy.
        module_type: Linear or embedding layer.
        weight_shape: Weight matrix rows and columns before conversion.
        device: Weight device at conversion, including its index when specified.
        dtype: Floating-point weight dtype before conversion.
    """

    name: str
    module_type: QuantizationModuleType
    weight_shape: tuple[int, int]
    device: str
    dtype: str


@dataclass
class QuantizationTargetMetadata:
    """Record conversion settings, layer selection and resulting weight types.

    Attributes:
        module_path: Configured submodule path, or an empty string for the policy.
        base_config: Fully qualified TorchAO numerical configuration class.
        base_config_parameters: Base configuration fields represented as strings.
        schema: Fully qualified quantization schema class.
        schema_parameters: Schema settings represented as strings.
        requires_calibration: Whether the schema requires representative inference.
        group_size: Number of input-channel weights sharing a scale, or None for
            configurations with another granularity.
        selected: Selected layers and their pre-conversion properties.
        skipped: Excluded layer names mapped to their exclusion reasons.
        weight_representations: Selected layer names mapped to the weight tensor
            class names observed after conversion.
    """

    module_path: str
    base_config: str
    base_config_parameters: dict[str, str]
    schema: str
    schema_parameters: dict[str, str]
    requires_calibration: bool
    group_size: int | None
    selected: list[QuantizedLayerMetadata]
    skipped: dict[str, str]
    weight_representations: dict[str, str] = field(default_factory=dict)
