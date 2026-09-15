"""Module selection and conversion metadata for quantization workflows."""

from dataclasses import fields, is_dataclass

import torch.nn as nn
from torchao.quantization import IntxWeightOnlyConfig
from torchao.quantization.quant_api import AOBaseConfig

from versatil.quantization.constants import QuantizationModuleType
from versatil.quantization.metadata import (
    QuantizationTargetMetadata,
    QuantizedLayerMetadata,
)
from versatil.quantization.pt2e.backends.base import BasePT2EBackend
from versatil.quantization.schemas.base import QuantizationSchema
from versatil.quantization.schemas.direct import DirectQuantizationSchema


class QuantizationModuleTarget:
    """Base class for a quantized policy submodule target."""

    def __init__(self, module_path: str) -> None:
        """Initialize the target.

        Args:
            module_path: Dotted path to the target module, or ``""`` for root.
        """
        self.module_path = module_path

    @property
    def label(self) -> str:
        """Return a readable module label for logs and errors.

        Returns:
            ``module_path`` for submodule targets, or ``"(root)"`` for the
            full-policy target.
        """
        return self.module_path or "(root)"

    def contains_module(self, module_name: str) -> bool:
        """Return whether a named module is inside this target.

        Args:
            module_name: Fully qualified module name from ``named_modules()``.

        Returns:
            Whether ``module_name`` is the target module itself or a child of
            the target module. The root target contains every module.
        """
        if self.module_path == "":
            return True
        return module_name == self.module_path or module_name.startswith(
            self.module_path + "."
        )

    def overlaps(self, other: "QuantizationModuleTarget") -> bool:
        """Return whether two targets can select the same submodule.

        Args:
            other: Target to compare against this target.

        Returns:
            Whether either target is root, both targets are the same path, or
            one target is nested under the other.
        """
        if self.module_path == "" or other.module_path == "":
            return True
        return (
            self.module_path == other.module_path
            or self.module_path.startswith(other.module_path + ".")
            or other.module_path.startswith(self.module_path + ".")
        )


class EagerQuantizationModuleTarget(QuantizationModuleTarget):
    """Select layers by scope and type and associate them with a schema."""

    def __init__(
        self,
        module_path: str,
        quantize_config: AOBaseConfig | None = None,
        schema: QuantizationSchema | None = None,
        module_type: str = QuantizationModuleType.LINEAR.value,
    ) -> None:
        """Initialize an eager quantization target.

        Args:
            module_path: Dotted path to the target module, or ``""`` for root.
            quantize_config: TorchAO base configuration for direct PTQ or QAT.
            schema: TorchAO preparation and conversion settings, including
                calibration and weight-format requirements.
                Specify this or quantize_config.
            module_type: Layer type selected within the module scope.

        Raises:
            ValueError: If configuration forms conflict, both are absent, or
                the module type and quantization configuration are incompatible.
        """
        super().__init__(module_path=module_path)
        self.module_type = QuantizationModuleType(module_type)
        if (quantize_config is None) == (schema is None):
            raise ValueError("Specify exactly one of quantize_config or schema.")
        self.schema = (
            schema
            if schema is not None
            else DirectQuantizationSchema(base_config=quantize_config)
        )
        if self.module_type == QuantizationModuleType.EMBEDDING and not isinstance(
            self.quantize_config, IntxWeightOnlyConfig
        ):
            raise ValueError(
                f"Target '{self.label}': embedding quantization requires IntxWeightOnlyConfig."
            )

    @property
    def quantize_config(self) -> AOBaseConfig:
        """Return the schema's base weight and activation quantization configuration."""
        return self.schema.base_config

    def overlaps(self, other: QuantizationModuleTarget) -> bool:
        """Return whether the targets can select the same layers."""
        if (
            isinstance(other, EagerQuantizationModuleTarget)
            and self.module_type != other.module_type
        ):
            return False
        return super().overlaps(other=other)

    def select_modules(
        self, model: nn.Module, auto_filter_incompatible: bool
    ) -> tuple[list[str], dict[str, str]]:
        """Resolve eligible layers within this target's module scope and type.

        Args:
            model: Initialized model whose module tree will be inspected.
            auto_filter_incompatible: Skip layers whose weight row widths
                are incompatible with the schema's group size. False raises an
                error for the first incompatible layer.

        Returns:
            Selected fully qualified layer names and excluded names mapped to
            their dimension mismatch reasons.

        Raises:
            ValueError: If the group size is nonpositive, a dimension mismatch
                requires filtering, or the selection contains no eligible layers.
        """
        group_size = self.schema.weight_group_size
        if group_size is not None and group_size <= 0:
            raise ValueError(
                f"Target '{self.label}' has invalid group_size {group_size}."
            )
        selected: list[str] = []
        skipped: dict[str, str] = {}
        layer_type = (
            nn.Embedding
            if self.module_type == QuantizationModuleType.EMBEDDING
            else nn.Linear
        )
        for name, module in model.named_modules():
            if not isinstance(module, layer_type):
                continue
            if not self.contains_module(module_name=name):
                continue
            dimension = (
                module.embedding_dim
                if self.module_type == QuantizationModuleType.EMBEDDING
                else module.in_features
            )
            if group_size is not None and dimension % group_size != 0:
                reason = (
                    f"Weight row width {dimension} requires divisibility by "
                    f"group_size {group_size}"
                )
                if not auto_filter_incompatible:
                    raise ValueError(f"Module '{name}': {reason}.")
                skipped[name] = reason
                continue
            selected.append(name)
        if not selected:
            raise ValueError(
                f"Target '{self.label}' selects zero eligible {self.module_type} modules; "
                f"skipped modules: {skipped}."
            )
        return selected, skipped

    def build_metadata(
        self, model: nn.Module, module_names: set[str], skipped: dict[str, str]
    ) -> QuantizationTargetMetadata:
        """Record settings and weight properties for an existing layer selection.

        Args:
            model: Model containing the selected weights immediately before
                conversion, after any observer calibration.
            module_names: Fully qualified names previously selected for conversion.
            skipped: Excluded layer names mapped to their selection reasons.

        Returns:
            Target settings and pre-conversion layer properties.

        Note:
            The workflow records resulting weight classes after conversion.
        """
        config = self.quantize_config
        selected = []
        for name in sorted(module_names):
            module = model.get_submodule(name)
            selected.append(
                QuantizedLayerMetadata(
                    name=name,
                    module_type=self.module_type,
                    weight_shape=tuple(module.weight.shape),
                    device=str(module.weight.device),
                    dtype=str(module.weight.dtype),
                )
            )
        return QuantizationTargetMetadata(
            module_path=self.module_path,
            base_config=f"{type(config).__module__}.{type(config).__qualname__}",
            base_config_parameters={
                parameter.name: str(getattr(config, parameter.name))
                for parameter in fields(config)
            }
            if is_dataclass(config)
            else {},
            schema=f"{type(self.schema).__module__}.{type(self.schema).__qualname__}",
            schema_parameters=self.schema.parameters,
            requires_calibration=self.schema.needs_calibration,
            group_size=self.schema.weight_group_size,
            selected=selected,
            skipped=skipped.copy(),
        )


class PT2EQuantizationModuleTarget(QuantizationModuleTarget):
    """Target using a PyTorch 2 Export backend quantizer config."""

    def __init__(
        self,
        module_path: str,
        pt2e_backend: BasePT2EBackend,
    ) -> None:
        """Initialize a PT2E quantization target.

        Args:
            module_path: Dotted path to the target module, or ``""`` for root.
            pt2e_backend: PT2E backend that creates the quantizer for this
                target.
        """
        super().__init__(module_path=module_path)
        self.pt2e_backend = pt2e_backend

    @property
    def needs_calibration(self) -> bool:
        """Return whether this target requires calibration batches.

        Returns:
            ``True`` for static PT2E backends and ``False`` for dynamic PT2E
            backends.
        """
        return not self.pt2e_backend.is_dynamic
