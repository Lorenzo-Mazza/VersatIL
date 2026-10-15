"""Resolve eager quantization targets and validate their layer selections."""

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torchao.core.config import AOBaseConfig

from versatil.post_training_compression.deployment_backends.base import (
    DeploymentBackend,
)
from versatil.quantization.module_target import (
    EagerQuantizationModuleTarget,
    QuantizationModuleTarget,
)

logger = logging.getLogger(__name__)


@dataclass
class ResolvedQuantizationTarget:
    """Selected layer names and TorchAO configurations for a module target.

    Attributes:
        target: Layer scope and quantization schema used for preparation and conversion.
        module_names: Fully qualified layer names resolved before preparation.
        preparation_config: Observer or fake-quant configuration, or None when
            the schema supplies a direct PTQ conversion configuration.
        conversion_config: Configuration that produces the quantized weights.
        skipped: Excluded layer names and their dimension mismatch reasons.
    """

    target: EagerQuantizationModuleTarget
    module_names: set[str]
    preparation_config: AOBaseConfig | None
    conversion_config: AOBaseConfig
    skipped: dict[str, str] = field(default_factory=dict)


def validate_quantization_targets(
    model: nn.Module,
    targets: list[QuantizationModuleTarget],
) -> None:
    """Validate quantization target paths and overlap.

    Args:
        model: Policy model whose submodule tree should contain every target.
        targets: Module targets configured for one quantization workflow.

    Raises:
        ValueError: If a target path does not exist or two targets overlap.
    """
    for target in targets:
        if target.module_path == "":
            continue
        try:
            model.get_submodule(target.module_path)
        except AttributeError as error:
            available = list(dict(model.named_children()).keys())
            raise ValueError(
                f"Quantization target '{target.module_path}' not found in model. "
                f"Available top-level modules: {available}."
            ) from error
    for index, target in enumerate(targets):
        for other in targets[index + 1 :]:
            if target.overlaps(other=other):
                raise ValueError(
                    "Quantization targets overlap: "
                    f"'{target.module_path}' and '{other.module_path}'."
                )


def resolve_quantization_targets(
    model: nn.Module,
    targets: list[EagerQuantizationModuleTarget],
    is_qat: bool,
    auto_filter_incompatible_linears: bool,
    activation_dtype: torch.dtype | None = None,
    deployment_backend: DeploymentBackend | None = None,
    for_conversion: bool = True,
    conversion_device: torch.device | None = None,
) -> list[ResolvedQuantizationTarget]:
    """Resolve layer names and validate every target before quantization mutation.

    Args:
        model: Policy or module containing the floating-point layers.
        targets: Configured module scopes and quantization schemas.
        is_qat: Select fake-quant preparation and conversion for QAT.
        auto_filter_incompatible_linears: Skip layers whose weight row widths
            conflict with the configured group size.
        activation_dtype: Known dtype of inputs to the selected linear layers.
        deployment_backend: Backend selected for the exported artifact.
        for_conversion: Apply weight-device and dtype requirements for the
            conversion. QAT preparation checks configuration and dimensions.
        conversion_device: Destination for final weight conversion and packing.
            None uses each selected weight's current device for validation.

    Returns:
        Selected and excluded layer names, with preparation and conversion configs.

    Raises:
        ValueError: If target paths overlap, a target selects no eligible
            layers, or a dimension, representation or device requirement fails.

    Note:
        Observer preparation can replace layers or add submodules. Resolve and
        validate all targets first so preparation and conversion reuse the same
        layer selection.
    """
    validate_quantization_targets(model=model, targets=targets)
    resolved_targets = []
    for target in targets:
        selected, skipped = target.select_modules(
            model=model,
            auto_filter_incompatible=auto_filter_incompatible_linears,
        )
        module_names = set(selected)
        validate_target_configuration(
            model=model,
            target=target,
            module_names=module_names,
            activation_dtype=activation_dtype,
            deployment_backend=deployment_backend,
            for_conversion=for_conversion,
            conversion_device=conversion_device,
        )
        resolved_targets.append(
            ResolvedQuantizationTarget(
                target=target,
                module_names=module_names,
                preparation_config=target.schema.preparation_config(is_qat=is_qat),
                conversion_config=target.schema.conversion_config(is_qat=is_qat),
                skipped=skipped,
            )
        )
        for name, reason in skipped.items():
            logger.info(f"Skipping quantization module {name}: {reason}")
    return resolved_targets


def validate_target_configuration(
    model: nn.Module,
    target: EagerQuantizationModuleTarget,
    module_names: set[str],
    activation_dtype: torch.dtype | None = None,
    deployment_backend: DeploymentBackend | None = None,
    for_conversion: bool = True,
    conversion_device: torch.device | None = None,
) -> None:
    """Run schema and deployment checks for the resolved layer selection.

    Args:
        model: Model containing the selected layers.
        target: Module scope and quantization schema.
        module_names: Fully qualified names selected for this target.
        activation_dtype: Known input dtype for quantized linear layers.
        deployment_backend: Backend responsible for artifact requirements.
        for_conversion: Check the weight placement required for conversion.
            QAT preparation checks numerical settings and dimensions.
        conversion_device: Destination for final weight conversion and packing.
            None uses each selected weight's current device for validation.
    """
    target.schema.validate_configuration(
        model=model,
        module_names=module_names,
        label=target.label,
        activation_dtype=activation_dtype,
        for_conversion=for_conversion,
        conversion_device=conversion_device,
    )
    if deployment_backend is not None:
        deployment_backend.validate_eager_target(
            model=model,
            target=target,
            module_names=module_names,
            for_conversion=for_conversion,
            conversion_device=conversion_device,
        )
