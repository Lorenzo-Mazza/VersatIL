"""Resolve target modules for parameter-efficient adaptation."""

import torch.nn as nn

from versatil.models.adaptation.constants import PEFTTargetModulePreset
from versatil.models.adaptation.target_patterns import (
    LLAMA_ATTENTION_AND_FEEDFORWARD_MODULES,
    LLAMA_QUERY_VALUE_MODULES,
    VLM_TEXT_MODEL_ATTENTION_AND_FEEDFORWARD_PATTERN,
    VLM_TEXT_MODEL_QUERY_VALUE_PATTERN,
)


def resolve_peft_target_modules(
    target_modules: str,
    scoped_target_modules: list[str] | None = None,
) -> str | list[str] | None:
    """Map a target-module preset to the value expected by PEFT.

    Args:
        target_modules: VersatIL target-module preset.
        scoped_target_modules: Resolved module names for a scope-based preset.

    Returns:
        Target-module value accepted by PEFT.

    Raises:
        ValueError: If the preset is unknown or requires an unresolved scope.
    """
    match target_modules:
        case PEFTTargetModulePreset.AUTO.value:
            return None
        case PEFTTargetModulePreset.ALL_LINEAR.value:
            return PEFTTargetModulePreset.ALL_LINEAR.value
        case PEFTTargetModulePreset.LLAMA_ATTENTION_AND_FEEDFORWARD.value:
            return LLAMA_ATTENTION_AND_FEEDFORWARD_MODULES
        case PEFTTargetModulePreset.LLAMA_QUERY_VALUE_PROJECTIONS.value:
            return LLAMA_QUERY_VALUE_MODULES
        case PEFTTargetModulePreset.VLM_TEXT_MODEL_ATTENTION_AND_FEEDFORWARD.value:
            return VLM_TEXT_MODEL_ATTENTION_AND_FEEDFORWARD_PATTERN
        case PEFTTargetModulePreset.VLM_TEXT_MODEL_QUERY_VALUE_PROJECTIONS.value:
            return VLM_TEXT_MODEL_QUERY_VALUE_PATTERN
        case PEFTTargetModulePreset.VLM_VISION_MODULES.value:
            if not scoped_target_modules:
                raise ValueError(
                    f"PEFT target preset '{target_modules}' requires at least one "
                    "resolved scoped module."
                )
            return scoped_target_modules
        case _:
            valid_targets = [preset.value for preset in PEFTTargetModulePreset]
            raise ValueError(
                f"Invalid PEFT target_modules '{target_modules}'. "
                f"Must be one of: {valid_targets}."
            )


def resolve_scoped_module_names(
    model: nn.Module,
    scoped_modules: list[nn.Module] | None,
    module_types: tuple[type[nn.Module], ...],
) -> list[str]:
    """Find matching layers contained in explicitly scoped modules.

    Args:
        model: Root model containing the requested scope.
        scoped_modules: Submodules that define the search scope.
        module_types: Layer types to include in the result.

    Returns:
        Fully qualified names of matching layers within the scope.

    Raises:
        ValueError: If no scope is provided, a scope is outside ``model``, or
            no layers of the requested types exist within the scope.
    """
    if not scoped_modules:
        raise ValueError("At least one scoped module is required.")
    named_modules = list(model.named_modules())
    scope_names = []
    for scoped_module in scoped_modules:
        matching_names = [
            name for name, module in named_modules if module is scoped_module
        ]
        if not matching_names:
            raise ValueError(
                f"Scoped module '{type(scoped_module).__name__}' is not registered "
                f"under model '{type(model).__name__}'."
            )
        scope_names.extend(matching_names)
    target_module_names = [
        name
        for name, module in named_modules
        if isinstance(module, module_types)
        and any(
            not scope_name or name == scope_name or name.startswith(f"{scope_name}.")
            for scope_name in scope_names
        )
    ]
    if not target_module_names:
        raise ValueError(
            "No modules of the requested types were found in the provided scope."
        )
    return list(dict.fromkeys(target_module_names))
