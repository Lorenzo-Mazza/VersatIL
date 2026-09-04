"""LoRA adaptation helpers for HuggingFace modules."""

from dataclasses import dataclass

import torch.nn as nn
from peft import LoraConfig as PeftLoRAConfig
from peft import PeftModel, get_peft_model

from versatil.models.adaptation.constants import (
    DEFAULT_LORA_INIT_WEIGHTS,
    PEFTTargetModulePreset,
)
from versatil.models.adaptation.target_resolution import (
    resolve_peft_target_modules,
    resolve_scoped_module_names,
)


@dataclass
class LoRAAdaptation:
    """Runtime configuration for Low-Rank Adaptation.

    Args:
        enabled: Whether to wrap the model with LoRA adapters. Disabled
            configurations leave the original module unchanged.
        rank: Adapter rank ``r``. Higher ranks give the adapter more capacity
            and increase trainable parameters.
        alpha: Adapter scaling factor. PEFT applies the learned update with
            scale ``alpha / rank``, so this controls how strongly the adapter
            update is added to the base weights.
        dropout: Dropout probability on the adapter path. Larger values add
            regularization before the low-rank update.
        target_modules: PEFT target-module preset. ``auto`` lets PEFT infer
            supported module names from the model type, ``all-linear`` adapts
            linear layers, and the VLM presets restrict LoRA to text- or
            vision-model projections inside VLM wrappers.
        exclude_modules: Optional module names to leave unwrapped even if they
            match the selected target preset.
        bias: PEFT bias handling mode.
        init_lora_weights: PEFT adapter weight initialization strategy passed to
            ``LoraConfig``. ``gaussian`` matches the OpenVLA-OFT recipe.
    """

    enabled: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: str = PEFTTargetModulePreset.AUTO.value
    exclude_modules: list[str] | None = None
    bias: str = "none"
    init_lora_weights: str = DEFAULT_LORA_INIT_WEIGHTS

    def __post_init__(self) -> None:
        """Validate LoRA hyperparameters."""
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}.")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}.")
        valid_targets = [preset.value for preset in PEFTTargetModulePreset]
        if self.target_modules not in valid_targets:
            raise ValueError(
                f"Invalid LoRA target_modules '{self.target_modules}'. "
                f"Must be one of: {valid_targets}."
            )


def is_lora_enabled(lora_config: LoRAAdaptation | None) -> bool:
    """Return whether LoRA adaptation should wrap a model."""
    return lora_config is not None and lora_config.enabled


def to_peft_lora_config(
    lora_config: LoRAAdaptation,
    scoped_target_modules: list[str] | None = None,
) -> PeftLoRAConfig:
    """Convert a VersatIL LoRA config to PEFT's LoRA configuration.

    Args:
        lora_config: VersatIL LoRA configuration.
        scoped_target_modules: Resolved module names for a scope-based preset.

    Returns:
        PEFT LoRA configuration.
    """
    return PeftLoRAConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=resolve_peft_target_modules(
            target_modules=lora_config.target_modules,
            scoped_target_modules=scoped_target_modules,
        ),
        exclude_modules=lora_config.exclude_modules,
        bias=lora_config.bias,
        init_lora_weights=lora_config.init_lora_weights,
    )


def apply_lora_config(
    model: nn.Module,
    lora_config: LoRAAdaptation | None,
    frozen: bool,
    scoped_modules: list[nn.Module] | None = None,
) -> nn.Module:
    """Wrap a HuggingFace module with LoRA adapters when configured.

    Args:
        model: HuggingFace module to adapt.
        lora_config: Optional LoRA configuration.
        frozen: Whether the owning wrapper requests a fully frozen model.
        scoped_modules: Optional submodules that constrain a scope-based target
            preset.

    Returns:
        The original model when LoRA is disabled, otherwise a PEFT-wrapped model.
    """
    if not is_lora_enabled(lora_config=lora_config):
        return model
    if frozen:
        raise ValueError(
            "LoRA adaptation cannot be enabled when frozen=True because LoRA "
            "adds trainable adapter parameters. Set frozen=False to train "
            "adapters, or disable LoRA for a fully frozen model."
        )
    if isinstance(model, PeftModel):
        raise ValueError(
            "LoRA adaptation is already applied to this model. Re-applying "
            "LoRA would add another adapter; instantiate a fresh base model "
            "or unload the existing adapter first."
        )
    scoped_target_modules = None
    if lora_config.target_modules == PEFTTargetModulePreset.VLM_VISION_MODULES.value:
        scoped_target_modules = resolve_scoped_module_names(
            model=model,
            scoped_modules=scoped_modules,
            module_types=(nn.Linear,),
        )
    peft_config = to_peft_lora_config(
        lora_config=lora_config,
        scoped_target_modules=scoped_target_modules,
    )
    return get_peft_model(model, peft_config)
