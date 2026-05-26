"""LoRA adaptation helpers for HuggingFace modules."""

from dataclasses import dataclass

import torch.nn as nn
from peft import LoraConfig as PeftLoRAConfig
from peft import PeftModel, get_peft_model

from versatil.models.adaptation.constants import LoRATargetModulePreset

LLAMA_ATTENTION_AND_FEEDFORWARD_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
VLM_TEXT_MODEL_ATTENTION_AND_FEEDFORWARD_PATTERN = (
    r".*(language_model|text_model)\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)
VLM_TEXT_MODEL_QUERY_VALUE_PATTERN = (
    r".*(language_model|text_model)\..*\.self_attn\.(q_proj|v_proj)$"
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
            linear layers, and the language-model presets restrict LoRA to
            text-model projections inside VLM wrappers.
        exclude_modules: Optional module names to leave unwrapped even if they
            match the selected target preset.
        bias: PEFT bias handling mode.
    """

    enabled: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: str = LoRATargetModulePreset.AUTO.value
    exclude_modules: list[str] | None = None
    bias: str = "none"

    def __post_init__(self) -> None:
        """Validate LoRA hyperparameters."""
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}.")
        if self.alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.alpha}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}.")
        valid_targets = [preset.value for preset in LoRATargetModulePreset]
        if self.target_modules not in valid_targets:
            raise ValueError(
                f"Invalid LoRA target_modules '{self.target_modules}'. "
                f"Must be one of: {valid_targets}."
            )


def is_lora_enabled(lora_config: LoRAAdaptation | None) -> bool:
    """Return whether LoRA adaptation should wrap a model."""
    return lora_config is not None and lora_config.enabled


def _to_peft_target_modules(target_modules: str) -> str | list[str] | None:
    """Map VersatIL target-module presets to PEFT values.

    Args:
        target_modules: VersatIL LoRA target-module preset.

    Returns:
        PEFT target_modules value.
    """
    if target_modules == LoRATargetModulePreset.AUTO.value:
        return None
    if target_modules == LoRATargetModulePreset.ALL_LINEAR.value:
        return LoRATargetModulePreset.ALL_LINEAR.value
    if target_modules == LoRATargetModulePreset.LLAMA_ATTENTION_AND_FEEDFORWARD.value:
        return LLAMA_ATTENTION_AND_FEEDFORWARD_MODULES
    if (
        target_modules
        == LoRATargetModulePreset.VLM_TEXT_MODEL_ATTENTION_AND_FEEDFORWARD.value
    ):
        return VLM_TEXT_MODEL_ATTENTION_AND_FEEDFORWARD_PATTERN
    if (
        target_modules
        == LoRATargetModulePreset.VLM_TEXT_MODEL_QUERY_VALUE_PROJECTIONS.value
    ):
        return VLM_TEXT_MODEL_QUERY_VALUE_PATTERN
    valid_targets = [preset.value for preset in LoRATargetModulePreset]
    raise ValueError(
        f"Invalid LoRA target_modules '{target_modules}'. "
        f"Must be one of: {valid_targets}."
    )


def to_peft_lora_config(lora_config: LoRAAdaptation) -> PeftLoRAConfig:
    """Convert a VersatIL LoRA config to PEFT's LoRA configuration.

    Args:
        lora_config: VersatIL LoRA configuration.

    Returns:
        PEFT LoRA configuration.
    """
    return PeftLoRAConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=_to_peft_target_modules(lora_config.target_modules),
        exclude_modules=lora_config.exclude_modules,
        bias=lora_config.bias,
    )


def apply_lora_config(
    model: nn.Module,
    lora_config: LoRAAdaptation | None,
    frozen: bool,
) -> nn.Module:
    """Wrap a HuggingFace module with LoRA adapters when configured.

    Args:
        model: HuggingFace module to adapt.
        lora_config: Optional LoRA configuration.
        frozen: Whether the owning wrapper requests a fully frozen model.

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
    return get_peft_model(model, to_peft_lora_config(lora_config=lora_config))
