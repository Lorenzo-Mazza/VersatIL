"""Configuration classes for model adaptation for parameters-efficient fine-tuning."""

from dataclasses import dataclass

from versatil.models.adaptation.constants import LoRATargetModulePreset


@dataclass
class LoRAAdaptationConfig:
    """Hydra configuration for Low-Rank Adaptation.

    Args:
        enabled: Whether Hydra should instantiate LoRA adaptation for the
            target module.
        rank: Adapter rank ``r``. Higher ranks increase adapter capacity and
            trainable parameters.
        alpha: Adapter scaling factor. PEFT applies the learned update with
            scale ``alpha / rank``.
        dropout: Dropout probability on the adapter path before the low-rank
            update.
        target_modules: PEFT target-module preset.
        exclude_modules: Optional module names to leave unwrapped by LoRA.
        bias: PEFT bias handling mode.
    """

    _target_: str = "versatil.models.adaptation.lora.LoRAAdaptation"
    enabled: bool = True
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: str = LoRATargetModulePreset.AUTO.value
    exclude_modules: list[str] | None = None
    bias: str = "none"
