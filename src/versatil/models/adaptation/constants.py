"""Constants for parameter-efficient model adaptation."""

import enum


class LoRATargetModulePreset(enum.StrEnum):
    """PEFT target-module presets exposed by VersatIL."""

    AUTO = "auto"
    ALL_LINEAR = "all-linear"
    LLAMA_ATTENTION_AND_FEEDFORWARD = "llama-attention-and-feedforward"
    VLM_TEXT_MODEL_ATTENTION_AND_FEEDFORWARD = (
        "vlm-text-model-attention-and-feedforward"
    )
    VLM_TEXT_MODEL_QUERY_VALUE_PROJECTIONS = "vlm-text-model-query-value-projections"
