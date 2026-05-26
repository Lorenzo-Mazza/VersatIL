"""Tests for versatil.models.adaptation.lora module."""

import re
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from versatil.models.adaptation.constants import LoRATargetModulePreset
from versatil.models.adaptation.lora import (
    LoRAAdaptation,
    apply_lora_config,
    to_peft_lora_config,
)


class TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)


class TinyAdaptedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)


@pytest.mark.unit
class TestLoRAAdaptation:
    def test_converts_auto_target_modules_to_peft_default(self) -> None:
        config = LoRAAdaptation(
            enabled=True,
            rank=4,
            alpha=8,
            dropout=0.25,
            target_modules=LoRATargetModulePreset.AUTO.value,
            exclude_modules=["head"],
            bias="none",
        )

        peft_config = to_peft_lora_config(lora_config=config)

        assert peft_config.r == 4
        assert peft_config.lora_alpha == 8
        assert peft_config.lora_dropout == pytest.approx(0.25)
        assert peft_config.target_modules is None
        assert peft_config.exclude_modules == {"head"}
        assert peft_config.bias == "none"

    def test_converts_all_linear_target_modules_to_peft_string(self) -> None:
        config = LoRAAdaptation(
            enabled=True,
            rank=4,
            alpha=8,
            dropout=0.0,
            target_modules=LoRATargetModulePreset.ALL_LINEAR.value,
        )

        peft_config = to_peft_lora_config(lora_config=config)

        assert peft_config.target_modules == LoRATargetModulePreset.ALL_LINEAR.value

    @pytest.mark.parametrize(
        "target_modules, expected_target_modules",
        [
            (
                LoRATargetModulePreset.LLAMA_ATTENTION_AND_FEEDFORWARD.value,
                {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                },
            ),
            (
                LoRATargetModulePreset.VLM_TEXT_MODEL_ATTENTION_AND_FEEDFORWARD.value,
                r".*(language_model|text_model)\..*\."
                r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            ),
            (
                LoRATargetModulePreset.VLM_TEXT_MODEL_QUERY_VALUE_PROJECTIONS.value,
                r".*(language_model|text_model)\..*\.self_attn\.(q_proj|v_proj)$",
            ),
        ],
    )
    def test_converts_language_model_target_presets_to_peft_values(
        self,
        target_modules: str,
        expected_target_modules: str | set[str],
    ) -> None:
        config = LoRAAdaptation(
            enabled=True,
            target_modules=target_modules,
        )

        peft_config = to_peft_lora_config(lora_config=config)

        assert peft_config.target_modules == expected_target_modules

    @pytest.mark.parametrize(
        "rank, alpha, dropout, target_modules, expected_message",
        [
            (
                0,
                8,
                0.0,
                LoRATargetModulePreset.AUTO.value,
                "LoRA rank must be positive, got 0.",
            ),
            (
                4,
                0,
                0.0,
                LoRATargetModulePreset.AUTO.value,
                "LoRA alpha must be positive, got 0.",
            ),
            (
                4,
                8,
                1.0,
                LoRATargetModulePreset.AUTO.value,
                "LoRA dropout must be in [0, 1), got 1.0.",
            ),
            (
                4,
                8,
                0.0,
                "manual",
                "Invalid LoRA target_modules 'manual'. "
                "Must be one of: ['auto', 'all-linear', "
                "'llama-attention-and-feedforward', "
                "'vlm-text-model-attention-and-feedforward', "
                "'vlm-text-model-query-value-projections'].",
            ),
        ],
    )
    def test_validates_configuration(
        self,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: str,
        expected_message: str,
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            LoRAAdaptation(
                enabled=True,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
            )


@pytest.mark.unit
class TestApplyLoRAAdaptation:
    def test_returns_original_model_when_unconfigured(self) -> None:
        model = TinyModule()

        result = apply_lora_config(model=model, lora_config=None, frozen=True)

        assert result is model

    def test_returns_original_model_when_disabled(self) -> None:
        model = TinyModule()
        config = LoRAAdaptation(enabled=False)

        result = apply_lora_config(model=model, lora_config=config, frozen=True)

        assert result is model

    def test_raises_when_lora_is_enabled_on_frozen_model(self) -> None:
        model = TinyModule()
        config = LoRAAdaptation(enabled=True)
        expected_message = (
            "LoRA adaptation cannot be enabled when frozen=True because LoRA "
            "adds trainable adapter parameters. Set frozen=False to train "
            "adapters, or disable LoRA for a fully frozen model."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            apply_lora_config(model=model, lora_config=config, frozen=True)

    def test_wraps_model_with_enabled_lora(self) -> None:
        model = TinyModule()
        adapted_model = TinyModule()
        config = LoRAAdaptation(enabled=True)

        with patch(
            "versatil.models.adaptation.lora.get_peft_model",
            return_value=adapted_model,
        ) as mock_get_peft_model:
            result = apply_lora_config(model=model, lora_config=config, frozen=False)

        assert result is adapted_model
        assert mock_get_peft_model.call_args.args[0] is model

    def test_raises_when_lora_is_applied_twice(self) -> None:
        model = TinyAdaptedModule()
        config = LoRAAdaptation(enabled=True)
        expected_message = (
            "LoRA adaptation is already applied to this model. Re-applying "
            "LoRA would add another adapter; instantiate a fresh base model "
            "or unload the existing adapter first."
        )

        with (
            patch("versatil.models.adaptation.lora.PeftModel", TinyAdaptedModule),
            pytest.raises(ValueError, match=re.escape(expected_message)),
        ):
            apply_lora_config(model=model, lora_config=config, frozen=False)

    def test_wraps_model_with_peft_when_enabled(self) -> None:
        model = TinyModule()
        adapted_model = TinyModule()
        config = LoRAAdaptation(
            enabled=True,
            rank=2,
            alpha=4,
            target_modules=LoRATargetModulePreset.ALL_LINEAR.value,
        )

        with patch(
            "versatil.models.adaptation.lora.get_peft_model",
            return_value=adapted_model,
        ) as mock_get_peft_model:
            result = apply_lora_config(
                model=model,
                lora_config=config,
                frozen=False,
            )

        assert result is adapted_model
        call_model, call_config = mock_get_peft_model.call_args.args
        assert call_model is model
        assert call_config.r == 2
        assert call_config.lora_alpha == 4
        assert call_config.target_modules == LoRATargetModulePreset.ALL_LINEAR.value
        assert call_config.exclude_modules is None

    def test_wraps_model_with_excluded_modules(self) -> None:
        model = TinyModule()
        adapted_model = TinyModule()
        config = LoRAAdaptation(
            enabled=True,
            target_modules=LoRATargetModulePreset.ALL_LINEAR.value,
            exclude_modules=["projection"],
        )

        with patch(
            "versatil.models.adaptation.lora.get_peft_model",
            return_value=adapted_model,
        ) as mock_get_peft_model:
            apply_lora_config(
                model=model,
                lora_config=config,
                frozen=False,
            )

        _, call_config = mock_get_peft_model.call_args.args
        assert call_config.exclude_modules == {"projection"}


@pytest.mark.integration
def test_lora_wraps_tiny_gpt2_and_keeps_only_adapter_weights_trainable(
    parameter_count: Callable[[torch.nn.Module], int],
    trainable_parameter_count: Callable[[torch.nn.Module], int],
) -> None:
    model = GPT2LMHeadModel(
        GPT2Config(
            n_layer=1,
            n_head=2,
            n_embd=16,
            vocab_size=32,
        )
    )
    config = LoRAAdaptation(
        enabled=True,
        rank=2,
        alpha=4,
        target_modules=LoRATargetModulePreset.ALL_LINEAR.value,
        exclude_modules=["c_fc"],
    )

    adapted_model = apply_lora_config(
        model=model,
        lora_config=config,
        frozen=False,
    )
    trainable_parameter_names = [
        name
        for name, parameter in adapted_model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_parameters = trainable_parameter_count(adapted_model)
    total_parameters = parameter_count(adapted_model)
    output = adapted_model(input_ids=torch.tensor([[1, 2, 3]]))

    assert len(trainable_parameter_names) == 6
    assert all("lora_" in name for name in trainable_parameter_names)
    assert all("c_fc" not in name for name in trainable_parameter_names)
    assert 0 < trainable_parameters < total_parameters
    assert output.logits.shape == (1, 3, 32)
