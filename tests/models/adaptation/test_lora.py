"""Tests for versatil.models.adaptation.lora module."""

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from peft import LoraConfig as PeftLoRAConfig
from transformers import GPT2Config, GPT2LMHeadModel

from versatil.models.adaptation.constants import PEFTTargetModulePreset
from versatil.models.adaptation.lora import (
    LoRAAdaptation,
    apply_lora_config,
    is_lora_enabled,
    to_peft_lora_config,
)


@pytest.fixture
def peft_config_factory() -> Callable[[], MagicMock]:
    def factory() -> MagicMock:
        return MagicMock(spec=PeftLoRAConfig)

    return factory


@pytest.fixture(scope="session")
def gpt2_model_factory() -> Callable[..., GPT2LMHeadModel]:
    def factory(vocabulary_size: int = 32) -> GPT2LMHeadModel:
        return GPT2LMHeadModel(
            GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=vocabulary_size)
        )

    return factory


@pytest.fixture
def token_ids_factory(rng: np.random.Generator) -> Callable[..., torch.Tensor]:
    def factory(
        batch_size: int = 1,
        sequence_length: int = 3,
        vocabulary_size: int = 32,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.integers(
                low=0, high=vocabulary_size, size=(batch_size, sequence_length)
            )
        )

    return factory


@pytest.mark.unit
class TestLoRAAdaptation:
    @pytest.mark.parametrize("enabled", [False, True])
    @pytest.mark.parametrize("rank, alpha, dropout", [(2, 4, 0.0), (4, 8, 0.25)])
    @pytest.mark.parametrize(
        "target_modules", [preset.value for preset in PEFTTargetModulePreset]
    )
    @pytest.mark.parametrize(
        "exclude_modules, bias, init_lora_weights",
        [(None, "none", "gaussian"), (["head"], "all", "pissa")],
    )
    def test_stores_configuration(
        self,
        lora_config_factory: Callable[..., LoRAAdaptation],
        enabled: bool,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: str,
        exclude_modules: list[str] | None,
        bias: str,
        init_lora_weights: str,
    ) -> None:
        config = lora_config_factory(
            enabled=enabled,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            bias=bias,
            init_lora_weights=init_lora_weights,
        )
        assert config.enabled == enabled
        assert config.rank == rank
        assert config.alpha == alpha
        assert config.dropout == dropout
        assert config.target_modules == target_modules
        assert config.exclude_modules == exclude_modules
        assert config.bias == bias
        assert config.init_lora_weights == init_lora_weights

    @pytest.mark.parametrize(
        "rank, alpha, dropout, target_modules, expectation",
        [
            (2, 4, 0.0, PEFTTargetModulePreset.AUTO.value, does_not_raise()),
            (
                4,
                8,
                0.5,
                PEFTTargetModulePreset.VLM_VISION_MODULES.value,
                does_not_raise(),
            ),
            (
                0,
                8,
                0.0,
                PEFTTargetModulePreset.AUTO.value,
                pytest.raises(
                    ValueError, match=re.escape("LoRA rank must be positive, got 0.")
                ),
            ),
            (
                4,
                0,
                0.0,
                PEFTTargetModulePreset.AUTO.value,
                pytest.raises(
                    ValueError, match=re.escape("LoRA alpha must be positive, got 0.")
                ),
            ),
            (
                4,
                8,
                1.0,
                PEFTTargetModulePreset.AUTO.value,
                pytest.raises(
                    ValueError,
                    match=re.escape("LoRA dropout must be in [0, 1), got 1.0."),
                ),
            ),
            (
                4,
                8,
                -0.1,
                PEFTTargetModulePreset.AUTO.value,
                pytest.raises(
                    ValueError,
                    match=re.escape("LoRA dropout must be in [0, 1), got -0.1."),
                ),
            ),
            (
                4,
                8,
                0.0,
                "manual",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Invalid LoRA target_modules 'manual'. "
                        f"Must be one of: {[preset.value for preset in PEFTTargetModulePreset]}."
                    ),
                ),
            ),
        ],
    )
    def test_validates_configuration(
        self,
        lora_config_factory: Callable[..., LoRAAdaptation],
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: str,
        expectation: AbstractContextManager,
    ) -> None:
        with expectation:
            config = lora_config_factory(
                enabled=True,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                target_modules=target_modules,
            )
            assert (
                config.rank,
                config.alpha,
                config.dropout,
                config.target_modules,
            ) == (
                rank,
                alpha,
                dropout,
                target_modules,
            )


@pytest.mark.unit
@pytest.mark.parametrize(
    "enabled, expected_enabled", [(None, False), (False, False), (True, True)]
)
def test_lora_enabled_requires_an_enabled_config(
    lora_config_factory: Callable[..., LoRAAdaptation],
    enabled: bool | None,
    expected_enabled: bool,
) -> None:
    config = None if enabled is None else lora_config_factory(enabled=enabled)
    assert is_lora_enabled(lora_config=config) == expected_enabled


@pytest.mark.unit
@pytest.mark.parametrize("init_lora_weights", ["gaussian", "pissa", "olora"])
@pytest.mark.parametrize(
    "target_modules, scoped_target_modules, resolved_targets",
    [
        (PEFTTargetModulePreset.AUTO.value, None, None),
        (PEFTTargetModulePreset.ALL_LINEAR.value, None, "all-linear"),
        (
            PEFTTargetModulePreset.VLM_VISION_MODULES.value,
            ["vision_tower.projection", "projector"],
            ["vision_tower.projection", "projector"],
        ),
    ],
)
def test_converts_configuration_and_resolved_targets_to_peft(
    lora_config_factory: Callable[..., LoRAAdaptation],
    peft_config_factory: Callable[[], MagicMock],
    init_lora_weights: str,
    target_modules: str,
    scoped_target_modules: list[str] | None,
    resolved_targets: str | list[str] | None,
) -> None:
    config = lora_config_factory(
        enabled=True,
        rank=4,
        alpha=8,
        dropout=0.25,
        target_modules=target_modules,
        exclude_modules=["head"],
        bias="none",
        init_lora_weights=init_lora_weights,
    )
    peft_config = peft_config_factory()
    with (
        patch(
            "versatil.models.adaptation.lora.resolve_peft_target_modules",
            autospec=True,
            return_value=resolved_targets,
        ) as resolve_targets,
        patch(
            "versatil.models.adaptation.lora.PeftLoRAConfig",
            autospec=True,
            return_value=peft_config,
        ) as build_config,
    ):
        result = to_peft_lora_config(
            lora_config=config, scoped_target_modules=scoped_target_modules
        )

    resolve_targets.assert_called_once_with(
        target_modules=target_modules, scoped_target_modules=scoped_target_modules
    )
    build_config.assert_called_once_with(
        r=4,
        lora_alpha=8,
        lora_dropout=0.25,
        target_modules=resolved_targets,
        exclude_modules=["head"],
        bias="none",
        init_lora_weights=init_lora_weights,
    )
    assert result == peft_config


@pytest.mark.unit
class TestApplyLoRAAdaptation:
    @pytest.mark.parametrize(
        "enabled, frozen, adapted, expectation",
        [
            (False, True, False, does_not_raise()),
            (
                True,
                True,
                False,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "LoRA adaptation cannot be enabled when frozen=True because LoRA "
                        "adds trainable adapter parameters. Set frozen=False to train "
                        "adapters, or disable LoRA for a fully frozen model."
                    ),
                ),
            ),
            (
                True,
                False,
                True,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "LoRA adaptation is already applied to this model. Re-applying "
                        "LoRA would add another adapter; instantiate a fresh base model "
                        "or unload the existing adapter first."
                    ),
                ),
            ),
        ],
    )
    def test_skips_wrapping_for_disabled_or_invalid_adaptation(
        self,
        adaptation_model_factory: Callable[..., MagicMock],
        lora_config_factory: Callable[..., LoRAAdaptation],
        enabled: bool,
        frozen: bool,
        adapted: bool,
        expectation: AbstractContextManager,
    ) -> None:
        model = adaptation_model_factory(adapted=adapted)
        config = lora_config_factory(enabled=enabled)
        with (
            patch(
                "versatil.models.adaptation.lora.is_lora_enabled",
                autospec=True,
                return_value=enabled,
            ) as check_enabled,
            patch(
                "versatil.models.adaptation.lora.resolve_scoped_module_names",
                autospec=True,
            ) as resolve_scope,
            patch(
                "versatil.models.adaptation.lora.to_peft_lora_config", autospec=True
            ) as build_config,
            patch(
                "versatil.models.adaptation.lora.get_peft_model", autospec=True
            ) as wrap_model,
            expectation,
        ):
            result = apply_lora_config(model=model, lora_config=config, frozen=frozen)
            assert result == model

        check_enabled.assert_called_once_with(lora_config=config)
        resolve_scope.assert_not_called()
        build_config.assert_not_called()
        wrap_model.assert_not_called()

    @pytest.mark.parametrize(
        "target_modules",
        [
            PEFTTargetModulePreset.AUTO.value,
            PEFTTargetModulePreset.ALL_LINEAR.value,
            PEFTTargetModulePreset.VLM_VISION_MODULES.value,
        ],
    )
    def test_passes_scoped_targets_and_configuration_to_peft(
        self,
        adaptation_model_factory: Callable[..., MagicMock],
        lora_config_factory: Callable[..., LoRAAdaptation],
        peft_config_factory: Callable[[], MagicMock],
        target_modules: str,
    ) -> None:
        model = adaptation_model_factory(adapted=False)
        adapted_model = adaptation_model_factory(adapted=True)
        config = lora_config_factory(enabled=True, target_modules=target_modules)
        peft_config = peft_config_factory()
        scopes = [model.vision_tower, model.projector]
        resolved_targets = ["vision_tower.projection", "projector"]
        with (
            patch(
                "versatil.models.adaptation.lora.is_lora_enabled",
                autospec=True,
                return_value=True,
            ) as check_enabled,
            patch(
                "versatil.models.adaptation.lora.resolve_scoped_module_names",
                autospec=True,
                return_value=resolved_targets,
            ) as resolve_scope,
            patch(
                "versatil.models.adaptation.lora.to_peft_lora_config",
                autospec=True,
                return_value=peft_config,
            ) as build_config,
            patch(
                "versatil.models.adaptation.lora.get_peft_model",
                autospec=True,
                return_value=adapted_model,
            ) as wrap_model,
        ):
            result = apply_lora_config(
                model=model, lora_config=config, frozen=False, scoped_modules=scopes
            )

        check_enabled.assert_called_once_with(lora_config=config)
        if target_modules == PEFTTargetModulePreset.VLM_VISION_MODULES.value:
            resolve_scope.assert_called_once_with(
                model=model, scoped_modules=scopes, module_types=(nn.Linear,)
            )
            build_config.assert_called_once_with(
                lora_config=config, scoped_target_modules=resolved_targets
            )
        else:
            resolve_scope.assert_not_called()
            build_config.assert_called_once_with(
                lora_config=config, scoped_target_modules=None
            )
        wrap_model.assert_called_once_with(model, peft_config)
        assert result == adapted_model


@pytest.mark.integration
def test_lora_wraps_tiny_gpt2_and_keeps_only_adapter_weights_trainable(
    gpt2_model_factory: Callable[..., GPT2LMHeadModel],
    token_ids_factory: Callable[..., torch.Tensor],
    lora_config_factory: Callable[..., LoRAAdaptation],
    parameter_count: Callable[[torch.nn.Module], int],
    trainable_parameter_count: Callable[[torch.nn.Module], int],
) -> None:
    model = gpt2_model_factory(vocabulary_size=32)
    config = lora_config_factory(
        enabled=True,
        rank=2,
        alpha=4,
        target_modules=PEFTTargetModulePreset.ALL_LINEAR.value,
        exclude_modules=["c_fc"],
    )
    adapted_model = apply_lora_config(model=model, lora_config=config, frozen=False)
    trainable_parameters = [
        (name, parameter)
        for name, parameter in adapted_model.named_parameters()
        if parameter.requires_grad
    ]
    input_ids = token_ids_factory(batch_size=1, sequence_length=3, vocabulary_size=32)
    output = adapted_model(input_ids=input_ids)
    output.logits.square().mean().backward()

    assert len(trainable_parameters) == 6
    assert all("lora_" in name for name, _ in trainable_parameters)
    assert all("c_fc" not in name for name, _ in trainable_parameters)
    assert 0 < trainable_parameter_count(adapted_model) < parameter_count(adapted_model)
    assert output.logits.shape == (1, 3, 32)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for _, parameter in trainable_parameters
    )
