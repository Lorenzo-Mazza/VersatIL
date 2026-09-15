"""Tests for versatil.models.adaptation.target_resolution module."""

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from versatil.models.adaptation.constants import PEFTTargetModulePreset
from versatil.models.adaptation.target_resolution import (
    resolve_peft_target_modules,
    resolve_scoped_module_names,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "scope_names, module_types, expected_names, expectation",
    [
        (
            ["vision_tower", "vision_tower.nested", "projector"],
            (nn.Linear,),
            ["vision_tower.projection", "vision_tower.nested.projection", "projector"],
            does_not_raise(),
        ),
        (
            [""],
            (nn.Linear,),
            [
                "vision_tower.projection",
                "vision_tower.nested.projection",
                "projector",
                "language_model",
            ],
            does_not_raise(),
        ),
        (["projector"], (nn.Linear,), ["projector"], does_not_raise()),
        (
            ["vision_tower"],
            (nn.LayerNorm,),
            ["vision_tower.normalization"],
            does_not_raise(),
        ),
        (
            None,
            (nn.Linear,),
            None,
            pytest.raises(
                ValueError, match=re.escape("At least one scoped module is required.")
            ),
        ),
        (
            [],
            (nn.Linear,),
            None,
            pytest.raises(
                ValueError, match=re.escape("At least one scoped module is required.")
            ),
        ),
        (
            ["outside"],
            (nn.Linear,),
            None,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Scoped module 'MagicMock' is not registered under model 'MagicMock'."
                ),
            ),
        ),
        (
            ["vision_tower.normalization"],
            (nn.Linear,),
            None,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "No modules of the requested types were found in the provided scope."
                ),
            ),
        ),
    ],
)
def test_resolves_scoped_modules(
    adaptation_model_factory: Callable[..., MagicMock],
    scope_names: list[str] | None,
    module_types: tuple[type[nn.Module], ...],
    expected_names: list[str] | None,
    expectation: AbstractContextManager,
) -> None:
    model = adaptation_model_factory(adapted=False)
    modules = dict(model.named_modules.return_value)
    modules["outside"] = adaptation_model_factory(adapted=False).projector
    scopes = None if scope_names is None else [modules[name] for name in scope_names]

    with expectation:
        result = resolve_scoped_module_names(
            model=model,
            scoped_modules=scopes,
            module_types=module_types,
        )
        assert result == expected_names

    if scope_names:
        model.named_modules.assert_called_once_with()
    else:
        model.named_modules.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "target_modules, scoped_target_modules, expected_targets, expectation",
    [
        (PEFTTargetModulePreset.AUTO.value, None, None, does_not_raise()),
        (PEFTTargetModulePreset.ALL_LINEAR.value, None, "all-linear", does_not_raise()),
        (
            PEFTTargetModulePreset.LLAMA_ATTENTION_AND_FEEDFORWARD.value,
            None,
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            does_not_raise(),
        ),
        (
            PEFTTargetModulePreset.LLAMA_QUERY_VALUE_PROJECTIONS.value,
            None,
            ["q_proj", "v_proj"],
            does_not_raise(),
        ),
        (
            PEFTTargetModulePreset.VLM_TEXT_MODEL_ATTENTION_AND_FEEDFORWARD.value,
            None,
            r".*(language_model|text_model)\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            does_not_raise(),
        ),
        (
            PEFTTargetModulePreset.VLM_TEXT_MODEL_QUERY_VALUE_PROJECTIONS.value,
            None,
            r".*(language_model|text_model)\..*\.self_attn\.(q_proj|v_proj)$",
            does_not_raise(),
        ),
        (
            PEFTTargetModulePreset.VLM_VISION_MODULES.value,
            ["vision_tower.projection", "projector"],
            ["vision_tower.projection", "projector"],
            does_not_raise(),
        ),
        *[
            (
                PEFTTargetModulePreset.VLM_VISION_MODULES.value,
                unresolved_scope,
                None,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "PEFT target preset 'vlm-vision-modules' requires at least one resolved scoped module."
                    ),
                ),
            )
            for unresolved_scope in (None, [])
        ],
        (
            "manual",
            None,
            None,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Invalid PEFT target_modules 'manual'. "
                    f"Must be one of: {[preset.value for preset in PEFTTargetModulePreset]}."
                ),
            ),
        ),
    ],
)
def test_resolves_peft_target_presets(
    target_modules: str,
    scoped_target_modules: list[str] | None,
    expected_targets: str | list[str] | None,
    expectation: AbstractContextManager,
) -> None:
    with expectation:
        assert (
            resolve_peft_target_modules(
                target_modules=target_modules,
                scoped_target_modules=scoped_target_modules,
            )
            == expected_targets
        )
