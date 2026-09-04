"""Tests for adaptation target-module resolution."""

import re

import pytest
import torch.nn as nn

from versatil.models.adaptation.constants import PEFTTargetModulePreset
from versatil.models.adaptation.target_resolution import (
    resolve_peft_target_modules,
    resolve_scoped_module_names,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(4, 4),
            nn.LayerNorm(4),
            nn.Sequential(nn.Linear(4, 4)),
        )
        self.decoder = nn.Linear(4, 4)


@pytest.mark.unit
class TestScopedModuleResolution:
    def test_resolves_nested_scopes_without_duplicates(self) -> None:
        model = TinyModel()

        result = resolve_scoped_module_names(
            model=model,
            scoped_modules=[model.encoder, model.encoder[2]],
            module_types=(nn.Linear,),
        )

        assert result == ["encoder.0", "encoder.2.0"]

    def test_requires_at_least_one_scope(self) -> None:
        model = TinyModel()
        expected_message = "At least one scoped module is required."

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            resolve_scoped_module_names(
                model=model,
                scoped_modules=None,
                module_types=(nn.Linear,),
            )

    def test_rejects_scope_outside_root_model(self) -> None:
        model = TinyModel()
        outside_module = nn.Linear(4, 4)
        expected_message = (
            "Scoped module 'Linear' is not registered under model 'TinyModel'."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            resolve_scoped_module_names(
                model=model,
                scoped_modules=[outside_module],
                module_types=(nn.Linear,),
            )

    def test_rejects_scope_without_requested_module_type(self) -> None:
        model = TinyModel()
        expected_message = (
            "No modules of the requested types were found in the provided scope."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            resolve_scoped_module_names(
                model=model,
                scoped_modules=[model.encoder[1]],
                module_types=(nn.Linear,),
            )


@pytest.mark.unit
def test_rejects_unknown_peft_target_preset() -> None:
    unknown_preset = "manual"
    valid_targets = [preset.value for preset in PEFTTargetModulePreset]
    expected_message = (
        f"Invalid PEFT target_modules '{unknown_preset}'. "
        f"Must be one of: {valid_targets}."
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        resolve_peft_target_modules(target_modules=unknown_preset)
