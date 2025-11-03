"""Tests for decoder configuration dataclasses."""
import dataclasses
import inspect

import pytest
from hydra.utils import instantiate

from refactoring.configs.decoding.decoder import ACTConfig
from refactoring.models.decoding.decoders.factory.act import ACT


@pytest.mark.unit
class TestACTConfig:

    def test_config_has_correct_target(self):
        config = ACTConfig(action_heads={})
        assert config._target_ == "refactoring.models.decoding.decoders.factory.act.ACT"

    def test_config_params_match_class_signature(self):
        sig = inspect.signature(ACT.__init__)
        params = set(sig.parameters.keys()) - {'self'}

        config = ACTConfig(action_heads={})
        config_keys = {f.name for f in dataclasses.fields(config)} - {'_target_'}

        assert config_keys.issubset(params), f"Extra keys: {config_keys - params}"
