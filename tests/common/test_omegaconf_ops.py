"""Tests for versatil.common.omegaconf_ops module."""

import pytest
from omegaconf import OmegaConf

from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.configs import register_resolvers
from versatil.data.constants import Cameras, MetadataPassthroughSource, SyntheticObsKey

register_resolvers()


@pytest.mark.unit
class TestResolveDictKeys:
    def test_plain_string_keys_returned_unchanged(self) -> None:
        input_dict = {"key_a": 1, "key_b": 2}
        result = resolve_dict_keys(input_dict)
        assert result == {"key_a": 1, "key_b": 2}

    def test_non_string_keys_returned_unchanged(self) -> None:
        input_dict = {42: "value", True: "other"}
        result = resolve_dict_keys(input_dict)
        assert result == {42: "value", True: "other"}

    def test_nested_dict_values_resolved_recursively(self) -> None:
        input_dict = {"outer": {"inner_key": "inner_value"}}
        result = resolve_dict_keys(input_dict)
        assert result["outer"]["inner_key"] == "inner_value"

    def test_nested_dictconfig_keys_resolved_recursively(self) -> None:
        input_config = OmegaConf.create(
            {
                "${metadata_passthrough_source:OBSERVATION}": {
                    "${synthetic_obs_key:MODE_ID}": "latent_color_label"
                }
            }
        )
        result = resolve_dict_keys(input_config)
        assert result == {
            MetadataPassthroughSource.OBSERVATION.value: {
                SyntheticObsKey.MODE_ID.value: "latent_color_label"
            }
        }

    def test_non_dict_values_preserved(self) -> None:
        input_dict = {"key": [1, 2, 3], "key2": "string", "key3": 42}
        result = resolve_dict_keys(input_dict)
        assert result["key"] == [1, 2, 3]
        assert result["key2"] == "string"
        assert result["key3"] == 42

    def test_empty_dict_returns_empty(self) -> None:
        result = resolve_dict_keys({})
        assert result == {}

    def test_interpolation_key_resolved_via_registered_resolver(self) -> None:
        input_dict = {"${cameras:LEFT}": "value"}
        result = resolve_dict_keys(input_dict)
        assert Cameras.LEFT.value in result
        assert result[Cameras.LEFT.value] == "value"

    def test_deeply_nested_dicts_resolved(self) -> None:
        input_dict = {"level1": {"level2": {"level3": "deep_value"}}}
        result = resolve_dict_keys(input_dict)
        assert result["level1"]["level2"]["level3"] == "deep_value"

    def test_mixed_interpolation_and_plain_keys(self) -> None:
        input_dict = {
            "plain_key": "value1",
            "${cameras:RIGHT}": "value2",
        }
        result = resolve_dict_keys(input_dict)
        assert result["plain_key"] == "value1"
        assert result[Cameras.RIGHT.value] == "value2"

    def test_returns_new_dict_not_mutating_input(self) -> None:
        input_dict = {"key": "value"}
        result = resolve_dict_keys(input_dict)
        assert result is not input_dict
        result["new_key"] = "new_value"
        assert "new_key" not in input_dict
