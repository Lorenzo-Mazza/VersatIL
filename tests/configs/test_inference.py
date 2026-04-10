"""Tests for versatil.configs.inference module."""

import dataclasses

import pytest

from versatil.configs.inference import InferenceConfig


@pytest.mark.unit
class TestInferenceConfig:
    @pytest.mark.parametrize("temporal_agg", [True, False])
    @pytest.mark.parametrize("rotate_images", [True, False])
    def test_stores_configuration(self, temporal_agg, rotate_images):
        config = InferenceConfig(temporal_agg=temporal_agg, rotate_images=rotate_images)
        assert config.temporal_agg == temporal_agg
        assert config.rotate_images == rotate_images

    @pytest.mark.parametrize("update_rate_hz", [3.0, 10.0])
    def test_stores_update_rate(self, update_rate_hz):
        config = InferenceConfig(update_rate_hz=update_rate_hz)
        assert config.update_rate_hz == update_rate_hz

    def test_has_all_expected_fields(self):
        field_names = {f.name for f in dataclasses.fields(InferenceConfig)}
        expected = {
            "temporal_agg",
            "update_rate_hz",
            "rotate_images",
            "action_execution_horizon",
        }
        assert expected == field_names
