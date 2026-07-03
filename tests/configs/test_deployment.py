"""Tests for versatil.configs.deployment module."""

import dataclasses

import pytest
from omegaconf import MISSING

from versatil.configs.deployment import DeploymentConfig


@pytest.mark.unit
class TestDeploymentConfig:
    def test_checkpoint_path_is_required(self):
        config = DeploymentConfig()
        assert config.checkpoint_path == MISSING

    @pytest.mark.parametrize("temporal_aggregation", [True, False])
    @pytest.mark.parametrize("request_timeout_seconds", [None, 2.5])
    def test_stores_configuration(
        self,
        temporal_aggregation: bool,
        request_timeout_seconds: float | None,
    ):
        config = DeploymentConfig(
            checkpoint_path="/ckpt",
            temporal_aggregation=temporal_aggregation,
            request_timeout_seconds=request_timeout_seconds,
        )
        assert config.temporal_aggregation == temporal_aggregation
        assert config.request_timeout_seconds == request_timeout_seconds

    def test_has_all_expected_fields(self):
        field_names = {field.name for field in dataclasses.fields(DeploymentConfig)}
        expected = {
            "checkpoint_path",
            "checkpoint_name",
            "device",
            "model_server_address",
            "model_server_port",
            "temporal_aggregation",
            "action_execution_horizon",
            "update_rate_hz",
            "max_steps",
            "temporal_max_timesteps",
            "timing_log",
            "compile_model",
            "request_timeout_seconds",
        }
        assert expected == field_names
