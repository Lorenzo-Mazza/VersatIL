"""Tests for versatil.configs.deployment module."""

import dataclasses

import pytest
from omegaconf import MISSING

from versatil.configs.deployment import DeploymentConfig
from versatil.configs.inference_client import InferenceClientConfig


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
            client=InferenceClientConfig(
                temporal_aggregation=temporal_aggregation,
                request_timeout_seconds=request_timeout_seconds,
            ),
        )
        assert config.client.temporal_aggregation == temporal_aggregation
        assert config.client.request_timeout_seconds == request_timeout_seconds

    def test_has_all_expected_fields(self):
        field_names = {field.name for field in dataclasses.fields(DeploymentConfig)}
        expected = {
            "checkpoint_path",
            "checkpoint_name",
            "device",
            "max_steps",
            "compile_model",
            "client",
        }
        assert expected == field_names
