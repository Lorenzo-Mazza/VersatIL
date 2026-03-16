"""Tests for versatil.configs.main module."""
import dataclasses

import pytest

from versatil.configs.experiment import ExperimentConfig
from versatil.configs.inference import InferenceConfig
from versatil.configs.main import MainConfig
from versatil.configs.policy import PolicyConfig
from versatil.configs.data.task import TaskSpaceConfig
from versatil.configs.training import TrainingConfig


@pytest.mark.unit
class TestMainConfig:

    def test_has_all_expected_fields(self):
        field_names = {f.name for f in dataclasses.fields(MainConfig)}
        expected = {
            "defaults",
            "experiment",
            "task",
            "training",
            "policy",
            "inference",
        }
        assert expected == field_names

    def test_experiment_default_is_experiment_config(self):
        config = MainConfig()
        assert isinstance(config.experiment, ExperimentConfig)

    def test_task_default_is_task_space_config(self):
        config = MainConfig()
        assert isinstance(config.task, TaskSpaceConfig)

    def test_training_default_is_training_config(self):
        config = MainConfig()
        assert isinstance(config.training, TrainingConfig)

    def test_policy_default_is_policy_config(self):
        config = MainConfig()
        assert isinstance(config.policy, PolicyConfig)

    def test_inference_default_is_inference_config(self):
        config = MainConfig()
        assert isinstance(config.inference, InferenceConfig)

    def test_defaults_list_has_expected_structure(self):
        config = MainConfig()
        assert len(config.defaults) == 5
