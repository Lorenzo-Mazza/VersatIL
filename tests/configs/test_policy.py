"""Tests for versatil.configs.policy module."""
import dataclasses

import pytest
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf, flag_override

from versatil.configs.data.task import ActionSpaceConfig, ObservationSpaceConfig
from versatil.configs.decoding.algorithm import BehavioralCloningConfig
from versatil.configs.decoding.decoder import ACTConfig
from versatil.configs.encoding.encoder import ProprioEncoderConfig
from versatil.configs.encoding.pipeline import EncodingPipelineConfig
from versatil.configs.loss import CompositeLossConfig
from versatil.configs.policy import PolicyConfig
from versatil.models.policy import Policy


@pytest.mark.unit
class TestPolicyConfig:

    def test_target_points_to_policy_class(self):
        config = PolicyConfig()
        assert config._target_ == "versatil.models.policy.Policy"

    def test_encoding_pipeline_defaults_to_missing(self):
        config = PolicyConfig()
        assert config.encoding_pipeline == MISSING

    def test_algorithm_defaults_to_missing(self):
        config = PolicyConfig()
        assert config.algorithm == MISSING

    def test_decoder_defaults_to_missing(self):
        config = PolicyConfig()
        assert config.decoder == MISSING

    def test_loss_defaults_to_missing(self):
        config = PolicyConfig()
        assert config.loss == MISSING

    @pytest.mark.parametrize("validate_loss_keys", [True, False])
    def test_stores_validate_loss_keys(self, validate_loss_keys):
        config = PolicyConfig(validate_loss_keys=validate_loss_keys)
        assert config.validate_loss_keys == validate_loss_keys

    def test_has_all_expected_fields(self):
        field_names = {f.name for f in dataclasses.fields(PolicyConfig)}
        expected = {
            "_target_",
            "encoding_pipeline",
            "algorithm",
            "decoder",
            "observation_space",
            "action_space",
            "prediction_horizon",
            "observation_horizon",
            "device",
            "loss",
            "validate_loss_keys",
        }
        assert expected == field_names

    def test_interpolation_references_in_defaults(self):
        config = PolicyConfig()
        # These use Hydra interpolation strings as defaults
        assert config.observation_space == "${task.observation_space}"
        assert config.action_space == "${task.action_space}"
        assert config.prediction_horizon == "${task.prediction_horizon}"
        assert config.observation_horizon == "${task.observation_horizon}"
        assert config.device == "${experiment.device}"


@pytest.mark.unit
class TestPolicyInstantiation:

    def test_policy_instantiates_with_real_configs(self):
        observation_space = ObservationSpaceConfig()
        action_space = ActionSpaceConfig()
        encoding_pipeline = EncodingPipelineConfig(
            encoders={
                "proprio": ProprioEncoderConfig(
                    input_keys=["proprio"],
                    output_dim=64,
                    pretrained=False,
                ),
            },
        )
        decoder_config = ACTConfig(input_keys=["proprio_embedding"])
        algorithm_config = BehavioralCloningConfig()
        loss_config = CompositeLossConfig(loss_modules={})

        config = PolicyConfig(
            encoding_pipeline=encoding_pipeline,
            algorithm=algorithm_config,
            decoder=decoder_config,
            loss=loss_config,
        )

        structured = OmegaConf.structured(config)
        with flag_override(structured, "struct", False):
            structured.observation_space = observation_space
            structured.action_space = action_space
            structured.prediction_horizon = 10
            structured.observation_horizon = 2
            structured.device = "cpu"
            structured.decoder.observation_space = observation_space
            structured.decoder.action_space = action_space
            structured.decoder.prediction_horizon = 10
            structured.decoder.observation_horizon = 2
            structured.decoder.device = "cpu"
            structured.decoder.action_heads = {}

        instance = instantiate(structured)
        assert isinstance(instance, Policy)
