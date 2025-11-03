import dataclasses

import pytest

from refactoring.configs.decoding.algorithm import BehavioralCloningConfig
from refactoring.configs.decoding.decoder import ACTConfig
from refactoring.configs.encoding.pipeline import EncodingPipelineConfig
from refactoring.configs.loss import ActionReconstructionLossConfig
from refactoring.configs.policy import PolicyConfig


@pytest.mark.unit
class TestPolicyConfig:

    def test_config_has_correct_target(self):
        config = PolicyConfig(
            encoding_pipeline=EncodingPipelineConfig(encoders={}),
            algorithm=BehavioralCloningConfig(),
            decoder=ACTConfig(action_heads={}, input_keys=[]),
            loss=ActionReconstructionLossConfig(),
        )
        assert config._target_ == "refactoring.models.policy.Policy"

    def test_config_can_be_instantiated(self):
        config = PolicyConfig(
            encoding_pipeline=EncodingPipelineConfig(encoders={}),
            algorithm=BehavioralCloningConfig(),
            decoder=ACTConfig(action_heads={}, input_keys=[]),
            loss=ActionReconstructionLossConfig(),
        )
        assert isinstance(config, PolicyConfig)
        assert config.validate_loss_keys is True

    def test_config_has_expected_fields(self):
        fields = {f.name for f in dataclasses.fields(PolicyConfig)}
        expected = {
            '_target_', 'encoding_pipeline', 'algorithm', 'decoder',
            'observation_space', 'action_space', 'prediction_horizon',
            'device', 'loss', 'validate_loss_keys'
        }
        assert expected.issubset(fields)
