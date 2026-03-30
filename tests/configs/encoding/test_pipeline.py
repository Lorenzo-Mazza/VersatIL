"""Tests for versatil.configs.encoding.pipeline module."""

import pytest
from hydra.utils import instantiate
from omegaconf import MISSING

from versatil.configs.encoding.encoder import ProprioEncoderConfig
from versatil.configs.encoding.pipeline import EncodingPipelineConfig
from versatil.data.task import ObservationSpace
from versatil.models.encoding.pipeline import EncodingPipeline


@pytest.fixture
def empty_observation_space() -> ObservationSpace:
    """Empty observation space for pipeline config tests."""
    return ObservationSpace(observations_metadata={})


@pytest.mark.unit
class TestEncodingPipelineConfig:
    def test_target_points_to_encoding_pipeline(self):
        config = EncodingPipelineConfig(encoders={})
        assert config._target_ == "versatil.models.encoding.pipeline.EncodingPipeline"

    def test_encoders_required(self):
        config = EncodingPipelineConfig()
        assert config.encoders == MISSING

    def test_fusion_stages_default_to_none(self):
        config = EncodingPipelineConfig(encoders={})
        assert config.fusion_stages is None


@pytest.mark.unit
class TestEncodingPipelineInstantiation:
    def test_empty_pipeline_instantiates(self, empty_observation_space):
        config = EncodingPipelineConfig(
            encoders={},
            observation_space=empty_observation_space,
        )
        instance = instantiate(config)
        assert isinstance(instance, EncodingPipeline)

    def test_pipeline_instantiates_with_encoder(self, empty_observation_space):
        config = EncodingPipelineConfig(
            encoders={
                "proprio": ProprioEncoderConfig(
                    input_keys=["proprio"],
                    output_dim=64,
                    pretrained=False,
                    model_dtype=None,
                ),
            },
            observation_space=empty_observation_space,
        )
        instance = instantiate(config)
        assert isinstance(instance, EncodingPipeline)
