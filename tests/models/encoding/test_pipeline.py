"""Tests for EncodingPipeline with real encoder implementations."""
import pytest
import torch
import logging
from unittest.mock import MagicMock

from refactoring.models.encoding.pipeline import EncodingPipeline
from refactoring.models.encoding.encoders.base import EncodingMixin, EncoderInput, EncoderOutput
from tests.conftest import DummyRGBEncoder, DummyDepthEncoder, DummyProprioEncoder


class DummyEncoderWithMixin(EncodingMixin):
    """Minimal encoder that inherits from EncodingMixin for testing tokenizer propagation."""

    def __init__(self):
        input_spec = EncoderInput(keys=["test_input"])
        super().__init__(input_specification=input_spec, pretrained=False)
        self.name = "test_encoder"
        self.linear = torch.nn.Linear(10, 128)

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=["features"],
            dimensions={"features": 128}
        )

    def forward(self, inputs: dict) -> dict:
        return {"features": torch.randn(inputs["test_input"].shape[0], 128)}


class DummyMultiOutputEncoder(torch.nn.Module):
    """Encoder that produces multiple output features (like VLM)."""

    def __init__(self):
        super().__init__()
        self.name = "vlm"
        self.conv = torch.nn.Conv2d(3, 512, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.visual_proj = torch.nn.Linear(512, 512)
        self.language_proj = torch.nn.Linear(512, 768)
        self.input_specification = EncoderInput(keys=["image"])
        self._output_spec = EncoderOutput(
            features=["visual", "language"],
            dimensions={"visual": 512, "language": 768}
        )

    def get_output_specification(self):
        return self._output_spec

    def forward(self, inputs: dict) -> dict:
        image = inputs["image"]
        x = self.conv(image)
        x = self.pool(x).flatten(1)
        return {
            "visual": self.visual_proj(x),
            "language": self.language_proj(x),
        }


class SimpleFusion(torch.nn.Module):
    """Simple fusion that concatenates and projects features."""

    def __init__(self, input_features, output_name, output_dim):
        super().__init__()
        self.input_features = input_features
        self.output_name = output_name
        self.output_dim = output_dim
        self.linear = None

    def setup(self, feature_keys_to_dims):
        input_dim = 0
        for feat_name in self.input_features:
            dims = feature_keys_to_dims[feat_name]
            if isinstance(dims, tuple):
                dim = 1
                for d in dims:
                    dim *= d
                input_dim += dim
            else:
                input_dim += dims
        self.linear = torch.nn.Linear(input_dim, self.output_dim)

    def forward(self, input_features):
        flattened = []
        for feat in input_features:
            if feat.dim() > 2:
                feat = feat.flatten(1)
            flattened.append(feat)
        concatenated = torch.cat(flattened, dim=-1)
        return self.linear(concatenated)

    def get_output_dim(self):
        return self.output_dim


@pytest.fixture
def rgb_encoder_factory():
    """Factory for creating RGB encoders."""
    def factory(**kwargs):
        encoder = DummyRGBEncoder()
        for key, value in kwargs.items():
            setattr(encoder, key, value)
        return encoder
    return factory


@pytest.fixture
def depth_encoder_factory():
    """Factory for creating depth encoders."""
    def factory(**kwargs):
        encoder = DummyDepthEncoder()
        for key, value in kwargs.items():
            setattr(encoder, key, value)
        return encoder
    return factory


@pytest.fixture
def proprio_encoder_factory():
    """Factory for creating proprioceptive encoders."""
    def factory(input_dim=7, output_dim=128, **kwargs):
        encoder = DummyProprioEncoder(input_dim=input_dim, output_dim=output_dim)
        for key, value in kwargs.items():
            setattr(encoder, key, value)
        return encoder
    return factory


@pytest.fixture
def multi_output_encoder_factory():
    """Factory for creating multi-output encoders."""
    def factory(**kwargs):
        encoder = DummyMultiOutputEncoder()
        for key, value in kwargs.items():
            setattr(encoder, key, value)
        return encoder
    return factory


@pytest.fixture
def fusion_factory():
    """Factory for creating fusion modules."""
    def factory(input_features, output_name="fused", output_dim=256):
        return SimpleFusion(input_features, output_name, output_dim)
    return factory


@pytest.mark.unit
class TestEncodingPipelineBasic:
    """Test basic encoding pipeline initialization and properties."""

    def test_init_single_encoder(self, rgb_encoder_factory):
        """Test initializing pipeline with single encoder."""
        rgb_encoder = rgb_encoder_factory()
        pipeline = EncodingPipeline(encoders={"rgb": rgb_encoder})

        assert len(pipeline.encoders) == 1
        assert "rgb" in pipeline.encoders
        assert "rgb_features" in pipeline.get_feature_names()

    def test_init_multiple_encoders(self, rgb_encoder_factory, depth_encoder_factory, proprio_encoder_factory):
        """Test initializing pipeline with multiple encoders."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "depth": depth_encoder_factory(),
            "proprio": proprio_encoder_factory()
        }
        pipeline = EncodingPipeline(encoders=encoders)

        assert len(pipeline.encoders) == 3
        feature_names = pipeline.get_feature_names()
        assert "rgb_features" in feature_names
        assert "depth_features" in feature_names
        assert "proprio_features" in feature_names

    def test_get_feature_names(self, rgb_encoder_factory, proprio_encoder_factory):
        """Test getting all feature names from pipeline."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "proprio": proprio_encoder_factory()
        }
        pipeline = EncodingPipeline(encoders=encoders)

        feature_names = pipeline.get_feature_names()
        assert "rgb_features" in feature_names
        assert "proprio_features" in feature_names

    def test_get_features_to_dimensions(self, rgb_encoder_factory, proprio_encoder_factory):
        """Test getting feature dimensions mapping."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "proprio": proprio_encoder_factory(output_dim=128)
        }
        pipeline = EncodingPipeline(encoders=encoders)

        feature_dims = pipeline.get_features_to_dimensions()
        assert feature_dims["rgb_features"] == (256, 7, 7)
        assert feature_dims["proprio_features"] == 128

    def test_empty_pipeline(self):
        """Test empty pipeline with no encoders."""
        pipeline = EncodingPipeline(encoders={})

        assert len(pipeline.encoders) == 0
        assert len(pipeline.get_feature_names()) == 0


@pytest.mark.unit
class TestEncodingPipelineForward:
    """Test encoding pipeline forward pass."""

    def test_forward_single_encoder(self, rgb_encoder_factory, batch_size, device):
        """Test forward pass with single encoder."""
        rgb_encoder = rgb_encoder_factory()
        pipeline = EncodingPipeline(encoders={"rgb": rgb_encoder}).to(device)

        observations = {"rgb": torch.randn(batch_size, 3, 224, 224, device=device)}
        features = pipeline(observations)

        assert "rgb_features" in features
        assert features["rgb_features"].shape == (batch_size, 256, 7, 7)
        assert features["rgb_features"].device.type == device.type

    def test_forward_multiple_encoders(self, rgb_encoder_factory, depth_encoder_factory,
                                      proprio_encoder_factory, batch_size, device):
        """Test forward pass with multiple encoders."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "depth": depth_encoder_factory(),
            "proprio": proprio_encoder_factory()
        }
        pipeline = EncodingPipeline(encoders=encoders).to(device)

        observations = {
            "rgb": torch.randn(batch_size, 3, 224, 224, device=device),
            "depth": torch.randn(batch_size, 1, 224, 224, device=device),
            "proprio": torch.randn(batch_size, 7, device=device),
        }
        features = pipeline(observations)

        assert "rgb_features" in features
        assert "depth_features" in features
        assert "proprio_features" in features
        assert features["rgb_features"].shape == (batch_size, 256, 7, 7)
        assert features["depth_features"].shape == (batch_size, 128, 7, 7)
        assert features["proprio_features"].shape == (batch_size, 128)

    def test_forward_mixed_spatial_and_flat(self, rgb_encoder_factory, proprio_encoder_factory,
                                           batch_size, device):
        """Test forward with mix of spatial and flat features."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "proprio": proprio_encoder_factory()
        }
        pipeline = EncodingPipeline(encoders=encoders).to(device)

        observations = {
            "rgb": torch.randn(batch_size, 3, 224, 224, device=device),
            "proprio": torch.randn(batch_size, 7, device=device),
        }
        features = pipeline(observations)

        assert features["rgb_features"].shape == (batch_size, 256, 7, 7)
        assert len(features["rgb_features"].shape) == 4

        assert features["proprio_features"].shape == (batch_size, 128)
        assert len(features["proprio_features"].shape) == 2


@pytest.mark.unit
class TestEncodingPipelineParametrized:
    """Parametrized tests for encoding pipeline with different configurations."""

    @pytest.mark.parametrize("output_dim", [64, 128, 256, 512])
    def test_different_output_dimensions(self, proprio_encoder_factory, output_dim, batch_size, device):
        """Test pipeline with different proprioceptive output dimensions."""
        proprio_encoder = proprio_encoder_factory(output_dim=output_dim)
        pipeline = EncodingPipeline(encoders={"proprio": proprio_encoder}).to(device)

        observations = {"proprio": torch.randn(batch_size, 7, device=device)}
        features = pipeline(observations)

        assert features["proprio_features"].shape == (batch_size, output_dim)

    @pytest.mark.parametrize("test_batch_size", [1, 4, 8, 16])
    def test_different_batch_sizes(self, rgb_encoder_factory, test_batch_size, device):
        """Test pipeline with different batch sizes."""
        rgb_encoder = rgb_encoder_factory()
        pipeline = EncodingPipeline(encoders={"rgb": rgb_encoder}).to(device)

        observations = {"rgb": torch.randn(test_batch_size, 3, 224, 224, device=device)}
        features = pipeline(observations)

        assert features["rgb_features"].shape[0] == test_batch_size

    @pytest.mark.parametrize("num_spatial_encoders", [1, 2, 3])
    def test_different_number_of_spatial_encoders(self, rgb_encoder_factory, num_spatial_encoders,
                                                  batch_size, device):
        """Test pipeline with different numbers of spatial encoders."""
        encoders = {}
        observations = {}

        for i in range(num_spatial_encoders):
            encoder = rgb_encoder_factory()
            encoder.name = f"encoder_{i}"
            encoder.input_specification = EncoderInput(keys=[f"input_{i}"])
            encoders[f"encoder_{i}"] = encoder
            observations[f"input_{i}"] = torch.randn(batch_size, 3, 224, 224, device=device)

        pipeline = EncodingPipeline(encoders=encoders).to(device)
        features = pipeline(observations)

        assert len(features) == num_spatial_encoders
        for i in range(num_spatial_encoders):
            assert f"encoder_{i}_features" in features


@pytest.mark.unit
class TestEncodingPipelineMultiOutput:
    """Test encoding pipeline with multi-output encoders."""

    def test_multi_output_encoder(self, multi_output_encoder_factory, batch_size, device):
        """Test encoder that produces multiple outputs (like VLM)."""
        vlm_encoder = multi_output_encoder_factory()
        pipeline = EncodingPipeline(encoders={"vlm": vlm_encoder}).to(device)

        observations = {"image": torch.randn(batch_size, 3, 224, 224, device=device)}
        features = pipeline(observations)

        assert "vlm_visual" in features
        assert "vlm_language" in features
        assert features["vlm_visual"].shape == (batch_size, 512)
        assert features["vlm_language"].shape == (batch_size, 768)


@pytest.mark.unit
class TestEncodingPipelineEdgeCases:
    """Test edge cases for encoding pipeline."""

    def test_single_sample_batch(self, rgb_encoder_factory, device):
        """Test pipeline with single sample."""
        rgb_encoder = rgb_encoder_factory()
        pipeline = EncodingPipeline(encoders={"rgb": rgb_encoder}).to(device)

        observations = {"rgb": torch.randn(1, 3, 224, 224, device=device)}
        features = pipeline(observations)

        assert features["rgb_features"].shape == (1, 256, 7, 7)

    def test_large_batch(self, rgb_encoder_factory, device):
        """Test pipeline with large batch size."""
        rgb_encoder = rgb_encoder_factory()
        pipeline = EncodingPipeline(encoders={"rgb": rgb_encoder}).to(device)

        batch_size = 32
        observations = {"rgb": torch.randn(batch_size, 3, 224, 224, device=device)}
        features = pipeline(observations)

        assert features["rgb_features"].shape[0] == batch_size


@pytest.mark.unit
class TestEncodingPipelineValidation:
    """Test validation logic of encoding pipeline."""

    def test_valid_single_encoder_configuration(self, rgb_encoder_factory):
        """Test that valid single encoder configuration passes validation."""
        rgb_encoder = rgb_encoder_factory()
        pipeline = EncodingPipeline(encoders={"rgb": rgb_encoder})

        assert pipeline is not None
        assert len(pipeline.encoders) == 1

    def test_valid_multiple_encoder_configuration(self, rgb_encoder_factory, depth_encoder_factory,
                                                  proprio_encoder_factory):
        """Test that valid multiple encoder configuration passes validation."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "depth": depth_encoder_factory(),
            "proprio": proprio_encoder_factory()
        }
        pipeline = EncodingPipeline(encoders=encoders)

        assert pipeline is not None
        assert len(pipeline.encoders) == 3

    def test_valid_multi_output_encoder_configuration(self, multi_output_encoder_factory):
        """Test that multi-output encoder configuration passes validation."""
        vlm_encoder = multi_output_encoder_factory()
        pipeline = EncodingPipeline(encoders={"vlm": vlm_encoder})

        assert pipeline is not None
        feature_names = pipeline.get_feature_names()
        assert "vlm_visual" in feature_names
        assert "vlm_language" in feature_names

    def test_feature_names_are_prefixed_correctly(self, rgb_encoder_factory):
        """Test that encoder outputs are correctly prefixed with encoder name."""
        rgb_encoder = rgb_encoder_factory()
        pipeline = EncodingPipeline(encoders={"rgb": rgb_encoder})

        feature_names = pipeline.get_feature_names()
        assert "rgb_features" in feature_names
        assert all(name.startswith("rgb_") for name in feature_names)

    def test_multiple_encoders_unique_feature_names(self, rgb_encoder_factory, depth_encoder_factory):
        """Test that multiple encoders produce uniquely named features."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "depth": depth_encoder_factory()
        }
        pipeline = EncodingPipeline(encoders=encoders)

        feature_names = pipeline.get_feature_names()
        assert len(feature_names) == len(set(feature_names))
        assert "rgb_features" in feature_names
        assert "depth_features" in feature_names

    def test_get_features_to_dimensions_returns_correct_mapping(self, rgb_encoder_factory,
                                                                proprio_encoder_factory):
        """Test that feature dimensions mapping is correct."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "proprio": proprio_encoder_factory(output_dim=256)
        }
        pipeline = EncodingPipeline(encoders=encoders)

        feature_dims = pipeline.get_features_to_dimensions()
        assert feature_dims["rgb_features"] == (256, 7, 7)
        assert feature_dims["proprio_features"] == 256

    def test_encoder_output_matches_specification(self, rgb_encoder_factory, batch_size, device):
        """Test that encoder forward output matches its specification."""
        rgb_encoder = rgb_encoder_factory()
        pipeline = EncodingPipeline(encoders={"rgb": rgb_encoder}).to(device)

        observations = {"rgb": torch.randn(batch_size, 3, 224, 224, device=device)}
        features = pipeline(observations)

        assert "rgb_features" in features
        output_spec = rgb_encoder.get_output_specification()
        expected_dims = output_spec.dimensions["features"]
        actual_shape = features["rgb_features"].shape[1:]
        assert actual_shape == expected_dims

    def test_missing_encoder_input_logs_warning(self, rgb_encoder_factory, batch_size, device, caplog):
        """Test that pipeline logs warning when encoder input is missing."""
        rgb_encoder = rgb_encoder_factory()
        rgb_encoder.input_specification = EncoderInput(keys=["rgb", "missing_key"])
        pipeline = EncodingPipeline(encoders={"rgb": rgb_encoder}).to(device)

        observations = {"rgb": torch.randn(batch_size, 3, 224, 224, device=device)}

        with caplog.at_level(logging.WARNING):
            features = pipeline(observations)

        assert any("missing" in record.message.lower() for record in caplog.records)
        assert "rgb_features" not in features

    def test_multiple_encoders_independent_validation(self, rgb_encoder_factory, depth_encoder_factory,
                                                     proprio_encoder_factory):
        """Test that encoders are validated independently."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "depth": depth_encoder_factory(),
            "proprio": proprio_encoder_factory()
        }
        pipeline = EncodingPipeline(encoders=encoders)

        assert len(pipeline.encoders) == 3
        feature_names = pipeline.get_feature_names()
        assert "rgb_features" in feature_names
        assert "depth_features" in feature_names
        assert "proprio_features" in feature_names


@pytest.mark.unit
class TestEncodingPipelineGradients:
    """Test gradient flow through encoding pipeline."""

    def test_backward_pass_single_encoder(self, rgb_encoder_factory, batch_size, device):
        """Test that gradients flow through pipeline with single encoder."""
        rgb_encoder = rgb_encoder_factory()
        pipeline = EncodingPipeline(encoders={"rgb": rgb_encoder}).to(device)

        observations = {"rgb": torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)}
        features = pipeline(observations)

        loss = features["rgb_features"].sum()
        loss.backward()

        assert observations["rgb"].grad is not None
        assert not torch.isnan(observations["rgb"].grad).any()

    def test_backward_pass_multiple_encoders(self, rgb_encoder_factory, depth_encoder_factory,
                                            batch_size, device):
        """Test gradient flow with multiple encoders."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "depth": depth_encoder_factory()
        }
        pipeline = EncodingPipeline(encoders=encoders).to(device)

        observations = {
            "rgb": torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True),
            "depth": torch.randn(batch_size, 1, 224, 224, device=device, requires_grad=True),
        }
        features = pipeline(observations)

        loss = features["rgb_features"].sum() + features["depth_features"].sum()
        loss.backward()

        assert observations["rgb"].grad is not None
        assert observations["depth"].grad is not None
        assert not torch.isnan(observations["rgb"].grad).any()
        assert not torch.isnan(observations["depth"].grad).any()


@pytest.mark.unit
class TestEncodingPipelineFeatureConsumption:
    """Test that fusion modules properly consume their input features."""

    def test_no_fusion_no_consumption(self, rgb_encoder_factory, proprio_encoder_factory):
        """Without fusion, all encoder features should be available."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "proprio": proprio_encoder_factory()
        }
        pipeline = EncodingPipeline(encoders=encoders)

        assert len(pipeline._consumed_features) == 0

        all_features = set(pipeline.get_feature_names())
        final_features = set(pipeline.get_final_feature_names())
        assert all_features == final_features
        assert "rgb_features" in final_features
        assert "proprio_features" in final_features

    def test_single_fusion_consumes_inputs(self, rgb_encoder_factory, proprio_encoder_factory,
                                          fusion_factory, batch_size):
        """Fusion module should consume its input features."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "proprio": proprio_encoder_factory()
        }
        fusion = fusion_factory(
            input_features=["rgb_features", "proprio_features"],
            output_name="fused",
            output_dim=256
        )
        pipeline = EncodingPipeline(encoders=encoders, fusion_stages=[fusion])

        assert "rgb_features" in pipeline._consumed_features
        assert "proprio_features" in pipeline._consumed_features

        final_features = set(pipeline.get_final_feature_names())
        assert "rgb_features" not in final_features
        assert "proprio_features" not in final_features
        assert "fused" in final_features

        observations = {
            "rgb": torch.randn(batch_size, 3, 224, 224),
            "proprio": torch.randn(batch_size, 7),
        }
        features = pipeline(observations)
        assert "rgb_features" not in features
        assert "proprio_features" not in features
        assert "fused" in features

    def test_chained_fusion_consumes_transitively(self, proprio_encoder_factory, fusion_factory, batch_size):
        """Chained fusions should consume all intermediate features."""
        encoder_a = proprio_encoder_factory(output_dim=64)
        encoder_a.name = "a"
        encoder_b = proprio_encoder_factory(output_dim=64)
        encoder_b.name = "b"
        encoder_c = proprio_encoder_factory(output_dim=64)
        encoder_c.name = "c"

        encoders = {"a": encoder_a, "b": encoder_b, "c": encoder_c}
        fusion1 = fusion_factory(
            input_features=["a_features", "b_features"],
            output_name="ab",
            output_dim=128
        )
        fusion2 = fusion_factory(
            input_features=["ab", "c_features"],
            output_name="abc",
            output_dim=256
        )
        pipeline = EncodingPipeline(encoders=encoders, fusion_stages=[fusion1, fusion2])

        assert "a_features" in pipeline._consumed_features
        assert "b_features" in pipeline._consumed_features
        assert "ab" in pipeline._consumed_features
        assert "c_features" in pipeline._consumed_features

        final_features = set(pipeline.get_final_feature_names())
        assert "abc" in final_features
        assert len(final_features) == 1

        observations = {"proprio": torch.randn(batch_size, 7)}
        features = pipeline(observations)
        assert "abc" in features
        assert len(features) == 1

    def test_partial_fusion_keeps_unused_features(self, rgb_encoder_factory, proprio_encoder_factory,
                                                  depth_encoder_factory, fusion_factory):
        """Features not used by fusion should remain available."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "proprio": proprio_encoder_factory(),
            "depth": depth_encoder_factory()
        }
        fusion = fusion_factory(
            input_features=["rgb_features", "proprio_features"],
            output_name="fused",
            output_dim=256
        )
        pipeline = EncodingPipeline(encoders=encoders, fusion_stages=[fusion])

        assert "depth_features" not in pipeline._consumed_features

        final_features = set(pipeline.get_final_feature_names())
        assert "depth_features" in final_features
        assert "fused" in final_features
        assert "rgb_features" not in final_features
        assert "proprio_features" not in final_features

    def test_get_final_features_to_dimensions(self, rgb_encoder_factory, proprio_encoder_factory,
                                             fusion_factory):
        """get_final_features_to_dimensions should exclude consumed features."""
        encoders = {
            "rgb": rgb_encoder_factory(),
            "proprio": proprio_encoder_factory(output_dim=64)
        }
        fusion = fusion_factory(
            input_features=["rgb_features", "proprio_features"],
            output_name="fused",
            output_dim=256
        )
        pipeline = EncodingPipeline(encoders=encoders, fusion_stages=[fusion])

        all_features_to_dims = pipeline.get_features_to_dimensions()
        assert "rgb_features" in all_features_to_dims
        assert "proprio_features" in all_features_to_dims
        assert "fused" in all_features_to_dims

        final_features_to_dims = pipeline.get_final_features_to_dimensions()
        assert "rgb_features" not in final_features_to_dims
        assert "proprio_features" not in final_features_to_dims
        assert "fused" in final_features_to_dims
        assert final_features_to_dims["fused"] == 256


@pytest.mark.unit
class TestEncodingPipelineSetTokenizer:
    """Test EncodingPipeline.set_tokenizer() validation."""

    def test_set_tokenizer_with_none(self, rgb_encoder_factory):
        """Test setting None tokenizer doesn't crash."""
        encoder = rgb_encoder_factory()
        pipeline = EncodingPipeline(encoders={"encoder1": encoder})
        pipeline.set_tokenizer(None)

    def test_set_tokenizer_without_observation_tokenizer(self, rgb_encoder_factory):
        """Test tokenizer without observation_tokenizer doesn't crash."""
        encoder = rgb_encoder_factory()
        pipeline = EncodingPipeline(encoders={"encoder1": encoder})

        tokenizer = MagicMock()
        tokenizer.observation_tokenizer = None
        pipeline.set_tokenizer(tokenizer)