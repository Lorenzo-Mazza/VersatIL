import pytest
import torch

from refactoring.models.encoding.encoders.multimodal.vlm import VLMEncoder
from refactoring.models.encoding.encoders.constants import (
    PoolingMethod,
    ImageTextModelType,
    EncoderOutputKeys
)
from refactoring.data.constants import Cameras, LANGUAGE_KEY


IMAGE_TEXT_MODELS_TO_OUTPUT_DIM = [
    (ImageTextModelType.CLIP_VITB32.value, 768, 512),  # (model, vision_dim, text_dim)
    (ImageTextModelType.CLIP_VITB16.value, 768, 512),
    (ImageTextModelType.SIGLIP_BASE_PATCH16.value, 768, 768),
]

FEATURE_EXTRACTION_METHODS = [
    PoolingMethod.DEFAULT.value,
    PoolingMethod.AVERAGE.value,
    PoolingMethod.LEARNED_AGGREGATION.value,
    PoolingMethod.NONE.value,
]


@pytest.fixture
def text_inputs(batch_size):
    """Sample text inputs for batch."""
    return ["a robot arm picking up an object"] * batch_size


@pytest.fixture
def text_inputs_5d(batch_size, temporal_length):
    """Sample text inputs for 5D temporal input as 2D list (batch action_embedding time)."""
    text = "a robot arm picking up an object"
    return [[text for _ in range(temporal_length)] for _ in range(batch_size)]


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def temporal_length():
    return 4


@pytest.fixture
def image_size():
    return (224, 224)


@pytest.fixture
def input_dict_4d(batch_size, image_size):
    H, W = image_size
    return {"rgb": torch.randn(batch_size, 3, H, W)}


@pytest.fixture
def input_dict_5d(batch_size, temporal_length, image_size):
    H, W = image_size
    return {"rgb": torch.randn(batch_size, temporal_length, 3, H, W)}

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CLIP")
@pytest.mark.integration
class TestImageTextEncoderInitialization:
    """Test VLMEncoder initialization."""

    @pytest.mark.parametrize("model_name,vision_dim,text_dim", IMAGE_TEXT_MODELS_TO_OUTPUT_DIM)
    def test_initialization(self, model_name, vision_dim, text_dim):
        """Test VLMEncoder initialization with different models."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=model_name,
            pretrained=True,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        assert encoder.hidden_vision_dim == vision_dim
        output_spec = encoder.get_output_specification()
        assert output_spec.dimensions[EncoderOutputKeys.RGB.value] == vision_dim
        assert output_spec.dimensions[EncoderOutputKeys.LANGUAGE.value] == text_dim
        assert encoder.feature_extraction_method == PoolingMethod.AVERAGE.value

    def test_get_output_specification(self):
        """Test get_output_specification returns correct structure."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        output_spec = encoder.get_output_specification()
        assert set(output_spec.features) == {EncoderOutputKeys.RGB.value, EncoderOutputKeys.LANGUAGE.value}
        assert isinstance(output_spec.features, list)
        assert len(output_spec.features) == 2
        assert isinstance(output_spec.dimensions, dict)

    def test_input_specification(self):
        """Test input specification is correctly set."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        assert set(encoder.input_specification.keys) == {Cameras.LEFT.value, LANGUAGE_KEY}
        assert LANGUAGE_KEY in encoder.input_specification.required
        assert [Cameras.LEFT.value, Cameras.RIGHT.value] in encoder.input_specification.one_of_groups

    def test_init_missing_camera(self):
        """Test initialization without camera key raises error."""
        with pytest.raises(ValueError, match="Exactly one from"):
            VLMEncoder(
                input_keys=[LANGUAGE_KEY],
                model_name=ImageTextModelType.CLIP_VITB32.value,
                pretrained=False,
                frozen=True,
                feature_extraction_method=PoolingMethod.AVERAGE.value,
            )

    def test_init_missing_language(self):
        """Test initialization without language key raises error."""
        with pytest.raises(ValueError, match="Missing required inputs"):
            VLMEncoder(
                input_keys=[Cameras.LEFT.value],
                model_name=ImageTextModelType.CLIP_VITB32.value,
                pretrained=False,
                frozen=True,
                feature_extraction_method=PoolingMethod.AVERAGE.value,
            )

    def test_init_multiple_cameras(self):
        """Test initialization with multiple camera keys raises error."""
        with pytest.raises(ValueError, match="Exactly one from"):
            VLMEncoder(
                input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value, LANGUAGE_KEY],
                model_name=ImageTextModelType.CLIP_VITB32.value,
                pretrained=False,
                frozen=True,
                feature_extraction_method=PoolingMethod.AVERAGE.value,
            )

    def test_init_frozen(self):
        """Test initialization with frozen weights."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        for param in encoder.encoder.parameters():
            assert not param.requires_grad

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CLIP")
class TestImageTextEncoderForward:
    """Test VLMEncoder forward pass."""

    @pytest.mark.parametrize("model_name,vision_dim,text_dim", IMAGE_TEXT_MODELS_TO_OUTPUT_DIM)
    def test_forward_4d_input(self, model_name, vision_dim, text_dim, input_dict_4d, text_inputs, batch_size):
        """Test forward pass with 4D input."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=model_name,
            pretrained=False,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"].cuda(),
            LANGUAGE_KEY: text_inputs
        }
        output_dict = encoder(input_dict)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.RGB.value in output_dict
        assert EncoderOutputKeys.LANGUAGE.value in output_dict

        image_output = output_dict[EncoderOutputKeys.RGB.value]
        text_output = output_dict[EncoderOutputKeys.LANGUAGE.value]

        assert image_output.shape == (batch_size, vision_dim)
        assert text_output.shape == (batch_size, text_dim)
        assert image_output.dtype == torch.float32
        assert text_output.dtype == torch.float32
        assert not torch.isnan(image_output).any()
        assert not torch.isnan(text_output).any()

    @pytest.mark.parametrize("model_name,vision_dim,text_dim", IMAGE_TEXT_MODELS_TO_OUTPUT_DIM)
    def test_forward_5d_input(self, model_name, vision_dim, text_dim, input_dict_5d, text_inputs_5d, batch_size, temporal_length):
        """Test forward pass with 5D temporal input."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=model_name,
            pretrained=False,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        input_dict = {
            Cameras.LEFT.value: input_dict_5d["rgb"].cuda(),
            LANGUAGE_KEY: text_inputs_5d
        }
        output_dict = encoder(input_dict)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.RGB.value in output_dict
        assert EncoderOutputKeys.LANGUAGE.value in output_dict

        image_output = output_dict[EncoderOutputKeys.RGB.value]
        text_output = output_dict[EncoderOutputKeys.LANGUAGE.value]

        assert image_output.shape == (batch_size, temporal_length, vision_dim)
        assert text_output.shape == (batch_size, temporal_length, text_dim)
        assert not torch.isnan(image_output).any()
        assert not torch.isnan(text_output).any()

    @pytest.mark.parametrize("feature_method", FEATURE_EXTRACTION_METHODS)
    def test_feature_extraction_methods(self, feature_method, input_dict_4d, text_inputs, batch_size):
        """Test different feature extraction methods."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            feature_extraction_method=feature_method,
        ).to("cuda")
        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"].cuda(),
            LANGUAGE_KEY: text_inputs
        }
        output_dict = encoder(input_dict)
        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.RGB.value in output_dict
        assert EncoderOutputKeys.LANGUAGE.value in output_dict
        image_output = output_dict[EncoderOutputKeys.RGB.value]
        text_output = output_dict[EncoderOutputKeys.LANGUAGE.value]
        if feature_method != PoolingMethod.NONE.value:
            assert image_output.shape == (batch_size, 768)
            assert text_output.shape == (batch_size, 512)
        else:
            assert image_output.ndim == 3  # (B, N, C)
            assert text_output.ndim == 3  # (B, N, C)
            assert image_output.shape[0] == batch_size
            assert text_output.shape[0] == batch_size
            assert image_output.shape[2] == 768
            assert text_output.shape[2] == 512


    def test_forward_output_keys_match_specification(self, input_dict_4d, text_inputs):
        """Test forward output keys match specification."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"].cuda(),
            LANGUAGE_KEY: text_inputs
        }
        output_dict = encoder(input_dict)
        output_spec = encoder.get_output_specification()

        assert set(output_dict.keys()) == set(output_spec.features)

    @pytest.mark.parametrize("feature_method", FEATURE_EXTRACTION_METHODS)
    def test_forward_dimensions_match_specification(self, feature_method, input_dict_4d, text_inputs):
        """Test forward output dimensions match specification."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            feature_extraction_method=feature_method,
        ).to("cuda")

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"].cuda(),
            LANGUAGE_KEY: text_inputs
        }
        output_dict = encoder(input_dict)
        output_spec = encoder.get_output_specification()

        for key in output_spec.features:
            spec_dim = output_spec.dimensions[key]
            assert output_dict[key].shape[-1] == spec_dim

    def test_gradients_enabled_unfrozen(self, input_dict_4d, text_inputs):
        """Test gradients flow when not frozen."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=False,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"].cuda().requires_grad_(True),
            LANGUAGE_KEY: text_inputs
        }
        output_dict = encoder(input_dict)

        assert output_dict[EncoderOutputKeys.RGB.value].requires_grad

        encoder_params_trainable = any(param.requires_grad for param in encoder.encoder.parameters())
        assert encoder_params_trainable

    def test_eval_mode_determinism(self, input_dict_4d, text_inputs):
        """Test encoder produces deterministic outputs in eval mode."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        encoder.eval()

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"].cuda(),
            LANGUAGE_KEY: text_inputs
        }

        with torch.no_grad():
            output1 = encoder(input_dict)
            output2 = encoder(input_dict)

        assert torch.allclose(output1[EncoderOutputKeys.RGB.value], output2[EncoderOutputKeys.RGB.value], atol=1e-6)
        assert torch.allclose(output1[EncoderOutputKeys.LANGUAGE.value], output2[EncoderOutputKeys.LANGUAGE.value], atol=1e-6)

    def test_multiple_camera_types(self):
        """Test encoders work with different camera types."""
        text_inputs = ["a robot arm"] * 2

        for camera in [Cameras.LEFT.value, Cameras.RIGHT.value]:
            encoder = VLMEncoder(
                input_keys=[camera, LANGUAGE_KEY],
                model_name=ImageTextModelType.CLIP_VITB32.value,
                pretrained=False,
                frozen=True,
                feature_extraction_method=PoolingMethod.AVERAGE.value,
            ).to("cuda")

            input_dict = {
                camera: torch.randn(2, 3, 224, 224).cuda(),
                LANGUAGE_KEY: text_inputs
            }
            output_dict = encoder(input_dict)

            assert output_dict[EncoderOutputKeys.RGB.value].shape == (2, 768)
            assert output_dict[EncoderOutputKeys.LANGUAGE.value].shape == (2, 512)


@pytest.mark.integration
class TestImageTextEncoderOutputSpecification:
    """Test encoder output specification methods."""

    @pytest.mark.parametrize("model_name,vision_dim,text_dim", IMAGE_TEXT_MODELS_TO_OUTPUT_DIM)
    def test_output_specification_structure(self, model_name, vision_dim, text_dim):
        """Test output specification returns proper structure."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=model_name,
            pretrained=False,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        output_spec = encoder.get_output_specification()

        assert isinstance(output_spec.dimensions, dict)
        assert len(output_spec.dimensions) == 2
        assert EncoderOutputKeys.RGB.value in output_spec.dimensions
        assert EncoderOutputKeys.LANGUAGE.value in output_spec.dimensions
        assert output_spec.dimensions[EncoderOutputKeys.RGB.value] == vision_dim
        assert output_spec.dimensions[EncoderOutputKeys.LANGUAGE.value] == text_dim
        assert all(isinstance(k, str) for k in output_spec.dimensions.keys())
        assert all(isinstance(v, int) for v in output_spec.dimensions.values())

    def test_output_specification_multi_output(self):
        """Test output specification correctly identifies multi-output encoder."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        )

        output_spec = encoder.get_output_specification()
        assert output_spec.is_multi_output is True


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CLIP")
class TestImageTextEncoderIntegration:
    """Integration tests for complete workflows."""

    def test_complete_forward_backward_pass(self, input_dict_4d, text_inputs):
        """Test complete forward and backward pass."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=False,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        encoder.train()
        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"].cuda().requires_grad_(True),
            LANGUAGE_KEY: text_inputs
        }
        output_dict = encoder(input_dict)
        loss = output_dict[EncoderOutputKeys.RGB.value].mean() + output_dict[EncoderOutputKeys.LANGUAGE.value].mean()
        loss.backward()

        assert output_dict[EncoderOutputKeys.RGB.value].requires_grad
        assert loss.requires_grad

    def test_eval_mode_no_gradients(self, input_dict_4d, text_inputs):
        """Test eval mode doesn't compute gradients."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        encoder.eval()
        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"].cuda(),
            LANGUAGE_KEY: text_inputs
        }

        with torch.no_grad():
            output_dict = encoder(input_dict)

        assert not output_dict[EncoderOutputKeys.RGB.value].requires_grad
        assert not output_dict[EncoderOutputKeys.LANGUAGE.value].requires_grad

    def test_consistent_output_shapes(self, batch_size, image_size):
        """Test output shapes are consistent across multiple forward passes."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        H, W = image_size
        text_inputs = ["test text"] * batch_size

        input1 = {
            Cameras.LEFT.value: torch.randn(batch_size, 3, H, W).cuda(),
            LANGUAGE_KEY: text_inputs
        }
        input2 = {
            Cameras.LEFT.value: torch.randn(batch_size, 3, H, W).cuda(),
            LANGUAGE_KEY: text_inputs
        }

        output1 = encoder(input1)
        output2 = encoder(input2)

        assert output1[EncoderOutputKeys.RGB.value].shape == output2[EncoderOutputKeys.RGB.value].shape
        assert output1[EncoderOutputKeys.LANGUAGE.value].shape == output2[EncoderOutputKeys.LANGUAGE.value].shape

    def test_text_image_alignment(self, batch_size):
        """Test that image and text features are in the same embedding space."""
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, LANGUAGE_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            feature_extraction_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        encoder.eval()

        # Create inputs with matching text
        images = torch.randn(batch_size, 3, 224, 224).cuda()
        texts = ["a photo of a cat"] * batch_size

        input_dict = {
            Cameras.LEFT.value: images,
            LANGUAGE_KEY: texts
        }

        with torch.no_grad():
            output = encoder(input_dict)
            image_features = output[EncoderOutputKeys.RGB.value]
            text_features = output[EncoderOutputKeys.LANGUAGE.value]

        # Normalize features (CLIP uses normalized embeddings)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = (image_features * text_features).sum(dim=-1)

        # All similarities should be in valid range [-1, 1]
        assert (similarity >= -1.0).all()
        assert (similarity <= 1.0).all()