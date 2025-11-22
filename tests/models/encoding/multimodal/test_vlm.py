import pytest
import torch

from refactoring.models.encoding.encoders.multimodal.vlm import VLMEncoder
from refactoring.models.encoding.encoders.constants import (
    PoolingMethod,
    ImageTextModelType,
    EncoderOutputKeys
)
from refactoring.data.constants import Cameras, TOKENIZED_OBSERVATIONS_KEY


IMAGE_TEXT_MODELS_TO_OUTPUT_DIM = [
    (ImageTextModelType.CLIP_VITB32.value, 768, 512),
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
def text_inputs_factory():
    """Factory for creating tokenized text inputs."""
    def factory(batch_size=2, seq_len=77, device="cpu"):
        return torch.randint(0, 49408, (batch_size, seq_len), device=device)
    return factory


@pytest.fixture
def image_inputs_factory():
    """Factory for creating image inputs with customizable shape and device."""
    def factory(batch_size=2, channels=3, height=224, width=224, temporal_length=None, device="cpu"):
        if temporal_length is None:
            return torch.randn(batch_size, channels, height, width, device=device)
        else:
            return torch.randn(batch_size, temporal_length, channels, height, width, device=device)
    return factory


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CLIP")
@pytest.mark.integration
class TestImageTextEncoderInitialization:
    """Test VLMEncoder initialization."""

    @pytest.mark.parametrize("model_name,vision_dim,text_dim", IMAGE_TEXT_MODELS_TO_OUTPUT_DIM)
    def test_initialization(self, model_name, vision_dim, text_dim):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=model_name,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        assert encoder.hidden_vision_dim == vision_dim
        output_spec = encoder.get_output_specification()
        assert output_spec.dimensions[EncoderOutputKeys.RGB.value] == vision_dim
        assert output_spec.dimensions[EncoderOutputKeys.LANGUAGE.value] == text_dim
        assert encoder.pooling_method == PoolingMethod.AVERAGE.value

    def test_get_output_specification(self):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        output_spec = encoder.get_output_specification()
        assert set(output_spec.features) == {EncoderOutputKeys.RGB.value, EncoderOutputKeys.LANGUAGE.value}
        assert isinstance(output_spec.features, list)
        assert len(output_spec.features) == 2
        assert isinstance(output_spec.dimensions, dict)

    def test_input_specification(self):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        assert set(encoder.input_specification.keys) == {Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY}
        assert TOKENIZED_OBSERVATIONS_KEY in encoder.input_specification.required
        assert [Cameras.LEFT.value, Cameras.RIGHT.value] in encoder.input_specification.one_of_groups

    def test_init_missing_camera(self):
        with pytest.raises(ValueError, match="Exactly one from"):
            VLMEncoder(
                input_keys=[TOKENIZED_OBSERVATIONS_KEY],
                model_name=ImageTextModelType.CLIP_VITB32.value,
                pretrained=False,
                frozen=True,
                pooling_method=PoolingMethod.AVERAGE.value,
            )

    def test_init_missing_language(self):
        with pytest.raises(ValueError, match="Missing required inputs"):
            VLMEncoder(
                input_keys=[Cameras.LEFT.value],
                model_name=ImageTextModelType.CLIP_VITB32.value,
                pretrained=False,
                frozen=True,
                pooling_method=PoolingMethod.AVERAGE.value,
            )

    def test_init_multiple_cameras(self):
        with pytest.raises(ValueError, match="Exactly one from"):
            VLMEncoder(
                input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value, TOKENIZED_OBSERVATIONS_KEY],
                model_name=ImageTextModelType.CLIP_VITB32.value,
                pretrained=False,
                frozen=True,
                pooling_method=PoolingMethod.AVERAGE.value,
            )

    def test_init_frozen(self):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        for param in encoder.encoder.parameters():
            assert not param.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CLIP")
@pytest.mark.integration
class TestImageTextEncoderForward:
    """Test VLMEncoder forward pass."""

    @pytest.mark.parametrize("model_name,vision_dim,text_dim", IMAGE_TEXT_MODELS_TO_OUTPUT_DIM)
    def test_forward_4d_input(self, model_name, vision_dim, text_dim, image_inputs_factory, text_inputs_factory):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=model_name,
            pretrained=False,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        batch_size = 2
        images = image_inputs_factory(batch_size=batch_size, device="cuda")
        text = text_inputs_factory(batch_size=batch_size, device="cuda")

        input_dict = {
            Cameras.LEFT.value: images,
            TOKENIZED_OBSERVATIONS_KEY: text
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
    def test_forward_5d_input(self, model_name, vision_dim, text_dim, image_inputs_factory, text_inputs_factory):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=model_name,
            pretrained=False,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        batch_size = 2
        temporal_length = 4
        images = image_inputs_factory(batch_size=batch_size, temporal_length=temporal_length, device="cuda")
        text = text_inputs_factory(batch_size=batch_size, seq_len=77, device="cuda")
        text = text.unsqueeze(1).expand(-1, temporal_length, -1)

        input_dict = {
            Cameras.LEFT.value: images,
            TOKENIZED_OBSERVATIONS_KEY: text
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
    def test_feature_extraction_methods(self, feature_method, image_inputs_factory, text_inputs_factory):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            pooling_method=feature_method,
        ).to("cuda")

        batch_size = 2
        images = image_inputs_factory(batch_size=batch_size, device="cuda")
        text = text_inputs_factory(batch_size=batch_size, device="cuda")

        input_dict = {
            Cameras.LEFT.value: images,
            TOKENIZED_OBSERVATIONS_KEY: text
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
            assert image_output.ndim == 3
            assert text_output.ndim == 3
            assert image_output.shape[0] == batch_size
            assert text_output.shape[0] == batch_size
            assert image_output.shape[2] == 768
            assert text_output.shape[2] == 512

    def test_forward_output_keys_match_specification(self, image_inputs_factory, text_inputs_factory):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        batch_size = 2
        images = image_inputs_factory(batch_size=batch_size, device="cuda")
        text = text_inputs_factory(batch_size=batch_size, device="cuda")

        input_dict = {
            Cameras.LEFT.value: images,
            TOKENIZED_OBSERVATIONS_KEY: text
        }
        output_dict = encoder(input_dict)
        output_spec = encoder.get_output_specification()

        assert set(output_dict.keys()) == set(output_spec.features)

    @pytest.mark.parametrize("feature_method", FEATURE_EXTRACTION_METHODS)
    def test_forward_dimensions_match_specification(self, feature_method, image_inputs_factory, text_inputs_factory):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            pooling_method=feature_method,
        ).to("cuda")

        batch_size = 2
        images = image_inputs_factory(batch_size=batch_size, device="cuda")
        text = text_inputs_factory(batch_size=batch_size, device="cuda")

        input_dict = {
            Cameras.LEFT.value: images,
            TOKENIZED_OBSERVATIONS_KEY: text
        }
        output_dict = encoder(input_dict)
        output_spec = encoder.get_output_specification()

        for key in output_spec.features:
            spec_dim = output_spec.dimensions[key]
            if feature_method != PoolingMethod.NONE.value:
                assert output_dict[key].shape[-1] == spec_dim
            else:
                assert output_dict[key].shape[-1] == spec_dim[-1]


    def test_gradients_enabled_unfrozen(self, image_inputs_factory, text_inputs_factory):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=False,
            pooling_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        batch_size = 2
        images = image_inputs_factory(batch_size=batch_size, device="cuda").requires_grad_(True)
        text = text_inputs_factory(batch_size=batch_size, device="cuda")

        input_dict = {
            Cameras.LEFT.value: images,
            TOKENIZED_OBSERVATIONS_KEY: text
        }
        output_dict = encoder(input_dict)

        assert output_dict[EncoderOutputKeys.RGB.value].requires_grad

        encoder_params_trainable = any(param.requires_grad for param in encoder.encoder.parameters())
        assert encoder_params_trainable

    def test_eval_mode_determinism(self, image_inputs_factory, text_inputs_factory):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        encoder.eval()

        batch_size = 2
        images = image_inputs_factory(batch_size=batch_size, device="cuda")
        text = text_inputs_factory(batch_size=batch_size, device="cuda")

        input_dict = {
            Cameras.LEFT.value: images,
            TOKENIZED_OBSERVATIONS_KEY: text
        }

        with torch.no_grad():
            output1 = encoder(input_dict)
            output2 = encoder(input_dict)

        assert torch.allclose(output1[EncoderOutputKeys.RGB.value], output2[EncoderOutputKeys.RGB.value], atol=1e-6)
        assert torch.allclose(output1[EncoderOutputKeys.LANGUAGE.value], output2[EncoderOutputKeys.LANGUAGE.value], atol=1e-6)

    def test_multiple_camera_types(self, text_inputs_factory):
        for camera in [Cameras.LEFT.value, Cameras.RIGHT.value]:
            encoder = VLMEncoder(
                input_keys=[camera, TOKENIZED_OBSERVATIONS_KEY],
                model_name=ImageTextModelType.CLIP_VITB32.value,
                pretrained=False,
                frozen=True,
                pooling_method=PoolingMethod.AVERAGE.value,
            ).to("cuda")

            batch_size = 2
            images = torch.randn(batch_size, 3, 224, 224, device="cuda")
            text = text_inputs_factory(batch_size=batch_size, device="cuda")

            input_dict = {
                camera: images,
                TOKENIZED_OBSERVATIONS_KEY: text
            }
            output_dict = encoder(input_dict)

            assert output_dict[EncoderOutputKeys.RGB.value].shape == (batch_size, 768)
            assert output_dict[EncoderOutputKeys.LANGUAGE.value].shape == (batch_size, 512)


@pytest.mark.integration
class TestImageTextEncoderOutputSpecification:
    """Test encoder output specification methods."""

    @pytest.mark.parametrize("model_name,vision_dim,text_dim", IMAGE_TEXT_MODELS_TO_OUTPUT_DIM)
    def test_output_specification_structure(self, model_name, vision_dim, text_dim):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=model_name,
            pretrained=False,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
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
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        output_spec = encoder.get_output_specification()
        assert output_spec.is_multi_output is True


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for CLIP")
class TestImageTextEncoderIntegration:
    """Integration tests for complete workflows."""

    def test_complete_forward_backward_pass(self, image_inputs_factory, text_inputs_factory):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=False,
            pooling_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        encoder.train()

        batch_size = 2
        images = image_inputs_factory(batch_size=batch_size, device="cuda").requires_grad_(True)
        text = text_inputs_factory(batch_size=batch_size, device="cuda")

        input_dict = {
            Cameras.LEFT.value: images,
            TOKENIZED_OBSERVATIONS_KEY: text
        }
        output_dict = encoder(input_dict)
        loss = output_dict[EncoderOutputKeys.RGB.value].mean() + output_dict[EncoderOutputKeys.LANGUAGE.value].mean()
        loss.backward()

        assert output_dict[EncoderOutputKeys.RGB.value].requires_grad
        assert loss.requires_grad

    def test_eval_mode_no_gradients(self, image_inputs_factory, text_inputs_factory):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        encoder.eval()

        batch_size = 2
        images = image_inputs_factory(batch_size=batch_size, device="cuda")
        text = text_inputs_factory(batch_size=batch_size, device="cuda")

        input_dict = {
            Cameras.LEFT.value: images,
            TOKENIZED_OBSERVATIONS_KEY: text
        }

        with torch.no_grad():
            output_dict = encoder(input_dict)

        assert not output_dict[EncoderOutputKeys.RGB.value].requires_grad
        assert not output_dict[EncoderOutputKeys.LANGUAGE.value].requires_grad

    def test_consistent_output_shapes(self, image_inputs_factory, text_inputs_factory):
        encoder = VLMEncoder(
            input_keys=[Cameras.LEFT.value, TOKENIZED_OBSERVATIONS_KEY],
            model_name=ImageTextModelType.CLIP_VITB32.value,
            pretrained=False,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        ).to("cuda")

        batch_size = 2
        images1 = image_inputs_factory(batch_size=batch_size, device="cuda")
        images2 = image_inputs_factory(batch_size=batch_size, device="cuda")
        text1 = text_inputs_factory(batch_size=batch_size, device="cuda")
        text2 = text_inputs_factory(batch_size=batch_size, device="cuda")

        input1 = {
            Cameras.LEFT.value: images1,
            TOKENIZED_OBSERVATIONS_KEY: text1
        }
        input2 = {
            Cameras.LEFT.value: images2,
            TOKENIZED_OBSERVATIONS_KEY: text2
        }

        output1 = encoder(input1)
        output2 = encoder(input2)

        assert output1[EncoderOutputKeys.RGB.value].shape == output2[EncoderOutputKeys.RGB.value].shape
        assert output1[EncoderOutputKeys.LANGUAGE.value].shape == output2[EncoderOutputKeys.LANGUAGE.value].shape

