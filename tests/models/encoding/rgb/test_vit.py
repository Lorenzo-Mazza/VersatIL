import pytest
import torch

from refactoring.models.encoding.encoders.rgb import ViTEncoder
from refactoring.models.encoding.encoders.constants import RGBBackboneType, PoolingMethod, EncoderOutputKeys
from refactoring.data.constants import Cameras


VIT_BACKBONES = [
    (RGBBackboneType.VIT_BASE.value, 768),
    (RGBBackboneType.DINOV2_VITS14.value, 384),
    (RGBBackboneType.DINOV2_VITB14.value, 768),
    (RGBBackboneType.DINOV2_VITL14.value, 1024),
    (RGBBackboneType.DINOV3_VITS16.value, 384),
    (RGBBackboneType.DINOV3_VITS16PLUS.value, 384),
    (RGBBackboneType.DINOV3_VITB16.value, 768),
]

FEATURE_EXTRACTION_METHODS = [
    PoolingMethod.DEFAULT.value,
    PoolingMethod.AVERAGE.value,
    PoolingMethod.LEARNED_AGGREGATION.value,
    PoolingMethod.NONE.value,
    PoolingMethod.MAX.value,
]


@pytest.fixture
def input_dict_4d():
    """4D input tensor fixture (batch, channels, height, width)."""
    return {"rgb": torch.randn(2, 3, 224, 224)}


@pytest.fixture
def input_dict_5d():
    """5D input tensor fixture (batch, time, channels, height, width)."""
    return {"rgb": torch.randn(2, 4, 3, 224, 224)}


@pytest.fixture
def batch_size():
    """Batch size fixture."""
    return 2


@pytest.fixture
def temporal_length():
    """Temporal length fixture."""
    return 4


@pytest.fixture
def device():
    """Device fixture."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.integration
class TestViTEncoder:
    """Test ViT encoder with TIMM models."""

    @pytest.mark.parametrize("backbone,expected_dim", VIT_BACKBONES)
    def test_initialization(self, backbone, expected_dim):
        """Test ViT encoder initialization with different backbones."""
        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        assert encoder.feature_dim == expected_dim
        spec = encoder.get_output_specification()
        assert spec.dimensions[EncoderOutputKeys.RGB.value] == expected_dim
        assert encoder.pooling_method == PoolingMethod.AVERAGE.value

    def test_get_output_specification(self):
        """Test get_output_specification returns correct structure."""
        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.DINOV2_VITB14.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        spec = encoder.get_output_specification()
        assert spec.features == [EncoderOutputKeys.RGB.value]
        assert isinstance(spec.features, list)
        assert EncoderOutputKeys.RGB.value in spec.dimensions
        assert isinstance(spec.dimensions, dict)

    def test_init_invalid_input_keys(self):
        """Test initialization with invalid input keys raises error."""
        with pytest.raises(ValueError, match="Exactly one from"):
            ViTEncoder(
                input_keys="invalid_camera",
                backbone=RGBBackboneType.DINOV2_VITB14.value,
                pretrained=True,
                frozen=True,
                pooling_method=PoolingMethod.AVERAGE.value,
            )

    def test_init_multiple_cameras(self):
        """Test initialization with multiple camera keys raises error."""
        with pytest.raises(ValueError, match="Exactly one from"):
            ViTEncoder(
                input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
                backbone=RGBBackboneType.DINOV2_VITB14.value,
                pretrained=True,
                frozen=True,
                pooling_method=PoolingMethod.AVERAGE.value,
            )

    @pytest.mark.parametrize("backbone,expected_dim", VIT_BACKBONES)
    def test_forward_4d(self, backbone, expected_dim, input_dict_4d, batch_size):
        """Test ViT encoder forward pass with 4D input."""
        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output_dict = encoder(input_dict)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.RGB.value in output_dict

        output = output_dict[EncoderOutputKeys.RGB.value]
        assert output.shape == (batch_size, expected_dim)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("backbone,expected_dim", VIT_BACKBONES)
    def test_forward_5d(self, backbone, expected_dim, input_dict_5d, batch_size, temporal_length):
        """Test ViT encoder forward pass with 5D temporal input."""
        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        input_dict = {Cameras.LEFT.value: input_dict_5d["rgb"]}
        output_dict = encoder(input_dict)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.RGB.value in output_dict

        output = output_dict[EncoderOutputKeys.RGB.value]
        assert output.shape == (batch_size, temporal_length, expected_dim)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("feature_method", FEATURE_EXTRACTION_METHODS)
    def test_feature_extraction_methods(self, feature_method, input_dict_4d, batch_size):
        """Test different feature extraction methods."""
        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.DINOV2_VITB14.value,
            pretrained=True,
            frozen=True,
            pooling_method=feature_method,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output = encoder(input_dict)[EncoderOutputKeys.RGB.value]

        if feature_method == PoolingMethod.LEARNED_AGGREGATION.value:
            expected_dim = 767
            assert encoder.pooling_head is not None
        else:
            expected_dim = 768
            assert encoder.pooling_head is None

        assert output.shape == (batch_size, expected_dim)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()

        if feature_method == PoolingMethod.LEARNED_AGGREGATION.value:
            assert encoder.pooling_head is not None
        else:
            assert encoder.pooling_head is None

    def test_unfrozen_backward_pass(self, input_dict_4d):
        """Test that unfrozen encoder allows gradient backpropagation."""
        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.DINOV2_VITB14.value,
            pretrained=True,
            frozen=False,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        input_tensor = input_dict_4d["rgb"].clone()
        input_tensor.requires_grad = True
        input_dict = {Cameras.LEFT.value: input_tensor}

        output = encoder(input_dict)[EncoderOutputKeys.RGB.value]
        loss = output.mean()
        loss.backward()

        assert input_tensor.grad is not None
        backbone_has_grads = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in encoder.backbone.parameters() if param.requires_grad
        )
        assert backbone_has_grads

    def test_frozen_weights(self, input_dict_4d):
        """Test frozen encoder has frozen parameters."""
        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.DINOV2_VITB14.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output = encoder(input_dict)[EncoderOutputKeys.RGB.value]

        assert not output.requires_grad

        for param in encoder.backbone.parameters():
            assert not param.requires_grad

    def test_unfrozen_weights(self, input_dict_4d):
        """Test unfrozen encoder has trainable parameters."""
        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.DINOV2_VITB14.value,
            pretrained=True,
            frozen=False,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        input_tensor = input_dict_4d["rgb"].clone()
        input_tensor.requires_grad = True
        input_dict = {Cameras.LEFT.value: input_tensor}

        output = encoder(input_dict)[EncoderOutputKeys.RGB.value]

        assert output.requires_grad

        backbone_params_trainable = any(param.requires_grad for param in encoder.backbone.parameters())
        assert backbone_params_trainable

    def test_different_feature_methods_produce_different_outputs(self, input_dict_4d):
        """Test different feature extraction methods produce different features."""
        encoder_cls = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.DINOV2_VITB14.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.DEFAULT.value,
        )

        encoder_gap = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.DINOV2_VITB14.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        encoder_cls.eval()
        encoder_gap.eval()

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}

        with torch.no_grad():
            output_cls = encoder_cls(input_dict)[EncoderOutputKeys.RGB.value]
            output_gap = encoder_gap(input_dict)[EncoderOutputKeys.RGB.value]

        assert output_cls.shape == output_gap.shape
        assert not torch.allclose(output_cls, output_gap, atol=1e-5)

    def test_eval_mode_determinism(self, input_dict_4d):
        """Test encoder produces deterministic outputs in eval mode."""
        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.DINOV2_VITB14.value,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        encoder.eval()

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}

        with torch.no_grad():
            output1 = encoder(input_dict)[EncoderOutputKeys.RGB.value]
            output2 = encoder(input_dict)[EncoderOutputKeys.RGB.value]

        assert torch.allclose(output1, output2, atol=1e-6)

    @pytest.mark.parametrize("backbone,expected_dim", VIT_BACKBONES)
    def test_different_image_sizes(self, backbone, expected_dim, batch_size):
        """Test encoder handles different image sizes with dynamic_img_size."""
        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        encoder.eval()

        input_224 = {Cameras.LEFT.value: torch.randn(batch_size, 3, 224, 224)}
        input_448 = {Cameras.LEFT.value: torch.randn(batch_size, 3, 448, 448)}

        with torch.no_grad():
            output_224 = encoder(input_224)[EncoderOutputKeys.RGB.value]
            output_448 = encoder(input_448)[EncoderOutputKeys.RGB.value]

        assert output_224.shape == (batch_size, expected_dim)
        assert output_448.shape == (batch_size, expected_dim)

    def test_multiple_camera_types(self):
        """Test encoders work with different camera types."""
        for camera in [Cameras.LEFT.value, Cameras.RIGHT.value]:
            encoder = ViTEncoder(
                input_keys=camera,
                backbone=RGBBackboneType.DINOV2_VITB14.value,
                pretrained=True,
                frozen=True,
                pooling_method=PoolingMethod.AVERAGE.value,
            )

            input_dict = {camera: torch.randn(2, 3, 224, 224)}
            output = encoder(input_dict)[EncoderOutputKeys.RGB.value]

            assert output.shape == (2, 768)


@pytest.mark.integration
class TestViTEncoderOutputSpecification:
    """Test ViT encoder output specification."""

    @pytest.mark.parametrize("backbone,expected_dim", VIT_BACKBONES)
    def test_output_dimensions(self, backbone, expected_dim):
        """Test output dimensions for different backbones."""
        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        spec = encoder.get_output_specification()

        assert isinstance(spec.dimensions, dict)
        assert EncoderOutputKeys.RGB.value in spec.dimensions
        assert spec.dimensions[EncoderOutputKeys.RGB.value] == expected_dim

    @pytest.mark.parametrize("feature_method", FEATURE_EXTRACTION_METHODS)
    def test_output_specification_structure(self, feature_method):
        """Test output specification structure for different feature methods."""
        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.DINOV2_VITB14.value,
            pretrained=True,
            frozen=True,
            pooling_method=feature_method,
        )

        spec = encoder.get_output_specification()

        assert hasattr(spec, 'features')
        assert hasattr(spec, 'dimensions')
        assert isinstance(spec.features, list)
        assert isinstance(spec.dimensions, dict)
        assert len(spec.features) == 1
        assert spec.features[0] == EncoderOutputKeys.RGB.value
        assert isinstance(spec.dimensions[EncoderOutputKeys.RGB.value], int)


@pytest.mark.integration
class TestViTEncoderComparison:
    """Compare different ViT encoder configurations."""

    def test_different_vit_sizes(self, input_dict_4d, batch_size):
        """Test different ViT sizes produce expected dimensions."""
        encoders_and_dims = [
            (RGBBackboneType.DINOV2_VITS14.value, 384),
            (RGBBackboneType.DINOV2_VITB14.value, 768),
            (RGBBackboneType.DINOV2_VITL14.value, 1024),
            (RGBBackboneType.DINOV3_VITS16.value, 384),
            (RGBBackboneType.DINOV3_VITB16.value, 768),
        ]

        outputs = []
        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}

        for backbone, expected_dim in encoders_and_dims:
            encoder = ViTEncoder(
                input_keys=Cameras.LEFT.value,
                backbone=backbone,
                pretrained=True,
                frozen=True,
                pooling_method=PoolingMethod.AVERAGE.value,
            )
            encoder.eval()

            with torch.no_grad():
                output = encoder(input_dict)[EncoderOutputKeys.RGB.value]

            assert output.shape == (batch_size, expected_dim)
            outputs.append((backbone, expected_dim, output))


@pytest.mark.integration
@pytest.mark.requires_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestViTEncodersGPU:
    """Test ViT encoders on GPU."""

    @pytest.mark.parametrize("backbone,expected_dim", VIT_BACKBONES)
    def test_encoder_on_gpu(self, backbone, expected_dim, input_dict_4d, batch_size, device):
        """Test ViT encoder execution on GPU."""
        if device.type != "cuda":
            pytest.skip("GPU not available")

        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pretrained=True,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        encoder = encoder.to(device)

        input_gpu = {Cameras.LEFT.value: input_dict_4d["rgb"].to(device)}
        output = encoder(input_gpu)[EncoderOutputKeys.RGB.value]

        assert output.device.type == "cuda"
        assert output.shape == (batch_size, expected_dim)

    def test_mixed_precision(self, input_dict_4d, batch_size, device):
        """Test encoder with mixed precision training."""
        if device.type != "cuda":
            pytest.skip("GPU not available")

        encoder = ViTEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.DINOV2_VITB14.value,
            pretrained=True,
            frozen=False,
            pooling_method=PoolingMethod.AVERAGE.value,
        )
        encoder = encoder.to(device)

        input_gpu = {Cameras.LEFT.value: input_dict_4d["rgb"].to(device)}

        with torch.cuda.amp.autocast():
            output = encoder(input_gpu)[EncoderOutputKeys.RGB.value]

        assert output.device.type == "cuda"
        assert output.shape == (batch_size, 768)
        assert output.dtype in [torch.float16, torch.float32]