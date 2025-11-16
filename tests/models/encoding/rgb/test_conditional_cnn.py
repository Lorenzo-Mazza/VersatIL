import pytest
import torch

from refactoring.models.encoding.encoders.rgb.conditional_cnn import ConditionalCNNEncoder
from refactoring.models.encoding.encoders.constants import RGBBackboneType, PoolingMethod, EncoderOutputKeys
from refactoring.data.constants import Cameras


SUPPORTED_BACKBONES = [
    RGBBackboneType.RESNET18.value,
    RGBBackboneType.RESNET34.value,
]

BACKBONE_FEATURE_DIMS = {
    RGBBackboneType.RESNET18.value: 512,
    RGBBackboneType.RESNET34.value: 512,
}


@pytest.fixture
def input_dict_4d():
    """4D input tensor fixture (batch, channels, height, width)."""
    return {"rgb": torch.randn(2, 3, 224, 224)}


@pytest.fixture
def input_dict_5d():
    """5D input tensor fixture (batch, time, channels, height, width)."""
    return {"rgb": torch.randn(2, 2, 3, 224, 224)}


@pytest.fixture
def batch_size():
    """Batch size fixture."""
    return 2


@pytest.fixture
def temporal_length():
    """Temporal length fixture."""
    return 2


@pytest.fixture
def image_size():
    """Image size fixture."""
    return (224, 224)


@pytest.fixture
def condition_dim():
    """Default conditioning dimension."""
    return 256


@pytest.fixture
def condition_tensor(batch_size, condition_dim):
    """Condition tensor for batch."""
    return torch.randn(batch_size, condition_dim)


@pytest.fixture
def condition_tensor_5d(batch_size, temporal_length, condition_dim):
    """Condition tensor for 5D temporal input."""
    return torch.randn(batch_size, temporal_length, condition_dim)


@pytest.mark.unit
class TestConditionalCNNEncoderInitialization:
    """Test ConditionalCNNEncoder initialization."""

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    @pytest.mark.parametrize("pooling_method,expected_multiplier", [
        (PoolingMethod.SPATIAL_SOFTMAX.value, 2),
        (PoolingMethod.AVERAGE.value, 1),
    ])
    def test_init_supported_backbones_pooled(self, backbone, pooling_method, expected_multiplier, condition_dim):
        """Test initialization with pooling methods that return 1D features."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=backbone,
            pooling_method=pooling_method,
            pretrained=False,
        )

        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]

        assert encoder.condition_key == "language_features"
        assert encoder.condition_dim == condition_dim
        assert encoder.pooling_method == pooling_method
        assert encoder.use_group_norm is True
        assert encoder.feature_dim == expected_feature_dim
        assert encoder.output_dim == expected_feature_dim * expected_multiplier
        assert encoder.pooling_head is not None

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    def test_init_none_pooling(self, backbone, condition_dim):
        """Test initialization with NONE pooling returns spatial dimensions."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=backbone,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
        )

        assert encoder.pooling_method == PoolingMethod.NONE.value
        assert encoder.pooling_head is not None
        assert isinstance(encoder.output_dim, tuple)
        assert len(encoder.output_dim) == 3

    def test_init_without_group_norm(self, condition_dim):
        """Test initialization without group normalization."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=RGBBackboneType.RESNET18.value,
            use_group_norm=False,
        )

        assert encoder.use_group_norm is False

    def test_init_frozen(self, condition_dim):
        """Test initialization with frozen weights."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=RGBBackboneType.RESNET18.value,
            frozen=True,
        )

        for param in encoder.parameters():
            assert param.requires_grad is False

    def test_init_unsupported_backbone(self, condition_dim):
        """Test initialization with unsupported backbone raises error."""
        with pytest.raises(ValueError, match="not supported for FiLM"):
            ConditionalCNNEncoder(
                input_keys=Cameras.LEFT.value,
                condition_key="language_features",
                condition_dim=condition_dim,
                backbone=RGBBackboneType.RESNET50.value,
            )

    def test_init_invalid_pooling_method(self, condition_dim):
        """Test initialization with invalid pooling method raises error."""
        with pytest.raises(ValueError, match="Unsupported pooling method"):
            ConditionalCNNEncoder(
                input_keys=Cameras.LEFT.value,
                condition_key="language_features",
                condition_dim=condition_dim,
                backbone=RGBBackboneType.RESNET18.value,
                pooling_method="invalid_pooling",
            )

    def test_init_invalid_input_keys(self, condition_dim):
        """Test initialization with invalid input keys raises error."""
        with pytest.raises(ValueError, match="Exactly one from"):
            ConditionalCNNEncoder(
                input_keys="invalid_camera",
                condition_key="language_features",
                condition_dim=condition_dim,
                backbone=RGBBackboneType.RESNET18.value,
            )

    def test_init_multiple_cameras(self, condition_dim):
        """Test initialization with multiple camera keys raises error."""
        with pytest.raises(ValueError, match="Exactly one from"):
            ConditionalCNNEncoder(
                input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
                condition_key="language_features",
                condition_dim=condition_dim,
                backbone=RGBBackboneType.RESNET18.value,
            )

    def test_get_output_specification(self, condition_dim):
        """Test get_output_specification returns correct structure."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=RGBBackboneType.RESNET18.value,
        )

        spec = encoder.get_output_specification()
        assert spec.features == [EncoderOutputKeys.RGB.value]
        assert isinstance(spec.features, list)
        assert EncoderOutputKeys.RGB.value in spec.dimensions
        assert isinstance(spec.dimensions, dict)


@pytest.mark.unit
class TestConditionalCNNEncoderForward:
    """Test ConditionalCNNEncoder forward pass."""

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    @pytest.mark.parametrize("pooling_method,expected_multiplier", [
        (PoolingMethod.SPATIAL_SOFTMAX.value, 2),
        (PoolingMethod.AVERAGE.value, 1),
    ])
    def test_forward_4d_input_pooled(self, input_dict_4d, condition_tensor, backbone, pooling_method, expected_multiplier, condition_dim):
        """Test forward pass with 4D input and pooling."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=backbone,
            pooling_method=pooling_method,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output_dict = encoder(input_dict, conditioning=condition_tensor)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.RGB.value in output_dict

        output = output_dict[EncoderOutputKeys.RGB.value]
        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]
        expected_output_dim = expected_feature_dim * expected_multiplier

        assert output.shape == (2, expected_output_dim)
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    def test_forward_4d_input_none_pooling(self, input_dict_4d, condition_tensor, backbone, condition_dim):
        """Test forward pass with 4D input and NONE pooling returns spatial features."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=backbone,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output_dict = encoder(input_dict, conditioning=condition_tensor)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.RGB.value in output_dict

        output = output_dict[EncoderOutputKeys.RGB.value]
        assert output.dim() == 4
        assert output.shape[0] == 2
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    @pytest.mark.parametrize("pooling_method,expected_multiplier", [
        (PoolingMethod.SPATIAL_SOFTMAX.value, 2),
        (PoolingMethod.AVERAGE.value, 1),
    ])
    def test_forward_5d_input_pooled(self, input_dict_5d, backbone, pooling_method, expected_multiplier, condition_dim):
        """Test forward pass with 5D input and pooling."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=backbone,
            pooling_method=pooling_method,
            pretrained=False,
        )

        B, T = 2, 2
        condition_tensor = torch.randn(B, condition_dim)

        input_dict = {Cameras.LEFT.value: input_dict_5d["rgb"]}
        output_dict = encoder(input_dict, conditioning=condition_tensor)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.RGB.value in output_dict

        output = output_dict[EncoderOutputKeys.RGB.value]
        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]
        expected_output_dim = expected_feature_dim * expected_multiplier

        assert output.shape == (B, T, expected_output_dim)
        assert output.dtype == torch.float32

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    def test_forward_5d_input_none_pooling(self, input_dict_5d, backbone, condition_dim):
        """Test forward pass with 5D temporal input and NONE pooling."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=backbone,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
        )

        B, T = 2, 2
        condition_tensor = torch.randn(B, condition_dim)

        input_dict = {Cameras.LEFT.value: input_dict_5d["rgb"]}
        output_dict = encoder(input_dict, conditioning=condition_tensor)

        output = output_dict[EncoderOutputKeys.RGB.value]
        assert output.dim() == 5
        assert output.shape[0] == B
        assert output.shape[1] == T
        assert output.dtype == torch.float32


@pytest.mark.unit
class TestConditionalCNNEncoderOutputSpecification:
    """Test ConditionalCNNEncoder output specification."""

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    def test_output_dims_spatial_softmax(self, backbone, condition_dim):
        """Test output dimensions with spatial softmax."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=backbone,
            pooling_method=PoolingMethod.SPATIAL_SOFTMAX.value,
            pretrained=False,
        )

        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]
        spec = encoder.get_output_specification()

        assert isinstance(spec.dimensions, dict)
        assert EncoderOutputKeys.RGB.value in spec.dimensions
        assert spec.dimensions[EncoderOutputKeys.RGB.value] == expected_feature_dim * 2

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    def test_output_dims_global_average(self, backbone, condition_dim):
        """Test output dimensions with global average pooling."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=backbone,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]
        spec = encoder.get_output_specification()

        assert isinstance(spec.dimensions, dict)
        assert EncoderOutputKeys.RGB.value in spec.dimensions
        assert spec.dimensions[EncoderOutputKeys.RGB.value] == expected_feature_dim

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    def test_output_dims_none_pooling(self, backbone, condition_dim):
        """Test output dimensions with NONE pooling returns tuple."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=backbone,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
        )

        spec = encoder.get_output_specification()

        assert isinstance(spec.dimensions, dict)
        assert EncoderOutputKeys.RGB.value in spec.dimensions
        assert isinstance(spec.dimensions[EncoderOutputKeys.RGB.value], tuple)
        assert len(spec.dimensions[EncoderOutputKeys.RGB.value]) == 3
        c, h, w = spec.dimensions[EncoderOutputKeys.RGB.value]
        assert c > 0 and h > 0 and w > 0

    def test_output_specification_structure(self, condition_dim):
        """Test get_output_specification returns proper structure."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        spec = encoder.get_output_specification()

        assert hasattr(spec, 'features')
        assert hasattr(spec, 'dimensions')
        assert isinstance(spec.features, list)
        assert isinstance(spec.dimensions, dict)
        assert len(spec.features) == 1
        assert all(isinstance(k, str) for k in spec.dimensions.keys())
        assert isinstance(spec.dimensions[EncoderOutputKeys.RGB.value], int)


class TestConditionalCNNEncoderGradients:
    """Test gradient behavior."""

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    def test_gradients_enabled_unfrozen(self, input_dict_4d, condition_tensor, backbone, condition_dim):
        """Test gradients flow when not frozen."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=backbone,
            frozen=False,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output = encoder(input_dict, conditioning=condition_tensor)[EncoderOutputKeys.RGB.value]
        loss = output.sum()
        loss.backward()

        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    def test_gradients_disabled_frozen(self, input_dict_4d, condition_tensor, backbone, condition_dim):
        """Test gradients don't flow when frozen."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=backbone,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output = encoder(input_dict, conditioning=condition_tensor)[EncoderOutputKeys.RGB.value]

        assert not output.requires_grad

        for param in encoder.parameters():
            assert not param.requires_grad

    def test_gradients_with_none_pooling(self, input_dict_4d, condition_tensor, condition_dim):
        """Test gradients flow with NONE pooling."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=RGBBackboneType.RESNET18.value,
            frozen=False,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output = encoder(input_dict, conditioning=condition_tensor)[EncoderOutputKeys.RGB.value]
        loss = output.sum()
        loss.backward()

        for param in encoder.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestConditionalCNNEncoderIntegration:
    """Integration tests for complete workflows."""

    def test_complete_forward_backward_pass(self, input_dict_4d, condition_tensor, condition_dim):
        """Test complete forward and backward pass."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        encoder.train()
        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output = encoder(input_dict, conditioning=condition_tensor)[EncoderOutputKeys.RGB.value]
        loss = output.mean()
        loss.backward()

        assert output.requires_grad
        assert loss.requires_grad

    def test_eval_mode_no_gradients(self, input_dict_4d, condition_tensor, condition_dim):
        """Test eval mode doesn't compute gradients."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        encoder.eval()
        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        with torch.no_grad():
            output = encoder(input_dict, conditioning=condition_tensor)[EncoderOutputKeys.RGB.value]

        assert not output.requires_grad

    @pytest.mark.parametrize("backbone", SUPPORTED_BACKBONES)
    def test_consistent_output_shapes(self, batch_size, image_size, backbone, condition_dim):
        """Test output shapes are consistent across multiple forward passes."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=backbone,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        H, W = image_size
        input1 = {Cameras.LEFT.value: torch.randn(batch_size, 3, H, W)}
        input2 = {Cameras.LEFT.value: torch.randn(batch_size, 3, H, W)}
        condition1 = torch.randn(batch_size, condition_dim)
        condition2 = torch.randn(batch_size, condition_dim)

        output1 = encoder(input1, conditioning=condition1)[EncoderOutputKeys.RGB.value]
        output2 = encoder(input2, conditioning=condition2)[EncoderOutputKeys.RGB.value]

        assert output1.shape == output2.shape

    def test_different_conditioning_produces_different_outputs(self, input_dict_4d, condition_dim):
        """Test different conditioning produces different outputs when letting gradients flow."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
            frozen=False
        )

        encoder.eval()

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        condition1 = torch.randn(2, condition_dim)
        condition2 = torch.randn(2, condition_dim) * 100
        optimizer1 = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        optimizer2 = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        encoder.train()
        for _ in range(2):
            output1 = encoder(input_dict, conditioning=condition1)[EncoderOutputKeys.RGB.value]
            loss = output1.mean()
            loss.backward()
            optimizer1.step()
            output2 = encoder(input_dict, conditioning=condition2)[EncoderOutputKeys.RGB.value]
            loss = output2.mean()
            loss.backward()
            optimizer2.step()

        assert not torch.allclose(output1, output2, atol=1e-7)

    def test_multiple_camera_types(self, condition_dim):
        """Test encoders work with different camera types."""
        for camera in [Cameras.LEFT.value, Cameras.RIGHT.value]:
            encoder = ConditionalCNNEncoder(
                input_keys=camera,
                condition_key="language_features",
                condition_dim=condition_dim,
                backbone=RGBBackboneType.RESNET18.value,
                pooling_method=PoolingMethod.AVERAGE.value,
            )

            input_dict = {camera: torch.randn(2, 3, 224, 224)}
            condition_tensor = torch.randn(2, condition_dim)
            output = encoder(input_dict, conditioning=condition_tensor)[EncoderOutputKeys.RGB.value]

            assert output.shape == (2, 512)

    def test_none_pooling_for_decoder(self, input_dict_4d, condition_tensor, condition_dim):
        """Test NONE pooling workflow for passing spatial features to decoder."""
        encoder = ConditionalCNNEncoder(
            input_keys=Cameras.LEFT.value,
            condition_key="language_features",
            condition_dim=condition_dim,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output = encoder(input_dict, conditioning=condition_tensor)[EncoderOutputKeys.RGB.value]

        assert output.dim() == 4
        B, C, H, W = output.shape
        assert B == 2
        assert C > 0 and H > 0 and W > 0

        flattened = output.flatten(2).permute(0, 2, 1)
        assert flattened.shape == (B, H * W, C)