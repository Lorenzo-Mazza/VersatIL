import pytest
import torch

from refactoring.models.encoding.encoders.rgb import CNNEncoder
from refactoring.models.encoding.encoders.constants import RGBBackboneType, PoolingMethod, EncoderOutputKeys
from refactoring.data.constants import Cameras


BACKBONE_FEATURE_DIMS = {
    RGBBackboneType.RESNET18.value: 512,
    RGBBackboneType.RESNET34.value: 512,
    RGBBackboneType.RESNET50.value: 2048,
    RGBBackboneType.EFFICIENTNET_B0.value: 320,
    RGBBackboneType.EDGENEXT_XX_SMALL.value: 168,
    RGBBackboneType.EDGENEXT_X_SMALL.value: 192,
    RGBBackboneType.EDGENEXT_SMALL.value: 304,
    RGBBackboneType.EDGENEXT_BASE.value: 584,
}


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
def image_size():
    """Image size fixture."""
    return (224, 224)


@pytest.mark.unit
class TestCNNEncoderInitialization:
    """Test CNNEncoder initialization."""


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET34.value,
        RGBBackboneType.RESNET50.value,
        RGBBackboneType.EFFICIENTNET_B0.value,
        RGBBackboneType.EDGENEXT_XX_SMALL.value,
        RGBBackboneType.EDGENEXT_X_SMALL.value,
        RGBBackboneType.EDGENEXT_SMALL.value,
        RGBBackboneType.EDGENEXT_BASE.value,
    ])
    @pytest.mark.parametrize("pooling_method,expected_multiplier", [
        (PoolingMethod.SPATIAL_SOFTMAX.value, 2),
        (PoolingMethod.AVERAGE.value, 1),
    ])
    def test_init_all_backbones_pooled(self, backbone, pooling_method, expected_multiplier):
        """Test initialization with pooling methods that return 1D features."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pooling_method=pooling_method,
            pretrained=False,
        )

        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]

        assert encoder.pooling_method == pooling_method
        assert encoder.use_group_norm is True
        assert encoder.feature_dim == expected_feature_dim
        assert encoder.output_dim == expected_feature_dim * expected_multiplier
        assert encoder.pooling_head is not None


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET50.value,
        RGBBackboneType.EFFICIENTNET_B0.value,
    ])
    def test_init_none_pooling(self, backbone):
        """Test initialization with NONE pooling returns spatial dimensions."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
        )

        assert encoder.pooling_method == PoolingMethod.NONE.value
        assert encoder.pooling_head is not None
        assert isinstance(encoder.output_dim, tuple)
        assert len(encoder.output_dim) == 3


    def test_init_without_group_norm(self):
        """Test initialization without group normalization."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.RESNET18.value,
            use_group_norm=False,
        )

        assert encoder.use_group_norm is False


    def test_init_frozen(self):
        """Test initialization with frozen weights."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.RESNET18.value,
            frozen=True,
        )

        for param in encoder.backbone.parameters():
            assert param.requires_grad is False


    def test_init_invalid_backbone(self):
        """Test initialization with invalid backbone raises error."""
        with pytest.raises(ValueError, match="Invalid backbone 'invalid_backbone'."):
            CNNEncoder(
                input_keys=Cameras.LEFT.value,
                backbone="invalid_backbone",
            )


    def test_init_invalid_pooling_method(self):
        """Test initialization with invalid pooling method raises error."""
        with pytest.raises(ValueError, match="Unsupported pooling method"):
            CNNEncoder(
                input_keys=Cameras.LEFT.value,
                backbone=RGBBackboneType.RESNET18.value,
                pooling_method="invalid_pooling",
            )


    def test_init_invalid_input_keys(self):
        """Test initialization with invalid input keys raises error."""
        with pytest.raises(ValueError, match="Exactly one from"):
            CNNEncoder(
                input_keys="invalid_camera",
                backbone=RGBBackboneType.RESNET18.value,
            )


    def test_init_multiple_cameras(self):
        """Test initialization with multiple camera keys raises error."""
        with pytest.raises(ValueError, match="Exactly one from"):
            CNNEncoder(
                input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
                backbone=RGBBackboneType.RESNET18.value,
            )


    def test_get_output_specification(self):
        """Test get_output_specification returns correct structure."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.RESNET18.value,
        )

        spec = encoder.get_output_specification()
        assert spec.features == [EncoderOutputKeys.RGB.value]
        assert isinstance(spec.features, list)
        assert EncoderOutputKeys.RGB.value in spec.dimensions
        assert isinstance(spec.dimensions, dict)


@pytest.mark.unit
class TestCNNEncoderForward:
    """Test CNNEncoder forward pass."""


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET34.value,
        RGBBackboneType.RESNET50.value,
        RGBBackboneType.EFFICIENTNET_B0.value,
        RGBBackboneType.EDGENEXT_XX_SMALL.value,
        RGBBackboneType.EDGENEXT_X_SMALL.value,
        RGBBackboneType.EDGENEXT_SMALL.value,
        RGBBackboneType.EDGENEXT_BASE.value,
    ])
    @pytest.mark.parametrize("pooling_method,expected_multiplier", [
        (PoolingMethod.SPATIAL_SOFTMAX.value, 2),
        (PoolingMethod.AVERAGE.value, 1),
    ])
    def test_forward_4d_input_pooled(self, input_dict_4d, backbone, pooling_method, expected_multiplier):
        """Test forward pass with 4D input and pooling."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pooling_method=pooling_method,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output_dict = encoder(input_dict)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.RGB.value in output_dict

        output = output_dict[EncoderOutputKeys.RGB.value]
        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]
        expected_output_dim = expected_feature_dim * expected_multiplier

        assert output.shape == (2, expected_output_dim)
        assert output.dtype == torch.float32


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET50.value,
        RGBBackboneType.EFFICIENTNET_B0.value,
    ])
    def test_forward_4d_input_none_pooling(self, input_dict_4d, backbone):
        """Test forward pass with 4D input and NONE pooling returns spatial features."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output_dict = encoder(input_dict)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.RGB.value in output_dict

        output = output_dict[EncoderOutputKeys.RGB.value]
        assert output.dim() == 4
        assert output.shape[0] == 2
        assert output.dtype == torch.float32


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET50.value,
        RGBBackboneType.EFFICIENTNET_B0.value,
        RGBBackboneType.EDGENEXT_SMALL.value,
    ])
    @pytest.mark.parametrize("pooling_method,expected_multiplier", [
        (PoolingMethod.SPATIAL_SOFTMAX.value, 2),
        (PoolingMethod.AVERAGE.value, 1),
    ])
    def test_forward_5d_input_pooled(self, input_dict_5d, backbone, pooling_method, expected_multiplier):
        """Test forward pass with 5D input (temporal) and pooling."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pooling_method=pooling_method,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_5d["rgb"]}
        output_dict = encoder(input_dict)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.RGB.value in output_dict

        output = output_dict[EncoderOutputKeys.RGB.value]
        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]
        expected_output_dim = expected_feature_dim * expected_multiplier

        assert output.shape == (2, 4, expected_output_dim)
        assert output.dtype == torch.float32


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET50.value,
        RGBBackboneType.EFFICIENTNET_B0.value,
    ])
    def test_forward_5d_input_none_pooling(self, input_dict_5d, backbone):
        """Test forward pass with 5D input and NONE pooling returns spatial features."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_5d["rgb"]}
        output_dict = encoder(input_dict)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.RGB.value in output_dict

        output = output_dict[EncoderOutputKeys.RGB.value]
        assert output.dim() == 5
        assert output.shape[0] == 2
        assert output.shape[1] == 4
        assert output.dtype == torch.float32


    def test_forward_dict_structure(self, input_dict_4d):
        """Test forward returns proper dict structure."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output_dict = encoder(input_dict)

        assert isinstance(output_dict, dict)
        assert len(output_dict) == 1
        assert all(isinstance(k, str) for k in output_dict.keys())
        assert EncoderOutputKeys.RGB.value in output_dict


@pytest.mark.unit
class TestCNNEncoderOutputSpecification:
    """Test CNNEncoder output specification."""


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET50.value,
        RGBBackboneType.EFFICIENTNET_B0.value,
        RGBBackboneType.EDGENEXT_SMALL.value,
    ])
    @pytest.mark.parametrize("pooling_method,expected_multiplier", [
        (PoolingMethod.SPATIAL_SOFTMAX.value, 2),
        (PoolingMethod.AVERAGE.value, 1),
    ])
    def test_output_dims_pooled(self, backbone, pooling_method, expected_multiplier):
        """Test output dimensions with pooling methods."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pooling_method=pooling_method,
            pretrained=False,
        )

        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]
        expected_output_dim = expected_feature_dim * expected_multiplier

        spec = encoder.get_output_specification()

        assert isinstance(spec.dimensions, dict)
        assert EncoderOutputKeys.RGB.value in spec.dimensions
        assert spec.dimensions[EncoderOutputKeys.RGB.value] == expected_output_dim


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET50.value,
        RGBBackboneType.EFFICIENTNET_B0.value,
    ])
    def test_output_dims_none_pooling(self, backbone):
        """Test output dimensions with NONE pooling returns tuple."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
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


    def test_output_specification_structure(self):
        """Test get_output_specification returns proper structure."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
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


class TestCNNEncoderGradients:
    """Test gradient behavior."""


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.EDGENEXT_SMALL.value,
    ])
    def test_gradients_enabled_unfrozen(self, input_dict_4d, backbone):
        """Test gradients flow when not frozen."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            frozen=False,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output = encoder(input_dict)[EncoderOutputKeys.RGB.value]
        loss = output.sum()
        loss.backward()

        for param in encoder.backbone.parameters():
            if param.requires_grad:
                assert param.grad is not None


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.EDGENEXT_SMALL.value,
    ])
    def test_gradients_disabled_frozen(self, input_dict_4d, backbone):
        """Test gradients don't flow when frozen."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output = encoder(input_dict)[EncoderOutputKeys.RGB.value]

        assert not output.requires_grad

        for param in encoder.backbone.parameters():
            assert not param.requires_grad


    def test_gradients_with_none_pooling(self, input_dict_4d):
        """Test gradients flow with NONE pooling."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.RESNET18.value,
            frozen=False,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output = encoder(input_dict)[EncoderOutputKeys.RGB.value]
        loss = output.sum()
        loss.backward()

        for param in encoder.backbone.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestCNNEncoderIntegration:
    """Integration tests for complete workflows."""


    def test_complete_forward_backward_pass(self, input_dict_4d):
        """Test complete forward and backward pass."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        encoder.train()
        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output = encoder(input_dict)[EncoderOutputKeys.RGB.value]
        loss = output.mean()
        loss.backward()

        assert output.requires_grad
        assert loss.requires_grad


    def test_eval_mode_no_gradients(self, input_dict_4d):
        """Test eval mode doesn't compute gradients."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.AVERAGE.value,
        )

        encoder.eval()
        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        with torch.no_grad():
            output = encoder(input_dict)[EncoderOutputKeys.RGB.value]

        assert not output.requires_grad


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET50.value,
    ])
    def test_consistent_output_shapes(self, batch_size, image_size, backbone):
        """Test output shapes are consistent across multiple forward passes."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=backbone,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        H, W = image_size
        input1 = {Cameras.LEFT.value: torch.randn(batch_size, 3, H, W)}
        input2 = {Cameras.LEFT.value: torch.randn(batch_size, 3, H, W)}

        output1 = encoder(input1)[EncoderOutputKeys.RGB.value]
        output2 = encoder(input2)[EncoderOutputKeys.RGB.value]

        assert output1.shape == output2.shape


    def test_multiple_camera_types(self):
        """Test encoders work with different camera types."""
        for camera in [Cameras.LEFT.value, Cameras.RIGHT.value]:
            encoder = CNNEncoder(
                input_keys=camera,
                backbone=RGBBackboneType.RESNET18.value,
            )

            input_dict = {camera: torch.randn(2, 3, 224, 224)}
            output = encoder(input_dict)[EncoderOutputKeys.RGB.value]

            assert output.shape == (2, 512)


    def test_none_pooling_for_decoder(self, input_dict_4d):
        """Test NONE pooling workflow for passing spatial features to decoder."""
        encoder = CNNEncoder(
            input_keys=Cameras.LEFT.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.NONE.value,
            pretrained=False,
        )

        input_dict = {Cameras.LEFT.value: input_dict_4d["rgb"]}
        output = encoder(input_dict)[EncoderOutputKeys.RGB.value]

        assert output.dim() == 4
        B, C, H, W = output.shape
        assert B == 2
        assert C > 0 and H > 0 and W > 0

        flattened = output.flatten(2).permute(0, 2, 1)
        assert flattened.shape == (B, H * W, C)