import pytest
import torch

from refactoring.models.encoding.encoders.depth.cnn import DepthCNNEncoder
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
    return {Cameras.DEPTH.value: torch.randn(batch_size, 1, H, W)}


@pytest.fixture
def input_dict_5d(batch_size, temporal_length, image_size):
    H, W = image_size
    return {Cameras.DEPTH.value: torch.randn(batch_size, temporal_length, 1, H, W)}


@pytest.mark.unit
class TestDepthCNNEncoderInitialization:
    """Test DepthCNNEncoder initialization."""


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
        (PoolingMethod.GLOBAL_AVERAGE.value, 1),
    ])
    def test_init_all_backbones(self, backbone, pooling_method, expected_multiplier):
        """Test initialization with all backbones and pooling methods."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
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


    def test_init_without_group_norm(self):
        """Test initialization without group normalization."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
            use_group_norm=False,
        )

        assert encoder.use_group_norm is False


    def test_init_frozen(self):
        """Test initialization with frozen weights."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
            frozen=True,
        )

        for param in encoder.backbone.parameters():
            assert param.requires_grad is False


    def test_init_invalid_pooling_method(self):
        """Test initialization with invalid pooling method raises error."""
        with pytest.raises(ValueError, match="Unsupported pooling method"):
            DepthCNNEncoder(
                input_keys=Cameras.DEPTH.value,
                backbone=RGBBackboneType.RESNET18.value,
                pooling_method="invalid_pooling",
            )


    def test_init_custom_image_size(self):
        """Test initialization with custom image dimensions."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
            image_height=128,
            image_width=128,
        )

        assert encoder.image_height == 128
        assert encoder.image_width == 128


    def test_get_output_specification(self):
        """Test get_output_specification returns correct structure."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
        )

        output_spec = encoder.get_output_specification()

        assert output_spec.features == [EncoderOutputKeys.DEPTH.value]
        assert isinstance(output_spec.features, list)
        assert isinstance(output_spec.dimensions, dict)
        assert EncoderOutputKeys.DEPTH.value in output_spec.dimensions
        assert output_spec.dimensions[EncoderOutputKeys.DEPTH.value] == encoder.output_dim


    def test_input_specification(self):
        """Test input specification is correctly set."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
        )

        assert encoder.input_specification.keys == [Cameras.DEPTH.value]
        assert Cameras.DEPTH.value in encoder.input_specification.required


@pytest.mark.unit
class TestDepthCNNEncoderForward:
    """Test DepthCNNEncoder forward pass."""


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
        (PoolingMethod.GLOBAL_AVERAGE.value, 1),
    ])
    def test_forward_4d_input(self, input_dict_4d, backbone, pooling_method, expected_multiplier):
        """Test forward pass with 4D input (B, C, H, W) for all backbones."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=backbone,
            pooling_method=pooling_method,
            pretrained=False,
        )

        output_dict = encoder(input_dict_4d)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.DEPTH.value in output_dict

        output = output_dict[EncoderOutputKeys.DEPTH.value]
        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]
        expected_output_dim = expected_feature_dim * expected_multiplier

        assert output.shape == (2, expected_output_dim)
        assert output.dtype == torch.float32


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET50.value,
        RGBBackboneType.EFFICIENTNET_B0.value,
        RGBBackboneType.EDGENEXT_SMALL.value,
    ])
    @pytest.mark.parametrize("pooling_method,expected_multiplier", [
        (PoolingMethod.SPATIAL_SOFTMAX.value, 2),
        (PoolingMethod.GLOBAL_AVERAGE.value, 1),
    ])
    def test_forward_5d_input(self, input_dict_5d, backbone, pooling_method, expected_multiplier):
        """Test forward pass with 5D input (B, T, C, H, W) for selected backbones."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=backbone,
            pooling_method=pooling_method,
            pretrained=False,
        )

        output_dict = encoder(input_dict_5d)

        assert isinstance(output_dict, dict)
        assert EncoderOutputKeys.DEPTH.value in output_dict

        output = output_dict[EncoderOutputKeys.DEPTH.value]
        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]
        expected_output_dim = expected_feature_dim * expected_multiplier

        assert output.shape == (2, 4, expected_output_dim)
        assert output.dtype == torch.float32


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET50.value,
        RGBBackboneType.EDGENEXT_SMALL.value,
    ])
    def test_forward_train_mode(self, input_dict_4d, backbone):
        """Test forward pass respects training mode."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=backbone,
            pooling_method=PoolingMethod.GLOBAL_AVERAGE.value,
            pretrained=False,
        )

        encoder.train()
        output_train = encoder(input_dict_4d)[EncoderOutputKeys.DEPTH.value]

        encoder.eval()
        with torch.no_grad():
            output_eval = encoder(input_dict_4d)[EncoderOutputKeys.DEPTH.value]

        assert output_train.requires_grad is True
        assert output_eval.requires_grad is False


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.EDGENEXT_SMALL.value,
    ])
    def test_forward_consistent_feature_dim(self, input_dict_4d, backbone):
        """Test feature dimension matches specification."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=backbone,
            pooling_method=PoolingMethod.GLOBAL_AVERAGE.value,
            pretrained=False,
        )

        output = encoder(input_dict_4d)[EncoderOutputKeys.DEPTH.value]
        output_spec = encoder.get_output_specification()

        assert output.shape[-1] == output_spec.dimensions[EncoderOutputKeys.DEPTH.value]


    def test_forward_output_keys_match_specification(self):
        """Test forward output keys match specification."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
        )

        depth_input = torch.randn(2, 1, 224, 224)
        output_dict = encoder({Cameras.DEPTH.value: depth_input})
        output_spec = encoder.get_output_specification()

        assert set(output_dict.keys()) == set(output_spec.features)


@pytest.mark.unit
class TestDepthCNNEncoderOutputSpecification:
    """Test encoder output specification methods."""


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET50.value,
        RGBBackboneType.EFFICIENTNET_B0.value,
        RGBBackboneType.EDGENEXT_SMALL.value,
    ])
    def test_output_specification_spatial_softmax(self, backbone):
        """Test output specification with spatial softmax."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=backbone,
            pooling_method=PoolingMethod.SPATIAL_SOFTMAX.value,
            pretrained=False,
        )

        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]
        output_spec = encoder.get_output_specification()

        assert isinstance(output_spec.dimensions, dict)
        assert EncoderOutputKeys.DEPTH.value in output_spec.dimensions
        assert output_spec.dimensions[EncoderOutputKeys.DEPTH.value] == expected_feature_dim * 2


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET50.value,
        RGBBackboneType.EFFICIENTNET_B0.value,
        RGBBackboneType.EDGENEXT_SMALL.value,
    ])
    def test_output_specification_global_average(self, backbone):
        """Test output specification with global average pooling."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=backbone,
            pooling_method=PoolingMethod.GLOBAL_AVERAGE.value,
            pretrained=False,
        )

        expected_feature_dim = BACKBONE_FEATURE_DIMS[backbone]
        output_spec = encoder.get_output_specification()

        assert isinstance(output_spec.dimensions, dict)
        assert EncoderOutputKeys.DEPTH.value in output_spec.dimensions
        assert output_spec.dimensions[EncoderOutputKeys.DEPTH.value] == expected_feature_dim


    def test_output_specification_structure(self):
        """Test output specification returns proper structure."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
        )

        output_spec = encoder.get_output_specification()

        assert hasattr(output_spec, 'features')
        assert hasattr(output_spec, 'dimensions')
        assert isinstance(output_spec.features, list)
        assert isinstance(output_spec.dimensions, dict)
        assert len(output_spec.features) == 1
        assert all(isinstance(k, str) for k in output_spec.dimensions.keys())
        assert all(isinstance(v, int) for v in output_spec.dimensions.values())


@pytest.mark.unit
class TestDepthCNNEncoderGradients:
    """Test gradient behavior."""


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.EDGENEXT_SMALL.value,
    ])
    def test_gradients_enabled_unfrozen(self, input_dict_4d, backbone):
        """Test gradients flow when not frozen."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=backbone,
            frozen=False,
            pooling_method=PoolingMethod.GLOBAL_AVERAGE.value,
            pretrained=False,
        )

        output = encoder(input_dict_4d)[EncoderOutputKeys.DEPTH.value]
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
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=backbone,
            frozen=True,
            pooling_method=PoolingMethod.GLOBAL_AVERAGE.value,
            pretrained=False,
        )

        output = encoder(input_dict_4d)[EncoderOutputKeys.DEPTH.value]

        assert not output.requires_grad

        for param in encoder.backbone.parameters():
            assert not param.requires_grad


@pytest.mark.unit
class TestDepthCNNEncoderIntegration:
    """Integration tests for complete workflows."""


    def test_complete_forward_backward_pass(self, input_dict_4d):
        """Test complete forward and backward pass."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.GLOBAL_AVERAGE.value,
        )

        encoder.train()
        output = encoder(input_dict_4d)[EncoderOutputKeys.DEPTH.value]
        loss = output.mean()
        loss.backward()

        assert output.requires_grad
        assert loss.requires_grad


    def test_eval_mode_no_gradients(self, input_dict_4d):
        """Test eval mode doesn't compute gradients."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.GLOBAL_AVERAGE.value,
        )

        encoder.eval()
        with torch.no_grad():
            output = encoder(input_dict_4d)[EncoderOutputKeys.DEPTH.value]

        assert not output.requires_grad


    @pytest.mark.parametrize("backbone", [
        RGBBackboneType.RESNET18.value,
        RGBBackboneType.RESNET50.value,
    ])
    def test_consistent_output_shapes(self, batch_size, image_size, backbone):
        """Test output shapes are consistent across multiple forward passes."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=backbone,
            pooling_method=PoolingMethod.GLOBAL_AVERAGE.value,
            pretrained=False,
        )

        H, W = image_size
        input1 = {Cameras.DEPTH.value: torch.randn(batch_size, 1, H, W)}
        input2 = {Cameras.DEPTH.value: torch.randn(batch_size, 1, H, W)}

        output1 = encoder(input1)[EncoderOutputKeys.DEPTH.value]
        output2 = encoder(input2)[EncoderOutputKeys.DEPTH.value]

        assert output1.shape == output2.shape


    def test_deterministic_output_eval_mode(self):
        """Test that eval mode produces deterministic output for same input."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.GLOBAL_AVERAGE.value,
            use_group_norm=False,
        )

        encoder.eval()
        depth_input = torch.randn(2, 1, 224, 224)
        input_dict = {Cameras.DEPTH.value: depth_input}

        with torch.no_grad():
            output1 = encoder(input_dict)[EncoderOutputKeys.DEPTH.value]
            output2 = encoder(input_dict)[EncoderOutputKeys.DEPTH.value]

        torch.testing.assert_close(output1, output2)


    def test_different_depth_ranges(self):
        """Test encoder handles different depth value ranges."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=PoolingMethod.GLOBAL_AVERAGE.value,
        )

        depth_normalized = torch.rand(2, 1, 224, 224)
        depth_large_values = torch.rand(2, 1, 224, 224) * 10.0

        output1 = encoder({Cameras.DEPTH.value: depth_normalized})[EncoderOutputKeys.DEPTH.value]
        output2 = encoder({Cameras.DEPTH.value: depth_large_values})[EncoderOutputKeys.DEPTH.value]

        assert output1.shape == output2.shape
        assert not torch.isnan(output1).any()
        assert not torch.isnan(output2).any()


    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.SPATIAL_SOFTMAX.value,
        PoolingMethod.GLOBAL_AVERAGE.value,
    ])
    def test_temporal_consistency(self, pooling_method):
        """Test temporal processing maintains consistency."""
        encoder = DepthCNNEncoder(
            input_keys=Cameras.DEPTH.value,
            backbone=RGBBackboneType.RESNET18.value,
            pooling_method=pooling_method,
        )

        depth_4d = torch.randn(2, 1, 224, 224)
        depth_5d = depth_4d.unsqueeze(1)

        output_4d = encoder({Cameras.DEPTH.value: depth_4d})[EncoderOutputKeys.DEPTH.value]
        output_5d = encoder({Cameras.DEPTH.value: depth_5d})[EncoderOutputKeys.DEPTH.value]

        output_5d_squeezed = output_5d.squeeze(1)
        torch.testing.assert_close(output_4d, output_5d_squeezed, rtol=1e-5, atol=1e-5)