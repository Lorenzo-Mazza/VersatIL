import pytest
import torch

from refactoring.models.encoding.encoders.depth import DFormerEncoder
from refactoring.models.encoding.encoders.constants import PoolingMethod, EncoderOutputKeys
from refactoring.data.constants import Cameras
from refactoring.models.encoding.encoders.depth.dformerv2 import DFormerVariant
from refactoring.models.layers.constants import AttentionDecompositionMode
from tests.conftest import CACHE_DIR


VARIANT_OUTPUT_DIMS = {
    DFormerVariant.SMALL.value: {
        PoolingMethod.SPATIAL_SOFTMAX.value: 512 * 2,
        PoolingMethod.AVERAGE.value: 512,
    },
    DFormerVariant.BASE.value: {
        PoolingMethod.SPATIAL_SOFTMAX.value: 512 * 2,
        PoolingMethod.AVERAGE.value: 512,
    },
    DFormerVariant.LARGE.value: {
        PoolingMethod.SPATIAL_SOFTMAX.value: 640 * 2,
        PoolingMethod.AVERAGE.value: 640,
    },
}


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 2


@pytest.fixture
def temporal_length():
    """Standard temporal length for sequential tests."""
    return 4


@pytest.fixture
def image_size():
    """Standard image size (H, W)."""
    return (224, 224)


@pytest.fixture
def input_dict_4d(batch_size, image_size):
    """4D input dict with RGB and depth (B, C, H, W)."""
    H, W = image_size
    return {
        "rgb": torch.randn(batch_size, 3, H, W),
        "depth": torch.randn(batch_size, 1, H, W),
    }


@pytest.fixture
def input_dict_5d(batch_size, temporal_length, image_size):
    """5D input dict with RGB and depth (B, T, C, H, W)."""
    H, W = image_size
    return {
        "rgb": torch.randn(batch_size, temporal_length, 3, H, W),
        "depth": torch.randn(batch_size, temporal_length, 1, H, W),
    }


@pytest.mark.unit
class TestDFormerEncoderInitialization:
    """Test DFormerEncoder initialization."""


    @pytest.mark.parametrize("variant", [
        DFormerVariant.SMALL.value,
        DFormerVariant.BASE.value,
        DFormerVariant.LARGE.value,
    ])
    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.SPATIAL_SOFTMAX.value,
        PoolingMethod.AVERAGE.value,
    ])
    def test_init_all_variants(self, variant, pooling_method):
        """Test initialization with all variants and pooling methods."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=variant,
            pooling_method=pooling_method,
            pretrained=False,
        )

        assert encoder.variant == variant
        assert encoder.pooling_method == pooling_method

        output_spec = encoder.get_output_specification()
        expected_output_dim = VARIANT_OUTPUT_DIMS[variant][pooling_method]
        assert output_spec.dimensions[EncoderOutputKeys.RGBD.value] == expected_output_dim

        assert encoder.pooling_head is not None


    @pytest.mark.parametrize("decomposition_mode", [
        AttentionDecompositionMode.FULL.value,
        AttentionDecompositionMode.SEPARABLE.value,
    ])
    def test_init_decomposition_modes(self, decomposition_mode):
        """Test initialization with different attention decomposition modes."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            decomposition_mode=decomposition_mode,
            pretrained=False,
        )

        assert encoder.decomposition_mode == AttentionDecompositionMode(decomposition_mode)


    def test_init_frozen(self):
        """Test initialization with frozen weights."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            frozen=True,
        )

        for param in encoder.parameters():
            assert param.requires_grad is False


    def test_init_invalid_variant(self):
        """Test initialization with invalid variant raises error."""
        with pytest.raises(ValueError, match="Variant.*not supported"):
            DFormerEncoder(
                input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
                variant="invalid_variant",
            )


    def test_init_missing_depth_input(self):
        """Test initialization without depth input raises error."""
        with pytest.raises(ValueError, match="Missing required inputs:.*"):
            DFormerEncoder(
                input_keys=Cameras.LEFT.value,
                variant=DFormerVariant.SMALL.value,
            )


    def test_init_missing_rgb_input(self):
        """Test initialization without RGB input raises error."""
        with pytest.raises(ValueError, match="Exactly one from"):
            DFormerEncoder(
                input_keys=Cameras.DEPTH.value,
                variant=DFormerVariant.SMALL.value,
            )


    def test_init_multiple_rgb_inputs(self):
        """Test initialization with multiple RGB inputs raises error."""
        with pytest.raises(ValueError, match="Exactly one from"):
            DFormerEncoder(
                input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value],
                variant=DFormerVariant.SMALL.value,
            )


    def test_init_with_right_camera(self):
        """Test initialization with right camera instead of left."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.RIGHT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pretrained=False,
        )

        assert Cameras.RIGHT.value in encoder.input_specification.keys
        assert Cameras.DEPTH.value in encoder.input_specification.keys


    def test_get_output_specification(self):
        """Test get_output_specification returns correct structure."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
        )

        output_spec = encoder.get_output_specification()
        assert output_spec.features == [EncoderOutputKeys.RGBD.value]
        assert isinstance(output_spec.features, list)
        assert EncoderOutputKeys.RGBD.value in output_spec.dimensions
        assert isinstance(output_spec.dimensions[EncoderOutputKeys.RGBD.value], int)


    @pytest.mark.parametrize("drop_path_rate", [0.0, 0.1, 0.3])
    def test_init_drop_path_rate(self, drop_path_rate):
        """Test initialization with different drop path rates."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            drop_path_rate=drop_path_rate,
            pretrained=False,
        )

        assert encoder is not None


    def test_init_pretrained(self):
        """Test initialization with pretrained weights."""
        checkpoint_path = CACHE_DIR / "pretrained_dformer" / "DFormerv2_Small_NYU.pth"
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pretrained=True,
            checkpoint_path=checkpoint_path,
        )

        H, W = 224, 224
        input_dict = {
            Cameras.LEFT.value: torch.randn(2, 3, H, W),
            Cameras.DEPTH.value: torch.randn(2, 1, H, W),
        }
        output = encoder(input_dict)
        assert EncoderOutputKeys.RGBD.value in output


@pytest.mark.unit
class TestDFormerEncoderForward:
    """Test DFormerEncoder forward pass."""


    @pytest.mark.parametrize("variant", [
        DFormerVariant.SMALL.value,
        DFormerVariant.BASE.value,
        DFormerVariant.LARGE.value,
    ])
    @pytest.mark.parametrize("pretrained", [True, False])
    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.SPATIAL_SOFTMAX.value,
        PoolingMethod.AVERAGE.value,
    ])
    def test_forward_4d_input_all_variants(self, input_dict_4d, variant, pooling_method, pretrained):
        """Test forward pass with 4D inputs for all variants."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=variant,
            pooling_method=pooling_method,
            pretrained=pretrained,
            checkpoint_path=CACHE_DIR / "pretrained_dformer" / "DFormerv2_Small_NYU.pth" if pretrained else None,
        )

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"],
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }
        output_dict = encoder(input_dict)

        assert EncoderOutputKeys.RGBD.value in output_dict
        output = output_dict[EncoderOutputKeys.RGBD.value]

        expected_output_dim = VARIANT_OUTPUT_DIMS[variant][pooling_method]
        assert output.shape == (2, expected_output_dim)
        assert output.dtype == torch.float32


    @pytest.mark.parametrize("variant", [
        DFormerVariant.SMALL.value,
        DFormerVariant.BASE.value,
    ])
    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.SPATIAL_SOFTMAX.value,
        PoolingMethod.AVERAGE.value,
    ])
    def test_forward_5d_input(self, input_dict_5d, variant, pooling_method):
        """Test forward pass with 5D temporal inputs."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=variant,
            pooling_method=pooling_method,
            pretrained=False,
        )

        input_dict = {
            Cameras.LEFT.value: input_dict_5d["rgb"],
            Cameras.DEPTH.value: input_dict_5d["depth"],
        }
        output_dict = encoder(input_dict)

        assert EncoderOutputKeys.RGBD.value in output_dict
        output = output_dict[EncoderOutputKeys.RGBD.value]

        expected_output_dim = VARIANT_OUTPUT_DIMS[variant][pooling_method]
        assert output.shape == (2, 4, expected_output_dim)


    def test_forward_train_mode(self, input_dict_4d):
        """Test forward pass in training mode."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pretrained=False,
        )

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"],
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }

        encoder.train()
        output_train = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        encoder.eval()
        with torch.no_grad():
            output_eval = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        assert output_train.shape == output_eval.shape
        assert output_train.requires_grad
        assert not output_eval.requires_grad


    def test_forward_with_right_camera(self, input_dict_4d):
        """Test forward pass with right camera."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.RIGHT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pretrained=False,
        )

        input_dict = {
            Cameras.RIGHT.value: input_dict_4d["rgb"],
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }
        output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        assert output.shape[0] == 2


    @pytest.mark.parametrize("decomposition_mode", [
        AttentionDecompositionMode.FULL.value,
        AttentionDecompositionMode.SEPARABLE.value,
    ])
    def test_forward_decomposition_modes(self, input_dict_4d, decomposition_mode):
        """Test forward pass with different decomposition modes."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            decomposition_mode=decomposition_mode,
            pretrained=False,
        )

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"],
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }
        output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        assert output.shape[0] == 2


@pytest.mark.unit
class TestDFormerEncoderTemporalHandling:
    """Test temporal input handling."""


    @pytest.mark.parametrize("temporal_length", [1, 4, 8, 16])
    def test_temporal_various_lengths(self, batch_size, image_size, temporal_length):
        """Test various temporal lengths."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        H, W = image_size
        input_dict = {
            Cameras.LEFT.value: torch.randn(batch_size, temporal_length, 3, H, W),
            Cameras.DEPTH.value: torch.randn(batch_size, temporal_length, 1, H, W),
        }
        output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        expected_output_dim = VARIANT_OUTPUT_DIMS[DFormerVariant.SMALL.value][
            PoolingMethod.AVERAGE.value
        ]
        assert output.shape == (batch_size, temporal_length, expected_output_dim)


    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.SPATIAL_SOFTMAX.value,
        PoolingMethod.AVERAGE.value,
    ])
    def test_temporal_single_timestep(self, batch_size, image_size, pooling_method):
        """Test single timestep temporal input."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pooling_method=pooling_method,
            pretrained=False,
        )

        H, W = image_size
        input_dict = {
            Cameras.LEFT.value: torch.randn(batch_size, 1, 3, H, W),
            Cameras.DEPTH.value: torch.randn(batch_size, 1, 1, H, W),
        }
        output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        expected_output_dim = VARIANT_OUTPUT_DIMS[DFormerVariant.SMALL.value][pooling_method]
        assert output.shape == (batch_size, 1, expected_output_dim)


    def test_temporal_consistency(self, batch_size, image_size):
        """Test temporal dimension is preserved correctly."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pretrained=False,
        )

        encoder.eval()
        H, W = image_size

        input_5d = {
            Cameras.LEFT.value: torch.randn(batch_size, 4, 3, H, W),
            Cameras.DEPTH.value: torch.randn(batch_size, 4, 1, H, W),
        }

        with torch.no_grad():
            output_5d = encoder(input_5d)[EncoderOutputKeys.RGBD.value]

        assert output_5d.shape[0] == batch_size
        assert output_5d.shape[1] == 4


@pytest.mark.unit
class TestDFormerEncoderOutputSpecification:
    """Test output specification methods."""


    def test_output_specification_structure(self):
        """Test output specification has correct structure."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
        )

        output_spec = encoder.get_output_specification()

        assert hasattr(output_spec, 'features')
        assert hasattr(output_spec, 'dimensions')
        assert isinstance(output_spec.features, list)
        assert isinstance(output_spec.dimensions, dict)


    @pytest.mark.parametrize("variant,pooling_method", [
        (DFormerVariant.SMALL.value, PoolingMethod.SPATIAL_SOFTMAX.value),
        (DFormerVariant.SMALL.value, PoolingMethod.AVERAGE.value),
        (DFormerVariant.BASE.value, PoolingMethod.SPATIAL_SOFTMAX.value),
        (DFormerVariant.LARGE.value, PoolingMethod.AVERAGE.value),
    ])
    def test_output_dimensions_correct(self, variant, pooling_method):
        """Test output dimensions match expected values."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=variant,
            pooling_method=pooling_method,
            pretrained=False,
        )

        output_spec = encoder.get_output_specification()
        expected_dim = VARIANT_OUTPUT_DIMS[variant][pooling_method]

        assert output_spec.dimensions[EncoderOutputKeys.RGBD.value] == expected_dim


    def test_output_keys_are_strings(self):
        """Test output keys are strings."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
        )

        output_spec = encoder.get_output_specification()
        assert all(isinstance(key, str) for key in output_spec.features)


    def test_output_dimensions_are_integers(self):
        """Test output dimensions are integers."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
        )

        output_spec = encoder.get_output_specification()
        assert all(isinstance(v, int) for v in output_spec.dimensions.values())


@pytest.mark.unit
class TestDFormerEncoderGradients:
    """Test gradient behavior."""


    @pytest.mark.parametrize("variant", [
        DFormerVariant.SMALL.value,
        DFormerVariant.BASE.value,
    ])
    def test_gradients_enabled_unfrozen(self, input_dict_4d, variant):
        """Test gradients flow when not frozen."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=variant,
            frozen=False,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"],
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }
        output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]
        loss = output.sum()
        loss.backward()

        has_grad = False
        for param in encoder.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        assert has_grad


    @pytest.mark.parametrize("variant", [
        DFormerVariant.SMALL.value,
        DFormerVariant.BASE.value,
    ])
    def test_gradients_disabled_frozen(self, input_dict_4d, variant):
        """Test gradients don't flow when frozen."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=variant,
            frozen=True,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"],
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }
        with torch.no_grad():
            output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        assert not output.requires_grad

        for param in encoder.parameters():
            assert not param.requires_grad


@pytest.mark.unit
class TestDFormerEncoderIntegration:
    """Integration tests for complete workflows."""


    def test_complete_forward_backward_pass(self, input_dict_4d):
        """Test complete forward and backward pass."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        encoder.train()
        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"],
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }
        output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]
        loss = output.mean()
        loss.backward()

        assert output.requires_grad
        assert loss.requires_grad


    def test_eval_mode_no_gradients(self, input_dict_4d):
        """Test eval mode doesn't compute gradients."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        encoder.eval()
        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"],
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }
        with torch.no_grad():
            output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        assert not output.requires_grad


    @pytest.mark.parametrize("variant", [
        DFormerVariant.SMALL.value,
        DFormerVariant.BASE.value,
    ])
    def test_consistent_output_shapes(self, batch_size, image_size, variant):
        """Test output shapes are consistent across multiple forward passes."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=variant,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        H, W = image_size
        input1 = {
            Cameras.LEFT.value: torch.randn(batch_size, 3, H, W),
            Cameras.DEPTH.value: torch.randn(batch_size, 1, H, W),
        }
        input2 = {
            Cameras.LEFT.value: torch.randn(batch_size, 3, H, W),
            Cameras.DEPTH.value: torch.randn(batch_size, 1, H, W),
        }

        output1 = encoder(input1)[EncoderOutputKeys.RGBD.value]
        output2 = encoder(input2)[EncoderOutputKeys.RGBD.value]

        assert output1.shape == output2.shape


    def test_multiple_camera_types(self, image_size):
        """Test encoders work with different camera types."""
        H, W = image_size

        for camera in [Cameras.LEFT.value, Cameras.RIGHT.value]:
            encoder = DFormerEncoder(
                input_keys=[camera, Cameras.DEPTH.value],
                variant=DFormerVariant.SMALL.value,
                pretrained=False,
            )

            input_dict = {
                camera: torch.randn(2, 3, H, W),
                Cameras.DEPTH.value: torch.randn(2, 1, H, W),
            }
            output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

            expected_output_dim = VARIANT_OUTPUT_DIMS[DFormerVariant.SMALL.value][
                PoolingMethod.AVERAGE.value
            ]
            assert output.shape == (2, expected_output_dim)


    @pytest.mark.parametrize("variant", [
        DFormerVariant.SMALL.value,
        DFormerVariant.BASE.value,
    ])
    def test_batch_independence(self, image_size, variant):
        """Test that samples in batch are processed independently."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=variant,
            pooling_method=PoolingMethod.AVERAGE.value,
            pretrained=False,
        )

        encoder.eval()

        H, W = image_size

        single_input = {
            Cameras.LEFT.value: torch.randn(1, 3, H, W),
            Cameras.DEPTH.value: torch.randn(1, 1, H, W),
        }
        batch_input = {
            Cameras.LEFT.value: torch.cat([single_input[Cameras.LEFT.value]] * 3, dim=0),
            Cameras.DEPTH.value: torch.cat([single_input[Cameras.DEPTH.value]] * 3, dim=0),
        }

        with torch.no_grad():
            single_output = encoder(single_input)[EncoderOutputKeys.RGBD.value]
            batch_output = encoder(batch_input)[EncoderOutputKeys.RGBD.value]

        assert torch.allclose(single_output, batch_output[0:1], atol=1e-5)
        assert torch.allclose(single_output, batch_output[1:2], atol=1e-5)
        assert torch.allclose(single_output, batch_output[2:3], atol=1e-5)


    def test_deterministic_output_eval_mode(self, input_dict_4d):
        """Test that eval mode produces deterministic outputs."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=DFormerVariant.SMALL.value,
            pretrained=False,
        )

        encoder.eval()
        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"],
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }

        with torch.no_grad():
            output1 = encoder(input_dict)[EncoderOutputKeys.RGBD.value]
            output2 = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        assert torch.allclose(output1, output2)


    @pytest.mark.parametrize("variant", [
        DFormerVariant.SMALL.value,
        DFormerVariant.BASE.value,
        DFormerVariant.LARGE.value,
    ])
    def test_all_variants_forward_pass(self, batch_size, image_size, variant):
        """Test forward pass works for all variants."""
        encoder = DFormerEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            variant=variant,
            pretrained=False,
        )

        H, W = image_size
        input_dict = {
            Cameras.LEFT.value: torch.randn(batch_size, 3, H, W),
            Cameras.DEPTH.value: torch.randn(batch_size, 1, H, W),
        }
        output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        assert output.shape[0] == batch_size
        assert output.ndim == 2