import pytest
import torch

from refactoring.models.encoding.encoders.depth import LightGeometricEncoder
from refactoring.models.encoding.encoders.constants import PoolingMethod, EncoderOutputKeys
from refactoring.data.constants import Cameras
from refactoring.models.layers.constants import AttentionDecompositionMode


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
    return {
        "rgb": torch.randn(batch_size, 3, H, W),
        "depth": torch.randn(batch_size, 1, H, W),
    }


@pytest.fixture
def input_dict_5d(batch_size, temporal_length, image_size):
    H, W = image_size
    return {
        "rgb": torch.randn(batch_size, temporal_length, 3, H, W),
        "depth": torch.randn(batch_size, temporal_length, 1, H, W),
    }


@pytest.mark.unit
class TestLightGeometricEncoderInitialization:

    @pytest.mark.parametrize("embedding_dimension", [256, 512])
    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.SPATIAL_SOFTMAX.value,
        PoolingMethod.GLOBAL_AVERAGE.value,
    ])
    def test_init(self, embedding_dimension, pooling_method):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            embedding_dimension=embedding_dimension,
            pooling_method=pooling_method,
        )

        output_spec = encoder.get_output_specification()
        expected_output_dim = embedding_dimension * 2 if pooling_method == PoolingMethod.SPATIAL_SOFTMAX.value else embedding_dimension
        assert output_spec.dimensions[EncoderOutputKeys.RGBD.value] == expected_output_dim


    @pytest.mark.parametrize("decomposition_mode", [
        AttentionDecompositionMode.FULL.value,
        AttentionDecompositionMode.SEPARABLE.value,
    ])
    def test_init_decomposition_modes(self, decomposition_mode):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            decomposition_mode=decomposition_mode,
        )

        assert encoder.decomposition_mode == AttentionDecompositionMode(decomposition_mode)


    def test_init_frozen(self):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            frozen=True,
        )

        for param in encoder.parameters():
            assert param.requires_grad is False


    def test_init_missing_depth_input(self):
        with pytest.raises(ValueError, match="Missing required inputs:.*"):
            LightGeometricEncoder(
                input_keys=Cameras.LEFT.value,
            )


    def test_init_missing_rgb_input(self):
        with pytest.raises(ValueError, match="Exactly one from"):
            LightGeometricEncoder(
                input_keys=Cameras.DEPTH.value,
            )


    def test_init_multiple_rgb_inputs(self):
        with pytest.raises(ValueError, match="Exactly one from"):
            LightGeometricEncoder(
                input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value],
            )


    def test_get_output_specification(self):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )

        output_spec = encoder.get_output_specification()
        assert output_spec.features == [EncoderOutputKeys.RGBD.value]
        assert isinstance(output_spec.features, list)
        assert EncoderOutputKeys.RGBD.value in output_spec.dimensions


@pytest.mark.unit
class TestLightGeometricEncoderForward:

    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.SPATIAL_SOFTMAX.value,
        PoolingMethod.GLOBAL_AVERAGE.value,
    ])
    def test_forward_4d_input(self, input_dict_4d, pooling_method):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            pooling_method=pooling_method,
        )

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"],
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }
        output_dict = encoder(input_dict)

        assert EncoderOutputKeys.RGBD.value in output_dict
        output = output_dict[EncoderOutputKeys.RGBD.value]
        expected_output_dim = 512 * 2 if pooling_method == PoolingMethod.SPATIAL_SOFTMAX.value else 512
        assert output.shape == (2, expected_output_dim)


    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.SPATIAL_SOFTMAX.value,
        PoolingMethod.GLOBAL_AVERAGE.value,
    ])
    def test_forward_5d_input(self, input_dict_5d, pooling_method):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            pooling_method=pooling_method,
        )

        input_dict = {
            Cameras.LEFT.value: input_dict_5d["rgb"],
            Cameras.DEPTH.value: input_dict_5d["depth"],
        }
        output_dict = encoder(input_dict)

        assert EncoderOutputKeys.RGBD.value in output_dict
        output = output_dict[EncoderOutputKeys.RGBD.value]
        expected_output_dim = 512 * 2 if pooling_method == PoolingMethod.SPATIAL_SOFTMAX.value else 512
        assert output.shape == (2, 4, expected_output_dim)


    def test_forward_train_mode(self, input_dict_4d):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
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
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.RIGHT.value, Cameras.DEPTH.value],
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
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            decomposition_mode=decomposition_mode,
        )

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"],
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }
        output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        assert output.shape[0] == 2


@pytest.mark.unit
class TestLightGeometricEncoderTemporalHandling:

    @pytest.mark.parametrize("temporal_length", [1, 4, 8, 16])
    def test_temporal_various_lengths(self, batch_size, image_size, temporal_length):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )

        H, W = image_size
        input_dict = {
            Cameras.LEFT.value: torch.randn(batch_size, temporal_length, 3, H, W),
            Cameras.DEPTH.value: torch.randn(batch_size, temporal_length, 1, H, W),
        }
        output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        assert output.shape == (batch_size, temporal_length, 512)


    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.SPATIAL_SOFTMAX.value,
        PoolingMethod.GLOBAL_AVERAGE.value,
    ])
    def test_temporal_single_timestep(self, batch_size, image_size, pooling_method):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            pooling_method=pooling_method,
        )

        H, W = image_size
        input_dict = {
            Cameras.LEFT.value: torch.randn(batch_size, 1, 3, H, W),
            Cameras.DEPTH.value: torch.randn(batch_size, 1, 1, H, W),
        }
        output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        expected_output_dim = 512 * 2 if pooling_method == PoolingMethod.SPATIAL_SOFTMAX.value else 512
        assert output.shape == (batch_size, 1, expected_output_dim)


    def test_temporal_consistency(self, batch_size, image_size):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
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
class TestLightGeometricEncoderOutputSpecification:
    """Test output specification methods."""


    def test_output_specification_structure(self):
        """Test output specification has correct structure."""
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )

        output_spec = encoder.get_output_specification()

        assert hasattr(output_spec, 'features')
        assert hasattr(output_spec, 'dimensions')
        assert isinstance(output_spec.features, list)
        assert isinstance(output_spec.dimensions, dict)


    @pytest.mark.parametrize("embedding_dimension,pooling_method", [
        (256, PoolingMethod.SPATIAL_SOFTMAX.value),
        (512, PoolingMethod.SPATIAL_SOFTMAX.value),
        (256, PoolingMethod.GLOBAL_AVERAGE.value),
        (512, PoolingMethod.GLOBAL_AVERAGE.value),
    ])
    def test_output_dimensions_correct(self, embedding_dimension, pooling_method):
        """Test output dimensions match expected values."""
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            embedding_dimension=embedding_dimension,
            pooling_method=pooling_method,
        )

        output_spec = encoder.get_output_specification()
        expected_dim = embedding_dimension * 2 if pooling_method == PoolingMethod.SPATIAL_SOFTMAX.value else embedding_dimension

        assert output_spec.dimensions[EncoderOutputKeys.RGBD.value] == expected_dim


    def test_output_keys_are_strings(self):
        """Test output keys are strings."""
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )

        output_spec = encoder.get_output_specification()
        assert all(isinstance(key, str) for key in output_spec.features)


    def test_output_dimensions_are_integers(self):
        """Test output dimensions are integers."""
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )

        output_spec = encoder.get_output_specification()
        assert all(isinstance(v, int) for v in output_spec.dimensions.values())


@pytest.mark.unit
class TestLightGeometricEncoderGradients:

    def test_gradients_enabled_unfrozen(self, input_dict_4d):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            frozen=False,
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


    def test_gradients_disabled_frozen(self, input_dict_4d):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
            frozen=True,
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
class TestLightGeometricEncoderIntegration:

    def test_complete_forward_backward_pass(self, input_dict_4d):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
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
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )

        encoder.eval()
        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"],
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }
        with torch.no_grad():
            output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        assert not output.requires_grad


    def test_consistent_output_shapes(self, batch_size, image_size):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
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
        H, W = image_size

        for camera in [Cameras.LEFT.value, Cameras.RIGHT.value]:
            encoder = LightGeometricEncoder(
                input_keys=[camera, Cameras.DEPTH.value],
            )

            input_dict = {
                camera: torch.randn(2, 3, H, W),
                Cameras.DEPTH.value: torch.randn(2, 1, H, W),
            }
            output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

            assert output.shape == (2, 512)


    def test_batch_independence(self, image_size):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
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
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
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


@pytest.mark.unit
class TestLightGeometricEncoderEdgeCases:

    def test_forward_missing_rgb_key(self, input_dict_4d):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )

        input_dict = {
            Cameras.DEPTH.value: input_dict_4d["depth"],
        }

        with pytest.raises(KeyError):
            encoder(input_dict)


    def test_forward_missing_depth_key(self, input_dict_4d):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )

        input_dict = {
            Cameras.LEFT.value: input_dict_4d["rgb"],
        }

        with pytest.raises(KeyError):
            encoder(input_dict)


    def test_forward_mismatched_spatial_dimensions(self, batch_size):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )

        input_dict = {
            Cameras.LEFT.value: torch.randn(batch_size, 3, 224, 224),
            Cameras.DEPTH.value: torch.randn(batch_size, 1, 112, 112),
        }

        output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]
        assert output.shape[0] == batch_size


    def test_forward_different_batch_sizes_error(self):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )

        input_dict = {
            Cameras.LEFT.value: torch.randn(2, 3, 224, 224),
            Cameras.DEPTH.value: torch.randn(4, 1, 224, 224),
        }

        with pytest.raises(RuntimeError):
            encoder(input_dict)


@pytest.mark.unit
class TestLightGeometricEncoderPerformance:

    def test_forward_pass_memory_efficient(self, batch_size, image_size):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )

        H, W = image_size
        input_dict = {
            Cameras.LEFT.value: torch.randn(batch_size, 3, H, W),
            Cameras.DEPTH.value: torch.randn(batch_size, 1, H, W),
        }

        encoder.eval()
        with torch.no_grad():
            output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        assert output is not None


    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_scalability_batch_size(self, batch_size, image_size):
        encoder = LightGeometricEncoder(
            input_keys=[Cameras.LEFT.value, Cameras.DEPTH.value],
        )

        H, W = image_size
        input_dict = {
            Cameras.LEFT.value: torch.randn(batch_size, 3, H, W),
            Cameras.DEPTH.value: torch.randn(batch_size, 1, H, W),
        }

        encoder.eval()
        with torch.no_grad():
            output = encoder(input_dict)[EncoderOutputKeys.RGBD.value]

        assert output.shape[0] == batch_size