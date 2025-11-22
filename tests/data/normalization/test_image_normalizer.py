import pytest
import torch
import numpy as np

from refactoring.data.normalization.image_normalizer import (
    create_image_normalizer,
    get_rgb_image_normalizer,
    get_depth_image_normalizer,
    get_range_normalizer_from_stat,
    array_to_stats,
    _get_output_range,
    _to_tensor,
    _create_linear_scaling_normalizer,
    _create_standardization_normalizer,
    _compute_scaled_values,
)
from refactoring.data.normalization.normalizer import (
    SingleFieldLinearNormalizer,
    SequentialNormalizer,
    LinearNormalizer
)
from refactoring.data.constants import (
    ImageNormalizationType,
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    IMAGENET_DEPTH_MEAN,
    IMAGENET_DEPTH_STD,
)


@pytest.fixture
def scalar_stats():
    return {
        'input_min': 0.0,
        'input_max': 10.0,
        'input_mean': 5.0,
        'input_std': 2.0
    }


@pytest.fixture
def array_stats():
    return {
        'input_min': np.array([0.0, 0.0, 0.0]),
        'input_max': np.array([1.0, 1.0, 1.0]),
        'input_mean': np.array([0.5, 0.5, 0.5]),
        'input_std': np.array([0.288675, 0.288675, 0.288675])
    }


@pytest.fixture
def rgb_imagenet_stats():
    return {
        'mean': np.array(IMAGENET_RGB_MEAN, dtype=np.float32),
        'std': np.array(IMAGENET_RGB_STD, dtype=np.float32)
    }


@pytest.fixture
def test_tensor_scalar():
    return torch.linspace(0.0, 10.0, 101).unsqueeze(-1)


@pytest.fixture
def test_tensor_rgb():
    return torch.rand(5, 3, 32, 32)


class TestGetOutputRange:

    @pytest.mark.parametrize("norm_type,expected", [
        (ImageNormalizationType.ZERO_TO_ONE.value, (0.0, 1.0)),
        (ImageNormalizationType.MINUS_ONE_TO_ONE.value, (-1.0, 1.0)),
        (ImageNormalizationType.IMAGENET.value, (0.0, 1.0)),
    ])
    def test_valid_norm_types(self, norm_type, expected):
        result = _get_output_range(norm_type)
        assert result == expected


    def test_invalid_norm_type(self):
        with pytest.raises(ValueError, match="Unsupported normalization type"):
            _get_output_range("invalid_type")


class TestToTensor:

    def test_scalar_conversion(self):
        result = _to_tensor(5.0)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)
        assert result.item() == 5.0


    def test_numpy_array_conversion(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _to_tensor(arr)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))


    def test_device_placement(self):
        result = _to_tensor(5.0, device='cpu')
        assert result.device.type == 'cpu'


    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_placement(self):
        result = _to_tensor(5.0, device='cuda:0')
        assert result.device.type == 'cuda'


class TestComputeScaledValues:

    def test_mean_scaling(self):
        result = _compute_scaled_values(
            values=5.0,
            input_min=0.0,
            input_max=10.0,
            output_min=0.0,
            output_max=1.0,
            is_std=False
        )
        assert np.isclose(result, 0.5)


    def test_std_scaling(self):
        result = _compute_scaled_values(
            values=2.0,
            input_min=0.0,
            input_max=10.0,
            output_min=0.0,
            output_max=1.0,
            is_std=True
        )
        assert np.isclose(result, 0.2)


    def test_array_scaling(self):
        values = np.array([2.0, 4.0, 6.0])
        input_min = np.array([0.0, 0.0, 0.0])
        input_max = np.array([10.0, 10.0, 10.0])
        result = _compute_scaled_values(
            values=values,
            input_min=input_min,
            input_max=input_max,
            output_min=0.0,
            output_max=1.0,
            is_std=False
        )
        expected = np.array([0.2, 0.4, 0.6])
        assert np.allclose(result, expected)


class TestCreateLinearScalingNormalizer:

    def test_scalar_inputs(self, scalar_stats):
        normalizer = _create_linear_scaling_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            output_min=0.0,
            output_max=1.0
        )
        assert isinstance(normalizer, SingleFieldLinearNormalizer)
        assert normalizer.params_dict['scale'].shape == (1,)
        assert normalizer.params_dict['offset'].shape == (1,)


    def test_array_inputs(self, array_stats):
        normalizer = _create_linear_scaling_normalizer(
            input_min=array_stats['input_min'],
            input_max=array_stats['input_max'],
            input_mean=array_stats['input_mean'],
            input_std=array_stats['input_std'],
            output_min=0.0,
            output_max=1.0
        )
        assert isinstance(normalizer, SingleFieldLinearNormalizer)
        assert normalizer.params_dict['scale'].shape == (3,)
        assert normalizer.params_dict['offset'].shape == (3,)


    def test_correct_scaling_formula(self, scalar_stats):
        normalizer = _create_linear_scaling_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            output_min=-1.0,
            output_max=1.0
        )
        scale = normalizer.params_dict['scale']
        offset = normalizer.params_dict['offset']
        expected_scale = 2.0 / 10.0
        expected_offset = -1.0
        assert torch.isclose(scale, torch.tensor([expected_scale]))
        assert torch.isclose(offset, torch.tensor([expected_offset]))


class TestCreateStandardizationNormalizer:

    def test_scalar_standardization(self):
        normalizer = _create_standardization_normalizer(
            input_min=0.0,
            input_max=1.0,
            input_mean=0.5,
            input_std=0.288675,
            standardization_mean=0.5,
            standardization_std=0.25,
            device=None
        )
        assert isinstance(normalizer, SingleFieldLinearNormalizer)
        expected_scale = 1.0 / 0.25
        expected_offset = -0.5 / 0.25
        assert torch.isclose(normalizer.params_dict['scale'], torch.tensor([expected_scale]))
        assert torch.isclose(normalizer.params_dict['offset'], torch.tensor([expected_offset]))


    def test_array_standardization(self, rgb_imagenet_stats):
        normalizer = _create_standardization_normalizer(
            input_min=np.zeros(3),
            input_max=np.ones(3),
            input_mean=np.full(3, 0.5),
            input_std=np.full(3, 0.288675),
            standardization_mean=rgb_imagenet_stats['mean'],
            standardization_std=rgb_imagenet_stats['std'],
            device=None
        )
        assert isinstance(normalizer, SingleFieldLinearNormalizer)
        assert normalizer.params_dict['scale'].shape == (3,)
        assert normalizer.params_dict['offset'].shape == (3,)


class TestArrayToStats:

    def test_1d_array(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = array_to_stats(arr)
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['mean'] == 3.0
        assert np.isclose(stats['std'], np.std([1, 2, 3, 4, 5]))


    def test_2d_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats = array_to_stats(arr)
        assert np.array_equal(stats['min'], np.array([1.0, 2.0]))
        assert np.array_equal(stats['max'], np.array([3.0, 4.0]))


class TestGetRangeNormalizerFromStat:

    def test_creates_normalizer_from_stats(self):
        stat = {
            'min': torch.tensor([0.0, 0.0]),
            'max': torch.tensor([10.0, 20.0]),
            'mean': torch.tensor([5.0, 10.0]),
            'std': torch.tensor([2.0, 4.0])
        }
        normalizer = get_range_normalizer_from_stat(
            stat=stat,
            output_min=-1.0,
            output_max=1.0
        )
        assert isinstance(normalizer, SingleFieldLinearNormalizer)


    def test_handles_zero_range_dimensions(self):
        stat = {
            'min': torch.tensor([0.0, 5.0]),
            'max': torch.tensor([10.0, 5.0]),
            'mean': torch.tensor([5.0, 5.0]),
            'std': torch.tensor([2.0, 0.0])
        }
        normalizer = get_range_normalizer_from_stat(
            stat=stat,
            output_min=-1.0,
            output_max=1.0,
            range_eps=1e-7
        )
        assert isinstance(normalizer, SingleFieldLinearNormalizer)


class TestCreateImageNormalizer:

    def test_simple_scaling_no_standardization(self, scalar_stats):
        normalizer = create_image_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value
        )
        assert isinstance(normalizer, SingleFieldLinearNormalizer)


    def test_with_standardization(self, scalar_stats):
        normalizer = create_image_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            norm_type=ImageNormalizationType.IMAGENET.value,
            standardization_mean=IMAGENET_DEPTH_MEAN,
            standardization_std=IMAGENET_DEPTH_STD
        )
        assert isinstance(normalizer, SequentialNormalizer)
        assert len(normalizer.normalizers) == 2


    def test_round_trip_normalization(self, scalar_stats, test_tensor_scalar):
        normalizer = create_image_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            norm_type=ImageNormalizationType.MINUS_ONE_TO_ONE.value
        )
        normalized = normalizer.normalize(test_tensor_scalar)
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(test_tensor_scalar, unnormalized, atol=1e-6)


    def test_calls_scaling_normalizer_without_standardization(self, scalar_stats):
        normalizer = create_image_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value
        )
        assert isinstance(normalizer, SingleFieldLinearNormalizer)
        assert not isinstance(normalizer, SequentialNormalizer)


    def test_calls_both_when_standardization_provided(self, scalar_stats):
        normalizer = create_image_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            norm_type=ImageNormalizationType.IMAGENET.value,
            standardization_mean=IMAGENET_DEPTH_MEAN,
            standardization_std=IMAGENET_DEPTH_STD
        )
        assert isinstance(normalizer, SequentialNormalizer)
        assert len(normalizer.normalizers) == 2


class TestGetRgbImageNormalizer:

    @pytest.mark.parametrize("norm_type", [
        ImageNormalizationType.ZERO_TO_ONE.value,
        ImageNormalizationType.MINUS_ONE_TO_ONE.value
    ])
    def test_simple_norm_types(self, norm_type):
        normalizer = get_rgb_image_normalizer(norm_type=norm_type)
        assert isinstance(normalizer, SingleFieldLinearNormalizer)


    def test_imagenet_returns_single_field_normalizer(self):
        normalizer = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.IMAGENET.value
        )
        assert isinstance(normalizer, SingleFieldLinearNormalizer)
        assert normalizer.params_dict['scale'].shape == (3,)
        assert normalizer.params_dict['offset'].shape == (3,)


    def test_imagenet_per_channel_statistics(self):
        normalizer = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.IMAGENET.value
        )
        expected_scale = 1.0 / np.array(IMAGENET_RGB_STD)
        expected_offset = -np.array(IMAGENET_RGB_MEAN) / np.array(IMAGENET_RGB_STD)

        assert torch.allclose(
            normalizer.params_dict['scale'],
            torch.from_numpy(expected_scale).float(),
            atol=1e-6
        )
        assert torch.allclose(
            normalizer.params_dict['offset'],
            torch.from_numpy(expected_offset).float(),
            atol=1e-6
        )


    def test_zero_to_one_is_identity(self):
        normalizer = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value
        )
        test_data = torch.rand(5, 3, 32, 32)
        normalized = normalizer.normalize(test_data)
        assert torch.allclose(test_data, normalized, atol=1e-6)


    def test_minus_one_to_one_range(self):
        normalizer = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.MINUS_ONE_TO_ONE.value
        )
        test_data = torch.tensor([[0.0], [0.5], [1.0]])
        normalized = normalizer.normalize(test_data)
        expected = torch.tensor([[-1.0], [0.0], [1.0]])
        assert torch.allclose(normalized, expected, atol=1e-6)


    def test_imagenet_normalizes_mean_to_zero(self):
        normalizer = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.IMAGENET.value
        )
        mean_data = torch.zeros(5, 3)
        for c in range(3):
            mean_data[:, c] = IMAGENET_RGB_MEAN[c]

        normalized = normalizer.normalize(mean_data)
        assert torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-6)


    def test_imagenet_uses_standardization_directly(self):
        normalizer = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.IMAGENET.value
        )
        assert isinstance(normalizer, SingleFieldLinearNormalizer)
        assert normalizer.params_dict['scale'].shape == (3,)


    def test_non_imagenet_uses_create_image_normalizer(self):
        normalizer = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value
        )
        assert isinstance(normalizer, SingleFieldLinearNormalizer)
        assert normalizer.params_dict['scale'].shape == (1,)


class TestGetDepthImageNormalizer:

    def test_zero_to_one(self, scalar_stats, test_tensor_scalar):
        normalizer = get_depth_image_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value
        )
        normalized = normalizer.normalize(test_tensor_scalar)
        assert torch.allclose(normalized.min(), torch.tensor([0.0]), atol=1e-6)
        assert torch.allclose(normalized.max(), torch.tensor([1.0]), atol=1e-6)


    def test_minus_one_to_one(self, scalar_stats, test_tensor_scalar):
        normalizer = get_depth_image_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            norm_type=ImageNormalizationType.MINUS_ONE_TO_ONE.value
        )
        normalized = normalizer.normalize(test_tensor_scalar)
        assert torch.allclose(normalized.min(), torch.tensor([-1.0]), atol=1e-6)
        assert torch.allclose(normalized.max(), torch.tensor([1.0]), atol=1e-6)


    def test_imagenet_returns_sequential(self, scalar_stats):
        normalizer = get_depth_image_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            norm_type=ImageNormalizationType.IMAGENET.value
        )
        assert isinstance(normalizer, SequentialNormalizer)
        assert len(normalizer.normalizers) == 2


    def test_imagenet_round_trip(self, scalar_stats, test_tensor_scalar):
        normalizer = get_depth_image_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            norm_type=ImageNormalizationType.IMAGENET.value
        )
        normalized = normalizer.normalize(test_tensor_scalar)
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(test_tensor_scalar, unnormalized, atol=1e-6)


    def test_delegates_to_create_image_normalizer_without_standardization(self, scalar_stats):
        normalizer = get_depth_image_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value
        )
        assert isinstance(normalizer, SingleFieldLinearNormalizer)
        assert not isinstance(normalizer, SequentialNormalizer)


    def test_adds_imagenet_standardization(self, scalar_stats):
        normalizer = get_depth_image_normalizer(
            input_min=scalar_stats['input_min'],
            input_max=scalar_stats['input_max'],
            input_mean=scalar_stats['input_mean'],
            input_std=scalar_stats['input_std'],
            norm_type=ImageNormalizationType.IMAGENET.value
        )
        assert isinstance(normalizer, SequentialNormalizer)
        assert len(normalizer.normalizers) == 2


class TestLinearNormalizerIntegration:

    def test_store_and_retrieve_rgb_normalizers(self):
        normalizer = LinearNormalizer()
        rgb_zero = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value
        )
        rgb_imagenet = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.IMAGENET.value
        )

        normalizer['rgb_zero'] = rgb_zero
        normalizer['rgb_imagenet'] = rgb_imagenet

        retrieved_zero = normalizer['rgb_zero']
        retrieved_imagenet = normalizer['rgb_imagenet']

        assert isinstance(retrieved_zero, SingleFieldLinearNormalizer)
        assert isinstance(retrieved_imagenet, SingleFieldLinearNormalizer)


    def test_store_and_retrieve_depth_normalizers(self, scalar_stats):
        normalizer = LinearNormalizer()
        depth_zero = get_depth_image_normalizer(
            **scalar_stats,
            norm_type=ImageNormalizationType.ZERO_TO_ONE.value
        )
        depth_imagenet = get_depth_image_normalizer(
            **scalar_stats,
            norm_type=ImageNormalizationType.IMAGENET.value
        )

        normalizer['depth_zero'] = depth_zero
        normalizer['depth_imagenet'] = depth_imagenet

        retrieved_zero = normalizer['depth_zero']
        retrieved_imagenet = normalizer['depth_imagenet']

        assert isinstance(retrieved_zero, SingleFieldLinearNormalizer)
        assert isinstance(retrieved_imagenet, SequentialNormalizer)


    def test_state_dict_save_load(self):
        normalizer = LinearNormalizer()
        normalizer['rgb'] = get_rgb_image_normalizer(
            norm_type=ImageNormalizationType.IMAGENET.value
        )

        state_dict = normalizer.state_dict()
        new_normalizer = LinearNormalizer()
        new_normalizer.load_state_dict(state_dict)

        test_data = torch.rand(10, 3)
        orig_out = normalizer['rgb'].normalize(test_data)
        loaded_out = new_normalizer['rgb'].normalize(test_data)

        assert torch.allclose(orig_out, loaded_out, atol=1e-6)