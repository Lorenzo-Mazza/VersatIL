import pytest
import torch
from refactoring.data.constants import KinematicsNormalizationType
from refactoring.data.normalize.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
    SequentialNormalizer,
)


@pytest.fixture
def random_data_4d():
    """Generate random 4D test data."""
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    return data


@pytest.fixture
def random_data_4d_with_constant():
    """Generate random 4D test data with one constant channel."""
    data = torch.zeros((100, 10, 9, 2)).uniform_()
    data[..., 0, 0] = 0  # Make one channel constant
    return data


@pytest.fixture
def random_dict_data():
    """Generate random dictionary data."""
    return {
        'obs': torch.zeros((1000, 128, 9, 2)).uniform_() * 512,
        'action': torch.zeros((1000, 128, 2)).uniform_() * 512
    }


class TestSingleFieldLinearNormalizer:
    """Tests for SingleFieldLinearNormalizer."""


    @pytest.mark.parametrize("last_n_dims", [0, 1, 2])
    def test_min_max_normalization_shape_preserved(self, random_data_4d_with_constant, last_n_dims):
        """Test that min-max normalization preserves shape."""
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            random_data_4d_with_constant,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            last_n_dims=last_n_dims
        )

        normalized = normalizer.normalize(random_data_4d_with_constant)
        assert normalized.shape == random_data_4d_with_constant.shape


    def test_min_max_normalization_range(self, random_data_4d):
        """Test that min-max normalization produces correct range for non-constant data."""
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            random_data_4d,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            last_n_dims=2
        )

        normalized = normalizer.normalize(random_data_4d)

        # Check bounds (allowing small numerical error)
        assert torch.allclose(normalized.max(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(normalized.min(), torch.tensor(-1.0), atol=1e-6)


    def test_min_max_normalization_invertible(self, random_data_4d_with_constant):
        """Test that normalization is invertible."""
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            random_data_4d_with_constant,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            last_n_dims=2
        )

        normalized = normalizer.normalize(random_data_4d_with_constant)
        unnormalized = normalizer.unnormalize(normalized)

        assert torch.allclose(random_data_4d_with_constant, unnormalized, atol=1e-6)


    def test_min_max_without_offset(self, random_data_4d):
        """Test min-max normalization without offset (zero-centered)."""
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            random_data_4d,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            last_n_dims=1,
            fit_offset=False
        )

        normalized = normalizer.normalize(random_data_4d)

        # Without offset, range should be [0, 1] for positive data
        assert normalized.shape == random_data_4d.shape
        assert torch.allclose(normalized.max(), torch.tensor(1.0), atol=1e-3)
        assert torch.allclose(normalized.min(), torch.tensor(0.0), atol=1e-3)

        # Check invertibility
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(random_data_4d, unnormalized, atol=1e-6)


    def test_gaussian_normalization(self, random_data_4d):
        """Test Gaussian (z-score) normalization."""
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            random_data_4d,
            mode=KinematicsNormalizationType.GAUSSIAN.value,
            last_n_dims=0  # Normalize entire tensor
        )

        normalized = normalizer.normalize(random_data_4d)

        # Check shape
        assert normalized.shape == random_data_4d.shape

        # Check mean ≈ 0 and std ≈ 1
        assert torch.allclose(normalized.mean(), torch.tensor(0.0), atol=1e-3)
        assert torch.allclose(normalized.std(), torch.tensor(1.0), atol=1e-3)

        # Check invertibility
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(random_data_4d, unnormalized, atol=1e-6)


    @pytest.mark.parametrize("output_min,output_max", [
        (-1.0, 1.0),
        (0.0, 1.0),
        (-2.0, 2.0),
    ])
    def test_custom_output_range(self, random_data_4d, output_min, output_max):
        """Test normalization with custom output range."""
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            random_data_4d,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            output_min=output_min,
            output_max=output_max,
            last_n_dims=0
        )

        normalized = normalizer.normalize(random_data_4d)

        assert torch.allclose(normalized.min(), torch.tensor(output_min), atol=1e-4)
        assert torch.allclose(normalized.max(), torch.tensor(output_max), atol=1e-4)


    def test_input_output_stats(self, random_data_4d_with_constant):
        """Test that input and output stats are computed correctly."""
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            random_data_4d_with_constant,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            last_n_dims=2
        )

        input_stats = normalizer.get_input_stats()
        output_stats = normalizer.get_output_stats()

        # Check all expected keys exist
        for key in ['min', 'max', 'mean', 'std']:
            assert key in input_stats
            assert key in output_stats

        # With last_n_dims=2, we have 9*2=18 channels
        # The first channel (index 0) is constant at 0.0 and should map to 0.0 (middle of [-1, 1])
        # The other channels should map to [-1, 1]
        assert output_stats['min'].shape[0] == 18
        assert output_stats['max'].shape[0] == 18

        # Check constant channel: should be at middle of output range
        assert torch.allclose(output_stats['min'][0], torch.tensor(0.0), atol=1e-4)
        assert torch.allclose(output_stats['max'][0], torch.tensor(0.0), atol=1e-4)

        # Check non-constant channels: should span the full output range
        for i in range(1, 18):
            assert torch.allclose(output_stats['min'][i], torch.tensor(-1.0), atol=1e-4)
            assert torch.allclose(output_stats['max'][i], torch.tensor(1.0), atol=1e-4)


    def test_input_output_stats_without_constant(self, random_data_4d):
        """Test input/output stats for data without constant channels."""
        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            random_data_4d,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            last_n_dims=2,
            output_min=-1.0,
            output_max=1.0
        )

        output_stats = normalizer.get_output_stats()

        # All channels should span the full output range
        assert torch.allclose(output_stats['min'].min(), torch.tensor(-1.0), atol=1e-4)
        assert torch.allclose(output_stats['max'].max(), torch.tensor(1.0), atol=1e-4)


    def test_create_manual(self):
        """Test manual creation with explicit scale and offset."""
        scale = torch.tensor([2.0], dtype=torch.float32)
        offset = torch.tensor([1.0], dtype=torch.float32)
        input_stats = {
            'min': torch.tensor([0.0], dtype=torch.float32),
            'max': torch.tensor([10.0], dtype=torch.float32),
            'mean': torch.tensor([5.0], dtype=torch.float32),
            'std': torch.tensor([2.0], dtype=torch.float32),
        }

        normalizer = SingleFieldLinearNormalizer.create_manual(
            scale=scale,
            offset=offset,
            input_stats_dict=input_stats
        )

        # Test normalization: action_embedding * scale + offset
        test_data = torch.tensor([5.0])
        normalized = normalizer.normalize(test_data)
        expected = test_data * scale + offset
        assert torch.allclose(normalized, expected)


    def test_create_identity(self):
        """Test identity normalizer (no transformation)."""
        normalizer = SingleFieldLinearNormalizer.create_identity()

        test_data = torch.randn(10, 5)
        normalized = normalizer.normalize(test_data)

        # Identity should not change data
        assert torch.allclose(normalized, test_data)


    def test_numpy_input(self, random_data_4d):
        """Test that numpy arrays work as input."""
        normalizer = SingleFieldLinearNormalizer()
        numpy_data = random_data_4d.numpy()

        normalizer.fit(numpy_data, mode=KinematicsNormalizationType.MIN_MAX.value)
        normalized = normalizer.normalize(numpy_data)

        # Output should be torch tensor
        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == random_data_4d.shape


    def test_device_handling(self, random_data_4d):
        """Test that device parameter works correctly."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            random_data_4d,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            device=device
        )

        # Check that params are on correct device
        assert normalizer.params_dict['scale'].device.type == device.type


class TestLinearNormalizer:
    """Tests for LinearNormalizer (dict-based)."""


    def test_dict_normalization(self, random_dict_data):
        """Test normalization of dictionary data."""
        normalizer = LinearNormalizer()
        normalizer.fit(random_dict_data, mode=KinematicsNormalizationType.MIN_MAX.value)

        normalized = normalizer.normalize(random_dict_data)

        # Check all keys present
        assert set(normalized.keys()) == set(random_dict_data.keys())

        # Check shapes preserved
        for key in random_dict_data:
            assert normalized[key].shape == random_dict_data[key].shape


    def test_dict_normalization_invertible(self, random_dict_data):
        """Test that dict normalization is invertible."""
        normalizer = LinearNormalizer()
        normalizer.fit(random_dict_data, mode=KinematicsNormalizationType.MIN_MAX.value)

        normalized = normalizer.normalize(random_dict_data)
        unnormalized = normalizer.unnormalize(normalized)

        for key in random_dict_data:
            assert torch.allclose(random_dict_data[key], unnormalized[key], atol=1e-4)


    def test_dict_input_stats(self, random_dict_data):
        """Test that input stats are computed for dict data."""
        normalizer = LinearNormalizer()
        normalizer.fit(random_dict_data, mode=KinematicsNormalizationType.MIN_MAX.value)

        input_stats = normalizer.get_input_stats()

        # Should have stats for each key
        assert set(input_stats.keys()) == set(random_dict_data.keys())

        for key in random_dict_data:
            assert 'min' in input_stats[key]
            assert 'max' in input_stats[key]
            assert 'mean' in input_stats[key]
            assert 'std' in input_stats[key]


    def test_single_tensor_normalization(self, random_data_4d):
        """Test that LinearNormalizer works with single tensor (non-dict)."""
        normalizer = LinearNormalizer()
        normalizer.fit(random_data_4d, mode=KinematicsNormalizationType.MIN_MAX.value)

        normalized = normalizer.normalize(random_data_4d)

        assert normalized.shape == random_data_4d.shape
        assert torch.allclose(normalized.max(), torch.tensor(1.0), atol=1e-4)
        assert torch.allclose(normalized.min(), torch.tensor(-1.0), atol=1e-4)


    def test_getitem_setitem(self, random_data_4d):
        """Test __getitem__ and __setitem__ for field access."""
        normalizer = LinearNormalizer()
        data_dict = {'field1': random_data_4d, 'field2': random_data_4d * 2}
        normalizer.fit(data_dict)

        # Test __getitem__
        field1_norm = normalizer['field1']
        assert isinstance(field1_norm, SingleFieldLinearNormalizer)

        # Test that retrieved normalizer works
        test_data = torch.randn(5, 2)
        normalized = field1_norm.normalize(test_data)
        assert normalized.shape == test_data.shape


    def test_state_dict_save_load(self, random_dict_data):
        """Test saving and loading state dict."""
        normalizer = LinearNormalizer()
        normalizer.fit(random_dict_data)

        # Save state
        state_dict = normalizer.state_dict()

        # Create new normalizer and load
        new_normalizer = LinearNormalizer()
        new_normalizer.load_state_dict(state_dict)

        # Test that loaded normalizer works identically
        normalized1 = normalizer.normalize(random_dict_data)
        normalized2 = new_normalizer.normalize(random_dict_data)

        for key in random_dict_data:
            assert torch.allclose(normalized1[key], normalized2[key], atol=1e-6)


class TestSequentialNormalizer:
    """Tests for SequentialNormalizer."""


    def test_sequential_composition(self):
        """Test that sequential normalizers compose correctly."""
        # Create test data
        data = torch.zeros((100, 10, 9, 2)).uniform_() * 10

        # First normalizer: scale to [0, 1]
        output1_max = 1.0
        output1_min = 0.0
        input_min = data.min()
        input_max = data.max()
        input_mean = data.mean()
        input_std = data.std()

        scale1 = (output1_max - output1_min) / (input_max - input_min)
        offset1 = output1_min - scale1 * input_min
        input_stats1 = {
            'min': torch.tensor([input_min], dtype=torch.float32),
            'max': torch.tensor([input_max], dtype=torch.float32),
            'mean': torch.tensor([input_mean], dtype=torch.float32),
            'std': torch.tensor([input_std], dtype=torch.float32),
        }

        # Second normalizer: ImageNet-style normalization
        imagenet_mean = 0.456
        imagenet_std = 0.21
        input_mean2 = input_mean * scale1 + offset1
        input_std2 = input_std * scale1
        input_stats2 = {
            'min': torch.tensor([output1_min], dtype=torch.float32),
            'max': torch.tensor([output1_max], dtype=torch.float32),
            'mean': torch.tensor([input_mean2], dtype=torch.float32),
            'std': torch.tensor([input_std2], dtype=torch.float32),
        }
        scale2 = torch.tensor([1.0 / imagenet_std], dtype=torch.float32)
        offset2 = torch.tensor([-imagenet_mean / imagenet_std], dtype=torch.float32)

        # Create normalizers
        normalizer1 = SingleFieldLinearNormalizer.create_manual(
            scale=torch.tensor([scale1], dtype=torch.float32),
            offset=torch.tensor([offset1], dtype=torch.float32),
            input_stats_dict=input_stats1
        )
        normalizer2 = SingleFieldLinearNormalizer.create_manual(
            scale=scale2,
            offset=offset2,
            input_stats_dict=input_stats2
        )

        # Create sequential normalizer
        seq_normalizer = SequentialNormalizer(normalizers=[normalizer1, normalizer2])

        # Normalize
        normalized = seq_normalizer.normalize(data)

        # Check shape
        assert normalized.shape == data.shape

        # Check that it matches applying normalizers sequentially
        norm1 = normalizer1.normalize(data)
        norm2 = normalizer2.normalize(norm1)
        assert torch.allclose(normalized, norm2, atol=1e-6)

        # Check expected statistics
        expected_mean = (input_mean2 - imagenet_mean) / imagenet_std
        expected_std = input_std2 / imagenet_std
        assert torch.allclose(normalized.mean(), torch.tensor(expected_mean), atol=1e-3)
        assert torch.allclose(normalized.std(), torch.tensor(expected_std), atol=1e-3)


    def test_sequential_invertible(self):
        """Test that sequential normalization is invertible."""
        data = torch.randn(100, 10) * 5 + 3

        # Create two normalizers
        norm1 = SingleFieldLinearNormalizer.create_fit(
            data,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            output_min=0,
            output_max=1
        )
        norm2 = SingleFieldLinearNormalizer.create_fit(
            norm1.normalize(data),
            mode=KinematicsNormalizationType.GAUSSIAN.value
        )

        seq_norm = SequentialNormalizer(normalizers=[norm1, norm2])

        # Test invertibility
        normalized = seq_norm.normalize(data)
        unnormalized = seq_norm.unnormalize(normalized)

        assert torch.allclose(data, unnormalized, atol=1e-5)


    def test_sequential_state_dict_save_load(self):
        """Test that SequentialNormalizer can be saved/loaded via LinearNormalizer."""
        data = torch.randn(100, 10) * 5

        # Create sequential normalizer
        norm1 = SingleFieldLinearNormalizer.create_fit(
            data,
            mode=KinematicsNormalizationType.MIN_MAX.value
        )
        norm2 = SingleFieldLinearNormalizer.create_fit(
            norm1.normalize(data),
            mode=KinematicsNormalizationType.GAUSSIAN.value
        )
        seq_norm = SequentialNormalizer(normalizers=[norm1, norm2])

        # Store in LinearNormalizer
        linear_norm = LinearNormalizer()
        linear_norm['test_field'] = seq_norm

        # Save and load
        state_dict = linear_norm.state_dict()
        new_linear_norm = LinearNormalizer()
        new_linear_norm.load_state_dict(state_dict)

        # Retrieve and test
        loaded_seq_norm = new_linear_norm['test_field']
        assert isinstance(loaded_seq_norm, SequentialNormalizer)

        # Test that it works
        test_data = torch.randn(10, 10) * 5
        result1 = seq_norm.normalize(test_data)
        result2 = loaded_seq_norm.normalize(test_data)
        assert torch.allclose(result1, result2, atol=1e-6)


    def test_sequential_empty_raises(self):
        """Test that creating SequentialNormalizer with empty list raises error."""
        with pytest.raises(ValueError, match="At least one normalizer required"):
            SequentialNormalizer(normalizers=[])


    def test_sequential_fit_raises(self):
        """Test that calling fit on SequentialNormalizer raises error."""
        data = torch.randn(10, 5)
        norm = SingleFieldLinearNormalizer.create_fit(data)
        seq_norm = SequentialNormalizer(normalizers=[norm])

        with pytest.raises(NotImplementedError):
            seq_norm.fit(data)


    def test_sequential_get_input_output_stats(self):
        """Test that get_input_stats and get_output_stats work correctly."""
        data = torch.randn(100, 5) * 10 + 5

        # Create sequential normalizer with two stages
        norm1 = SingleFieldLinearNormalizer.create_fit(
            data,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            output_min=0,
            output_max=1
        )
        norm2 = SingleFieldLinearNormalizer.create_fit(
            norm1.normalize(data),
            mode=KinematicsNormalizationType.GAUSSIAN.value
        )
        seq_norm = SequentialNormalizer(normalizers=[norm1, norm2])

        # Get input stats (should be from first normalizer)
        input_stats = seq_norm.get_input_stats()
        expected_input_stats = norm1.get_input_stats()

        for key in ['min', 'max', 'mean', 'std']:
            assert torch.allclose(input_stats[key], expected_input_stats[key])

        # Get output stats (should be from composition)
        output_stats = seq_norm.get_output_stats()

        # Verify by manual calculation
        test_min = norm1.normalize(expected_input_stats['min'])
        test_min = norm2.normalize(test_min)
        assert torch.allclose(output_stats['min'], test_min, atol=1e-5)


class TestEdgeCases:
    """Test edge cases and error handling."""


    def test_constant_data(self):
        """Test normalization of constant data."""
        data = torch.ones(100, 10) * 5.0

        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(data, mode=KinematicsNormalizationType.MIN_MAX.value)

        # Should handle gracefully (map to middle of output range)
        normalized = normalizer.normalize(data)
        assert torch.all(torch.isfinite(normalized))

        # Should map to 0.0 (middle of [-1, 1])
        assert torch.allclose(normalized, torch.tensor(0.0), atol=1e-5)

        # Should be invertible
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(data, unnormalized, atol=1e-5)


    def test_very_small_range(self):
        """Test data with very small range."""
        data = torch.ones(100, 10) * 5.0
        data += torch.randn(100, 10) * 1e-8  # Add tiny noise

        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(
            data,
            mode=KinematicsNormalizationType.MIN_MAX.value,
            range_eps=1e-6  # Should treat as constant
        )

        normalized = normalizer.normalize(data)
        assert torch.all(torch.isfinite(normalized))

        # Should map to 0.0 (middle of [-1, 1]) since range < eps
        assert torch.allclose(normalized, torch.tensor(0.0), atol=1e-4)


    def test_mixed_constant_and_variable_dims(self):
        """Test data where some dimensions are constant."""
        data = torch.randn(100, 10, 5)
        data[..., 0] = 1.0  # Make first channel constant

        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(data, mode=KinematicsNormalizationType.MIN_MAX.value, last_n_dims=1)

        normalized = normalizer.normalize(data)
        unnormalized = normalizer.unnormalize(normalized)

        assert torch.allclose(data, unnormalized, atol=1e-5)

        # Check that constant channel maps to middle of range
        # With last_n_dims=1, we have 5 channels, first is constant
        assert torch.allclose(normalized[..., 0], torch.tensor(0.0), atol=1e-5)


    def test_gaussian_with_zero_std(self):
        """Test Gaussian normalization with zero std deviation."""
        data = torch.ones(100, 5) * 3.0

        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(data, mode=KinematicsNormalizationType.GAUSSIAN.value)

        normalized = normalizer.normalize(data)

        # Should handle gracefully (scale = 1, offset = 0 for constant data)
        assert torch.all(torch.isfinite(normalized))

        # For Gaussian mode with constant data, should map to 0 (offset = -mean * scale)
        assert torch.allclose(normalized, torch.tensor(0.0), atol=1e-5)


    def test_negative_data_min_max(self):
        """Test min-max normalization with negative data."""
        data = torch.randn(100, 10) * 10 - 5  # Data centered around -5

        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(data, mode=KinematicsNormalizationType.MIN_MAX.value)

        normalized = normalizer.normalize(data)

        # Should still map to [-1, 1]
        assert torch.allclose(normalized.min(), torch.tensor(-1.0), atol=1e-4)
        assert torch.allclose(normalized.max(), torch.tensor(1.0), atol=1e-4)

        # Should be invertible
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(data, unnormalized, atol=1e-5)


    def test_single_sample(self):
        """Test normalization with a single sample."""
        data = torch.tensor([[1.0, 2.0, 3.0]])

        normalizer = SingleFieldLinearNormalizer()
        normalizer.fit(data, mode=KinematicsNormalizationType.MIN_MAX.value, last_n_dims=1)

        normalized = normalizer.normalize(data)

        # Should normalize each channel independently
        assert normalized.shape == data.shape

        # Should be invertible
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(data, unnormalized, atol=1e-5)


    def test_different_dtypes(self):
        """Test normalization with different dtypes."""
        data_float32 = torch.randn(50, 10).float()
        data_float64 = torch.randn(50, 10).double()

        # Test float32
        norm32 = SingleFieldLinearNormalizer()
        norm32.fit(data_float32, dtype=torch.float32)
        normalized32 = norm32.normalize(data_float32)
        assert normalized32.dtype == torch.float32

        # Test float64
        norm64 = SingleFieldLinearNormalizer()
        norm64.fit(data_float64, dtype=torch.float64)
        normalized64 = norm64.normalize(data_float64)
        assert normalized64.dtype == torch.float64