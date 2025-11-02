import pytest
import torch

from refactoring.models.encoding.encoders.constants import PoolingMethod
from refactoring.models.layers.pooling.pooling_head import (
    PoolingHead,
    SpatialSoftmaxPooling,
    GlobalAveragePooling,
    IdentityPooling,
    create_pooling_head,
)


class TestPoolingHeads:
    """Test suite for pooling head implementations."""


    @pytest.fixture
    def feature_dimensions(self):
        """Standard feature dimensions for testing."""
        return {
            'channels': 512,
            'height': 7,
            'width': 7,
        }


    @pytest.fixture
    def sample_features(self, feature_dimensions):
        """Create sample feature tensor."""
        batch_size = 4
        return torch.randn(
            batch_size,
            feature_dimensions['channels'],
            feature_dimensions['height'],
            feature_dimensions['width']
        )


    def test_spatial_softmax_pooling_output_shape(self, feature_dimensions, sample_features):
        """Test spatial softmax pooling output shape."""
        pooling = SpatialSoftmaxPooling(
            feature_dimensions['height'],
            feature_dimensions['width'],
            feature_dimensions['channels']
        )

        output = pooling(sample_features)
        expected_dim = feature_dimensions['channels'] * 2

        assert output.shape == (sample_features.shape[0], expected_dim)
        assert pooling.get_output_dim(feature_dimensions['channels']) == expected_dim


    def test_global_average_pooling_output_shape(self, feature_dimensions, sample_features):
        """Test global average pooling output shape."""
        pooling = GlobalAveragePooling()

        output = pooling(sample_features)

        assert output.shape == (sample_features.shape[0], feature_dimensions['channels'])
        assert pooling.get_output_dim(feature_dimensions['channels']) == feature_dimensions['channels']


    def test_global_average_pooling_correctness(self, feature_dimensions, sample_features):
        """Test global average pooling computes correct values."""
        pooling = GlobalAveragePooling()

        output = pooling(sample_features)
        expected = sample_features.mean(dim=[2, 3])

        assert torch.allclose(output, expected)


    def test_identity_pooling_output_shape(self, feature_dimensions, sample_features):
        """Test identity pooling preserves input shape."""
        pooling = IdentityPooling(
            feature_dimensions['height'],
            feature_dimensions['width'],
            feature_dimensions['channels']
        )

        output = pooling(sample_features)
        expected_dim = (
            feature_dimensions['channels'],
            feature_dimensions['height'],
            feature_dimensions['width']
        )

        assert output.shape == sample_features.shape
        assert pooling.get_output_dim(feature_dimensions['channels']) == expected_dim


    def test_identity_pooling_preserves_values(self, feature_dimensions, sample_features):
        """Test identity pooling doesn't modify values."""
        pooling = IdentityPooling(
            feature_dimensions['height'],
            feature_dimensions['width'],
            feature_dimensions['channels']
        )

        output = pooling(sample_features)

        assert torch.allclose(output, sample_features)


    @pytest.mark.parametrize("pooling_method,expected_class", [
        (PoolingMethod.SPATIAL_SOFTMAX.value, SpatialSoftmaxPooling),
        (PoolingMethod.GLOBAL_AVERAGE.value, GlobalAveragePooling),
        (PoolingMethod.NONE.value, IdentityPooling),
    ])
    def test_factory_creates_correct_type(
            self, pooling_method, expected_class, feature_dimensions
    ):
        """Test factory function creates correct pooling head types."""
        pooling = create_pooling_head(
            pooling_method=pooling_method,
            feature_channels=feature_dimensions['channels'],
            spatial_height=feature_dimensions['height'],
            spatial_width=feature_dimensions['width'],
        )

        assert isinstance(pooling, expected_class)


    def test_factory_invalid_method_raises_error(self, feature_dimensions):
        """Test factory raises error for invalid pooling method."""
        with pytest.raises(ValueError, match="Unsupported pooling method"):
            create_pooling_head(
                pooling_method="invalid_method",
                feature_channels=feature_dimensions['channels'],
                spatial_height=feature_dimensions['height'],
                spatial_width=feature_dimensions['width'],
            )


    def test_all_pooling_heads_are_modules(self, feature_dimensions):
        """Test all pooling heads are proper nn.Modules."""
        pooling_classes = [
            SpatialSoftmaxPooling(7, 7, 512),
            GlobalAveragePooling(),
            IdentityPooling(7, 7, 512),
        ]

        for pooling in pooling_classes:
            assert isinstance(pooling, torch.nn.Module)


    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    def test_batch_size_independence(self, batch_size, feature_dimensions):
        """Test pooling works with different batch sizes."""
        pooling = GlobalAveragePooling()

        features = torch.randn(
            batch_size,
            feature_dimensions['channels'],
            feature_dimensions['height'],
            feature_dimensions['width']
        )
        output = pooling(features)

        assert output.shape[0] == batch_size
        assert output.shape[1] == feature_dimensions['channels']


    @pytest.mark.parametrize("channels,height,width", [
        (256, 14, 14),
        (512, 7, 7),
        (1024, 4, 4),
        (128, 28, 28),
    ])
    def test_different_spatial_dimensions(self, channels, height, width):
        """Test pooling with various spatial dimensions."""
        pooling = create_pooling_head(
            pooling_method=PoolingMethod.SPATIAL_SOFTMAX.value,
            feature_channels=channels,
            spatial_height=height,
            spatial_width=width,
        )

        features = torch.randn(2, channels, height, width)
        output = pooling(features)

        assert output.shape == (2, channels * 2)


    @pytest.mark.parametrize("pooling_method", [
        PoolingMethod.SPATIAL_SOFTMAX.value,
        PoolingMethod.GLOBAL_AVERAGE.value,
        PoolingMethod.NONE.value,
    ])
    def test_gradient_flow(self, pooling_method, feature_dimensions):
        """Test gradients flow through all pooling methods."""
        pooling = create_pooling_head(
            pooling_method=pooling_method,
            feature_channels=feature_dimensions['channels'],
            spatial_height=feature_dimensions['height'],
            spatial_width=feature_dimensions['width'],
        )

        features = torch.randn(
            2,
            feature_dimensions['channels'],
            feature_dimensions['height'],
            feature_dimensions['width'],
            requires_grad=True
        )

        output = pooling(features)
        loss = output.sum()
        loss.backward()

        assert features.grad is not None
        assert not torch.allclose(features.grad, torch.zeros_like(features.grad))


    def test_pooling_head_is_abstract(self):
        """Test that PoolingHead cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PoolingHead()


    def test_output_dim_consistency(self, feature_dimensions):
        """Test output_dim matches actual output shape."""
        pooling_configs = [
            (PoolingMethod.SPATIAL_SOFTMAX.value, lambda c: c * 2),
            (PoolingMethod.GLOBAL_AVERAGE.value, lambda c: c),
            (PoolingMethod.NONE.value, lambda c: (c, 7, 7)),
        ]

        for method, expected_dim_fn in pooling_configs:
            pooling = create_pooling_head(
                pooling_method=method,
                feature_channels=feature_dimensions['channels'],
                spatial_height=feature_dimensions['height'],
                spatial_width=feature_dimensions['width'],
            )

            features = torch.randn(
                4,
                feature_dimensions['channels'],
                feature_dimensions['height'],
                feature_dimensions['width']
            )
            output = pooling(features)

            reported_dim = pooling.get_output_dim(feature_dimensions['channels'])
            expected_dim = expected_dim_fn(feature_dimensions['channels'])

            assert reported_dim == expected_dim

            if isinstance(reported_dim, int):
                assert output.shape[1] == reported_dim
            else:
                assert output.shape[1:] == reported_dim


    def test_device_compatibility(self, feature_dimensions):
        """Test pooling works on different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        pooling = GlobalAveragePooling().cuda()
        features = torch.randn(
            2,
            feature_dimensions['channels'],
            feature_dimensions['height'],
            feature_dimensions['width']
        ).cuda()

        output = pooling(features)

        assert output.device.type == 'cuda'
        assert output.shape == (2, feature_dimensions['channels'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])