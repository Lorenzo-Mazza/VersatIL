
import pytest
import torch
from torch import nn

from versatil.models.layers.geometric_attention import (
    SpatialDecayMask,
    DepthAwareDecayMask,
    GeometricAttentionBias,
    GeometricSelfAttention,
)
from versatil.models.layers.constants import AttentionDecompositionMode, Axis
from versatil.models.layers.convolution.depthwise_conv2d import DepthwiseConv2D
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding2D



@pytest.fixture
def num_heads():
    return 4


@pytest.fixture
def initial_decay():
    return 5.0


@pytest.fixture
def decay_range():
    return 3.0


@pytest.fixture
def height():
    return 4


@pytest.fixture
def width():
    return 4


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def embedding_dimension():
    return 64


@pytest.fixture
def depth_map(batch_size, height, width):
    return torch.randn(batch_size, 1, height, width)


@pytest.fixture
def input_tensor(batch_size, height, width, embedding_dimension):
    return torch.randn(batch_size, height, width, embedding_dimension)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def decay_rates(num_heads):
    return torch.linspace(0.1, 0.1 * num_heads, num_heads)


@pytest.mark.unit
class TestSpatialDecayMask:

    @pytest.mark.parametrize("num_heads, initial_decay, decay_range, expected", [
        (1, 1.0, 0.0, torch.tensor([-0.6931])),  # log(1 - 2^(-1)) = log(0.5)
        (2, 1.0, 2.0, torch.tensor([-0.6931, -0.2877])),  # log(0.5), log(0.75)
    ])
    def test_compute_per_head_decay(self, num_heads, initial_decay, decay_range, expected):
        result = SpatialDecayMask._compute_per_head_decay(num_heads, initial_decay, decay_range)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)


    @pytest.mark.parametrize("height, width", [(2, 2), (3, 3)])
    def test_compute_2d_distance_matrix(self, height, width):
        module = SpatialDecayMask(num_heads=1)
        result = module.compute_2d_distance_matrix(height, width)
        assert result.shape == (height * width, height * width)
        assert torch.all(result >= 0)
        assert torch.allclose(result, result.T)
        assert torch.all(result.diag() == 0)


    @pytest.mark.parametrize("length", [2, 4])
    def test_compute_1d_distance_matrix(self, length):
        module = SpatialDecayMask(num_heads=1)
        result = module.compute_1d_distance_matrix(length)
        assert result.shape == (length, length)
        assert torch.all(result >= 0)
        assert torch.allclose(result, result.T)
        assert torch.all(result.diag() == 0)


    @pytest.mark.parametrize("decomposition_mode", [
        AttentionDecompositionMode.FULL.value,
        AttentionDecompositionMode.SEPARABLE.value,
    ])
    def test_forward(self, num_heads, initial_decay, decay_range, height, width, decomposition_mode):
        module = SpatialDecayMask(num_heads, initial_decay, decay_range)
        masks = module.forward(height, width, decomposition_mode)
        if decomposition_mode == AttentionDecompositionMode.FULL.value:
            assert len(masks) == 1
            assert masks[0].shape == (num_heads, height * width, height * width)
        else:
            assert len(masks) == 2
            assert masks[0].shape == (num_heads, height, height)
            assert masks[1].shape == (num_heads, width, width)


@pytest.mark.unit
class TestDepthAwareDecayMask:

    @pytest.mark.parametrize("height, width", [(2, 2), (3, 3)])
    def test_compute_depth_difference_matrix(self, batch_size, height, width):
        depth_map = torch.arange(batch_size * 1 * height * width).view(batch_size, 1, height, width).float()
        module = DepthAwareDecayMask(num_heads=1)
        result = module.compute_depth_difference_matrix(depth_map, height, width)
        assert result.shape == (batch_size, height * width, height * width)
        assert torch.all(result >= 0)
        assert torch.allclose(result, result.transpose(1, 2))
        assert torch.all(result.diagonal(dim1=1, dim2=2) == 0)


    @pytest.mark.parametrize("axis", [Axis.HEIGHT.value, Axis.WIDTH.value])
    def test_compute_1d_depth_difference_matrix(self, batch_size, height, width, axis):
        depth_map = torch.arange(batch_size * 1 * height * width).view(batch_size, 1, height, width).float()
        module = DepthAwareDecayMask(num_heads=1)
        result = module.compute_1d_depth_difference_matrix(depth_map, axis)

        if axis == Axis.HEIGHT.value:
            assert result.shape == (batch_size, width, height, height)
        else:
            assert result.shape == (batch_size, height, width, width)
        assert torch.all(result >= 0)


    @pytest.mark.parametrize("decomposition_mode", [
        AttentionDecompositionMode.FULL.value,
        AttentionDecompositionMode.SEPARABLE.value,
    ])
    def test_forward(self, num_heads, height, width, depth_map, decay_rates, decomposition_mode):
        module = DepthAwareDecayMask(num_heads)
        masks = module.forward(depth_map, height, width, decay_rates, decomposition_mode)
        batch_size = depth_map.shape[0]
        if decomposition_mode == AttentionDecompositionMode.FULL.value:
            assert len(masks) == 1
            assert masks[0].shape == (batch_size, num_heads, height * width, height * width)
        else:
            assert len(masks) == 2
            assert masks[0].shape == (batch_size, num_heads, width, height, height)
            assert masks[1].shape == (batch_size, num_heads, height, width, width)


@pytest.mark.unit
class TestGeometricAttentionBias:

    @pytest.mark.parametrize("embedding_dimension, num_heads", [(64, 4), (128, 8)])
    def test_init(self, embedding_dimension, num_heads, initial_decay, decay_range):
        module = GeometricAttentionBias(embedding_dimension, num_heads, initial_decay, decay_range)
        assert module.embedding_dimension == embedding_dimension
        assert module.num_heads == num_heads
        assert isinstance(module.rotary_encoding, RotaryPositionalEncoding2D)
        assert isinstance(module.spatial_decay, SpatialDecayMask)
        assert isinstance(module.depth_decay, DepthAwareDecayMask)
        assert module.bias_weights.shape == (2, 1, 1, 1)


    @pytest.mark.parametrize("decomposition_mode", [
        AttentionDecompositionMode.FULL.value,
        AttentionDecompositionMode.SEPARABLE.value,
    ])
    def test_forward(self, embedding_dimension, num_heads, height, width, depth_map, device, decomposition_mode):
        module = GeometricAttentionBias(embedding_dimension, num_heads)
        (sine, cosine), bias_masks = module.forward(height, width, depth_map, device, decomposition_mode)

        head_dim = embedding_dimension // num_heads
        assert sine.shape == (height, width, head_dim)
        assert cosine.shape == (height, width, head_dim)

        batch_size = depth_map.shape[0]
        if decomposition_mode == AttentionDecompositionMode.FULL.value:
            assert len(bias_masks) == 1
            assert bias_masks[0].shape == (batch_size, num_heads, height * width, height * width)
        else:
            assert len(bias_masks) == 2
            assert bias_masks[0].shape == (batch_size, num_heads, width, height, height)
            assert bias_masks[1].shape == (batch_size, num_heads, height, width, width)


@pytest.mark.unit
class TestGeometricSelfAttention:

    @pytest.mark.parametrize("embedding_dimension, num_heads, value_dimension_factor", [
        (64, 4, 1),
        (128, 8, 2),
    ])
    @pytest.mark.parametrize("decomposition_mode", [
        AttentionDecompositionMode.FULL.value,
        AttentionDecompositionMode.SEPARABLE.value,
    ])
    def test_init(self, embedding_dimension, num_heads, value_dimension_factor, decomposition_mode, initial_decay, decay_range):
        module = GeometricSelfAttention(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            value_dimension_factor=value_dimension_factor,
            decomposition_mode=decomposition_mode,
            initial_decay=initial_decay,
            decay_range=decay_range,
        )

        assert module.embedding_dimension == embedding_dimension
        assert module.num_heads == num_heads
        assert module.value_dimension_factor == value_dimension_factor
        assert module.decomposition_mode == decomposition_mode
        assert module.head_dimension_key == embedding_dimension // num_heads
        assert module.head_dimension_value == (embedding_dimension * value_dimension_factor) // num_heads
        assert isinstance(module.query_projection, nn.Linear)
        assert isinstance(module.key_projection, nn.Linear)
        assert isinstance(module.value_projection, nn.Linear)
        assert isinstance(module.learned_positional_encodings, DepthwiseConv2D)
        assert isinstance(module.output_projection, nn.Linear)
        assert isinstance(module.geometric_bias, GeometricAttentionBias)


    def test_forward_full(self, input_tensor, depth_map, embedding_dimension, num_heads):
        module = GeometricSelfAttention(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            decomposition_mode=AttentionDecompositionMode.FULL.value,
        )
        output = module.forward(input_tensor, depth_map)
        assert output.shape == input_tensor.shape


    def test_forward_separable(self, input_tensor, depth_map, embedding_dimension, num_heads):
        module = GeometricSelfAttention(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )
        output = module.forward(input_tensor, depth_map)
        assert output.shape == input_tensor.shape


    @pytest.mark.parametrize("batch_size, height, width, head_dim", [(2, 4, 4, 16)])
    def test_compute_attention_full(self, batch_size, height, width, head_dim, embedding_dimension, num_heads):
        module = GeometricSelfAttention(embedding_dimension, num_heads)
        query = torch.ones(batch_size, num_heads, height, width, head_dim)
        key = torch.ones(batch_size, num_heads, height, width, head_dim)
        value = torch.ones(batch_size, num_heads, height, width, head_dim)
        sine = torch.zeros(height, width, head_dim)
        cosine = torch.ones(height, width, head_dim)
        attention_bias = torch.zeros(batch_size, num_heads, height * width, height * width)
        output = module._compute_attention_full(query, key, value, sine, cosine, attention_bias)
        assert output.shape == (batch_size, height, width, num_heads * head_dim)


    @pytest.mark.parametrize("batch_size, height, width, head_dim", [(2, 4, 4, 16)])
    def test_compute_attention_separable(self, batch_size, height, width, head_dim, embedding_dimension, num_heads):
        module = GeometricSelfAttention(embedding_dimension, num_heads)
        query = torch.ones(batch_size, num_heads, height, width, head_dim)
        key = torch.ones(batch_size, num_heads, height, width, head_dim)
        value = torch.ones(batch_size, num_heads, height, width, head_dim)
        sine = torch.zeros(height, width, head_dim)
        cosine = torch.ones(height, width, head_dim)
        height_bias = torch.zeros(batch_size, num_heads, width, height, height)
        width_bias = torch.zeros(batch_size, num_heads, height, width, width)

        output = module._compute_attention_separable(query, key, value, sine, cosine, height_bias, width_bias)
        assert output.shape == (batch_size, height, width, num_heads * head_dim)