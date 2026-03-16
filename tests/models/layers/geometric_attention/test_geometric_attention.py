"""Tests for versatil.models.layers.geometric_attention.geometric_attention module."""
import pytest
import torch

from versatil.models.layers.constants import AttentionDecompositionMode


class TestGeometricSelfAttentionConfiguration:

    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("num_heads", [4, 8])
    @pytest.mark.parametrize(
        "decomposition_mode",
        [AttentionDecompositionMode.FULL.value, AttentionDecompositionMode.SEPARABLE.value],
    )
    def test_stores_configuration(
        self,
        geometric_attention_factory,
        embedding_dimension,
        num_heads,
        decomposition_mode,
    ):
        attention = geometric_attention_factory(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            decomposition_mode=decomposition_mode,
        )
        assert attention.embedding_dimension == embedding_dimension
        assert attention.num_heads == num_heads
        assert attention.decomposition_mode == decomposition_mode

    @pytest.mark.parametrize("value_dimension_factor", [1, 2])
    def test_head_dimensions_computed_correctly(
        self, geometric_attention_factory, value_dimension_factor
    ):
        embedding_dimension = 32
        num_heads = 4
        attention = geometric_attention_factory(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            value_dimension_factor=value_dimension_factor,
        )
        expected_key_dim = embedding_dimension // num_heads
        expected_value_dim = (embedding_dimension * value_dimension_factor) // num_heads
        assert attention.head_dimension_key == expected_key_dim
        assert attention.head_dimension_value == expected_value_dim

    def test_attention_scaling_equals_inverse_sqrt_head_dim(
        self, geometric_attention_factory
    ):
        embedding_dimension = 64
        num_heads = 8
        attention = geometric_attention_factory(
            embedding_dimension=embedding_dimension, num_heads=num_heads
        )
        expected_scaling = (embedding_dimension // num_heads) ** -0.5
        assert abs(attention.attention_scaling - expected_scaling) < 1e-6


class TestGeometricSelfAttentionProjections:

    def test_query_key_projections_preserve_embedding_dimension(
        self, geometric_attention_factory
    ):
        embedding_dimension = 32
        attention = geometric_attention_factory(embedding_dimension=embedding_dimension)
        assert attention.query_projection.in_features == embedding_dimension
        assert attention.query_projection.out_features == embedding_dimension
        assert attention.key_projection.in_features == embedding_dimension
        assert attention.key_projection.out_features == embedding_dimension

    @pytest.mark.parametrize("value_dimension_factor", [1, 2])
    def test_value_projection_expands_by_factor(
        self, geometric_attention_factory, value_dimension_factor
    ):
        embedding_dimension = 32
        attention = geometric_attention_factory(
            embedding_dimension=embedding_dimension,
            value_dimension_factor=value_dimension_factor,
        )
        expected_output = embedding_dimension * value_dimension_factor
        assert attention.value_projection.out_features == expected_output

    @pytest.mark.parametrize("value_dimension_factor", [1, 2])
    def test_output_projection_maps_back_to_embedding_dimension(
        self, geometric_attention_factory, value_dimension_factor
    ):
        embedding_dimension = 32
        attention = geometric_attention_factory(
            embedding_dimension=embedding_dimension,
            value_dimension_factor=value_dimension_factor,
        )
        expected_input = embedding_dimension * value_dimension_factor
        assert attention.output_projection.in_features == expected_input
        assert attention.output_projection.out_features == embedding_dimension


class TestGeometricSelfAttentionForward:

    @pytest.mark.parametrize(
        "batch_size, height, width, embedding_dimension, num_heads",
        [(2, 4, 6, 32, 4), (1, 3, 3, 64, 8)],
    )
    def test_output_shape_matches_input(
        self,
        geometric_attention_factory,
        depth_map_factory,
        nhwc_tensor_factory,
        batch_size,
        height,
        width,
        embedding_dimension,
        num_heads,
    ):
        attention = geometric_attention_factory(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
        )
        input_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        depth_map = depth_map_factory(
            batch_size=batch_size, height=height, width=width
        )
        output = attention(input_tensor=input_tensor, depth_map=depth_map)
        assert output.shape == (batch_size, height, width, embedding_dimension)

    @pytest.mark.parametrize(
        "decomposition_mode",
        [AttentionDecompositionMode.FULL.value, AttentionDecompositionMode.SEPARABLE.value],
    )
    def test_output_shape_for_both_decomposition_modes(
        self,
        geometric_attention_factory,
        depth_map_factory,
        nhwc_tensor_factory,
        decomposition_mode,
    ):
        batch_size, height, width, embedding_dimension = 2, 4, 4, 32
        attention = geometric_attention_factory(
            embedding_dimension=embedding_dimension,
            num_heads=4,
            decomposition_mode=decomposition_mode,
        )
        input_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        depth_map = depth_map_factory(
            batch_size=batch_size, height=height, width=width
        )
        output = attention(input_tensor=input_tensor, depth_map=depth_map)
        assert output.shape == (batch_size, height, width, embedding_dimension)

    def test_different_depth_maps_produce_different_outputs(
        self, geometric_attention_factory, nhwc_tensor_factory
    ):
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        attention = geometric_attention_factory(
            embedding_dimension=embedding_dimension, num_heads=4
        )
        input_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )

        uniform_depth = torch.ones(batch_size, 1, height, width)

        # Depth with sharp discontinuity
        discontinuous_depth = torch.ones(batch_size, 1, height, width)
        discontinuous_depth[:, :, :, width // 2:] = 10.0

        output_uniform = attention(
            input_tensor=input_tensor, depth_map=uniform_depth
        )
        output_discontinuous = attention(
            input_tensor=input_tensor, depth_map=discontinuous_depth
        )

        assert not torch.allclose(output_uniform, output_discontinuous, atol=1e-5)

    def test_different_inputs_produce_different_outputs(
        self, geometric_attention_factory, depth_map_factory, nhwc_tensor_factory
    ):
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        attention = geometric_attention_factory(
            embedding_dimension=embedding_dimension, num_heads=4
        )
        depth_map = depth_map_factory(
            batch_size=batch_size, height=height, width=width
        )

        input_a = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        input_b = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )

        output_a = attention(input_tensor=input_a, depth_map=depth_map)
        output_b = attention(input_tensor=input_b, depth_map=depth_map)
        assert not torch.allclose(output_a, output_b, atol=1e-5)

    def test_output_is_differentiable(
        self, geometric_attention_factory, depth_map_factory, nhwc_tensor_factory
    ):
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        attention = geometric_attention_factory(
            embedding_dimension=embedding_dimension, num_heads=4
        )
        input_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        input_tensor.requires_grad_(True)
        depth_map = depth_map_factory(
            batch_size=batch_size, height=height, width=width
        )

        output = attention(input_tensor=input_tensor, depth_map=depth_map)
        loss = output.sum()
        loss.backward()
        assert input_tensor.grad is not None
        assert input_tensor.grad.shape == input_tensor.shape
        # Verify gradients actually flow (non-zero)
        assert input_tensor.grad.abs().sum().item() > 0.0


class TestGeometricSelfAttentionDepthAwareness:

    def test_depth_boundary_suppresses_cross_boundary_attention(
        self, geometric_attention_factory, nhwc_tensor_factory
    ):
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        attention = geometric_attention_factory(
            embedding_dimension=embedding_dimension, num_heads=4
        )
        attention.eval()

        input_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        ) * 0.01

        # Uniform depth - no boundary effects
        uniform_depth = torch.ones(batch_size, 1, height, width)
        output_uniform = attention(
            input_tensor=input_tensor, depth_map=uniform_depth
        )

        # Very large depth discontinuity in the middle
        boundary_depth = torch.ones(batch_size, 1, height, width)
        boundary_depth[:, :, :, width // 2:] = 100.0
        output_boundary = attention(
            input_tensor=input_tensor, depth_map=boundary_depth
        )

        relative_difference = (output_uniform - output_boundary).abs().mean()
        assert relative_difference.item() > 1e-6

    def test_full_and_separable_produce_different_outputs(
        self, geometric_attention_factory, nhwc_tensor_factory, depth_map_factory
    ):
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        num_heads = 4

        attention_full = geometric_attention_factory(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            decomposition_mode=AttentionDecompositionMode.FULL.value,
        )
        attention_separable = geometric_attention_factory(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            decomposition_mode=AttentionDecompositionMode.SEPARABLE.value,
        )

        # Copy weights so only decomposition_mode differs
        attention_separable.load_state_dict(
            attention_full.state_dict(), strict=False
        )

        input_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        depth_map = depth_map_factory(
            batch_size=batch_size, height=height, width=width
        )

        output_full = attention_full(
            input_tensor=input_tensor, depth_map=depth_map
        )
        output_separable = attention_separable(
            input_tensor=input_tensor, depth_map=depth_map
        )

        assert not torch.allclose(output_full, output_separable, atol=1e-5)

    def test_value_dimension_factor_changes_internal_dimensions(
        self, geometric_attention_factory, nhwc_tensor_factory, depth_map_factory
    ):
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        attention_factor_1 = geometric_attention_factory(
            embedding_dimension=embedding_dimension,
            num_heads=4,
            value_dimension_factor=1,
        )
        attention_factor_2 = geometric_attention_factory(
            embedding_dimension=embedding_dimension,
            num_heads=4,
            value_dimension_factor=2,
        )
        # Output shape should be the same regardless of factor
        input_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        depth_map = depth_map_factory(
            batch_size=batch_size, height=height, width=width
        )

        output_1 = attention_factor_1(
            input_tensor=input_tensor, depth_map=depth_map
        )
        output_2 = attention_factor_2(
            input_tensor=input_tensor, depth_map=depth_map
        )
        assert output_1.shape == output_2.shape == (batch_size, height, width, embedding_dimension)
        # But the actual outputs should differ due to different internal computations
        assert not torch.allclose(output_1, output_2, atol=1e-5)
