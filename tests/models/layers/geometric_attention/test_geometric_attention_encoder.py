"""Tests for versatil.models.layers.geometric_attention.geometric_attention_encoder module."""

import pytest
import torch

from versatil.models.layers.constants import AttentionDecompositionMode


class TestEncoderBlockConfiguration:
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("use_layer_scale", [True, False])
    @pytest.mark.parametrize(
        "decomposition_mode",
        [AttentionDecompositionMode.FULL, AttentionDecompositionMode.SEPARABLE],
    )
    def test_stores_configuration(
        self,
        encoder_block_factory,
        embedding_dimension,
        use_layer_scale,
        decomposition_mode,
    ):
        block = encoder_block_factory(
            embedding_dimension=embedding_dimension,
            use_layer_scale=use_layer_scale,
            decomposition_mode=decomposition_mode,
        )
        assert block.embedding_dimension == embedding_dimension
        assert block.use_layer_scale == use_layer_scale

    def test_layer_scale_parameters_initialized_correctly(self, encoder_block_factory):
        layer_scale_init_value = 1e-5
        embedding_dimension = 32
        block = encoder_block_factory(
            use_layer_scale=True,
            layer_scale_init_value=layer_scale_init_value,
            embedding_dimension=embedding_dimension,
        )
        assert block.gamma1.shape == (1, 1, 1, embedding_dimension)
        assert block.gamma2.shape == (1, 1, 1, embedding_dimension)
        assert torch.allclose(
            block.gamma1,
            layer_scale_init_value * torch.ones(1, 1, 1, embedding_dimension),
        )
        assert torch.allclose(
            block.gamma2,
            layer_scale_init_value * torch.ones(1, 1, 1, embedding_dimension),
        )
        # Verify they are learnable parameters
        assert block.gamma1.requires_grad is True
        assert block.gamma2.requires_grad is True

    def test_layer_scale_disabled_removes_gamma_parameters(
        self, encoder_block_factory, nhwc_tensor_factory, depth_map_factory
    ):
        # When layer_scale is disabled, forward should work without gamma parameters
        block = encoder_block_factory(use_layer_scale=False, embedding_dimension=32)
        # Verify gamma1/gamma2 are not in the module's parameters
        param_names = [name for name, _ in block.named_parameters()]
        assert "gamma1" not in param_names
        assert "gamma2" not in param_names


class TestEncoderBlockForward:
    @pytest.mark.parametrize(
        "batch_size, height, width, embedding_dimension, number_of_heads",
        [(2, 4, 6, 32, 4), (1, 8, 8, 64, 8)],
    )
    def test_output_shape_matches_input(
        self,
        encoder_block_factory,
        nhwc_tensor_factory,
        depth_map_factory,
        batch_size,
        height,
        width,
        embedding_dimension,
        number_of_heads,
    ):
        block = encoder_block_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
        )
        rgb_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        depth_map = depth_map_factory(batch_size=batch_size, height=height, width=width)
        output = block(rgb_tensor=rgb_tensor, depth_map=depth_map)
        assert output.shape == (batch_size, height, width, embedding_dimension)

    @pytest.mark.parametrize(
        "decomposition_mode",
        [AttentionDecompositionMode.FULL, AttentionDecompositionMode.SEPARABLE],
    )
    def test_forward_works_for_both_decomposition_modes(
        self,
        encoder_block_factory,
        nhwc_tensor_factory,
        depth_map_factory,
        decomposition_mode,
    ):
        batch_size, height, width, embedding_dimension = 2, 4, 4, 32
        block = encoder_block_factory(
            decomposition_mode=decomposition_mode,
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
        )
        rgb_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        depth_map = depth_map_factory(batch_size=batch_size, height=height, width=width)
        output = block(rgb_tensor=rgb_tensor, depth_map=depth_map)
        assert output.shape == (batch_size, height, width, embedding_dimension)

    def test_output_is_differentiable(
        self, encoder_block_factory, nhwc_tensor_factory, depth_map_factory
    ):
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        block = encoder_block_factory(embedding_dimension=embedding_dimension)
        rgb_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        rgb_tensor.requires_grad_(True)
        depth_map = depth_map_factory(batch_size=batch_size, height=height, width=width)

        output = block(rgb_tensor=rgb_tensor, depth_map=depth_map)
        loss = output.sum()
        loss.backward()
        assert rgb_tensor.grad is not None
        assert rgb_tensor.grad.shape == rgb_tensor.shape
        # Verify gradients actually flow (non-zero)
        assert rgb_tensor.grad.abs().sum().item() > 0.0


class TestEncoderBlockResidualConnection:
    def test_residual_connection_preserves_input_scale(
        self, encoder_block_factory, nhwc_tensor_factory, depth_map_factory
    ):
        # With layer_scale init at very small value, the residual should dominate
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        block = encoder_block_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            use_layer_scale=True,
            layer_scale_init_value=1e-10,
            drop_path_rate=0.0,
        )
        block.eval()

        rgb_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        depth_map = depth_map_factory(batch_size=batch_size, height=height, width=width)

        output = block(rgb_tensor=rgb_tensor, depth_map=depth_map)

        # With near-zero layer scale, output should be close to
        # input + input_positional_encoding(input)
        input_with_position = rgb_tensor + block.input_positional_encoding(rgb_tensor)
        relative_error = (
            output - input_with_position
        ).abs().mean() / input_with_position.abs().mean()
        assert relative_error.item() < 0.01


class TestEncoderBlockLayerScale:
    def test_layer_scale_modulates_attention_contribution(
        self, encoder_block_factory, nhwc_tensor_factory, depth_map_factory
    ):
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        block_with_scale = encoder_block_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            use_layer_scale=True,
            layer_scale_init_value=1e-5,
        )
        block_without_scale = encoder_block_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            use_layer_scale=False,
        )

        # Copy weights to ensure only layer_scale differs
        state_dict = block_with_scale.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key not in ("gamma1", "gamma2")
        }
        block_without_scale.load_state_dict(compatible_state)

        block_with_scale.eval()
        block_without_scale.eval()

        rgb_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        depth_map = depth_map_factory(batch_size=batch_size, height=height, width=width)

        output_with_scale = block_with_scale(rgb_tensor=rgb_tensor, depth_map=depth_map)
        output_without_scale = block_without_scale(
            rgb_tensor=rgb_tensor, depth_map=depth_map
        )

        # With tiny layer_scale, the attention/mlp contributions are nearly zero
        # so output is closer to the residual. Without layer_scale, contributions are full.
        assert not torch.allclose(output_with_scale, output_without_scale, atol=1e-4)

    def test_large_layer_scale_approaches_no_scale_behavior(
        self, encoder_block_factory, nhwc_tensor_factory, depth_map_factory
    ):
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        block_with_ones_scale = encoder_block_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            use_layer_scale=True,
            layer_scale_init_value=1.0,
        )
        block_without_scale = encoder_block_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            use_layer_scale=False,
        )

        # Copy weights
        state_dict = block_with_ones_scale.state_dict()
        compatible_state = {
            key: value
            for key, value in state_dict.items()
            if key not in ("gamma1", "gamma2")
        }
        block_without_scale.load_state_dict(compatible_state)

        block_with_ones_scale.eval()
        block_without_scale.eval()

        rgb_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        depth_map = depth_map_factory(batch_size=batch_size, height=height, width=width)

        output_with_scale = block_with_ones_scale(
            rgb_tensor=rgb_tensor, depth_map=depth_map
        )
        output_without_scale = block_without_scale(
            rgb_tensor=rgb_tensor, depth_map=depth_map
        )

        # With layer_scale=1.0, behavior should be identical to no scale
        assert torch.allclose(output_with_scale, output_without_scale, atol=1e-5)


class TestEncoderBlockDropPath:
    def test_drop_path_is_identity_in_eval_mode(
        self, encoder_block_factory, nhwc_tensor_factory, depth_map_factory
    ):
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        block = encoder_block_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            drop_path_rate=0.5,
        )
        block.eval()

        rgb_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        depth_map = depth_map_factory(batch_size=batch_size, height=height, width=width)

        # In eval mode, drop_path should be deterministic (no dropout)
        output_first = block(rgb_tensor=rgb_tensor, depth_map=depth_map)
        output_second = block(rgb_tensor=rgb_tensor, depth_map=depth_map)
        assert torch.allclose(output_first, output_second)

    def test_drop_path_zero_rate_is_identity(
        self, encoder_block_factory, nhwc_tensor_factory, depth_map_factory
    ):
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        block = encoder_block_factory(
            embedding_dimension=embedding_dimension,
            number_of_heads=4,
            drop_path_rate=0.0,
        )
        block.train()

        rgb_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )
        depth_map = depth_map_factory(batch_size=batch_size, height=height, width=width)

        # With drop_path_rate=0.0, training mode should be deterministic
        output_first = block(rgb_tensor=rgb_tensor, depth_map=depth_map)
        output_second = block(rgb_tensor=rgb_tensor, depth_map=depth_map)
        assert torch.allclose(output_first, output_second)


class TestEncoderBlockDepthConditioning:
    def test_different_depth_maps_produce_different_outputs(
        self, encoder_block_factory, nhwc_tensor_factory
    ):
        batch_size, height, width, embedding_dimension = 1, 4, 4, 32
        block = encoder_block_factory(
            embedding_dimension=embedding_dimension, number_of_heads=4
        )
        block.eval()

        rgb_tensor = nhwc_tensor_factory(
            batch_size=batch_size,
            height=height,
            width=width,
            channels=embedding_dimension,
        )

        uniform_depth = torch.ones(batch_size, 1, height, width)
        boundary_depth = torch.ones(batch_size, 1, height, width)
        boundary_depth[:, :, :, width // 2 :] = 50.0

        output_uniform = block(rgb_tensor=rgb_tensor, depth_map=uniform_depth)
        output_boundary = block(rgb_tensor=rgb_tensor, depth_map=boundary_depth)

        assert not torch.allclose(output_uniform, output_boundary, atol=1e-5)
