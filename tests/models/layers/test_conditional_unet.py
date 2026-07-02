"""Tests for versatil.models.layers.conditional_unet module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.conditional_unet import ConditionalUnet1D
from versatil.models.layers.modulation.conditional_modulation import (
    ConditionalModulation,
)


def _activate_modulation_weights(module: ConditionalUnet1D) -> None:
    for child in module.modules():
        if isinstance(child, ConditionalModulation):
            for layer in child.projection.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)


@pytest.fixture
def unet_factory() -> Callable[..., ConditionalUnet1D]:
    def factory(
        input_dimension: int = 8,
        local_conditioning_dimension: int | None = None,
        global_conditioning_dimension: int | None = None,
        diffusion_step_embedding_dimension: int = 32,
        down_dimensions: list[int] | None = None,
        kernel_size: int = 3,
        num_groups: int = 8,
        condition_predict_scale: bool = False,
        initializer_range: float = 0.02,
    ) -> ConditionalUnet1D:
        if down_dimensions is None:
            down_dimensions = [32, 64]
        return ConditionalUnet1D(
            input_dimension=input_dimension,
            local_conditioning_dimension=local_conditioning_dimension,
            global_conditioning_dimension=global_conditioning_dimension,
            diffusion_step_embedding_dimension=diffusion_step_embedding_dimension,
            down_dimensions=down_dimensions,
            kernel_size=kernel_size,
            num_groups=num_groups,
            condition_predict_scale=condition_predict_scale,
            initializer_range=initializer_range,
        )

    return factory


@pytest.fixture
def local_conditioning_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    def factory(
        batch_size: int = 2,
        sequence_length: int = 16,
        local_conditioning_dimension: int = 16,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, sequence_length, local_conditioning_dimension)
        ).astype(np.float32)
        return torch.from_numpy(data)

    return factory


class TestConditionalUnet1DInitialization:
    @pytest.mark.parametrize("input_dimension", [8, 16])
    @pytest.mark.parametrize("diffusion_step_embedding_dimension", [32, 64])
    @pytest.mark.parametrize("initializer_range", [0.02, 0.05])
    def test_stores_configuration(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        input_dimension: int,
        diffusion_step_embedding_dimension: int,
        initializer_range: float,
    ):
        module = unet_factory(
            input_dimension=input_dimension,
            diffusion_step_embedding_dimension=diffusion_step_embedding_dimension,
            initializer_range=initializer_range,
        )
        assert module.initializer_range == initializer_range
        assert len(list(module.diffusion_step_encoder.parameters())) > 0
        assert len(module.downsampling_modules) > 0
        assert len(module.upsampling_modules) > 0
        assert len(module.middle_modules) > 0
        assert len(list(module.final_convolution.parameters())) > 0

    def test_local_condition_encoder_created_when_dimension_provided(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
    ):
        module = unet_factory(local_conditioning_dimension=16)
        assert module.local_condition_encoder is not None
        assert len(module.local_condition_encoder) == 2

    def test_local_condition_encoder_absent_when_dimension_not_provided(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
    ):
        module = unet_factory(local_conditioning_dimension=None)
        assert module.local_condition_encoder is None

    @pytest.mark.parametrize(
        "down_dimensions",
        [
            [32, 64],
            [32, 64, 128],
        ],
    )
    def test_downsampling_module_count_matches_down_dimensions(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        down_dimensions: list[int],
    ):
        module = unet_factory(down_dimensions=down_dimensions)
        assert len(module.downsampling_modules) == len(down_dimensions)

    @pytest.mark.parametrize(
        "down_dimensions",
        [
            [32, 64],
            [32, 64, 128],
        ],
    )
    def test_upsampling_module_count_matches_down_dimensions(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        down_dimensions: list[int],
    ):
        module = unet_factory(down_dimensions=down_dimensions)
        expected_upsampling_count = len(down_dimensions) - 1
        assert len(module.upsampling_modules) == expected_upsampling_count

    def test_last_downsample_is_identity(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
    ):
        module = unet_factory(down_dimensions=[32, 64])
        last_downsample_group = module.downsampling_modules[-1]
        downsample_layer = last_downsample_group[2]
        assert type(downsample_layer) is nn.Identity

    def test_non_last_downsample_performs_spatial_reduction(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        conv1d_tensor_factory: Callable[..., torch.Tensor],
    ):
        module = unet_factory(down_dimensions=[32, 64])
        first_downsample_group = module.downsampling_modules[0]
        downsample_layer = first_downsample_group[2]
        test_input = conv1d_tensor_factory(
            batch_size=1,
            channels=32,
            sequence_length=16,
        )
        with torch.no_grad():
            output = downsample_layer(test_input)
        assert output.shape[2] == 8

    def test_middle_modules_count(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
    ):
        module = unet_factory(down_dimensions=[32, 64])
        assert len(module.middle_modules) == 2

    def test_weight_initialization_linear_layers_have_zero_bias(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
    ):
        initializer_range = 0.02
        module = unet_factory(initializer_range=initializer_range)
        for child_module in module.modules():
            if isinstance(child_module, nn.Linear):
                if (
                    hasattr(child_module, "_is_modulation_layer")
                    and child_module._is_modulation_layer
                ):
                    continue
                if child_module.bias is not None:
                    assert torch.all(child_module.bias == 0)

    def test_weight_initialization_modulation_layers_are_zero(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
    ):
        module = unet_factory()
        for child_module in module.modules():
            if isinstance(child_module, ConditionalModulation):
                for layer in child_module.projection.modules():
                    if isinstance(layer, nn.Linear):
                        assert torch.all(layer.weight == 0)
                        if layer.bias is not None:
                            assert torch.all(layer.bias == 0)

    def test_weight_initialization_group_norm_layers(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
    ):
        module = unet_factory()
        for child_module in module.modules():
            if isinstance(child_module, nn.GroupNorm):
                assert torch.all(child_module.weight == 1.0)
                assert torch.all(child_module.bias == 0.0)


class TestConditionalUnet1DForward:
    @pytest.mark.parametrize("input_dimension", [8, 16])
    @pytest.mark.parametrize("sequence_length", [16, 32])
    def test_output_shape_without_conditioning(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        timestep_factory: Callable[..., torch.Tensor],
        batch_size: int,
        input_dimension: int,
        sequence_length: int,
    ):
        module = unet_factory(input_dimension=input_dimension)
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = timestep_factory(batch_size=batch_size)
        with torch.no_grad():
            output = module(
                noisy_input=noisy_input,
                timesteps=timesteps,
            )
        assert output.shape == (batch_size, sequence_length, input_dimension)

    def test_output_shape_with_global_conditioning(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        timestep_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        input_dimension = 8
        sequence_length = 16
        global_conditioning_dimension = 16
        module = unet_factory(
            input_dimension=input_dimension,
            global_conditioning_dimension=global_conditioning_dimension,
        )
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = timestep_factory(batch_size=batch_size)
        global_conditioning = condition_factory(
            batch_size=batch_size,
            condition_dim=global_conditioning_dimension,
        )
        with torch.no_grad():
            output = module(
                noisy_input=noisy_input,
                timesteps=timesteps,
                global_conditioning=global_conditioning,
            )
        assert output.shape == (batch_size, sequence_length, input_dimension)

    @pytest.mark.parametrize("down_dimensions", [[32], [32, 64], [32, 64, 128]])
    def test_output_shape_with_local_conditioning(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        timestep_factory: Callable[..., torch.Tensor],
        local_conditioning_factory: Callable[..., torch.Tensor],
        batch_size: int,
        down_dimensions: list[int],
    ):
        # Regression: multi-level UNets used to crash adding the
        # full-resolution up-path local conditioning inside the up loop.
        input_dimension = 8
        sequence_length = 16
        local_conditioning_dimension = 16
        module = unet_factory(
            input_dimension=input_dimension,
            local_conditioning_dimension=local_conditioning_dimension,
            down_dimensions=down_dimensions,
        )
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = timestep_factory(batch_size=batch_size)
        local_conditioning = local_conditioning_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            local_conditioning_dimension=local_conditioning_dimension,
        )
        with torch.no_grad():
            output = module(
                noisy_input=noisy_input,
                timesteps=timesteps,
                local_conditioning=local_conditioning,
            )
        assert output.shape == (batch_size, sequence_length, input_dimension)

    def test_output_shape_with_both_conditionings(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        timestep_factory: Callable[..., torch.Tensor],
        local_conditioning_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        input_dimension = 8
        sequence_length = 16
        local_conditioning_dimension = 16
        global_conditioning_dimension = 16
        module = unet_factory(
            input_dimension=input_dimension,
            local_conditioning_dimension=local_conditioning_dimension,
            global_conditioning_dimension=global_conditioning_dimension,
            down_dimensions=[32],
        )
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = timestep_factory(batch_size=batch_size)
        local_conditioning = local_conditioning_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            local_conditioning_dimension=local_conditioning_dimension,
        )
        global_conditioning = condition_factory(
            batch_size=batch_size,
            condition_dim=global_conditioning_dimension,
        )
        with torch.no_grad():
            output = module(
                noisy_input=noisy_input,
                timesteps=timesteps,
                local_conditioning=local_conditioning,
                global_conditioning=global_conditioning,
            )
        assert output.shape == (batch_size, sequence_length, input_dimension)

    def test_scalar_timestep_input(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        input_dimension = 8
        sequence_length = 16
        module = unet_factory(input_dimension=input_dimension)
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        scalar_timestep = 5
        with torch.no_grad():
            output = module(
                noisy_input=noisy_input,
                timesteps=scalar_timestep,
            )
        assert output.shape == (batch_size, sequence_length, input_dimension)

    def test_zero_dim_tensor_timestep_input(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        input_dimension = 8
        sequence_length = 16
        module = unet_factory(input_dimension=input_dimension)
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        zero_dim_timestep = torch.tensor(5)
        with torch.no_grad():
            output = module(
                noisy_input=noisy_input,
                timesteps=zero_dim_timestep,
            )
        assert output.shape == (batch_size, sequence_length, input_dimension)


class TestConditionalUnet1DConditioning:
    def test_modulation_is_identity_at_init(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        # With identity-initialized modulation, different timesteps produce the same output
        input_dimension = 8
        sequence_length = 16
        module = unet_factory(input_dimension=input_dimension)
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps_a = torch.zeros(batch_size, dtype=torch.long)
        timesteps_b = torch.full((batch_size,), fill_value=50, dtype=torch.long)
        with torch.no_grad():
            output_a = module(noisy_input=noisy_input, timesteps=timesteps_a)
            output_b = module(noisy_input=noisy_input, timesteps=timesteps_b)
        assert torch.allclose(output_a, output_b)

    def test_different_timesteps_produce_different_outputs_after_activation(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        input_dimension = 8
        sequence_length = 16
        module = unet_factory(input_dimension=input_dimension)
        _activate_modulation_weights(module)
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps_a = torch.zeros(batch_size, dtype=torch.long)
        timesteps_b = torch.full((batch_size,), fill_value=50, dtype=torch.long)
        with torch.no_grad():
            output_a = module(noisy_input=noisy_input, timesteps=timesteps_a)
            output_b = module(noisy_input=noisy_input, timesteps=timesteps_b)
        assert not torch.allclose(output_a, output_b)

    def test_different_global_conditions_produce_different_outputs(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        timestep_factory: Callable[..., torch.Tensor],
        condition_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        input_dimension = 8
        sequence_length = 16
        global_conditioning_dimension = 16
        module = unet_factory(
            input_dimension=input_dimension,
            global_conditioning_dimension=global_conditioning_dimension,
        )
        _activate_modulation_weights(module)
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = timestep_factory(batch_size=batch_size)
        global_conditioning_a = condition_factory(
            batch_size=batch_size,
            condition_dim=global_conditioning_dimension,
        )
        global_conditioning_b = condition_factory(
            batch_size=batch_size,
            condition_dim=global_conditioning_dimension,
        )
        with torch.no_grad():
            output_a = module(
                noisy_input=noisy_input,
                timesteps=timesteps,
                global_conditioning=global_conditioning_a,
            )
            output_b = module(
                noisy_input=noisy_input,
                timesteps=timesteps,
                global_conditioning=global_conditioning_b,
            )
        assert not torch.allclose(output_a, output_b)

    def test_different_local_conditions_produce_different_outputs(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        timestep_factory: Callable[..., torch.Tensor],
        local_conditioning_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        # Single down_dimension avoids sequence mismatch for local conditioning
        input_dimension = 8
        sequence_length = 16
        local_conditioning_dimension = 16
        module = unet_factory(
            input_dimension=input_dimension,
            local_conditioning_dimension=local_conditioning_dimension,
            down_dimensions=[32],
        )
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = timestep_factory(batch_size=batch_size)
        local_conditioning_a = local_conditioning_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            local_conditioning_dimension=local_conditioning_dimension,
        )
        local_conditioning_b = local_conditioning_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            local_conditioning_dimension=local_conditioning_dimension,
        )
        with torch.no_grad():
            output_a = module(
                noisy_input=noisy_input,
                timesteps=timesteps,
                local_conditioning=local_conditioning_a,
            )
            output_b = module(
                noisy_input=noisy_input,
                timesteps=timesteps,
                local_conditioning=local_conditioning_b,
            )
        assert not torch.allclose(output_a, output_b)

    def test_same_inputs_produce_same_outputs(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        timestep_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        input_dimension = 8
        sequence_length = 16
        module = unet_factory(input_dimension=input_dimension)
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = timestep_factory(batch_size=batch_size)
        with torch.no_grad():
            output_a = module(noisy_input=noisy_input, timesteps=timesteps)
            output_b = module(noisy_input=noisy_input, timesteps=timesteps)
        assert torch.allclose(output_a, output_b)


class TestConditionalUnet1DSkipConnections:
    def test_zeroing_skip_connections_changes_output(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        # Verify skip connections actually affect the output by intercepting
        # the hidden_states list and zeroing the skip tensors
        input_dimension = 8
        sequence_length = 16
        module = unet_factory(
            input_dimension=input_dimension,
            down_dimensions=[32, 64],
        )
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = torch.zeros(batch_size, dtype=torch.long)
        with torch.no_grad():
            output_normal = module(noisy_input=noisy_input, timesteps=timesteps)
        # Zero out the skip connection by monkey-patching the first upsample block
        # to receive zeros instead of the encoder hidden states
        original_forward = module.upsampling_modules[0][0].forward

        def zeroed_skip_forward(x, condition):
            # x has doubled channels from skip concat; zero out the skip half
            half_channels = x.shape[1] // 2
            x_zeroed = x.clone()
            x_zeroed[:, half_channels:, :] = 0.0
            return original_forward(x=x_zeroed, condition=condition)

        module.upsampling_modules[0][0].forward = zeroed_skip_forward
        with torch.no_grad():
            output_zeroed = module(noisy_input=noisy_input, timesteps=timesteps)
        assert not torch.allclose(output_normal, output_zeroed)

    def test_encoder_hidden_states_are_concatenated_in_decoder(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        input_dimension = 8
        sequence_length = 16
        module = unet_factory(
            input_dimension=input_dimension,
            down_dimensions=[32, 64],
        )
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = torch.zeros(batch_size, dtype=torch.long)
        # Capture the input to the first upsample residual block to verify
        # it receives doubled channels from skip connection concatenation
        captured_inputs = []
        original_forward = module.upsampling_modules[0][0].forward

        def capturing_forward(x, condition):
            captured_inputs.append(x.shape)
            return original_forward(x=x, condition=condition)

        module.upsampling_modules[0][0].forward = capturing_forward
        with torch.no_grad():
            module(noisy_input=noisy_input, timesteps=timesteps)
        # The first residual block in upsampling receives 2x channels (skip concat)
        channel_dimension = 1
        assert captured_inputs[0][channel_dimension] == 64 * 2


class TestConditionalUnet1DLocalConditioningIntegration:
    def test_local_conditioning_forward_pass_completes(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        local_conditioning_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        # Single down_dimension avoids sequence mismatch for local conditioning
        input_dimension = 8
        sequence_length = 16
        local_conditioning_dimension = 16
        module = unet_factory(
            input_dimension=input_dimension,
            local_conditioning_dimension=local_conditioning_dimension,
            down_dimensions=[32],
        )
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = torch.zeros(batch_size, dtype=torch.long)
        local_conditioning = local_conditioning_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            local_conditioning_dimension=local_conditioning_dimension,
        )
        with torch.no_grad():
            output = module(
                noisy_input=noisy_input,
                timesteps=timesteps,
                local_conditioning=local_conditioning,
            )
        assert output.shape == (batch_size, sequence_length, input_dimension)

    def test_local_conditioning_affects_first_downsample_block(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        local_conditioning_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        # Verify the local conditioning encoder produces hidden states that get added
        # at the first downsample block (index == 0)
        input_dimension = 8
        sequence_length = 16
        local_conditioning_dimension = 16
        module = unet_factory(
            input_dimension=input_dimension,
            local_conditioning_dimension=local_conditioning_dimension,
            down_dimensions=[32],
        )
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = torch.zeros(batch_size, dtype=torch.long)
        local_conditioning = local_conditioning_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            local_conditioning_dimension=local_conditioning_dimension,
        )
        call_count = [0]
        original_forwards = [
            module.local_condition_encoder[0].forward,
            module.local_condition_encoder[1].forward,
        ]

        def counting_forward_0(x, condition):
            call_count[0] += 1
            return original_forwards[0](x=x, condition=condition)

        def counting_forward_1(x, condition):
            call_count[0] += 1
            return original_forwards[1](x=x, condition=condition)

        module.local_condition_encoder[0].forward = counting_forward_0
        module.local_condition_encoder[1].forward = counting_forward_1
        with torch.no_grad():
            module(
                noisy_input=noisy_input,
                timesteps=timesteps,
                local_conditioning=local_conditioning,
            )
        # Both local condition encoders are called during forward pass
        assert call_count[0] == 2


class TestConditionalUnet1DConditionPredictScale:
    @pytest.mark.parametrize("condition_predict_scale", [True, False])
    def test_forward_with_condition_predict_scale(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        timestep_factory: Callable[..., torch.Tensor],
        batch_size: int,
        condition_predict_scale: bool,
    ):
        input_dimension = 8
        sequence_length = 16
        module = unet_factory(
            input_dimension=input_dimension,
            condition_predict_scale=condition_predict_scale,
        )
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = timestep_factory(batch_size=batch_size)
        with torch.no_grad():
            output = module(noisy_input=noisy_input, timesteps=timesteps)
        assert output.shape == (batch_size, sequence_length, input_dimension)


@pytest.mark.unit
class TestConditionalUnet1DLocalConditioningEffect:
    def test_local_conditioning_changes_output(
        self,
        unet_factory: Callable[..., ConditionalUnet1D],
        sequence_tensor_factory: Callable[..., torch.Tensor],
        timestep_factory: Callable[..., torch.Tensor],
        local_conditioning_factory: Callable[..., torch.Tensor],
        batch_size: int,
    ):
        input_dimension = 8
        sequence_length = 16
        local_conditioning_dimension = 16
        module = unet_factory(
            input_dimension=input_dimension,
            local_conditioning_dimension=local_conditioning_dimension,
            down_dimensions=[32, 64],
        )
        module.eval()
        noisy_input = sequence_tensor_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            embedding_dimension=input_dimension,
        )
        timesteps = timestep_factory(batch_size=batch_size)
        local_conditioning = local_conditioning_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            local_conditioning_dimension=local_conditioning_dimension,
        )
        with torch.no_grad():
            with_conditioning = module(
                noisy_input=noisy_input,
                timesteps=timesteps,
                local_conditioning=local_conditioning,
            )
            with_other_conditioning = module(
                noisy_input=noisy_input,
                timesteps=timesteps,
                local_conditioning=local_conditioning + 1.0,
            )

        assert not torch.allclose(with_conditioning, with_other_conditioning)
