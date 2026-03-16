"""Conditional 1-dimensional U-Net architecture, originally used in the Diffusion Policy paper https://arxiv.org/abs/2303.04137v4"""

import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from versatil.models.layers.convolution.conv1d import (
    Conv1dBlock,
    Downsample1d,
    Upsample1d,
)
from versatil.models.layers.modulation.conditional_residual_block import (
    ConditionalResidualBlock1D,
)
from versatil.models.layers.normalization.rms_norm import RMSNorm
from versatil.models.layers.positional_encoding.base import (
    DenominatorMode,
    OrderingMode,
    PositionSource,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        local_conditioning_dimension: int | None = None,
        global_conditioning_dimension: int | None = None,
        diffusion_step_embedding_dimension: int = 256,
        down_dimensions: list[int] | None = None,
        kernel_size: int = 3,
        num_groups: int = 8,
        condition_predict_scale: bool = False,
        initializer_range: float = 0.02,
    ):
        """Initialize the ConditionalUnet1D module.

        Args:
            input_dimension: Dimensionality of the input sequence features (e.g., action space size).
            local_conditioning_dimension: Dimensionality of per-timestep local conditioning (e.g., observations).
                If None, local conditioning is disabled.
            global_conditioning_dimension: Dimensionality of global conditioning (e.g., task embeddings).
                If None, global conditioning beyond diffusion steps is disabled.
            diffusion_step_embedding_dimension: Hidden size for diffusion timestep embeddings.
            down_dimensions: List of channel dimensions for downsampling layers.
            kernel_size: Kernel size for convolutions in residual blocks.
            num_groups: Number of groups for group normalization in residual blocks.
            condition_predict_scale: If True, conditions predict scaling factors in residual blocks.
            initializer_range: std of the initial weights for conv and linear layers.
        """
        super().__init__()
        if down_dimensions is None:
            down_dimensions = [256, 512, 1024]
        all_dimensions = [input_dimension] + list(down_dimensions)
        starting_dimension = down_dimensions[0]
        diffusion_step_embedding_dimension = diffusion_step_embedding_dimension
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPositionalEncoding1D(
                embedding_dimension=diffusion_step_embedding_dimension,
                denominator_mode=DenominatorMode.HALF_MINUS_ONE.value,
                ordering_mode=OrderingMode.CAT_COS_SIN.value,
                position_source=PositionSource.SCALAR.value,
                precompute_encodings=False,
                temperature=10000.0,
            ),
            nn.Linear(
                diffusion_step_embedding_dimension,
                diffusion_step_embedding_dimension * 4,
            ),
            nn.Mish(),
            nn.Linear(
                diffusion_step_embedding_dimension * 4,
                diffusion_step_embedding_dimension,
            ),
        )
        condition_dimension = diffusion_step_embedding_dimension
        if global_conditioning_dimension is not None:
            condition_dimension += global_conditioning_dimension

        input_output_pairs = list(zip(all_dimensions[:-1], all_dimensions[1:]))

        local_condition_encoder = None
        if local_conditioning_dimension is not None:
            _, dimension_out = input_output_pairs[0]
            dimension_in = local_conditioning_dimension
            local_condition_encoder = nn.ModuleList(
                [
                    # down encoder
                    ConditionalResidualBlock1D(
                        dimension_in,
                        output_channels=dimension_out,
                        condition_dimension=condition_dimension,
                        kernel_size=kernel_size,
                        num_groups=num_groups,
                        condition_predict_scale=condition_predict_scale,
                    ),
                    # up encoder
                    ConditionalResidualBlock1D(
                        dimension_in,
                        output_channels=dimension_out,
                        condition_dimension=condition_dimension,
                        kernel_size=kernel_size,
                        num_groups=num_groups,
                        condition_predict_scale=condition_predict_scale,
                    ),
                ]
            )

        middle_dimension = all_dimensions[-1]
        self.middle_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    input_channels=middle_dimension,
                    output_channels=middle_dimension,
                    condition_dimension=condition_dimension,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    condition_predict_scale=condition_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    input_channels=middle_dimension,
                    output_channels=middle_dimension,
                    condition_dimension=condition_dimension,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    condition_predict_scale=condition_predict_scale,
                ),
            ]
        )

        downsampling_modules = nn.ModuleList([])
        for index, (dimension_in, dimension_out) in enumerate(input_output_pairs):
            is_last = index >= (len(input_output_pairs) - 1)
            downsampling_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            input_channels=dimension_in,
                            output_channels=dimension_out,
                            condition_dimension=condition_dimension,
                            kernel_size=kernel_size,
                            num_groups=num_groups,
                            condition_predict_scale=condition_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            input_channels=dimension_out,
                            output_channels=dimension_out,
                            condition_dimension=condition_dimension,
                            kernel_size=kernel_size,
                            num_groups=num_groups,
                            condition_predict_scale=condition_predict_scale,
                        ),
                        Downsample1d(dimension_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        upsampling_modules = nn.ModuleList([])
        for index, (dimension_in, dimension_out) in enumerate(
            reversed(input_output_pairs[1:])
        ):
            is_last = index >= (len(input_output_pairs) - 1)
            upsampling_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            input_channels=dimension_out * 2,
                            output_channels=dimension_in,
                            condition_dimension=condition_dimension,
                            kernel_size=kernel_size,
                            num_groups=num_groups,
                            condition_predict_scale=condition_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            input_channels=dimension_in,
                            output_channels=dimension_in,
                            condition_dimension=condition_dimension,
                            kernel_size=kernel_size,
                            num_groups=num_groups,
                            condition_predict_scale=condition_predict_scale,
                        ),
                        Upsample1d(dimension_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_convolution = nn.Sequential(
            Conv1dBlock(
                starting_dimension, starting_dimension, kernel_size=kernel_size
            ),
            nn.Conv1d(starting_dimension, input_dimension, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_condition_encoder = local_condition_encoder
        self.upsampling_modules = upsampling_modules
        self.downsampling_modules = downsampling_modules
        self.final_convolution = final_convolution
        self.initializer_range = initializer_range
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if hasattr(module, "_is_modulation_layer") and module._is_modulation_layer:
                return
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm, nn.GroupNorm, _BatchNorm)):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        noisy_input: torch.Tensor,
        timesteps: torch.Tensor | float | int,
        local_conditioning: torch.Tensor | None = None,
        global_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the conditional U-Net.

        Processes noisy input sequences through the U-Net, injecting conditions at each residual block.
        The input is transposed to (batch, channels, sequence) for 1D convolutions and transposed back
        at the output.

        Args:
            noisy_input: Noisy input tensor of shape (batch_size, sequence_length, input_dimension).
            timesteps: Diffusion timesteps; can be a tensor of shape (batch_size,) or a scalar value.
            local_conditioning: Optional local conditioning tensor of shape
                (batch_size, sequence_length, local_condition_dimension).
            global_conditioning: Optional global conditioning tensor of shape
                (batch_size, global_condition_dimension).

        Returns:
            Denoised output tensor of shape (batch_size, sequence_length, input_dimension).
        """
        noisy_input = noisy_input.permute(
            0, 2, 1
        )  # Shape: (batch_size, horizon, input_dimension) -> (batch_size, input_dimension, horizon)
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=noisy_input.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(noisy_input.device)
        # broadcast to batch dimension
        timesteps = timesteps.expand(noisy_input.shape[0])

        global_features = self.diffusion_step_encoder(timesteps)

        if global_conditioning is not None:
            global_features = torch.cat([global_features, global_conditioning], dim=-1)

        # encode local features
        local_hidden_states = []
        if local_conditioning is not None:
            local_conditioning = local_conditioning.permute(
                0, 2, 1
            )  # Shape: (batch_size, horizon, local_conditioning_dimension) -> (batch_size, local_conditioning_dimension, horizon)
            first_residual_block, second_residual_block = self.local_condition_encoder
            x = first_residual_block(x=local_conditioning, condition=global_features)
            local_hidden_states.append(x)
            x = second_residual_block(x=local_conditioning, condition=global_features)
            local_hidden_states.append(x)

        x = noisy_input
        hidden_states = []
        for index, (
            first_residual_block,
            second_residual_block,
            downsample,
        ) in enumerate(self.downsampling_modules):
            x = first_residual_block(x=x, condition=global_features)
            if index == 0 and len(local_hidden_states) > 0:
                x = x + local_hidden_states[0]
            x = second_residual_block(x=x, condition=global_features)
            hidden_states.append(x)
            x = downsample(x)

        for middle_module in self.middle_modules:
            x = middle_module(x=x, condition=global_features)

        for index, (first_residual_block, second_residual_block, upsample) in enumerate(
            self.upsampling_modules
        ):
            x = torch.cat((x, hidden_states.pop()), dim=1)
            x = first_residual_block(x=x, condition=global_features)
            if (
                index == (len(self.upsampling_modules) - 1)
                and len(local_hidden_states) > 0
            ):
                x = x + local_hidden_states[1]
            x = second_residual_block(x=x, condition=global_features)
            x = upsample(x)

        x = self.final_convolution(x)

        x = x.permute(
            0, 2, 1
        )  # Shape: (batch_size, input_dimension, horizon) -> (batch_size, horizon, input_dimension)
        return x
