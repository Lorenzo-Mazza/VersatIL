"""Conditional Residual Block 1D Module originally used in the Diffusion Policy U-Net architecture."""

import torch
from torch import nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.convolution.conv1d import Conv1dBlock
from versatil.models.layers.modulation.conditional_modulation import (
    ConditionalModulation,
)


class ConditionalResidualBlock1D(nn.Module):
    """Conditioned residual block for 1D diffusion-policy feature maps."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        condition_dimension: int,
        kernel_size: int = 3,
        num_groups: int = 8,
        condition_predict_scale: bool = False,
    ) -> None:
        """Initialize the conditioned residual block.

        Args:
            input_channels: Number of input channels.
            output_channels: Number of output channels.
            condition_dimension: Dimension of the conditioning vector.
            kernel_size: Convolution kernel size.
            num_groups: Number of groups used by Conv1dBlock normalization.
            condition_predict_scale: Whether conditioning predicts both scale
                and shift instead of scale only.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    input_channels, output_channels, kernel_size, num_groups=num_groups
                ),
                Conv1dBlock(
                    output_channels, output_channels, kernel_size, num_groups=num_groups
                ),
            ]
        )
        self.modulator = ConditionalModulation(
            condition_dim=condition_dimension,
            feature_dim=output_channels,
            use_shift=condition_predict_scale,
            activation=ActivationFunction.MISH.value,
            init_strategy="zero",
            feature_axis=1,
        )
        self.residual_convolution = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass of ConditionalResidualBlock1D.

        Args:
            x: Input tensor of shape ``(batch_size, input_channels, prediction_horizon)``.
            condition: Conditioning tensor of shape ``(batch_size, condition_dimension)``.

        Returns:
            Output tensor of shape ``(batch_size, output_channels, prediction_horizon)``.
        """
        out = self.blocks[0](x)
        out, _ = self.modulator(out, condition)
        out = self.blocks[1](out)
        out = out + self.residual_convolution(x)
        return out
