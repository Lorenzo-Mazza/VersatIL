"""Conditional Residual Block 1D Module originally used in the Diffusion Policy U-Net architecture."""
from torch import nn

from refactoring.models.layers import ConditionalModulation
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.convolution.conv1d import Conv1dBlock


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        condition_dimension,
        kernel_size=3,
        num_groups=8,
        condition_predict_scale=False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(
                    input_channels, output_channels, kernel_size, n_groups=num_groups
                ),
                Conv1dBlock(
                    output_channels, output_channels, kernel_size, n_groups=num_groups
                ),
            ]
        )
        self.modulator = ConditionalModulation(
            condition_dim=condition_dimension,
            feature_dim=output_channels,
            use_shift=condition_predict_scale,
            activation=ActivationFunction.MISH.value,
            init_strategy="identity",
        )
        self.residual_convolution = (
            nn.Conv1d(input_channels, output_channels, 1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, x, condition):
        """Forward pass of ConditionalResidualBlock1D.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_channels, prediction horizon).
            condition (torch.Tensor): Conditioning tensor of shape (batch_size, condition_dimension).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_channels, prediction horizon).
        """
        out = self.blocks[0](x)
        out = self.modulator(out, condition)
        out = self.blocks[1](out)
        out = out + self.residual_convolution(x)
        return out
