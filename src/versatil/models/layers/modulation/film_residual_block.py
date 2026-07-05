import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.modulation.conditional_modulation import (
    ConditionalModulation,
)


class FiLMedResBlock(nn.Module):
    """ResNet residual block with FiLM conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conditioning_dimension: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.film1 = ConditionalModulation(
            conditioning_dimension=conditioning_dimension,
            feature_dim=out_channels,
            use_shift=True,
            activation=ActivationFunction.LINEAR.value,
            init_strategy="zero",
        )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.film2 = ConditionalModulation(
            conditioning_dimension=conditioning_dimension,
            feature_dim=out_channels,
            use_shift=True,
            activation=ActivationFunction.LINEAR.value,
            init_strategy="zero",
        )

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply the FiLM-modulated residual convolution block."""
        identity = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out, _ = self.film1(out, condition)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out, _ = self.film2(out, condition)

        out = out + identity
        result: torch.Tensor = self.relu(out)

        return result
