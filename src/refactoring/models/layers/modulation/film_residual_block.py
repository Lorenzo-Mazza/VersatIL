import torch
import torch.nn as nn

from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.modulation.conditional_modulation import (
    ConditionalModulation,
)


class FiLMedResBlock(nn.Module):
    """ResNet residual block with FiLM conditioning."""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            condition_dim: int,
            stride: int = 1,
            downsample: nn.Module | None = None
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.film1 = ConditionalModulation(
            condition_dim=condition_dim,
            feature_dim=out_channels,
            use_shift=True,
            activation=ActivationFunction.LINEAR.value,
            init_strategy="identity"
        )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.film2 = ConditionalModulation(
            condition_dim=condition_dim,
            feature_dim=out_channels,
            use_shift=True,
            activation=ActivationFunction.LINEAR.value,
            init_strategy="identity"
        )

        self.relu = nn.ReLU()
        self.downsample = downsample


    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.film1(out, condition)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.film2(out, condition)

        out = out + identity
        result: torch.Tensor = self.relu(out)

        return result



