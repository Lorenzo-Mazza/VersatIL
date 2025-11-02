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



'''class FiLMedResNetBasicBlock(nn.Module):
    """FiLM-conditioned version of TIMM's ResNet BasicBlock."""
    def __init__(self, condition_dim: int, original_block: nn.Module):
        super().__init__()
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.film1 = ConditionalModulation(
            condition_dim=condition_dim,
            feature_dim=self.conv1.out_channels,
            use_shift=True,
            use_activation=False,
            init_strategy="identity"
        )
        self.act1 = original_block.act1
        self.drop_block = getattr(original_block, 'drop_block', nn.Identity())
        self.aa = getattr(original_block, 'aa', nn.Identity())
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.film2 = ConditionalModulation(
            condition_dim=condition_dim,
            feature_dim=self.conv2.out_channels,
            use_shift=True,
            use_activation=False,
            init_strategy="identity"
        )
        self.downsample = original_block.downsample
        self.act2 = getattr(original_block, 'act2', nn.ReLU(inplace=True))

    def forward(self, action_embedding: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(action_embedding) if self.downsample else action_embedding
        out = self.conv1(action_embedding)
        out = self.bn1(out)
        out = self.film1(out, condition)
        out = self.act1(out)
        out = self.drop_block(out)
        out = self.aa(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.film2(out, condition)
        out += identity
        out = self.act2(out)
        return out


class FiLMedBottleneck(nn.Module):
    """ResNet bottleneck block with FiLM conditioning."""
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            mid_channels: int,
            condition_dim: int,
            stride: int = 1,
            downsample: nn.Module = None
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.film1 = ConditionalModulation(
            condition_dim=condition_dim,
            feature_dim=mid_channels,
            use_shift=True,
            use_activation=False,
            init_strategy="identity"
        )

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.film2 = ConditionalModulation(
            condition_dim=condition_dim,
            feature_dim=mid_channels,
            use_shift=True,
            use_activation=False,
            init_strategy="identity"
        )

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.film3 = ConditionalModulation(
            condition_dim=condition_dim,
            feature_dim=out_channels,
            use_shift=True,
            use_activation=False,
            init_strategy="identity"
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, action_embedding: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        identity = action_embedding if self.downsample is None else self.downsample(action_embedding)

        out = self.conv1(action_embedding)
        out = self.bn1(out)
        out = self.film1(out, condition)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.film2(out, condition)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.film3(out, condition)

        out = out + identity
        out = self.relu(out)

        return out'''
