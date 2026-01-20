"""Module for depth-wise 2D Convolution layer."""
import torch
from torch import nn


class DepthwiseConv2D(nn.Module):
    """A depth-wise 2D convolution applies a single convolutional filter to each input channel independently,
    without mixing channels (unlike standard convolution).
    """

    def __init__(self, dimension, kernel_size, stride, padding):
        super().__init__()
        self.convolution = nn.Conv2d(
            dimension, dimension, kernel_size, stride, padding, groups=dimension
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input of shape (batch, height, width, channels).

        Returns:
            output: tensor after applying depth-wise spatial convolutions, shape (batch, height, width, channels).
        """
        x = x.permute(0, 3, 1, 2)
        x = self.convolution(x)
        x = x.permute(0, 2, 3, 1)
        return x
