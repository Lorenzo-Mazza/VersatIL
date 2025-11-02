import torch
import torch.nn as nn


class SpatialSoftmax(nn.Module):
    """Spatial softmax layer for extracting keypoints from feature maps, based on https://rll.berkeley.edu/dsae/dsae.pdf"""
    def __init__(self,
                 height: int,
                 width: int,
                 channel: int,
                 temperature: float = 1.0,
                 learnable_temperature: bool = False,
                 ):
        """Initializes the SpatialSoftmax module.

        Args:
            height: Height of the input feature map
            width: Width of the input feature map
            channel: Number of channels in the input feature map
            temperature: Temperature parameter for softmax
            learnable_temperature: If True, temperature is a learnable parameter
        """
        super().__init__()
        self.height = height
        self.width = width
        self.channel = channel
        self.learnable_temperature = learnable_temperature
        if self.learnable_temperature:
            temperature_param = nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter("temperature", temperature_param)
            self.temperature = temperature_param
        else:
            # temperature held constant after initialization
            temperature_param = nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer("temperature", temperature_param)
            self.temperature = temperature_param
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, width),
            torch.linspace(-1, 1, height),
            indexing='xy'
        )
        self.register_buffer('pos_x', pos_x.reshape(1, height * width))
        self.register_buffer('pos_y', pos_y.reshape(1, height * width))


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, C, H, W) feature map
        Returns:
            keypoints: (B, C * 2)
        """
        B, C, H, W = features.shape
        features = features.reshape(B, C, H * W)
        softmax_attn = torch.softmax(features / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * softmax_attn, dim=-1, keepdim=False)
        expected_y = torch.sum(self.pos_y * softmax_attn, dim=-1, keepdim=False)
        return torch.cat([expected_x, expected_y], dim=-1)

