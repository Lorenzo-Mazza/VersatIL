"""Taken from DFormerV2 paper: https://arxiv.org/pdf/2504.04701"""
import torch
from torch import nn

from refactoring.models.layers.constants import AttentionDecompositionMode
from refactoring.models.layers.convolution.depthwise_conv2d import DepthwiseConv2D
from refactoring.models.layers.drop_path import DropPath
from refactoring.models.layers.geometric_attention import GeometricSelfAttention
from refactoring.models.layers.mlp import MLP


class GeometricAttentionEncoderBlock(nn.Module):
    """Geometric attention encoder block for conditioning with depth maps on RGB images.

    Integrates depth-conditioned attention, feed-forward network,
    and optional layer scaling for residual connections.
    """

    def __init__(
        self,
        decomposition_mode: AttentionDecompositionMode,
        embedding_dimension: int,
        num_heads: int,
        ffn_dimension: int,
        drop_path_rate: float = 0.0,
        use_layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        initial_decay: float = 2.0,
        decay_range: float = 4.0,
        value_dimension_factor: int = 1,
        depthwise_kernel_size: int = 5,
        depthwise_padding: int = 2,
        input_positional_kernel_size: int = 3,
        input_positional_padding: int = 1,
    ):
        """Initializes the geometric attention encoder block.

        Args:
            decomposition_mode: Attention mode (full or separable).
            embedding_dimension: Feature dimension.
            num_heads: Number of attention heads.
            ffn_dimension: Hidden dimension for the fully-connected layer that follows the self-attention layer.
            drop_path_rate: Stochastic depth rate.
            use_layer_scale: Whether to use layer scaling.
            layer_scale_init_value: Initial value for layer scale parameters.
            initial_decay: Initial decay rate for spatial biases.
            decay_range: Range of decay rates across heads.
            value_dimension_factor: Expansion factor for value dimension.
            depthwise_kernel_size: Kernel size for value positional encoding.
            depthwise_padding: Padding for value positional encoding.
            input_positional_kernel_size: Kernel size for input positional encoding.
            input_positional_padding: Padding for input positional encoding.
        """
        super().__init__()
        self.use_layer_scale = use_layer_scale
        self.embedding_dimension = embedding_dimension

        self.norm1 = nn.LayerNorm(embedding_dimension, eps=1e-6)
        self.norm2 = nn.LayerNorm(embedding_dimension, eps=1e-6)

        self.attention = GeometricSelfAttention(
            embedding_dimension=embedding_dimension,
            num_heads=num_heads,
            value_dimension_factor=value_dimension_factor,
            decomposition_mode=decomposition_mode.value,
            initial_decay=initial_decay,
            decay_range=decay_range,
            depthwise_convolution_kernel_size=depthwise_kernel_size,
            depthwise_convolution_padding=depthwise_padding,
        )

        self.drop_path = DropPath(drop_path_rate)
        self.mlp = MLP(
            input_dim=embedding_dimension,
            hidden_dims=[ffn_dimension],
            output_dim=embedding_dimension,
            activation_function=nn.GELU,
            dropout=0.0,
        )

        self.input_positional_encoding = DepthwiseConv2D(
            dimension=embedding_dimension,
            kernel_size=input_positional_kernel_size,
            stride=1,
            padding=input_positional_padding,
        )

        if use_layer_scale:
            self.gamma1 = nn.Parameter(
                layer_scale_init_value * torch.ones(1, 1, 1, embedding_dimension),
                requires_grad=True,
            )
            self.gamma2 = nn.Parameter(
                layer_scale_init_value * torch.ones(1, 1, 1, embedding_dimension),
                requires_grad=True,
            )

    def forward(self, rgb_tensor: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
        """Applies the geometric attention encoder block.

        Args:
            rgb_tensor: Input images of shape (B, H, W, C).
            depth_map: Depth map of shape (B, 1, H, W).

        Returns:
            Updated features of shape (B, H, W, C).
        """
        features = rgb_tensor + self.input_positional_encoding(rgb_tensor)

        residual = features
        features = self.norm1(features)
        attention_output = self.attention(features, depth_map)

        if self.use_layer_scale:
            features = residual + self.drop_path(self.gamma1 * attention_output)
        else:
            features = residual + self.drop_path(attention_output)

        residual = features
        features = self.norm2(features)
        mlp_output = self.mlp(features)

        if self.use_layer_scale:
            features = residual + self.drop_path(self.gamma2 * mlp_output)
        else:
            features = residual + self.drop_path(mlp_output)

        return features  # type: ignore[no-any-return]
