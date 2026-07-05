import torch
from torch import nn

from versatil.models.layers.constants import AttentionDecompositionMode
from versatil.models.layers.convolution.depthwise_conv2d import DepthwiseConv2D
from versatil.models.layers.geometric_attention.geometric_bias import (
    GeometricAttentionBias,
)


class GeometricSelfAttention(nn.Module):
    """Geometric self-attention with spatial and depth-aware biases.

    Combines:
    1. Rotary positional encoding for relative position modeling
    2. Geometric attention bias (spatial + depth)
    3. Learned positional encoding (through depth-wise spatial convolution) on values
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        value_dimension_factor: int = 1,
        decomposition_mode: str = AttentionDecompositionMode.FULL.value,
        initial_decay: float = 5.0,
        decay_range: float = 3.0,
        depthwise_convolution_kernel_size: int = 5,
        depthwise_convolution_padding: int = 2,
        use_raster_positions: bool = False,
    ):
        """Initializes geometric self-attention.

        Args:
            embedding_dimension: Embedding dimension.
            number_of_heads: Number of attention heads.
            value_dimension_factor: Factor to expand value dimension.
            decomposition_mode: Full or separable attention computation.
            initial_decay: Initial spatial decay rate.
            decay_range: Range of decay rates across heads.
            depthwise_convolution_kernel_size: Kernel size for depth-wise convolution, used for learned positional encodings.
            depthwise_convolution_padding: Padding for depth-wise convolution
            use_raster_positions: Whether rotary encoding uses flattened raster
                grid positions (the DFormerv2 reference convention).
        """
        if number_of_heads < 1:
            raise ValueError(
                f"number_of_heads must be positive, got {number_of_heads}."
            )
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.value_dimension_factor = value_dimension_factor
        self.decomposition_mode = decomposition_mode

        self.head_dimension_key = embedding_dimension // number_of_heads
        self.head_dimension_value = (
            embedding_dimension * value_dimension_factor
        ) // number_of_heads
        self.attention_scaling = self.head_dimension_key**-0.5

        self.query_projection = nn.Linear(
            embedding_dimension, embedding_dimension, bias=True
        )
        self.key_projection = nn.Linear(
            embedding_dimension, embedding_dimension, bias=True
        )
        self.value_projection = nn.Linear(
            embedding_dimension, embedding_dimension * value_dimension_factor, bias=True
        )

        self.learned_positional_encodings = DepthwiseConv2D(
            dimension=embedding_dimension * value_dimension_factor,
            kernel_size=depthwise_convolution_kernel_size,
            stride=1,
            padding=depthwise_convolution_padding,
        )

        self.output_projection = nn.Linear(
            embedding_dimension * value_dimension_factor, embedding_dimension, bias=True
        )

        self.geometric_bias = GeometricAttentionBias(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            initial_decay=initial_decay,
            decay_range=decay_range,
            use_raster_positions=use_raster_positions,
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initializes projection weights with careful scaling."""
        nn.init.xavier_normal_(self.query_projection.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.key_projection.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.value_projection.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0.0)

    def _compute_attention_full(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sine: torch.Tensor,
        cosine: torch.Tensor,
        attention_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Computes full 2D attention over all spatial positions.

        Args:
            query: Query tensor (B, number_of_heads, H, W, head_dim).
            key: Key tensor (B, number_of_heads, H, W, head_dim).
            value: Value tensor (B, number_of_heads, H, W, head_dim_value).
            sine: Sine components for rotation.
            cosine: Cosine components for rotation.
            attention_bias: Geometric bias (B or 1, number_of_heads, H*W, H*W).

        Returns:
            Attention output (B, H, W, embedding_dimension * value_factor).
        """
        batch_size, _, height, width, _ = query.shape

        query_rotated = self.geometric_bias.rotary_encoding.apply_rotation(
            query, sine, cosine
        )
        key_rotated = self.geometric_bias.rotary_encoding.apply_rotation(
            key, sine, cosine
        )

        query_flat = query_rotated.flatten(2, 3)
        key_flat = key_rotated.flatten(2, 3)
        value_flat = value.flatten(2, 3)

        attention_scores = torch.matmul(query_flat, key_flat.transpose(-1, -2))
        attention_scores = attention_scores + attention_bias
        attention_weights = torch.softmax(attention_scores, dim=-1)

        attended_values = torch.matmul(attention_weights, value_flat)
        attended_values = attended_values.transpose(1, 2).reshape(
            batch_size, height, width, -1
        )

        return attended_values

    def _compute_attention_separable(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        sine: torch.Tensor,
        cosine: torch.Tensor,
        height_bias: torch.Tensor,
        width_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Computes separable attention (horizontal then vertical).

        Args:
            query: Query tensor (B, number_of_heads, H, W, head_dim).
            key: Key tensor (B, number_of_heads, H, W, head_dim).
            value: Value tensor (B, number_of_heads, H, W, head_dim_value).
            sine: Sine components for rotation.
            cosine: Cosine components for rotation.
            height_bias: Height attention bias.
            width_bias: Width attention bias.

        Returns:
            Attention output (B, H, W, embedding_dimension * value_factor).
        """
        batch_size, _, height, width, _ = query.shape

        query_rotated = self.geometric_bias.rotary_encoding.apply_rotation(
            query, sine, cosine
        )
        key_rotated = self.geometric_bias.rotary_encoding.apply_rotation(
            key, sine, cosine
        )

        query_width = query_rotated.transpose(1, 2)
        key_width = key_rotated.transpose(1, 2)
        value_height_first = value.transpose(1, 2)  # (B, H, number_of_heads, W, dv)

        attention_scores_width = torch.matmul(query_width, key_width.transpose(-1, -2))
        attention_scores_width = attention_scores_width + width_bias.transpose(1, 2)
        attention_weights_width = torch.softmax(attention_scores_width, dim=-1)
        value_after_width = torch.matmul(attention_weights_width, value_height_first)

        query_height = query_rotated.permute(0, 3, 1, 2, 4)
        key_height = key_rotated.permute(0, 3, 1, 2, 4)
        value_for_height = value_after_width.permute(0, 3, 2, 1, 4)

        attention_scores_height = torch.matmul(
            query_height, key_height.transpose(-1, -2)
        )
        attention_scores_height = attention_scores_height + height_bias.transpose(1, 2)
        attention_weights_height = torch.softmax(attention_scores_height, dim=-1)
        attended_values = torch.matmul(attention_weights_height, value_for_height)

        attended_values = attended_values.permute(0, 3, 1, 2, 4).flatten(-2, -1)

        return attended_values

    def forward(
        self, input_tensor: torch.Tensor, depth_map: torch.Tensor
    ) -> torch.Tensor:
        """Applies geometric self-attention.

        Args:
            input_tensor: Input of shape (B, H, W, C).
            depth_map: Depth map of shape (B, 1, H, W).

        Returns:
            Attention output of shape (B, H, W, C).
        """
        batch_size, height, width, _ = input_tensor.shape

        query = self.query_projection(input_tensor)
        key = self.key_projection(input_tensor)
        value = self.value_projection(input_tensor)
        key = key * self.attention_scaling

        query = query.view(
            batch_size, height, width, self.number_of_heads, self.head_dimension_key
        )
        query = query.permute(0, 3, 1, 2, 4)

        key = key.view(
            batch_size, height, width, self.number_of_heads, self.head_dimension_key
        )
        key = key.permute(0, 3, 1, 2, 4)

        value = value.view(
            batch_size, height, width, self.number_of_heads, self.head_dimension_value
        )
        value = value.permute(0, 3, 1, 2, 4)

        (sine, cosine), bias_masks = self.geometric_bias(
            height=height,
            width=width,
            depth_map=depth_map,
            device=input_tensor.device,
            decomposition_mode=self.decomposition_mode,
        )

        if self.decomposition_mode == AttentionDecompositionMode.SEPARABLE.value:
            attended_values = self._compute_attention_separable(
                query, key, value, sine, cosine, bias_masks[0], bias_masks[1]
            )
        else:
            attended_values = self._compute_attention_full(
                query, key, value, sine, cosine, bias_masks[0]
            )

        positional_encoding = self.learned_positional_encodings(
            self.value_projection(input_tensor)
        )
        output = attended_values + positional_encoding
        output = self.output_projection(output)

        return output
