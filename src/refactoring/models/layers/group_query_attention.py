import torch
import torch.nn as nn
import torch.nn.functional as F
from refactoring.models.layers.positional_encoding.rotary import RotaryPositionalEncoding1D  # Assume import

class GroupQueryAttention(nn.Module):
    """Group Query Attention module.

    This module implements Group Query Attention (https://arxiv.org/pdf/2305.13245), where query heads are divided into groups,
    each sharing the same key and value heads. This reduces parameters compared to standard
    multi-head attention while maintaining performance.

    Args:
        embedding_dimension (int): The embedding dimension.
        number_of_heads (int): The number of query attention heads.
        number_of_key_value_heads (int): The number of key-value heads (must divide number_of_heads evenly).
        dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.
        bias (bool, optional): Whether to include bias in linear projections. Defaults to True.
        use_rope (bool, optional): Whether to use Rotary Position Embeddings. Defaults to False.
        rope_base (float, optional): Base frequency for RoPE. Defaults to 10000

    Raises:
        ValueError: If embedding_dimension is not divisible by number_of_heads, or number_of_heads is not divisible by number_of_key_value_heads.
    """

    def __init__(self,
                 embedding_dimension: int = 1536,
                 number_of_heads: int = 12,
                 number_of_key_value_heads: int = 2,
                 dropout: float = 0.0,
                 bias: bool = True,
                 use_rope: bool = False,
                 rope_base: float = 10000.0
                 ):
        super().__init__()
        if embedding_dimension % number_of_heads != 0:
            raise ValueError("embedding_dimension must be divisible by number_of_heads")
        if number_of_heads % number_of_key_value_heads != 0:
            raise ValueError("number_of_heads must be divisible by number_of_key_value_heads")

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.number_of_key_value_heads = number_of_key_value_heads
        self.dropout = dropout
        self.head_dimension = embedding_dimension // number_of_heads
        self.group_size = number_of_heads // number_of_key_value_heads

        self.query_projection = nn.Linear(embedding_dimension, number_of_heads * self.head_dimension, bias=bias)
        self.key_projection = nn.Linear(embedding_dimension, number_of_key_value_heads * self.head_dimension, bias=bias)
        self.value_projection = nn.Linear(embedding_dimension, number_of_key_value_heads * self.head_dimension, bias=bias)
        self.output_projection = nn.Linear(number_of_heads * self.head_dimension, embedding_dimension, bias=bias)
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEncoding1D(embedding_dimension=embedding_dimension, num_heads=number_of_heads, base_frequency=rope_base)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass for Group Query Attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, query_sequence_length, embedding_dimension).
            key (torch.Tensor): Key tensor of shape (batch_size, key_sequence_length, embedding_dimension).
            value (torch.Tensor): Value tensor of shape (batch_size, key_sequence_length, embedding_dimension).
            attention_mask (torch.Tensor | None, optional): Attention mask of shape
                (batch_size, number_of_heads, query_sequence_length, key_sequence_length). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, query_sequence_length, embedding_dimension).
        """
        batch_size, query_sequence_length, _ = query.shape
        _, key_sequence_length, _ = key.shape

        projected_query = self.query_projection(query)  # (batch_size, query_sequence_length, number_of_heads * head_dimension)
        projected_key = self.key_projection(key)        # (batch_size, key_sequence_length, number_of_key_value_heads * head_dimension)
        projected_value = self.value_projection(value)  # (batch_size, key_sequence_length, number_of_key_value_heads * head_dimension)
        # Reshape to include head dimension
        projected_query = projected_query.view(batch_size, query_sequence_length, self.number_of_heads, self.head_dimension)
        projected_key = projected_key.view(batch_size, key_sequence_length, self.number_of_key_value_heads, self.head_dimension)
        projected_value = projected_value.view(batch_size, key_sequence_length, self.number_of_key_value_heads, self.head_dimension)
        # Repeat key and value heads to match query heads
        projected_key = torch.repeat_interleave(projected_key, self.group_size, dim=2)
        projected_value = torch.repeat_interleave(projected_value, self.group_size, dim=2)
        # Transpose for attention computation: (batch_size, number_of_heads, sequence_length, head_dimension)
        projected_query = projected_query.transpose(1, 2)
        projected_key = projected_key.transpose(1, 2)
        projected_value = projected_value.transpose(1, 2)
        if self.use_rope:
            q_sine, q_cosine = self.rope.compute_rotation_components(query_sequence_length)
            q_sine = q_sine.unsqueeze(0).unsqueeze(0)  # (1, 1, Q_len, head_dim)
            q_cosine = q_cosine.unsqueeze(0).unsqueeze(0)
            projected_query = self.rope.apply_rotation(projected_query, q_sine, q_cosine)
            # For key (reuse rope instance or make separate if needed)
            k_sine, k_cosine = self.rope.compute_rotation_components(key_sequence_length)
            k_sine = k_sine.unsqueeze(0).unsqueeze(0)  # (1, 1, K_len, head_dim)
            k_cosine = k_cosine.unsqueeze(0).unsqueeze(0)
            projected_key = self.rope.apply_rotation(projected_key, k_sine, k_cosine)

        attention_scores = torch.matmul(projected_query, projected_key.transpose(-1, -2)) / (self.head_dimension ** 0.5)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout, training=self.training)
        attended_values = torch.matmul(attention_weights, projected_value)
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.view(batch_size, query_sequence_length, self.number_of_heads * self.head_dimension)
        output = self.output_projection(attended_values)
        return output