"""Attention mechanisms for GPT transformer with KV cache support."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from refactoring.models.layers.constants import AttentionType
from refactoring.models.layers.transformer.kv_cache import LayerKVCache
from refactoring.models.layers.transformer.positional_encoding import apply_rope_positional_encoding


class CachedAttention(nn.Module):
    """Base attention module with KV cache support.

    Supports both Multi-Head Attention (MHA) and Grouped Query Attention (GQA).
    Can be used for self-attention or cross-attention.
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        number_of_key_value_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        attention_type: str = AttentionType.MULTI_HEAD.value,
    ):
        """Initialize cached attention module.

        Args:
            embedding_dimension: Model embedding dimension
            number_of_heads: Number of query heads
            number_of_key_value_heads: Number of key/value heads (for GQA)
            dropout: Dropout probability for attention weights
            bias: Whether to include bias in projections
            attention_type: Type of attention (use AttentionType enum values)

        Raises:
            ValueError: If dimensions don't match or invalid attention type
        """
        super().__init__()
        if embedding_dimension % number_of_heads != 0:
            raise ValueError(
                f"embedding_dimension ({embedding_dimension}) must be divisible "
                f"by number_of_heads ({number_of_heads})"
            )
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = embedding_dimension // number_of_heads
        self.dropout = dropout
        self.attention_type = attention_type
        if attention_type == AttentionType.GROUPED_QUERY.value:
            if number_of_key_value_heads is None:
                raise ValueError("number_of_key_value_heads required for GQA")
            if number_of_heads % number_of_key_value_heads != 0:
                raise ValueError(
                    f"number_of_heads ({number_of_heads}) must be divisible "
                    f"by number_of_key_value_heads ({number_of_key_value_heads})"
                )
            self.number_of_key_value_heads = number_of_key_value_heads
            self.group_size = number_of_heads // number_of_key_value_heads
        elif attention_type == AttentionType.MULTI_HEAD.value:
            self.number_of_key_value_heads = number_of_heads
            self.group_size = 1
        else:
            raise ValueError(
                f"Unsupported attention type: {attention_type}. "
                f"Must be one of {[e.value for e in AttentionType]}."
            )
        self.query_projection = nn.Linear(
            embedding_dimension,
            number_of_heads * self.head_dimension,
            bias=bias,
        )
        self.key_projection = nn.Linear(
            embedding_dimension,
            self.number_of_key_value_heads * self.head_dimension,
            bias=bias,
        )
        self.value_projection = nn.Linear(
            embedding_dimension,
            self.number_of_key_value_heads * self.head_dimension,
            bias=bias,
        )
        self.output_projection = nn.Linear(
            number_of_heads * self.head_dimension,
            embedding_dimension,
            bias=bias,
        )
        self.output_projection.SQUARE_ROOT_WEIGHT = True # Flag for initialization (GPT2 style)

    def compute_query_key_value(
        self,
        query_input: torch.Tensor,
        key_input: torch.Tensor,
        value_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project inputs to query, key, value.

        Args:
            query_input: Query input (B, query_len, D)
            key_input: Key input (B, key_len, D)
            value_input: Value input (B, value_len, D)

        Returns:
            Tuple of (queries, keys, values). Queries have shape (B, num_heads, seq_len, head_dim).
            For GQA, keys/values have compact shape (B, kv_heads, seq_len, head_dim) for efficient caching.
        """
        batch_size = query_input.shape[0]
        query_length = query_input.shape[1]
        key_length = key_input.shape[1]
        projected_query = self.query_projection(query_input)
        projected_key = self.key_projection(key_input)
        projected_value = self.value_projection(value_input)
        # Reshape: (B, L, num_heads * head_dim) -> (B, L, num_heads, head_dim)
        projected_query = projected_query.view(
            batch_size, query_length, self.number_of_heads, self.head_dimension
        )
        projected_key = projected_key.view(
            batch_size, key_length, self.number_of_key_value_heads, self.head_dimension
        )
        projected_value = projected_value.view(
            batch_size, key_length, self.number_of_key_value_heads, self.head_dimension
        )
        # Transpose: (B, length, num_heads, head_dim) -> (B, num_heads, length, head_dim)
        projected_query = projected_query.transpose(1, 2)
        projected_key = projected_key.transpose(1, 2)
        projected_value = projected_value.transpose(1, 2)
        return projected_query, projected_key, projected_value


    def compute_attention(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Args:
            queries: Query tensor (B, num_heads, query_len, head_dim)
            keys: Key tensor (B, kv_heads, key_len, head_dim) - compact for GQA
            values: Value tensor (B, kv_heads, value_len, head_dim) - compact for GQA
            attention_mask: Optional bool mask (B, 1, query_len, key_len) where True means masked.

        Returns:
            Attention output (B, query_len, embedding_dim)
        """
        batch_size = queries.shape[0]
        query_length = queries.shape[2]
        if self.group_size > 1: # For GQ attention
            keys = torch.repeat_interleave(keys, self.group_size, dim=1) # (B, num_heads, kv_length, head_dim)
            values = torch.repeat_interleave(values, self.group_size, dim=1) # (B, num_heads, kv_length, head_dim)

        sdpa_mask = None
        if attention_mask is not None:
            mask_shape = (batch_size, self.number_of_heads, query_length, keys.shape[2])  #(B, num_heads, query_len, key_len)
            sdpa_mask = torch.full(mask_shape, False, dtype=torch.bool, device=queries.device)
            sdpa_mask = sdpa_mask.masked_fill_(attention_mask, True)  # Broadcast over num_heads
            sdpa_mask = ~sdpa_mask # False means don't attend/padded
            # cf. https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

        attended_values = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.head_dimension ** -0.5,
        )
        attended_values = attended_values.transpose(1, 2).contiguous()  # (B, query_len, num_heads, head_dim)
        attended_values = attended_values.view(
            batch_size, query_length, self.number_of_heads * self.head_dimension  # (B, query_len, embedding_dim)
        )
        output = self.output_projection(attended_values)
        return output

    def forward(
        self,
        query_input: torch.Tensor,
        key_input: torch.Tensor | None = None,
        value_input: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        layer_cache: LayerKVCache | None = None,
        use_self_attention_cache: bool = False,
        positional_encoding: nn.Module | None = None,
        use_cross_attention_cache: bool = False,
    ) -> tuple[torch.Tensor, LayerKVCache | None]:
        """Forward pass with optional KV caching.

        Args:
            query_input: Query input (B, query_len, D)
            key_input: Key input (B, key_len, D). If None and use_cross_attention_cache=True,
             uses precomputed K from layer_cache
            value_input: Value input (B, value_len, D). If None and use_cross_attention_cache=True,
             uses precomputed V from layer_cache
            attention_mask: Optional attention mask (B,  1, query_len, key_len) where True means masked.
             If None, no masking is applied.
            layer_cache: Optional cached keys/values
            use_self_attention_cache: Whether to return updated cache for self-attention
            positional_encoding: Optional positional encoding module (for RoPE)
            use_cross_attention_cache: If True, use precomputed cross-attention K/V from cache

        Returns:
            Tuple of (attention_output, updated_cache), where attention_output has shape (B, query_len, D) and
            updated_cache is a LayerKVCache or None.
        """
        if use_cross_attention_cache and layer_cache is not None:
            if layer_cache.cross_attention_keys is None or layer_cache.cross_attention_values is None:
                raise ValueError("layer_cache must contain precomputed cross_attention K/V when use_cross_attention_cache=True")
            # Use precomputed cross K/V, only project queries
            queries = self.query_projection(query_input)
            batch_size = query_input.shape[0]
            query_length = query_input.shape[1]
            queries = queries.view(batch_size, query_length, self.number_of_heads, self.head_dimension)
            queries = queries.transpose(1, 2)
            keys = layer_cache.cross_attention_keys
            values = layer_cache.cross_attention_values
        else:
            if key_input is None or value_input is None:
                raise ValueError("key_input and value_input required when not using cross_attention_cache")

            queries, keys, values = self.compute_query_key_value(
                query_input, key_input, value_input
            )
            cache_position = 0
            if layer_cache is not None and layer_cache.self_attention_keys.numel() > 0:
                cache_position = layer_cache.get_length()
            # Apply RoPE positional encoding BEFORE concatenation
            # This ensures cached keys retain their original rotations
            if positional_encoding is not None:
                queries, keys = apply_rope_positional_encoding(
                    queries=queries,
                    keys=keys,
                    positional_encoding=positional_encoding,
                    cache_position=cache_position,
                )
            if layer_cache is not None and layer_cache.self_attention_keys.numel() > 0:
                keys = torch.cat([layer_cache.self_attention_keys, keys], dim=2)
                values = torch.cat([layer_cache.self_attention_values, values], dim=2)

        output = self.compute_attention(queries, keys, values, attention_mask)

        new_cache = None
        if use_self_attention_cache and not use_cross_attention_cache:
            new_cache = LayerKVCache(
                self_attention_keys=keys,
                self_attention_values=values,
                cross_attention_keys=layer_cache.cross_attention_keys if layer_cache is not None else None,
                cross_attention_values=layer_cache.cross_attention_values if layer_cache is not None else None,
            )

        return output, new_cache


