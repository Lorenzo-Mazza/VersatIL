"""Attention with generation and conditioning cache support."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from versatil.models.layers.constants import AttentionType
from versatil.models.layers.transformer.cache.conditioning import (
    ConditioningLayerCache,
)
from versatil.models.layers.transformer.cache.generation import GenerationLayerCache
from versatil.models.layers.transformer.positional_encoding import (
    apply_rope_positional_encoding,
)


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
        head_dimension: int | None = None,
        dropout: float = 0.0,
        bias: bool = True,
        attention_type: str = AttentionType.MULTI_HEAD.value,
    ):
        """Initialize cached attention module.

        Args:
            embedding_dimension: Model embedding dimension
            number_of_heads: Number of query heads
            number_of_key_value_heads: Number of key/value heads (for GQA)
            head_dimension: Per-head dimension. Defaults to embedding_dimension // number_of_heads.
                Override for architectures where hidden_size != num_heads * head_dim (e.g. Gemma2).
            dropout: Dropout probability for attention weights
            bias: Whether to include bias in projections
            attention_type: Type of attention (use AttentionType enum values)

        Raises:
            ValueError: If dimensions don't match or invalid attention type
        """
        super().__init__()
        if number_of_heads <= 0:
            raise ValueError(
                f"number_of_heads must be positive, got {number_of_heads}."
            )
        if head_dimension is not None and head_dimension <= 0:
            raise ValueError(f"head_dimension must be positive, got {head_dimension}.")
        if head_dimension is None and embedding_dimension % number_of_heads != 0:
            raise ValueError(
                f"embedding_dimension ({embedding_dimension}) must be divisible "
                f"by number_of_heads ({number_of_heads})."
            )
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension: int = (
            head_dimension
            if head_dimension is not None
            else embedding_dimension // number_of_heads
        )
        self.dropout = dropout
        self.attention_type = attention_type
        if attention_type == AttentionType.GROUPED_QUERY.value:
            if number_of_key_value_heads is None:
                raise ValueError("number_of_key_value_heads required for GQA")
            if number_of_key_value_heads <= 0:
                raise ValueError(
                    "number_of_key_value_heads must be positive, "
                    f"got {number_of_key_value_heads}."
                )
            if number_of_heads % number_of_key_value_heads != 0:
                raise ValueError(
                    f"number_of_heads ({number_of_heads}) must be divisible "
                    f"by number_of_key_value_heads ({number_of_key_value_heads})."
                )
            self.number_of_key_value_heads = number_of_key_value_heads
            self.group_size = number_of_heads // number_of_key_value_heads
        elif attention_type == AttentionType.MULTI_HEAD.value:
            if (
                number_of_key_value_heads is not None
                and number_of_key_value_heads != number_of_heads
            ):
                raise ValueError(
                    "number_of_key_value_heads must be None or equal to "
                    "number_of_heads for multi-head attention, got "
                    f"{number_of_key_value_heads}."
                )
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
        self.output_projection.SQUARE_ROOT_WEIGHT = (
            True  # Flag for initialization (GPT2 style)
        )

    def compute_query(self, query_input: torch.Tensor) -> torch.Tensor:
        """Project and reshape query input.

        Args:
            query_input: (B, query_len, D)

        Returns:
            Projected queries (B, num_heads, query_len, head_dim).
        """
        batch_size, query_length, _ = query_input.shape
        projected = self.query_projection(query_input)
        # (B, L, num_heads * head_dim) -> (B, num_heads, L, head_dim)
        return projected.view(
            batch_size, query_length, self.number_of_heads, self.head_dimension
        ).transpose(1, 2)

    def compute_key(self, key_input: torch.Tensor) -> torch.Tensor:
        """Project and reshape key input.

        Args:
            key_input: (B, key_len, D)

        Returns:
            Projected keys (B, kv_heads, key_len, head_dim).
        """
        batch_size, key_length, _ = key_input.shape
        projected = self.key_projection(key_input)
        # (B, L, kv_heads * head_dim) -> (B, kv_heads, L, head_dim)
        return projected.view(
            batch_size, key_length, self.number_of_key_value_heads, self.head_dimension
        ).transpose(1, 2)

    def compute_value(self, value_input: torch.Tensor) -> torch.Tensor:
        """Project and reshape value input.

        Args:
            value_input: (B, value_len, D)

        Returns:
            Projected values (B, kv_heads, value_len, head_dim).
        """
        batch_size, value_length, _ = value_input.shape
        projected = self.value_projection(value_input)
        # (B, L, kv_heads * head_dim) -> (B, kv_heads, L, head_dim)
        return projected.view(
            batch_size,
            value_length,
            self.number_of_key_value_heads,
            self.head_dimension,
        ).transpose(1, 2)

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
            Tuple of (queries, keys, values). Queries: (B, num_heads, query_len, head_dim).
            Keys/values: (B, kv_heads, key_len, head_dim).
        """
        return (
            self.compute_query(query_input),
            self.compute_key(key_input),
            self.compute_value(value_input),
        )

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
        if self.group_size > 1:  # For GQ attention
            keys = torch.repeat_interleave(
                keys, self.group_size, dim=1
            )  # (B, num_heads, kv_length, head_dim)
            values = torch.repeat_interleave(
                values, self.group_size, dim=1
            )  # (B, num_heads, kv_length, head_dim)

        sdpa_mask = None
        if attention_mask is not None:
            sdpa_mask = (
                ~attention_mask if attention_mask is not None else None
            )  # False means don't attend/padded
            # cf. https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

        attended_values = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.head_dimension**-0.5,
        )
        attended_values = attended_values.transpose(
            1, 2
        ).contiguous()  # (B, query_len, num_heads, head_dim)
        attended_values = attended_values.view(
            batch_size,
            query_length,
            self.number_of_heads * self.head_dimension,  # (B, query_len, embedding_dim)
        )
        output = self.output_projection(attended_values)
        return output

    def forward(
        self,
        query_input: torch.Tensor,
        key_input: torch.Tensor | None = None,
        value_input: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        generation_cache: GenerationLayerCache | None = None,
        positional_encoding: nn.Module | None = None,
        conditioning_cache: ConditioningLayerCache | None = None,
    ) -> tuple[torch.Tensor, GenerationLayerCache | None]:
        """Forward pass with optional generation and conditioning caches.

        Args:
            query_input: Query input (B, query_len, D).
            key_input: Key input (B, key_len, D). None when using conditioning_cache.
            value_input: Value input (B, value_len, D). None when using conditioning_cache.
            attention_mask: Bool mask (B, 1, query_len, key_len), True = masked.
            generation_cache: Cached K/V from the main sequence. When provided,
                an updated cache is returned.
            positional_encoding: Optional RoPE module.
            conditioning_cache: Precomputed K/V for static conditioning. When present,
                key_input/value_input are ignored and cached K/V is used directly.

        Returns:
            Tuple of (output (B, query_len, D), updated GenerationLayerCache or None).
        """
        if conditioning_cache is not None:
            queries = self.compute_query(query_input)
            keys = conditioning_cache.keys
            values = conditioning_cache.values
        else:
            if key_input is None or value_input is None:
                raise ValueError(
                    "key_input and value_input required when conditioning_cache is not provided"
                )

            queries, keys, values = self.compute_query_key_value(
                query_input, key_input, value_input
            )
            cache_position = 0
            if generation_cache is not None and generation_cache.keys.numel() > 0:
                cache_position = generation_cache.get_length()
            # Apply RoPE before concatenation so cached keys retain original rotations
            if positional_encoding is not None:
                queries, keys = apply_rope_positional_encoding(
                    queries=queries,
                    keys=keys,
                    positional_encoding=positional_encoding,
                    cache_position=cache_position,
                )
            if generation_cache is not None and generation_cache.keys.numel() > 0:
                keys = torch.cat([generation_cache.keys, keys], dim=2)
                values = torch.cat([generation_cache.values, values], dim=2)

        output = self.compute_attention(queries, keys, values, attention_mask)

        new_cache = None
        if generation_cache is not None and conditioning_cache is None:
            new_cache = GenerationLayerCache(keys=keys, values=values)

        return output, new_cache
