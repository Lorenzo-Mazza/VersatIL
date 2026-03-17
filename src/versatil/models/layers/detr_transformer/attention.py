import torch
import torch.nn.functional as F
from torch import nn

from versatil.models.layers.positional_encoding.base import add_positional_encoding


class FlashAttention(nn.Module):
    def __init__(
        self, embedding_dimension: int, number_of_heads: int, dropout: float = 0.0
    ):
        super().__init__()
        if embedding_dimension % number_of_heads != 0:
            raise ValueError(
                "Attention layer embedding_dimension must be divisible by number_of_heads."
            )
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.head_size = embedding_dimension // number_of_heads
        self.q_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.k_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.v_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.out_proj = nn.Linear(embedding_dimension, embedding_dimension)
        self.out_proj.SQUARE_ROOT_WEIGHT = True  # Flag for initialization (GPT2 style)
        self.dropout = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        query_positional_encoding: torch.Tensor | None = None,
        key_positional_encoding: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the attention layer.

        Note: attention_mask and key_padding_mask contain boolean values where True indicates positions to be masked.
        """
        B, query_length, C = query.shape
        key_length = key.shape[1]
        q = self.q_proj(
            add_positional_encoding(query, query_positional_encoding)
        )  # (B, query_length, embedding_dimension)
        k = self.k_proj(
            add_positional_encoding(key, key_positional_encoding)
        )  # (B, key_length, embedding_dimension)
        v = self.v_proj(value)  # (B, key_length, embedding_dimension)
        q = q.view(B, query_length, self.number_of_heads, self.head_size).transpose(
            1, 2
        )  # (B,query_length, nh, hs)
        k = k.view(B, key_length, self.number_of_heads, self.head_size).transpose(
            1, 2
        )  # (B, key_length, nh, hs)
        v = v.view(B, key_length, self.number_of_heads, self.head_size).transpose(
            1, 2
        )  # (B, key_length, nh, hs)
        mask = None
        if attention_mask is not None or key_padding_mask is not None:
            attn_bool = None
            if attention_mask is not None:
                attn_bool = (
                    attention_mask
                    if attention_mask.dtype == torch.bool
                    else torch.isneginf(attention_mask)
                )
                if attn_bool.dim() == 2:
                    attn_bool = attn_bool.unsqueeze(0)  # (1, query_length, key_length)
            padding_bool = (
                key_padding_mask.bool() if key_padding_mask is not None else None
            )
            if attn_bool is not None and padding_bool is not None:
                mask = torch.zeros(
                    B, query_length, key_length, dtype=torch.bool, device=query.device
                )
                mask |= padding_bool[:, None, :]  # (B, 1, key_length)
                mask |= attn_bool
                mask = mask.unsqueeze(1)  # → [B, 1, query_length, key_length]
            elif padding_bool is not None:
                mask = padding_bool[:, None, None, :]  # → [B, 1, 1, key_length]
            elif attn_bool is not None:
                mask = attn_bool.unsqueeze(1)  # → [1, 1, query_length, key_length]
        if mask is not None:
            mask = ~mask  # bool False means don't attend,
            # cf.https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, nh, query_length, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, query_length, C)
        )  # (B, query_length, embedding_dimension)
        return self.out_proj(y)
