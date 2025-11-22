"""Based on "The Free Transformer" (Fleuret, 2025) - arXiv:2510.17558
https://arxiv.org/abs/2510.17558

The Free Transformer extends decoder transformers by injecting learnable latent
variables into the middle layer, enabling conditional generation through a variational
autoencoder framework.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from refactoring.models.layers.normalization.rms_norm import RMSNorm
from refactoring.models.layers.binary_mapper import BinaryMapper
from refactoring.models.layers.positional_encoding.rotary import RotaryPositionalEncoding1D
from refactoring.models.layers.group_query_attention import GroupQueryAttention
from refactoring.models.layers.activation import ActivationFunction


class LatentConditionedTransformerBlock(nn.Module):
    """Transformer block with latent variable injection into K and V.

    Latent variables are projected and added to features before K/V projections,
    following the paper's specification: K, V = proj(X_{L/2} + R) where R = proj(Z).
    Query Q is computed from plain X_{L/2}.

    Note: Uses manual Q/K/V projections with RoPE (not GroupQueryAttention module)
    because latent conditioning requires adding latent projection before K/V computation.

    Args:
        embedding_dimension: Model embedding dimension
        number_of_heads: Number of query attention heads
        number_of_key_value_heads: Number of key-value heads (for GQA)
        feedforward_dimension: FFN hidden dimension
        latent_dim: Dimension of latent codes
        dropout: Dropout rate
        causal: Whether to use causal (autoregressive) attention mask
        use_rope: Whether to apply manual RoPE to Q and K
        rope_base: Base frequency for RoPE
        norm_layer: Normalization layer class
        activation: Activation function for feedforward network
    """

    def __init__(
        self,
        embedding_dimension: int = 256,
        number_of_heads: int = 8,
        number_of_key_value_heads: int = 2,
        feedforward_dimension: int = 1024,
        latent_dim: int = 65536,
        dropout: float = 0.1,
        causal: bool = False,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        norm_layer: type[nn.Module] = RMSNorm,
        activation: ActivationFunction = ActivationFunction.SWIGLU,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.number_of_key_value_heads = number_of_key_value_heads
        self.head_dim = embedding_dimension // number_of_heads
        self.latent_dim = latent_dim
        self.causal = causal
        self.use_rope = use_rope

        if embedding_dimension % number_of_heads != 0:
            raise ValueError(f"embedding_dimension must be divisible by number_of_heads")
        if number_of_heads % number_of_key_value_heads != 0:
            raise ValueError(f"number_of_heads must be divisible by number_of_key_value_heads")

        self.latent_proj = nn.Linear(latent_dim, embedding_dimension, bias=False)

        if use_rope:
            self.rope = RotaryPositionalEncoding1D(
                embedding_dimension=embedding_dimension,
                num_heads=number_of_heads,
                base_frequency=rope_base,
            )

        self.q_proj = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.k_proj = nn.Linear(embedding_dimension, number_of_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dimension, number_of_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.group_size = number_of_heads // number_of_key_value_heads

        activation_class = activation.to_torch_activation()
        if activation == ActivationFunction.SWIGLU:
            self.feedforward = nn.Sequential(
                activation_class(input_dim=embedding_dimension, hidden_dim=feedforward_dimension, bias=False),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=False),
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(embedding_dimension, feedforward_dimension, bias=False),
                activation_class(),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=False),
            )

        self.norm1 = norm_layer(embedding_dimension)
        self.norm2 = norm_layer(embedding_dimension)

        self.dropout = nn.Dropout(dropout)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with latent-conditioned attention.

        Args:
            x: Input tensor (B, T, embedding_dimension)
            latent: Latent codes (B, T, latent_dim) or (B, 1, latent_dim)
            key_padding_mask: Padding mask (B, T) with True for padding

        Returns:
            Output tensor (B, T, embedding_dimension)
        """
        B, T, D = x.shape

        x_normed = self.norm1(x)
        R = self.latent_proj(latent)
        if R.shape[1] == 1 and T > 1:
            R = R.expand(-1, T, -1)
        Q = self.q_proj(x_normed)
        x_for_kv = x_normed + R
        K = self.k_proj(x_for_kv)
        V = self.v_proj(x_for_kv)
        Q = Q.view(B, T, self.number_of_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        K = K.view(B, T, self.number_of_key_value_heads, self.head_dim).transpose(1, 2)  # (B, num_kv_heads, T, head_dim)
        V = V.view(B, T, self.number_of_key_value_heads, self.head_dim).transpose(1, 2)  # (B, num_kv_heads, T, head_dim)

        if self.group_size > 1:
            K = torch.repeat_interleave(K, self.group_size, dim=1)  # (B, num_heads, T, head_dim)
            V = torch.repeat_interleave(V, self.group_size, dim=1)  # (B, num_heads, T, head_dim)

        if self.use_rope:
            sine, cosine = self.rope.compute_rotation_components(T)
            sine = sine.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
            cosine = cosine.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
            Q = self.rope.apply_rotation(Q, sine, cosine)
            K = self.rope.apply_rotation(K, sine, cosine)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.causal:
            causal_mask = self._generate_causal_mask(T, x.device)
            attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float("-inf")
            )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, V)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        attn_out = self.out_proj(attn_out)

        x = x + self.dropout(attn_out)

        x_normed = self.norm2(x)
        ff_out = self.feedforward(x_normed)
        x = x + self.dropout(ff_out)

        return x


class FreeTransformerDecoderBlock(nn.Module):
    """Free transformer decoder block.

    Uses Group Query Attention (GQA) for efficiency with RoPE handled internally.
    Supports configurable normalization and activation functions.

    Args:
        embedding_dimension: Model embedding dimension
        number_of_heads: Number of query attention heads
        number_of_key_value_heads: Number of key-value heads (for GQA). If None, defaults to number_of_heads (standard MHA)
        feedforward_dimension: FFN hidden dimension
        dropout: Dropout rate
        causal: Whether to use causal (autoregressive) attention mask
        rope_base: Base frequency for RoPE
        norm_layer: Normalization layer class
        activation: Activation function for feedforward network
    """

    def __init__(
        self,
        embedding_dimension: int = 256,
        number_of_heads: int = 8,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int = 1024,
        dropout: float = 0.1,
        causal: bool = True,
        rope_base: float = 10000.0,
        norm_layer: type[nn.Module] = RMSNorm,
        activation: ActivationFunction = ActivationFunction.SWIGLU,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.number_of_key_value_heads = number_of_key_value_heads if number_of_key_value_heads is not None else number_of_heads
        self.causal = causal
        self.attention = GroupQueryAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=self.number_of_key_value_heads,
            dropout=dropout,
            bias=False,
            use_rope=True,
            rope_base=rope_base,
        )
        activation_class = activation.to_torch_activation()
        if activation == ActivationFunction.SWIGLU:
            self.feedforward = nn.Sequential(
                activation_class(input_dim=embedding_dimension, hidden_dim=feedforward_dimension, bias=False),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=False),
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(embedding_dimension, feedforward_dimension, bias=False),
                activation_class(),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=False),
            )

        self.norm1 = norm_layer(embedding_dimension)
        self.norm2 = norm_layer(embedding_dimension)
        self.dropout = nn.Dropout(dropout)


    def _generate_causal_mask(self, target_length: int, source_length: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask for possibly different target/source lengths."""
        mask = torch.full((target_length, source_length), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)  # upper -inf, lower/diagonal 0 after fill below

    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with cross-attention or self-attention.

        Args:
            target: Input tensor (B, target sequence length, embedding_dimension)
            memory: Memory tensor coming from encoder (B, memory sequence length, embedding_dimension).
                If None, defaults to target (self-attention mode).
            target_mask: Target attention mask of shape (target_length, target_length).
            memory_mask: Memory attention mask of shape (target_length, source_length).
            memory_key_padding_mask: Target padding mask of shape (batch, target_length).

        Returns:
            Output tensor (B, T, embedding_dimension)
        """
        # If no memory provided, use self-attention
        if memory is None:
            memory = target

        B, T, D = target.shape
        _, S, _ = memory.shape  # Support T != S
        residual = target
        target_normed = self.norm1(target)
        memory_normed = self.norm1(memory)

        attention_mask = None
        if self.causal:
            causal_mask = self._generate_causal_mask(T, S, target.device)  # (T, S)
            attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, S)

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = memory_key_padding_mask.unsqueeze(1).unsqueeze(2).masked_fill(
                memory_key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )  # (B, 1, 1, T)
            if attention_mask is not None:
                attention_mask = attention_mask + memory_key_padding_mask
            else:
                attention_mask = memory_key_padding_mask
        if target_mask is not None:
            target_mask = target_mask.unsqueeze(0).unsqueeze(0)
            attention_mask = attention_mask + target_mask if attention_mask is not None else target_mask
        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(0).unsqueeze(0)
            attention_mask = attention_mask + memory_mask if attention_mask is not None else memory_mask

        attn_out = self.attention(
            query=target_normed,
            key=memory_normed,
            value=memory_normed,
            attention_mask=attention_mask,
        )

        x = residual + self.dropout(attn_out)
        x_normed = self.norm2(x)
        ff_out = self.feedforward(x_normed)
        x = x + self.dropout(ff_out)
        return x


class FreeTransformerEncoderBlock(nn.Module):
    """Encoder block with cross-attention using learned queries.

    Uses learned query embeddings for cross-attention with input features as K/V.
    Non-causal attention for access to full sequence context during training.

    Args:
        embedding_dimension: Model embedding dimension
        number_of_heads: Number of query attention heads
        number_of_key_value_heads: Number of key-value heads (for GQA). If None, defaults to number_of_heads (standard MHA)
        feedforward_dimension: FFN hidden dimension
        dropout: Dropout rate
        causal: Must be False for encoder blocks (cross-attention is always non-causal)
        use_rope: Whether to use Rotary Position Embeddings (handled by GroupQueryAttention)
        rope_base: Base frequency for RoPE
        norm_layer: Normalization layer class
        activation: Activation function for feedforward network
    """

    def __init__(
        self,
        embedding_dimension: int = 256,
        number_of_heads: int = 8,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int = 1024,
        dropout: float = 0.1,
        causal: bool = False,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        norm_layer: type[nn.Module] = RMSNorm,
        activation: ActivationFunction = ActivationFunction.SWIGLU,
    ):
        super().__init__()
        if causal:
            raise ValueError("Encoder blocks must be non-causal for cross-attention")

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.number_of_key_value_heads = number_of_key_value_heads if number_of_key_value_heads is not None else number_of_heads

        self.attention = GroupQueryAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=self.number_of_key_value_heads,
            dropout=dropout,
            bias=False,
            use_rope=use_rope,
            rope_base=rope_base,
        )

        activation_class = activation.to_torch_activation()
        if activation == ActivationFunction.SWIGLU:
            self.feedforward = nn.Sequential(
                activation_class(input_dim=embedding_dimension, hidden_dim=feedforward_dimension, bias=False),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=False),
            )
        else:
            self.feedforward = nn.Sequential(
                nn.Linear(embedding_dimension, feedforward_dimension, bias=False),
                activation_class(),
                nn.Linear(feedforward_dimension, embedding_dimension, bias=False),
            )

        self.norm1 = norm_layer(embedding_dimension)
        self.norm2 = norm_layer(embedding_dimension)
        self.norm_q = norm_layer(embedding_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        queries: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with cross-attention.

        Args:
            x: Input features from mid-decoder (B, T, embedding_dimension)
            queries: Learned queries (B, T, embedding_dimension) from shared parameter
            key_padding_mask: Padding mask (B, T) with True for padding

        Returns:
            Encoded features (B, T, embedding_dimension)
        """
        queries_normed = self.norm_q(queries)
        x_normed = self.norm1(x)

        attention_mask = None
        if key_padding_mask is not None:
            attention_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )  # (B, 1, 1, T)

        attn_out = self.attention(
            query=queries_normed,
            key=x_normed,
            value=x_normed,
            attention_mask=attention_mask,
        )

        out = queries + self.dropout(attn_out)

        out_normed = self.norm2(out)
        ff_out = self.feedforward(out_normed)
        out = out + self.dropout(ff_out)

        return out


class FreeTransformerEncoder(nn.Module):
    """Free Transformer encoder stack for latent prediction during training.

    Stack of encoder blocks with cross-attention followed by binary mapper
    to produce discrete latent codes.

    Args:
        embedding_dimension: Model embedding dimension
        number_of_layers: Number of encoder layers
        number_of_heads: Number of attention heads per layer
        feedforward_dimension: FFN hidden dimension
        latent_bits: Number of bits for latent code
        dropout: Dropout rate
        use_rope: Whether to use RoPE
        rope_base: Base frequency for RoPE
        norm_layer: Normalization layer class
        activation: Activation function for feedforward network
    """

    def __init__(
        self,
        embedding_dimension: int = 256,
        number_of_layers: int = 1,
        number_of_heads: int = 8,
        feedforward_dimension: int = 1024,
        latent_bits: int = 16,
        dropout: float = 0.1,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        norm_layer: type[nn.Module] = RMSNorm,
        activation: ActivationFunction = ActivationFunction.SWIGLU,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.latent_bits = latent_bits
        self.latent_dim = 2**latent_bits
        self.learned_query = nn.Parameter(torch.randn(1, 1, embedding_dimension))
        self.layers = nn.ModuleList([
            FreeTransformerEncoderBlock(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                feedforward_dimension=feedforward_dimension,
                dropout=dropout,
                use_rope=use_rope,
                rope_base=rope_base,
                norm_layer=norm_layer,
                activation=activation,
            )
            for _ in range(number_of_layers)
        ])
        self.norm = norm_layer(embedding_dimension)
        self.binary_mapper = BinaryMapper(
            latent_bits=latent_bits,
            embedding_dimension=embedding_dimension,
        )

    def forward(
        self,
        mid_decoder_features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode mid-decoder features and produce latent codes.

        Args:
            mid_decoder_features: Features from decoder mid-point (B, T, embedding_dimension)
            key_padding_mask: Padding mask (B, T) with True for padding
            deterministic: If True, use deterministic latent sampling

        Returns:
            Tuple of:
                - latent_codes: One-hot latent codes (B, T, latent_dim)
                - logits: Raw logits for KL divergence (B, T, latent_bits)
        """
        x = mid_decoder_features
        B, T, _ = x.shape
        queries = self.learned_query.expand(B, T, -1)

        for layer in self.layers:
            queries = layer(x, queries=queries, key_padding_mask=key_padding_mask)

        queries = self.norm(queries)

        latent_codes, logits = self.binary_mapper(queries, deterministic=deterministic)

        return latent_codes, logits


class FreeTransformerDecoder(nn.Module):
    """Free Transformer decoder with latent injection at middle layer.

    Structure:
    - L/2 self-attention blocks
    - 1 latent-conditioned block at layer L/2+1
    - L/2-1 self-attention blocks

    Args:
        embedding_dimension: Model embedding dimension
        number_of_layers: Total number of layers (must be even)
        number_of_heads: Number of attention heads per layer
        feedforward_dimension: FFN hidden dimension
        latent_dim: Dimension of latent codes
        dropout: Dropout rate
        causal: Whether to use causal (autoregressive) attention mask
        use_rope: Whether to use RoPE (handled by GroupQueryAttention in self-attention blocks, manual in latent block)
        rope_base: Base frequency for RoPE
        norm_layer: Normalization layer class
        activation: Activation function for feedforward network
    """

    def __init__(
        self,
        embedding_dimension: int = 256,
        number_of_layers: int = 6,
        number_of_heads: int = 8,
        feedforward_dimension: int = 1024,
        latent_dim: int = 65536,
        dropout: float = 0.1,
        causal: bool = False,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        norm_layer: type[nn.Module] = RMSNorm,
        activation: ActivationFunction = ActivationFunction.SWIGLU,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.latent_dim = latent_dim

        if number_of_layers % 2 != 0:
            raise ValueError(f"number_of_layers must be even, got {number_of_layers}")

        mid = number_of_layers // 2
        self.pre_latent_layers = nn.ModuleList([
            FreeTransformerDecoderBlock(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                feedforward_dimension=feedforward_dimension,
                dropout=dropout,
                causal=causal,
                rope_base=rope_base,
                norm_layer=norm_layer,
                activation=activation,
            )
            for _ in range(mid)
        ])

        self.latent_block = LatentConditionedTransformerBlock(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            latent_dim=latent_dim,
            dropout=dropout,
            causal=causal,
            use_rope=use_rope,
            rope_base=rope_base,
            norm_layer=norm_layer,
            activation=activation,
        )

        self.post_latent_layers = nn.ModuleList([
            FreeTransformerDecoderBlock(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                feedforward_dimension=feedforward_dimension,
                dropout=dropout,
                causal=causal,
                rope_base=rope_base,
                norm_layer=norm_layer,
                activation=activation,
            )
            for _ in range(mid - 1)
        ])

        self.norm = norm_layer(embedding_dimension)

    def forward_to_mid(
        self,
        source: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run first L/2 layers to get mid-layer features.

        Args:
            source: Input tensor (B, T, embedding_dimension)
            memory : Optional memory tensor (B, S, embedding_dimension) for cross-attention
            key_padding_mask: Padding mask (B, T) with True for padding

        Returns:
            Mid-layer features (B, T, embedding_dimension)
        """
        for layer in self.pre_latent_layers:
            source = layer(target=source, memory=memory, memory_key_padding_mask=key_padding_mask)
        return source

    def forward_from_mid(
        self,
        mid_features: torch.Tensor,
        latent: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Continue decoding from mid-layer with latent conditioning.

        Args:
            mid_features: Features from mid-layer (B, T, embedding_dimension)
            latent: Latent codes (B, T, latent_dim) or (B, 1, latent_dim)
            memory : Optional memory tensor (B, S, embedding_dimension) for cross-attention
            key_padding_mask: Padding mask (B, T) with True for padding

        Returns:
            Decoded features (B, T, embedding_dimension)
        """
        x = self.latent_block(mid_features, latent=latent, key_padding_mask=key_padding_mask)
        for layer in self.post_latent_layers:
            if memory is None:
                memory = x
            x = layer(target=x, memory=memory, memory_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_mid_features: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode with latent conditioning at middle layer.

        Args:
            x: Input tensor (B, T, embedding_dimension)
            latent: Latent codes (B, T, latent_dim) or (B, 1, latent_dim)
            memory : Optional memory tensor (B, S, embedding_dimension) for cross-attention
            key_padding_mask: Padding mask (B, T) with True for padding
            return_mid_features: If True, return features before latent injection

        Returns:
            Tuple of:
                - Decoded features (B, T, embedding_dimension)
                - Mid-layer features if return_mid_features=True, else None
        """
        mid_features = self.forward_to_mid(source=x, memory=memory, key_padding_mask=key_padding_mask)
        mid_features_out = mid_features if return_mid_features else None
        out = self.forward_from_mid(mid_features=mid_features, memory=memory, latent=latent, key_padding_mask=key_padding_mask)
        return out, mid_features_out