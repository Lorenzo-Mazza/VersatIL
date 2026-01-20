"""DiT transformer encoder layer."""

import torch
import torch.nn as nn

from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.constants import AttentionType
from refactoring.models.layers.normalization.ada_norm import AdaNorm
from refactoring.models.layers.normalization.constants import NormalizationType
from refactoring.models.layers.normalization.factory import create_normalization_layer
from refactoring.models.layers.transformer.attention import CachedAttention


class DiTEncoderLayer(nn.Module):
    """Single self-attention encoder layer for DiT.

    Similar to standard transformer encoder layer but simpler, designed for
    sequence-first format (S, B, D) used by MultiheadAttention.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        activation_name: str = "gelu",
        normalization_type: str = NormalizationType.ADALN.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
    ) -> None:
        """Initialize the DiTEncoderLayer.

        Args:
            embedding_dim: Dimensionality of the embeddings.
            num_heads: Number of attention heads.
            feedforward_dim: Hidden dimensionality in the feedforward network.
            dropout: Dropout rate.
            activation_name: Name of the activation function.
        """
        super().__init__()

        # Self-attention (custom attention uses batch-first inputs)
        self.self_attention = CachedAttention(
            embedding_dimension=embedding_dim,
            number_of_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
        )

        # Feedforward network
        if activation_name == "gelu":
            activation = nn.GELU(approximate="tanh")
        elif activation_name == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        self.feedforward_network = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
        )

        # Normalization layers
        condition_dim = None
        if normalization_type in (
            NormalizationType.ADALN.value,
            NormalizationType.ADARMS.value,
        ):
            condition_dim = embedding_dim
        self.norm1 = create_normalization_layer(
            normalization_type, embedding_dim, condition_dim=condition_dim
        )
        self.norm2 = create_normalization_layer(
            normalization_type, embedding_dim, condition_dim=condition_dim
        )

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        source_tensor: torch.Tensor,
        positional_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the encoder layer.

        Args:
            source_tensor: Input tensor (sequence_length, batch_size, embedding_dim).
            positional_embedding: Positional embedding (sequence_length, batch_size, embedding_dim).

        Returns:
            Processed tensor (sequence_length, batch_size, embedding_dim).
        """
        # Add positional embedding to queries and keys for self-attention
        query = key = source_tensor + positional_embedding
        condition = positional_embedding.mean(dim=0)

        # Custom attention expects batch-first: (B, S, D)
        query_bf = query.transpose(0, 1)
        key_bf = key.transpose(0, 1)
        value_bf = source_tensor.transpose(0, 1)
        attended_bf, _ = self.self_attention(
            query_bf, key_bf, value_bf, attention_mask=None
        )  # (B, S, D)
        attended = attended_bf.transpose(0, 1)  # (S, B, D)

        # Residual connection and layer norm after attention
        source_tensor = source_tensor + self.dropout1(attended)
        source_tensor = self._apply_norm(self.norm1, source_tensor, condition)

        # Feedforward subnetwork with residual and norm
        # Linear layers work on last dimension, so we can reshape: (S, B, D) -> (S*B, D) -> (S*B, D) -> (S, B, D)
        seq_len, batch_size, emb_dim = source_tensor.shape
        source_flat = source_tensor.reshape(-1, emb_dim)  # (S*B, D)
        feedforward_flat = self.feedforward_network(source_flat)  # (S*B, D)
        feedforward = feedforward_flat.reshape(
            seq_len, batch_size, emb_dim
        )  # (S, B, D)

        source_tensor = source_tensor + self.dropout2(feedforward)
        source_tensor = self._apply_norm(self.norm2, source_tensor, condition)

        return source_tensor

    def _apply_norm(
        self,
        norm_layer: nn.Module,
        x: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(norm_layer, AdaNorm):
            result = norm_layer(x, condition)
            if isinstance(result, tuple):
                return result[0]
            return result
        return norm_layer(x)

    def reset_parameters(self) -> None:
        """Reset parameters using Xavier uniform initialization."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

