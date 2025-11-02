import copy

import torch
import torch.nn as nn

from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.positional_encoding.base import add_positional_encoding


class TransformerDecoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dimension: int,
            number_of_heads: int,
            feedforward_dimension: int = 2048,
            dropout: float = 0.1,
            activation: str = ActivationFunction.RELU.value,
            normalize_before: bool = False,
    ):
        """Initialize transformer decoder layer.

        Args:
            embedding_dimension: Model embedding dimension.
            number_of_heads: Number of attention heads.
            feedforward_dimension: Dimension of feedforward network.
            dropout: Dropout rate.
            activation: Activation function name from ActivationFunction enum.
            normalize_before: If True, use pre-normalization (norm before attention/FFN).
                             If False, use post-normalization (norm after attention/FFN).
        """
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension, num_heads=number_of_heads, dropout=dropout, batch_first=False
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension, num_heads=number_of_heads, dropout=dropout, batch_first=False
        )
        self.feedforward_linear1 = nn.Linear(embedding_dimension, feedforward_dimension)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_linear2 = nn.Linear(feedforward_dimension, embedding_dimension)
        self.normalization1 = nn.LayerNorm(embedding_dimension)
        self.normalization2 = nn.LayerNorm(embedding_dimension)
        self.normalization3 = nn.LayerNorm(embedding_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = ActivationFunction(activation).to_torch_activation()()


    def forward(
            self,
            target: torch.Tensor,
            memory: torch.Tensor,
            target_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
            target_key_padding_mask: torch.Tensor | None = None,
            memory_key_padding_mask: torch.Tensor | None = None,
            memory_positional_encoding: torch.Tensor | None = None,
            query_positional_encoding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through decoder layer.

        Args:
            target: Target tensor of shape (target_length, batch, embedding_dimension).
            memory: Encoder output of shape (source_length, batch, embedding_dimension), used for cross-attention.
            target_mask: Target attention mask of shape (target_length, target_length).
            memory_mask: Memory attention mask of shape (target_length, source_length).
            target_key_padding_mask: Target padding mask of shape (batch, target_length).
            memory_key_padding_mask: Memory padding mask of shape (batch, source_length).
            memory_positional_encoding: Memory positional encoding of shape (source_length, batch, embedding_dimension).
            query_positional_encoding: Query positional encoding of shape (target_length, batch, embedding_dimension).

        Returns:
            Output tensor of shape (target_length, batch, embedding_dimension).
        """
        residual = target
        target = self.normalization1(target) if self.normalize_before else target
        # DETR-style: Add positional encoding to queries and keys, but not to values
        query = key = add_positional_encoding(target, query_positional_encoding)
        target = self.self_attention(
            query, key, value=target,
            attn_mask=target_mask,
            key_padding_mask=target_key_padding_mask,
        )[0]
        target = residual + self.dropout1(target)
        target = target if self.normalize_before else self.normalization1(target)
        residual = target
        target = self.normalization2(target) if self.normalize_before else target
        target = self.cross_attention(
            query=add_positional_encoding(target, query_positional_encoding),
            key=add_positional_encoding(memory, memory_positional_encoding),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        target = residual + self.dropout2(target)
        target = target if self.normalize_before else self.normalization2(target)
        residual = target
        target = self.normalization3(target) if self.normalize_before else target
        target = self.feedforward_linear2(
            self.feedforward_dropout(
                self.activation(self.feedforward_linear1(target))
            )
        )
        target = residual + self.dropout3(target)
        target = target if self.normalize_before else self.normalization3(target)
        return target




class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers."""
    def __init__(
            self,
            decoder_layer: TransformerDecoderLayer,
            number_of_layers: int,
            normalization: nn.Module | None = None,
            return_intermediate: bool = False,
    ):
        """Initialize transformer decoder.

        Args:
            decoder_layer: Single decoder layer to be stacked.
            number_of_layers: Number of decoder layers.
            normalization: Optional final normalization layer.
            return_intermediate: If True, return outputs from all layers stacked.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(number_of_layers)
        ])
        self.number_of_layers = number_of_layers
        self.normalization = normalization
        self.return_intermediate = return_intermediate


    def forward(
            self,
            target: torch.Tensor,
            memory: torch.Tensor,
            target_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
            target_key_padding_mask: torch.Tensor | None = None,
            memory_key_padding_mask: torch.Tensor | None = None,
            memory_positional_encoding: torch.Tensor | None = None,
            query_positional_encoding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through all decoder layers.

        Args:
            target: Target tensor of shape (target_length, batch, embedding_dimension).
            memory: Encoder output of shape (source_length, batch, embedding_dimension).
            target_mask: Target attention mask of shape (target_length, target_length).
            memory_mask: Memory attention mask of shape (target_length, source_length).
            target_key_padding_mask: Target padding mask of shape (batch, target_length).
            memory_key_padding_mask: Memory padding mask of shape (batch, source_length).
            memory_positional_encoding: Memory positional encoding of shape (source_length, batch, embedding_dimension).
            query_positional_encoding: Query positional encoding of shape (target_length, batch, embedding_dimension).

        Returns:
            If return_intermediate is True, returns tensor of shape (number_of_layers, target_length, batch, embedding_dimension).
            Otherwise, returns tensor of shape (1, target_length, batch, embedding_dimension).
        """
        output = target
        intermediate = []
        for layer in self.layers:
            output = layer(
                output, memory,
                target_mask=target_mask,
                memory_mask=memory_mask,
                target_key_padding_mask=target_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_positional_encoding=memory_positional_encoding,
                query_positional_encoding=query_positional_encoding,
            )
            if self.return_intermediate:
                intermediate.append(self.normalization(output) if self.normalization else output)

        if self.normalization is not None:
            output = self.normalization(output)
            if self.return_intermediate:
                intermediate[-1] = output

        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)
