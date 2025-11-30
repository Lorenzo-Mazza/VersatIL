import copy
import math

import torch
import torch.nn as nn

from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.detr_transformer.attention import FlashAttention
from refactoring.models.layers.normalization.ada_norm import AdaNorm
from refactoring.models.layers.normalization.rms_norm import RMSNorm

RESIDUAL_STREAM_FLAG = "SQUARE_ROOT_WEIGHT"


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
        self.self_attention = FlashAttention(embedding_dimension=embedding_dimension,
                                        number_of_heads=number_of_heads, dropout=dropout)
        self.cross_attention = FlashAttention(embedding_dimension=embedding_dimension,
                                        number_of_heads=number_of_heads, dropout=dropout)
        self.feedforward_linear1 = nn.Linear(embedding_dimension, feedforward_dimension)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.feedforward_linear2 = nn.Linear(feedforward_dimension, embedding_dimension)
        self.normalization1 = nn.LayerNorm(embedding_dimension)
        self.normalization2 = nn.LayerNorm(embedding_dimension)
        self.normalization3 = nn.LayerNorm(embedding_dimension)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        if activation == ActivationFunction.SWIGLU.value:
            self.activation = ActivationFunction(activation).to_torch_activation()(
                input_dim=embedding_dimension, hidden_dim=feedforward_dimension)
            self.feedforward_network = nn.Sequential(
                self.activation,
                self.feedforward_dropout,
                self.feedforward_linear2,
            )
        else:
            self.activation = ActivationFunction(activation).to_torch_activation()()
            self.feedforward_network = nn.Sequential(
                self.feedforward_linear1,
                self.activation,
                self.feedforward_dropout,
                self.feedforward_linear2,
            )
        self.feedforward_linear2.SQUARE_ROOT_WEIGHT = True  # Flag for initialization (GPT-2 style)


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

        Returns:
            Output tensor of shape (batch size, target_length, embedding_dimension).
        """
        residual = target
        target = self.self_attention(
            query=target,
            key=target,
            value=target,
            query_positional_encoding=query_positional_encoding,
            attention_mask=target_mask,
            key_padding_mask=target_key_padding_mask,
        )
        target = residual + self.dropout1(target)
        target = target if self.normalize_before else self.normalization1(target)
        residual = target
        target = self.normalization2(target) if self.normalize_before else target
        target = self.cross_attention(
            query=target,
            key=memory,
            value=memory,
            query_positional_encoding=query_positional_encoding,
            key_positional_encoding=memory_positional_encoding,
            attention_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        target = residual + self.dropout2(target)
        target = target if self.normalize_before else self.normalization2(target)
        residual = target
        target = self.normalization3(target) if self.normalize_before else target
        target = self.feedforward_network(target)
        target = residual + self.dropout3(target)
        target = target if self.normalize_before else self.normalization3(target)
        return target # (B, target_length, C)


def generate_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """Generate causal attention mask.

    Args:
        size: Sequence length
        device: Device for tensor

    Returns:
        Causal mask (size, size) as boolean tensor where True means masked position
    """
    # Return boolean mask: True for positions that should be masked (future positions)
    mask = torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)
    return mask


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers."""
    def __init__(
            self,
            decoder_layer: TransformerDecoderLayer,
            number_of_layers: int,
            normalization: nn.Module | None = None,
            return_intermediate: bool = False,
            initializer_range: float = 0.02,
    ):
        """Initialize transformer decoder.

        Args:
            decoder_layer: Single decoder layer to be stacked.
            number_of_layers: Number of decoder layers.
            normalization: Optional final normalization layer.
            return_intermediate: If True, return outputs from all layers stacked.
            initializer_range: Standard deviation for weight initialization.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(number_of_layers)
        ])
        self.number_of_layers = number_of_layers
        self.normalization = normalization
        self.return_intermediate = return_intermediate
        self.initializer_range = initializer_range
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        # > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        # > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        if hasattr(module, RESIDUAL_STREAM_FLAG):  # Residual stream correction
            num_norm_layers = 3 # Two attention + one FFN normalization layers
            std = self.initializer_range / math.sqrt(num_norm_layers * self.number_of_layers)
        else:
            std = self.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm, AdaNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)


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
            target: Target tensor of shape (batch size, target_length, embedding_dimension).
            memory: Encoder output of shape (batch size, source_length, embedding_dimension).
            target_mask: Target attention mask of shape (target_length, target_length).
            memory_mask: Memory attention mask of shape (target_length, source_length).
            target_key_padding_mask: Target padding mask of shape (batch size, target_length).
            memory_key_padding_mask: Memory padding mask of shape (batch size, source_length).
            memory_positional_encoding: Memory positional encoding of shape (batch size,source_length,embedding_dimension).
            query_positional_encoding: Query positional encoding of shape (batch size,target_length,embedding_dimension).

        Returns:
            If return_intermediate is True, a tensor with shape (number_of_layers, batch_size, target_length,
             embedding_dimension). Otherwise, with shape  (1, batch_size, target_length, embedding_dimension).
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
