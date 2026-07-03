"""Individual building blocks for composing action heads."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.mlp import MLP
from versatil.models.layers.normalization.ada_norm import AdaNorm


class ActionHeadBlock(nn.Module, ABC):
    """Abstract base class for action head building blocks.

    Action head blocks are modular components that can be composed together
    to create complex action prediction heads. Each block processes embeddings
    and outputs tensors with the same shape.

    """

    @abstractmethod
    def forward(self, action_embedding: torch.Tensor) -> torch.Tensor:
        """Process embeddings through this block.

        Args:
            action_embedding: Input tensor (B, prediction horizon, embedding_dimension) or (B, embedding_dimension)

        Returns:
            Processed tensor with same shape as input
        """
        raise NotImplementedError


class LayerNormBlock(ActionHeadBlock):
    """Layer-normalization block for action heads."""

    def __init__(self, input_dimension: int) -> None:
        """Initialize the layer-normalization block.

        Args:
            input_dimension: Input and output feature dimension.
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dim = input_dimension
        self.norm = nn.LayerNorm(input_dimension)

    def forward(self, action_embedding: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization."""
        return self.norm(action_embedding)


class MLPBlock(ActionHeadBlock):
    """Multi-layer perceptron block for action heads.

    This block applies layer normalization followed by an MLP with configurable
    hidden layers, activation function, and dropout.
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimensions: list[int] | None = None,
        output_dim: int | None = None,
        activation: str = ActivationFunction.GELU.value,
        dropout: float = 0.0,
        normalization: bool = True,
    ) -> None:
        """Initialize MLP block.

        Args:
            input_dimension: Input dimension
            hidden_dimensions: List of hidden dimensions
            output_dim: Output dimension (None to keep same as last hidden)
            activation: Activation function name
            dropout: Dropout rate
            normalization: Whether to apply layer normalization before MLP
        """
        super().__init__()
        if output_dim is None and not hidden_dimensions:
            raise ValueError(
                "Either output_dim or hidden_dimensions must be specified."
            )
        self.input_dimension = input_dimension
        self.output_dim = output_dim or hidden_dimensions[-1]
        self.norm = nn.LayerNorm(input_dimension) if normalization else nn.Identity()

        self.mlp = MLP(
            input_dimension=input_dimension,
            hidden_dimensions=hidden_dimensions,
            output_dim=output_dim,
            activation_function=ActivationFunction(activation).to_torch_activation(),
            dropout=dropout,
        )

    def forward(self, action_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass through normalized MLP.

        Args:
            action_embedding: Input tensor (B, prediction horizon, embedding_dimension) or (B, embedding_dimension)

        Returns:
            Output tensor with same shape
        """
        result: torch.Tensor = self.mlp(self.norm(action_embedding))
        return result


class AttentionBlock(ActionHeadBlock):
    """Self-attention block for action heads with residual connection.

    This block applies layer normalization, self-attention, and adds a residual
    connection. Useful for allowing action tokens to attend to each other across
    the prediction horizon.
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int = 8,
        dropout: float = 0.0,
        normalization: bool = True,
    ) -> None:
        """Initialize attention block.

        Args:
            embedding_dimension: Embedding dimension
            number_of_heads: Number of attention heads
            dropout: Dropout rate
            normalization: Whether to apply layer normalization
        """
        super().__init__()
        self.norm = (
            nn.LayerNorm(embedding_dimension) if normalization else nn.Identity()
        )
        self.input_dimension = embedding_dimension
        self.output_dim = embedding_dimension
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=number_of_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, action_embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            action_embedding: Input (B, prediction horizon, embedding_dimension)

        Returns:
            Output with residual (B, prediction horizon, embedding_dimension)
        """
        normed = self.norm(action_embedding)
        attn_out, _ = self.attention(normed, normed, normed)
        result: torch.Tensor = action_embedding + self.dropout(attn_out)
        return result


class ResidualBlock(ActionHeadBlock):
    """Residual block wrapper for any ActionHeadBlock.

    Wraps another block and adds a residual connection around it.
    """

    def __init__(self, block: ActionHeadBlock, dropout: float = 0.0) -> None:
        """Initialize residual block.

        Args:
            block: Block to wrap with residual connection
            dropout: Dropout rate after block
        """
        super().__init__()
        self.block = block
        self.input_dimension = block.input_dimension
        self.output_dim = block.output_dim
        if self.input_dimension != self.output_dim:
            raise ValueError(
                "Input and output dimensions must match for ResidualBlock."
            )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, action_embedding: torch.Tensor) -> torch.Tensor:
        """Forward with residual connection.

        Args:
            action_embedding: Input tensor

        Returns:
            action_embedding + dropout(block(action_embedding))
        """
        result: torch.Tensor = action_embedding + self.dropout(
            self.block(action_embedding)
        )
        return result


class ConditionalActionHeadBlock(nn.Module, ABC):
    """Abstract base class for action-head blocks with a conditioning input."""

    @abstractmethod
    def forward(
        self,
        action_embedding: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Process action embeddings with a conditioning vector."""
        raise NotImplementedError


class AdaNormBlock(ConditionalActionHeadBlock):
    """Adaptive layer-normalization block for conditional action heads."""

    def __init__(
        self,
        input_dimension: int,
        conditioning_dimension: int,
        activation: str = ActivationFunction.SILU.value,
    ) -> None:
        """Initialize adaptive normalization.

        Args:
            input_dimension: Action embedding feature dimension.
            conditioning_dimension: Conditioning vector dimension.
            activation: Activation used inside the modulation projection.
        """
        super().__init__()
        self.input_dimension = input_dimension
        self.output_dim = input_dimension
        base_norm = nn.LayerNorm(
            input_dimension,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.ada_norm = AdaNorm(
            base_norm=base_norm,
            conditioning_dimension=conditioning_dimension,
            feature_dim=input_dimension,
            use_gate=False,
            activation=activation,
        )

    def forward(
        self,
        action_embedding: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive normalization."""
        modulated_embedding, _ = self.ada_norm(action_embedding, condition)
        return modulated_embedding
