"""ActionHead class for composing blocks into action prediction heads."""

import torch
import torch.nn as nn

from refactoring.models.decoding.action_heads import (
    ActionHeadBlock,
    AttentionBlock,
    MLPBlock,
)


class ActionHead(nn.Module):
    """Composable action head from sequence of blocks.

    Converts decoder embeddings (B, horizon, embedding_dimension) to action predictions
    (B, horizon, action_dim). Action heads are composed of a sequence of blocks
    followed by a final linear projection to the action dimension.

    Example:
        # Attention + MLP head
        head = ActionHead(
            input_dim=256,
            output_dim=3,
            blocks=[
                AttentionBlock(256, num_heads=8),
                MLPBlock(256, hidden_dims=[128])
            ]
        )
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        blocks: list[ActionHeadBlock] | None = None,
    ):
        """Initialize action head.

        Args:
            input_dim: Input embedding dimension from decoder
            output_dim: Output action dimension
            blocks: List of blocks to apply (if None, uses simple linear projection)
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        if blocks is None:
            blocks = []

        self.blocks = nn.ModuleList(blocks)
        self.output_proj = nn.Linear(input_dim, output_dim)


    def forward(self, action_embedding: torch.Tensor) -> torch.Tensor:
        """Convert embeddings to actions.

        Args:
            action_embedding: Decoder embeddings (B,prediction horizon, embedding_dimension) or (B, embedding_dimension)

        Returns:
            Action predictions (B, prediction horizon, action_dim) or (B, action_dim)
        """
        for block in self.blocks:
            action_embedding = block(action_embedding)
        result: torch.Tensor = self.output_proj(action_embedding)
        return result


def create_default_action_head(
    input_dim: int,
    output_dim: int,
    hidden_dim: int | None = None,
    activation: str = "silu",
    dropout: float = 0.1,
) -> ActionHead:
    """Create default action head with single MLP layer.

    This is a convenience factory for creating a simple action head with
    a single hidden layer MLP block.

    Args:
        input_dim: Input embedding dimension from decoder
        output_dim: Output action dimension
        hidden_dim: Hidden layer dimension (defaults to input_dim // 2)
        activation: Activation function name
        dropout: Dropout rate

    Returns:
        ActionHead with single MLP block
    """
    if hidden_dim is None:
        hidden_dim = input_dim // 2

    mlp_block = MLPBlock(
        input_dim=input_dim,
        hidden_dims=[hidden_dim],
        output_dim=input_dim,  # Keep same dim for output projection
        activation=activation,
        dropout=dropout,
        normalization=True,
    )

    return ActionHead(
        input_dim=input_dim,
        output_dim=output_dim,
        blocks=[mlp_block],
    )


def create_mlp_action_head(
    input_dim: int,
    output_dim: int,
    hidden_dims: list[int] | None = None,
    activation: str = "silu",
    dropout: float = 0.1,
) -> ActionHead:
    """Create multi-layer MLP action head.

    This is a convenience factory for creating an action head with
    a multi-layer MLP block.

    Args:
        input_dim: Input embedding dimension from decoder
        output_dim: Output action dimension
        hidden_dims: List of hidden layer dimensions (defaults to [128, 64])
        activation: Activation function name
        dropout: Dropout rate

    Returns:
        ActionHead with multi-layer MLP block
    """
    if hidden_dims is None:
        hidden_dims = [128, 64]

    mlp_block = MLPBlock(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=input_dim,  # Keep same dim for output projection
        activation=activation,
        dropout=dropout,
        normalization=True,
    )

    return ActionHead(
        input_dim=input_dim,
        output_dim=output_dim,
        blocks=[mlp_block],
    )


def create_attention_mlp_head(
    input_dim: int,
    output_dim: int,
    num_heads: int = 8,
    mlp_hidden_dim: int | None = None,
    activation: str = "silu",
    dropout: float = 0.1,
) -> ActionHead:
    """Create attention + MLP action head.

    This is a convenience factory for creating an action head with
    an attention block followed by an MLP block.

    Args:
        input_dim: Input embedding dimension from decoder
        output_dim: Output action dimension
        num_heads: Number of attention heads
        mlp_hidden_dim: MLP hidden dimension (defaults to input_dim // 2)
        activation: Activation function name
        dropout: Dropout rate

    Returns:
        ActionHead with attention and MLP blocks
    """
    if mlp_hidden_dim is None:
        mlp_hidden_dim = input_dim // 2

    attention_block = AttentionBlock(
        embedding_dimension=input_dim,
        num_heads=num_heads,
        dropout=dropout,
        normalization=True,
    )

    mlp_block = MLPBlock(
        input_dim=input_dim,
        hidden_dims=[mlp_hidden_dim],
        output_dim=input_dim,  # Keep same dim for output projection
        activation=activation,
        dropout=dropout,
        normalization=True,
    )

    return ActionHead(
        input_dim=input_dim,
        output_dim=output_dim,
        blocks=[attention_block, mlp_block],
    )
