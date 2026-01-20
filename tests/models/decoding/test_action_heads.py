"""Tests for modular action prediction heads."""
import pytest
import torch

from versatil.models.decoding.action_heads import (
    ActionHead,
    MLPBlock,
    AttentionBlock,
    ResidualBlock,
)


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    """Default batch size."""
    return 4


@pytest.fixture
def prediction_horizon():
    """Default prediction horizon."""
    return 10


@pytest.fixture
def embedding_dimension():
    """Default embedding dimension."""
    return 256


@pytest.fixture
def action_dimension():
    """Default action dimension."""
    return 3


@pytest.mark.unit
class TestMLPBlock:
    """Test MLP building block for action heads."""

    def test_init_basic(self, embedding_dimension):
        """Test basic MLP block initialization."""
        block = MLPBlock(
            input_dim=embedding_dimension,
            hidden_dims=[128],
            output_dim=embedding_dimension,
        )
        assert block.mlp is not None
        assert block.norm is not None

    def test_init_without_normalization(self, embedding_dimension):
        """Test MLP block without normalization."""
        block = MLPBlock(
            input_dim=embedding_dimension,
            hidden_dims=[128],
            output_dim=embedding_dimension,
            normalization=False,
        )
        assert isinstance(block.norm, torch.nn.Identity)

    def test_forward_temporal(self, batch_size, prediction_horizon, embedding_dimension, device):
        """Test forward pass with temporal input (B, T, D)."""
        block = MLPBlock(
            input_dim=embedding_dimension,
            hidden_dims=[128, 64],
            output_dim=embedding_dimension,
            activation="relu",
            dropout=0.1,
        ).to(device)

        x = torch.randn(batch_size, prediction_horizon, embedding_dimension, device=device)
        output = block(x)

        assert output.shape == (batch_size, prediction_horizon, embedding_dimension)
        assert not torch.isnan(output).any()

    def test_forward_flat(self, batch_size, embedding_dimension, device):
        """Test forward pass with flat input (B, D)."""
        block = MLPBlock(
            input_dim=embedding_dimension,
            hidden_dims=[128],
            output_dim=embedding_dimension,
        ).to(device)

        x = torch.randn(batch_size, embedding_dimension, device=device)
        output = block(x)

        assert output.shape == (batch_size, embedding_dimension)
        assert not torch.isnan(output).any()

    def test_multiple_hidden_layers(self, batch_size, embedding_dimension, device):
        """Test MLP with multiple hidden layers."""
        block = MLPBlock(
            input_dim=embedding_dimension,
            hidden_dims=[512, 256, 128],
            output_dim=embedding_dimension,
        ).to(device)

        x = torch.randn(batch_size, embedding_dimension, device=device)
        output = block(x)

        assert output.shape == (batch_size, embedding_dimension)

    def test_different_activations(self, batch_size, embedding_dimension, device):
        """Test MLP with different activation functions."""
        activations = ["relu", "gelu", "silu"]

        for activation in activations:
            block = MLPBlock(
                input_dim=embedding_dimension,
                hidden_dims=[128],
                output_dim=embedding_dimension,
                activation=activation,
            ).to(device)

            x = torch.randn(batch_size, embedding_dimension, device=device)
            output = block(x)

            assert output.shape == (batch_size, embedding_dimension)


@pytest.mark.unit
class TestAttentionBlock:
    """Test self-attention building block for action heads."""

    def test_init_basic(self, embedding_dimension):
        """Test basic attention block initialization."""
        block = AttentionBlock(
            embedding_dimension=embedding_dimension,
            num_heads=8,
        )
        assert block.attention is not None
        assert block.norm is not None
        assert block.dropout is not None

    def test_init_without_normalization(self, embedding_dimension):
        """Test attention block without normalization."""
        block = AttentionBlock(
            embedding_dimension=embedding_dimension,
            num_heads=8,
            normalization=False,
        )
        assert isinstance(block.norm, torch.nn.Identity)

    def test_forward_with_residual(self, batch_size, prediction_horizon, embedding_dimension, device):
        """Test forward pass includes residual connection."""
        block = AttentionBlock(
            embedding_dimension=embedding_dimension,
            num_heads=8,
            dropout=0.1,
        ).to(device)

        x = torch.randn(batch_size, prediction_horizon, embedding_dimension, device=device)
        output = block(x)

        # Output should have same shape due to residual
        assert output.shape == (batch_size, prediction_horizon, embedding_dimension)
        assert not torch.isnan(output).any()

        # Output should be different from input (not identity)
        assert not torch.allclose(output, x)

    def test_different_num_heads(self, batch_size, prediction_horizon, embedding_dimension, device):
        """Test attention with different numbers of heads."""
        for num_heads in [4, 8, 16]:
            block = AttentionBlock(
                embedding_dimension=embedding_dimension,
                num_heads=num_heads,
            ).to(device)

            x = torch.randn(batch_size, prediction_horizon, embedding_dimension, device=device)
            output = block(x)

            assert output.shape == x.shape


@pytest.mark.unit
class TestResidualBlock:
    """Test residual wrapper block for action heads."""

    def test_init_basic(self, embedding_dimension):
        """Test basic residual block initialization."""
        inner_block = MLPBlock(
            input_dim=embedding_dimension,
            hidden_dims=[128],
            output_dim=embedding_dimension,
        )
        block = ResidualBlock(block=inner_block, dropout=0.1)

        assert block.block is not None
        assert block.dropout is not None

    def test_forward_with_residual(self, batch_size, prediction_horizon, embedding_dimension, device):
        """Test that residual connection is added."""
        inner_block = MLPBlock(
            input_dim=embedding_dimension,
            hidden_dims=[128],
            output_dim=embedding_dimension,
        )
        block = ResidualBlock(block=inner_block).to(device)

        x = torch.randn(batch_size, prediction_horizon, embedding_dimension, device=device)
        output = block(x)

        # Output should include residual connection
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_nested_residuals(self, batch_size, embedding_dimension, device):
        """Test nesting multiple residual blocks."""
        block1 = MLPBlock(
            input_dim=embedding_dimension,
            hidden_dims=[128],
            output_dim=embedding_dimension,
        )
        block2 = MLPBlock(
            input_dim=embedding_dimension,
            hidden_dims=[128],
            output_dim=embedding_dimension,
        )

        residual1 = ResidualBlock(block1)
        residual2 = ResidualBlock(block2)

        composite = ResidualBlock(
            torch.nn.Sequential(residual1, residual2)
        ).to(device)

        x = torch.randn(batch_size, embedding_dimension, device=device)
        output = composite(x)

        assert output.shape == x.shape


@pytest.mark.unit
class TestActionHead:
    """Test complete action head composition."""

    def test_init_empty_blocks(self, embedding_dimension, action_dimension):
        """Test action head with no blocks (simple linear projection)."""
        head = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_dimension,
            blocks=None,
        )

        assert len(head.blocks) == 0
        assert head.output_proj is not None
        assert head.output_dim == action_dimension

    def test_init_with_single_block(self, embedding_dimension, action_dimension):
        """Test action head with single MLP block."""
        head = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_dimension,
            blocks=[
                MLPBlock(
                    input_dim=embedding_dimension,
                    hidden_dims=[128],
                    output_dim=embedding_dimension,
                )
            ],
        )

        assert len(head.blocks) == 1

    def test_init_with_multiple_blocks(self, embedding_dimension, action_dimension):
        """Test action head with multiple blocks."""
        head = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_dimension,
            blocks=[
                AttentionBlock(embedding_dimension=embedding_dimension, num_heads=8),
                MLPBlock(
                    input_dim=embedding_dimension,
                    hidden_dims=[128],
                    output_dim=embedding_dimension,
                ),
            ],
        )

        assert len(head.blocks) == 2

    def test_forward_temporal_no_blocks(
        self, batch_size, prediction_horizon, embedding_dimension, action_dimension, device
    ):
        """Test forward with temporal input and no blocks."""
        head = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_dimension,
            blocks=[],
        ).to(device)

        x = torch.randn(batch_size, prediction_horizon, embedding_dimension, device=device)
        output = head(x)

        assert output.shape == (batch_size, prediction_horizon, action_dimension)
        assert not torch.isnan(output).any()

    def test_forward_temporal_with_blocks(
        self, batch_size, prediction_horizon, embedding_dimension, action_dimension, device
    ):
        """Test forward with temporal input and processing blocks."""
        head = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_dimension,
            blocks=[
                MLPBlock(
                    input_dim=embedding_dimension,
                    hidden_dims=[128],
                    output_dim=embedding_dimension,
                    activation="relu",
                    dropout=0.1,
                ),
            ],
        ).to(device)

        x = torch.randn(batch_size, prediction_horizon, embedding_dimension, device=device)
        output = head(x)

        assert output.shape == (batch_size, prediction_horizon, action_dimension)
        assert not torch.isnan(output).any()

    def test_forward_flat(self, batch_size, embedding_dimension, action_dimension, device):
        """Test forward with flat input (B, D)."""
        head = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_dimension,
            blocks=[
                MLPBlock(
                    input_dim=embedding_dimension,
                    hidden_dims=[128],
                    output_dim=embedding_dimension,
                ),
            ],
        ).to(device)

        x = torch.randn(batch_size, embedding_dimension, device=device)
        output = head(x)

        assert output.shape == (batch_size, action_dimension)
        assert not torch.isnan(output).any()

    def test_complex_composition(self, batch_size, prediction_horizon, embedding_dimension, device):
        """Test complex composition with attention + residual MLP."""
        head = ActionHead(
            input_dim=embedding_dimension,
            output_dim=7,  # Position (3) + Quaternion (4)
            blocks=[
                AttentionBlock(
                    embedding_dimension=embedding_dimension,
                    num_heads=8,
                    dropout=0.1,
                ),
                ResidualBlock(
                    MLPBlock(
                        input_dim=embedding_dimension,
                        hidden_dims=[512, 256],
                        output_dim=embedding_dimension,
                        activation="gelu",
                        dropout=0.1,
                    )
                ),
                MLPBlock(
                    input_dim=embedding_dimension,
                    hidden_dims=[128],
                    output_dim=embedding_dimension,
                    activation="relu",
                ),
            ],
        ).to(device)

        x = torch.randn(batch_size, prediction_horizon, embedding_dimension, device=device)
        output = head(x)

        assert output.shape == (batch_size, prediction_horizon, 7)
        assert not torch.isnan(output).any()

    def test_different_output_dimensions(self, batch_size, prediction_horizon, embedding_dimension, device):
        """Test heads with different output dimensions for different action types."""
        output_dims = [3, 4, 1]  # position, quaternion, gripper

        for output_dim in output_dims:
            head = ActionHead(
                input_dim=embedding_dimension,
                output_dim=output_dim,
                blocks=[
                    MLPBlock(
                        input_dim=embedding_dimension,
                        hidden_dims=[128],
                        output_dim=embedding_dimension,
                    )
                ],
            ).to(device)

            x = torch.randn(batch_size, prediction_horizon, embedding_dimension, device=device)
            output = head(x)

            assert output.shape == (batch_size, prediction_horizon, output_dim)

    def test_backward_pass(self, batch_size, prediction_horizon, embedding_dimension, action_dimension, device):
        """Test that gradients flow through action head."""
        head = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_dimension,
            blocks=[
                MLPBlock(
                    input_dim=embedding_dimension,
                    hidden_dims=[128],
                    output_dim=embedding_dimension,
                )
            ],
        ).to(device)

        x = torch.randn(
            batch_size, prediction_horizon, embedding_dimension, device=device, requires_grad=True
        )
        output = head(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


@pytest.mark.unit
class TestActionHeadEdgeCases:
    """Test edge cases and error handling."""

    def test_single_timestep(self, batch_size, embedding_dimension, action_dimension, device):
        """Test with single timestep (horizon=1)."""
        head = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_dimension,
            blocks=[
                MLPBlock(
                    input_dim=embedding_dimension,
                    hidden_dims=[128],
                    output_dim=embedding_dimension,
                )
            ],
        ).to(device)

        x = torch.randn(batch_size, 1, embedding_dimension, device=device)
        output = head(x)

        assert output.shape == (batch_size, 1, action_dimension)

    def test_large_prediction_horizon(self, batch_size, embedding_dimension, action_dimension, device):
        """Test with large prediction horizon."""
        large_horizon = 100
        head = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_dimension,
            blocks=[
                AttentionBlock(embedding_dimension=embedding_dimension, num_heads=8),
            ],
        ).to(device)

        x = torch.randn(batch_size, large_horizon, embedding_dimension, device=device)
        output = head(x)

        assert output.shape == (batch_size, large_horizon, action_dimension)

    def test_batch_size_one(self, embedding_dimension, action_dimension, device):
        """Test with single sample in batch."""
        head = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_dimension,
            blocks=[],
        ).to(device)

        x = torch.randn(1, 10, embedding_dimension, device=device)
        output = head(x)

        assert output.shape == (1, 10, action_dimension)

    def test_zero_dropout(self, batch_size, prediction_horizon, embedding_dimension, action_dimension, device):
        """Test that zero dropout works correctly."""
        head = ActionHead(
            input_dim=embedding_dimension,
            output_dim=action_dimension,
            blocks=[
                MLPBlock(
                    input_dim=embedding_dimension,
                    hidden_dims=[128],
                    output_dim=embedding_dimension,
                    dropout=0.0,
                )
            ],
        ).to(device)

        # In eval mode with zero dropout, outputs should be deterministic
        head.eval()
        x = torch.randn(batch_size, prediction_horizon, embedding_dimension, device=device)

        output1 = head(x)
        output2 = head(x)

        assert torch.allclose(output1, output2)
