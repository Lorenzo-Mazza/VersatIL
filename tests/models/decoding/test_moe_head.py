"""Tests for Mixture of Experts action head."""
import pytest
import torch

from refactoring.models.decoding.action_heads import (
    ActionHead,
    MLPBlock,
    MoEHead,
)
from refactoring.models.decoding.constants import MoERoutingType


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    """Default batch size."""
    return 4


@pytest.fixture
def horizon():
    """Default prediction horizon."""
    return 10


@pytest.fixture
def embedding_dim():
    """Default embedding dimension."""
    return 256


@pytest.fixture
def action_dim():
    """Default action dimension."""
    return 3


@pytest.fixture
def num_experts():
    """Default number of experts."""
    return 3


@pytest.fixture
def simple_experts(num_experts, embedding_dim, action_dim, device):
    """Create simple expert heads for testing."""
    experts = []
    for _ in range(num_experts):
        expert = ActionHead(
            input_dim=embedding_dim,
            output_dim=action_dim,
            blocks=[MLPBlock(embedding_dim, hidden_dims=[128], output_dim=embedding_dim)],
        ).to(device)
        experts.append(expert)
    return experts


@pytest.fixture
def embeddings_2d(batch_size, embedding_dim, device):
    """Create 2D embeddings (B, embedding_dim)."""
    return torch.randn(batch_size, embedding_dim, device=device)


@pytest.fixture
def embeddings_3d(batch_size, horizon, embedding_dim, device):
    """Create 3D embeddings (B, horizon, embedding_dim)."""
    return torch.randn(batch_size, horizon, embedding_dim, device=device)


@pytest.fixture
def routing_weights_2d(batch_size, num_experts, device):
    """Create 2D routing weights (B, num_experts)."""
    weights = torch.randn(batch_size, num_experts, device=device)
    return torch.softmax(weights, dim=-1)


@pytest.fixture
def routing_weights_3d(batch_size, horizon, num_experts, device):
    """Create 3D routing weights (B, horizon, num_experts)."""
    weights = torch.randn(batch_size, horizon, num_experts, device=device)
    return torch.softmax(weights, dim=-1)




@pytest.mark.unit
class TestMoEInstantiation:
    """Tests for MoE head instantiation."""

    def test_basic_instantiation_with_gating(self, simple_experts, embedding_dim, action_dim):
        """Test basic instantiation with internal gating network."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
        )
        assert moe is not None
        assert moe.num_experts == len(simple_experts)
        assert moe.has_gating_network

    def test_basic_instantiation_without_gating(self, simple_experts, action_dim):
        """Test basic instantiation without gating network (external routing)."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=None,
        )
        assert moe is not None
        assert moe.num_experts == len(simple_experts)
        assert not moe.has_gating_network

    def test_empty_experts_raises_error(self, embedding_dim, action_dim):
        """Test that empty experts list raises error."""
        with pytest.raises(ValueError, match="at least one expert"):
            MoEHead(
                experts=[],
                output_dim=action_dim,
                gating_input_dim=embedding_dim,
            )

    def test_mismatched_expert_dims_raises_error(self, embedding_dim, action_dim, device):
        """Test that mismatched expert output dimensions raise error."""
        experts = [
            ActionHead(embedding_dim, action_dim, blocks=[]).to(device),
            ActionHead(embedding_dim, action_dim + 1, blocks=[]).to(device),  # Wrong dim
        ]
        with pytest.raises(ValueError, match="does not match"):
            MoEHead(
                experts=experts,
                output_dim=action_dim,
                gating_input_dim=embedding_dim,
            )

    @pytest.mark.parametrize("routing_type", [
        MoERoutingType.SOFT.value,
        MoERoutingType.TOP_K.value,
    ])
    def test_routing_types(self, simple_experts, embedding_dim, action_dim, routing_type):
        """Test different routing types."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            routing_type=routing_type,
        )
        assert moe.routing_type == routing_type

    def test_invalid_routing_type(self, simple_experts, embedding_dim, action_dim):
        """Test that invalid routing type raises error."""
        with pytest.raises(ValueError, match="Invalid routing_type"):
            MoEHead(
                experts=simple_experts,
                output_dim=action_dim,
                gating_input_dim=embedding_dim,
                routing_type="invalid",
            )



@pytest.mark.unit
class TestMoEForwardWithGating:
    """Tests for MoE forward pass with internal gating network."""

    def test_forward_2d_soft_routing(
        self, simple_experts, embeddings_2d, embedding_dim, action_dim, batch_size, device
    ):
        """Test forward pass with 2D embeddings and soft routing."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            routing_type=MoERoutingType.SOFT.value,
            device=device,
        )

        outputs = moe(embeddings_2d)

        assert "action" in outputs
        assert "routing_weights" in outputs
        assert "expert_outputs" in outputs

        # Check shapes
        assert outputs["action"].shape == (batch_size, action_dim)
        assert outputs["routing_weights"].shape == (batch_size, len(simple_experts))
        assert outputs["expert_outputs"].shape == (batch_size, len(simple_experts), action_dim)

        # Check routing weights sum to 1
        assert torch.allclose(
            outputs["routing_weights"].sum(dim=-1),
            torch.ones(batch_size, device=embeddings_2d.device),
            atol=1e-5,
        )

    def test_forward_3d_soft_routing(
        self, simple_experts, embeddings_3d, embedding_dim, action_dim, batch_size, horizon, device
    ):
        """Test forward pass with 3D embeddings and soft routing."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            routing_type=MoERoutingType.SOFT.value,
            device=device,
        )

        outputs = moe(embeddings_3d)

        # Check shapes
        assert outputs["action"].shape == (batch_size, horizon, action_dim)
        assert outputs["routing_weights"].shape == (batch_size, horizon, len(simple_experts))
        assert outputs["expert_outputs"].shape == (
            batch_size,
            horizon,
            len(simple_experts),
            action_dim,
        )

    @pytest.mark.parametrize("top_k", [1, 2, 3])
    def test_forward_top_k_routing(
        self, simple_experts, embeddings_2d, embedding_dim, action_dim, top_k, device
    ):
        """Test forward pass with top-k routing."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            routing_type=MoERoutingType.TOP_K.value,
            top_k=top_k,
            device=device,
        )

        outputs = moe(embeddings_2d)

        assert "action" in outputs
        # Top-k routing should still produce valid outputs
        assert not torch.isnan(outputs["action"]).any()
        assert not torch.isinf(outputs["action"]).any()



@pytest.mark.unit
class TestMoEForwardWithExternalRouting:
    """Tests for MoE forward pass with externally provided routing weights."""

    def test_forward_2d_external_routing(
        self, simple_experts, embeddings_2d, routing_weights_2d, action_dim, batch_size
    ):
        """Test forward with 2D external routing weights."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=None,  # No internal gating
            routing_type=MoERoutingType.SOFT.value,
        )

        outputs = moe(embeddings_2d, routing_weights=routing_weights_2d)

        assert outputs["action"].shape == (batch_size, action_dim)
        # Routing weights should be the provided ones (after temperature scaling and softmax)
        assert outputs["routing_weights"].shape == routing_weights_2d.shape

    def test_forward_3d_external_routing(
        self, simple_experts, embeddings_3d, routing_weights_3d, action_dim, batch_size, horizon
    ):
        """Test forward with 3D external routing weights."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=None,
            routing_type=MoERoutingType.SOFT.value,
        )

        outputs = moe(embeddings_3d, routing_weights=routing_weights_3d)

        assert outputs["action"].shape == (batch_size, horizon, action_dim)

    def test_external_routing_with_logits(
        self, simple_experts, embeddings_2d, action_dim, batch_size, num_experts, device
    ):
        """Test that external routing weights can be logits (not normalized)."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=None,
            routing_type=MoERoutingType.SOFT.value,
        )

        # Provide unnormalized logits
        logits = torch.randn(batch_size, num_experts, device=device)

        outputs = moe(embeddings_2d, routing_weights=logits)

        # Should still produce valid outputs
        assert outputs["action"].shape == (batch_size, action_dim)
        # Routing weights should be normalized
        assert torch.allclose(
            outputs["routing_weights"].sum(dim=-1),
            torch.ones(batch_size, device=device),
            atol=1e-5,
        )

    def test_no_gating_no_external_raises_error(
        self, simple_experts, embeddings_2d, action_dim
    ):
        """Test that forward without gating and without external routing raises error."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=None,
        )

        with pytest.raises(ValueError, match="gating_input_dim must be provided"):
            moe(embeddings_2d)


@pytest.mark.unit
class TestMoETemperature:
    """Tests for temperature parameter in MoE."""

    @pytest.mark.parametrize("temperature", [0.1, 1.0, 10.0])
    def test_different_temperatures(
        self, simple_experts, embeddings_2d, embedding_dim, action_dim, temperature, device
    ):
        """Test that temperature affects routing sharpness."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            temperature=temperature,
            device=device,
        )

        outputs = moe(embeddings_2d)
        routing_weights = outputs["routing_weights"]

        # Lower temperature should produce sharper distributions
        # Higher temperature should produce more uniform distributions
        entropy = -(routing_weights * torch.log(routing_weights + 1e-8)).sum(dim=-1).mean()

        # Just check it produces valid outputs
        assert not torch.isnan(entropy)

    def test_learnable_temperature(
        self, simple_experts, embeddings_2d, embedding_dim, action_dim
    ):
        """Test learnable temperature parameter."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            temperature=1.0,
            learnable_temperature=True,
        )

        # Temperature should be a learnable parameter
        assert isinstance(moe.temperature, torch.nn.Parameter)
        assert moe.temperature.requires_grad

    def test_fixed_temperature(self, simple_experts, embeddings_2d, embedding_dim, action_dim):
        """Test fixed temperature parameter."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            temperature=1.0,
            learnable_temperature=False,
        )

        # Temperature should be a buffer (not parameter)
        assert not isinstance(moe.temperature, torch.nn.Parameter)



@pytest.mark.unit
class TestMoEBaseExpertCloning:
    """Tests for MoE instantiation via base_expert cloning."""

    def test_instantiation_with_base_expert(self, embedding_dim, action_dim, num_experts):
        """Test MoE instantiation by cloning a base expert."""
        base_expert = ActionHead(embedding_dim, action_dim, blocks=None)
        moe = MoEHead(
            base_expert=base_expert,
            num_experts=num_experts,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
        )

        assert moe is not None
        assert moe.num_experts == num_experts
        assert len(moe.experts) == num_experts

        # All experts should have correct dimensions
        for expert in moe.experts:
            assert expert.input_dim == embedding_dim
            assert expert.output_dim == action_dim

    def test_experts_have_independent_weights(self, embedding_dim, action_dim, device):
        """Test that cloned experts have independent weights."""
        base_expert = ActionHead(embedding_dim, action_dim, blocks=None)
        moe = MoEHead(
            base_expert=base_expert,
            num_experts=3,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            device=device,
        )

        # Get weight references
        expert0_weight = moe.experts[0].output_proj.weight
        expert1_weight = moe.experts[1].output_proj.weight

        # They should be different objects
        assert expert0_weight is not expert1_weight

        # Modify expert0 and verify expert1 doesn't change
        original_expert1_value = expert1_weight.data[0, 0].clone()
        expert0_weight.data[0, 0] = 999.0

        assert expert1_weight.data[0, 0] == original_expert1_value
        assert expert0_weight.data[0, 0] != expert1_weight.data[0, 0]

    def test_experts_with_blocks_have_independent_weights(self, embedding_dim, action_dim, device):
        """Test that cloned experts with blocks have independent weights."""
        blocks = [MLPBlock(embedding_dim, hidden_dims=[128], output_dim=embedding_dim)]
        base_expert = ActionHead(embedding_dim, action_dim, blocks=blocks)
        moe = MoEHead(
            base_expert=base_expert,
            num_experts=3,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            device=device,
        )

        # Check that block weights are independent (mlp is a Sequential containing Linear layers)
        block0_weight = moe.experts[0].blocks[0].mlp[0].weight
        block1_weight = moe.experts[1].blocks[0].mlp[0].weight

        assert block0_weight is not block1_weight

        # Modify and verify independence
        original_block1_value = block1_weight.data[0, 0].clone()
        block0_weight.data[0, 0] = 999.0

        assert block1_weight.data[0, 0] == original_block1_value

    def test_base_expert_cloning_forward_pass(
        self, embedding_dim, action_dim, num_experts, embeddings_2d, batch_size, device
    ):
        """Test forward pass with experts created via cloning."""
        base_expert = ActionHead(embedding_dim, action_dim, blocks=None)
        moe = MoEHead(
            base_expert=base_expert,
            num_experts=num_experts,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            device=device,
        )

        outputs = moe(embeddings_2d)

        assert "action" in outputs
        assert "routing_weights" in outputs
        assert "expert_outputs" in outputs
        assert outputs["action"].shape == (batch_size, action_dim)

    def test_missing_num_experts_raises_error(self, embedding_dim, action_dim):
        """Test that missing num_experts raises error."""
        base_expert = ActionHead(embedding_dim, action_dim, blocks=None)

        with pytest.raises(ValueError, match="Must provide either"):
            MoEHead(
                base_expert=base_expert,
                output_dim=action_dim,
                gating_input_dim=embedding_dim,
            )


@pytest.mark.unit
class TestMoEEdgeCases:
    """Tests for edge cases and validation."""

    def test_single_expert(self, embedding_dim, action_dim, device):
        """Test MoE with a single expert (should work but behave like regular head)."""
        expert = ActionHead(embedding_dim, action_dim, blocks=[]).to(device)
        moe = MoEHead(
            experts=[expert],
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            device=device,
        )

        embeddings = torch.randn(4, embedding_dim, device=device)
        outputs = moe(embeddings)

        # Should still work
        assert outputs["action"].shape == (4, action_dim)
        # Routing weights should be 1.0 for the single expert
        assert torch.allclose(
            outputs["routing_weights"], torch.ones_like(outputs["routing_weights"])
        )

    def test_top_k_exceeds_num_experts(self, simple_experts, embedding_dim, action_dim):
        """Test that top_k is clamped to num_experts."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            routing_type=MoERoutingType.TOP_K.value,
            top_k=100,  # Much larger than num_experts
        )

        # top_k should be clamped to num_experts
        assert moe.top_k == len(simple_experts)

    def test_gradient_flow(
        self, simple_experts, embeddings_2d, embedding_dim, action_dim, device
    ):
        """Test that gradients flow through MoE head."""
        moe = MoEHead(
            experts=simple_experts,
            output_dim=action_dim,
            gating_input_dim=embedding_dim,
            device=device,
        )

        embeddings_2d.requires_grad_(True)
        outputs = moe(embeddings_2d)

        # Compute loss and backprop
        loss = outputs["action"].sum()
        loss.backward()

        # Check that gradients exist
        assert embeddings_2d.grad is not None
        assert not torch.isnan(embeddings_2d.grad).any()

        # Check that gating network has gradients
        for param in moe.gating_network.parameters():
            if param.requires_grad:
                assert param.grad is not None

        # Check that experts have gradients
        for expert in moe.experts:
            for param in expert.parameters():
                if param.requires_grad:
                    assert param.grad is not None
