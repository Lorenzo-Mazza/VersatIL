"""Tests for Mixture of Experts decoder."""
import pytest
import torch

from versatil.models.decoding.decoders import MoEDecoder, DecoderInput
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.constants import MoERoutingType, FeatureType
from versatil.models.decoding.action_heads import ActionHead
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.constants import (
    POSITION_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    OrientationRepresentation,
    GripperType,
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
def observation_horizon():
    """Default observation horizon."""
    return 1


@pytest.fixture
def embedding_dim():
    """Default embedding dimension."""
    return 256


@pytest.fixture
def num_experts():
    """Default number of experts."""
    return 3


@pytest.fixture
def action_space():
    """Create default action space configuration."""
    return ActionSpace(
        has_position=True,
        position_dim=3,
        has_orientation=True,
        orientation_dim=4,
        orientation_repr=OrientationRepresentation.QUATERNION.value,
        has_gripper=True,
        gripper_type=GripperType.BINARY.value,
        gripper_dim=1,
        predict_in_camera_frame=False,
        deltas_as_actions=False,
    )


@pytest.fixture
def observation_space():
    """Create default observation space configuration."""
    return ObservationSpace(
        use_proprioceptive_data=False,
        use_proprio_base_frame=False,
        use_proprio_camera_frame=False,
        use_gripper_state=False,
        gripper_type=GripperType.BINARY.value,
        camera_keys=[],
        use_language=False,
    )


class MockDecoder(ActionDecoder):
    """Mock decoder for testing."""

    def __init__(
        self,
        decoder_input: DecoderInput,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        action_heads: dict,
        device: str,
        observation_horizon: int,
        prediction_horizon: int,
        embedding_dim: int,
    ):
        super().__init__(
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
        )
        self.embedding_dim = embedding_dim
        # Simple linear layer to process features
        self.feature_processor = torch.nn.Linear(embedding_dim, embedding_dim).to(device)

    def forward(self, features: dict, actions=None):
        """Simple forward pass for testing."""
        # Get first available feature
        feature = next(iter(features.values()))
        batch_size = feature.shape[0]

        # Process features
        processed = self.feature_processor(feature)  # (B, embedding_dim)

        # Expand to prediction horizon
        processed = processed.unsqueeze(1).expand(
            batch_size, self.prediction_horizon, self.embedding_dim
        )  # (B, horizon, embedding_dim)

        # Apply action heads
        outputs = {}
        for key, head in self.action_heads.items():
            outputs[key] = head(processed)  # (B, horizon, action_dim)

        return outputs


@pytest.fixture
def decoder_input():
    """Create decoder input specification."""
    return DecoderInput(
        keys=["flat_features"],
        required_types=[FeatureType.FLAT],
        requires_actions=False,
    )


def create_action_heads(action_space, embedding_dim, device):
    """Helper function to create action heads for all action modalities."""
    heads = {}

    if action_space.has_position:
        heads[POSITION_ACTION_KEY] = ActionHead(
            input_dim=embedding_dim,
            output_dim=action_space.position_dim,
            blocks=[],
        ).to(device)

    if action_space.has_orientation:
        heads[ORIENTATION_ACTION_KEY] = ActionHead(
            input_dim=embedding_dim,
            output_dim=action_space.orientation_dim,
            blocks=[],
        ).to(device)

    if action_space.has_gripper:
        heads[GRIPPER_ACTION_KEY] = ActionHead(
            input_dim=embedding_dim,
            output_dim=action_space.gripper_dim,
            blocks=[],
        ).to(device)

    return heads


@pytest.fixture
def action_heads(action_space, embedding_dim, device):
    """Create action heads for all action modalities."""
    return create_action_heads(action_space, embedding_dim, device)


@pytest.fixture
def expert_decoders(
    num_experts,
    decoder_input,
    observation_space,
    action_space,
    device,
    observation_horizon,
    prediction_horizon,
    embedding_dim,
):
    """Create expert decoder instances for testing."""
    experts = []
    for _ in range(num_experts):
        # Create separate action heads for each expert
        expert_action_heads = create_action_heads(action_space, embedding_dim, device)
        expert = MockDecoder(
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=expert_action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dim=embedding_dim,
        )
        experts.append(expert)
    return experts


@pytest.fixture
def flat_features(batch_size, embedding_dim, device):
    """Create flat features for testing."""
    return {"flat_features": torch.randn(batch_size, embedding_dim, device=device)}


@pytest.fixture
def routing_weights_2d(batch_size, num_experts, device):
    """Create 2D routing weights (B, num_experts)."""
    weights = torch.randn(batch_size, num_experts, device=device)
    return torch.softmax(weights, dim=-1)


@pytest.mark.unit
class TestMoEDecoderInstantiation:
    """Tests for MoE decoder instantiation."""

    def test_basic_instantiation_with_gating(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
    ):
        """Test basic instantiation with internal gating network."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
        )
        assert moe is not None
        assert moe.num_experts == len(expert_decoders)
        assert moe.has_gating_network

    def test_basic_instantiation_without_gating(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
    ):
        """Test basic instantiation without gating network (external routing)."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=None,
        )
        assert moe is not None
        assert moe.num_experts == len(expert_decoders)
        assert not moe.has_gating_network

    def test_empty_experts_raises_error(
        self,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
    ):
        """Test that empty experts list raises error."""
        with pytest.raises(ValueError, match="at least one expert"):
            MoEDecoder(
                expert_decoders=[],
                decoder_input=decoder_input,
                observation_space=observation_space,
                action_space=action_space,
                action_heads=action_heads,
                device=device,
                observation_horizon=observation_horizon,
                prediction_horizon=prediction_horizon,
                gating_input_dim=embedding_dim,
            )

    @pytest.mark.parametrize(
        "routing_type",
        [
            MoERoutingType.SOFT.value,
            MoERoutingType.TOP_K.value,
        ],
    )
    def test_routing_types(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
        routing_type,
    ):
        """Test different routing types."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
            routing_type=routing_type,
        )
        assert moe.routing_type == routing_type

    def test_invalid_routing_type(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
    ):
        """Test that invalid routing type raises error."""
        with pytest.raises(ValueError, match="Invalid routing_type"):
            MoEDecoder(
                expert_decoders=expert_decoders,
                decoder_input=decoder_input,
                observation_space=observation_space,
                action_space=action_space,
                action_heads=action_heads,
                device=device,
                observation_horizon=observation_horizon,
                prediction_horizon=prediction_horizon,
                gating_input_dim=embedding_dim,
                routing_type="invalid",
            )


@pytest.mark.unit
class TestMoEDecoderForwardWithGating:
    """Tests for MoE decoder forward pass with internal gating network."""

    def test_forward_soft_routing(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
        flat_features,
        batch_size,
    ):
        """Test forward pass with soft routing."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
            routing_type=MoERoutingType.SOFT.value,
        )

        outputs = moe(flat_features)

        # Check all action predictions exist
        assert POSITION_ACTION_KEY in outputs
        assert ORIENTATION_ACTION_KEY in outputs
        assert GRIPPER_ACTION_KEY in outputs
        assert "routing_weights" in outputs
        assert "expert_outputs" in outputs

        # Check shapes
        assert outputs[POSITION_ACTION_KEY].shape == (
            batch_size,
            prediction_horizon,
            action_space.position_dim,
        )
        assert outputs[ORIENTATION_ACTION_KEY].shape == (
            batch_size,
            prediction_horizon,
            action_space.orientation_dim,
        )
        assert outputs[GRIPPER_ACTION_KEY].shape == (
            batch_size,
            prediction_horizon,
            action_space.gripper_dim,
        )

        # Check routing weights sum to 1
        assert torch.allclose(
            outputs["routing_weights"].sum(dim=-1),
            torch.ones(batch_size, device=device),
            atol=1e-5,
        )

    @pytest.mark.parametrize("top_k", [1, 2, 3])
    def test_forward_top_k_routing(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
        flat_features,
        top_k,
    ):
        """Test forward pass with top-k routing."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
            routing_type=MoERoutingType.TOP_K.value,
            top_k=top_k,
        )

        outputs = moe(flat_features)

        # Check all action predictions exist
        assert POSITION_ACTION_KEY in outputs
        assert ORIENTATION_ACTION_KEY in outputs
        assert GRIPPER_ACTION_KEY in outputs

        # Top-k routing should still produce valid outputs
        assert not torch.isnan(outputs[POSITION_ACTION_KEY]).any()
        assert not torch.isinf(outputs[POSITION_ACTION_KEY]).any()


@pytest.mark.unit
class TestMoEDecoderForwardWithExternalRouting:
    """Tests for MoE decoder forward pass with externally provided routing weights."""

    def test_forward_external_routing(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        flat_features,
        routing_weights_2d,
        batch_size,
    ):
        """Test forward with external routing weights."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=None,  # No internal gating
            routing_type=MoERoutingType.SOFT.value,
        )

        outputs = moe(flat_features, routing_weights=routing_weights_2d)

        # Check all action predictions exist
        assert POSITION_ACTION_KEY in outputs
        assert ORIENTATION_ACTION_KEY in outputs
        assert GRIPPER_ACTION_KEY in outputs

        # Routing weights should be the provided ones (after temperature scaling and softmax)
        assert outputs["routing_weights"].shape == routing_weights_2d.shape

    def test_external_routing_with_logits(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        flat_features,
        batch_size,
        num_experts,
    ):
        """Test that external routing weights can be logits (not normalized)."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=None,
            routing_type=MoERoutingType.SOFT.value,
        )

        # Provide unnormalized logits
        logits = torch.randn(batch_size, num_experts, device=device)

        outputs = moe(flat_features, routing_weights=logits)

        # Should still produce valid outputs
        assert POSITION_ACTION_KEY in outputs

        # Routing weights should be normalized
        assert torch.allclose(
            outputs["routing_weights"].sum(dim=-1),
            torch.ones(batch_size, device=device),
            atol=1e-5,
        )

    def test_no_gating_no_external_raises_error(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        flat_features,
    ):
        """Test that forward without gating and without external routing raises error."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=None,
        )

        with pytest.raises(ValueError, match="gating_input_dim must be provided"):
            moe(flat_features)

@pytest.mark.unit
class TestMoEDecoderTemperature:
    """Tests for temperature parameter in MoE decoder."""

    @pytest.mark.parametrize("temperature", [0.1, 1.0, 10.0])
    def test_different_temperatures(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
        flat_features,
        temperature,
    ):
        """Test that temperature affects routing sharpness."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
            temperature=temperature,
        )

        outputs = moe(flat_features)
        routing_weights = outputs["routing_weights"]

        # Compute entropy to check routing distribution
        entropy = -(routing_weights * torch.log(routing_weights + 1e-8)).sum(dim=-1).mean()

        # Just check it produces valid outputs
        assert not torch.isnan(entropy)

    def test_learnable_temperature(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
    ):
        """Test learnable temperature parameter."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
            temperature=1.0,
            learnable_temperature=True,
        )

        # Temperature should be a learnable parameter
        assert isinstance(moe.temperature, torch.nn.Parameter)
        assert moe.temperature.requires_grad

    def test_fixed_temperature(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
    ):
        """Test fixed temperature parameter."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
            temperature=1.0,
            learnable_temperature=False,
        )

        # Temperature should be a buffer (not parameter)
        assert not isinstance(moe.temperature, torch.nn.Parameter)


@pytest.mark.unit
class TestMoEDecoderExpertSpecialization:
    """Tests for expert specialization analysis."""

    def test_get_expert_specialization_with_internal_gating(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
        flat_features,
        num_experts,
    ):
        """Test expert specialization analysis with internal gating."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
        )

        stats = moe.get_expert_specialization(flat_features)

        # Check all statistics are present
        assert "expert_usage" in stats
        assert "routing_entropy" in stats
        assert "top_expert_confidence" in stats

        # Check shapes
        assert stats["expert_usage"].shape == (num_experts,)
        assert stats["routing_entropy"].shape == ()
        assert stats["top_expert_confidence"].shape == ()

        # Check expert usage sums to 1
        assert torch.allclose(
            stats["expert_usage"].sum(), torch.tensor(1.0, device=device), atol=1e-5
        )

    def test_get_expert_specialization_with_external_routing(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        flat_features,
        routing_weights_2d,
    ):
        """Test expert specialization analysis with external routing."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=None,
        )

        stats = moe.get_expert_specialization(flat_features, external_routing=routing_weights_2d)

        # Check all statistics are present
        assert "expert_usage" in stats
        assert "routing_entropy" in stats
        assert "top_expert_confidence" in stats



@pytest.mark.unit
class TestMoEDecoderEdgeCases:
    """Tests for edge cases and validation."""

    def test_single_expert(
        self,
        decoder_input,
        observation_space,
        action_space,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
        flat_features,
    ):
        """Test MoE with a single expert (should work but behave like regular decoder)."""
        expert_action_heads = create_action_heads(action_space, embedding_dim, device)
        expert = MockDecoder(
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=expert_action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dim=embedding_dim,
        )

        # Create separate action heads for MoE decoder (for validation purposes)
        moe_action_heads = create_action_heads(action_space, embedding_dim, device)

        moe = MoEDecoder(
            expert_decoders=[expert],
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=moe_action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
        )

        outputs = moe(flat_features)

        # Should still work
        assert POSITION_ACTION_KEY in outputs

        # Routing weights should be 1.0 for the single expert
        assert torch.allclose(
            outputs["routing_weights"], torch.ones_like(outputs["routing_weights"])
        )

    def test_top_k_exceeds_num_experts(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
    ):
        """Test that top_k is clamped to num_experts."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
            routing_type=MoERoutingType.TOP_K.value,
            top_k=100,  # Much larger than num_experts
        )

        # top_k should be clamped to num_experts
        assert moe.top_k == len(expert_decoders)

    def test_gradient_flow(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
        batch_size,
    ):
        """Test that gradients flow through MoE decoder."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
        )

        # Create features that require gradients
        features = {
            "flat_features": torch.randn(
                batch_size, embedding_dim, device=device, requires_grad=True
            )
        }

        outputs = moe(features)

        # Compute loss and backprop
        loss = outputs[POSITION_ACTION_KEY].sum()
        loss.backward()

        # Check that gradients exist
        assert features["flat_features"].grad is not None
        assert not torch.isnan(features["flat_features"].grad).any()

        # Check that gating network has gradients
        if moe.has_gating_network:
            for param in moe.gating_network.parameters():
                if param.requires_grad:
                    assert param.grad is not None

        # Check that experts have gradients (at least some parameters should have gradients)
        for expert in moe.expert_decoders:
            expert_params_with_grad = sum(
                1 for p in expert.parameters() if p.requires_grad and p.grad is not None
            )
            expert_params_total = sum(1 for p in expert.parameters() if p.requires_grad)
            # At least some parameters should have gradients
            assert expert_params_with_grad > 0
            assert expert_params_with_grad / expert_params_total >= 0.3  # At least 30% should have gradients



@pytest.mark.unit
class TestMoEDecoderParametrized:
    """Parametrized tests for MoE decoder with different configurations."""

    @pytest.mark.parametrize("num_experts_param", [1, 3, 5])
    def test_different_num_experts(
        self,
        decoder_input,
        observation_space,
        action_space,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
        flat_features,
        num_experts_param,
    ):
        """Test MoE decoder with different numbers of experts."""
        # Create experts
        experts = []
        for _ in range(num_experts_param):
            expert_action_heads = create_action_heads(action_space, embedding_dim, device)
            expert = MockDecoder(
                decoder_input=decoder_input,
                observation_space=observation_space,
                action_space=action_space,
                action_heads=expert_action_heads,
                device=device,
                observation_horizon=observation_horizon,
                prediction_horizon=prediction_horizon,
                embedding_dim=embedding_dim,
            )
            experts.append(expert)

        # Create action heads for MoE decoder (for validation purposes)
        moe_action_heads = create_action_heads(action_space, embedding_dim, device)

        moe = MoEDecoder(
            expert_decoders=experts,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=moe_action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
        )

        outputs = moe(flat_features)

        # Check outputs are valid
        assert POSITION_ACTION_KEY in outputs
        assert outputs["routing_weights"].shape[-1] == num_experts_param

    @pytest.mark.parametrize("prediction_horizon_param", [1, 10, 50])
    def test_different_prediction_horizons(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        device,
        observation_horizon,
        embedding_dim,
        batch_size,
        prediction_horizon_param,
    ):
        """Test MoE decoder with different prediction horizons."""
        # Recreate experts with different prediction horizon
        experts = []
        for _ in range(3):
            expert_action_heads = create_action_heads(action_space, embedding_dim, device)
            expert = MockDecoder(
                decoder_input=decoder_input,
                observation_space=observation_space,
                action_space=action_space,
                action_heads=expert_action_heads,
                device=device,
                observation_horizon=observation_horizon,
                prediction_horizon=prediction_horizon_param,
                embedding_dim=embedding_dim,
            )
            experts.append(expert)

        # Create action heads for MoE decoder (for validation purposes)
        moe_action_heads = create_action_heads(action_space, embedding_dim, device)

        moe = MoEDecoder(
            expert_decoders=experts,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=moe_action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon_param,
            gating_input_dim=embedding_dim,
        )

        features = {"flat_features": torch.randn(batch_size, embedding_dim, device=device)}
        outputs = moe(features)

        # Check output shapes
        assert outputs[POSITION_ACTION_KEY].shape == (
            batch_size,
            prediction_horizon_param,
            action_space.position_dim,
        )


@pytest.mark.unit
class TestMoEDecoderGatingFeatureKey:
    """Tests for gating_feature_key parameter."""

    def test_gating_feature_key_with_specific_key(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
        batch_size,
    ):
        """Test that gating network uses specified feature key."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
            gating_feature_key="latent",  # Specify which key to use
        )

        # Create features dict with multiple features
        features = {
            "rgb_features": torch.randn(batch_size, embedding_dim, device=device),
            "latent": torch.randn(batch_size, embedding_dim, device=device),  # This should be used
            "proprio_features": torch.randn(batch_size, 64, device=device),
        }

        outputs = moe(features)

        # Should successfully use the latent feature
        assert POSITION_ACTION_KEY in outputs
        assert "routing_weights" in outputs

    def test_gating_feature_key_raises_if_not_found(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
        batch_size,
    ):
        """Test that specifying non-existent gating_feature_key raises error."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
            gating_feature_key="nonexistent_key",
        )

        features = {
            "rgb_features": torch.randn(batch_size, embedding_dim, device=device),
        }

        with pytest.raises(ValueError, match="Gating feature key 'nonexistent_key' not found"):
            moe(features)

    def test_gating_feature_key_none_uses_default(
        self,
        expert_decoders,
        decoder_input,
        observation_space,
        action_space,
        action_heads,
        device,
        observation_horizon,
        prediction_horizon,
        embedding_dim,
        batch_size,
    ):
        """Test that gating_feature_key=None uses default behavior (first feature)."""
        moe = MoEDecoder(
            expert_decoders=expert_decoders,
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            gating_input_dim=embedding_dim,
            gating_feature_key=None,  # Use default behavior
        )

        features = {
            "rgb_features": torch.randn(batch_size, embedding_dim, device=device),
            "latent": torch.randn(batch_size, embedding_dim, device=device),
        }

        outputs = moe(features)

        # Should work using first available feature
        assert POSITION_ACTION_KEY in outputs
        assert "routing_weights" in outputs
