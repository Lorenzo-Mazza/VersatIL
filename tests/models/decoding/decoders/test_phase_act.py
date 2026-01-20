"""Tests for Phase-conditioned ACT decoder."""
import pytest
import torch

from versatil.models.decoding.decoders.factory.phase_act import PhaseACT
from versatil.models.decoding.action_heads import ActionHead, MLPBlock, MoEHead
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.constants import (
    POSITION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    PHASE_LABEL_KEY,
    IS_PAD_ACTION_KEY,
    Cameras,
    GripperType,
)
from versatil.models.decoding.constants import (
    MoERoutingType,
    ROUTING_WEIGHT,
    EXPERT_OUTPUTS,
    MU_KEY,
    LOGVAR_KEY,
)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def observation_horizon():
    return 1


@pytest.fixture
def prediction_horizon():
    return 10


@pytest.fixture
def embedding_dimension():
    return 256


@pytest.fixture
def num_phases():
    return 5


@pytest.fixture
def action_space(num_phases):
    return ActionSpace(
        has_position=True,
        position_dim=3,
        has_orientation=False,
        has_gripper=True,
        gripper_type=GripperType.BINARY.value,
        gripper_dim=1,
        predict_in_camera_frame=False,
        deltas_as_actions=False,
        task_has_phases=True,
        number_of_phases=num_phases,
    )


@pytest.fixture
def observation_space():
    return ObservationSpace(
        use_proprioceptive_data=False,
        use_proprio_base_frame=False,
        use_proprio_camera_frame=False,
        use_gripper_state=False,
        gripper_type=GripperType.BINARY.value,
        camera_keys=[Cameras.LEFT.value],
        use_language=False,
    )


@pytest.fixture
def phase_head(embedding_dimension, num_phases, device):
    return ActionHead(
        input_dim=embedding_dimension,
        output_dim=num_phases,
        blocks=[
            MLPBlock(
                input_dim=embedding_dimension,
                hidden_dims=[128],
                output_dim=embedding_dimension,
                activation="relu",
                dropout=0.1,
                normalization=True,
            )
        ]
    ).to(device)


@pytest.fixture
def position_moe_head(embedding_dimension, num_phases, device):
    base_expert = ActionHead(
        input_dim=embedding_dimension,
        output_dim=3,
        blocks=[
            MLPBlock(
                input_dim=embedding_dimension,
                hidden_dims=[128],
                output_dim=embedding_dimension,
                activation="silu",
                dropout=0.1,
                normalization=True,
            )
        ]
    )

    experts = [
        ActionHead(
            input_dim=embedding_dimension,
            output_dim=3,
            blocks=[
                MLPBlock(
                    input_dim=embedding_dimension,
                    hidden_dims=[128],
                    output_dim=embedding_dimension,
                    activation="silu",
                    dropout=0.1,
                    normalization=True,
                )
            ]
        ).to(device)
        for _ in range(num_phases)
    ]

    return MoEHead(
        experts=experts,
        output_dim=3,
        gating_input_dim=None,
        routing_type=MoERoutingType.SOFT.value,
        temperature=100.0,
        learnable_temperature=True,
    )


@pytest.fixture
def gripper_moe_head(embedding_dimension, num_phases, device):
    experts = [
        ActionHead(
            input_dim=embedding_dimension,
            output_dim=1,
            blocks=[
                MLPBlock(
                    input_dim=embedding_dimension,
                    hidden_dims=[64],
                    output_dim=embedding_dimension,
                    activation="silu",
                    dropout=0.1,
                    normalization=True,
                )
            ]
        ).to(device)
        for _ in range(num_phases)
    ]

    return MoEHead(
        experts=experts,
        output_dim=1,
        gating_input_dim=None,
        routing_type=MoERoutingType.SOFT.value,
        temperature=100.0,
        learnable_temperature=True,
    )


@pytest.fixture
def action_heads_phase(phase_head, position_moe_head, gripper_moe_head):
    return {
        PHASE_LABEL_KEY: phase_head,
        POSITION_ACTION_KEY: position_moe_head,
        GRIPPER_ACTION_KEY: gripper_moe_head,
    }


@pytest.fixture
def action_heads_no_phase(position_moe_head, gripper_moe_head):
    return {
        POSITION_ACTION_KEY: position_moe_head,
        GRIPPER_ACTION_KEY: gripper_moe_head,
    }


@pytest.fixture
def action_heads_no_moe(phase_head, embedding_dimension, device):
    return {
        PHASE_LABEL_KEY: phase_head,
        POSITION_ACTION_KEY: ActionHead(
            input_dim=embedding_dimension, output_dim=3, blocks=[]
        ).to(device),
        GRIPPER_ACTION_KEY: ActionHead(
            input_dim=embedding_dimension, output_dim=1, blocks=[]
        ).to(device),
    }


@pytest.fixture
def spatial_features(batch_size, device):
    return {
        "rgb_left_features": torch.randn(batch_size, 2048, 7, 7, device=device)
    }


@pytest.fixture
def actions_dict(batch_size, prediction_horizon, num_phases, device):
    return {
        POSITION_ACTION_KEY: torch.randn(batch_size, prediction_horizon, 3, device=device),
        GRIPPER_ACTION_KEY: torch.randint(0, 2, (batch_size, prediction_horizon, 1), device=device).float(),
        PHASE_LABEL_KEY: torch.randint(0, num_phases, (batch_size, prediction_horizon, 1), device=device).long(),
        IS_PAD_ACTION_KEY: torch.zeros(batch_size, prediction_horizon, dtype=torch.bool, device=device),
    }


def create_phase_act(
    input_keys,
    action_space,
    observation_space,
    action_heads,
    observation_horizon,
    prediction_horizon,
    embedding_dimension,
    device,
    **kwargs
):
    return PhaseACT(
        input_keys=input_keys,
        action_space=action_space,
        observation_space=observation_space,
        action_heads=action_heads,
        observation_horizon=observation_horizon,
        prediction_horizon=prediction_horizon,
        embedding_dimension=embedding_dimension,
        device=device,
        **kwargs
    )


class TestPhaseACTInstantiation:

    def test_basic_instantiation(
        self, action_space, observation_space, action_heads_phase,
        observation_horizon, prediction_horizon, embedding_dimension, device
    ):
        decoder = create_phase_act(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads_phase,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        assert decoder is not None
        assert decoder.phase_routing_key == PHASE_LABEL_KEY
        assert PHASE_LABEL_KEY in decoder.action_heads
        assert POSITION_ACTION_KEY in decoder.action_heads
        assert GRIPPER_ACTION_KEY in decoder.action_heads

    def test_missing_phase_head_raises_error(
        self, action_space, observation_space, action_heads_no_phase,
        observation_horizon, prediction_horizon, embedding_dimension, device
    ):
        with pytest.raises(ValueError, match="phase_label"):
            create_phase_act(
                input_keys=["rgb_left_features"],
                action_space=action_space,
                observation_space=observation_space,
                action_heads=action_heads_no_phase,
                observation_horizon=observation_horizon,
                prediction_horizon=prediction_horizon,
                embedding_dimension=embedding_dimension,
                device=device,
            )

    def test_no_moe_heads_raises_error(
        self, action_space, observation_space, action_heads_no_moe,
        observation_horizon, prediction_horizon, embedding_dimension, device
    ):
        with pytest.raises(ValueError, match="at least one MoE action head"):
            create_phase_act(
                input_keys=["rgb_left_features"],
                action_space=action_space,
                observation_space=observation_space,
                action_heads=action_heads_no_moe,
                observation_horizon=observation_horizon,
                prediction_horizon=prediction_horizon,
                embedding_dimension=embedding_dimension,
                device=device,
            )


    @pytest.mark.parametrize("num_experts", [1, 3, 5, 7])
    def test_different_num_experts(
        self, action_space, observation_space, phase_head, gripper_moe_head, embedding_dimension,
        observation_horizon, prediction_horizon, device, num_experts
    ):
        experts = [
            ActionHead(embedding_dimension, 3, blocks=[]).to(device)
            for _ in range(num_experts)
        ]

        position_moe = MoEHead(
            experts=experts,
            output_dim=3,
            gating_input_dim=None,
            routing_type=MoERoutingType.SOFT.value,
        )

        action_heads = {
            PHASE_LABEL_KEY: phase_head,
            POSITION_ACTION_KEY: position_moe,
            GRIPPER_ACTION_KEY: gripper_moe_head,
        }

        decoder = create_phase_act(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        assert decoder.action_heads[POSITION_ACTION_KEY].num_experts == num_experts


class TestPhaseACTForward:

    def test_forward_pass_training(
        self, action_space, observation_space, action_heads_phase,
        observation_horizon, prediction_horizon, embedding_dimension,
        spatial_features, actions_dict, device, batch_size
    ):
        decoder = create_phase_act(
            input_keys=list(spatial_features.keys()),
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads_phase,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        outputs = decoder(spatial_features, actions_dict)

        assert PHASE_LABEL_KEY in outputs
        assert POSITION_ACTION_KEY in outputs
        assert GRIPPER_ACTION_KEY in outputs
        assert outputs[PHASE_LABEL_KEY].shape == (batch_size, prediction_horizon, 5)
        assert outputs[POSITION_ACTION_KEY].shape == (batch_size, prediction_horizon, 3)
        assert outputs[GRIPPER_ACTION_KEY].shape == (batch_size, prediction_horizon, 1)

    def test_forward_pass_inference(
        self, action_space, observation_space, action_heads_phase,
        observation_horizon, prediction_horizon, embedding_dimension,
        spatial_features, device, batch_size
    ):
        decoder = create_phase_act(
            input_keys=list(spatial_features.keys()),
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads_phase,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        outputs = decoder(spatial_features, actions=None)

        assert PHASE_LABEL_KEY in outputs
        assert POSITION_ACTION_KEY in outputs
        assert GRIPPER_ACTION_KEY in outputs
        assert outputs[PHASE_LABEL_KEY].shape == (batch_size, prediction_horizon, 5)

    def test_no_vae_in_decoder(
        self, action_space, observation_space, action_heads_phase,
        observation_horizon, prediction_horizon, embedding_dimension,
        spatial_features, actions_dict, device
    ):
        """Test that VAE is not present in decoder (handled at algorithm level)."""
        decoder = create_phase_act(
            input_keys=list(spatial_features.keys()),
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads_phase,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        outputs = decoder(spatial_features, actions_dict)

        # VAE keys should NOT be in decoder output (handled at algorithm level)
        assert MU_KEY not in outputs
        assert LOGVAR_KEY not in outputs

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(
        self, action_space, observation_space, action_heads_phase,
        observation_horizon, prediction_horizon, embedding_dimension, device, batch_size
    ):
        decoder = create_phase_act(
            input_keys=["rgb_left_features"],
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads_phase,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        features = {
            "rgb_left_features": torch.randn(batch_size, 2048, 7, 7, device=device)
        }

        outputs = decoder(features, actions=None)

        assert outputs[PHASE_LABEL_KEY].shape[0] == batch_size
        assert outputs[POSITION_ACTION_KEY].shape[0] == batch_size


class TestPhaseACTRouting:

    def test_routing_weights_present(
        self, action_space, observation_space, action_heads_phase,
        observation_horizon, prediction_horizon, embedding_dimension,
        spatial_features, actions_dict, device, batch_size, num_phases
    ):
        decoder = create_phase_act(
            input_keys=list(spatial_features.keys()),
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads_phase,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        outputs = decoder(spatial_features, actions_dict)

        assert f'{POSITION_ACTION_KEY}_{ROUTING_WEIGHT}' in outputs
        assert f'{GRIPPER_ACTION_KEY}_{ROUTING_WEIGHT}' in outputs

        position_routing = outputs[f'{POSITION_ACTION_KEY}_{ROUTING_WEIGHT}']
        assert position_routing.shape == (batch_size, prediction_horizon, num_phases)

        assert torch.allclose(position_routing.sum(dim=-1), torch.ones_like(position_routing.sum(dim=-1)), atol=1e-5)

    def test_expert_outputs_present(
        self, action_space, observation_space, action_heads_phase,
        observation_horizon, prediction_horizon, embedding_dimension,
        spatial_features, actions_dict, device, batch_size, num_phases
    ):
        decoder = create_phase_act(
            input_keys=list(spatial_features.keys()),
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads_phase,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        outputs = decoder(spatial_features, actions_dict)

        assert f'{POSITION_ACTION_KEY}_{EXPERT_OUTPUTS}' in outputs
        assert f'{GRIPPER_ACTION_KEY}_{EXPERT_OUTPUTS}' in outputs

        position_experts = outputs[f'{POSITION_ACTION_KEY}_{EXPERT_OUTPUTS}']
        assert position_experts.shape == (batch_size, prediction_horizon, num_phases, 3)

    def test_phase_logits_used_for_routing(
        self, action_space, observation_space, action_heads_phase,
        observation_horizon, prediction_horizon, embedding_dimension,
        spatial_features, actions_dict, device, batch_size
    ):
        decoder = create_phase_act(
            input_keys=list(spatial_features.keys()),
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads_phase,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        outputs = decoder(spatial_features, actions_dict)

        phase_logits = outputs[PHASE_LABEL_KEY]
        position_routing = outputs[f'{POSITION_ACTION_KEY}_{ROUTING_WEIGHT}']

        phase_probs = torch.softmax(phase_logits / 100.0, dim=-1)

        assert phase_probs.shape == position_routing.shape


class TestPhaseACTOutputStructure:

    def test_output_keys_complete(
        self, action_space, observation_space, action_heads_phase,
        observation_horizon, prediction_horizon, embedding_dimension,
        spatial_features, actions_dict, device
    ):
        decoder = create_phase_act(
            input_keys=list(spatial_features.keys()),
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads_phase,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        outputs = decoder(spatial_features, actions_dict)

        expected_keys = {
            PHASE_LABEL_KEY,
            POSITION_ACTION_KEY,
            GRIPPER_ACTION_KEY,
            f'{POSITION_ACTION_KEY}_{ROUTING_WEIGHT}',
            f'{POSITION_ACTION_KEY}_{EXPERT_OUTPUTS}',
            f'{GRIPPER_ACTION_KEY}_{ROUTING_WEIGHT}',
            f'{GRIPPER_ACTION_KEY}_{EXPERT_OUTPUTS}',
            # VAE keys (MU_KEY, LOGVAR_KEY) removed - handled at algorithm level
        }

        assert set(outputs.keys()) == expected_keys

    def test_output_dtypes(
        self, action_space, observation_space, action_heads_phase,
        observation_horizon, prediction_horizon, embedding_dimension,
        spatial_features, device
    ):
        decoder = create_phase_act(
            input_keys=list(spatial_features.keys()),
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads_phase,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        outputs = decoder(spatial_features, actions=None)

        for key, value in outputs.items():
            assert isinstance(value, torch.Tensor)
            assert value.dtype in [torch.float32, torch.float16]


class TestPhaseACTGradients:

    def test_gradients_flow_through_phase_head(
        self, action_space, observation_space, action_heads_phase,
        observation_horizon, prediction_horizon, embedding_dimension,
        spatial_features, actions_dict, device
    ):
        decoder = create_phase_act(
            input_keys=list(spatial_features.keys()),
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads_phase,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        outputs = decoder(spatial_features, actions_dict)

        loss = outputs[PHASE_LABEL_KEY].sum()
        loss.backward()

        phase_head_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in decoder.action_heads[PHASE_LABEL_KEY].parameters()
        )
        assert phase_head_has_grad

    def test_gradients_flow_through_moe_experts(
        self, action_space, observation_space, action_heads_phase,
        observation_horizon, prediction_horizon, embedding_dimension,
        spatial_features, actions_dict, device
    ):
        decoder = create_phase_act(
            input_keys=list(spatial_features.keys()),
            action_space=action_space,
            observation_space=observation_space,
            action_heads=action_heads_phase,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            embedding_dimension=embedding_dimension,
            device=device,
        )

        outputs = decoder(spatial_features, actions_dict)

        loss = outputs[POSITION_ACTION_KEY].sum()
        loss.backward()

        moe_head = decoder.action_heads[POSITION_ACTION_KEY]
        expert_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for expert in moe_head.experts
            for p in expert.parameters()
        )
        assert expert_has_grad
