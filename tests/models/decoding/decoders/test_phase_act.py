"""Tests for Phase-conditioned ACT decoder."""
import pytest
import torch

from versatil.models.decoding.decoders.factory.phase_act import PhaseACT
from versatil.models.decoding.action_heads import ActionHead, MLPBlock, MoEHead
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.constants import (
    Cameras,
    GripperType,
    ObsKey,
    ProprioceptiveType,
    SampleKey,
)
from versatil.models.decoding.constants import DecoderOutputKey, LatentKey, MoERoutingType


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
        ObsKey.PHASE_LABEL.value: phase_head,
        ProprioceptiveType.POSITION.value: position_moe_head,
        ProprioceptiveType.GRIPPER.value: gripper_moe_head,
    }


@pytest.fixture
def action_heads_no_phase(position_moe_head, gripper_moe_head):
    return {
        ProprioceptiveType.POSITION.value: position_moe_head,
        ProprioceptiveType.GRIPPER.value: gripper_moe_head,
    }


@pytest.fixture
def action_heads_no_moe(phase_head, embedding_dimension, device):
    return {
        ObsKey.PHASE_LABEL.value: phase_head,
        ProprioceptiveType.POSITION.value: ActionHead(
            input_dim=embedding_dimension, output_dim=3, blocks=[]
        ).to(device),
        ProprioceptiveType.GRIPPER.value: ActionHead(
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
        ProprioceptiveType.POSITION.value: torch.randn(batch_size, prediction_horizon, 3, device=device),
        ProprioceptiveType.GRIPPER.value: torch.randint(0, 2, (batch_size, prediction_horizon, 1), device=device).float(),
        ObsKey.PHASE_LABEL.value: torch.randint(0, num_phases, (batch_size, prediction_horizon, 1), device=device).long(),
        SampleKey.IS_PAD_ACTION.value: torch.zeros(batch_size, prediction_horizon, dtype=torch.bool, device=device),
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
        assert decoder.phase_routing_key == ObsKey.PHASE_LABEL.value
        assert ObsKey.PHASE_LABEL.value in decoder.action_heads
        assert ProprioceptiveType.POSITION.value in decoder.action_heads
        assert ProprioceptiveType.GRIPPER.value in decoder.action_heads

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
            ObsKey.PHASE_LABEL.value: phase_head,
            ProprioceptiveType.POSITION.value: position_moe,
            ProprioceptiveType.GRIPPER.value: gripper_moe_head,
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

        assert decoder.action_heads[ProprioceptiveType.POSITION.value].num_experts == num_experts


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

        assert ObsKey.PHASE_LABEL.value in outputs
        assert ProprioceptiveType.POSITION.value in outputs
        assert ProprioceptiveType.GRIPPER.value in outputs
        assert outputs[ObsKey.PHASE_LABEL.value].shape == (batch_size, prediction_horizon, 5)
        assert outputs[ProprioceptiveType.POSITION.value].shape == (batch_size, prediction_horizon, 3)
        assert outputs[ProprioceptiveType.GRIPPER.value].shape == (batch_size, prediction_horizon, 1)

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

        assert ObsKey.PHASE_LABEL.value in outputs
        assert ProprioceptiveType.POSITION.value in outputs
        assert ProprioceptiveType.GRIPPER.value in outputs
        assert outputs[ObsKey.PHASE_LABEL.value].shape == (batch_size, prediction_horizon, 5)

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
        assert LatentKey.POSTERIOR_MU.value not in outputs
        assert LatentKey.POSTERIOR_LOGVAR.value not in outputs

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

        assert outputs[ObsKey.PHASE_LABEL.value].shape[0] == batch_size
        assert outputs[ProprioceptiveType.POSITION.value].shape[0] == batch_size


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

        assert f'{ProprioceptiveType.POSITION.value}_{DecoderOutputKey.ROUTING_WEIGHTS.value}' in outputs
        assert f'{ProprioceptiveType.GRIPPER.value}_{DecoderOutputKey.ROUTING_WEIGHTS.value}' in outputs

        position_routing = outputs[f'{ProprioceptiveType.POSITION.value}_{DecoderOutputKey.ROUTING_WEIGHTS.value}']
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

        assert f'{ProprioceptiveType.POSITION.value}_{DecoderOutputKey.EXPERT_OUTPUTS.value}' in outputs
        assert f'{ProprioceptiveType.GRIPPER.value}_{DecoderOutputKey.EXPERT_OUTPUTS.value}' in outputs

        position_experts = outputs[f'{ProprioceptiveType.POSITION.value}_{DecoderOutputKey.EXPERT_OUTPUTS.value}']
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

        phase_logits = outputs[ObsKey.PHASE_LABEL.value]
        position_routing = outputs[f'{ProprioceptiveType.POSITION.value}_{DecoderOutputKey.ROUTING_WEIGHTS.value}']

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
            ObsKey.PHASE_LABEL.value,
            ProprioceptiveType.POSITION.value,
            ProprioceptiveType.GRIPPER.value,
            f'{ProprioceptiveType.POSITION.value}_{DecoderOutputKey.ROUTING_WEIGHTS.value}',
            f'{ProprioceptiveType.POSITION.value}_{DecoderOutputKey.EXPERT_OUTPUTS.value}',
            f'{ProprioceptiveType.GRIPPER.value}_{DecoderOutputKey.ROUTING_WEIGHTS.value}',
            f'{ProprioceptiveType.GRIPPER.value}_{DecoderOutputKey.EXPERT_OUTPUTS.value}',
            # VAE keys (LatentKey) removed - handled at algorithm level
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

        loss = outputs[ObsKey.PHASE_LABEL.value].sum()
        loss.backward()

        phase_head_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in decoder.action_heads[ObsKey.PHASE_LABEL.value].parameters()
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

        loss = outputs[ProprioceptiveType.POSITION.value].sum()
        loss.backward()

        moe_head = decoder.action_heads[ProprioceptiveType.POSITION.value]
        expert_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for expert in moe_head.experts
            for p in expert.parameters()
        )
        assert expert_has_grad
