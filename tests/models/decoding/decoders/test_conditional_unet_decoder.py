import pytest
import torch

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.constants import (
    GRIPPER_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    POSITION_ACTION_KEY,
    GripperType,
    OrientationRepresentation,
)
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.constants import FeatureType
from versatil.models.decoding.constants import TIMESTEP_KEY
from versatil.models.decoding.decoders.factory.conditional_unet_decoder import (
    ConditionalUNetDecoder,
)


@pytest.fixture
def observation_horizon():
    return 2


@pytest.fixture
def prediction_horizon():
    return 4


@pytest.fixture
def action_space():
    return ActionSpace(
        has_position=True,
        position_dim=3,
        has_orientation=True,
        orientation_dim=4,
        orientation_repr=OrientationRepresentation.QUATERNION.value,
        has_gripper=True,
        gripper_type=GripperType.BINARY.value,
        predict_in_camera_frame=False,
        deltas_as_actions=False,
        denoise_actions=False,
        task_has_phases=False,
    )


@pytest.fixture
def observation_space():
    return ObservationSpace(
        camera_keys=[],
        use_proprio_base_frame=True,
        use_proprio_camera_frame=False,
        use_language=False,
        use_gripper_state=False,
        gripper_type=GripperType.BINARY.value,
    )


@pytest.fixture
def action_heads(action_space):
    action_heads = {}

    if action_space.has_position:
        action_heads[POSITION_ACTION_KEY] = ActionHead(
            input_dim=action_space.position_dim,
            output_dim=action_space.position_dim,
            blocks=[],
        )

    if action_space.has_orientation:
        action_heads[ORIENTATION_ACTION_KEY] = ActionHead(
            input_dim=action_space.orientation_dim,
            output_dim=action_space.orientation_dim,
            blocks=[],
        )

    if action_space.has_gripper:
        action_heads[GRIPPER_ACTION_KEY] = ActionHead(
            input_dim=action_space.gripper_dim,
            output_dim=action_space.gripper_dim,
            blocks=[],
        )

    return action_heads


@pytest.fixture
def input_keys():
    return ["fused_features"]


def create_conditional_unet_decoder(
    input_keys: list[str],
    action_space: ActionSpace,
    action_heads: dict[str, ActionHead],
    observation_space: ObservationSpace,
    observation_horizon: int,
    prediction_horizon: int,
    device: str,
    embedding_dimension: int = 128,
    down_dimensions: list[int] = None,
    kernel_size: int = 5,
    num_groups: int = 8,
    use_local_conditioning: bool = False,
    condition_predict_scale: bool = False,
) -> ConditionalUNetDecoder:
    if down_dimensions is None:
        down_dimensions = [256, 512, 1024]

    decoder = ConditionalUNetDecoder(
        input_keys=input_keys,
        action_space=action_space,
        action_heads=action_heads,
        observation_space=observation_space,
        observation_horizon=observation_horizon,
        prediction_horizon=prediction_horizon,
        device=device,
        embedding_dimension=embedding_dimension,
        down_dimensions=down_dimensions,
        kernel_size=kernel_size,
        num_groups=num_groups,
        use_local_conditioning=use_local_conditioning,
        condition_predict_scale=condition_predict_scale,
    )
    return decoder.to(device)


@pytest.fixture
def flat_features_single(batch_size, device):
    return {"fused_features": torch.randn(batch_size, 256, device=device)}


@pytest.fixture
def flat_features_multiple(batch_size, device):
    return {
        "visual_features": torch.randn(batch_size, 128, device=device),
        "proprio_features": torch.randn(batch_size, 64, device=device),
    }


@pytest.fixture
def sequential_features_single(batch_size, observation_horizon, device):
    return {
        "temporal_features": torch.randn(batch_size, observation_horizon, 256, device=device)
    }


@pytest.fixture
def sequential_features_multiple(batch_size, observation_horizon, device):
    return {
        "visual_history": torch.randn(batch_size, observation_horizon, 128, device=device),
        "proprio_history": torch.randn(batch_size, observation_horizon, 64, device=device),
    }


@pytest.fixture
def spatial_features(batch_size, device):
    return {"spatial_features": torch.randn(batch_size, 256, 7, 7, device=device)}


@pytest.fixture
def mixed_features(batch_size, observation_horizon, device):
    return {
        "flat_feature": torch.randn(batch_size, 128, device=device),
        "sequential_feature": torch.randn(batch_size, observation_horizon, 64, device=device),
    }


@pytest.fixture
def timesteps(batch_size, device):
    return torch.randint(0, 100, (batch_size,), device=device, dtype=torch.long)


@pytest.fixture
def noisy_actions(batch_size, prediction_horizon, action_space, device):
    actions = {}

    if action_space.has_position:
        actions[POSITION_ACTION_KEY] = torch.randn(
            batch_size, prediction_horizon, action_space.position_dim, device=device
        )

    if action_space.has_orientation:
        actions[ORIENTATION_ACTION_KEY] = torch.randn(
            batch_size, prediction_horizon, action_space.orientation_dim, device=device
        )

    if action_space.has_gripper:
        actions[GRIPPER_ACTION_KEY] = torch.randn(
            batch_size, prediction_horizon, action_space.gripper_dim, device=device
        )

    return actions


@pytest.mark.unit
class TestConditionalUNetInitialization:

    def test_basic_initialization(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
    ):
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        assert decoder is not None
        assert decoder.action_dim == action_space.get_total_action_dim()
        assert decoder.prediction_horizon == prediction_horizon
        assert decoder.observation_horizon == observation_horizon
        assert decoder._unet is None

    def test_initialization_with_custom_params(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
    ):
        custom_embedding_dim = 256
        custom_down_dims = [128, 256, 512]
        custom_kernel_size = 3
        custom_num_groups = 4

        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            embedding_dimension=custom_embedding_dim,
            down_dimensions=custom_down_dims,
            kernel_size=custom_kernel_size,
            num_groups=custom_num_groups,
        )

        assert decoder.embedding_dimension == custom_embedding_dim
        assert decoder.down_dimensions == custom_down_dims
        assert decoder.kernel_size == custom_kernel_size
        assert decoder.num_groups == custom_num_groups

    def test_local_conditioning_raises_error(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
    ):
        with pytest.raises(NotImplementedError, match="Local conditioning is not yet implemented"):
            create_conditional_unet_decoder(
                input_keys=input_keys,
                action_space=action_space,
                action_heads=action_heads,
                observation_space=observation_space,
                observation_horizon=observation_horizon,
                prediction_horizon=prediction_horizon,
                device=device,
                use_local_conditioning=True,
            )

    def test_decoder_input_specification(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
    ):
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        assert decoder.decoder_input.keys == input_keys
        assert decoder.decoder_input.requires_actions is True
        assert FeatureType.SPATIAL.value in decoder.decoder_input.raises_for_types


@pytest.mark.unit
class TestConditionalUNetFeaturePreparation:

    def test_prepare_flat_features_single(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        flat_features_single,
        batch_size,
    ):
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        global_conditioning = decoder._prepare_global_conditioning(flat_features_single)

        assert global_conditioning.shape == (batch_size, 256)
        assert decoder._unet is not None
        assert decoder._global_conditioning_dimension == 256

    def test_prepare_flat_features_multiple(
        self,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        flat_features_multiple,
        batch_size,
    ):
        input_keys = ["visual_features", "proprio_features"]
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        global_conditioning = decoder._prepare_global_conditioning(flat_features_multiple)

        assert global_conditioning.shape == (batch_size, 192)
        assert decoder._global_conditioning_dimension == 192

    def test_prepare_sequential_features_flattening(
        self,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        sequential_features_single,
        batch_size,
    ):
        input_keys = ["temporal_features"]
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        global_conditioning = decoder._prepare_global_conditioning(sequential_features_single)

        expected_dimension = observation_horizon * 256
        assert global_conditioning.shape == (batch_size, expected_dimension)
        assert decoder._global_conditioning_dimension == expected_dimension

    def test_prepare_mixed_features(
        self,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        mixed_features,
        batch_size,
    ):
        input_keys = ["flat_feature", "sequential_feature"]
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        global_conditioning = decoder._prepare_global_conditioning(mixed_features)

        expected_dimension = 128 + (observation_horizon * 64)
        assert global_conditioning.shape == (batch_size, expected_dimension)

    def test_spatial_features_raise_error(
        self,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        spatial_features,
    ):
        input_keys = ["spatial_features"]
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        with pytest.raises(ValueError, match="Spatial features not supported"):
            decoder._prepare_global_conditioning(spatial_features)

    def test_missing_feature_key_raises_error(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        flat_features_single,
    ):
        decoder = create_conditional_unet_decoder(
            input_keys=["missing_key"],
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        with pytest.raises(ValueError, match="Expected feature .* not found"):
            decoder._prepare_global_conditioning(flat_features_single)

    def test_empty_features_raises_error(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
    ):
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        with pytest.raises(ValueError, match="Expected feature .* not found"):
            decoder._prepare_global_conditioning({})


@pytest.mark.unit
class TestConditionalUNetForwardPass:

    def test_forward_with_flat_features(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        flat_features_single,
        timesteps,
        noisy_actions,
        batch_size,
    ):
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        features = {**flat_features_single, TIMESTEP_KEY: timesteps}
        outputs = decoder.forward(features=features, actions=noisy_actions)

        assert POSITION_ACTION_KEY in outputs
        assert ORIENTATION_ACTION_KEY in outputs
        assert GRIPPER_ACTION_KEY in outputs

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

    def test_forward_with_sequential_features(
        self,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        sequential_features_single,
        timesteps,
        noisy_actions,
        batch_size,
    ):
        input_keys = ["temporal_features"]
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        features = {**sequential_features_single, TIMESTEP_KEY: timesteps}
        outputs = decoder.forward(features=features, actions=noisy_actions)

        assert POSITION_ACTION_KEY in outputs
        assert outputs[POSITION_ACTION_KEY].shape[0] == batch_size
        assert outputs[POSITION_ACTION_KEY].shape[1] == prediction_horizon

    def test_forward_missing_timesteps_raises_error(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        flat_features_single,
        noisy_actions,
    ):
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        with pytest.raises(ValueError, match="Missing 'timestep' in features dict"):
            decoder.forward(features=flat_features_single, actions=noisy_actions)

    def test_forward_missing_actions_raises_error(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        flat_features_single,
        timesteps,
    ):
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        features = {**flat_features_single, TIMESTEP_KEY: timesteps}

        with pytest.raises(ValueError, match="ConditionalUNetDecoder requires 'actions' parameter"):
            decoder.forward(features=features, actions=None)

    def test_forward_with_2d_timesteps(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        flat_features_single,
        noisy_actions,
        batch_size,
    ):
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        timesteps_2d = torch.randint(0, 100, (batch_size, 1), device=device, dtype=torch.long)
        features = {**flat_features_single, TIMESTEP_KEY: timesteps_2d}

        outputs = decoder.forward(features=features, actions=noisy_actions)

        assert POSITION_ACTION_KEY in outputs
        assert outputs[POSITION_ACTION_KEY].shape[0] == batch_size


@pytest.mark.unit
class TestConditionalUNetEdgeCases:

    def test_lazy_initialization_consistency(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        flat_features_single,
        timesteps,
        noisy_actions,
    ):
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        assert decoder._unet is None
        assert decoder._global_conditioning_dimension is None

        features = {**flat_features_single, TIMESTEP_KEY: timesteps}
        decoder.forward(features=features, actions=noisy_actions)

        assert decoder._unet is not None
        assert decoder._global_conditioning_dimension == 256

        unet_id = id(decoder._unet)
        decoder.forward(features=features, actions=noisy_actions)
        assert id(decoder._unet) == unet_id

    def test_gradient_flow(
        self,
        input_keys,
        action_space,
        action_heads,
        observation_space,
        observation_horizon,
        prediction_horizon,
        device,
        flat_features_single,
        timesteps,
        noisy_actions,
    ):
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )
        decoder.to(device)

        for param in decoder.parameters():
            param.requires_grad = True

        features = {**flat_features_single, TIMESTEP_KEY: timesteps}
        for key in features:
            if features[key].dtype == torch.float32:
                features[key].requires_grad = True

        for key in noisy_actions:
            noisy_actions[key].requires_grad = True

        outputs = decoder.forward(features=features, actions=noisy_actions)

        loss = sum(output.sum() for output in outputs.values())
        loss.backward()

        assert any(param.grad is not None for param in decoder.parameters() if param.requires_grad)


@pytest.mark.unit
@pytest.mark.parametrize(
    "observation_horizon,prediction_horizon",
    [
        (1, 4),
        (2, 8),
        (4, 16),
    ],
)
class TestConditionalUNetParametrized:

    def test_different_horizons(
        self,
        observation_horizon,
        prediction_horizon,
        action_space,
        action_heads,
        observation_space,
        device,
        batch_size,
    ):
        input_keys = ["features"]
        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        sequential_features = {
            "features": torch.randn(batch_size, observation_horizon, 256, device=device)
        }
        timesteps = torch.randint(0, 100, (batch_size,), device=device, dtype=torch.long)
        features = {**sequential_features, TIMESTEP_KEY: timesteps}

        noisy_actions = {}
        if action_space.has_position:
            noisy_actions[POSITION_ACTION_KEY] = torch.randn(
                batch_size, prediction_horizon, action_space.position_dim, device=device
            )
        if action_space.has_orientation:
            noisy_actions[ORIENTATION_ACTION_KEY] = torch.randn(
                batch_size, prediction_horizon, action_space.orientation_dim, device=device
            )
        if action_space.has_gripper:
            noisy_actions[GRIPPER_ACTION_KEY] = torch.randn(
                batch_size, prediction_horizon, action_space.gripper_dim, device=device
            )

        outputs = decoder.forward(features=features, actions=noisy_actions)

        for key, output in outputs.items():
            assert output.shape[0] == batch_size
            assert output.shape[1] == prediction_horizon


@pytest.mark.unit
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
class TestConditionalUNetBatchSizes:

    def test_different_batch_sizes(
        self,
        batch_size,
        action_space,
        action_heads,
        observation_space,
        device,
    ):
        input_keys = ["features"]
        observation_horizon = 2
        prediction_horizon = 4

        decoder = create_conditional_unet_decoder(
            input_keys=input_keys,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        flat_features = {"features": torch.randn(batch_size, 256, device=device)}
        timesteps = torch.randint(0, 100, (batch_size,), device=device, dtype=torch.long)
        features = {**flat_features, TIMESTEP_KEY: timesteps}

        noisy_actions = {}
        if action_space.has_position:
            noisy_actions[POSITION_ACTION_KEY] = torch.randn(
                batch_size, prediction_horizon, action_space.position_dim, device=device
            )
        if action_space.has_orientation:
            noisy_actions[ORIENTATION_ACTION_KEY] = torch.randn(
                batch_size, prediction_horizon, action_space.orientation_dim, device=device
            )
        if action_space.has_gripper:
            noisy_actions[GRIPPER_ACTION_KEY] = torch.randn(
                batch_size, prediction_horizon, action_space.gripper_dim, device=device
            )

        outputs = decoder.forward(features=features, actions=noisy_actions)

        for output in outputs.values():
            assert output.shape[0] == batch_size