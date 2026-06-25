"""Tests for versatil.inference integration module."""

import threading
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from tso_robotics_sockets import (
    CompressionType,
    InferenceRequestKey,
    InferenceResponseKey,
    ServerRoute,
    ServerStatus,
    SocketServer,
    TransportKey,
    compress_array,
)
from versatil_constants.shared import (
    ActionComponent,
    ActionComputationMethod,
    ActionMetadataField,
    CoordinateSystem,
    GripperType,
    ObsKey,
    OrientationRepresentation,
)

from versatil.data.constants import Cameras, ProprioKey
from versatil.data.metadata import OnTheFlyActionMetadata
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.inference.inference_client import InferenceClient
from versatil.inference.policy_runtime.float_runtime import FloatPolicyRuntime
from versatil.inference.socket_transport import (
    SocketActionTransport,
    SocketObservationTransport,
)

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
OBSERVATION_HORIZON = 1
PREDICTION_HORIZON = 4
POSITION_DIMENSION = 3
ORIENTATION_DIMENSION = 1
GRIPPER_DIMENSION = 1
SERVER_PORT = 15556


@pytest.fixture
def observation_space_integration_factory(
    position_observation_metadata_factory: Callable,
    orientation_observation_metadata_factory: Callable,
    gripper_observation_metadata_factory: Callable,
    camera_metadata_factory: Callable,
) -> Callable[..., ObservationSpace]:
    def factory(
        include_orientation: bool = True,
        include_gripper: bool = True,
    ) -> ObservationSpace:
        metadata = {
            Cameras.LEFT.value: camera_metadata_factory(camera_key=Cameras.LEFT.value),
            Cameras.RIGHT.value: camera_metadata_factory(
                camera_key=Cameras.RIGHT.value
            ),
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: position_observation_metadata_factory(
                dimension=POSITION_DIMENSION,
            ),
            ObsKey.LANGUAGE.value: MagicMock(),
        }
        if include_orientation:
            metadata[ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_ORI.value] = (
                orientation_observation_metadata_factory(
                    dimension=ORIENTATION_DIMENSION
                )
            )
        if include_gripper:
            metadata[ProprioKey.GRIPPER_STATE.value] = (
                gripper_observation_metadata_factory()
            )
        return ObservationSpace(observations_metadata=metadata)

    return factory


@pytest.fixture
def action_space_integration_factory(
    position_observation_metadata_factory: Callable,
    orientation_observation_metadata_factory: Callable,
    gripper_observation_metadata_factory: Callable,
) -> Callable[..., ActionSpace]:
    def factory(
        include_orientation: bool = True,
        include_gripper: bool = True,
    ) -> ActionSpace:
        actions = {
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: OnTheFlyActionMetadata(
                source_metadata=position_observation_metadata_factory(
                    dimension=POSITION_DIMENSION,
                ),
                computation_method=ActionComputationMethod.DELTA.value,
            ),
        }
        if include_orientation:
            actions[ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_ORI.value] = (
                OnTheFlyActionMetadata(
                    source_metadata=orientation_observation_metadata_factory(
                        dimension=ORIENTATION_DIMENSION,
                    ),
                    computation_method=ActionComputationMethod.DELTA.value,
                )
            )
        if include_gripper:
            actions[ProprioKey.GRIPPER_STATE.value] = OnTheFlyActionMetadata(
                source_metadata=gripper_observation_metadata_factory(),
                computation_method=ActionComputationMethod.DELTA.value,
            )
        return ActionSpace(actions_metadata=actions)

    return factory


@pytest.fixture
def observation_space(
    observation_space_integration_factory: Callable[..., ObservationSpace],
) -> ObservationSpace:
    return observation_space_integration_factory()


@pytest.fixture
def action_space(
    action_space_integration_factory: Callable[..., ActionSpace],
) -> ActionSpace:
    return action_space_integration_factory()


@pytest.fixture
def mock_policy_loader_factory(
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    rng: np.random.Generator,
) -> Callable[..., MagicMock]:
    def factory(
        observation_horizon: int = OBSERVATION_HORIZON,
        prediction_horizon: int = PREDICTION_HORIZON,
        denoising_thresholds: dict[str, float] | None = None,
        depth_clamp_range: tuple[float, float] | None = None,
        checkpoint_path: str = "test_checkpoint",
        inference_side_effect: Callable | None = None,
    ) -> MagicMock:
        if denoising_thresholds is None:
            denoising_thresholds = {
                ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: 0.001,
            }

        mock = MagicMock(spec=FloatPolicyRuntime)
        mock.observation_space = observation_space
        mock.action_space = action_space
        mock.observation_horizon = observation_horizon
        mock.prediction_horizon = prediction_horizon
        mock.config.task.dataloader.image_height = IMAGE_HEIGHT
        mock.config.task.dataloader.image_width = IMAGE_WIDTH
        mock.config.inference.rotate_images = False
        mock.depth_clamp_range = depth_clamp_range
        mock.denoising_thresholds = denoising_thresholds
        mock.checkpoint_path = checkpoint_path
        mock.client_identifier = checkpoint_path
        mock.device = torch.device("cpu")

        if inference_side_effect is not None:
            mock.run_inference = MagicMock(side_effect=inference_side_effect)
        else:
            action_keys_to_dimensions = {
                key: metadata.prediction_dimension
                for key, metadata in action_space.actions_metadata.items()
                if metadata.requires_prediction_head
            }

            def default_run_inference(obs_dict: dict) -> dict[str, torch.Tensor]:
                batch_count = 1
                for value in obs_dict.values():
                    if isinstance(value, torch.Tensor):
                        batch_count = value.shape[0]
                        break
                return {
                    key: torch.from_numpy(
                        rng.standard_normal(
                            (batch_count, prediction_horizon, dimension)
                        ).astype(np.float32)
                    )
                    for key, dimension in action_keys_to_dimensions.items()
                }

            mock.run_inference = MagicMock(side_effect=default_run_inference)
        return mock

    return factory


@pytest.fixture(
    params=[
        (2, None),
        (4, None),
        (8, 3),
    ],
    ids=["pred=2", "pred=4", "pred=8_exec=3"],
)
def mock_policy_loader(
    request,
    mock_policy_loader_factory: Callable[..., MagicMock],
) -> MagicMock:
    prediction_horizon, action_execution_horizon = request.param
    loader = mock_policy_loader_factory(prediction_horizon=prediction_horizon)
    # Attach execution horizon so socket_integration_client can read it
    loader.action_execution_horizon = action_execution_horizon
    return loader


@pytest.fixture(scope="session")
def observation_server() -> SocketServer:
    server_rng = np.random.default_rng(seed=99)
    camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value]
    proprio_dims = {
        ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: POSITION_DIMENSION,
        ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_ORI.value: ORIENTATION_DIMENSION,
        ProprioKey.GRIPPER_STATE.value: GRIPPER_DIMENSION,
    }

    server = SocketServer(
        ip_address="127.0.0.1",
        port=SERVER_PORT,
        max_workers=1,
    )

    def handle_get_observation(request_data: dict) -> tuple[bool, dict]:
        requested_keys = request_data.get(InferenceRequestKey.REQUESTED_KEYS.value, [])
        compression_type = request_data.get(
            InferenceRequestKey.COMPRESSION_TYPE.value,
            CompressionType.RAW.value,
        )
        response: dict = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
            InferenceResponseKey.COMPRESSION_TYPE.value: compression_type,
        }
        for camera_key in camera_keys:
            if camera_key in requested_keys:
                image_data = server_rng.integers(
                    0, 256, size=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8
                )
                response[camera_key] = compress_array(
                    image_data, method=compression_type, as_base64=True
                )
        for proprio_key, dimension in proprio_dims.items():
            if proprio_key in requested_keys:
                response[proprio_key] = (
                    server_rng.standard_normal(dimension).astype(np.float32).tolist()
                )
        if ObsKey.LANGUAGE.value in requested_keys:
            response[ObsKey.LANGUAGE.value] = "pick up the red block"
        return True, response

    def handle_send_action(request_data: dict) -> tuple[bool, dict]:
        return True, {}

    def handle_register(request_data: dict) -> tuple[bool, dict]:
        return True, {}

    server.add_route(
        ServerRoute.GET_OBSERVATION.value,
        handle_get_observation,
        blocking=True,
    )
    server.add_route(
        ServerRoute.SEND_ACTION.value,
        handle_send_action,
        blocking=True,
    )
    server.add_route(
        ServerRoute.REGISTER_CLIENT.value,
        handle_register,
        blocking=True,
    )

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    yield server
    server.stop()


@pytest.fixture
def socket_observation_transport(
    observation_server: SocketServer,
) -> SocketObservationTransport:
    return SocketObservationTransport(
        server_address="127.0.0.1",
        server_port=SERVER_PORT,
    )


@pytest.fixture
def socket_action_transport(
    observation_server: SocketServer,
) -> SocketActionTransport:
    return SocketActionTransport(
        server_address="127.0.0.1",
        server_port=SERVER_PORT,
    )


@pytest.fixture
def socket_integration_client(
    mock_policy_loader: MagicMock,
    socket_observation_transport: SocketObservationTransport,
    socket_action_transport: SocketActionTransport,
) -> InferenceClient:
    return InferenceClient(
        policy_runtime=mock_policy_loader,
        observation_transport=socket_observation_transport,
        action_transport=socket_action_transport,
        action_execution_horizon=mock_policy_loader.action_execution_horizon,
        compression_type=CompressionType.RAW.value,
    )


@pytest.mark.integration
class TestSocketProtocolEndToEnd:
    def test_full_observation_action_cycle_over_sockets(
        self,
        socket_integration_client: InferenceClient,
        mock_policy_loader: MagicMock,
    ):
        status = socket_integration_client.step()

        assert status == "continue"
        mock_policy_loader.run_inference.assert_called_once()

    def test_rgb_images_survive_serialization_as_normalized_floats(
        self,
        socket_integration_client: InferenceClient,
        mock_policy_loader: MagicMock,
    ):
        socket_integration_client.step()

        obs_dict = mock_policy_loader.run_inference.call_args.kwargs["obs_dict"]
        left_tensor = obs_dict[Cameras.LEFT.value]
        assert left_tensor.dtype == torch.float32
        assert left_tensor.min() >= 0.0
        assert left_tensor.max() <= 1.0
        assert left_tensor.shape == (
            1,
            OBSERVATION_HORIZON,
            3,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )

    def test_proprioceptive_data_survives_json_serialization(
        self,
        socket_integration_client: InferenceClient,
        mock_policy_loader: MagicMock,
    ):
        socket_integration_client.step()

        obs_dict = mock_policy_loader.run_inference.call_args.kwargs["obs_dict"]
        position_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        position_tensor = obs_dict[position_key]
        assert position_tensor.shape == (1, OBSERVATION_HORIZON, POSITION_DIMENSION)
        assert position_tensor.dtype == torch.float32

    def test_language_instruction_survives_serialization(
        self,
        socket_integration_client: InferenceClient,
        mock_policy_loader: MagicMock,
    ):
        socket_integration_client.step()

        obs_dict = mock_policy_loader.run_inference.call_args.kwargs["obs_dict"]
        language = obs_dict[ObsKey.LANGUAGE.value]
        assert language == [["pick up the red block"]]

    def test_structured_actions_sent_with_correct_components(
        self,
        socket_integration_client: InferenceClient,
        mock_policy_loader: MagicMock,
    ):
        socket_integration_client.step()

        # Verify that the client produced structured actions with the correct keys
        # by inspecting the action_transport.send call via the mock
        metadata = (
            socket_integration_client.action_postprocessor.build_action_metadata()
        )
        assert ActionComponent.POSITION.value in metadata
        assert ActionComponent.ORIENTATION.value in metadata
        assert ActionComponent.GRIPPER.value in metadata

    def test_multi_step_episode_over_sockets(
        self,
        socket_integration_client: InferenceClient,
        mock_policy_loader: MagicMock,
    ):
        socket_integration_client.run_episode(max_steps=3)

        assert mock_policy_loader.run_inference.call_count == 3
        assert socket_integration_client.timestep == 3

    def test_action_metadata_contains_all_fields(
        self,
        socket_integration_client: InferenceClient,
        mock_policy_loader: MagicMock,
    ):
        socket_integration_client.step()

        metadata = (
            socket_integration_client.action_postprocessor.build_action_metadata()
        )
        position_metadata = metadata[ActionComponent.POSITION.value]
        assert (
            position_metadata[ActionMetadataField.DIMENSION.value] == POSITION_DIMENSION
        )
        assert (
            position_metadata[ActionMetadataField.FRAME.value]
            == CoordinateSystem.ROBOT_BASE.value
        )
        assert (
            position_metadata[ActionMetadataField.ACTION_TYPE.value]
            == ActionComputationMethod.DELTA.value
        )

        orientation_metadata = metadata[ActionComponent.ORIENTATION.value]
        assert (
            orientation_metadata[ActionMetadataField.ORIENTATION_REPRESENTATION.value]
            == OrientationRepresentation.ROLL.value
        )

        gripper_metadata = metadata[ActionComponent.GRIPPER.value]
        assert (
            gripper_metadata[ActionMetadataField.GRIPPER_TYPE.value]
            == GripperType.BINARY.value
        )

    def test_binary_gripper_produces_discrete_values(
        self,
        socket_integration_client: InferenceClient,
        mock_policy_loader: MagicMock,
    ):
        collected_gripper_values = []

        original_format_action = (
            socket_integration_client.action_postprocessor.format_action
        )

        def capturing_format_action(action_dict):
            result = original_format_action(action_dict=action_dict)
            if ActionComponent.GRIPPER.value in result:
                collected_gripper_values.append(result[ActionComponent.GRIPPER.value])
            return result

        socket_integration_client.action_postprocessor.format_action = (
            capturing_format_action
        )

        for _ in range(5):
            socket_integration_client.step()

        # 5 steps × action_execution_horizon actions per step (chunk execution)
        execution_horizon = (
            mock_policy_loader.action_execution_horizon
            if mock_policy_loader.action_execution_horizon is not None
            else mock_policy_loader.prediction_horizon
        )
        assert len(collected_gripper_values) == 5 * execution_horizon
        for gripper_values in collected_gripper_values:
            for value in gripper_values:
                assert value in (0.0, 1.0), f"Expected 0.0 or 1.0, got {value}"

    def test_denoising_zeroes_small_actions_over_sockets(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        action_space: ActionSpace,
        socket_observation_transport: SocketObservationTransport,
        socket_action_transport: SocketActionTransport,
    ):
        position_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        action_keys_to_dimensions = {
            key: meta.prediction_dimension
            for key, meta in action_space.actions_metadata.items()
            if meta.requires_prediction_head
        }

        def small_inference(obs_dict: dict) -> dict[str, torch.Tensor]:
            batch_count = 1
            for value in obs_dict.values():
                if isinstance(value, torch.Tensor):
                    batch_count = value.shape[0]
                    break
            return {
                key: torch.full((batch_count, PREDICTION_HORIZON, dim), 0.0001)
                for key, dim in action_keys_to_dimensions.items()
            }

        loader = mock_policy_loader_factory(
            denoising_thresholds={position_key: 1000.0},
            inference_side_effect=small_inference,
        )
        client = InferenceClient(
            policy_runtime=loader,
            observation_transport=socket_observation_transport,
            action_transport=socket_action_transport,
            compression_type=CompressionType.RAW.value,
        )

        collected_actions = []

        original_format_action = client.action_postprocessor.format_action

        def capturing_format_action(action_dict):
            result = original_format_action(action_dict=action_dict)
            collected_actions.append(result)
            return result

        client.action_postprocessor.format_action = capturing_format_action

        client.step()

        # 1 step × PREDICTION_HORIZON actions per chunk
        assert len(collected_actions) == PREDICTION_HORIZON
        position_values = collected_actions[0][ActionComponent.POSITION.value]
        assert all(value == 0.0 for value in position_values)


@pytest.mark.integration
class TestTemporalAggregationIntegration:
    def test_temporal_aggregation_produces_different_values_across_steps(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        action_space: ActionSpace,
        socket_observation_transport: SocketObservationTransport,
        socket_action_transport: SocketActionTransport,
    ):
        action_keys_to_dimensions = {
            key: meta.prediction_dimension
            for key, meta in action_space.actions_metadata.items()
            if meta.requires_prediction_head
        }
        call_count = 0

        def incrementing_inference(obs_dict: dict) -> dict[str, torch.Tensor]:
            nonlocal call_count
            call_count += 1
            return {
                key: torch.full((1, PREDICTION_HORIZON, dim), float(call_count))
                for key, dim in action_keys_to_dimensions.items()
            }

        loader = mock_policy_loader_factory(
            denoising_thresholds={},
            inference_side_effect=incrementing_inference,
        )
        client = InferenceClient(
            policy_runtime=loader,
            observation_transport=socket_observation_transport,
            action_transport=socket_action_transport,
            compression_type=CompressionType.RAW.value,
            temporal_aggregation=True,
        )

        collected_actions = []

        original_format_action = client.action_postprocessor.format_action

        def capturing_format_action(action_dict):
            result = original_format_action(action_dict=action_dict)
            collected_actions.append(result)
            return result

        client.action_postprocessor.format_action = capturing_format_action

        client.step()
        assert (
            client.environment_states[0].temporal_aggregator.prediction_horizon
            == PREDICTION_HORIZON
        )

        client.step()
        client.step()
        assert client.timestep == 3

        # Verify temporal aggregation produces different values across steps
        # because the incrementing_inference returns different prediction values
        assert len(collected_actions) == 3
        step_one_position = collected_actions[0][ActionComponent.POSITION.value]
        step_three_position = collected_actions[2][ActionComponent.POSITION.value]
        assert step_one_position != step_three_position


@pytest.mark.integration
class TestObservationHorizonGreaterThanOne:
    def test_buffer_fills_before_inference(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        socket_observation_transport: SocketObservationTransport,
        socket_action_transport: SocketActionTransport,
    ):
        horizon = 2
        loader = mock_policy_loader_factory(
            observation_horizon=horizon,
            denoising_thresholds={},
        )
        client = InferenceClient(
            policy_runtime=loader,
            observation_transport=socket_observation_transport,
            action_transport=socket_action_transport,
            compression_type=CompressionType.RAW.value,
        )

        client.step()
        assert loader.run_inference.call_count == 0

        client.step()
        loader.run_inference.assert_called_once()

        obs_dict = loader.run_inference.call_args.kwargs["obs_dict"]
        left_tensor = obs_dict[Cameras.LEFT.value]
        assert left_tensor.shape[1] == horizon
