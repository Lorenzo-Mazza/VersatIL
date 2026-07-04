"""Tests for versatil.inference.inference_client module."""

import logging
import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tso_robotics_sockets import (
    CompressionType,
    InferenceResponseKey,
    ServerStatus,
    TransportKey,
)
from versatil_constants.shared import ObsKey

from versatil.data.constants import Cameras
from versatil.data.metadata import (
    DepthCameraMetadata,
    ObservationMetadata,
    RGBCameraMetadata,
)
from versatil.inference.action_postprocessor import ActionPostprocessor
from versatil.inference.inference_client import (
    EpisodeStatus,
    InferenceClient,
    infer_rotate_images,
)
from versatil.inference.observation_preprocessor import ObservationPreprocessor
from versatil.inference.policy_runtime.float_runtime import FloatPolicyRuntime
from versatil.inference.protocol import ActionTransport, ObservationTransport
from versatil.inference.temporal_aggregation import TemporalAggregator


@pytest.fixture
def mock_observation_space_factory() -> Callable[..., MagicMock]:
    def factory(
        camera_keys: list[str] | None = None,
        state_keys: list[str] | None = None,
        has_language: bool = False,
    ) -> MagicMock:
        if camera_keys is None:
            camera_keys = ["left", "right"]
        if state_keys is None:
            state_keys = ["proprio_robot_frame"]

        cameras: dict[str, MagicMock] = {}
        for key in camera_keys:
            if key == Cameras.DEPTH.value:
                metadata = MagicMock(spec=DepthCameraMetadata)
                metadata.dtype = "float32"
                metadata.channels = 1
                metadata.image_height = 64
                metadata.image_width = 64
                metadata.max_pixel_value = None
                metadata.is_rgb = False
                metadata.is_depth = True
                metadata.is_single_channel = True
            else:
                metadata = MagicMock(spec=RGBCameraMetadata)
                metadata.dtype = "uint8"
                metadata.channels = 3
                metadata.image_height = 64
                metadata.image_width = 64
                metadata.max_pixel_value = 255.0
                metadata.is_rgb = True
                metadata.is_depth = False
                metadata.is_single_channel = False
            cameras[key] = metadata
        state_metadata = {
            key: MagicMock(dtype="float32", is_numerical=True) for key in state_keys
        }

        observations_metadata: dict[str, MagicMock] = {}
        observations_metadata.update(cameras)
        observations_metadata.update(state_metadata)
        if has_language:
            observations_metadata[ObsKey.LANGUAGE.value] = MagicMock()

        mock = MagicMock()
        mock.cameras = cameras
        mock.numerical_observations = state_metadata
        mock.observations_metadata = observations_metadata
        return mock

    return factory


@pytest.fixture
def mock_action_space_factory() -> Callable[..., MagicMock]:
    def factory(
        action_keys_to_dimensions: dict[str, int] | None = None,
    ) -> MagicMock:
        if action_keys_to_dimensions is None:
            action_keys_to_dimensions = {"position": 3}

        actions_metadata = {}
        for key, dimension in action_keys_to_dimensions.items():
            meta = MagicMock()
            meta.prediction_dimension = dimension
            meta.requires_prediction_head = True
            actions_metadata[key] = meta

        mock = MagicMock()
        mock.actions_metadata = actions_metadata
        return mock

    return factory


@pytest.fixture
def mock_policy_loader_factory(
    mock_observation_space_factory: Callable[..., MagicMock],
    mock_action_space_factory: Callable[..., MagicMock],
) -> Callable[..., MagicMock]:
    def factory(
        camera_keys: list[str] | None = None,
        state_keys: list[str] | None = None,
        has_language: bool = False,
        action_keys_to_dimensions: dict[str, int] | None = None,
        prediction_horizon: int = 4,
        observation_horizon: int = 2,
        image_height: int = 64,
        image_width: int = 64,
        rotate_images: bool = False,
        depth_clamp_range: tuple[float, float] | None = None,
    ) -> MagicMock:
        mock = MagicMock(spec=FloatPolicyRuntime)
        mock.observation_space = mock_observation_space_factory(
            camera_keys=camera_keys,
            state_keys=state_keys,
            has_language=has_language,
        )
        mock.action_space = mock_action_space_factory(
            action_keys_to_dimensions=action_keys_to_dimensions,
        )
        mock.prediction_horizon = prediction_horizon
        mock.observation_horizon = observation_horizon
        mock.device = torch.device("cpu")
        mock.checkpoint_path = "/mock/checkpoint"
        mock.client_identifier = "/mock/checkpoint/latest-99"
        dataset_schema = (
            {
                "_target_": "versatil.data.raw.schemas.lerobot.LeRobotDatasetSchemaV30",
                "dataset_type": "libero",
            }
            if rotate_images
            else {
                "_target_": "versatil.data.raw.schemas.hdf5.Hdf5DatasetSchema",
            }
        )
        mock.config = OmegaConf.create(
            {
                "task": {
                    "dataloader": {
                        "image_height": image_height,
                        "image_width": image_width,
                    },
                    "dataset_schema": dataset_schema,
                }
            }
        )
        mock.depth_clamp_ranges = depth_clamp_range
        mock.denoising_thresholds = {}
        return mock

    return factory


@pytest.fixture
def mock_observation_transport() -> MagicMock:
    return MagicMock(spec=ObservationTransport)


@pytest.fixture
def mock_action_transport() -> MagicMock:
    mock = MagicMock(spec=ActionTransport)
    mock.close = MagicMock()
    return mock


@pytest.fixture
def inference_client_factory(
    mock_policy_loader_factory: Callable[..., MagicMock],
    mock_observation_transport: MagicMock,
    mock_action_transport: MagicMock,
) -> Callable[..., InferenceClient]:
    def factory(
        camera_keys: list[str] | None = None,
        state_keys: list[str] | None = None,
        has_language: bool = False,
        action_keys_to_dimensions: dict[str, int] | None = None,
        prediction_horizon: int = 4,
        observation_horizon: int = 2,
        temporal_aggregation: bool = False,
        action_execution_horizon: int | None = None,
        compression_type: str = CompressionType.RAW.value,
        timing_log: bool = False,
        update_rate_hz: float | None = None,
    ) -> InferenceClient:
        policy_loader = mock_policy_loader_factory(
            camera_keys=camera_keys,
            state_keys=state_keys,
            has_language=has_language,
            action_keys_to_dimensions=action_keys_to_dimensions,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
        )
        return InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=mock_action_transport,
            temporal_aggregation=temporal_aggregation,
            action_execution_horizon=action_execution_horizon,
            compression_type=compression_type,
            timing_log=timing_log,
            update_rate_hz=update_rate_hz,
        )

    return factory


@pytest.mark.unit
class TestInferenceClientInitialization:
    @pytest.mark.parametrize("temporal_aggregation", [True, False])
    @pytest.mark.parametrize(
        "camera_keys",
        [["left"], ["left", "right"]],
    )
    def test_stores_configuration(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        temporal_aggregation: bool,
        camera_keys: list[str],
    ):
        client = inference_client_factory(
            temporal_aggregation=temporal_aggregation,
            camera_keys=camera_keys,
        )

        assert client.temporal_aggregation == temporal_aggregation
        assert client.camera_keys == camera_keys

    def test_derives_camera_and_state_keys_from_observation_space(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            camera_keys=["left", "right"],
            state_keys=["proprio_robot_frame"],
        )

        assert client.camera_keys == ["left", "right"]
        assert client.state_keys == ["proprio_robot_frame"]
        assert "left" in client.all_observation_keys
        assert "right" in client.all_observation_keys
        assert "proprio_robot_frame" in client.all_observation_keys

    def test_derives_action_keys_to_dimensions_from_action_space(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            action_keys_to_dimensions={"position": 3, "orientation": 4},
        )

        assert client.action_keys_to_dimensions == {
            "position": 3,
            "orientation": 4,
        }

    def test_has_language_true_when_language_in_observation_space(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(has_language=True)

        assert ObsKey.LANGUAGE.value in client.all_observation_keys

    def test_has_language_false_when_language_absent(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(has_language=False)

        assert ObsKey.LANGUAGE.value not in client.all_observation_keys

    def test_creates_observation_preprocessor(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(camera_keys=["left"])

        assert client.observation_preprocessor.camera_keys == ["left"]

    def test_creates_action_postprocessor(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            action_keys_to_dimensions={"position": 3},
        )

        assert (
            client.action_postprocessor.action_space.actions_metadata[
                "position"
            ].prediction_dimension
            == 3
        )
        assert client.action_postprocessor.denoising_thresholds == {}

    def test_initial_timestep_is_zero(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory()

        assert client.timestep == 0

    def test_environment_states_initially_empty(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory()

        assert client.environment_states == {}

    def test_raises_when_action_execution_horizon_exceeds_prediction_horizon(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "action_execution_horizon (10) cannot exceed prediction_horizon (4)."
            ),
        ):
            inference_client_factory(
                prediction_horizon=4,
                action_execution_horizon=10,
            )


@pytest.mark.unit
class TestBucketObservationKeys:
    def test_buckets_cameras_state_and_language(
        self,
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        observation_space = mock_observation_space_factory(
            camera_keys=["left", "right"],
            state_keys=["proprio_robot_frame", "gripper_state_obs"],
            has_language=True,
        )

        camera_keys, state_keys, has_language = (
            InferenceClient._bucket_observation_keys(
                observation_space=observation_space
            )
        )

        assert camera_keys == ["left", "right"]
        assert state_keys == ["proprio_robot_frame", "gripper_state_obs"]
        assert has_language is True

    def test_has_language_false_when_language_key_absent(
        self,
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        observation_space = mock_observation_space_factory(
            camera_keys=["left"],
            state_keys=["proprio_robot_frame"],
            has_language=False,
        )

        _, _, has_language = InferenceClient._bucket_observation_keys(
            observation_space=observation_space
        )

        assert has_language is False

    def test_empty_when_no_observations(
        self,
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        observation_space = mock_observation_space_factory(
            camera_keys=[],
            state_keys=[],
            has_language=False,
        )

        camera_keys, state_keys, has_language = (
            InferenceClient._bucket_observation_keys(
                observation_space=observation_space
            )
        )

        assert camera_keys == []
        assert state_keys == []
        assert has_language is False

    def test_raises_when_observation_has_no_dispatch(
        self,
        mock_observation_space_factory: Callable[..., MagicMock],
    ):
        observation_space = mock_observation_space_factory(
            camera_keys=["left"],
            state_keys=["proprio_robot_frame"],
            has_language=False,
        )
        unsupported_meta = MagicMock(spec=ObservationMetadata)
        unsupported_meta.is_numerical = False
        observation_space.observations_metadata["mystery_categorical"] = (
            unsupported_meta
        )

        with pytest.raises(
            TypeError,
            match=re.escape(
                "Observations ['mystery_categorical'] have no inference dispatch; "
                "expected CameraMetadata, numerical ObservationMetadata, or the "
                f"language key '{ObsKey.LANGUAGE.value}'."
            ),
        ):
            InferenceClient._bucket_observation_keys(
                observation_space=observation_space
            )


@pytest.mark.unit
class TestCheckStatus:
    def test_finished_status_returns_finished(self):
        response = {
            TransportKey.STATUS.value: ServerStatus.FINISHED.value,
        }

        result = InferenceClient._check_status(response=response)

        assert result == EpisodeStatus.FINISHED.value

    def test_error_status_raises_runtime_error(self):
        error_message = "simulation crashed"
        response = {
            TransportKey.STATUS.value: ServerStatus.ERROR.value,
            TransportKey.ERROR_MSG.value: error_message,
        }

        with pytest.raises(
            RuntimeError,
            match=re.escape(f"Server error: {error_message}"),
        ):
            InferenceClient._check_status(response=response)

    @pytest.mark.parametrize(
        "status",
        [
            ServerStatus.PROCESSING.value,
            ServerStatus.CREATING_ENV.value,
        ],
    )
    def test_transient_status_returns_skip(self, status: str):
        response = {TransportKey.STATUS.value: status}

        result = InferenceClient._check_status(response=response)

        assert result == EpisodeStatus.SKIP.value

    def test_waiting_action_status_returns_continue(self):
        response = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
        }

        result = InferenceClient._check_status(response=response)

        assert result == EpisodeStatus.CONTINUE.value

    def test_unknown_status_returns_continue(self):
        response = {TransportKey.STATUS.value: "some_unknown_status"}

        result = InferenceClient._check_status(response=response)

        assert result == EpisodeStatus.CONTINUE.value

    def test_missing_status_returns_continue(self):
        response = {}

        result = InferenceClient._check_status(response=response)

        assert result == EpisodeStatus.CONTINUE.value


@pytest.mark.unit
class TestInferRotateImages:
    @pytest.mark.parametrize(
        "dataset_schema, expected",
        [
            (
                {
                    "_target_": (
                        "versatil.data.raw.schemas.lerobot.LeRobotDatasetSchemaV30"
                    ),
                    "dataset_type": "libero",
                },
                True,
            ),
            (
                {
                    "_target_": (
                        "versatil.data.raw.schemas.lerobot.LeRobotDatasetSchemaV30"
                    ),
                    "dataset_type": "metaworld",
                },
                False,
            ),
            (
                {"_target_": "versatil.data.raw.schemas.hdf5.Hdf5DatasetSchema"},
                False,
            ),
        ],
        ids=["libero_lerobot", "metaworld_lerobot", "libero_hdf5"],
    )
    def test_rotation_derived_from_dataset_schema(
        self, dataset_schema: dict, expected: bool
    ):
        config = OmegaConf.create({"task": {"dataset_schema": dataset_schema}})
        assert infer_rotate_images(config=config) is expected

    def test_missing_schema_disables_rotation(self):
        config = OmegaConf.create({"task": {}})
        assert infer_rotate_images(config=config) is False


@pytest.mark.unit
class TestHandleResetSignal:
    def test_resets_buffer_of_indicated_environment(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            observation_horizon=1,
        )
        state = client._create_environment_state()
        state.observation_buffer.add(
            observations={
                "left": np.zeros((64, 64, 3), dtype=np.uint8),
                "proprio": np.zeros(3, dtype=np.float32),
            }
        )
        client.environment_states[0] = state
        assert state.observation_buffer.is_ready()

        client._handle_reset_signal(
            response={InferenceResponseKey.RESET_ENVIRONMENT_INDICES.value: [0]}
        )

        assert not state.observation_buffer.is_ready()

    def test_resets_temporal_aggregator_of_indicated_environment(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            temporal_aggregation=True,
            observation_horizon=1,
        )
        state = client._create_environment_state()
        state.temporal_aggregator.timestep = 5
        client.environment_states[0] = state

        client._handle_reset_signal(
            response={InferenceResponseKey.RESET_ENVIRONMENT_INDICES.value: [0]}
        )

        assert state.temporal_aggregator.timestep == 0

    def test_does_not_reset_non_indicated_environments(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            observation_horizon=1,
        )
        state_zero = client._create_environment_state()
        state_zero.observation_buffer.add(
            observations={
                "left": np.zeros((64, 64, 3), dtype=np.uint8),
                "proprio": np.zeros(3, dtype=np.float32),
            }
        )
        state_one = client._create_environment_state()
        state_one.observation_buffer.add(
            observations={
                "left": np.zeros((64, 64, 3), dtype=np.uint8),
                "proprio": np.zeros(3, dtype=np.float32),
            }
        )
        client.environment_states[0] = state_zero
        client.environment_states[1] = state_one

        client._handle_reset_signal(
            response={InferenceResponseKey.RESET_ENVIRONMENT_INDICES.value: [0]}
        )

        assert state_one.observation_buffer.is_ready()

    def test_ignores_unknown_environment_indices(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory()

        client._handle_reset_signal(
            response={InferenceResponseKey.RESET_ENVIRONMENT_INDICES.value: [99]}
        )

    def test_no_op_when_reset_key_missing(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            observation_horizon=1,
        )
        state = client._create_environment_state()
        state.observation_buffer.add(
            observations={
                "left": np.zeros((64, 64, 3), dtype=np.uint8),
                "proprio": np.zeros(3, dtype=np.float32),
            }
        )
        client.environment_states[0] = state

        client._handle_reset_signal(response={})

        assert state.observation_buffer.is_ready()


@pytest.mark.unit
class TestUpdateEnvironmentStates:
    def test_creates_new_state_for_unknown_environment(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
        )

        observations = {
            "left": np.zeros((64, 64, 3), dtype=np.uint8),
            "proprio": np.zeros(3, dtype=np.float32),
        }
        client._update_environment_states(
            per_environment_observations={0: observations}
        )

        assert 0 in client.environment_states
        state = client.environment_states[0]
        assert state.observation_buffer.buffer_size > 0

    def test_adds_observation_to_existing_buffer(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            observation_horizon=2,
        )
        observations = {
            "left": np.zeros((64, 64, 3), dtype=np.uint8),
            "proprio": np.zeros(3, dtype=np.float32),
        }

        client._update_environment_states(
            per_environment_observations={0: observations}
        )
        assert not client.environment_states[0].observation_buffer.is_ready()

        client._update_environment_states(
            per_environment_observations={0: observations}
        )
        assert client.environment_states[0].observation_buffer.is_ready()

    def test_handles_multiple_environments(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
        )
        observations = {
            "left": np.zeros((64, 64, 3), dtype=np.uint8),
            "proprio": np.zeros(3, dtype=np.float32),
        }

        client._update_environment_states(
            per_environment_observations={
                0: observations,
                1: observations,
                5: observations,
            }
        )

        assert 0 in client.environment_states
        assert 1 in client.environment_states
        assert 5 in client.environment_states


@pytest.mark.unit
class TestRemoveInactiveEnvironments:
    def test_removes_environments_not_in_response(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory()
        client.environment_states[0] = client._create_environment_state()
        client.environment_states[1] = client._create_environment_state()
        client.environment_states[2] = client._create_environment_state()

        client._remove_inactive_environments(
            per_environment_observations={0: {}, 2: {}}
        )

        assert 0 in client.environment_states
        assert 1 not in client.environment_states
        assert 2 in client.environment_states

    def test_keeps_all_when_all_active(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory()
        client.environment_states[0] = client._create_environment_state()
        client.environment_states[1] = client._create_environment_state()

        client._remove_inactive_environments(
            per_environment_observations={0: {}, 1: {}}
        )

        assert len(client.environment_states) == 2

    def test_removes_all_when_none_active(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory()
        client.environment_states[0] = client._create_environment_state()

        client._remove_inactive_environments(per_environment_observations={})

        assert len(client.environment_states) == 0


@pytest.mark.unit
class TestCreateEnvironmentState:
    def test_buffer_keys_include_cameras_and_state(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            camera_keys=["left", "right"],
            state_keys=["proprio_robot_frame"],
        )

        state = client._create_environment_state()

        assert "left" in state.observation_buffer.required_keys
        assert "right" in state.observation_buffer.required_keys
        assert "proprio_robot_frame" in state.observation_buffer.required_keys

    def test_buffer_keys_include_language_when_enabled(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(has_language=True)

        state = client._create_environment_state()

        assert ObsKey.LANGUAGE.value in state.observation_buffer.required_keys

    def test_buffer_keys_exclude_language_when_disabled(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(has_language=False)

        state = client._create_environment_state()

        assert ObsKey.LANGUAGE.value not in state.observation_buffer.required_keys

    def test_buffer_size_matches_observation_horizon(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(observation_horizon=3)

        state = client._create_environment_state()

        assert state.observation_buffer.buffer_size == 3

    def test_temporal_aggregator_created_when_enabled(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            temporal_aggregation=True,
            prediction_horizon=8,
        )

        state = client._create_environment_state()

        assert state.temporal_aggregator.prediction_horizon == 8

    def test_no_aggregation_when_disabled(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            temporal_aggregation=False,
            action_keys_to_dimensions={"position": 2},
            observation_horizon=2,
        )

        state = client._create_environment_state()
        client.environment_states[0] = state

        assert state.observation_buffer.buffer_size == 2

        # Without aggregation, distribute_actions returns the full chunk
        # prediction_horizon defaults to 4 in the factory
        prediction_horizon = 4
        client.action_postprocessor = MagicMock(spec=ActionPostprocessor)
        client.action_postprocessor.format_action.side_effect = [
            {"position": [float(i), float(i + 1)]} for i in range(prediction_horizon)
        ]

        action_dict = {
            "position": torch.arange(
                prediction_horizon * 2, dtype=torch.float32
            ).reshape(1, prediction_horizon, 2),
        }
        result = client._distribute_actions(
            action_dict=action_dict,
            ready_indices=[0],
        )

        assert len(result[0]) == prediction_horizon


@pytest.mark.unit
class TestGetActionsForReadyEnvironments:
    def test_returns_empty_when_no_environments_ready(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            observation_horizon=3,
        )
        state = client._create_environment_state()
        state.observation_buffer.add(
            observations={
                "left": np.zeros((64, 64, 3), dtype=np.uint8),
                "proprio": np.zeros(3, dtype=np.float32),
            }
        )
        client.environment_states[0] = state

        result = client._get_actions_for_ready_environments()

        assert result == {}

    def test_runs_inference_for_ready_environments(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
    ):
        policy_loader = mock_policy_loader_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            action_keys_to_dimensions={"position": 3},
            prediction_horizon=4,
            observation_horizon=1,
        )
        predicted_actions = torch.from_numpy(
            rng.standard_normal((1, 4, 3)).astype(np.float32)
        )
        policy_loader.run_inference.return_value = {
            "position": predicted_actions,
        }

        client = InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=mock_action_transport,
        )

        state = client._create_environment_state()
        state.observation_buffer.add(
            observations={
                "left": rng.integers(0, 255, (64, 64, 3)).astype(np.uint8),
                "proprio": rng.standard_normal(3).astype(np.float32),
            }
        )
        client.environment_states[0] = state

        result = client._get_actions_for_ready_environments()

        policy_loader.run_inference.assert_called_once()
        assert 0 in result

    def test_sends_ready_batch_to_online_explanation_source(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
    ):
        policy_loader = mock_policy_loader_factory(
            camera_keys=[Cameras.AGENTVIEW.value],
            state_keys=["proprio"],
            action_keys_to_dimensions={"position": 3},
            prediction_horizon=4,
            observation_horizon=1,
        )
        policy_loader.run_inference.return_value = {
            "position": torch.from_numpy(
                rng.standard_normal((1, 4, 3)).astype(np.float32)
            ),
        }
        online_explanation_source = MagicMock()
        client = InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=mock_action_transport,
            online_explanation_source=online_explanation_source,
        )
        state = client._create_environment_state()
        state.observation_buffer.add(
            observations={
                Cameras.AGENTVIEW.value: rng.integers(0, 255, (64, 64, 3)).astype(
                    np.uint8
                ),
                "proprio": rng.standard_normal(3).astype(np.float32),
            }
        )
        client.environment_states[7] = state

        client._get_actions_for_ready_environments()

        online_explanation_source.explain_observation_batch.assert_called_once()
        call_kwargs = (
            online_explanation_source.explain_observation_batch.call_args.kwargs
        )
        policy_loader.run_inference.assert_called_once_with(
            obs_dict=call_kwargs["observation"]
        )
        assert call_kwargs["environment_indices"] == [7]
        assert call_kwargs["timestep"] == 0
        assert Cameras.AGENTVIEW.value in call_kwargs["display_observation"]

    def test_passes_language_batch_to_inference_when_language_enabled(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
    ):
        policy_loader = mock_policy_loader_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            has_language=True,
            action_keys_to_dimensions={"position": 3},
            prediction_horizon=4,
            observation_horizon=1,
        )
        predicted_actions = torch.from_numpy(
            rng.standard_normal((1, 4, 3)).astype(np.float32)
        )
        policy_loader.run_inference.return_value = {
            "position": predicted_actions,
        }

        client = InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=mock_action_transport,
        )

        language_instruction = "pick up the red block"
        state = client._create_environment_state()
        state.observation_buffer.add(
            observations={
                "left": rng.integers(0, 255, (64, 64, 3)).astype(np.uint8),
                "proprio": rng.standard_normal(3).astype(np.float32),
                ObsKey.LANGUAGE.value: language_instruction,
            }
        )
        client.environment_states[0] = state

        client._get_actions_for_ready_environments()

        call_kwargs = policy_loader.run_inference.call_args.kwargs
        obs_dict = call_kwargs["obs_dict"]
        assert ObsKey.LANGUAGE.value in obs_dict
        # language_batch is a list per environment; each entry is the
        # recent buffer contents (a list of strings per timestep)
        assert obs_dict[ObsKey.LANGUAGE.value] == [[language_instruction]]

    def test_multiple_ready_environments_with_language(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
    ):
        policy_loader = mock_policy_loader_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            has_language=True,
            action_keys_to_dimensions={"position": 3},
            prediction_horizon=4,
            observation_horizon=1,
        )
        predicted_actions = torch.from_numpy(
            rng.standard_normal((2, 4, 3)).astype(np.float32)
        )
        policy_loader.run_inference.return_value = {
            "position": predicted_actions,
        }

        client = InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=mock_action_transport,
        )

        instruction_env_0 = "pick up the red block"
        instruction_env_1 = "place on the table"

        state_0 = client._create_environment_state()
        state_0.observation_buffer.add(
            observations={
                "left": rng.integers(0, 255, (64, 64, 3)).astype(np.uint8),
                "proprio": rng.standard_normal(3).astype(np.float32),
                ObsKey.LANGUAGE.value: instruction_env_0,
            }
        )
        client.environment_states[0] = state_0

        state_1 = client._create_environment_state()
        state_1.observation_buffer.add(
            observations={
                "left": rng.integers(0, 255, (64, 64, 3)).astype(np.uint8),
                "proprio": rng.standard_normal(3).astype(np.float32),
                ObsKey.LANGUAGE.value: instruction_env_1,
            }
        )
        client.environment_states[1] = state_1

        result = client._get_actions_for_ready_environments()

        policy_loader.run_inference.assert_called_once()
        call_kwargs = policy_loader.run_inference.call_args.kwargs
        obs_dict = call_kwargs["obs_dict"]

        # Both environments batched together
        assert obs_dict["left"].shape[0] == 2
        assert obs_dict["proprio"].shape[0] == 2
        assert obs_dict[ObsKey.LANGUAGE.value] == [
            [instruction_env_0],
            [instruction_env_1],
        ]
        assert 0 in result
        assert 1 in result


@pytest.mark.unit
class TestDistributeActions:
    @pytest.mark.parametrize(
        "prediction_horizon, action_execution_horizon, expected_steps",
        [
            (4, None, 4),
            (4, 2, 2),
            (4, 1, 1),
        ],
    )
    def test_chunk_execution_formats_correct_number_of_steps(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        prediction_horizon: int,
        action_execution_horizon: int | None,
        expected_steps: int,
    ):
        client = inference_client_factory(
            action_keys_to_dimensions={"position": 2},
            temporal_aggregation=False,
            prediction_horizon=prediction_horizon,
            action_execution_horizon=action_execution_horizon,
        )
        client.environment_states[0] = client._create_environment_state()
        client.action_postprocessor = MagicMock(spec=ActionPostprocessor)
        client.action_postprocessor.format_action.side_effect = [
            {"position": [float(i), float(i + 1)]} for i in range(expected_steps)
        ]

        action_dict = {
            "position": torch.arange(
                prediction_horizon * 2, dtype=torch.float32
            ).reshape(1, prediction_horizon, 2),
        }

        result = client._distribute_actions(
            action_dict=action_dict,
            ready_indices=[0],
        )

        assert len(result[0]) == expected_steps
        assert client.action_postprocessor.format_action.call_count == expected_steps
        # Verify each call received the correct timestep slice
        for step in range(expected_steps):
            call_dict = client.action_postprocessor.format_action.call_args_list[
                step
            ].kwargs["action_dict"]
            torch.testing.assert_close(
                call_dict["position"],
                action_dict["position"][0, step],
            )

    def test_with_temporal_aggregation_calls_store_and_average(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        rng: np.random.Generator,
    ):
        client = inference_client_factory(
            action_keys_to_dimensions={"position": 3},
            temporal_aggregation=True,
        )
        state = client._create_environment_state()
        mock_aggregator = MagicMock(spec=TemporalAggregator)
        averaged_action = {"position": torch.tensor([0.5, 0.5, 0.5])}
        mock_aggregator.store_and_average.return_value = averaged_action
        state.temporal_aggregator = mock_aggregator
        client.environment_states[0] = state
        client.action_postprocessor = MagicMock(spec=ActionPostprocessor)
        client.action_postprocessor.format_action.return_value = {
            "position": [0.5, 0.5, 0.5],
        }

        action_dict = {
            "position": torch.from_numpy(
                rng.standard_normal((1, 4, 3)).astype(np.float32)
            ),
        }

        result = client._distribute_actions(
            action_dict=action_dict,
            ready_indices=[0],
        )

        mock_aggregator.store_and_average.assert_called_once()
        assert 0 in result
        # Returns a single-element list for temporal ensemble
        assert len(result[0]) == 1
        # format_action receives the averaged output, not the raw predictions
        format_call = client.action_postprocessor.format_action.call_args
        passed_dict = format_call.kwargs["action_dict"]
        torch.testing.assert_close(
            passed_dict["position"],
            torch.tensor([0.5, 0.5, 0.5]),
        )

    def test_distributes_to_multiple_environments(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            action_keys_to_dimensions={"position": 2},
            temporal_aggregation=False,
            prediction_horizon=2,
        )
        client.environment_states[0] = client._create_environment_state()
        client.environment_states[1] = client._create_environment_state()

        # 2 environments × 2 steps each = 4 format_action calls
        client.action_postprocessor = MagicMock(spec=ActionPostprocessor)
        client.action_postprocessor.format_action.side_effect = [
            {"position": [1.0, 2.0]},
            {"position": [3.0, 4.0]},
            {"position": [5.0, 6.0]},
            {"position": [7.0, 8.0]},
        ]

        action_dict = {
            "position": torch.tensor(
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ),
        }

        result = client._distribute_actions(
            action_dict=action_dict,
            ready_indices=[0, 1],
        )

        assert 0 in result
        assert 1 in result
        assert len(result[0]) == 2
        assert len(result[1]) == 2
        assert result[0][0]["position"] == [1.0, 2.0]
        assert result[0][1]["position"] == [3.0, 4.0]
        assert result[1][0]["position"] == [5.0, 6.0]
        assert result[1][1]["position"] == [7.0, 8.0]


@pytest.mark.unit
class TestReset:
    def test_clears_all_environment_states(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory()
        client.environment_states[0] = client._create_environment_state()
        client.environment_states[1] = client._create_environment_state()

        client.reset()

        assert len(client.environment_states) == 0

    def test_resets_observation_buffers_before_clearing(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            observation_horizon=1,
        )
        state = client._create_environment_state()
        state.observation_buffer.add(
            observations={
                "left": np.zeros((64, 64, 3), dtype=np.uint8),
                "proprio": np.zeros(3, dtype=np.float32),
            }
        )
        client.environment_states[0] = state
        buffer_reference = state.observation_buffer

        client.reset()

        assert not buffer_reference.is_ready()

    def test_resets_temporal_aggregators_before_clearing(
        self,
        inference_client_factory: Callable[..., InferenceClient],
    ):
        client = inference_client_factory(
            temporal_aggregation=True,
            observation_horizon=1,
        )
        state = client._create_environment_state()
        state.temporal_aggregator.timestep = 10
        client.environment_states[0] = state
        aggregator_reference = state.temporal_aggregator

        client.reset()

        assert aggregator_reference.timestep == 0


@pytest.mark.unit
class TestStep:
    def test_receives_observations_with_correct_parameters(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
    ):
        client = inference_client_factory(
            compression_type=CompressionType.RAW.value,
        )
        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.FINISHED.value,
        }

        client.step()

        mock_observation_transport.receive.assert_called_once_with(
            requested_keys=client.all_observation_keys,
            compression_type=CompressionType.RAW.value,
        )

    def test_returns_finished_when_server_finished(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
    ):
        client = inference_client_factory()
        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.FINISHED.value,
        }

        result = client.step()

        assert result == EpisodeStatus.FINISHED.value

    def test_returns_skip_when_server_processing(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
    ):
        client = inference_client_factory()
        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.PROCESSING.value,
        }

        result = client.step()

        assert result == EpisodeStatus.SKIP.value

    def test_does_not_call_parse_when_non_continue_status(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
    ):
        client = inference_client_factory()
        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.FINISHED.value,
        }

        with patch.object(
            client.observation_preprocessor,
            "parse_response",
        ) as mock_parse:
            client.step()

            mock_parse.assert_not_called()

    def test_does_not_send_actions_when_buffer_not_ready(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
    ):
        client = inference_client_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            observation_horizon=3,
        )
        response = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
        }
        mock_observation_transport.receive.return_value = response
        parsed_observations = {
            0: {
                "left": np.zeros((64, 64, 3), dtype=np.uint8),
                "proprio": np.zeros(3, dtype=np.float32),
            }
        }
        client.observation_preprocessor = MagicMock(spec=ObservationPreprocessor)
        client.observation_preprocessor.parse_response.return_value = (
            parsed_observations
        )

        client.step()

        mock_action_transport.send.assert_not_called()

    def test_increments_timestep_on_continue(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
    ):
        client = inference_client_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            observation_horizon=3,
        )
        response = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
        }
        mock_observation_transport.receive.return_value = response
        parsed_observations = {
            0: {
                "left": np.zeros((64, 64, 3), dtype=np.uint8),
                "proprio": np.zeros(3, dtype=np.float32),
            }
        }
        client.observation_preprocessor = MagicMock(spec=ObservationPreprocessor)
        client.observation_preprocessor.parse_response.return_value = (
            parsed_observations
        )
        assert client.timestep == 0

        client.step()

        assert client.timestep == 1

    def test_does_not_increment_timestep_on_finished(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
    ):
        client = inference_client_factory()
        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.FINISHED.value,
        }

        client.step()

        assert client.timestep == 0


@pytest.mark.unit
class TestStepOrchestration:
    def test_full_receive_parse_buffer_infer_format_send_cycle(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
    ):
        policy_loader = mock_policy_loader_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            action_keys_to_dimensions={"position": 3},
            prediction_horizon=4,
            observation_horizon=1,
        )
        predicted_actions = torch.from_numpy(
            rng.standard_normal((1, 4, 3)).astype(np.float32)
        )
        policy_loader.run_inference.return_value = {
            "position": predicted_actions,
        }

        client = InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=mock_action_transport,
        )

        # Mock preprocessor to avoid decompress_array
        parsed_observations = {
            0: {
                "left": rng.integers(0, 255, (64, 64, 3)).astype(np.uint8),
                "proprio": rng.standard_normal(3).astype(np.float32),
            }
        }
        client.observation_preprocessor = MagicMock(spec=ObservationPreprocessor)
        client.observation_preprocessor.parse_response.return_value = (
            parsed_observations
        )
        camera_tensor = torch.from_numpy(
            rng.standard_normal((1, 3, 64, 64)).astype(np.float32)
        )
        client.observation_preprocessor.transform_camera_observations.return_value = {
            "left": camera_tensor,
        }
        # Mock postprocessor
        client.action_postprocessor = MagicMock(spec=ActionPostprocessor)
        client.action_postprocessor.format_action.return_value = {
            "position": [0.1, 0.2, 0.3],
        }
        client.action_postprocessor.build_action_metadata.return_value = {
            "position": {"dimension": 3},
        }

        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
        }

        result = client.step()

        assert result == EpisodeStatus.CONTINUE.value
        policy_loader.run_inference.assert_called_once()
        # prediction_horizon=4 → 4 sends (one per action in the chunk)
        assert mock_action_transport.send.call_count == 4

        # Each send contains environment 0
        for call in mock_action_transport.send.call_args_list:
            sent_actions = (
                call.kwargs.get("actions") if call.kwargs else call[1]["actions"]
            )
            assert 0 in sent_actions

    def _make_history_client(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
        observation_horizon: int,
    ) -> InferenceClient:
        policy_loader = mock_policy_loader_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            action_keys_to_dimensions={"position": 3},
            prediction_horizon=3,
            observation_horizon=observation_horizon,
        )
        policy_loader.run_inference.return_value = {
            "position": torch.from_numpy(
                rng.standard_normal((1, 3, 3)).astype(np.float32)
            ),
        }
        client = InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=mock_action_transport,
        )
        parsed_observations = {
            0: {
                "left": rng.integers(0, 255, (64, 64, 3)).astype(np.uint8),
                "proprio": rng.standard_normal(3).astype(np.float32),
            }
        }
        client.observation_preprocessor = MagicMock(spec=ObservationPreprocessor)
        client.observation_preprocessor.parse_response.return_value = (
            parsed_observations
        )
        client.observation_preprocessor.transform_camera_observations.return_value = {
            "left": torch.from_numpy(
                rng.standard_normal((1, 3, 64, 64)).astype(np.float32)
            ),
        }
        client.action_postprocessor = MagicMock(spec=ActionPostprocessor)
        client.action_postprocessor.format_action.return_value = {
            "position": [0.1, 0.2, 0.3],
        }
        client.action_postprocessor.build_action_metadata.return_value = {
            "position": {"dimension": 3},
        }
        return client

    def test_warns_on_history_gap_with_chunked_execution(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ):
        policy_loader = mock_policy_loader_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            action_keys_to_dimensions={"position": 3},
            prediction_horizon=3,
            observation_horizon=2,
        )

        with caplog.at_level(logging.WARNING):
            InferenceClient(
                policy_runtime=policy_loader,
                observation_transport=mock_observation_transport,
                action_transport=mock_action_transport,
                action_execution_horizon=3,
            )

        assert any(
            "training windows are contiguous" in record.message
            for record in caplog.records
        )

    def test_chunk_execution_reads_one_observation_per_prediction(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
    ):
        client = self._make_history_client(
            mock_policy_loader_factory,
            mock_observation_transport,
            mock_action_transport,
            rng,
            observation_horizon=1,
        )
        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
        }

        client.step()

        assert mock_action_transport.send.call_count == 3
        assert mock_observation_transport.receive.call_count == 1

    def test_step_calls_pipeline_in_correct_order(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
    ):
        policy_loader = mock_policy_loader_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            action_keys_to_dimensions={"position": 3},
            prediction_horizon=4,
            observation_horizon=1,
        )
        predicted_actions = torch.from_numpy(
            rng.standard_normal((1, 4, 3)).astype(np.float32)
        )
        policy_loader.run_inference.return_value = {
            "position": predicted_actions,
        }

        client = InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=mock_action_transport,
        )

        mock_preprocessor = MagicMock(spec=ObservationPreprocessor)
        parsed_observations = {
            0: {
                "left": rng.integers(0, 255, (64, 64, 3)).astype(np.uint8),
                "proprio": rng.standard_normal(3).astype(np.float32),
            }
        }
        mock_preprocessor.parse_response.return_value = parsed_observations
        camera_tensor = torch.from_numpy(
            rng.standard_normal((1, 3, 64, 64)).astype(np.float32)
        )
        mock_preprocessor.transform_camera_observations.return_value = {
            "left": camera_tensor,
        }
        client.observation_preprocessor = mock_preprocessor

        mock_postprocessor = MagicMock(spec=ActionPostprocessor)
        mock_postprocessor.format_action.return_value = {
            "position": [0.1, 0.2, 0.3],
        }
        mock_postprocessor.build_action_metadata.return_value = {}
        client.action_postprocessor = mock_postprocessor

        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
        }

        client.step()

        # Verify receive -> parse -> inference -> format (×K) -> send (×K) order
        mock_observation_transport.receive.assert_called_once()
        mock_preprocessor.parse_response.assert_called_once()
        policy_loader.run_inference.assert_called_once()
        # prediction_horizon=4 → format and send called 4 times
        assert mock_postprocessor.format_action.call_count == 4
        mock_postprocessor.build_action_metadata.assert_called_once()
        assert mock_action_transport.send.call_count == 4

    def test_temporal_aggregation_accumulates_across_steps(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
    ):
        policy_loader = mock_policy_loader_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            action_keys_to_dimensions={"position": 3},
            prediction_horizon=4,
            observation_horizon=1,
        )

        client = InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=mock_action_transport,
            temporal_aggregation=True,
        )

        # Mock preprocessor
        parsed_observations = {
            0: {
                "left": rng.integers(0, 255, (64, 64, 3)).astype(np.uint8),
                "proprio": rng.standard_normal(3).astype(np.float32),
            }
        }
        client.observation_preprocessor = MagicMock(spec=ObservationPreprocessor)
        client.observation_preprocessor.parse_response.return_value = (
            parsed_observations
        )
        camera_tensor = torch.from_numpy(
            rng.standard_normal((1, 3, 64, 64)).astype(np.float32)
        )
        client.observation_preprocessor.transform_camera_observations.return_value = {
            "left": camera_tensor,
        }
        # Mock postprocessor
        client.action_postprocessor = MagicMock(spec=ActionPostprocessor)
        client.action_postprocessor.format_action.return_value = {
            "position": [0.5, 0.5, 0.5],
        }
        client.action_postprocessor.build_action_metadata.return_value = {}

        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
        }

        step_one_predictions = torch.from_numpy(
            rng.standard_normal((1, 4, 3)).astype(np.float32)
        )
        policy_loader.run_inference.return_value = {
            "position": step_one_predictions,
        }
        client.step()

        step_two_predictions = torch.from_numpy(
            rng.standard_normal((1, 4, 3)).astype(np.float32)
        )
        policy_loader.run_inference.return_value = {
            "position": step_two_predictions,
        }
        client.step()

        assert mock_action_transport.send.call_count == 2
        state = client.environment_states[0]
        assert state.temporal_aggregator.timestep == 2

    def test_without_temporal_aggregation_sends_all_chunk_steps(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
    ):
        prediction_horizon = 4
        policy_loader = mock_policy_loader_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            action_keys_to_dimensions={"position": 3},
            prediction_horizon=prediction_horizon,
            observation_horizon=1,
        )

        fixed_predictions = torch.zeros(1, prediction_horizon, 3)
        for t in range(prediction_horizon):
            fixed_predictions[0, t] = torch.tensor(
                [float(t * 3 + 1), float(t * 3 + 2), float(t * 3 + 3)]
            )
        policy_loader.run_inference.return_value = {
            "position": fixed_predictions,
        }

        client = InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=mock_action_transport,
            temporal_aggregation=False,
        )

        parsed_observations = {
            0: {
                "left": rng.integers(0, 255, (64, 64, 3)).astype(np.uint8),
                "proprio": rng.standard_normal(3).astype(np.float32),
            }
        }
        client.observation_preprocessor = MagicMock(spec=ObservationPreprocessor)
        client.observation_preprocessor.parse_response.return_value = (
            parsed_observations
        )
        camera_tensor = torch.from_numpy(
            rng.standard_normal((1, 3, 64, 64)).astype(np.float32)
        )
        client.observation_preprocessor.transform_camera_observations.return_value = {
            "left": camera_tensor,
        }
        client.action_postprocessor = MagicMock(spec=ActionPostprocessor)
        client.action_postprocessor.format_action.side_effect = [
            {"position": [float(t * 3 + 1), float(t * 3 + 2), float(t * 3 + 3)]}
            for t in range(prediction_horizon)
        ]
        client.action_postprocessor.build_action_metadata.return_value = {}

        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
        }

        client.step()

        # Each step in the chunk produces a format_action call and a send
        assert (
            client.action_postprocessor.format_action.call_count == prediction_horizon
        )
        assert mock_action_transport.send.call_count == prediction_horizon
        # Verify each format_action call received the correct timestep
        for t in range(prediction_horizon):
            call_dict = client.action_postprocessor.format_action.call_args_list[
                t
            ].kwargs["action_dict"]
            torch.testing.assert_close(
                call_dict["position"],
                fixed_predictions[0, t],
            )


@pytest.mark.unit
class TestRunEpisode:
    def test_registers_with_client_identifier(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
    ):
        client = inference_client_factory()
        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.FINISHED.value,
        }

        client.run_episode(max_steps=10)

        mock_observation_transport.register.assert_called_once_with(
            client_name=client.policy_runtime.client_identifier,
        )

    def test_stops_on_finished_status(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
    ):
        client = inference_client_factory()
        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.FINISHED.value,
        }

        client.run_episode(max_steps=100)

        assert mock_observation_transport.receive.call_count == 1

    def test_respects_max_steps(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
    ):
        client = inference_client_factory()
        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.PROCESSING.value,
        }

        client.run_episode(max_steps=5)

        assert mock_observation_transport.receive.call_count == 5


@pytest.mark.unit
class TestShutdown:
    def test_closes_observation_transport(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
    ):
        client = inference_client_factory()

        client.shutdown()

        mock_observation_transport.close.assert_called_once()

    def test_closes_action_transport_when_close_method_exists(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_action_transport: MagicMock,
    ):
        client = inference_client_factory()

        client.shutdown()

        mock_action_transport.close.assert_called_once()

    def test_handles_action_transport_without_close(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
    ):
        policy_loader = mock_policy_loader_factory()
        action_transport = MagicMock(spec=[])
        client = InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=action_transport,
        )

        client.shutdown()

        mock_observation_transport.close.assert_called_once()


@pytest.mark.unit
class TestStepTimingLog:
    def _make_continue_client(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
    ) -> InferenceClient:
        policy_loader = mock_policy_loader_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            action_keys_to_dimensions={"position": 3},
            prediction_horizon=4,
            observation_horizon=1,
        )
        predicted_actions = torch.from_numpy(
            rng.standard_normal((1, 4, 3)).astype(np.float32)
        )
        policy_loader.run_inference.return_value = {
            "position": predicted_actions,
        }

        client = InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=mock_action_transport,
            timing_log=True,
        )

        parsed_observations = {
            0: {
                "left": rng.integers(0, 255, (64, 64, 3)).astype(np.uint8),
                "proprio": rng.standard_normal(3).astype(np.float32),
            }
        }
        client.observation_preprocessor = MagicMock(spec=ObservationPreprocessor)
        client.observation_preprocessor.parse_response.return_value = (
            parsed_observations
        )
        camera_tensor = torch.from_numpy(
            rng.standard_normal((1, 3, 64, 64)).astype(np.float32)
        )
        client.observation_preprocessor.transform_camera_observations.return_value = {
            "left": camera_tensor,
        }
        client.action_postprocessor = MagicMock(spec=ActionPostprocessor)
        client.action_postprocessor.format_action.return_value = {
            "position": [0.1, 0.2, 0.3],
        }
        client.action_postprocessor.build_action_metadata.return_value = {}

        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
        }

        return client

    def test_logs_timing_breakdown_on_successful_step(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
    ):
        client = self._make_continue_client(
            mock_policy_loader_factory=mock_policy_loader_factory,
            mock_observation_transport=mock_observation_transport,
            mock_action_transport=mock_action_transport,
            rng=rng,
        )
        # 8 calls to time.time() in the timing_log=True path:
        # step_start, preprocessing_start, end_preprocess, inference_start,
        # end_inference, postprocessing_start, end_postprocess, end_total
        time_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        with (
            patch(
                "versatil.inference.inference_client.time.time",
                side_effect=time_values,
            ),
            patch(
                "versatil.inference.inference_client.logging.info",
            ) as mock_log_info,
        ):
            client.step()

            mock_log_info.assert_called_once()
            call_args = mock_log_info.call_args[0]
            format_string = call_args[0]
            assert "[TIMING]" in format_string
            # preprocess=0.1, inference=0.1, postprocess=0.1, total=0.7, fps=1/0.7
            assert call_args[1] == 0  # timestep
            assert call_args[2] == pytest.approx(0.1)  # preprocess duration
            assert call_args[3] == pytest.approx(0.1)  # inference duration
            assert call_args[4] == pytest.approx(0.1)  # postprocess duration
            assert call_args[5] == pytest.approx(0.7)  # total duration
            assert call_args[6] == pytest.approx(1.0 / 0.7)  # fps

    def test_no_timing_log_when_disabled(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
    ):
        client = inference_client_factory(
            timing_log=False,
            camera_keys=["left"],
            state_keys=["proprio"],
            observation_horizon=3,
        )
        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
        }
        parsed_observations = {
            0: {
                "left": np.zeros((64, 64, 3), dtype=np.uint8),
                "proprio": np.zeros(3, dtype=np.float32),
            }
        }
        client.observation_preprocessor = MagicMock(spec=ObservationPreprocessor)
        client.observation_preprocessor.parse_response.return_value = (
            parsed_observations
        )

        with patch(
            "versatil.inference.inference_client.logging.info",
        ) as mock_log_info:
            client.step()

            mock_log_info.assert_not_called()


@pytest.mark.unit
class TestStepUpdateRateHz:
    def test_sleeps_for_target_period_when_update_rate_set(
        self,
        mock_policy_loader_factory: Callable[..., MagicMock],
        mock_observation_transport: MagicMock,
        mock_action_transport: MagicMock,
        rng: np.random.Generator,
    ):
        policy_loader = mock_policy_loader_factory(
            camera_keys=["left"],
            state_keys=["proprio"],
            action_keys_to_dimensions={"position": 3},
            prediction_horizon=4,
            observation_horizon=1,
        )
        predicted_actions = torch.from_numpy(
            rng.standard_normal((1, 4, 3)).astype(np.float32)
        )
        policy_loader.run_inference.return_value = {
            "position": predicted_actions,
        }

        client = InferenceClient(
            policy_runtime=policy_loader,
            observation_transport=mock_observation_transport,
            action_transport=mock_action_transport,
            update_rate_hz=10.0,
        )

        parsed_observations = {
            0: {
                "left": rng.integers(0, 255, (64, 64, 3)).astype(np.uint8),
                "proprio": rng.standard_normal(3).astype(np.float32),
            }
        }
        client.observation_preprocessor = MagicMock(spec=ObservationPreprocessor)
        client.observation_preprocessor.parse_response.return_value = (
            parsed_observations
        )
        camera_tensor = torch.from_numpy(
            rng.standard_normal((1, 3, 64, 64)).astype(np.float32)
        )
        client.observation_preprocessor.transform_camera_observations.return_value = {
            "left": camera_tensor,
        }
        client.action_postprocessor = MagicMock(spec=ActionPostprocessor)
        client.action_postprocessor.format_action.return_value = {
            "position": [0.1, 0.2, 0.3],
        }
        client.action_postprocessor.build_action_metadata.return_value = {}

        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
        }

        with patch(
            "versatil.inference.inference_client.time.sleep",
        ) as mock_sleep:
            client.step()

            assert mock_sleep.call_count == 4
            assert mock_sleep.call_args_list == [((1.0 / 10.0,),)] * 4

    def test_does_not_sleep_when_update_rate_is_none(
        self,
        inference_client_factory: Callable[..., InferenceClient],
        mock_observation_transport: MagicMock,
    ):
        client = inference_client_factory(
            update_rate_hz=None,
            camera_keys=["left"],
            state_keys=["proprio"],
            observation_horizon=3,
        )
        mock_observation_transport.receive.return_value = {
            TransportKey.STATUS.value: ServerStatus.WAITING_ACTION.value,
        }
        parsed_observations = {
            0: {
                "left": np.zeros((64, 64, 3), dtype=np.uint8),
                "proprio": np.zeros(3, dtype=np.float32),
            }
        }
        client.observation_preprocessor = MagicMock(spec=ObservationPreprocessor)
        client.observation_preprocessor.parse_response.return_value = (
            parsed_observations
        )

        with patch(
            "versatil.inference.inference_client.time.sleep",
        ) as mock_sleep:
            client.step()

            mock_sleep.assert_not_called()
