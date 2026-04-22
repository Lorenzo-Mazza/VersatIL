"""Tests for versatil.data.task module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from versatil_constants.shared import ObsKey
from versatil_constants.tso import TSOObsKey

from versatil.data.constants import (
    ActionComputationMethod,
    Cameras,
    CoordinateSystem,
    GripperType,
    OrientationRepresentation,
)
from versatil.data.metadata import (
    ActionMetadata,
    CameraMetadata,
    GripperActionMetadata,
    GripperObservationMetadata,
    ObservationMetadata,
    OnTheFlyActionMetadata,
    OrientationObservationMetadata,
    PositionObservationMetadata,
)
from versatil.data.task import ActionSpace, ObservationSpace, TaskSpace


@pytest.fixture
def position_on_the_fly(
    position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
    on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
) -> OnTheFlyActionMetadata:
    """On-the-fly position action (3D, delta, robot base frame)."""
    return on_the_fly_action_metadata_factory(
        source_metadata=position_observation_metadata_factory(dimension=3),
    )


@pytest.fixture
def orientation_on_the_fly(
    orientation_observation_metadata_factory: Callable[
        ..., OrientationObservationMetadata
    ],
    on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
) -> OnTheFlyActionMetadata:
    """On-the-fly orientation action (1D roll, delta)."""
    return on_the_fly_action_metadata_factory(
        source_metadata=orientation_observation_metadata_factory(dimension=1),
    )


@pytest.fixture
def precomputed_gripper(
    gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
) -> GripperActionMetadata:
    """Precomputed binary gripper action (1D)."""
    return gripper_action_metadata_factory(prediction_dimension=1)


@pytest.fixture
def mixed_action_space(
    action_space_factory: Callable[..., ActionSpace],
    position_on_the_fly: OnTheFlyActionMetadata,
    precomputed_gripper: GripperActionMetadata,
) -> ActionSpace:
    """Action space with both on-the-fly position and precomputed gripper."""
    return action_space_factory(
        actions_metadata={
            "position": position_on_the_fly,
            "gripper": precomputed_gripper,
        }
    )


def _make_mock_schema(
    zarr_keys: list[str],
    observations: dict = None,
) -> MagicMock:
    """Create a mock DatasetSchema with given zarr keys and observations."""
    schema = MagicMock()
    schema.get_required_zarr_keys.return_value = zarr_keys
    schema.metadata = MagicMock()
    schema.metadata.observations = observations or {}
    schema.metadata.get_observation = lambda key: (observations or {}).get(key)
    return schema


class TestActionSpaceInitialization:
    @pytest.mark.parametrize("denoise_actions", [True, False])
    def test_denoise_actions_stored(
        self,
        action_space_factory: Callable[..., ActionSpace],
        denoise_actions: bool,
    ):
        action_space = action_space_factory(denoise_actions=denoise_actions)
        assert action_space.denoise_actions is denoise_actions

    @pytest.mark.parametrize("denoising_percentile", [5.0, 15.0, 50.0])
    def test_denoising_percentile_stored(
        self,
        action_space_factory: Callable[..., ActionSpace],
        denoising_percentile: float,
    ):
        action_space = action_space_factory(denoising_percentile=denoising_percentile)
        assert action_space.denoising_percentile == denoising_percentile

    @pytest.mark.parametrize("use_gripper_class_weights", [True, False])
    def test_use_gripper_class_weights_stored(
        self,
        action_space_factory: Callable[..., ActionSpace],
        use_gripper_class_weights: bool,
    ):
        action_space = action_space_factory(
            use_gripper_class_weights=use_gripper_class_weights,
        )
        assert action_space.use_gripper_class_weights is use_gripper_class_weights


class TestActionSpacePropertyFiltering:
    def test_on_the_fly_actions_filters_correctly(
        self,
        mixed_action_space: ActionSpace,
    ):
        result = mixed_action_space.on_the_fly_actions
        assert "position" in result
        assert "gripper" not in result

    def test_precomputed_actions_filters_correctly(
        self,
        mixed_action_space: ActionSpace,
    ):
        result = mixed_action_space.precomputed_actions
        assert "gripper" in result
        assert "position" not in result

    def test_position_actions_includes_on_the_fly_from_position_source(
        self,
        action_space_factory: Callable[..., ActionSpace],
        position_on_the_fly: OnTheFlyActionMetadata,
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": position_on_the_fly,
            }
        )

        assert "position" in action_space.position_actions

    def test_position_actions_excludes_on_the_fly_from_orientation_source(
        self,
        action_space_factory: Callable[..., ActionSpace],
        orientation_on_the_fly: OnTheFlyActionMetadata,
    ):
        action_space = action_space_factory(
            actions_metadata={
                "orientation": orientation_on_the_fly,
            }
        )

        assert len(action_space.position_actions) == 0

    def test_orientation_actions_includes_on_the_fly_from_orientation_source(
        self,
        action_space_factory: Callable[..., ActionSpace],
        orientation_on_the_fly: OnTheFlyActionMetadata,
    ):
        action_space = action_space_factory(
            actions_metadata={
                "orientation": orientation_on_the_fly,
            }
        )

        assert "orientation" in action_space.orientation_actions

    def test_gripper_actions_includes_on_the_fly_from_gripper_source(
        self,
        action_space_factory: Callable[..., ActionSpace],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        gripper_source = gripper_observation_metadata_factory(
            gripper_type=GripperType.CONTINUOUS.value,
        )
        gripper_on_the_fly = on_the_fly_action_metadata_factory(
            source_metadata=gripper_source,
        )

        action_space = action_space_factory(
            actions_metadata={
                "gripper": gripper_on_the_fly,
            }
        )

        assert "gripper" in action_space.gripper_actions

    def test_gripper_actions_includes_precomputed_gripper(
        self,
        action_space_factory: Callable[..., ActionSpace],
        precomputed_gripper: GripperActionMetadata,
    ):
        action_space = action_space_factory(
            actions_metadata={
                "gripper": precomputed_gripper,
            }
        )

        assert "gripper" in action_space.gripper_actions

    def test_empty_metadata_returns_empty_for_all_properties(
        self,
        action_space_factory: Callable[..., ActionSpace],
    ):
        action_space = action_space_factory()

        assert len(action_space.on_the_fly_actions) == 0
        assert len(action_space.precomputed_actions) == 0
        assert len(action_space.position_actions) == 0
        assert len(action_space.orientation_actions) == 0
        assert len(action_space.gripper_actions) == 0


class TestActionSpaceDimensions:
    def test_position_dim_from_on_the_fly(
        self,
        action_space_factory: Callable[..., ActionSpace],
        position_on_the_fly: OnTheFlyActionMetadata,
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": position_on_the_fly,
            }
        )

        assert action_space.position_dim == 3

    def test_orientation_dim_from_on_the_fly(
        self,
        action_space_factory: Callable[..., ActionSpace],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        quaternion_source = orientation_observation_metadata_factory(
            dimension=4,
            orientation_representation=OrientationRepresentation.QUATERNION.value,
            raw_data_column_keys=["w", "x", "y", "z"],
        )
        action_space = action_space_factory(
            actions_metadata={
                "orientation": on_the_fly_action_metadata_factory(
                    source_metadata=quaternion_source,
                ),
            }
        )

        assert action_space.orientation_dim == 4

    def test_gripper_dim_from_precomputed(
        self,
        action_space_factory: Callable[..., ActionSpace],
        precomputed_gripper: GripperActionMetadata,
    ):
        action_space = action_space_factory(
            actions_metadata={
                "gripper": precomputed_gripper,
            }
        )

        assert action_space.gripper_dim == 1

    def test_get_total_action_dim_sums_all_prediction_dimensions(
        self,
        mixed_action_space: ActionSpace,
        orientation_on_the_fly: OnTheFlyActionMetadata,
        action_space_factory: Callable[..., ActionSpace],
        position_on_the_fly: OnTheFlyActionMetadata,
        precomputed_gripper: GripperActionMetadata,
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": position_on_the_fly,
                "orientation": orientation_on_the_fly,
                "gripper": precomputed_gripper,
            }
        )

        assert action_space.get_total_action_dim() == 3 + 1 + 1

    def test_get_total_action_dim_excludes_non_prediction_head_actions(
        self,
        action_space_factory: Callable[..., ActionSpace],
        position_on_the_fly: OnTheFlyActionMetadata,
    ):
        auxiliary_action = ActionMetadata(
            prediction_dimension=5,
            is_numerical=True,
            needs_normalization=False,
            dtype="int32",
            is_precomputed=True,
            requires_prediction_head=False,
        )

        action_space = action_space_factory(
            actions_metadata={
                "position": position_on_the_fly,
                "phase_label": auxiliary_action,
            }
        )

        assert action_space.get_total_action_dim() == 3

    def test_empty_metadata_has_zero_dims(
        self,
        action_space_factory: Callable[..., ActionSpace],
    ):
        action_space = action_space_factory()

        assert action_space.position_dim == 0
        assert action_space.orientation_dim == 0
        assert action_space.gripper_dim == 0
        assert action_space.get_total_action_dim() == 0


class TestActionSpaceBooleanProperties:
    def test_has_on_the_fly_actions(
        self,
        action_space_factory: Callable[..., ActionSpace],
        position_on_the_fly: OnTheFlyActionMetadata,
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": position_on_the_fly,
            }
        )

        assert action_space.has_on_the_fly_actions
        assert not action_space.has_precomputed_actions

    def test_has_precomputed_actions(
        self,
        action_space_factory: Callable[..., ActionSpace],
        precomputed_gripper: GripperActionMetadata,
    ):
        action_space = action_space_factory(
            actions_metadata={
                "gripper": precomputed_gripper,
            }
        )

        assert action_space.has_precomputed_actions
        assert not action_space.has_on_the_fly_actions

    @pytest.mark.parametrize(
        "case, expected",
        [
            ("empty", False),
            ("only_precomputed", True),
            ("only_on_the_fly", False),
            ("mixed", False),
        ],
    )
    def test_has_only_precomputed_actions(
        self,
        action_space_factory: Callable[..., ActionSpace],
        position_on_the_fly: OnTheFlyActionMetadata,
        precomputed_gripper: GripperActionMetadata,
        case: str,
        expected: bool,
    ):
        if case == "empty":
            actions_metadata = {}
        elif case == "only_precomputed":
            actions_metadata = {"gripper": precomputed_gripper}
        elif case == "only_on_the_fly":
            actions_metadata = {"position": position_on_the_fly}
        else:
            actions_metadata = {
                "position": position_on_the_fly,
                "gripper": precomputed_gripper,
            }
        action_space = action_space_factory(actions_metadata=actions_metadata)
        assert action_space.has_only_precomputed_actions is expected

    def test_has_delta_actions_true_when_delta_method(
        self,
        action_space_factory: Callable[..., ActionSpace],
        position_on_the_fly: OnTheFlyActionMetadata,
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": position_on_the_fly,
            }
        )

        assert action_space.has_delta_actions

    def test_has_delta_actions_false_when_next_timestep(
        self,
        action_space_factory: Callable[..., ActionSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": on_the_fly_action_metadata_factory(
                    source_metadata=position_observation_metadata_factory(),
                    computation_method=ActionComputationMethod.NEXT_TIMESTEP.value,
                ),
            }
        )

        assert not action_space.has_delta_actions

    def test_task_has_phases_when_phase_label_key_present(
        self,
        action_space_factory: Callable[..., ActionSpace],
    ):
        phase_metadata = ActionMetadata(
            prediction_dimension=5,
            is_numerical=True,
            needs_normalization=False,
            dtype="int32",
            is_precomputed=True,
        )
        action_space = action_space_factory(
            actions_metadata={
                TSOObsKey.PHASE_LABEL.value: phase_metadata,
            }
        )

        assert action_space.task_has_phases

    def test_task_has_phases_false_when_no_phase_label(
        self,
        action_space_factory: Callable[..., ActionSpace],
        position_on_the_fly: OnTheFlyActionMetadata,
    ):
        action_space = action_space_factory(
            actions_metadata={
                "position": position_on_the_fly,
            }
        )

        assert not action_space.task_has_phases


class TestActionSpaceZarrKeys:
    def test_get_required_zarr_keys_returns_all_metadata_keys(
        self,
        mixed_action_space: ActionSpace,
    ):
        keys = mixed_action_space.get_required_zarr_keys()
        assert set(keys) == {"position", "gripper"}


class TestObservationSpacePropertyFiltering:
    def test_cameras_filters_camera_metadata(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                Cameras.LEFT.value: camera_metadata_factory(
                    camera_key=Cameras.LEFT.value
                ),
                "position": position_observation_metadata_factory(),
            }
        )

        assert Cameras.LEFT.value in observation_space.cameras
        assert "position" not in observation_space.cameras

    def test_position_observations_filters_correctly(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                "position": position_observation_metadata_factory(),
                "orientation": orientation_observation_metadata_factory(),
            }
        )

        assert "position" in observation_space.position_observations
        assert "orientation" not in observation_space.position_observations

    def test_proprioceptive_observations_returns_only_robot_types(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        camera_metadata_factory: Callable[..., CameraMetadata],
    ):
        custom_numerical = ObservationMetadata(
            raw_data_column_keys=["feature_1"],
            dimension=1,
            dtype="float32",
            is_numerical=True,
            needs_normalization=True,
        )
        observation_space = observation_space_factory(
            observations_metadata={
                "position": position_observation_metadata_factory(),
                "orientation": orientation_observation_metadata_factory(),
                "gripper": gripper_observation_metadata_factory(),
                "custom_numerical": custom_numerical,
                Cameras.LEFT.value: camera_metadata_factory(),
            }
        )

        proprio = observation_space.proprioceptive_observations
        assert set(proprio.keys()) == {"position", "orientation", "gripper"}

    def test_numerical_observations_includes_proprio_and_custom_numerical(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        camera_metadata_factory: Callable[..., CameraMetadata],
    ):
        custom_numerical = ObservationMetadata(
            raw_data_column_keys=["object_pos"],
            dimension=2,
            dtype="float32",
            is_numerical=True,
            needs_normalization=True,
        )
        language = ObservationMetadata(
            raw_data_column_keys=["language"],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        observation_space = observation_space_factory(
            observations_metadata={
                "position": position_observation_metadata_factory(),
                "orientation": orientation_observation_metadata_factory(),
                "gripper": gripper_observation_metadata_factory(),
                "object_pos": custom_numerical,
                ObsKey.LANGUAGE.value: language,
                Cameras.LEFT.value: camera_metadata_factory(),
            }
        )

        numerical = observation_space.numerical_observations
        assert set(numerical.keys()) == {
            "position",
            "orientation",
            "gripper",
            "object_pos",
        }

    def test_custom_observations_excludes_proprio_and_cameras(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        camera_metadata_factory: Callable[..., CameraMetadata],
    ):
        custom = ObservationMetadata(
            raw_data_column_keys=["feature_1", "feature_2"],
            dimension=2,
            dtype="float32",
            is_numerical=True,
            needs_normalization=True,
        )
        observation_space = observation_space_factory(
            observations_metadata={
                "custom_feature": custom,
                "position": position_observation_metadata_factory(),
                Cameras.LEFT.value: camera_metadata_factory(),
            }
        )

        result = observation_space.custom_observations
        assert "custom_feature" in result
        assert "position" not in result
        assert Cameras.LEFT.value not in result


class TestObservationSpaceBooleanProperties:
    def test_has_cameras(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        camera_metadata_factory: Callable[..., CameraMetadata],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                Cameras.LEFT.value: camera_metadata_factory(),
            }
        )

        assert observation_space.has_cameras

    def test_has_gripper_state(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                "gripper": gripper_observation_metadata_factory(),
            }
        )

        assert observation_space.has_gripper_state

    def test_has_proprioceptive_state(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                "position": position_observation_metadata_factory(),
            }
        )

        assert observation_space.has_proprioceptive_state

    def test_has_proprioceptive_position(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                "position": position_observation_metadata_factory(),
            }
        )

        assert observation_space.has_proprioceptive_position

    def test_has_proprioceptive_orientation(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                "orientation": orientation_observation_metadata_factory(),
            }
        )

        assert observation_space.has_proprioceptive_orientation

    def test_empty_observations_all_false(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
    ):
        observation_space = observation_space_factory()

        assert not observation_space.has_cameras
        assert not observation_space.has_gripper_state
        assert not observation_space.has_proprioceptive_state
        assert not observation_space.has_proprioceptive_position
        assert not observation_space.has_proprioceptive_orientation


class TestObservationSpaceZarrKeys:
    def test_get_required_zarr_keys_returns_all_metadata_keys(
        self,
        observation_space_factory: Callable[..., ObservationSpace],
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        observation_space = observation_space_factory(
            observations_metadata={
                Cameras.LEFT.value: camera_metadata_factory(),
                "position": position_observation_metadata_factory(),
            }
        )

        keys = observation_space.get_required_zarr_keys()
        assert set(keys) == {Cameras.LEFT.value, "position"}


class TestTaskSpaceInitialization:
    def test_all_components_stored(
        self,
        action_space_factory: Callable[..., ActionSpace],
        observation_space_factory: Callable[..., ObservationSpace],
    ):
        schema = _make_mock_schema(zarr_keys=[])
        dataloader = MagicMock()
        action_space = action_space_factory()
        observation_space = observation_space_factory()

        task_space = TaskSpace(
            dataset_schema=schema,
            dataloader=dataloader,
            action_space=action_space,
            observation_space=observation_space,
            prediction_horizon=32,
            observation_horizon=4,
        )

        assert task_space.dataset_schema is schema
        assert task_space.dataloader is dataloader
        assert task_space.action_space is action_space
        assert task_space.observation_space is observation_space
        assert task_space.prediction_horizon == 32
        assert task_space.observation_horizon == 4


class TestTaskSpaceValidation:
    def test_valid_on_the_fly_action_passes(
        self,
        action_space_factory: Callable[..., ActionSpace],
        observation_space_factory: Callable[..., ObservationSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        camera_metadata_factory: Callable[..., CameraMetadata],
    ):
        position_observation = position_observation_metadata_factory()
        on_the_fly = on_the_fly_action_metadata_factory(
            source_metadata=position_observation,
        )

        schema = _make_mock_schema(
            zarr_keys=["proprio_robot_frame", Cameras.LEFT.value],
            observations={"proprio_robot_frame": position_observation},
        )

        task_space = TaskSpace(
            dataset_schema=schema,
            dataloader=MagicMock(),
            action_space=action_space_factory(
                actions_metadata={
                    "proprio_robot_frame": on_the_fly,
                }
            ),
            observation_space=observation_space_factory(
                observations_metadata={
                    Cameras.LEFT.value: camera_metadata_factory(),
                }
            ),
        )

        assert task_space.prediction_horizon == 16
        assert task_space.observation_horizon == 1

    def test_on_the_fly_action_missing_from_schema_raises(
        self,
        action_space_factory: Callable[..., ActionSpace],
        observation_space_factory: Callable[..., ObservationSpace],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        schema = _make_mock_schema(zarr_keys=[])

        with pytest.raises(ValueError, match="doesn't exist in dataset schema"):
            TaskSpace(
                dataset_schema=schema,
                dataloader=MagicMock(),
                action_space=action_space_factory(
                    actions_metadata={
                        "proprio_robot_frame": on_the_fly_action_metadata_factory(),
                    }
                ),
                observation_space=observation_space_factory(),
            )

    def test_on_the_fly_action_metadata_mismatch_raises(
        self,
        action_space_factory: Callable[..., ActionSpace],
        observation_space_factory: Callable[..., ObservationSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        # Task expects robot_base frame, schema has camera frame
        task_observation = position_observation_metadata_factory(
            frame=CoordinateSystem.ROBOT_BASE.value,
        )
        schema_observation = position_observation_metadata_factory(
            frame=CoordinateSystem.CAMERA.value,
        )
        on_the_fly = on_the_fly_action_metadata_factory(
            source_metadata=task_observation,
        )

        schema = _make_mock_schema(
            zarr_keys=["position"],
            observations={"position": schema_observation},
        )

        with pytest.raises(ValueError, match="metadata mismatch"):
            TaskSpace(
                dataset_schema=schema,
                dataloader=MagicMock(),
                action_space=action_space_factory(
                    actions_metadata={
                        "position": on_the_fly,
                    }
                ),
                observation_space=observation_space_factory(),
            )

    def test_precomputed_action_missing_from_schema_raises(
        self,
        action_space_factory: Callable[..., ActionSpace],
        observation_space_factory: Callable[..., ObservationSpace],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        schema = _make_mock_schema(zarr_keys=[])

        with pytest.raises(ValueError, match="not found in dataset schema"):
            TaskSpace(
                dataset_schema=schema,
                dataloader=MagicMock(),
                action_space=action_space_factory(
                    actions_metadata={
                        "gripper_action": gripper_action_metadata_factory(),
                    }
                ),
                observation_space=observation_space_factory(),
            )

    def test_observation_missing_from_schema_raises(
        self,
        action_space_factory: Callable[..., ActionSpace],
        observation_space_factory: Callable[..., ObservationSpace],
        camera_metadata_factory: Callable[..., CameraMetadata],
    ):
        schema = _make_mock_schema(zarr_keys=[])

        with pytest.raises(ValueError, match="not found in dataset schema"):
            TaskSpace(
                dataset_schema=schema,
                dataloader=MagicMock(),
                action_space=action_space_factory(),
                observation_space=observation_space_factory(
                    observations_metadata={
                        Cameras.LEFT.value: camera_metadata_factory(),
                    }
                ),
            )

    def test_observation_metadata_mismatch_raises(
        self,
        action_space_factory: Callable[..., ActionSpace],
        observation_space_factory: Callable[..., ObservationSpace],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        # Task expects 3D position, schema stores 6D
        task_observation = position_observation_metadata_factory(dimension=3)
        schema_observation = position_observation_metadata_factory(
            dimension=6,
            raw_data_column_keys=["a", "b", "c", "d", "e", "f"],
        )

        schema = _make_mock_schema(
            zarr_keys=["position"],
            observations={"position": schema_observation},
        )

        with pytest.raises(ValueError, match="metadata mismatch"):
            TaskSpace(
                dataset_schema=schema,
                dataloader=MagicMock(),
                action_space=action_space_factory(),
                observation_space=observation_space_factory(
                    observations_metadata={
                        "position": task_observation,
                    }
                ),
            )

    @pytest.mark.parametrize("observation_horizon", [0, -1])
    def test_invalid_observation_horizon_raises(
        self,
        action_space_factory: Callable[..., ActionSpace],
        observation_space_factory: Callable[..., ObservationSpace],
        observation_horizon: int,
    ):
        schema = _make_mock_schema(zarr_keys=[])

        with pytest.raises(ValueError, match="observation_horizon must be >= 1"):
            TaskSpace(
                dataset_schema=schema,
                dataloader=MagicMock(),
                action_space=action_space_factory(),
                observation_space=observation_space_factory(),
                observation_horizon=observation_horizon,
            )

    @pytest.mark.parametrize("prediction_horizon", [0, -1])
    def test_invalid_prediction_horizon_raises(
        self,
        action_space_factory: Callable[..., ActionSpace],
        observation_space_factory: Callable[..., ObservationSpace],
        prediction_horizon: int,
    ):
        schema = _make_mock_schema(zarr_keys=[])

        with pytest.raises(ValueError, match="prediction_horizon must be >= 1"):
            TaskSpace(
                dataset_schema=schema,
                dataloader=MagicMock(),
                action_space=action_space_factory(),
                observation_space=observation_space_factory(),
                prediction_horizon=prediction_horizon,
            )
