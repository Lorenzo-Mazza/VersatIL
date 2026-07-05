"""Tests for versatil.data.raw.zarr_meta module."""

from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import pytest

from versatil.data.constants import (
    BinaryGripperRange,
    Cameras,
    CoordinateSystem,
    GripperType,
    OrientationRepresentation,
    RawCameraKey,
)
from versatil.data.metadata import (
    CameraMetadata,
    GripperActionMetadata,
    GripperObservationMetadata,
    ObservationMetadata,
    OrientationActionMetadata,
    OrientationObservationMetadata,
    PositionActionMetadata,
    PositionObservationMetadata,
    PrecomputedActionMetadata,
)
from versatil.data.raw.zarr_meta import DatasetMetadata


class TestDatasetMetadataPostInit:
    def test_empty_observations_and_actions_succeeds(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        assert metadata.observations == {}
        assert metadata.precomputed_actions == {}

    def test_overlapping_keys_raises(
        self,
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
    ):
        shared_key = "shared_key"
        observations = {
            shared_key: position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            )
        }
        actions = {
            shared_key: precomputed_action_metadata_factory(
                storage_dimension=7, prediction_dimension=3
            )
        }

        with pytest.raises(ValueError, match="Keys cannot be both"):
            DatasetMetadata(observations=observations, precomputed_actions=actions)

    def test_disjoint_keys_succeeds(
        self,
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
    ):
        observations = {
            "position": position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            )
        }
        actions = {
            "action": precomputed_action_metadata_factory(
                storage_dimension=7, prediction_dimension=3
            )
        }

        metadata = DatasetMetadata(
            observations=observations, precomputed_actions=actions
        )

        assert len(metadata.get_all_keys()) == 2

    @pytest.mark.parametrize(
        "dict_name, call_index",
        [
            ("observations", 0),
            ("precomputed_actions", 1),
        ],
        ids=["observations", "precomputed_actions"],
    )
    def test_resolve_dict_keys_called_on_both_dicts(
        self,
        dict_name: str,
        call_index: int,
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
    ):
        observations = {
            "pos": position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            )
        }
        actions = {
            "act": precomputed_action_metadata_factory(
                storage_dimension=7, prediction_dimension=3
            )
        }
        target_dict = {"observations": observations, "precomputed_actions": actions}[
            dict_name
        ]

        with patch(
            "versatil.data.raw.zarr_meta.resolve_dict_keys",
            side_effect=lambda d: d,
        ) as mock_resolve:
            DatasetMetadata(observations=observations, precomputed_actions=actions)

        assert mock_resolve.call_args_list[call_index].args[0] is target_dict

    def test_duplicate_camera_key_check_is_unreachable(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        """The duplicate camera key validation in __post_init__ (lines 56-60) is dead code.

        Dict keys are unique by definition, so the list comprehension
        `[k for k, v in self.observations.items() if isinstance(v, CameraMetadata)]`
        can never produce duplicates. The `len(camera_keys) != len(set(camera_keys))`
        check is always False.
        """
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(camera_key=Cameras.LEFT.value),
            Cameras.RIGHT.value: camera_metadata_factory(
                camera_key=Cameras.RIGHT.value
            ),
        }
        metadata = dataset_metadata_factory(observations=observations)

        camera_keys = [
            k for k, v in metadata.observations.items() if isinstance(v, CameraMetadata)
        ]
        assert len(camera_keys) == len(set(camera_keys))


class TestDatasetMetadataObservationProperties:
    @pytest.mark.parametrize(
        "property_name, expected_key, expected_type",
        [
            ("cameras", Cameras.LEFT.value, CameraMetadata),
            ("position_observations", "position", PositionObservationMetadata),
            ("orientation_observations", "orientation", OrientationObservationMetadata),
            ("gripper_observations", "gripper", GripperObservationMetadata),
        ],
        ids=["cameras", "position", "orientation", "gripper"],
    )
    @pytest.mark.parametrize(
        "frame, orientation_repr, gripper_range",
        [
            (
                CoordinateSystem.ROBOT_BASE.value,
                OrientationRepresentation.ROLL.value,
                BinaryGripperRange.ZERO_ONE.value,
            ),
            (
                CoordinateSystem.CAMERA.value,
                OrientationRepresentation.EULER.value,
                BinaryGripperRange.MINUS_ONE_ONE.value,
            ),
        ],
        ids=["robot_base_roll_01", "camera_euler_minus11"],
    )
    def test_observation_property_returns_only_matching_type(
        self,
        property_name: str,
        expected_key: str,
        expected_type: type,
        frame: str,
        orientation_repr: str,
        gripper_range: str,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(camera_key=Cameras.LEFT.value),
            "position": position_observation_metadata_factory(dimension=3, frame=frame),
            "orientation": orientation_observation_metadata_factory(
                dimension=1, frame=frame, orientation_representation=orientation_repr
            ),
            "gripper": gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=gripper_range,
                dimension=1,
            ),
        }
        metadata = dataset_metadata_factory(observations=observations)

        result = getattr(metadata, property_name)

        assert len(result) == 1
        assert expected_key in result
        assert isinstance(result[expected_key], expected_type)

    def test_depth_cameras_returns_only_depth_metadata(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(
            observations={
                Cameras.LEFT.value: camera_metadata_factory(
                    camera_key=Cameras.LEFT.value
                ),
                Cameras.DEPTH.value: camera_metadata_factory(
                    camera_key=Cameras.DEPTH.value,
                    channels=1,
                ),
            }
        )

        assert set(metadata.depth_cameras) == {Cameras.DEPTH.value}

    def test_rgb_cameras_excludes_depth_metadata(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(
            observations={
                Cameras.LEFT.value: camera_metadata_factory(
                    camera_key=Cameras.LEFT.value
                ),
                Cameras.RIGHT.value: camera_metadata_factory(
                    camera_key=Cameras.RIGHT.value
                ),
                Cameras.DEPTH.value: camera_metadata_factory(
                    camera_key=Cameras.DEPTH.value,
                    channels=1,
                ),
            }
        )

        assert set(metadata.rgb_cameras) == {
            Cameras.LEFT.value,
            Cameras.RIGHT.value,
        }

    @pytest.mark.parametrize(
        "property_name",
        [
            "cameras",
            "position_observations",
            "orientation_observations",
            "gripper_observations",
            "proprioceptive_observations",
            "custom_observations",
        ],
        ids=[
            "cameras",
            "position",
            "orientation",
            "gripper",
            "proprioceptive",
            "custom",
        ],
    )
    def test_observation_property_empty_when_no_matching_type(
        self,
        property_name: str,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={})

        assert getattr(metadata, property_name) == {}

    def test_proprioceptive_observations_includes_position_orientation_gripper(
        self,
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        camera_metadata_factory: Callable[..., CameraMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            "position": position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            ),
            "orientation": orientation_observation_metadata_factory(
                dimension=1, frame=CoordinateSystem.ROBOT_BASE.value
            ),
            "gripper": gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
                dimension=1,
            ),
            Cameras.LEFT.value: camera_metadata_factory(camera_key=Cameras.LEFT.value),
        }
        metadata = dataset_metadata_factory(observations=observations)

        proprioceptive = metadata.proprioceptive_observations

        assert set(proprioceptive.keys()) == {"position", "orientation", "gripper"}

    def test_proprioceptive_observations_excludes_camera_and_custom(
        self,
        camera_metadata_factory: Callable[..., CameraMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        custom_observation = ObservationMetadata(
            raw_data_column_keys=["lang"],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        observations = {
            Cameras.LEFT.value: camera_metadata_factory(camera_key=Cameras.LEFT.value),
            "custom": custom_observation,
        }
        metadata = dataset_metadata_factory(observations=observations)

        assert metadata.proprioceptive_observations == {}

    def test_custom_observations_returns_base_observation_metadata_only(
        self,
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        camera_metadata_factory: Callable[..., CameraMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        custom_observation = ObservationMetadata(
            raw_data_column_keys=["lang"],
            dimension=1,
            dtype="str",
            is_numerical=False,
            needs_normalization=False,
        )
        observations = {
            "custom": custom_observation,
            "position": position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            ),
            Cameras.LEFT.value: camera_metadata_factory(camera_key=Cameras.LEFT.value),
        }
        metadata = dataset_metadata_factory(observations=observations)

        result = metadata.custom_observations

        assert list(result.keys()) == ["custom"]


class TestDatasetMetadataActionProperties:
    @pytest.mark.parametrize(
        "property_name, expected_key, expected_type",
        [
            ("position_actions", "pos_action", PositionActionMetadata),
            ("orientation_actions", "ori_action", OrientationActionMetadata),
            ("gripper_actions", "grip_action", GripperActionMetadata),
        ],
        ids=["position", "orientation", "gripper"],
    )
    @pytest.mark.parametrize(
        "frame, orientation_repr, gripper_range",
        [
            (
                CoordinateSystem.ROBOT_BASE.value,
                OrientationRepresentation.ROLL.value,
                BinaryGripperRange.ZERO_ONE.value,
            ),
            (
                CoordinateSystem.CAMERA.value,
                OrientationRepresentation.QUATERNION.value,
                BinaryGripperRange.MINUS_ONE_ONE.value,
            ),
        ],
        ids=["robot_base_roll_01", "camera_quaternion_minus11"],
    )
    def test_action_property_returns_only_matching_type(
        self,
        property_name: str,
        expected_key: str,
        expected_type: type,
        frame: str,
        orientation_repr: str,
        gripper_range: str,
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
        orientation_action_metadata_factory: Callable[..., OrientationActionMetadata],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        actions = {
            "pos_action": position_action_metadata_factory(
                frame=frame,
                storage_dimension=3,
                prediction_dimension=3,
            ),
            "ori_action": orientation_action_metadata_factory(
                frame=frame,
                orientation_representation=orientation_repr,
                storage_dimension=1,
                prediction_dimension=1,
            ),
            "grip_action": gripper_action_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=gripper_range,
                storage_dimension=1,
                prediction_dimension=1,
            ),
        }
        metadata = dataset_metadata_factory(precomputed_actions=actions)

        result = getattr(metadata, property_name)

        assert len(result) == 1
        assert expected_key in result
        assert isinstance(result[expected_key], expected_type)

    @pytest.mark.parametrize(
        "property_name",
        [
            "position_actions",
            "orientation_actions",
            "gripper_actions",
            "custom_actions",
        ],
        ids=["position", "orientation", "gripper", "custom"],
    )
    def test_action_property_empty_when_no_matching_type(
        self,
        property_name: str,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(precomputed_actions={})

        assert getattr(metadata, property_name) == {}

    def test_custom_actions_excludes_typed_subtypes(
        self,
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
        orientation_action_metadata_factory: Callable[..., OrientationActionMetadata],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        actions = {
            "custom": precomputed_action_metadata_factory(
                storage_dimension=1, prediction_dimension=1
            ),
            "pos": position_action_metadata_factory(
                frame=CoordinateSystem.ROBOT_BASE.value,
                storage_dimension=3,
                prediction_dimension=3,
            ),
            "ori": orientation_action_metadata_factory(
                frame=CoordinateSystem.ROBOT_BASE.value,
                orientation_representation=OrientationRepresentation.ROLL.value,
                storage_dimension=1,
                prediction_dimension=1,
            ),
            "grip": gripper_action_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
                storage_dimension=1,
                prediction_dimension=1,
            ),
        }
        metadata = dataset_metadata_factory(precomputed_actions=actions)

        result = metadata.custom_actions

        assert list(result.keys()) == ["custom"]


class TestDatasetMetadataUtilityMethods:
    @pytest.mark.parametrize(
        "num_observations, num_actions, expected_total",
        [
            (2, 1, 3),
            (0, 0, 0),
            (3, 0, 3),
            (0, 2, 2),
        ],
        ids=[
            "mixed_obs_and_actions",
            "empty_metadata",
            "observations_only",
            "actions_only",
        ],
    )
    def test_get_all_keys_returns_correct_count(
        self,
        num_observations: int,
        num_actions: int,
        expected_total: int,
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            f"obs_{i}": position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            )
            for i in range(num_observations)
        }
        actions = {
            f"act_{i}": precomputed_action_metadata_factory(
                storage_dimension=7, prediction_dimension=3
            )
            for i in range(num_actions)
        }
        metadata = dataset_metadata_factory(
            observations=observations, precomputed_actions=actions
        )

        assert len(metadata.get_all_keys()) == expected_total

    @pytest.mark.parametrize(
        "num_cameras, num_other_obs, expected_camera_count",
        [
            (2, 1, 2),
            (0, 1, 0),
            (1, 0, 1),
        ],
        ids=["two_cameras_one_other", "no_cameras", "one_camera_only"],
    )
    def test_get_camera_keys_count(
        self,
        num_cameras: int,
        num_other_obs: int,
        expected_camera_count: int,
        camera_metadata_factory: Callable[..., CameraMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        camera_enum_values = [Cameras.LEFT.value, Cameras.RIGHT.value]
        observations = {}
        for i in range(num_cameras):
            observations[camera_enum_values[i]] = camera_metadata_factory(
                camera_key=camera_enum_values[i]
            )
        for i in range(num_other_obs):
            observations[f"obs_{i}"] = position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            )
        metadata = dataset_metadata_factory(observations=observations)

        assert len(metadata.get_camera_keys()) == expected_camera_count

    @pytest.mark.parametrize(
        "position_dim, orientation_dim, gripper_dim, expected_total",
        [
            (3, 1, 1, 5),
            (3, 0, 0, 3),
            (0, 0, 0, 0),
            (3, 3, 1, 7),
        ],
        ids=[
            "position_orientation_gripper",
            "position_only",
            "no_proprioceptive",
            "large_orientation",
        ],
    )
    def test_get_proprio_dimension(
        self,
        position_dim: int,
        orientation_dim: int,
        gripper_dim: int,
        expected_total: int,
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {}
        if position_dim > 0:
            observations["position"] = position_observation_metadata_factory(
                dimension=position_dim, frame=CoordinateSystem.ROBOT_BASE.value
            )
        if orientation_dim > 0:
            observations["orientation"] = orientation_observation_metadata_factory(
                dimension=orientation_dim,
                frame=CoordinateSystem.ROBOT_BASE.value,
                orientation_representation=OrientationRepresentation.EULER.value,
                raw_data_column_keys=["r", "p", "y"][:orientation_dim],
            )
        if gripper_dim > 0:
            observations["gripper"] = gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
                dimension=gripper_dim,
            )
        metadata = dataset_metadata_factory(observations=observations)

        assert metadata.get_proprio_dimension() == expected_total

    @pytest.mark.parametrize(
        "has_gripper, expected_dimension",
        [
            (True, 1),
            (False, 0),
        ],
        ids=["with_gripper", "without_gripper"],
    )
    def test_get_gripper_dimension(
        self,
        has_gripper: bool,
        expected_dimension: int,
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {}
        if has_gripper:
            observations["gripper"] = gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
                binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
                dimension=1,
            )
        metadata = dataset_metadata_factory(observations=observations)

        assert metadata.get_gripper_dimension() == expected_dimension

    @pytest.mark.parametrize(
        "has_actions, expected",
        [
            (True, True),
            (False, False),
        ],
        ids=["with_actions", "without_actions"],
    )
    def test_has_precomputed_actions(
        self,
        has_actions: bool,
        expected: bool,
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        actions = {}
        if has_actions:
            actions["action"] = precomputed_action_metadata_factory(
                storage_dimension=7, prediction_dimension=3
            )
        metadata = dataset_metadata_factory(precomputed_actions=actions)

        assert metadata.has_precomputed_actions() is expected

    def test_get_precomputed_action_existing_key(
        self,
        precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        action = precomputed_action_metadata_factory(
            storage_dimension=7, prediction_dimension=3
        )
        metadata = dataset_metadata_factory(precomputed_actions={"pos_action": action})

        assert metadata.get_precomputed_action("pos_action") is action

    def test_get_precomputed_action_missing_key_returns_none(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        assert metadata.get_precomputed_action("nonexistent") is None

    def test_get_observation_existing_key(
        self,
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observation = position_observation_metadata_factory(
            dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
        )
        metadata = dataset_metadata_factory(observations={"position": observation})

        assert metadata.get_observation("position") is observation

    def test_get_observation_missing_key_returns_none(
        self,
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        metadata = dataset_metadata_factory(observations={}, precomputed_actions={})

        assert metadata.get_observation("nonexistent") is None

    @pytest.mark.parametrize(
        "key, key_present, expected",
        [
            ("position", True, True),
            ("nonexistent", False, False),
        ],
        ids=["key_present", "key_absent"],
    )
    def test_has_observation(
        self,
        key: str,
        key_present: bool,
        expected: bool,
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {}
        if key_present:
            observations[key] = position_observation_metadata_factory(
                dimension=3, frame=CoordinateSystem.ROBOT_BASE.value
            )
        metadata = dataset_metadata_factory(observations=observations)

        assert metadata.has_observation(key) is expected

    @pytest.mark.parametrize(
        "dict_key, raw_key, expectation",
        [
            (Cameras.LEFT.value, RawCameraKey.LEFT.value, does_not_raise()),
            (Cameras.AGENTVIEW.value, RawCameraKey.IMAGE.value, does_not_raise()),
            (Cameras.EYE_IN_HAND.value, RawCameraKey.IMAGE_2.value, does_not_raise()),
            (Cameras.AGENTVIEW.value, RawCameraKey.FRONT.value, does_not_raise()),
            (
                Cameras.EYE_IN_HAND.value,
                RawCameraKey.IMAGE.value,
                pytest.raises(ValueError, match="which maps to"),
            ),
            (
                Cameras.LEFT.value,
                RawCameraKey.IMAGE_METAWORLD.value,
                pytest.raises(ValueError, match="which maps to"),
            ),
        ],
        ids=[
            "tso_identity",
            "libero_lerobot_agentview",
            "libero_lerobot_eye_in_hand",
            "libero_plus_front",
            "mismatch_eye_in_hand_with_image",
            "mismatch_left_with_metaworld",
        ],
    )
    def test_camera_mapping_validation(
        self,
        dict_key: str,
        raw_key: str,
        expectation,
        camera_metadata_factory: Callable[..., CameraMetadata],
        dataset_metadata_factory: Callable[..., DatasetMetadata],
    ):
        observations = {
            dict_key: camera_metadata_factory(camera_key=raw_key),
        }
        with expectation:
            dataset_metadata_factory(observations=observations)
