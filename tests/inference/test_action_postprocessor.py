"""Tests for versatil.inference.action_postprocessor module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch
from versatil_constants.shared import (
    ActionComponent,
    ActionComputationMethod,
    ActionMetadataField,
    BinaryGripperRange,
    CoordinateSystem,
    GripperType,
    OrientationRepresentation,
)

from versatil.data.metadata import (
    ActionMetadata,
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
    OrientationActionMetadata,
    OrientationObservationMetadata,
    PositionActionMetadata,
    PositionObservationMetadata,
)
from versatil.data.task import ActionSpace
from versatil.inference.action_postprocessor import ActionPostprocessor


@pytest.fixture
def action_postprocessor_factory(
    action_space_factory: Callable[..., ActionSpace],
) -> Callable[..., ActionPostprocessor]:
    def factory(
        actions_metadata: dict[str, ActionMetadata] | None = None,
        denoising_thresholds: dict[str, float] | None = None,
    ) -> ActionPostprocessor:
        action_space = action_space_factory(
            actions_metadata=actions_metadata,
        )
        if denoising_thresholds is None:
            denoising_thresholds = {}
        return ActionPostprocessor(
            action_space=action_space,
            denoising_thresholds=denoising_thresholds,
        )

    return factory


@pytest.mark.unit
class TestActionPostprocessorInitialization:
    def test_stores_action_space_and_thresholds(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
    ):
        thresholds = {"position": 0.5}
        postprocessor = action_postprocessor_factory(
            denoising_thresholds=thresholds,
        )

        assert postprocessor.denoising_thresholds == thresholds
        assert postprocessor.action_space.actions_metadata == {}


@pytest.mark.unit
class TestPostprocessGripperAction:
    def test_binary_zero_one_large_positive_logit_maps_to_one(
        self,
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        metadata = gripper_action_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
        )
        raw_value = np.array([10.0])

        result = ActionPostprocessor._postprocess_gripper_action(
            raw_value=raw_value,
            action_meta=metadata,
        )

        assert result[0] == 1.0

    def test_binary_zero_one_large_negative_logit_maps_to_zero(
        self,
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        metadata = gripper_action_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
        )
        raw_value = np.array([-10.0])

        result = ActionPostprocessor._postprocess_gripper_action(
            raw_value=raw_value,
            action_meta=metadata,
        )

        assert result[0] == 0.0

    def test_binary_zero_one_zero_logit_maps_to_zero(
        self,
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        # sigmoid(0) = 0.5, threshold is > 0.5, so 0.5 is NOT > 0.5 => maps to 0
        metadata = gripper_action_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
        )
        raw_value = np.array([0.0])

        result = ActionPostprocessor._postprocess_gripper_action(
            raw_value=raw_value,
            action_meta=metadata,
        )

        assert result[0] == 0.0

    def test_binary_minus_one_one_large_positive_logit_maps_to_one(
        self,
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        metadata = gripper_action_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.MINUS_ONE_ONE.value,
        )
        raw_value = np.array([10.0])

        result = ActionPostprocessor._postprocess_gripper_action(
            raw_value=raw_value,
            action_meta=metadata,
        )

        # float(True) * 2.0 - 1.0 = 1.0
        assert result[0] == 1.0

    def test_binary_minus_one_one_large_negative_logit_maps_to_minus_one(
        self,
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        metadata = gripper_action_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.MINUS_ONE_ONE.value,
        )
        raw_value = np.array([-10.0])

        result = ActionPostprocessor._postprocess_gripper_action(
            raw_value=raw_value,
            action_meta=metadata,
        )

        # float(False) * 2.0 - 1.0 = -1.0
        assert result[0] == -1.0

    def test_binary_minus_one_one_zero_logit_maps_to_minus_one(
        self,
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        # sigmoid(0) = 0.5, 0.5 > 0.5 is False => 0.0 * 2.0 - 1.0 = -1.0
        metadata = gripper_action_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.MINUS_ONE_ONE.value,
        )
        raw_value = np.array([0.0])

        result = ActionPostprocessor._postprocess_gripper_action(
            raw_value=raw_value,
            action_meta=metadata,
        )

        assert result[0] == -1.0

    def test_continuous_gripper_passes_through_unchanged(
        self,
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        metadata = gripper_action_metadata_factory(
            gripper_type=GripperType.CONTINUOUS.value,
        )
        raw_value = np.array([0.42])

        result = ActionPostprocessor._postprocess_gripper_action(
            raw_value=raw_value,
            action_meta=metadata,
        )

        np.testing.assert_array_equal(result, raw_value)

    def test_on_the_fly_binary_gripper_applies_sigmoid(
        self,
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
    ):
        gripper_obs_meta = gripper_observation_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
        )
        metadata = on_the_fly_action_metadata_factory(
            source_metadata=gripper_obs_meta,
            computation_method=ActionComputationMethod.NEXT_TIMESTEP.value,
        )
        raw_value = np.array([10.0])

        result = ActionPostprocessor._postprocess_gripper_action(
            raw_value=raw_value,
            action_meta=metadata,
        )

        assert result[0] == 1.0

    def test_on_the_fly_continuous_gripper_passes_through(
        self,
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
    ):
        gripper_obs_meta = gripper_observation_metadata_factory(
            gripper_type=GripperType.CONTINUOUS.value,
        )
        metadata = on_the_fly_action_metadata_factory(
            source_metadata=gripper_obs_meta,
            computation_method=ActionComputationMethod.NEXT_TIMESTEP.value,
        )
        raw_value = np.array([0.75])

        result = ActionPostprocessor._postprocess_gripper_action(
            raw_value=raw_value,
            action_meta=metadata,
        )

        np.testing.assert_array_equal(result, raw_value)

    def test_non_gripper_metadata_passes_through(
        self,
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
    ):
        metadata = position_action_metadata_factory()
        raw_value = np.array([1.0, 2.0, 3.0])

        result = ActionPostprocessor._postprocess_gripper_action(
            raw_value=raw_value,
            action_meta=metadata,
        )

        np.testing.assert_array_equal(result, raw_value)


@pytest.mark.unit
class TestFormatAction:
    def test_maps_versatil_keys_to_action_component_keys(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        position_meta = position_action_metadata_factory(prediction_dimension=3)
        gripper_meta = gripper_action_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
        )
        postprocessor = action_postprocessor_factory(
            actions_metadata={
                "position_key": position_meta,
                "gripper_key": gripper_meta,
            },
        )
        action_dict = {
            "position_key": torch.tensor([1.0, 2.0, 3.0]),
            "gripper_key": torch.tensor([10.0]),
        }

        result = postprocessor.format_action(action_dict=action_dict)

        assert ActionComponent.POSITION.value in result
        assert ActionComponent.GRIPPER.value in result
        assert result[ActionComponent.POSITION.value] == [1.0, 2.0, 3.0]
        assert result[ActionComponent.GRIPPER.value] == [1.0]

    def test_skips_metadata_without_prediction_head(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
    ):
        position_meta = position_action_metadata_factory()
        position_meta.requires_prediction_head = False
        postprocessor = action_postprocessor_factory(
            actions_metadata={"position_key": position_meta},
        )

        result = postprocessor.format_action(action_dict={})

        assert result == {}

    def test_applies_denoising_zeroes_small_values(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
    ):
        position_meta = position_action_metadata_factory(prediction_dimension=3)
        postprocessor = action_postprocessor_factory(
            actions_metadata={"position_key": position_meta},
            denoising_thresholds={"position_key": 1.0},
        )
        # norm([0.001, 0.001, 0.001]) ~ 0.0017, below threshold 1.0
        action_dict = {
            "position_key": torch.tensor([0.001, 0.001, 0.001]),
        }

        result = postprocessor.format_action(action_dict=action_dict)

        assert result[ActionComponent.POSITION.value] == [0.0, 0.0, 0.0]

    def test_denoising_preserves_large_values(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
    ):
        position_meta = position_action_metadata_factory(prediction_dimension=3)
        postprocessor = action_postprocessor_factory(
            actions_metadata={"position_key": position_meta},
            denoising_thresholds={"position_key": 0.5},
        )
        # norm([1.0, 1.0, 1.0]) ~ 1.73, above threshold 0.5
        action_dict = {
            "position_key": torch.tensor([1.0, 1.0, 1.0]),
        }

        result = postprocessor.format_action(action_dict=action_dict)

        assert result[ActionComponent.POSITION.value] == [1.0, 1.0, 1.0]

    def test_no_denoising_when_key_not_in_thresholds(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
    ):
        position_meta = position_action_metadata_factory(prediction_dimension=3)
        postprocessor = action_postprocessor_factory(
            actions_metadata={"position_key": position_meta},
            denoising_thresholds={"other_key": 0.5},
        )
        action_dict = {
            "position_key": torch.tensor([0.001, 0.001, 0.001]),
        }

        result = postprocessor.format_action(action_dict=action_dict)

        # Value is small but no threshold for this key, so not zeroed
        expected = [
            pytest.approx(0.001, abs=1e-6),
            pytest.approx(0.001, abs=1e-6),
            pytest.approx(0.001, abs=1e-6),
        ]
        assert result[ActionComponent.POSITION.value] == expected

    def test_applies_gripper_sigmoid_before_denoising(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        gripper_meta = gripper_action_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
        )
        # Denoising threshold is 0.5, sigmoid(10) = ~1.0 which has norm > 0.5
        postprocessor = action_postprocessor_factory(
            actions_metadata={"gripper_key": gripper_meta},
            denoising_thresholds={"gripper_key": 0.5},
        )
        action_dict = {
            "gripper_key": torch.tensor([10.0]),
        }

        result = postprocessor.format_action(action_dict=action_dict)

        # After sigmoid+threshold: 1.0; norm(1.0) = 1.0 > 0.5, so not zeroed
        assert result[ActionComponent.GRIPPER.value] == [1.0]

    def test_format_action_with_on_the_fly_metadata(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        position_obs_meta = position_observation_metadata_factory(dimension=3)
        on_the_fly_meta = on_the_fly_action_metadata_factory(
            source_metadata=position_obs_meta,
            computation_method=ActionComputationMethod.DELTA.value,
        )
        postprocessor = action_postprocessor_factory(
            actions_metadata={"position_key": on_the_fly_meta},
        )
        action_dict = {
            "position_key": torch.tensor([1.0, 2.0, 3.0]),
        }

        result = postprocessor.format_action(action_dict=action_dict)

        assert ActionComponent.POSITION.value in result
        assert result[ActionComponent.POSITION.value] == [1.0, 2.0, 3.0]

    def test_format_action_with_empty_metadata(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
    ):
        postprocessor = action_postprocessor_factory(
            actions_metadata={},
        )

        result = postprocessor.format_action(action_dict={})

        assert result == {}

    def test_denoising_threshold_zero_never_zeroes(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
    ):
        # threshold=0.0 means np.linalg.norm(value) < 0.0 is always False,
        # so values are never zeroed regardless of magnitude.
        position_meta = position_action_metadata_factory(prediction_dimension=3)
        postprocessor = action_postprocessor_factory(
            actions_metadata={"position_key": position_meta},
            denoising_thresholds={"position_key": 0.0},
        )
        action_dict = {
            "position_key": torch.tensor([1e-10, 1e-10, 1e-10]),
        }

        result = postprocessor.format_action(action_dict=action_dict)

        # Even tiny values survive because norm >= 0.0 is always true
        expected = [
            pytest.approx(1e-10, abs=1e-15),
            pytest.approx(1e-10, abs=1e-15),
            pytest.approx(1e-10, abs=1e-15),
        ]
        assert result[ActionComponent.POSITION.value] == expected

    def test_on_the_fly_non_gripper_source_in_postprocess(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        # OnTheFlyActionMetadata wrapping PositionObservationMetadata:
        # the gripper postprocessing should fall through to passthrough
        # because the source_metadata is not GripperObservationMetadata.
        position_obs_meta = position_observation_metadata_factory(dimension=3)
        on_the_fly_meta = on_the_fly_action_metadata_factory(
            source_metadata=position_obs_meta,
            computation_method=ActionComputationMethod.DELTA.value,
        )
        raw_value = np.array([0.5, -0.3, 0.7])

        result = ActionPostprocessor._postprocess_gripper_action(
            raw_value=raw_value,
            action_meta=on_the_fly_meta,
        )

        # Values pass through unchanged -- no sigmoid applied
        np.testing.assert_array_equal(result, raw_value)


@pytest.mark.unit
class TestBuildActionMetadata:
    def test_position_action_metadata_includes_dimension_and_frame(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
    ):
        position_meta = position_action_metadata_factory(
            frame=CoordinateSystem.CAMERA.value,
            prediction_dimension=3,
        )
        postprocessor = action_postprocessor_factory(
            actions_metadata={"position_key": position_meta},
        )

        result = postprocessor.build_action_metadata()

        entry = result[ActionComponent.POSITION.value]
        assert entry[ActionMetadataField.DIMENSION.value] == 3
        assert entry[ActionMetadataField.FRAME.value] == CoordinateSystem.CAMERA.value
        assert ActionMetadataField.GRIPPER_TYPE.value not in entry
        assert ActionMetadataField.ORIENTATION_REPRESENTATION.value not in entry
        assert ActionMetadataField.ACTION_TYPE.value not in entry

    def test_orientation_action_metadata_includes_representation_and_frame(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        orientation_action_metadata_factory: Callable[..., OrientationActionMetadata],
    ):
        orientation_meta = orientation_action_metadata_factory(
            frame=CoordinateSystem.ROBOT_BASE.value,
            orientation_representation=OrientationRepresentation.QUATERNION.value,
            prediction_dimension=3,
        )
        postprocessor = action_postprocessor_factory(
            actions_metadata={"orientation_key": orientation_meta},
        )

        result = postprocessor.build_action_metadata()

        entry = result[ActionComponent.ORIENTATION.value]
        assert entry[ActionMetadataField.DIMENSION.value] == 3
        assert (
            entry[ActionMetadataField.FRAME.value] == CoordinateSystem.ROBOT_BASE.value
        )
        assert (
            entry[ActionMetadataField.ORIENTATION_REPRESENTATION.value]
            == OrientationRepresentation.QUATERNION.value
        )
        assert ActionMetadataField.GRIPPER_TYPE.value not in entry

    def test_gripper_action_metadata_includes_type_and_range(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        gripper_meta = gripper_action_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.MINUS_ONE_ONE.value,
        )
        postprocessor = action_postprocessor_factory(
            actions_metadata={"gripper_key": gripper_meta},
        )

        result = postprocessor.build_action_metadata()

        entry = result[ActionComponent.GRIPPER.value]
        assert entry[ActionMetadataField.DIMENSION.value] == 1
        assert entry[ActionMetadataField.GRIPPER_TYPE.value] == GripperType.BINARY.value
        assert (
            entry[ActionMetadataField.BINARY_GRIPPER_RANGE.value]
            == BinaryGripperRange.MINUS_ONE_ONE.value
        )
        assert ActionMetadataField.FRAME.value not in entry

    def test_on_the_fly_position_metadata_includes_frame_and_action_type(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        position_obs_meta = position_observation_metadata_factory(
            dimension=3,
            frame=CoordinateSystem.CAMERA.value,
        )
        on_the_fly_meta = on_the_fly_action_metadata_factory(
            source_metadata=position_obs_meta,
            computation_method=ActionComputationMethod.DELTA.value,
        )
        postprocessor = action_postprocessor_factory(
            actions_metadata={"position_key": on_the_fly_meta},
        )

        result = postprocessor.build_action_metadata()

        entry = result[ActionComponent.POSITION.value]
        assert entry[ActionMetadataField.DIMENSION.value] == 3
        assert entry[ActionMetadataField.FRAME.value] == CoordinateSystem.CAMERA.value
        assert (
            entry[ActionMetadataField.ACTION_TYPE.value]
            == ActionComputationMethod.DELTA.value
        )

    def test_on_the_fly_orientation_metadata_includes_representation(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
    ):
        orientation_obs_meta = orientation_observation_metadata_factory(
            dimension=3,
            frame=CoordinateSystem.ROBOT_BASE.value,
            orientation_representation=OrientationRepresentation.EULER.value,
        )
        on_the_fly_meta = on_the_fly_action_metadata_factory(
            source_metadata=orientation_obs_meta,
            computation_method=ActionComputationMethod.NEXT_TIMESTEP.value,
        )
        postprocessor = action_postprocessor_factory(
            actions_metadata={"orientation_key": on_the_fly_meta},
        )

        result = postprocessor.build_action_metadata()

        entry = result[ActionComponent.ORIENTATION.value]
        assert entry[ActionMetadataField.DIMENSION.value] == 3
        assert (
            entry[ActionMetadataField.FRAME.value] == CoordinateSystem.ROBOT_BASE.value
        )
        assert (
            entry[ActionMetadataField.ORIENTATION_REPRESENTATION.value]
            == OrientationRepresentation.EULER.value
        )
        assert (
            entry[ActionMetadataField.ACTION_TYPE.value]
            == ActionComputationMethod.NEXT_TIMESTEP.value
        )

    def test_on_the_fly_gripper_metadata_includes_type_and_range(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
    ):
        gripper_obs_meta = gripper_observation_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.ZERO_ONE.value,
        )
        on_the_fly_meta = on_the_fly_action_metadata_factory(
            source_metadata=gripper_obs_meta,
            computation_method=ActionComputationMethod.NEXT_TIMESTEP.value,
        )
        postprocessor = action_postprocessor_factory(
            actions_metadata={"gripper_key": on_the_fly_meta},
        )

        result = postprocessor.build_action_metadata()

        entry = result[ActionComponent.GRIPPER.value]
        assert entry[ActionMetadataField.DIMENSION.value] == 1
        assert entry[ActionMetadataField.GRIPPER_TYPE.value] == GripperType.BINARY.value
        assert (
            entry[ActionMetadataField.BINARY_GRIPPER_RANGE.value]
            == BinaryGripperRange.ZERO_ONE.value
        )
        assert (
            entry[ActionMetadataField.ACTION_TYPE.value]
            == ActionComputationMethod.NEXT_TIMESTEP.value
        )

    def test_skips_metadata_without_prediction_head(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
    ):
        position_meta = position_action_metadata_factory()
        position_meta.requires_prediction_head = False
        postprocessor = action_postprocessor_factory(
            actions_metadata={"position_key": position_meta},
        )

        result = postprocessor.build_action_metadata()

        assert result == {}

    def test_multiple_action_components(
        self,
        action_postprocessor_factory: Callable[..., ActionPostprocessor],
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
        orientation_action_metadata_factory: Callable[..., OrientationActionMetadata],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        postprocessor = action_postprocessor_factory(
            actions_metadata={
                "position_key": position_action_metadata_factory(
                    prediction_dimension=3,
                ),
                "orientation_key": orientation_action_metadata_factory(
                    prediction_dimension=3,
                ),
                "gripper_key": gripper_action_metadata_factory(),
            },
        )

        result = postprocessor.build_action_metadata()

        assert len(result) == 3
        assert ActionComponent.POSITION.value in result
        assert ActionComponent.ORIENTATION.value in result
        assert ActionComponent.GRIPPER.value in result


@pytest.mark.unit
class TestAddActionTypeMetadata:
    def test_adds_computation_method_for_on_the_fly_metadata(
        self,
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        metadata = on_the_fly_action_metadata_factory(
            computation_method=ActionComputationMethod.DELTA.value,
        )
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_action_type_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert (
            entry[ActionMetadataField.ACTION_TYPE.value]
            == ActionComputationMethod.DELTA.value
        )

    def test_does_not_add_for_precomputed_metadata(
        self,
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
    ):
        metadata = position_action_metadata_factory()
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_action_type_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert ActionMetadataField.ACTION_TYPE.value not in entry


@pytest.mark.unit
class TestAddFrameMetadata:
    @pytest.mark.parametrize(
        "frame",
        [CoordinateSystem.ROBOT_BASE.value, CoordinateSystem.CAMERA.value],
    )
    def test_adds_frame_for_position_action_metadata(
        self,
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
        frame: str,
    ):
        metadata = position_action_metadata_factory(frame=frame)
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_frame_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert entry[ActionMetadataField.FRAME.value] == frame

    def test_adds_frame_for_orientation_action_metadata(
        self,
        orientation_action_metadata_factory: Callable[..., OrientationActionMetadata],
    ):
        metadata = orientation_action_metadata_factory(
            frame=CoordinateSystem.CAMERA.value,
        )
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_frame_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert entry[ActionMetadataField.FRAME.value] == CoordinateSystem.CAMERA.value

    def test_adds_frame_for_on_the_fly_position_metadata(
        self,
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        position_obs = position_observation_metadata_factory(
            frame=CoordinateSystem.ROBOT_BASE.value,
        )
        metadata = on_the_fly_action_metadata_factory(source_metadata=position_obs)
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_frame_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert (
            entry[ActionMetadataField.FRAME.value] == CoordinateSystem.ROBOT_BASE.value
        )

    def test_adds_frame_for_on_the_fly_orientation_metadata(
        self,
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
    ):
        orientation_obs = orientation_observation_metadata_factory(
            frame=CoordinateSystem.CAMERA.value,
        )
        metadata = on_the_fly_action_metadata_factory(
            source_metadata=orientation_obs,
        )
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_frame_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert entry[ActionMetadataField.FRAME.value] == CoordinateSystem.CAMERA.value

    def test_does_not_add_frame_for_gripper_action_metadata(
        self,
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        metadata = gripper_action_metadata_factory()
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_frame_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert ActionMetadataField.FRAME.value not in entry

    def test_does_not_add_frame_for_on_the_fly_gripper_metadata(
        self,
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
    ):
        gripper_obs = gripper_observation_metadata_factory()
        metadata = on_the_fly_action_metadata_factory(source_metadata=gripper_obs)
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_frame_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert ActionMetadataField.FRAME.value not in entry


@pytest.mark.unit
class TestAddOrientationMetadata:
    @pytest.mark.parametrize(
        "orientation_representation",
        [
            OrientationRepresentation.EULER.value,
            OrientationRepresentation.QUATERNION.value,
        ],
    )
    def test_adds_representation_for_orientation_action_metadata(
        self,
        orientation_action_metadata_factory: Callable[..., OrientationActionMetadata],
        orientation_representation: str,
    ):
        metadata = orientation_action_metadata_factory(
            orientation_representation=orientation_representation,
        )
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_orientation_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert (
            entry[ActionMetadataField.ORIENTATION_REPRESENTATION.value]
            == orientation_representation
        )

    def test_adds_representation_for_on_the_fly_orientation_metadata(
        self,
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
    ):
        orientation_obs = orientation_observation_metadata_factory(
            orientation_representation=OrientationRepresentation.ROLL.value,
        )
        metadata = on_the_fly_action_metadata_factory(
            source_metadata=orientation_obs,
        )
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_orientation_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert (
            entry[ActionMetadataField.ORIENTATION_REPRESENTATION.value]
            == OrientationRepresentation.ROLL.value
        )

    def test_does_not_add_for_position_metadata(
        self,
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
    ):
        metadata = position_action_metadata_factory()
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_orientation_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert ActionMetadataField.ORIENTATION_REPRESENTATION.value not in entry

    def test_does_not_add_for_on_the_fly_position_metadata(
        self,
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        position_obs = position_observation_metadata_factory()
        metadata = on_the_fly_action_metadata_factory(source_metadata=position_obs)
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_orientation_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert ActionMetadataField.ORIENTATION_REPRESENTATION.value not in entry


@pytest.mark.unit
class TestAddGripperMetadata:
    @pytest.mark.parametrize(
        "gripper_type, binary_gripper_range",
        [
            (GripperType.BINARY.value, BinaryGripperRange.ZERO_ONE.value),
            (GripperType.BINARY.value, BinaryGripperRange.MINUS_ONE_ONE.value),
        ],
    )
    def test_adds_type_and_range_for_gripper_action_metadata(
        self,
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
        gripper_type: str,
        binary_gripper_range: str,
    ):
        metadata = gripper_action_metadata_factory(
            gripper_type=gripper_type,
            binary_gripper_range=binary_gripper_range,
        )
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_gripper_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert entry[ActionMetadataField.GRIPPER_TYPE.value] == gripper_type
        assert (
            entry[ActionMetadataField.BINARY_GRIPPER_RANGE.value]
            == binary_gripper_range
        )

    def test_adds_type_and_range_for_on_the_fly_gripper_metadata(
        self,
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
    ):
        gripper_obs = gripper_observation_metadata_factory(
            gripper_type=GripperType.BINARY.value,
            binary_gripper_range=BinaryGripperRange.MINUS_ONE_ONE.value,
        )
        metadata = on_the_fly_action_metadata_factory(source_metadata=gripper_obs)
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_gripper_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert entry[ActionMetadataField.GRIPPER_TYPE.value] == GripperType.BINARY.value
        assert (
            entry[ActionMetadataField.BINARY_GRIPPER_RANGE.value]
            == BinaryGripperRange.MINUS_ONE_ONE.value
        )

    def test_does_not_add_for_position_metadata(
        self,
        position_action_metadata_factory: Callable[..., PositionActionMetadata],
    ):
        metadata = position_action_metadata_factory()
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_gripper_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert ActionMetadataField.GRIPPER_TYPE.value not in entry
        assert ActionMetadataField.BINARY_GRIPPER_RANGE.value not in entry

    def test_does_not_add_for_on_the_fly_position_metadata(
        self,
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        position_obs = position_observation_metadata_factory()
        metadata = on_the_fly_action_metadata_factory(source_metadata=position_obs)
        entry: dict[str, str | int] = {}

        ActionPostprocessor._add_gripper_metadata(
            action_meta=metadata,
            entry=entry,
        )

        assert ActionMetadataField.GRIPPER_TYPE.value not in entry
        assert ActionMetadataField.BINARY_GRIPPER_RANGE.value not in entry
