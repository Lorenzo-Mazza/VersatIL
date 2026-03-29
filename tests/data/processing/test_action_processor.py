"""Tests for versatil.data.processing.action_processor module."""

import logging
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pytest

from versatil.data.constants import (
    ActionComputationMethod,
    GripperType,
    OrientationRepresentation,
    ProprioceptiveType,
)
from versatil.data.metadata import (
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
    OrientationObservationMetadata,
    PositionObservationMetadata,
)
from versatil.data.processing.action_processor import ActionProcessor
from versatil.data.task import ActionSpace


@pytest.fixture
def action_processor_factory(
    action_space_factory: Callable[..., ActionSpace],
) -> Callable[..., ActionProcessor]:
    """Factory for creating ActionProcessor instances."""

    def factory(
        actions_metadata: dict = None,
        denoise_actions: bool = False,
        denoising_percentile: float = 15.0,
    ) -> ActionProcessor:
        action_space = action_space_factory(
            actions_metadata=actions_metadata or {},
            denoise_actions=denoise_actions,
            denoising_percentile=denoising_percentile,
        )
        return ActionProcessor(action_space=action_space)

    return factory


class TestActionProcessorInitialization:
    def test_properties_stored_from_action_space(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory(
            denoise_actions=True,
            denoising_percentile=10.0,
        )

        assert processor.denoise_actions is True
        assert processor.denoising_percentile == 10.0
        assert processor.denoising_thresholds == {}
        assert processor._denoising_thresholds_computed is False

    def test_requires_denoising_setup_true_when_enabled_but_not_computed(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory(denoise_actions=True)

        assert processor.requires_denoising_setup is True

    def test_requires_denoising_setup_false_when_disabled(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory(denoise_actions=False)

        assert processor.requires_denoising_setup is False

    def test_requires_denoising_setup_false_after_thresholds_computed(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory(denoise_actions=True)
        processor._denoising_thresholds_computed = True

        assert processor.requires_denoising_setup is False


class TestComputePositionAction:
    def test_delta_method_returns_difference(self):
        current_position = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        )
        next_position = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)

        result = ActionProcessor.compute_position_action_from_observation(
            current_position=current_position,
            next_position=next_position,
            method=ActionComputationMethod.DELTA.value,
        )

        np.testing.assert_allclose(result[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(result[1], [0.0, 1.0, 0.0])

    def test_next_timestep_method_returns_next(self):
        current_position = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        next_position = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        result = ActionProcessor.compute_position_action_from_observation(
            current_position=current_position,
            next_position=next_position,
            method=ActionComputationMethod.NEXT_TIMESTEP.value,
        )

        np.testing.assert_allclose(result, next_position)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported position action"):
            ActionProcessor.compute_position_action_from_observation(
                current_position=np.zeros((1, 3)),
                next_position=np.zeros((1, 3)),
                method="velocity",
            )


class TestComputeOrientationAction:
    def test_next_timestep_returns_next_for_any_representation(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory()
        current_orientation = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        next_orientation = np.array([[0.4, 0.5, 0.6]], dtype=np.float32)

        result = processor.compute_orientation_action_from_observation(
            current_orientation=current_orientation,
            next_orientation=next_orientation,
            method=ActionComputationMethod.NEXT_TIMESTEP.value,
            representation=OrientationRepresentation.EULER.value,
        )

        np.testing.assert_allclose(result, next_orientation)

    def test_roll_delta_is_simple_subtraction(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory()
        current_roll = np.array([[0.0], [0.5]], dtype=np.float32)
        next_roll = np.array([[0.3], [1.0]], dtype=np.float32)

        result = processor.compute_orientation_action_from_observation(
            current_orientation=current_roll,
            next_orientation=next_roll,
            method=ActionComputationMethod.DELTA.value,
            representation=OrientationRepresentation.ROLL.value,
        )

        expected = np.array([[0.3], [0.5]], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_quaternion_delta_identity_when_same_orientation(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory()
        # (w, x, y, z) format — identity quaternion
        identity = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        result = processor.compute_orientation_action_from_observation(
            current_orientation=identity,
            next_orientation=identity,
            method=ActionComputationMethod.DELTA.value,
            representation=OrientationRepresentation.QUATERNION.value,
        )

        np.testing.assert_allclose(result, identity, atol=1e-6)

    def test_quaternion_delta_90_degrees_around_z(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        """Verifies (w,x,y,z) reorder logic produces correct relative rotation."""
        processor = action_processor_factory()
        # Identity → 90° around Z: relative rotation should be 90° around Z
        identity = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        # 90° around Z in (w,x,y,z): w=cos(45°), z=sin(45°)
        rotated_90z = np.array([[0.7071068, 0.0, 0.0, 0.7071068]], dtype=np.float64)

        result = processor.compute_orientation_action_from_observation(
            current_orientation=identity,
            next_orientation=rotated_90z,
            method=ActionComputationMethod.DELTA.value,
            representation=OrientationRepresentation.QUATERNION.value,
        )

        np.testing.assert_allclose(result, rotated_90z, atol=1e-5)

    def test_euler_delta_non_trivial_rotation(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory()
        # From identity (0,0,0) to small rotation — delta should equal the rotation
        current_euler = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        next_euler = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)

        result = processor.compute_orientation_action_from_observation(
            current_orientation=current_euler,
            next_orientation=next_euler,
            method=ActionComputationMethod.DELTA.value,
            representation=OrientationRepresentation.EULER.value,
        )

        # For small angles from identity, delta ≈ the target euler angles
        np.testing.assert_allclose(result, next_euler, atol=0.05)

    def test_euler_delta_zero_when_same_orientation(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory()
        euler = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

        result = processor.compute_orientation_action_from_observation(
            current_orientation=euler,
            next_orientation=euler,
            method=ActionComputationMethod.DELTA.value,
            representation=OrientationRepresentation.EULER.value,
        )

        np.testing.assert_allclose(result, np.zeros((1, 3)), atol=1e-5)

    def test_invalid_representation_raises(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory()
        with pytest.raises(ValueError, match="Unsupported orientation representation"):
            processor.compute_orientation_action_from_observation(
                current_orientation=np.zeros((1, 3)),
                next_orientation=np.zeros((1, 3)),
                method=ActionComputationMethod.DELTA.value,
                representation="rotation_matrix",
            )

    def test_invalid_method_raises(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory()
        with pytest.raises(ValueError, match="Unsupported method"):
            processor.compute_orientation_action_from_observation(
                current_orientation=np.zeros((1, 3)),
                next_orientation=np.zeros((1, 3)),
                method="velocity",
                representation=OrientationRepresentation.EULER.value,
            )


class TestComputeGripperAction:
    def test_binary_next_timestep_returns_next(self):
        current_gripper = np.array([[0]], dtype=np.float32)
        next_gripper = np.array([[1]], dtype=np.float32)

        result = ActionProcessor.compute_gripper_action_from_observation(
            current_gripper=current_gripper,
            next_gripper=next_gripper,
            method=ActionComputationMethod.NEXT_TIMESTEP.value,
            gripper_type=GripperType.BINARY.value,
        )

        np.testing.assert_array_equal(result, next_gripper)

    def test_binary_delta_raises(self):
        with pytest.raises(ValueError, match="Delta not supported for binary"):
            ActionProcessor.compute_gripper_action_from_observation(
                current_gripper=np.array([[0]], dtype=np.float32),
                next_gripper=np.array([[1]], dtype=np.float32),
                method=ActionComputationMethod.DELTA.value,
                gripper_type=GripperType.BINARY.value,
            )

    def test_continuous_delta_returns_difference(self):
        current_gripper = np.array([[0.2], [0.5]], dtype=np.float32)
        next_gripper = np.array([[0.7], [0.3]], dtype=np.float32)

        result = ActionProcessor.compute_gripper_action_from_observation(
            current_gripper=current_gripper,
            next_gripper=next_gripper,
            method=ActionComputationMethod.DELTA.value,
            gripper_type=GripperType.CONTINUOUS.value,
        )

        expected = np.array([[0.5], [-0.2]], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_continuous_next_timestep_returns_next(self):
        current_gripper = np.array([[0.2]], dtype=np.float32)
        next_gripper = np.array([[0.7]], dtype=np.float32)

        result = ActionProcessor.compute_gripper_action_from_observation(
            current_gripper=current_gripper,
            next_gripper=next_gripper,
            method=ActionComputationMethod.NEXT_TIMESTEP.value,
            gripper_type=GripperType.CONTINUOUS.value,
        )

        np.testing.assert_allclose(result, next_gripper)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported method"):
            ActionProcessor.compute_gripper_action_from_observation(
                current_gripper=np.zeros((1, 1)),
                next_gripper=np.zeros((1, 1)),
                method="velocity",
                gripper_type=GripperType.CONTINUOUS.value,
            )


class TestComputeActionOnTheFly:
    def test_dispatches_to_position(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        processor = action_processor_factory()
        metadata = on_the_fly_action_metadata_factory(
            source_metadata=position_observation_metadata_factory(dimension=3),
            computation_method=ActionComputationMethod.DELTA.value,
        )

        current = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        next_obs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

        result = processor.compute_action_on_the_fly(
            current_obs=current,
            next_obs=next_obs,
            metadata=metadata,
        )

        np.testing.assert_allclose(result, [[1.0, 2.0, 3.0]])

    def test_dispatches_to_orientation(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        processor = action_processor_factory()
        metadata = on_the_fly_action_metadata_factory(
            source_metadata=orientation_observation_metadata_factory(dimension=1),
            computation_method=ActionComputationMethod.DELTA.value,
        )

        current = np.array([[0.5]], dtype=np.float32)
        next_obs = np.array([[1.0]], dtype=np.float32)

        result = processor.compute_action_on_the_fly(
            current_obs=current,
            next_obs=next_obs,
            metadata=metadata,
        )

        np.testing.assert_allclose(result, [[0.5]])

    def test_dispatches_to_gripper(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        gripper_observation_metadata_factory: Callable[..., GripperObservationMetadata],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        processor = action_processor_factory()
        metadata = on_the_fly_action_metadata_factory(
            source_metadata=gripper_observation_metadata_factory(
                gripper_type=GripperType.BINARY.value,
            ),
            computation_method=ActionComputationMethod.NEXT_TIMESTEP.value,
        )

        current = np.array([[0]], dtype=np.float32)
        next_obs = np.array([[1]], dtype=np.float32)

        result = processor.compute_action_on_the_fly(
            current_obs=current,
            next_obs=next_obs,
            metadata=metadata,
        )

        np.testing.assert_array_equal(result, [[1]])

    def test_unsupported_action_type_raises(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        processor = action_processor_factory()
        metadata = on_the_fly_action_metadata_factory(
            source_metadata=position_observation_metadata_factory(),
        )
        # Override action_type to an unsupported value
        metadata.action_type = ProprioceptiveType.CUSTOM.value

        with pytest.raises(ValueError, match="Unsupported action type"):
            processor.compute_action_on_the_fly(
                current_obs=np.zeros((1, 3)),
                next_obs=np.zeros((1, 3)),
                metadata=metadata,
            )


class TestComputeDenosingThreshold:
    def test_computes_threshold_from_position_observations(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        processor = action_processor_factory(
            denoise_actions=True,
            denoising_percentile=50.0,
        )
        position_metadata = position_observation_metadata_factory(dimension=3)
        # 5 timesteps, all within one episode
        observation_data = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.001, 0.001, 0.001],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ],
            dtype=np.float32,
        )
        episode_ends = np.array([5])

        processor.compute_denoising_threshold(
            obs_data=observation_data,
            key="position",
            meta=position_metadata,
            episode_ends=episode_ends,
        )

        assert "position" in processor.denoising_thresholds
        assert processor.denoising_thresholds["position"] > 0
        assert processor._denoising_thresholds_computed is True
        assert "position" in processor.dataset_magnitudes
        assert (
            len(processor.dataset_magnitudes["position"]) == 4
        )  # 5 timesteps → 4 deltas

    def test_masks_cross_episode_transitions(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
    ):
        processor = action_processor_factory(
            denoise_actions=True,
            denoising_percentile=50.0,
        )
        position_metadata = position_observation_metadata_factory(dimension=1)
        # Two episodes: [0, 1, 2] and [100, 101, 102]
        # Cross-episode delta (2→100) should be excluded
        observation_data = np.array(
            [[0], [1], [2], [100], [101], [102]], dtype=np.float32
        )
        episode_ends = np.array([3, 6])

        processor.compute_denoising_threshold(
            obs_data=observation_data,
            key="position",
            meta=position_metadata,
            episode_ends=episode_ends,
        )

        # All valid deltas are 1.0, so the threshold should be close to 1.0
        # (without masking, the 98.0 cross-episode delta would skew the percentile)
        assert processor.denoising_thresholds["position"] == pytest.approx(1.0, abs=0.1)

    def test_skips_non_position_metadata(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
    ):
        processor = action_processor_factory(denoise_actions=True)
        orientation_metadata = orientation_observation_metadata_factory()

        processor.compute_denoising_threshold(
            obs_data=np.zeros((10, 1)),
            key="orientation",
            meta=orientation_metadata,
            episode_ends=np.array([10]),
        )

        assert "orientation" not in processor.denoising_thresholds


class TestApplyDeltaDenoising:
    def test_zeroes_movements_below_threshold(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory(denoise_actions=True)
        processor.denoising_thresholds = {"position": 0.5}
        processor._denoising_thresholds_computed = True

        current_values = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        next_values = np.array(
            [
                [0.001, 0.001, 0.001],  # norm ~0.0017, below threshold
                [1.0, 1.0, 1.0],  # norm ~1.73, above threshold
            ],
            dtype=np.float32,
        )

        denoised_next, _ = processor.apply_delta_denoising(
            next_values=next_values,
            current_values=current_values,
            key="position",
        )

        # Small movement should be snapped to current
        np.testing.assert_allclose(denoised_next[0], current_values[0])
        # Large movement should be unchanged
        np.testing.assert_allclose(denoised_next[1], next_values[1])

    def test_does_not_modify_original_arrays(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory(denoise_actions=True)
        processor.denoising_thresholds = {"position": 0.5}
        processor._denoising_thresholds_computed = True

        next_values = np.array([[0.001, 0.001, 0.001]], dtype=np.float32)
        original_copy = next_values.copy()

        processor.apply_delta_denoising(
            next_values=next_values,
            current_values=np.zeros((1, 3), dtype=np.float32),
            key="position",
        )

        np.testing.assert_array_equal(next_values, original_copy)

    def test_raises_if_thresholds_not_computed(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory(denoise_actions=True)

        with pytest.raises(RuntimeError, match="thresholds have not been computed"):
            processor.apply_delta_denoising(
                next_values=np.zeros((1, 3)),
                current_values=np.zeros((1, 3)),
                key="position",
            )

    def test_no_op_if_key_not_in_thresholds(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory(denoise_actions=True)
        processor.denoising_thresholds = {"other_key": 0.5}
        processor._denoising_thresholds_computed = True

        next_values = np.array([[0.001, 0.001, 0.001]], dtype=np.float32)

        denoised_next, _ = processor.apply_delta_denoising(
            next_values=next_values,
            current_values=np.zeros((1, 3), dtype=np.float32),
            key="position",
        )

        np.testing.assert_allclose(denoised_next, next_values)


class TestComputeSampleActions:
    def test_extracts_precomputed_actions_from_padded_data(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        gripper_metadata = gripper_action_metadata_factory()
        processor = action_processor_factory(
            actions_metadata={"gripper_action": gripper_metadata},
        )

        padded_data = {
            "gripper_action": np.array([[0], [1], [0], [1], [0]], dtype=np.float32),
        }

        action_data, action_meta = processor.compute_sample_actions(
            padded_data=padded_data,
            action_slice_start=1,
            action_slice_end=4,
        )

        np.testing.assert_array_equal(action_data["gripper_action"], [[1], [0], [1]])
        assert action_meta["gripper_action"] is gripper_metadata

    def test_computes_on_the_fly_position_actions(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        position_metadata = on_the_fly_action_metadata_factory(
            source_metadata=position_observation_metadata_factory(dimension=1),
            computation_method=ActionComputationMethod.DELTA.value,
        )
        processor = action_processor_factory(
            actions_metadata={"position": position_metadata},
        )

        # Observation sequence: [0, 1, 2, 3, 4]
        padded_data = {
            "position": np.array([[0], [1], [2], [3], [4]], dtype=np.float32),
        }

        # Slice [1:3] → current=[1,2], next=[2,3] → deltas=[1,1]
        action_data, _ = processor.compute_sample_actions(
            padded_data=padded_data,
            action_slice_start=1,
            action_slice_end=3,
        )

        np.testing.assert_allclose(action_data["position"], [[1], [1]])

    def test_applies_denoising_to_position_actions(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        position_metadata = on_the_fly_action_metadata_factory(
            source_metadata=position_observation_metadata_factory(dimension=1),
            computation_method=ActionComputationMethod.DELTA.value,
        )
        processor = action_processor_factory(
            actions_metadata={"position": position_metadata},
            denoise_actions=True,
        )
        processor.denoising_thresholds = {"position": 0.5}
        processor._denoising_thresholds_computed = True

        # Observation: [0, 0.001, 5, 6] — first delta is tiny (0.001), second is large (4.999)
        padded_data = {
            "position": np.array([[0], [0.001], [5], [6]], dtype=np.float32),
        }

        action_data, _ = processor.compute_sample_actions(
            padded_data=padded_data,
            action_slice_start=0,
            action_slice_end=2,
        )

        # First action: denoised → next snapped to current → delta=0
        assert action_data["position"][0, 0] == pytest.approx(0.0, abs=1e-6)
        # Second action: above threshold → delta preserved (~4.999)
        assert action_data["position"][1, 0] > 4.0

    def test_denoising_not_applied_to_orientation_actions(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        orientation_observation_metadata_factory: Callable[
            ..., OrientationObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
    ):
        orientation_metadata = on_the_fly_action_metadata_factory(
            source_metadata=orientation_observation_metadata_factory(dimension=1),
            computation_method=ActionComputationMethod.DELTA.value,
        )
        processor = action_processor_factory(
            actions_metadata={"orientation": orientation_metadata},
            denoise_actions=True,
        )
        processor.denoising_thresholds = {"orientation": 0.5}
        processor._denoising_thresholds_computed = True

        # Tiny delta (0.001) that would be zeroed if denoising were applied
        padded_data = {
            "orientation": np.array([[0.0], [0.001], [0.002]], dtype=np.float32),
        }

        action_data, _ = processor.compute_sample_actions(
            padded_data=padded_data,
            action_slice_start=0,
            action_slice_end=1,
        )

        # Orientation delta should NOT be denoised — raw delta=0.001 preserved
        assert action_data["orientation"][0, 0] == pytest.approx(0.001, abs=1e-6)

    def test_mixed_precomputed_and_on_the_fly(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        position_observation_metadata_factory: Callable[
            ..., PositionObservationMetadata
        ],
        on_the_fly_action_metadata_factory: Callable[..., OnTheFlyActionMetadata],
        gripper_action_metadata_factory: Callable[..., GripperActionMetadata],
    ):
        position_metadata = on_the_fly_action_metadata_factory(
            source_metadata=position_observation_metadata_factory(dimension=1),
            computation_method=ActionComputationMethod.DELTA.value,
        )
        gripper_metadata = gripper_action_metadata_factory()

        processor = action_processor_factory(
            actions_metadata={
                "position": position_metadata,
                "gripper": gripper_metadata,
            },
        )

        padded_data = {
            "position": np.array([[0], [1], [3], [6]], dtype=np.float32),
            "gripper": np.array([[0], [1], [1], [0]], dtype=np.float32),
        }

        action_data, action_meta = processor.compute_sample_actions(
            padded_data=padded_data,
            action_slice_start=0,
            action_slice_end=2,
        )

        np.testing.assert_allclose(action_data["position"], [[1], [2]])
        np.testing.assert_array_equal(action_data["gripper"], [[0], [1]])
        assert isinstance(action_meta["position"], OnTheFlyActionMetadata)
        assert isinstance(action_meta["gripper"], GripperActionMetadata)


class TestLogMovementDistribution:
    def test_logs_stats_for_each_key(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        caplog: pytest.LogCaptureFixture,
    ):
        processor = action_processor_factory(denoise_actions=True)
        processor.dataset_magnitudes = {
            "position": np.array([0.1, 0.5, 1.0, 2.0, 3.0]),
        }
        processor.denoising_thresholds = {}

        with caplog.at_level(logging.INFO):
            processor.log_movement_distribution()

        assert "position movement stats" in caplog.text
        assert "mean=" in caplog.text

    def test_logs_denoising_info_when_thresholds_exist(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
        caplog: pytest.LogCaptureFixture,
    ):
        processor = action_processor_factory(denoise_actions=True)
        processor.dataset_magnitudes = {
            "position": np.array([0.1, 0.5, 1.0, 2.0, 3.0]),
        }
        processor.denoising_thresholds = {"position": 0.5}

        with caplog.at_level(logging.INFO):
            processor.log_movement_distribution()

        assert "denoising" in caplog.text
        assert "threshold=" in caplog.text
        assert "movements zeroed" in caplog.text


class TestPlotActionMagnitudeDistribution:
    def test_returns_none_when_no_thresholds(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory()

        result = processor.plot_action_magnitude_distribution()

        assert result is None

    def test_returns_figure_when_thresholds_exist(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory(denoise_actions=True)
        processor.dataset_magnitudes = {
            "position": np.array([0.01, 0.1, 0.5, 1.0, 2.0, 3.0]),
        }
        processor.denoising_thresholds = {"position": 0.5}

        result = processor.plot_action_magnitude_distribution()

        assert isinstance(result, plt.Figure)
        plt.close(result)

    def test_handles_multiple_keys(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory(denoise_actions=True)
        processor.dataset_magnitudes = {
            "position_robot": np.array([0.1, 0.5, 1.0]),
            "position_camera": np.array([0.2, 0.6, 1.1]),
        }
        processor.denoising_thresholds = {
            "position_robot": 0.3,
            "position_camera": 0.4,
        }

        result = processor.plot_action_magnitude_distribution()

        assert isinstance(result, plt.Figure)
        axes = result.get_axes()
        assert len(axes) == 2
        plt.close(result)

    def test_handles_zero_threshold(
        self,
        action_processor_factory: Callable[..., ActionProcessor],
    ):
        processor = action_processor_factory(denoise_actions=True)
        processor.dataset_magnitudes = {
            "position": np.array([0.0, 0.0, 0.0]),
        }
        processor.denoising_thresholds = {"position": 0.0}

        result = processor.plot_action_magnitude_distribution()

        assert isinstance(result, plt.Figure)
        plt.close(result)
