"""Shared fixtures for versatil.metrics.losses tests."""

import pytest

from versatil.data.constants import BinaryGripperRange, GripperType
from versatil.data.metadata import GripperActionMetadata


@pytest.fixture
def binary_gripper_metadata_factory():
    def factory(
        gripper_type: str = GripperType.BINARY.value,
        binary_gripper_range: str = BinaryGripperRange.ZERO_ONE.value,
    ) -> dict[str, GripperActionMetadata]:
        return {
            "gripper": GripperActionMetadata(
                gripper_type=gripper_type,
                raw_data_column_keys=["gripper_state"],
                storage_dimension=1,
                prediction_dimension=1,
                needs_normalization=False,
                dtype="int32",
                binary_gripper_range=binary_gripper_range,
            )
        }

    return factory


@pytest.fixture
def continuous_gripper_metadata_factory():
    def factory() -> dict[str, GripperActionMetadata]:
        return {
            "gripper": GripperActionMetadata(
                gripper_type=GripperType.CONTINUOUS.value,
                raw_data_column_keys=["gripper_state"],
                storage_dimension=1,
                prediction_dimension=1,
                needs_normalization=True,
                dtype="float32",
            )
        }

    return factory
