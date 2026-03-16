"""Action postprocessing for the inference pipeline."""

import numpy as np
import torch
from versatil_constants.shared import (
    ActionComponent,
    ActionMetadataField,
    BinaryGripperRange,
    GripperType,
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


class ActionPostprocessor:
    """Converts raw policy output tensors into structured action dicts for the server.

    Handles gripper postprocessing (sigmoid for binary), denoising thresholds,
    and action metadata construction from the action space.
    """

    def __init__(
        self,
        action_space: ActionSpace,
        denoising_thresholds: dict[str, float],
    ):
        """Initialize the action postprocessor.

        Args:
            action_space: Policy's action space with metadata per key.
            denoising_thresholds: Per-VersatIL-key denoising thresholds.
        """
        self.action_space = action_space
        self.denoising_thresholds = denoising_thresholds

    def format_action(
        self, action_dict: dict[str, torch.Tensor]
    ) -> dict[str, list[float]]:
        """Format action tensors into a structured dict keyed by ActionComponent.

        Args:
            action_dict: Dict mapping VersatIL action key to tensor.

        Returns:
            Dict mapping ActionComponent value to action values list.
        """
        components: dict[str, list[float]] = {}
        for key, metadata in self.action_space.actions_metadata.items():
            if not metadata.requires_prediction_head:
                continue
            value = action_dict[key].cpu().detach().float().numpy().flatten()
            if metadata.action_type == ActionComponent.GRIPPER.value:
                value = self._postprocess_gripper_action(
                    raw_value=value, action_meta=metadata
                )
            threshold = self.denoising_thresholds.get(key)
            if threshold is not None and np.linalg.norm(value) < threshold:
                value = np.zeros_like(value)
            components[metadata.action_type] = value.tolist()
        return components

    def build_action_metadata(self) -> dict[str, dict[str, str | int]]:
        """Build action metadata dict keyed by ActionComponent.

        Returns:
            Dict mapping ActionComponent value to metadata entry.
        """
        metadata: dict[str, dict[str, str | int]] = {}
        for _key, action_meta in self.action_space.actions_metadata.items():
            if not action_meta.requires_prediction_head:
                continue
            entry: dict[str, str | int] = {
                ActionMetadataField.DIMENSION.value: action_meta.prediction_dimension,
            }
            self._add_action_type_metadata(action_meta=action_meta, entry=entry)
            self._add_frame_metadata(action_meta=action_meta, entry=entry)
            self._add_orientation_metadata(action_meta=action_meta, entry=entry)
            self._add_gripper_metadata(action_meta=action_meta, entry=entry)
            metadata[action_meta.action_type] = entry
        return metadata

    @staticmethod
    def _postprocess_gripper_action(
        raw_value: np.ndarray, action_meta: ActionMetadata
    ) -> np.ndarray:
        """Apply metadata-driven gripper postprocessing.

        Binary grippers: sigmoid -> probability -> threshold to discrete value.

        Args:
            raw_value: Raw predicted gripper value.
            action_meta: Gripper action metadata.

        Returns:
            Postprocessed gripper value.
        """
        gripper_type = None
        binary_range = None
        if isinstance(action_meta, GripperActionMetadata):
            gripper_type = action_meta.gripper_type
            binary_range = action_meta.binary_gripper_range
        elif isinstance(action_meta, OnTheFlyActionMetadata) and isinstance(
            action_meta.source_metadata, GripperObservationMetadata
        ):
            gripper_type = action_meta.source_metadata.gripper_type
            binary_range = action_meta.source_metadata.binary_gripper_range

        if gripper_type == GripperType.BINARY.value:
            probability = 1.0 / (1.0 + np.exp(-raw_value[0]))
            if binary_range == BinaryGripperRange.ZERO_ONE.value:
                return np.array([float(probability > 0.5)])
            return np.array([float(probability > 0.5) * 2.0 - 1.0])
        return raw_value

    @staticmethod
    def _add_action_type_metadata(
        action_meta: ActionMetadata, entry: dict[str, str | int]
    ) -> None:
        """Add action computation method (delta or next_timestep) to metadata."""
        if isinstance(action_meta, OnTheFlyActionMetadata):
            entry[ActionMetadataField.ACTION_TYPE.value] = (
                action_meta.computation_method
            )

    @staticmethod
    def _add_frame_metadata(
        action_meta: ActionMetadata, entry: dict[str, str | int]
    ) -> None:
        """Add coordinate frame to metadata entry if available."""
        if isinstance(action_meta, (PositionActionMetadata, OrientationActionMetadata)):
            entry[ActionMetadataField.FRAME.value] = action_meta.frame
        elif isinstance(action_meta, OnTheFlyActionMetadata) and isinstance(
            action_meta.source_metadata,
            (PositionObservationMetadata, OrientationObservationMetadata),
        ):
            entry[ActionMetadataField.FRAME.value] = action_meta.source_metadata.frame

    @staticmethod
    def _add_orientation_metadata(
        action_meta: ActionMetadata, entry: dict[str, str | int]
    ) -> None:
        """Add orientation representation to metadata entry if available."""
        if isinstance(action_meta, OrientationActionMetadata):
            entry[ActionMetadataField.ORIENTATION_REPRESENTATION.value] = (
                action_meta.orientation_representation
            )
        elif isinstance(action_meta, OnTheFlyActionMetadata) and isinstance(
            action_meta.source_metadata, OrientationObservationMetadata
        ):
            entry[ActionMetadataField.ORIENTATION_REPRESENTATION.value] = (
                action_meta.source_metadata.orientation_representation
            )

    @staticmethod
    def _add_gripper_metadata(
        action_meta: ActionMetadata, entry: dict[str, str | int]
    ) -> None:
        """Add gripper type and range to metadata entry if available."""
        if isinstance(action_meta, GripperActionMetadata):
            entry[ActionMetadataField.GRIPPER_TYPE.value] = action_meta.gripper_type
            entry[ActionMetadataField.BINARY_GRIPPER_RANGE.value] = (
                action_meta.binary_gripper_range
            )
        elif isinstance(action_meta, OnTheFlyActionMetadata):
            if isinstance(action_meta.source_metadata, GripperObservationMetadata):
                entry[ActionMetadataField.GRIPPER_TYPE.value] = (
                    action_meta.source_metadata.gripper_type
                )
                entry[ActionMetadataField.BINARY_GRIPPER_RANGE.value] = (
                    action_meta.source_metadata.binary_gripper_range
                )
