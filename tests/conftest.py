"""Root test fixtures shared across the entire test suite."""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import versatil  # noqa: F401 — triggers dotenv loading and cache directory setup
from versatil.data.constants import (
    ActionComputationMethod,
    BinaryGripperRange,
    Cameras,
    CoordinateSystem,
    GripperType,
    OrientationRepresentation,
)
from versatil.data.metadata import (
    CameraMetadata,
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
    OrientationActionMetadata,
    OrientationObservationMetadata,
    PositionActionMetadata,
    PositionObservationMetadata,
)
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.metrics.base import LossOutput
from versatil.models.policy import Policy

MINIMUM_VRAM_GB = 8.0


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip ``@pytest.mark.requires_gpu`` tests when CUDA is unavailable."""
    if torch.cuda.is_available():
        return
    skip_requires_gpu = pytest.mark.skip(
        reason="requires CUDA; unavailable in this environment"
    )
    for item in items:
        if "requires_gpu" in item.keywords:
            item.add_marker(skip_requires_gpu)


def get_test_device() -> torch.device:
    """Return CUDA device if available with sufficient VRAM, else CPU."""
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if vram_gb > MINIMUM_VRAM_GB:
            return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed RNG for data generators. Fresh instance per test for isolation."""
    return np.random.default_rng(42)


@pytest.fixture
def device() -> torch.device:
    """Get available device (CUDA if available with >8GB VRAM, else CPU)."""
    return get_test_device()


@pytest.fixture
def batch_size() -> int:
    """Default batch size for tests."""
    return 2


@pytest.fixture
def temporal_length() -> int:
    """Default temporal sequence length."""
    return 2


@pytest.fixture
def image_size() -> tuple[int, int]:
    """Default image size (height, width)."""
    return 224, 224


@pytest.fixture
def loss_output_factory() -> Callable[..., LossOutput]:
    """Factory for LossOutput instances with configurable loss values."""

    def factory(
        total_loss_value: float = 1.0,
        component_losses: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        device: str = "cpu",
        requires_grad: bool = False,
    ) -> LossOutput:
        total = torch.tensor(
            total_loss_value, device=device, requires_grad=requires_grad
        )
        components = {}
        if component_losses is not None:
            for key, value in component_losses.items():
                components[key] = torch.tensor(value, device=device)
        return LossOutput(
            total_loss=total,
            component_losses=components,
            metadata=metadata if metadata is not None else {},
        )

    return factory


@pytest.fixture
def padding_mask_factory() -> Callable[..., torch.Tensor]:
    """Factory for padding masks (B, S) with True=padded."""

    def factory(
        batch_size: int = 2,
        sequence_length: int = 4,
        padded_from: int | None = None,
        padded_positions: list[list[int]] | None = None,
        mask_last_n: int | None = None,
    ) -> torch.Tensor:
        mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
        if padded_positions is not None:
            for batch_index, positions in enumerate(padded_positions):
                for position in positions:
                    mask[batch_index, position] = True
        elif mask_last_n is not None:
            mask[:, -mask_last_n:] = True
        elif padded_from is not None:
            mask[:, padded_from:] = True
        return mask

    return factory


@pytest.fixture
def action_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for action tensors (B, T, D)."""

    def factory(
        batch_size: int = 2,
        sequence_length: int = 4,
        action_dimension: int = 3,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, sequence_length, action_dimension)
        ).astype(np.float32)
        return torch.from_numpy(data)

    return factory


@pytest.fixture
def mock_policy_factory(rng: np.random.Generator) -> Callable[..., MagicMock]:
    def factory(
        prediction_horizon: int = 4,
        observation_horizon: int = 1,
        observations_metadata: dict | None = None,
        predict_action_return: dict[str, torch.Tensor] | None = None,
        named_parameters: list[tuple[str, torch.nn.Parameter]] | None = None,
    ) -> MagicMock:
        mock = MagicMock(spec=Policy)
        mock.prediction_horizon = prediction_horizon
        mock.observation_horizon = observation_horizon
        mock.observation_space = MagicMock()
        mock.observation_space.observations_metadata = (
            observations_metadata if observations_metadata is not None else {}
        )
        if predict_action_return is not None:
            mock.predict_action.return_value = predict_action_return

        if named_parameters is None:
            weight_data = torch.from_numpy(
                rng.standard_normal((8, 4)).astype(np.float32)
            )
            bias_data = torch.from_numpy(rng.standard_normal((8,)).astype(np.float32))
            weight = torch.nn.Parameter(weight_data)
            bias = torch.nn.Parameter(bias_data)
            named_parameters = [("layer.weight", weight), ("layer.bias", bias)]
        all_parameters = [parameter for _, parameter in named_parameters]
        mock.parameters.return_value = iter(all_parameters)
        mock.named_parameters.return_value = iter(named_parameters)
        mock_module = MagicMock()
        mock_module.parameters.return_value = iter(all_parameters)
        mock.modules.return_value = iter([mock_module])
        return mock

    return factory


@pytest.fixture
def position_observation_metadata_factory() -> Callable[
    ..., PositionObservationMetadata
]:
    def factory(
        dimension: int = 3,
        frame: str = CoordinateSystem.ROBOT_BASE.value,
        needs_normalization: bool = True,
        raw_data_column_keys: list[str] = None,
        dtype: str = "float32",
        slice_start: int = None,
        slice_end: int = None,
    ) -> PositionObservationMetadata:
        if raw_data_column_keys is None:
            raw_data_column_keys = ["x", "y", "z"][:dimension]
        return PositionObservationMetadata(
            raw_data_column_keys=raw_data_column_keys,
            dimension=dimension,
            dtype=dtype,
            needs_normalization=needs_normalization,
            frame=frame,
            slice_start=slice_start,
            slice_end=slice_end,
        )

    return factory


@pytest.fixture
def orientation_observation_metadata_factory() -> Callable[
    ..., OrientationObservationMetadata
]:
    def factory(
        dimension: int = 1,
        frame: str = CoordinateSystem.ROBOT_BASE.value,
        orientation_representation: str = OrientationRepresentation.ROLL.value,
        needs_normalization: bool = True,
        raw_data_column_keys: list[str] = None,
        dtype: str = "float32",
        slice_start: int = None,
        slice_end: int = None,
    ) -> OrientationObservationMetadata:
        if raw_data_column_keys is None:
            raw_data_column_keys = ["roll", "pitch", "yaw"][:dimension]
        return OrientationObservationMetadata(
            raw_data_column_keys=raw_data_column_keys,
            dimension=dimension,
            dtype=dtype,
            needs_normalization=needs_normalization,
            frame=frame,
            orientation_representation=orientation_representation,
            slice_start=slice_start,
            slice_end=slice_end,
        )

    return factory


@pytest.fixture
def gripper_observation_metadata_factory() -> Callable[..., GripperObservationMetadata]:
    def factory(
        gripper_type: str = GripperType.BINARY.value,
        binary_gripper_range: str = BinaryGripperRange.ZERO_ONE.value,
        dimension: int = 1,
        raw_data_column_keys: list[str] = None,
        dtype: str = None,
        needs_normalization: bool = None,
    ) -> GripperObservationMetadata:
        if raw_data_column_keys is None:
            raw_data_column_keys = ["gripper_state"]
        if gripper_type == GripperType.BINARY.value:
            if dtype is None:
                dtype = "int32"
            if needs_normalization is None:
                needs_normalization = False
        else:
            if dtype is None:
                dtype = "float32"
            if needs_normalization is None:
                needs_normalization = True
        return GripperObservationMetadata(
            raw_data_column_keys=raw_data_column_keys,
            dimension=dimension,
            dtype=dtype,
            needs_normalization=needs_normalization,
            gripper_type=gripper_type,
            binary_gripper_range=binary_gripper_range,
        )

    return factory


@pytest.fixture
def camera_metadata_factory() -> Callable[..., CameraMetadata]:
    def factory(
        camera_key: str = Cameras.LEFT.value,
        dtype: str = "uint8",
        channels: int = 3,
        image_width: int = 64,
        image_height: int = 64,
    ) -> CameraMetadata:
        return CameraMetadata(
            camera_key=camera_key,
            dtype=dtype,
            channels=channels,
            image_width=image_width,
            image_height=image_height,
        )

    return factory


@pytest.fixture
def on_the_fly_action_metadata_factory(
    position_observation_metadata_factory: Callable[..., PositionObservationMetadata],
) -> Callable[..., OnTheFlyActionMetadata]:
    def factory(
        source_metadata: PositionObservationMetadata
        | OrientationObservationMetadata
        | GripperObservationMetadata = None,
        computation_method: str = ActionComputationMethod.DELTA.value,
    ) -> OnTheFlyActionMetadata:
        if source_metadata is None:
            source_metadata = position_observation_metadata_factory()
        return OnTheFlyActionMetadata(
            source_metadata=source_metadata,
            computation_method=computation_method,
        )

    return factory


@pytest.fixture
def gripper_action_metadata_factory() -> Callable[..., GripperActionMetadata]:
    def factory(
        gripper_type: str = GripperType.BINARY.value,
        binary_gripper_range: str = BinaryGripperRange.ZERO_ONE.value,
        raw_data_column_keys: list[str] = None,
        storage_dimension: int = 1,
        prediction_dimension: int = 1,
        dtype: str = None,
        needs_normalization: bool = None,
    ) -> GripperActionMetadata:
        if raw_data_column_keys is None:
            raw_data_column_keys = ["gripper_action"]
        if gripper_type == GripperType.BINARY.value:
            if dtype is None:
                dtype = "int32"
            if needs_normalization is None:
                needs_normalization = False
        else:
            if dtype is None:
                dtype = "float32"
            if needs_normalization is None:
                needs_normalization = True
        return GripperActionMetadata(
            gripper_type=gripper_type,
            raw_data_column_keys=raw_data_column_keys,
            storage_dimension=storage_dimension,
            prediction_dimension=prediction_dimension,
            needs_normalization=needs_normalization,
            dtype=dtype,
            binary_gripper_range=binary_gripper_range,
        )

    return factory


@pytest.fixture
def action_space_factory() -> Callable[..., ActionSpace]:
    def factory(
        actions_metadata: dict = None,
        use_gripper_class_weights: bool = False,
        denoise_actions: bool = True,
        denoising_percentile: float = 15.0,
    ) -> ActionSpace:
        if actions_metadata is None:
            actions_metadata = {}
        return ActionSpace(
            actions_metadata=actions_metadata,
            use_gripper_class_weights=use_gripper_class_weights,
            denoise_actions=denoise_actions,
            denoising_percentile=denoising_percentile,
        )

    return factory


@pytest.fixture
def position_action_metadata_factory() -> Callable[..., PositionActionMetadata]:
    def factory(
        frame: str = CoordinateSystem.ROBOT_BASE.value,
        raw_data_column_keys: list[str] = None,
        storage_dimension: int = 3,
        prediction_dimension: int = 3,
        needs_normalization: bool = True,
        dtype: str = "float32",
        slice_start: int = None,
        slice_end: int = None,
    ) -> PositionActionMetadata:
        if raw_data_column_keys is None:
            raw_data_column_keys = ["x", "y", "z"][:prediction_dimension]
        return PositionActionMetadata(
            frame=frame,
            raw_data_column_keys=raw_data_column_keys,
            storage_dimension=storage_dimension,
            prediction_dimension=prediction_dimension,
            needs_normalization=needs_normalization,
            dtype=dtype,
            slice_start=slice_start,
            slice_end=slice_end,
        )

    return factory


@pytest.fixture
def orientation_action_metadata_factory() -> Callable[..., OrientationActionMetadata]:
    def factory(
        frame: str = CoordinateSystem.ROBOT_BASE.value,
        orientation_representation: str = OrientationRepresentation.ROLL.value,
        raw_data_column_keys: list[str] = None,
        storage_dimension: int = 1,
        prediction_dimension: int = 1,
        needs_normalization: bool = True,
        dtype: str = "float32",
        slice_start: int = None,
        slice_end: int = None,
    ) -> OrientationActionMetadata:
        if raw_data_column_keys is None:
            raw_data_column_keys = ["roll", "pitch", "yaw"][:prediction_dimension]
        return OrientationActionMetadata(
            frame=frame,
            orientation_representation=orientation_representation,
            raw_data_column_keys=raw_data_column_keys,
            storage_dimension=storage_dimension,
            prediction_dimension=prediction_dimension,
            needs_normalization=needs_normalization,
            dtype=dtype,
            slice_start=slice_start,
            slice_end=slice_end,
        )

    return factory


@pytest.fixture
def observation_space_factory() -> Callable[..., ObservationSpace]:
    def factory(
        observations_metadata: dict = None,
    ) -> ObservationSpace:
        if observations_metadata is None:
            observations_metadata = {}
        return ObservationSpace(observations_metadata=observations_metadata)

    return factory
