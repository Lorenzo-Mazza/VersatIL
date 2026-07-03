"""Root test fixtures shared across the entire test suite."""

import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import (
    Gemma2Config,
    Idefics3Config,
    LlamaConfig,
    PaliGemmaConfig,
    SiglipVisionConfig,
)

import versatil  # noqa: F401 — triggers dotenv loading and cache directory setup
from versatil.data.constants import (
    ActionComputationMethod,
    BinaryGripperRange,
    Cameras,
    CoordinateSystem,
    GripperType,
    OrientationRepresentation,
    ProprioKey,
    SampleKey,
)
from versatil.data.metadata import (
    CameraMetadata,
    DepthCameraMetadata,
    GripperActionMetadata,
    GripperObservationMetadata,
    OnTheFlyActionMetadata,
    OrientationActionMetadata,
    OrientationObservationMetadata,
    PositionActionMetadata,
    PositionObservationMetadata,
    PrecomputedActionMetadata,
    RGBCameraMetadata,
)
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.raw.zarr_meta import DatasetMetadata
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import Tokenizer
from versatil.metrics.base import LossOutput
from versatil.models.decoding.action_heads.single_output import ActionHead
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.diffusion import Diffusion
from versatil.models.decoding.decoders.base import DecoderInput
from versatil.models.decoding.decoders.factory.smolvla import SmolVLADecoder
from versatil.models.decoding.generative_language_models.constants import (
    PRISMATIC_CONFIG_FILENAME,
    PRISMATIC_VISION_BACKBONES,
    PRISMATIC_VISION_IMAGE_SIZES,
    PaliGemmaModelType,
    PrismaticLLMBackboneType,
    PrismaticModelType,
    PrismaticVisionBackboneType,
    SmolVLMModelType,
)
from versatil.models.decoding.generative_language_models.vision_language.paligemma import (
    PaliGemmaVLM,
)
from versatil.models.decoding.generative_language_models.vision_language.prismatic import (
    PrismaticVLM,
)
from versatil.models.decoding.generative_language_models.vision_language.smolvlm import (
    SmolVLM,
)
from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    BatchNormHandling,
    EncoderOutputKeys,
    FlatBackboneType,
    PoolingMethod,
    SpatialBackboneType,
)
from versatil.models.encoding.encoders.rgb.flat import FlatRGBEncoder
from versatil.models.encoding.encoders.rgb.spatial import SpatialRGBEncoder
from versatil.models.encoding.pipeline import EncodingPipeline
from versatil.models.policy import Policy

MINIMUM_VRAM_GB = 8.0
MINIMUM_FREE_VRAM_GB = 2.0
EXECUTORCH_PACKAGE = "executorch"
REAL_MODEL_IMAGE_SIZE = 64
REAL_POLICY_ACTION_KEY = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
REAL_PIPELINE_ENCODER_NAME = "vision"
REAL_PIPELINE_FEATURE_KEY = "vision_rgb"
REAL_SMOLVLA_MODULE_NAME = "decoder.vlm_backbone"
REAL_VLM_MODULE_NAME = "decoder.vlm_backbone"
REAL_PRISMATIC_VISION_MODULE_NAME = "decoder.vlm_backbone.vision_encoders.0"
SMOLVLA_IMAGE_SIZE = 56
SMOLVLA_HIDDEN_DIMENSION = 32
SMOLVLA_EXPERT_WIDTH_MULTIPLIER = 0.5
SMOLVLA_EXPERT_HIDDEN_DIMENSION = int(
    SMOLVLA_HIDDEN_DIMENSION * SMOLVLA_EXPERT_WIDTH_MULTIPLIER
)
SMOLVLA_VOCAB_SIZE = 1000
SMOLVLA_TEXT_LENGTH = 4
PRISMATIC_TINY_VOCAB_SIZE = 128
VLM_TOKEN_ID_UPPER_BOUND = 128


@dataclass(frozen=True)
class RealExplainabilityPolicyCase:
    policy: Policy
    observation: dict[str, torch.Tensor]
    target_camera: str
    target_vision_module_names: list[str]
    expected_camera: str
    expected_vision_module_names: list[str]


class TinyContinuousAlgorithm(DecodingAlgorithm):
    def forward(
        self,
        network: nn.Module,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return network(features=features, actions=actions)

    def predict(
        self,
        network: nn.Module,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return network(features=features, actions=None)


class TinyContinuousDecoder(nn.Module):
    requires_tokenized_actions = False

    def __init__(
        self,
        input_keys: list[str],
        prediction_horizon: int,
        prediction_key: str = REAL_POLICY_ACTION_KEY,
    ) -> None:
        super().__init__()
        self.decoder_input = DecoderInput(keys=input_keys)
        self.prediction_horizon = prediction_horizon
        self.prediction_key = prediction_key

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer = normalizer

    def set_tokenizer(self, tokenizer: Tokenizer | None) -> None:
        self.tokenizer = tokenizer

    def get_prediction_output_keys(self) -> list[str]:
        return [self.prediction_key]

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        score = self._score_features(features=features)
        prediction = score[:, None, None].repeat(1, self.prediction_horizon, 3)
        return {self.prediction_key: prediction}

    def _score_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        scores = []
        for key in self.decoder_input.keys:
            tensor = features[key].float()
            scores.append(tensor.flatten(start_dim=1).mean(dim=1))
        return torch.stack(scores, dim=0).mean(dim=0)


class TinyVLMContinuousDecoder(nn.Module):
    requires_tokenized_actions = False

    def __init__(
        self,
        vlm_backbone: nn.Module,
        prediction_horizon: int,
        prediction_key: str = REAL_POLICY_ACTION_KEY,
    ) -> None:
        super().__init__()
        self.vlm_backbone = vlm_backbone
        self.decoder_input = DecoderInput(keys=vlm_backbone.input_specification.keys)
        self.prediction_horizon = prediction_horizon
        self.prediction_key = prediction_key

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer = normalizer

    def set_tokenizer(self, tokenizer: Tokenizer | None) -> None:
        self.tokenizer = tokenizer

    def get_prediction_output_keys(self) -> list[str]:
        return [self.prediction_key]

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        del actions
        vlm_features = self.vlm_backbone(inputs=features)
        fused_features = vlm_features[
            EncoderOutputKeys.FUSED_RGB_LANGUAGE.value
        ].float()
        score = fused_features.flatten(start_dim=1).mean(dim=1)
        prediction = score[:, None, None].repeat(1, self.prediction_horizon, 3)
        return {self.prediction_key: prediction}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip optional-capability tests when the current environment cannot run them."""
    skip_requires_gpu = None
    skip_requires_executorch = None
    if not _cuda_has_sufficient_memory():
        skip_requires_gpu = pytest.mark.skip(
            reason="requires CUDA with sufficient free memory; unavailable in this environment"
        )
    if not _executorch_available():
        skip_requires_executorch = pytest.mark.skip(
            reason="requires ExecuTorch with XNNPACK support; unavailable in this environment"
        )
    for item in items:
        if skip_requires_gpu is not None and "requires_gpu" in item.keywords:
            item.add_marker(skip_requires_gpu)
        if (
            skip_requires_executorch is not None
            and "requires_executorch" in item.keywords
        ):
            item.add_marker(skip_requires_executorch)


def _cuda_has_sufficient_memory() -> bool:
    """Return whether CUDA is available with enough total and free VRAM for tests."""
    if not torch.cuda.is_available():
        return False
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    try:
        free_memory_bytes, _ = torch.cuda.mem_get_info(0)
    except RuntimeError:
        return False
    free_vram_gb = free_memory_bytes / 1e9
    return total_vram_gb > MINIMUM_VRAM_GB and free_vram_gb > MINIMUM_FREE_VRAM_GB


def _executorch_available() -> bool:
    """Return whether the optional ExecuTorch package is installed."""
    return importlib.util.find_spec(EXECUTORCH_PACKAGE) is not None


def get_test_device() -> torch.device:
    """Return CUDA device if available with sufficient VRAM, else CPU."""
    if _cuda_has_sufficient_memory():
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
        dtype: str | None = None,
        channels: int = 3,
        image_width: int = 64,
        image_height: int = 64,
    ) -> CameraMetadata:
        if camera_key == Cameras.DEPTH.value:
            if channels != 1:
                raise ValueError(
                    f"Depth camera metadata uses one channel, got {channels}"
                )
            if dtype is None:
                dtype = "float32"
            return DepthCameraMetadata(
                camera_key=camera_key,
                dtype=dtype,
                image_width=image_width,
                image_height=image_height,
            )
        if dtype is None:
            dtype = "uint8"
        if channels != 3:
            return CameraMetadata(
                camera_key=camera_key,
                dtype=dtype,
                channels=channels,
                image_width=image_width,
                image_height=image_height,
            )
        return RGBCameraMetadata(
            camera_key=camera_key,
            dtype=dtype,
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
def precomputed_action_metadata_factory() -> Callable[..., PrecomputedActionMetadata]:
    def factory(
        raw_data_column_keys: list[str] = None,
        storage_dimension: int = 7,
        prediction_dimension: int = 3,
        is_numerical: bool = True,
        needs_normalization: bool = True,
        dtype: str = "float32",
        slice_start: int = None,
        slice_end: int = None,
        requires_prediction_head: bool = True,
    ) -> PrecomputedActionMetadata:
        if raw_data_column_keys is None:
            raw_data_column_keys = ["action_col"]
        return PrecomputedActionMetadata(
            raw_data_column_keys=raw_data_column_keys,
            storage_dimension=storage_dimension,
            prediction_dimension=prediction_dimension,
            is_numerical=is_numerical,
            needs_normalization=needs_normalization,
            dtype=dtype,
            slice_start=slice_start,
            slice_end=slice_end,
            requires_prediction_head=requires_prediction_head,
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
        computation_method: str = None,
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
            computation_method=computation_method,
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
def dataset_metadata_factory() -> Callable[..., DatasetMetadata]:
    def factory(
        observations: dict = None,
        precomputed_actions: dict = None,
    ) -> DatasetMetadata:
        if observations is None:
            observations = {}
        if precomputed_actions is None:
            precomputed_actions = {}
        return DatasetMetadata(
            observations=observations,
            precomputed_actions=precomputed_actions,
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


@pytest.fixture
def real_explainability_policy_case_factory(
    tmp_path: Path,
    rng: np.random.Generator,
    camera_metadata_factory: Callable[..., CameraMetadata],
    precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
    action_space_factory: Callable[..., ActionSpace],
    observation_space_factory: Callable[..., ObservationSpace],
) -> Callable[..., RealExplainabilityPolicyCase]:
    def factory(
        case_name: str,
        batch_size: int = 2,
        temporal_length: int = 1,
        image_height: int = REAL_MODEL_IMAGE_SIZE,
        image_width: int = REAL_MODEL_IMAGE_SIZE,
    ) -> RealExplainabilityPolicyCase:
        torch.manual_seed(17)
        action_space = _make_real_action_space(
            precomputed_action_metadata_factory=precomputed_action_metadata_factory,
            action_space_factory=action_space_factory,
        )
        camera_keys = _camera_keys_for_real_case(case_name=case_name)
        observation_space = _make_real_observation_space(
            camera_keys=camera_keys,
            image_height=image_height,
            image_width=image_width,
            camera_metadata_factory=camera_metadata_factory,
            observation_space_factory=observation_space_factory,
        )
        observation = _make_real_camera_observation(
            rng=rng,
            camera_keys=camera_keys,
            batch_size=batch_size,
            temporal_length=temporal_length,
            image_height=image_height,
            image_width=image_width,
        )
        if case_name in _VLM_REAL_CASES:
            observation.update(
                _make_real_tokenized_observation(
                    rng=rng,
                    batch_size=batch_size,
                    temporal_length=temporal_length,
                )
            )

        if case_name in _SPATIAL_REAL_BACKBONES:
            encoder = _make_real_spatial_encoder(
                backbone=_SPATIAL_REAL_BACKBONES[case_name],
            )
            policy = _make_pipeline_real_policy(
                encoder=encoder,
                observation_space=observation_space,
                action_space=action_space,
                temporal_length=temporal_length,
            )
            return RealExplainabilityPolicyCase(
                policy=policy,
                observation=observation,
                target_camera=Cameras.LEFT.value,
                target_vision_module_names=[REAL_PIPELINE_ENCODER_NAME],
                expected_camera=Cameras.LEFT.value,
                expected_vision_module_names=[REAL_PIPELINE_ENCODER_NAME],
            )

        if case_name in _FLAT_REAL_BACKBONES:
            encoder = _make_real_flat_encoder(
                input_keys=[Cameras.LEFT.value],
                backbone=_FLAT_REAL_BACKBONES[case_name],
                image_height=image_height,
                image_width=image_width,
            )
            policy = _make_pipeline_real_policy(
                encoder=encoder,
                observation_space=observation_space,
                action_space=action_space,
                temporal_length=temporal_length,
            )
            return RealExplainabilityPolicyCase(
                policy=policy,
                observation=observation,
                target_camera=Cameras.LEFT.value,
                target_vision_module_names=[REAL_PIPELINE_ENCODER_NAME],
                expected_camera=Cameras.LEFT.value,
                expected_vision_module_names=[REAL_PIPELINE_ENCODER_NAME],
            )

        if case_name == "smolvla":
            policy = _make_smolvla_real_policy(
                action_space=action_space,
                observation_space=observation_space,
                temporal_length=temporal_length,
            )
            return RealExplainabilityPolicyCase(
                policy=policy,
                observation=observation,
                target_camera=Cameras.LEFT.value,
                target_vision_module_names=[REAL_SMOLVLA_MODULE_NAME],
                expected_camera=Cameras.LEFT.value,
                expected_vision_module_names=[REAL_SMOLVLA_MODULE_NAME],
            )

        if case_name == "paligemma_vlm":
            policy = _make_vlm_real_policy(
                vlm_backbone=_make_real_paligemma_backbone(),
                action_space=action_space,
                observation_space=observation_space,
                temporal_length=temporal_length,
            )
            return RealExplainabilityPolicyCase(
                policy=policy,
                observation=observation,
                target_camera=Cameras.RIGHT.value,
                target_vision_module_names=[REAL_VLM_MODULE_NAME],
                expected_camera=Cameras.RIGHT.value,
                expected_vision_module_names=[REAL_VLM_MODULE_NAME],
            )

        if case_name == "prismatic_vlm":
            policy = _make_vlm_real_policy(
                vlm_backbone=_make_real_prismatic_backbone(root=tmp_path),
                action_space=action_space,
                observation_space=observation_space,
                temporal_length=temporal_length,
            )
            return RealExplainabilityPolicyCase(
                policy=policy,
                observation=observation,
                target_camera=Cameras.RIGHT.value,
                target_vision_module_names=[REAL_PRISMATIC_VISION_MODULE_NAME],
                expected_camera=Cameras.RIGHT.value,
                expected_vision_module_names=[REAL_PRISMATIC_VISION_MODULE_NAME],
            )

        valid_cases = [
            *_SPATIAL_REAL_BACKBONES,
            *_FLAT_REAL_BACKBONES,
            *_VLM_REAL_CASES,
        ]
        raise ValueError(
            f"Unknown real explainability policy case: {case_name}. "
            f"Valid cases: {valid_cases}"
        )

    return factory


_SPATIAL_REAL_BACKBONES = {
    "spatial_resnet18": SpatialBackboneType.RESNET18.value,
    "spatial_efficientnet_b0": SpatialBackboneType.EFFICIENTNET_B0.value,
    "spatial_convnext_nano": SpatialBackboneType.CONVNEXT_NANO.value,
    "spatial_tiny_vit": SpatialBackboneType.TINY_VIT_21M.value,
}
_FLAT_REAL_BACKBONES = {
    "flat_deit_tiny": FlatBackboneType.DEIT_TINY.value,
    "flat_deit_small": FlatBackboneType.DEIT_SMALL.value,
}
_VLM_REAL_CASES = {"smolvla", "paligemma_vlm", "prismatic_vlm"}


def _camera_keys_for_real_case(case_name: str) -> list[str]:
    if case_name in _VLM_REAL_CASES:
        return [Cameras.LEFT.value, Cameras.RIGHT.value]
    return [Cameras.LEFT.value]


def _make_real_action_space(
    precomputed_action_metadata_factory: Callable[..., PrecomputedActionMetadata],
    action_space_factory: Callable[..., ActionSpace],
) -> ActionSpace:
    action_metadata = precomputed_action_metadata_factory(
        raw_data_column_keys=["x", "y", "z"],
        storage_dimension=3,
        prediction_dimension=3,
        slice_start=None,
        slice_end=None,
    )
    return action_space_factory(
        actions_metadata={REAL_POLICY_ACTION_KEY: action_metadata},
    )


def _make_real_observation_space(
    camera_keys: list[str],
    image_height: int,
    image_width: int,
    camera_metadata_factory: Callable[..., CameraMetadata],
    observation_space_factory: Callable[..., ObservationSpace],
) -> ObservationSpace:
    observations_metadata = {
        camera_key: camera_metadata_factory(
            camera_key=camera_key,
            image_height=image_height,
            image_width=image_width,
        )
        for camera_key in camera_keys
    }
    return observation_space_factory(observations_metadata=observations_metadata)


def _make_real_camera_observation(
    rng: np.random.Generator,
    camera_keys: list[str],
    batch_size: int,
    temporal_length: int,
    image_height: int,
    image_width: int,
) -> dict[str, torch.Tensor]:
    observation = {}
    for camera_index, camera_key in enumerate(camera_keys):
        image_batch = rng.random(
            (batch_size, temporal_length, 3, image_height, image_width),
            dtype=np.float32,
        )
        image_batch = image_batch + np.float32(camera_index) * np.float32(0.05)
        observation[camera_key] = torch.from_numpy(np.clip(image_batch, 0.0, 1.0))
    return observation


def _make_real_tokenized_observation(
    rng: np.random.Generator,
    batch_size: int,
    temporal_length: int,
) -> dict[str, torch.Tensor]:
    token_shape = (batch_size, temporal_length, SMOLVLA_TEXT_LENGTH)
    token_ids = rng.integers(
        low=0,
        high=VLM_TOKEN_ID_UPPER_BOUND,
        size=token_shape,
        dtype=np.int64,
    )
    return {
        SampleKey.TOKENIZED_OBSERVATIONS.value: torch.from_numpy(token_ids),
        SampleKey.IS_PAD_OBSERVATION.value: torch.zeros(
            token_shape,
            dtype=torch.bool,
        ),
    }


def _make_real_spatial_encoder(backbone: str) -> SpatialRGBEncoder:
    return SpatialRGBEncoder(
        input_keys=Cameras.LEFT.value,
        backbone=backbone,
        pooling_method=PoolingMethod.NONE.value,
        batch_norm_handling=BatchNormHandling.DEFAULT.value,
        pretrained=False,
        frozen=False,
    )


def _make_real_flat_encoder(
    input_keys: list[str],
    backbone: str,
    image_height: int,
    image_width: int,
) -> FlatRGBEncoder:
    return FlatRGBEncoder(
        input_keys=input_keys,
        backbone=backbone,
        pooling_method=PoolingMethod.NONE.value,
        pretrained=False,
        frozen=False,
        image_size=(image_height, image_width),
        intermediate_layer_index=-2,
    )


def _make_tiny_smolvlm_config() -> Idefics3Config:
    text_config = LlamaConfig(
        num_hidden_layers=1,
        hidden_size=SMOLVLA_HIDDEN_DIMENSION,
        intermediate_size=SMOLVLA_HIDDEN_DIMENSION * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=SMOLVLA_HIDDEN_DIMENSION // 2,
        vocab_size=SMOLVLA_VOCAB_SIZE,
        max_position_embeddings=64,
    )
    vision_config = SiglipVisionConfig(
        hidden_size=SMOLVLA_HIDDEN_DIMENSION,
        intermediate_size=SMOLVLA_HIDDEN_DIMENSION * 2,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=SMOLVLA_IMAGE_SIZE,
        patch_size=14,
    )
    return Idefics3Config(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        scale_factor=4,
    )


def _make_tiny_paligemma_config() -> PaliGemmaConfig:
    text_config = Gemma2Config(
        num_hidden_layers=1,
        hidden_size=SMOLVLA_HIDDEN_DIMENSION,
        intermediate_size=SMOLVLA_HIDDEN_DIMENSION * 2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=SMOLVLA_HIDDEN_DIMENSION // 2,
        vocab_size=SMOLVLA_VOCAB_SIZE,
        max_position_embeddings=64,
    )
    vision_config = SiglipVisionConfig(
        hidden_size=SMOLVLA_HIDDEN_DIMENSION,
        intermediate_size=SMOLVLA_HIDDEN_DIMENSION * 2,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=SMOLVLA_IMAGE_SIZE,
        patch_size=14,
    )
    config = PaliGemmaConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        projection_dim=SMOLVLA_HIDDEN_DIMENSION,
    )
    config.vision_config.num_image_tokens = 16
    return config


def _make_real_smolvlm_backbone() -> SmolVLM:
    tiny_config = _make_tiny_smolvlm_config()
    with patch(
        "versatil.models.decoding.generative_language_models.vision_language"
        ".huggingface.AutoConfig.from_pretrained",
        return_value=tiny_config,
    ):
        backbone = SmolVLM(
            input_keys=[
                Cameras.LEFT.value,
                Cameras.RIGHT.value,
                SampleKey.TOKENIZED_OBSERVATIONS.value,
            ],
            pretrained=False,
            frozen=False,
            model_name=SmolVLMModelType.SMOLVLM_256M.value,
            max_text_length=SMOLVLA_TEXT_LENGTH,
        )
    backbone.vlm = backbone.vlm.float()
    return backbone


def _make_real_paligemma_backbone() -> PaliGemmaVLM:
    tiny_config = _make_tiny_paligemma_config()
    with patch(
        "versatil.models.decoding.generative_language_models.vision_language"
        ".huggingface.AutoConfig.from_pretrained",
        return_value=tiny_config,
    ):
        backbone = PaliGemmaVLM(
            input_keys=[
                Cameras.LEFT.value,
                Cameras.RIGHT.value,
                SampleKey.TOKENIZED_OBSERVATIONS.value,
            ],
            pretrained=False,
            frozen=False,
            model_name=PaliGemmaModelType.PALIGEMMA2_3B_224.value,
            max_text_length=SMOLVLA_TEXT_LENGTH,
        )
    backbone.vlm = backbone.vlm.float()
    return backbone


def _make_tiny_prismatic_config_dir(root: Path) -> Path:
    config_dir = root / "prismatic_tiny"
    config_dir.mkdir(exist_ok=True)
    config_path = config_dir / PRISMATIC_CONFIG_FILENAME
    config_path.write_text(
        json.dumps(
            {
                "model": {
                    "model_id": PrismaticModelType.PRISM_DINOSIGLIP_224PX_7B.value,
                    "vision_backbone_id": (
                        PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_224PX.value
                    ),
                    "llm_backbone_id": PrismaticLLMBackboneType.LLAMA2_7B_PURE.value,
                    "arch_specifier": "linear",
                    "image_resize_strategy": "resize-naive",
                    "llm_max_length": SMOLVLA_TEXT_LENGTH,
                }
            }
        )
    )
    return config_dir


def _make_real_prismatic_backbone(root: Path) -> PrismaticVLM:
    config_dir = _make_tiny_prismatic_config_dir(root=root)
    text_config = LlamaConfig(
        vocab_size=PRISMATIC_TINY_VOCAB_SIZE,
        hidden_size=SMOLVLA_HIDDEN_DIMENSION,
        intermediate_size=SMOLVLA_HIDDEN_DIMENSION * 2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    with (
        patch.dict(
            PRISMATIC_VISION_BACKBONES,
            {
                PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_224PX: (
                    FlatBackboneType.DEIT_TINY,
                    FlatBackboneType.DEIT_TINY,
                )
            },
        ),
        patch.dict(
            PRISMATIC_VISION_IMAGE_SIZES,
            {PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_224PX: 32},
        ),
        patch(
            "versatil.models.decoding.generative_language_models.vision_language"
            ".prismatic.AutoConfig.from_pretrained",
            autospec=True,
            return_value=text_config,
        ),
    ):
        backbone = PrismaticVLM(
            input_keys=[
                Cameras.LEFT.value,
                Cameras.RIGHT.value,
                SampleKey.TOKENIZED_OBSERVATIONS.value,
            ],
            pretrained=False,
            frozen=False,
            model_name=str(config_dir),
            repository_id="test/prismatic",
            attention_type=AttentionImplementation.SDPA.value,
            model_dtype=None,
            max_text_length=SMOLVLA_TEXT_LENGTH,
            lora_config=None,
            gradient_checkpointing=False,
        )
    backbone.eval()
    return backbone


def _make_smolvla_real_policy(
    action_space: ActionSpace,
    observation_space: ObservationSpace,
    temporal_length: int,
) -> Policy:
    decoder = SmolVLADecoder(
        input_keys=[],
        action_space=action_space,
        action_heads={
            "joint_action": ActionHead(input_dimension=SMOLVLA_EXPERT_HIDDEN_DIMENSION)
        },
        observation_space=observation_space,
        observation_horizon=temporal_length,
        prediction_horizon=1,
        device="cpu",
        vlm_backbone=_make_real_smolvlm_backbone(),
        expert_width_multiplier=SMOLVLA_EXPERT_WIDTH_MULTIPLIER,
        num_expert_layers=1,
        num_vlm_layers=1,
        self_attention_every_n_layers=0,
        freeze_vlm=False,
        dropout=0.0,
    )
    policy = Policy(
        encoding_pipeline=EncodingPipeline(
            encoders={},
            observation_space=observation_space,
        ),
        algorithm=Diffusion(num_inference_steps=1),
        decoder=decoder,
        observation_space=observation_space,
        action_space=action_space,
        prediction_horizon=1,
        observation_horizon=temporal_length,
        loss=MagicMock(),
        device="cpu",
    )
    policy.eval()
    return policy


def _make_vlm_real_policy(
    vlm_backbone: nn.Module,
    action_space: ActionSpace,
    observation_space: ObservationSpace,
    temporal_length: int,
) -> Policy:
    decoder = TinyVLMContinuousDecoder(
        vlm_backbone=vlm_backbone,
        prediction_horizon=1,
    )
    return _make_real_policy(
        encoding_pipeline=EncodingPipeline(
            encoders={},
            observation_space=observation_space,
        ),
        decoder=decoder,
        observation_space=observation_space,
        action_space=action_space,
        temporal_length=temporal_length,
    )


def _make_pipeline_real_policy(
    encoder: nn.Module,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    temporal_length: int,
) -> Policy:
    encoding_pipeline = EncodingPipeline(
        encoders={REAL_PIPELINE_ENCODER_NAME: encoder},
        observation_space=observation_space,
    )
    decoder = TinyContinuousDecoder(
        input_keys=[REAL_PIPELINE_FEATURE_KEY],
        prediction_horizon=1,
    )
    return _make_real_policy(
        encoding_pipeline=encoding_pipeline,
        decoder=decoder,
        observation_space=observation_space,
        action_space=action_space,
        temporal_length=temporal_length,
    )


def _make_real_policy(
    encoding_pipeline: EncodingPipeline,
    decoder: nn.Module,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    temporal_length: int,
) -> Policy:
    policy = Policy(
        encoding_pipeline=encoding_pipeline,
        algorithm=TinyContinuousAlgorithm(),
        decoder=decoder,
        observation_space=observation_space,
        action_space=action_space,
        prediction_horizon=1,
        observation_horizon=temporal_length,
        loss=MagicMock(),
        device="cpu",
    )
    policy.eval()
    return policy
