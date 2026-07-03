"""Fixtures for versatil.explainability.attribution tests."""

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from versatil.data.constants import Cameras, ProprioKey
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
    VisionExplanationTarget,
)
from versatil.models.input_specification import InputSpecification


class _SpatialEncodingPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 2, kernel_size=1)
        self.encoders = {"rgb": self}
        self.conditional_encoders = {}
        self.input_specification = InputSpecification(keys=[Cameras.LEFT.value])

    def get_explainability_targets(self) -> list[VisionExplanationTarget]:
        return [
            VisionExplanationTarget(
                layer=self.encoder,
                target_kind=ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
                activation_layout=ActivationLayout.NCHW.value,
            )
        ]

    def is_vision_encoder(self) -> bool:
        return True

    def forward(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image = observation[Cameras.LEFT.value]
        batch_size, temporal_length = image.shape[:2]
        flattened = image.reshape(batch_size * temporal_length, *image.shape[2:])
        features = self.encoder(flattened)
        return {ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: features.mean(dim=1)}


class _MultiCameraSpatialEncodingPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        nn.init.ones_(self.encoder.weight)
        self.encoders = {"rgb": self}
        self.conditional_encoders = {}
        self.input_specification = InputSpecification(
            keys=[Cameras.LEFT.value, Cameras.RIGHT.value]
        )
        self.camera_keys = [Cameras.LEFT.value, Cameras.RIGHT.value]
        self.is_multi_camera = True

    def get_explainability_targets(self) -> list[VisionExplanationTarget]:
        return [
            VisionExplanationTarget(
                layer=self.encoder,
                target_kind=ExplanationTargetKind.SPATIAL_FEATURE_MAP.value,
                activation_layout=ActivationLayout.NCHW.value,
            )
        ]

    def forward(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features_by_camera = []
        for camera_key in self.camera_keys:
            image = observation[camera_key]
            batch_size, temporal_length = image.shape[:2]
            flattened = image.reshape(batch_size * temporal_length, *image.shape[2:])
            features_by_camera.append(self.encoder(flattened).squeeze(1))
        features = torch.stack(features_by_camera, dim=0).sum(dim=0)
        return {ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: features}


class _TokenLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(2, 3))
        prefix = pooled.mean(dim=1, keepdim=True).unsqueeze(-1).expand(-1, 1, 3)
        patch_tokens = torch.stack([pooled, pooled * 2, pooled * 3, pooled * 4], dim=1)
        return self.projection(torch.cat([prefix, patch_tokens], dim=1))


class _TokenEncodingPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _TokenLayer()
        self.encoders = {"vit": self}
        self.conditional_encoders = {}
        self.input_specification = InputSpecification(keys=[Cameras.LEFT.value])

    def get_explainability_targets(self) -> list[VisionExplanationTarget]:
        return [
            VisionExplanationTarget(
                layer=self.encoder,
                target_kind=ExplanationTargetKind.TOKEN_SEQUENCE.value,
                activation_layout=ActivationLayout.NLC.value,
                prefix_token_count=1,
                patch_grid=(2, 2),
            )
        ]

    def is_vision_encoder(self) -> bool:
        return True

    def forward(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image = observation[Cameras.LEFT.value]
        batch_size, temporal_length = image.shape[:2]
        flattened = image.reshape(batch_size * temporal_length, *image.shape[2:])
        tokens = self.encoder(flattened)
        return {ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: tokens.mean(dim=-1)}


class _ActionDecoder(nn.Module):
    @property
    def encoder_cache_enabled(self) -> bool:
        return False

    def enable_encoder_cache(self) -> None:
        pass

    def disable_encoder_cache(self) -> None:
        pass

    def set_encoder_cache_suppressed(self, suppressed: bool) -> None:
        pass

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        prediction = features[ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value]
        return {ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: prediction}


class _Algorithm(nn.Module):
    def predict(
        self,
        network: _ActionDecoder,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return network(features=features, actions=None)


class _ExplainabilityPolicy:
    def __init__(self, encoding_pipeline: nn.Module) -> None:
        self.encoding_pipeline = encoding_pipeline
        self.decoder = _ActionDecoder()
        self.algorithm = _Algorithm()
        self.normalizer = MagicMock()
        self.observation_space = MagicMock()
        self.observation_space.cameras = {Cameras.LEFT.value: MagicMock()}
        self.tokenizer = None

    def _strip_metadata_passthrough_observations(
        self,
        observation: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return observation

    def _build_algorithm_features(
        self,
        observation: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return self.encoding_pipeline(observation=observation)

    def zero_grad(self) -> None:
        self.encoding_pipeline.zero_grad()
        self.decoder.zero_grad()


@pytest.fixture
def camera_observation_factory() -> Callable[[], dict[str, torch.Tensor]]:
    def factory() -> dict[str, torch.Tensor]:
        return {
            Cameras.LEFT.value: torch.arange(48, dtype=torch.float32).reshape(
                1,
                1,
                3,
                4,
                4,
            )
        }

    return factory


@pytest.fixture
def multi_camera_observation_factory() -> Callable[[], dict[str, torch.Tensor]]:
    def factory() -> dict[str, torch.Tensor]:
        left = torch.arange(48, dtype=torch.float32).reshape(1, 1, 3, 4, 4)
        right = torch.zeros(1, 1, 3, 4, 4)
        return {
            Cameras.LEFT.value: left,
            Cameras.RIGHT.value: right,
        }

    return factory


@pytest.fixture(
    params=[_SpatialEncodingPipeline, _TokenEncodingPipeline],
    ids=["spatial", "token"],
)
def explainability_encoding_pipeline_factory(
    request: pytest.FixtureRequest,
) -> Callable[[], nn.Module]:
    pipeline_class = request.param

    def factory() -> nn.Module:
        return pipeline_class()

    return factory


@pytest.fixture
def multi_camera_encoding_pipeline_factory() -> Callable[[], nn.Module]:
    def factory() -> nn.Module:
        return _MultiCameraSpatialEncodingPipeline()

    return factory


@pytest.fixture
def explainability_policy_factory() -> Callable[[nn.Module], _ExplainabilityPolicy]:
    def factory(encoding_pipeline: nn.Module) -> _ExplainabilityPolicy:
        return _ExplainabilityPolicy(encoding_pipeline=encoding_pipeline)

    return factory
