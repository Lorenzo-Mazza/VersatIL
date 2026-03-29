"""Encoding package test fixtures: mock factories for encoder and fusion dependencies."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.data.metadata import CameraMetadata
from versatil.data.task import ObservationSpace
from versatil.models.encoding.encoders.base import (
    EncoderInput,
    EncodingMixin,
)
from versatil.models.encoding.encoders.conditional import ConditionalEncoder
from versatil.models.encoding.fusion.base import FusionModule
from versatil.models.feature_meta import (
    FeatureMetadata,
    FeatureType,
    infer_feature_type,
)


@pytest.fixture
def encoder_mock_factory(rng: np.random.Generator) -> Callable[..., MagicMock]:
    """Factory for mock encoders compatible with EncodingPipeline setup."""

    def factory(
        output_features: list[str] | None = None,
        output_dimensions: dict[str, tuple[int, ...]] | None = None,
        input_keys: list[str] | None = None,
        requires_tokenized: bool = False,
        vocab_size: int | None = None,
        forward_return: dict[str, torch.Tensor] | None = None,
        batch_size: int = 2,
    ) -> MagicMock:
        if output_features is None:
            output_features = ["embedding"]
        if output_dimensions is None:
            output_dimensions = dict.fromkeys(output_features, (64,))
        if input_keys is None:
            input_keys = ["left"]
        encoder = MagicMock(spec=EncodingMixin)
        encoder.get_output_specification.return_value = [
            FeatureMetadata(
                key=feat,
                feature_type=infer_feature_type(output_dimensions[feat]),
                dimension=output_dimensions[feat],
            )
            for feat in output_features
        ]
        encoder.input_specification = EncoderInput(
            keys=input_keys,
            requires_tokenized=requires_tokenized,
        )
        encoder.get_vocab_size.return_value = vocab_size
        if forward_return is None:
            forward_return = {
                feat: torch.from_numpy(
                    rng.standard_normal((batch_size, *dim)).astype(np.float32)
                )
                for feat, dim in output_dimensions.items()
            }
        encoder.return_value = forward_return
        return encoder

    return factory


@pytest.fixture
def conditional_encoder_mock_factory(
    rng: np.random.Generator,
) -> Callable[..., MagicMock]:
    """Factory for mock conditional encoders compatible with EncodingPipeline setup."""

    def factory(
        output_features: list[str] | None = None,
        output_dimensions: dict[str, tuple[int, ...]] | None = None,
        input_keys: list[str] | None = None,
        condition_key: str = "rgb_encoder_embedding",
        forward_return: dict[str, torch.Tensor] | None = None,
        batch_size: int = 2,
    ) -> MagicMock:
        if output_features is None:
            output_features = ["embedding"]
        if output_dimensions is None:
            output_dimensions = dict.fromkeys(output_features, (64,))
        if input_keys is None:
            input_keys = ["right"]
        encoder = MagicMock(spec=ConditionalEncoder)
        encoder.get_output_specification.return_value = [
            FeatureMetadata(
                key=feat,
                feature_type=infer_feature_type(output_dimensions[feat]),
                dimension=output_dimensions[feat],
            )
            for feat in output_features
        ]
        encoder.input_specification = EncoderInput(
            keys=input_keys,
            conditioning_key=condition_key,
            requires_tokenized=False,
        )
        encoder.condition_key = condition_key
        encoder.get_vocab_size.return_value = None
        if forward_return is None:
            forward_return = {
                feat: torch.from_numpy(
                    rng.standard_normal((batch_size, *dim)).astype(np.float32)
                )
                for feat, dim in output_dimensions.items()
            }
        encoder.return_value = forward_return
        return encoder

    return factory


@pytest.fixture
def fusion_module_mock_factory(
    rng: np.random.Generator,
) -> Callable[..., MagicMock]:
    """Factory for mock fusion modules compatible with EncodingPipeline setup."""

    def factory(
        input_features: list[str] | None = None,
        output_name: str = "fused",
        output_dimension: int = 128,
        forward_return: torch.Tensor | None = None,
        batch_size: int = 2,
    ) -> MagicMock:
        if input_features is None:
            input_features = ["rgb_encoder_embedding"]
        fusion = MagicMock(spec=FusionModule)
        fusion.input_features = input_features
        fusion.output_name = output_name
        fusion.get_output_specification.return_value = FeatureMetadata(
            key=output_name,
            feature_type=FeatureType.FLAT.value,
            dimension=(output_dimension,),
        )
        if forward_return is None:
            forward_return = torch.from_numpy(
                rng.standard_normal((batch_size, output_dimension)).astype(np.float32)
            )
        fusion.return_value = forward_return
        return fusion

    return factory


@pytest.fixture
def default_observation_space(
    observation_space_factory,
) -> ObservationSpace:
    """Default ObservationSpace with left/right cameras for pipeline tests."""
    return observation_space_factory(
        observations_metadata={
            "left": CameraMetadata(
                camera_key="left",
                dtype="uint8",
                channels=3,
                image_height=224,
                image_width=224,
            ),
            "right": CameraMetadata(
                camera_key="right",
                dtype="uint8",
                channels=3,
                image_height=224,
                image_width=224,
            ),
        }
    )
