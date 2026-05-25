"""Models package test fixtures: mock factories for Policy dependencies."""

from collections.abc import Callable
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.metrics.base import BaseLoss
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.encoding.encoders.base import EncoderInput, EncodingMixin
from versatil.models.encoding.pipeline import EncodingPipeline
from versatil.models.policy import Policy


@pytest.fixture
def parameter_count() -> Callable[[torch.nn.Module], int]:
    """Return a module parameter counter."""

    def count_parameters(module: torch.nn.Module) -> int:
        return sum(parameter.numel() for parameter in module.parameters())

    return count_parameters


@pytest.fixture
def trainable_parameter_count() -> Callable[[torch.nn.Module], int]:
    """Return a trainable-parameter counter."""

    def count_trainable_parameters(module: torch.nn.Module) -> int:
        return sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    return count_trainable_parameters


@pytest.fixture
def input_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for generic input tensors with configurable shape."""

    def factory(
        batch_size: int = 2,
        input_dimension: int = 64,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        if sequence_length is not None:
            shape = (batch_size, sequence_length, input_dimension)
        else:
            shape = (batch_size, input_dimension)
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    return factory


@pytest.fixture
def sequence_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for sequence tensors (B, S, D)."""

    def factory(
        batch_size: int = 2,
        sequence_length: int = 4,
        embedding_dimension: int = 32,
    ) -> torch.Tensor:
        data = rng.standard_normal(
            (batch_size, sequence_length, embedding_dimension)
        ).astype(np.float32)
        return torch.from_numpy(data)

    return factory


@pytest.fixture
def embedding_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for action embedding tensors (B, prediction_horizon, embedding_dimension)."""

    def factory(
        batch_size: int = 2,
        prediction_horizon: int = 8,
        embedding_dimension: int = 64,
    ) -> torch.Tensor:
        shape = (batch_size, prediction_horizon, embedding_dimension)
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    return factory


@pytest.fixture
def mock_action_decoder_factory() -> Callable[..., MagicMock]:
    """Factory for mock ActionDecoder with configurable action space metadata."""

    def factory(
        action_keys: list[str] | None = None,
        prediction_dimension: int = 3,
        prediction_horizon: int = 8,
        return_value: dict[str, torch.Tensor] | None = None,
    ) -> MagicMock:
        if action_keys is None:
            action_keys = ["position_action"]

        @dataclass
        class MockMeta:
            requires_prediction_head: bool = True
            prediction_dimension: int = 3

        network = MagicMock()
        network.action_space.actions_metadata = {
            key: MockMeta(prediction_dimension=prediction_dimension)
            for key in action_keys
        }
        network.prediction_horizon = prediction_horizon
        if return_value is not None:
            network.return_value = return_value
        else:
            network.return_value = {
                key: torch.zeros(2, prediction_horizon, prediction_dimension)
                for key in action_keys
            }
        return network

    return factory


@pytest.fixture
def feature_dictionary_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for feature dictionaries produced by encoding pipeline."""

    def factory(
        batch_size: int = 2,
        feature_dimension: int = 64,
        feature_keys: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if feature_keys is None:
            feature_keys = ["rgb_features"]
        return {
            key: torch.from_numpy(
                rng.standard_normal((batch_size, feature_dimension)).astype(np.float32)
            )
            for key in feature_keys
        }

    return factory


@pytest.fixture
def observation_dictionary_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for observation dictionaries."""

    def factory(
        batch_size: int = 2,
        observation_dimension: int = 7,
        observation_keys: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if observation_keys is None:
            observation_keys = ["proprio_robot_frame"]
        return {
            key: torch.from_numpy(
                rng.standard_normal((batch_size, observation_dimension)).astype(
                    np.float32
                )
            )
            for key in observation_keys
        }

    return factory


@pytest.fixture
def action_dictionary_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for action dictionaries."""

    def factory(
        batch_size: int = 2,
        prediction_horizon: int = 4,
        action_dimension: int = 7,
        action_keys: list[str] | None = None,
        include_padding_mask: bool = True,
    ) -> dict[str, torch.Tensor]:
        if action_keys is None:
            action_keys = ["proprio_robot_frame"]
        result = {
            key: torch.from_numpy(
                rng.standard_normal(
                    (batch_size, prediction_horizon, action_dimension)
                ).astype(np.float32)
            )
            for key in action_keys
        }
        if include_padding_mask:
            result[SampleKey.IS_PAD_ACTION.value] = torch.zeros(
                batch_size, prediction_horizon, dtype=torch.bool
            )
        return result

    return factory


@pytest.fixture
def batch_dictionary_factory(
    observation_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
    action_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
) -> Callable[..., dict[str, dict[str, torch.Tensor]]]:
    """Factory for full batch dictionaries with observation and action sub-dicts."""

    def factory(
        batch_size: int = 2,
        observation_dimension: int = 7,
        action_dimension: int = 7,
        prediction_horizon: int = 4,
    ) -> dict[str, dict[str, torch.Tensor]]:
        return {
            SampleKey.OBSERVATION.value: observation_dictionary_factory(
                batch_size=batch_size,
                observation_dimension=observation_dimension,
            ),
            SampleKey.ACTION.value: action_dictionary_factory(
                batch_size=batch_size,
                prediction_horizon=prediction_horizon,
                action_dimension=action_dimension,
            ),
        }

    return factory


@pytest.fixture
def vision_encoder_factory() -> Callable[..., MagicMock]:
    """Factory for mock vision encoders with configurable architecture attributes.

    With spec=EncodingMixin, attributes not in the spec are absent by default.
    Pass has_backbone=True etc. to explicitly add the corresponding attribute.
    """

    def factory(
        has_backbone: bool = False,
        has_stages: bool = False,
        has_layer4: bool = False,
        has_attention_block: bool = False,
        input_keys: list[str] | None = None,
    ) -> MagicMock:
        encoder = MagicMock(spec=EncodingMixin)
        if has_backbone:
            encoder.backbone = MagicMock()
        if has_stages:
            encoder.stages = [MagicMock()]
        if has_layer4:
            encoder.layer4 = MagicMock()
        if has_attention_block:
            encoder.attention_block = MagicMock()
        if input_keys is not None:
            input_spec = MagicMock(spec=EncoderInput)
            input_spec.keys = input_keys
            encoder.input_specification = input_spec
        return encoder

    return factory


@pytest.fixture
def encoding_pipeline_factory() -> Callable[..., MagicMock]:
    """Factory for mock encoding pipelines with configurable encoder dicts."""

    def factory(
        encoders: dict[str, MagicMock] | None = None,
        conditional_encoders: dict[str, MagicMock] | None = None,
    ) -> MagicMock:
        pipeline = MagicMock(spec=EncodingPipeline)
        pipeline.encoders = encoders if encoders is not None else {}
        pipeline.conditional_encoders = (
            conditional_encoders if conditional_encoders is not None else {}
        )
        return pipeline

    return factory


@pytest.fixture
def policy_factory(
    feature_dictionary_factory: Callable[..., dict[str, torch.Tensor]],
) -> Callable[..., Policy]:
    """Factory for creating Policy instances with mocked dependencies."""

    def factory(
        encoding_pipeline: EncodingPipeline | None = None,
        algorithm: DecodingAlgorithm | None = None,
        decoder: ActionDecoder | None = None,
        observation_space: ObservationSpace | None = None,
        action_space: ActionSpace | None = None,
        prediction_horizon: int = 4,
        observation_horizon: int = 2,
        loss: BaseLoss | None = None,
        device: str = "cpu",
        metadata_passthrough: dict[str, dict[str, str]] | None = None,
        feature_return_value: dict[str, torch.Tensor] | None = None,
        algorithm_forward_return: dict[str, torch.Tensor] | None = None,
        algorithm_predict_return: dict[str, torch.Tensor] | None = None,
    ) -> Policy:
        if encoding_pipeline is None:
            encoding_pipeline = MagicMock(spec=EncodingPipeline)
            if feature_return_value is None:
                feature_return_value = feature_dictionary_factory()
            encoding_pipeline.return_value = feature_return_value
            encoding_pipeline.encoders = {}
            encoding_pipeline.conditional_encoders = {}
        if algorithm is None:
            algorithm = MagicMock(spec=DecodingAlgorithm)
            if algorithm_forward_return is not None:
                algorithm.forward.return_value = algorithm_forward_return
            if algorithm_predict_return is not None:
                algorithm.predict.return_value = algorithm_predict_return
        if decoder is None:
            if feature_return_value is None and isinstance(
                encoding_pipeline.return_value, dict
            ):
                feature_return_value = encoding_pipeline.return_value
            decoder_input_keys = (
                list(feature_return_value.keys())
                if feature_return_value is not None
                else []
            )
            decoder = MagicMock(
                spec=ActionDecoder,
                decoder_input=DecoderInput(keys=decoder_input_keys),
            )
        if observation_space is None:
            observation_space = MagicMock(spec=ObservationSpace)
            observation_space.observations_metadata = {}
        if action_space is None:
            action_space = MagicMock(spec=ActionSpace)
        if loss is None:
            loss = MagicMock(spec=BaseLoss)
        return Policy(
            encoding_pipeline=encoding_pipeline,
            algorithm=algorithm,
            decoder=decoder,
            observation_space=observation_space,
            action_space=action_space,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            loss=loss,
            device=device,
            metadata_passthrough=metadata_passthrough,
        )

    return factory
