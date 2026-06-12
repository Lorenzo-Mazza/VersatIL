"""Shared fixtures for versatil.metrics tests."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.metrics.base import LossOutput
from versatil.metrics.regularization_context import (
    PolicyForwardContext,
    PolicyRegularizationGraph,
)
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.decoders.base import DecoderInput
from versatil.models.policy import Policy


@pytest.fixture
def latent_sample_factory(rng: np.random.Generator) -> Callable[..., torch.Tensor]:
    def factory(
        batch_size: int = 16,
        latent_dimension: int = 8,
    ) -> torch.Tensor:
        data = rng.standard_normal((batch_size, latent_dimension)).astype(np.float32)
        return torch.from_numpy(data)

    return factory


class ScalingEncodingPipeline(torch.nn.Module):
    """Encoding pipeline fixture that emits one scaled feature tensor."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.encoders = {}
        self.conditional_encoders = {}

    def forward(
        self,
        observation: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {"feature": self.scale * observation["raw"]}


class ScalingDecoder(torch.nn.Module):
    """Decoder fixture that scales one feature into one action output.

    With ``output_noise_scale > 0`` the decoder adds Gaussian noise drawn from
    the global RNG on every forward, emulating stochastic algorithms whose
    sampling must be replayed across regularization graph re-entries.
    """

    def __init__(
        self,
        scale: float = 3.0,
        output_dimension: int | None = None,
        chunk_count: int = 1,
        output_noise_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(scale))
        self.output_dimension = output_dimension
        self.chunk_count = chunk_count
        self.output_noise_scale = output_noise_scale
        self.decoder_input = DecoderInput(keys=["feature"])

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        feature = features["feature"]
        if self.output_dimension is not None:
            feature = feature[..., : self.output_dimension]
        output = self.scale * feature.unsqueeze(1)  # (B, D) -> (B, 1, D)
        if self.output_noise_scale > 0.0:
            output = output + self.output_noise_scale * torch.randn_like(output)
        return {"action": output.repeat(1, self.chunk_count, 1)}

    def get_loss_output_keys(self) -> set[str]:
        return {"action"}

    def get_prediction_output_keys(self) -> set[str]:
        return {"action"}


class DirectAlgorithm(DecodingAlgorithm):
    """Algorithm fixture that delegates directly to the decoder."""

    def forward(
        self,
        network: ScalingDecoder,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        return network(features=features, actions=actions)

    def predict(
        self,
        network: ScalingDecoder,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return network(features=features, actions=None)


class ZeroLoss(torch.nn.Module):
    """Loss fixture that contributes no training loss."""

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        is_pad: torch.Tensor | None = None,
    ) -> LossOutput:
        return LossOutput(
            total_loss=torch.tensor(0.0, device=next(iter(predictions.values())).device)
        )

    def get_required_keys(self) -> set[str]:
        return set()


@pytest.fixture
def regularizer_batch_factory(
    rng: np.random.Generator,
    action_tensor_factory: Callable[..., torch.Tensor],
    padding_mask_factory: Callable[..., torch.Tensor],
) -> Callable[..., dict[str, dict[str, torch.Tensor]]]:
    def factory(
        batch_size: int = 2,
        feature_dimension: int = 3,
        prediction_horizon: int = 1,
    ) -> dict[str, dict[str, torch.Tensor]]:
        raw_observation = torch.from_numpy(
            rng.standard_normal((batch_size, feature_dimension)).astype(np.float32)
        )
        return {
            SampleKey.OBSERVATION.value: {"raw": raw_observation},
            SampleKey.ACTION.value: {
                "action": action_tensor_factory(
                    batch_size=batch_size,
                    sequence_length=prediction_horizon,
                    action_dimension=feature_dimension,
                ),
                SampleKey.IS_PAD_ACTION.value: padding_mask_factory(
                    batch_size=batch_size,
                    sequence_length=prediction_horizon,
                ),
            },
        }

    return factory


@pytest.fixture
def regularizer_policy_factory() -> Callable[..., Policy]:
    def factory(
        encoder_scale: float = 2.0,
        decoder_scale: float = 3.0,
        decoder_output_dimension: int | None = None,
        decoder_chunk_count: int = 1,
        decoder_output_noise_scale: float = 0.0,
    ) -> Policy:
        observation_space = MagicMock(spec=ObservationSpace)
        observation_space.observations_metadata = {}
        action_space = MagicMock(spec=ActionSpace)
        action_space.actions_metadata = {}
        return Policy(
            encoding_pipeline=ScalingEncodingPipeline(scale=encoder_scale),
            algorithm=DirectAlgorithm(),
            decoder=ScalingDecoder(
                scale=decoder_scale,
                output_dimension=decoder_output_dimension,
                chunk_count=decoder_chunk_count,
                output_noise_scale=decoder_output_noise_scale,
            ),
            observation_space=observation_space,
            action_space=action_space,
            prediction_horizon=1,
            observation_horizon=1,
            loss=ZeroLoss(),
            device="cpu",
        )

    return factory


@pytest.fixture
def regularizer_forward_context_factory(
    regularizer_batch_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
) -> Callable[..., PolicyForwardContext]:
    def factory(
        policy: Policy,
        batch_size: int = 2,
        feature_dimension: int = 3,
    ) -> PolicyForwardContext:
        batch = regularizer_batch_factory(
            batch_size=batch_size,
            feature_dimension=feature_dimension,
        )
        return policy._build_forward_context(batch=batch)

    return factory


@pytest.fixture
def regularizer_graph_factory(
    regularizer_forward_context_factory: Callable[..., PolicyForwardContext],
) -> Callable[..., PolicyRegularizationGraph]:
    def factory(
        policy: Policy,
        batch_size: int = 2,
        feature_dimension: int = 3,
    ) -> PolicyRegularizationGraph:
        context = regularizer_forward_context_factory(
            policy=policy,
            batch_size=batch_size,
            feature_dimension=feature_dimension,
        )
        return policy._build_regularization_graph(context=context)

    return factory
