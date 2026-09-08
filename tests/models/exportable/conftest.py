"""Shared fixtures for policy export adapters."""

from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from versatil.data.task import ActionSpace
from versatil.data.tokenization.action_discretizer import (
    BinnedActionDiscretizer,
    FastActionDiscretizer,
)
from versatil.data.tokenization.action_tokenizer import ActionTokenizer
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.behavior_cloning import BehavioralCloning
from versatil.models.decoding.algorithm.diffusion import Diffusion
from versatil.models.decoding.algorithm.flow_matching import FlowMatching
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.autoregressive_vla import (
    AutoregressiveVLADecoder,
)
from versatil.models.decoding.decoders.factory.gpt_action_transformer import (
    GPTActionTransformer,
)
from versatil.models.exportable.base import ExportablePolicy
from versatil.models.exportable.metadata import (
    PolicyExportMetadata,
)
from versatil.models.layers.denoising.diffusion_schedule import DiffusionSchedule
from versatil.models.policy import Policy


@pytest.fixture
def exportable_factory(
    encoding_pipeline_factory: Callable[..., MagicMock],
) -> Callable[..., ExportablePolicy]:
    """Factory for ExportablePolicy with configurable keys."""

    def factory(
        observation_keys: list[str] | None = None,
        action_keys: list[str] | None = None,
        pipeline: MagicMock | None = None,
        export_metadata: PolicyExportMetadata | None = None,
    ) -> ExportablePolicy:
        if observation_keys is None:
            observation_keys = ["depth", "left"]
        if action_keys is None:
            action_keys = ["orientation", "position"]
        return ExportablePolicy(
            encoding_pipeline=pipeline or encoding_pipeline_factory(),
            algorithm=MagicMock(),
            decoder=MagicMock(),
            observation_keys=observation_keys,
            action_keys=action_keys,
            export_metadata=export_metadata,
        )

    return factory


@pytest.fixture
def from_policy_factory(
    policy_factory: Callable[..., Policy],
    vision_encoder_factory: Callable[..., MagicMock],
    encoding_pipeline_factory: Callable[..., MagicMock],
) -> Callable[..., Policy]:
    """Factory for Policy instances configured for from_policy tests."""

    def factory(
        encoder_keys: dict[str, list[str]] | None = None,
        conditional_encoder_keys: dict[str, list[str]] | None = None,
        action_keys: list[str] | None = None,
    ) -> Policy:
        if encoder_keys is None:
            encoder_keys = {"rgb": ["left", "right"]}
        if conditional_encoder_keys is None:
            conditional_encoder_keys = {}
        if action_keys is None:
            action_keys = ["position"]
        encoders = nn.ModuleDict(
            {
                name: vision_encoder_factory(input_keys=keys)
                for name, keys in encoder_keys.items()
            }
        )
        conditional_encoders = nn.ModuleDict(
            {
                name: vision_encoder_factory(input_keys=keys)
                for name, keys in conditional_encoder_keys.items()
            }
        )
        pipeline = encoding_pipeline_factory(
            encoders=encoders,
            conditional_encoders=conditional_encoders,
        )
        decoder = MagicMock()
        decoder.decoder_input.needs_raw_observations = False
        decoder.requires_tokenized_actions = False
        decoder.action_heads = nn.ModuleDict(
            {key: nn.Identity() for key in action_keys}
        )
        return policy_factory(
            encoding_pipeline=pipeline,
            decoder=decoder,
        )

    return factory


@pytest.fixture
def observation_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for observation tensors with configurable shape."""

    def factory(
        batch_size: int = 2,
        channels: int = 3,
        height: int = 64,
        width: int = 64,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, channels, height, width)).astype(
                np.float32
            )
        )

    return factory


@pytest.fixture
def tokenized_export_policy_factory(
    encoding_pipeline_factory: Callable[..., MagicMock],
) -> Callable[..., MagicMock]:
    def factory(
        supported_algorithm: bool = True,
        supported_decoder: bool = True,
        has_tokenizer: bool = True,
        binned: bool = True,
        fitted: bool = True,
        tokenizer_horizon: int = 2,
        tokenizer_dimension: int = 3,
        max_token_len: int = 7,
        deterministic: bool = True,
        vlm: bool = False,
    ) -> MagicMock:
        policy = MagicMock(spec=Policy)
        policy.encoding_pipeline = encoding_pipeline_factory()
        policy.algorithm = MagicMock(
            spec=BehavioralCloning if supported_algorithm else DecodingAlgorithm
        )
        policy.algorithm.injected_feature_keys.return_value = set()
        decoder_type = AutoregressiveVLADecoder if vlm else GPTActionTransformer
        policy.decoder = MagicMock(
            spec=decoder_type if supported_decoder else ActionDecoder
        )
        policy.decoder.requires_tokenized_actions = True
        policy.decoder.deterministic = deterministic
        policy.prediction_horizon = 2
        policy.action_space = MagicMock(spec=ActionSpace)
        policy.action_space.get_total_action_dim.return_value = 3
        policy.input_keys = ["observation"]
        policy.output_keys = ["position"]
        if has_tokenizer:
            discretizer = MagicMock(
                spec=BinnedActionDiscretizer if binned else FastActionDiscretizer
            )
            discretizer.is_fitted = fitted
            discretizer.time_horizon = tokenizer_horizon
            discretizer.action_dim = tokenizer_dimension
            action_tokenizer = MagicMock(spec=ActionTokenizer)
            action_tokenizer.action_discretizer = discretizer
            action_tokenizer.max_token_len = max_token_len
            policy.tokenizer = MagicMock(spec=Tokenizer)
            policy.tokenizer.action_tokenizer = action_tokenizer
            policy.decoder.tokenizer = action_tokenizer
        else:
            policy.tokenizer = None
        return policy

    return factory


@pytest.fixture
def token_generation_tensors_factory() -> Callable[..., tuple[MagicMock, MagicMock]]:
    def factory(token_count: int = 7) -> tuple[MagicMock, MagicMock]:
        tokens = MagicMock(spec=torch.Tensor)
        tokens.shape = (2, token_count)
        return MagicMock(spec=torch.Tensor), tokens

    return factory


@pytest.fixture
def denoising_reference_factory() -> Iterator[Callable[..., MagicMock]]:
    with patch(
        "versatil.models.exportable.denoising.resolve_feature_reference"
    ) as resolve_reference:

        def factory(
            dtype: torch.dtype = torch.bfloat16,
            device: torch.device = torch.device("cpu"),
        ) -> MagicMock:
            resolve_reference.return_value = (2, device, dtype)
            return resolve_reference

        yield factory


@pytest.fixture
def denoising_export_policy_factory(
    encoding_pipeline_factory: Callable[..., MagicMock],
    mock_action_decoder_factory: Callable[..., MagicMock],
) -> Callable[..., MagicMock]:
    def factory(diffusion: bool, stochastic: bool) -> MagicMock:
        policy = MagicMock(spec=Policy)
        policy.encoding_pipeline = encoding_pipeline_factory()
        policy.algorithm = MagicMock(spec=Diffusion if diffusion else FlowMatching)
        policy.algorithm.injected_feature_keys.return_value = set()
        if diffusion:
            policy.algorithm.inference_schedule = MagicMock(spec=DiffusionSchedule)
            policy.algorithm.inference_schedule.stochastic = stochastic
            policy.algorithm.inference_schedule.timesteps = (9, 4, 0)
        policy.decoder = mock_action_decoder_factory(
            action_keys=["position", "orientation"],
            prediction_dimension=3,
            prediction_horizon=2,
        )
        policy.decoder.requires_tokenized_actions = False
        policy.action_space = policy.decoder.action_space
        policy.prediction_horizon = 2
        policy.input_keys = ["observation"]
        policy.output_keys = ["position", "orientation"]
        return policy

    return factory
