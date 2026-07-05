"""Shared fixtures for quantization tests."""

from collections.abc import Callable, Iterator
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock

import hydra
import numpy as np
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir

import versatil.configs  # noqa: F401
from versatil.configs.paths import get_hydra_configs_dir
from versatil.data.constants import ProprioKey, SampleKey
from versatil.models.layers.frozen_batchnorm import FrozenBatchNorm2d
from versatil.models.policy import Policy
from versatil.quantization.calibration import CalibrationDataProvider
from versatil.quantization.constants import PT2EBackendName
from versatil.quantization.pt2e.backends.base import BasePT2EBackend
from versatil.quantization.pt2e.backends.x86_inductor import X86InductorBackend
from versatil.quantization.workflows.eager import EagerQuantizationWorkflow

HYDRA_CONFIGS_ROOT = str(get_hydra_configs_dir())
LANGUAGE_ACTION_TRANSFORMER_TINY_CONFIG = (
    "end_to_end_training_runs/libero_lerobot/bcat_language_tiny"
)


class _ScopedLinearModel(nn.Module):
    """Small module with scoped and group-incompatible linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.ModuleDict(
            {
                "compatible": nn.Linear(in_features=32, out_features=16),
                "incompatible": nn.Linear(in_features=8, out_features=8),
            }
        )
        self.head = nn.Linear(in_features=16, out_features=4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = torch.relu(self.encoder["compatible"](inputs))
        return self.head(encoded)


class _SyntheticConvModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_dimension: int,
        use_frozen_bn: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
        )
        self.batchnorm = (
            FrozenBatchNorm2d(dimension=hidden_channels)
            if use_frozen_bn
            else nn.BatchNorm2d(num_features=hidden_channels)
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=hidden_channels,
            out_features=output_dimension,
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        features = self.relu(self.batchnorm(self.conv(image)))
        return (self.linear(self.flatten(self.pool(features))),)


class _TwoPartConvModel(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_dimension: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_channels, out_features=hidden_channels),
            nn.ReLU(),
            nn.Linear(in_features=hidden_channels, out_features=output_dimension),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.decoder(self.encoder(image)),)


class _CountingCalibration:
    """Calibration iterable that records consumed batches."""

    def __init__(self, batches: list[tuple[torch.Tensor, ...]]) -> None:
        self._batches = batches
        self.consumed_batches = 0

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """Yield calibration batches and count consumption."""
        for batch in self._batches:
            self.consumed_batches += 1
            yield batch


@pytest.fixture
def x86_inductor_backend_factory() -> Callable[..., X86InductorBackend]:
    """Factory for X86InductorBackend with configurable flags."""

    def factory(
        is_dynamic: bool = False,
        is_qat: bool = False,
        reduce_range: bool = False,
    ) -> X86InductorBackend:
        return X86InductorBackend(
            is_dynamic=is_dynamic,
            is_qat=is_qat,
            reduce_range=reduce_range,
        )

    return factory


@pytest.fixture
def mock_pt2e_backend_factory() -> Callable[..., MagicMock]:
    """Factory for mock PT2E backends with configurable flags."""

    def factory(is_dynamic: bool = False, is_qat: bool = False) -> MagicMock:
        backend = MagicMock(spec=BasePT2EBackend)
        backend.name = PT2EBackendName.X86_INDUCTOR.value
        backend.is_dynamic = is_dynamic
        backend.is_qat = is_qat
        backend.create_quantizer.return_value = MagicMock()
        backend.environment_context.return_value = does_not_raise()
        return backend

    return factory


@pytest.fixture
def language_action_transformer_qat_policy_factory() -> Callable[
    [str], tuple[Policy, EagerQuantizationWorkflow]
]:
    """Factory for a real language ActionTransformer policy with QAT config."""

    def factory(qat_preset: str) -> tuple[Policy, EagerQuantizationWorkflow]:
        overrides = [
            f"+quantization={qat_preset}",
            "experiment.device=cuda",
        ]
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            yaml_config = compose(
                config_name=LANGUAGE_ACTION_TRANSFORMER_TINY_CONFIG,
                overrides=overrides,
            )
        policy = hydra.utils.instantiate(yaml_config.policy)
        qat_workflow = hydra.utils.instantiate(yaml_config.quantization)
        return policy, qat_workflow

    return factory


@pytest.fixture
def language_action_transformer_batch_factory(
    rng: np.random.Generator,
) -> Callable[[Policy, torch.device], dict[str, dict[str, torch.Tensor]]]:
    """Factory for language ActionTransformer training batches."""

    def factory(
        policy: Policy,
        device: torch.device,
    ) -> dict[str, dict[str, torch.Tensor]]:
        batch_size = 2
        observation_horizon = policy.observation_horizon
        prediction_horizon = policy.prediction_horizon
        max_token_length = 32
        vocab_size = policy.encoding_pipeline.encoders["instruction"].get_vocab_size()

        observation: dict[str, torch.Tensor] = {}
        for camera_key, metadata in policy.observation_space.cameras.items():
            image_data = rng.standard_normal(
                (
                    batch_size,
                    observation_horizon,
                    metadata.channels,
                    metadata.image_height,
                    metadata.image_width,
                )
            ).astype(np.float32)
            observation[camera_key] = torch.from_numpy(image_data).to(device=device)
        token_data = rng.integers(
            low=1,
            high=min(vocab_size, 128),
            size=(batch_size, observation_horizon, max_token_length),
            dtype=np.int64,
        )
        observation[SampleKey.TOKENIZED_OBSERVATIONS.value] = torch.from_numpy(
            token_data
        ).to(device=device)
        observation[SampleKey.IS_PAD_OBSERVATION.value] = torch.zeros(
            batch_size,
            observation_horizon,
            max_token_length,
            dtype=torch.bool,
            device=device,
        )

        gripper_data = (
            rng.integers(
                low=0,
                high=2,
                size=(batch_size, prediction_horizon, 1),
            ).astype(np.float32)
            * 2.0
            - 1.0
        )
        action = {
            ProprioKey.EE_POS_ACTION.value: torch.from_numpy(
                rng.standard_normal((batch_size, prediction_horizon, 3)).astype(
                    np.float32
                )
            ).to(device=device),
            ProprioKey.EE_ORI_ACTION.value: torch.from_numpy(
                rng.standard_normal((batch_size, prediction_horizon, 3)).astype(
                    np.float32
                )
            ).to(device=device),
            ProprioKey.GRIPPER_STATE_ACTION.value: torch.from_numpy(gripper_data).to(
                device=device
            ),
            SampleKey.IS_PAD_ACTION.value: torch.zeros(
                batch_size,
                prediction_horizon,
                dtype=torch.bool,
                device=device,
            ),
        }
        return {
            SampleKey.OBSERVATION.value: observation,
            SampleKey.ACTION.value: action,
        }

    return factory


@pytest.fixture
def scoped_linear_model_factory(rng: np.random.Generator) -> Callable[[], nn.Module]:
    """Factory for deterministic small linear models."""

    def factory() -> nn.Module:
        model = _ScopedLinearModel()
        for parameter in model.parameters():
            values = rng.standard_normal(parameter.shape).astype(np.float32)
            parameter.data.copy_(torch.from_numpy(values))
        return model

    return factory


@pytest.fixture
def linear_input_factory(
    rng: np.random.Generator,
) -> Callable[[torch.device], torch.Tensor]:
    """Factory for small linear inputs."""

    def factory(device: torch.device) -> torch.Tensor:
        values = rng.standard_normal((2, 32)).astype(np.float32)
        return torch.from_numpy(values).to(device=device)

    return factory


@pytest.fixture
def synthetic_model_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Module]:
    """Factory for deterministic synthetic Conv-BN-Linear models."""

    def factory(
        input_channels: int = 3,
        hidden_channels: int = 16,
        output_dimension: int = 4,
        use_frozen_bn: bool = False,
    ) -> nn.Module:
        model = _SyntheticConvModel(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_dimension=output_dimension,
            use_frozen_bn=use_frozen_bn,
        )
        generator = torch.Generator()
        generator.manual_seed(int(rng.integers(0, 2**31)))
        for parameter in model.parameters():
            nn.init.normal_(parameter, generator=generator)
        if use_frozen_bn:
            for buffer in model.buffers():
                buffer.data.normal_(generator=generator)
            model.batchnorm.running_var.data.abs_()
        model.eval()
        return model

    return factory


@pytest.fixture
def example_inputs_factory(
    rng: np.random.Generator,
) -> Callable[..., tuple[torch.Tensor, ...]]:
    """Factory for spatial input tensors as tuple."""

    def factory(
        batch_size: int = 2,
        channels: int = 3,
        image_size: int = 16,
    ) -> tuple[torch.Tensor, ...]:
        image = torch.from_numpy(
            rng.standard_normal((batch_size, channels, image_size, image_size)).astype(
                np.float32
            )
        )
        return (image,)

    return factory


@pytest.fixture
def two_part_model_factory(
    rng: np.random.Generator,
) -> Callable[..., nn.Module]:
    """Factory for deterministic two-stage Conv-BN-Linear models."""

    def factory(
        input_channels: int = 3,
        hidden_channels: int = 16,
        output_dimension: int = 4,
    ) -> nn.Module:
        model = _TwoPartConvModel(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            output_dimension=output_dimension,
        )
        generator = torch.Generator()
        generator.manual_seed(int(rng.integers(0, 2**31)))
        for parameter in model.parameters():
            nn.init.normal_(parameter, generator=generator)
        model.eval()
        return model

    return factory


@pytest.fixture
def counting_calibration_factory() -> Callable[
    [list[tuple[torch.Tensor, ...]]], _CountingCalibration
]:
    """Factory for calibration iterables that record consumption."""

    def factory(batches: list[tuple[torch.Tensor, ...]]) -> _CountingCalibration:
        return _CountingCalibration(batches=batches)

    return factory


@pytest.fixture
def mock_calibration_provider_factory(
    rng: np.random.Generator,
) -> Callable[..., MagicMock]:
    """Factory for mock CalibrationDataProvider with deterministic batches."""

    def factory(
        batch_size: int = 2,
        input_dimension: int = 4,
        num_batches: int = 3,
    ) -> MagicMock:
        provider = MagicMock(spec=CalibrationDataProvider)
        batches = []
        for _ in range(num_batches):
            data = rng.standard_normal((batch_size, input_dimension)).astype(np.float32)
            batches.append((torch.from_numpy(data),))
        provider.__iter__ = MagicMock(return_value=iter(batches))
        single_data = rng.standard_normal((batch_size, input_dimension)).astype(
            np.float32
        )
        provider.get_single_batch.return_value = (torch.from_numpy(single_data),)
        return provider

    return factory
