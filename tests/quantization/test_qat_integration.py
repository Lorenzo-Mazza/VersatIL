"""Integration tests for torchao QAT on real VersatIL policies."""

from collections.abc import Callable
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    PerGroup,
)

import versatil.configs  # noqa: F401
from versatil.data.constants import ProprioKey, SampleKey
from versatil.models.policy import Policy
from versatil.quantization.strategies import QATStrategy

HYDRA_CONFIGS_ROOT = str(Path(__file__).parents[2] / "hydra_configs")
LIBERO_LANGUAGE_TINY_CONFIG = (
    "end_to_end_training_runs/libero_lerobot/action_transformer_language_tiny"
)
FAKE_QUANTIZED_LINEAR_CLASS_NAME = "FakeQuantizedLinear"


class _ScopedQATModel(nn.Module):
    """Small module with scoped and group-incompatible linear layers."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.ModuleDict(
            {
                "compatible": nn.Linear(32, 16),
                "incompatible": nn.Linear(8, 8),
            }
        )
        self.head = nn.Linear(16, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = torch.relu(self.encoder["compatible"](inputs))
        return self.head(encoded)


@pytest.fixture
def libero_language_qat_policy_factory() -> Callable[[str], tuple[Policy, QATStrategy]]:
    """Factory for a real Libero language ActionTransformer with QAT config."""

    def factory(qat_preset: str) -> tuple[Policy, QATStrategy]:
        overrides = [
            f"+quantization=../../../quantization/{qat_preset}",
            "experiment.device=cuda",
        ]
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            yaml_config = compose(
                config_name=LIBERO_LANGUAGE_TINY_CONFIG,
                overrides=overrides,
            )
        policy = hydra.utils.instantiate(yaml_config.policy)
        qat_strategy = hydra.utils.instantiate(yaml_config.quantization)
        return policy, qat_strategy

    return factory


@pytest.fixture
def libero_language_batch_factory(
    rng: np.random.Generator,
) -> Callable[[Policy, torch.device], dict[str, dict[str, torch.Tensor]]]:
    """Factory for a real Libero language ActionTransformer training batch."""

    def factory(
        policy: Policy,
        device: torch.device,
    ) -> dict[str, dict[str, torch.Tensor]]:
        batch_size = 2
        observation_horizon = policy.observation_horizon
        prediction_horizon = policy.prediction_horizon
        max_token_length = 32
        vocab_size = policy.encoding_pipeline.encoders["instruction"].get_vocab_size()

        observation = {}
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
def scoped_qat_model_factory(rng: np.random.Generator) -> Callable[[], _ScopedQATModel]:
    """Factory for deterministic small QAT models."""

    def factory() -> _ScopedQATModel:
        model = _ScopedQATModel()
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


def _nonzero_gradient_count(module: nn.Module) -> int:
    return sum(
        1
        for parameter in module.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum().item() > 0.0
    )


def _is_fake_quantized_linear(module: nn.Module) -> bool:
    return module.__class__.__name__ == FAKE_QUANTIZED_LINEAR_CLASS_NAME


@pytest.mark.integration
@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    "qat_preset",
    [
        "qat_int8_dynamic_intx_int4",
        "qat_int4_weight_only",
    ],
)
def test_libero_language_action_transformer_qat_forward_backward(
    qat_preset: str,
    libero_language_qat_policy_factory: Callable[[str], tuple[Policy, QATStrategy]],
    libero_language_batch_factory: Callable[
        [Policy, torch.device], dict[str, dict[str, torch.Tensor]]
    ],
) -> None:
    device = torch.device("cuda")
    policy, qat_strategy = libero_language_qat_policy_factory(qat_preset)
    policy.to(device=device)
    policy.train()
    batch = libero_language_batch_factory(policy, device)

    with torch.no_grad():
        warmup_loss = policy.compute_loss(batch=batch).total_loss

    qat_strategy.prepare_model(model=policy)
    loss_output = policy.compute_loss(batch=batch)
    loss_output.total_loss.backward()

    fake_quantized_linear_count = sum(
        1 for module in policy.modules() if _is_fake_quantized_linear(module=module)
    )
    nonzero_gradient_count = _nonzero_gradient_count(module=policy)

    assert torch.isfinite(warmup_loss)
    assert torch.isfinite(loss_output.total_loss)
    assert len(qat_strategy._prepared_module_names) > 0
    assert fake_quantized_linear_count == len(qat_strategy._prepared_module_names)
    assert nonzero_gradient_count > 0


@pytest.mark.integration
@pytest.mark.requires_gpu
def test_qat_prepare_filters_scoped_group_incompatible_linears(
    scoped_qat_model_factory: Callable[[], _ScopedQATModel],
    linear_input_factory: Callable[[torch.device], torch.Tensor],
) -> None:
    device = torch.device("cuda")
    model = scoped_qat_model_factory().to(device=device)
    inputs = linear_input_factory(device)
    strategy = QATStrategy(
        base_config=Int4WeightOnlyConfig(group_size=32),
        module_paths=["encoder"],
    )

    strategy.prepare_model(model=model)
    output = model(inputs)
    output.square().mean().backward()

    assert _is_fake_quantized_linear(module=model.encoder["compatible"])
    assert isinstance(model.encoder["incompatible"], nn.Linear)
    assert isinstance(model.head, nn.Linear)
    assert torch.isfinite(output).all()
    assert _nonzero_gradient_count(module=model) > 0


@pytest.mark.integration
@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    "base_config",
    [
        Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(32),
        ),
        IntxWeightOnlyConfig(
            weight_dtype=torch.int4,
            granularity=PerGroup(32),
        ),
    ],
)
def test_qat_convert_runs_supported_torchao_configs(
    base_config: Int8DynamicActivationIntxWeightConfig | IntxWeightOnlyConfig,
    scoped_qat_model_factory: Callable[[], _ScopedQATModel],
    linear_input_factory: Callable[[torch.device], torch.Tensor],
) -> None:
    device = torch.device("cuda")
    model = scoped_qat_model_factory().to(device=device)
    inputs = linear_input_factory(device)
    strategy = QATStrategy(base_config=base_config)

    strategy.prepare_model(model=model)
    prepared_output = model(inputs)
    prepared_output.square().mean().backward()
    strategy.convert_model(model=model)

    with torch.no_grad():
        converted_output = model(inputs)

    fake_quantized_linear_count = sum(
        1 for module in model.modules() if _is_fake_quantized_linear(module=module)
    )
    assert torch.isfinite(prepared_output).all()
    assert torch.isfinite(converted_output).all()
    assert fake_quantized_linear_count == 0
    assert converted_output.shape == (2, 4)
