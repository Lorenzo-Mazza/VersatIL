"""Tests for versatil.quantization.workflows.eager QAT integration on real policies."""

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import hydra
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as functional
from hydra import compose, initialize_config_dir
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    PerGroup,
)
from torchao.quantization.qat import FakeQuantizedEmbedding

from versatil.configs.paths import get_hydra_configs_dir
from versatil.inference.policy_runtime.executorch_adapter import ExecuTorchModuleAdapter
from versatil.models.policy import Policy
from versatil.post_training_compression.deployment_backends.executorch_xnnpack import (
    ExecutorchXNNPACKBackend,
)
from versatil.quantization.module_target import EagerQuantizationModuleTarget
from versatil.quantization.workflows.eager import EagerQuantizationWorkflow

FAKE_QUANTIZED_LINEAR_CLASS_NAME = "FakeQuantizedLinear"


@pytest.fixture(scope="session")
def embedding_model_factory() -> Callable[..., nn.Sequential]:
    def factory(device: str, padding_idx: int | None = 0) -> nn.Sequential:
        model = nn.Sequential(
            OrderedDict(
                embedding=nn.Embedding(
                    num_embeddings=64, embedding_dim=32, padding_idx=padding_idx
                ),
                projection=nn.Linear(in_features=32, out_features=16),
            )
        ).to(device=device)
        with torch.no_grad():
            model.embedding.weight[0].fill_(0.25)  # (64, 32) -> (32,)
        return model

    return factory


@pytest.fixture(scope="session")
def embedding_qat_workflow_factory() -> Callable[..., EagerQuantizationWorkflow]:
    def factory(is_qat: bool = True) -> EagerQuantizationWorkflow:
        with initialize_config_dir(
            config_dir=str(get_hydra_configs_dir()), version_base=None
        ):
            config = compose(
                config_name="quantization/qat_int4_embeddings_and_linears",
                overrides=[f"quantization.is_qat={is_qat}"],
            )
        return hydra.utils.instantiate(config.quantization)

    return factory


@pytest.fixture
def embedding_indices_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    def factory(device: str, batch_size: int = 2) -> torch.Tensor:
        indices = rng.integers(low=1, high=64, size=(batch_size, 4), dtype=np.int64)
        indices[:, 0] = 0  # (batch_size, 4) -> (batch_size,)
        return torch.from_numpy(indices).to(device=device)  # (batch_size, 4)

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
    "qat_preset, full_language",
    [
        ("qat_int8_dynamic_intx_int4", False),
        ("qat_int4_weight_only", False),
        ("qat_int4_embeddings_and_linears", True),
    ],
    ids=[
        "int8_dynamic_activation_int4_weight",
        "int4_weight_only",
        "full_language_int4",
    ],
)
def test_language_action_transformer_qat_forward_backward(
    qat_preset: str,
    full_language: bool,
    language_action_transformer_qat_policy_factory: Callable[
        ..., tuple[Policy, EagerQuantizationWorkflow]
    ],
    language_action_transformer_batch_factory: Callable[
        [Policy, torch.device], dict[str, dict[str, torch.Tensor]]
    ],
) -> None:
    device = torch.device("cuda")
    policy, qat_workflow = language_action_transformer_qat_policy_factory(
        qat_preset=qat_preset, full_language=full_language
    )
    policy.to(device=device)
    policy.train()
    batch = language_action_transformer_batch_factory(policy=policy, device=device)

    with torch.no_grad():
        warmup_loss = policy.compute_loss(batch=batch).total_loss

    qat_workflow.prepare_model(model=policy)
    loss_output = policy.compute_loss(batch=batch)
    loss_output.total_loss.backward()

    fake_quantized_linear_count = sum(
        1 for module in policy.modules() if _is_fake_quantized_linear(module=module)
    )
    prepared_module_count = sum(
        len(prepared.module_names) for prepared in qat_workflow._prepared_targets
    )
    quantized_embeddings = [
        module
        for module in policy.modules()
        if isinstance(module, FakeQuantizedEmbedding)
    ]
    nonzero_gradient_count = _nonzero_gradient_count(module=policy)

    assert torch.isfinite(warmup_loss)
    assert torch.isfinite(loss_output.total_loss)
    assert prepared_module_count > 0
    assert (
        fake_quantized_linear_count + len(quantized_embeddings) == prepared_module_count
    )
    assert nonzero_gradient_count > 0
    if full_language:
        assert len(quantized_embeddings) == 3
        assert all(
            _nonzero_gradient_count(module=layer) == 1 for layer in quantized_embeddings
        )


@pytest.mark.integration
@pytest.mark.requires_gpu
def test_qat_prepare_filters_scoped_group_incompatible_linears(
    scoped_linear_model_factory: Callable[[], nn.Module],
    linear_input_factory: Callable[[torch.device], torch.Tensor],
) -> None:
    device = torch.device("cuda")
    model = scoped_linear_model_factory().to(device=device)
    inputs = linear_input_factory(device)
    strategy = EagerQuantizationWorkflow(
        targets=[
            EagerQuantizationModuleTarget(
                module_path="encoder",
                quantize_config=Int4WeightOnlyConfig(group_size=32),
            )
        ],
        is_qat=True,
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
    "quantize_config",
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
    ids=["int8_dynamic_activation_int4_weight", "int4_weight_only"],
)
def test_qat_convert_runs_supported_torchao_configs(
    quantize_config: Int8DynamicActivationIntxWeightConfig | IntxWeightOnlyConfig,
    scoped_linear_model_factory: Callable[[], nn.Module],
    linear_input_factory: Callable[[torch.device], torch.Tensor],
) -> None:
    device = torch.device("cuda")
    model = scoped_linear_model_factory().to(device=device)
    inputs = linear_input_factory(device)
    strategy = EagerQuantizationWorkflow(
        targets=[
            EagerQuantizationModuleTarget(
                module_path="",
                quantize_config=quantize_config,
            )
        ],
        is_qat=True,
    )

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


@pytest.mark.integration
@pytest.mark.parametrize(
    "training_device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
)
def test_embedding_qat_gradients_checkpoint_reload_and_conversion(
    embedding_model_factory: Callable[..., nn.Sequential],
    embedding_qat_workflow_factory: Callable[[], EagerQuantizationWorkflow],
    embedding_indices_factory: Callable[..., torch.Tensor],
    tmp_path: Path,
    training_device: str,
) -> None:
    model = embedding_model_factory(device=training_device)
    workflow = embedding_qat_workflow_factory()
    workflow.prepare_model(model=model)
    indices = embedding_indices_factory(device=training_device)  # (2, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial_weight = model.embedding.weight.detach().clone()  # (64, 32)
    prediction = model(indices)  # (2, 4) -> (2, 4, 16)
    prediction.square().mean().backward()  # (2, 4, 16) -> ()

    for layer in (model.embedding, model.projection):
        gradient = layer.weight.grad  # (64, 32) or (16, 32)
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum().item() > 0
    torch.testing.assert_close(
        model.embedding.weight.grad[0], torch.zeros_like(initial_weight[0])
    )  # (32,)
    optimizer.step()
    assert not torch.equal(model.embedding.weight, initial_weight)
    expected_embedding = model.embedding(indices).detach()  # (2, 4, 32)
    float_embedding = functional.embedding(
        indices, model.embedding.weight
    ).detach()  # (2, 4) -> (2, 4, 32)
    assert not torch.equal(expected_embedding, float_embedding)
    expected = model(indices).detach()  # (2, 4) -> (2, 4, 16)
    checkpoint = tmp_path / "embedding_qat.pt"
    torch.save(model.state_dict(), checkpoint)

    restored = embedding_model_factory(device=training_device)
    restored_workflow = embedding_qat_workflow_factory()
    restored_workflow.prepare_model(model=restored)
    restored.load_state_dict(
        torch.load(checkpoint, weights_only=True, map_location=training_device)
    )
    restored.eval()
    torch.testing.assert_close(restored(indices), expected)  # (2, 4, 16)
    torch.testing.assert_close(
        restored.embedding(indices), expected_embedding
    )  # (2, 4, 32)
    restored.cpu()
    cpu_indices = indices.cpu()  # (2, 4)
    expected_embedding = restored.embedding(cpu_indices).detach()  # (2, 4, 32)
    expected = restored(cpu_indices).detach()  # (2, 4, 16)
    restored_workflow.convert_model(model=restored)
    torch.testing.assert_close(
        restored.embedding(cpu_indices), expected_embedding
    )  # (2, 4, 32)
    torch.testing.assert_close(
        restored(cpu_indices), expected, atol=1e-5, rtol=1e-4
    )  # (2, 4, 16)


@pytest.mark.integration
@pytest.mark.requires_executorch
@pytest.mark.parametrize("padding_idx", [None, 0])
@pytest.mark.parametrize("is_qat", [False, True], ids=["ptq", "qat"])
def test_embedding_and_linear_export_runs_in_executorch(
    embedding_model_factory: Callable[..., nn.Sequential],
    embedding_qat_workflow_factory: Callable[..., EagerQuantizationWorkflow],
    embedding_indices_factory: Callable[..., torch.Tensor],
    tmp_path: Path,
    padding_idx: int | None,
    is_qat: bool,
) -> None:
    model = embedding_model_factory(device="cpu", padding_idx=padding_idx)
    workflow = embedding_qat_workflow_factory(is_qat=is_qat)
    indices = embedding_indices_factory(device="cpu")  # (2, 4)
    if is_qat:
        workflow.prepare_model(model=model)
        model(indices).square().mean().backward()  # (2, 4) -> ()
    model.eval()
    backend = ExecutorchXNNPACKBackend(max_batch_size=4)
    for target in workflow.targets:
        selected, _ = target.select_modules(model=model, auto_filter_incompatible=False)
        backend.validate_eager_target(
            model=model, target=target, module_names=set(selected), for_conversion=True
        )
    if is_qat:
        workflow.convert_model(model=model)
    else:
        workflow._apply_ptq(model=model, deployment_backend=backend)
    artifact = backend.export(model=model, example_inputs=(indices,))
    assert b"embedding_4bit" in artifact.model_bytes
    assert b"XnnpackBackend" in artifact.model_bytes
    assert model.embedding.padding_idx == padding_idx
    model_path = tmp_path / artifact.model_filename
    model_path.write_bytes(artifact.model_bytes)
    runtime = ExecuTorchModuleAdapter(model_path=str(model_path))

    with torch.no_grad():
        for batch_size in (1, 2, 4):
            tokens = embedding_indices_factory(
                device="cpu", batch_size=batch_size
            )  # (batch_size, 4)
            expected = model(tokens)  # (batch_size, 4) -> (batch_size, 4, 16)
            actual = runtime(observation_tensors=(tokens,))[0]  # (batch_size, 4, 16)
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
