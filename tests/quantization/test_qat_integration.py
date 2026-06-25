"""Integration tests for torchao QAT on real VersatIL policies."""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationIntxWeightConfig,
    IntxWeightOnlyConfig,
    PerGroup,
)

from versatil.models.policy import Policy
from versatil.quantization.workflows.eager import EagerQuantizationWorkflow

FAKE_QUANTIZED_LINEAR_CLASS_NAME = "FakeQuantizedLinear"


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
    ids=["int8_dynamic_activation_int4_weight", "int4_weight_only"],
)
def test_language_action_transformer_qat_forward_backward(
    qat_preset: str,
    language_action_transformer_qat_policy_factory: Callable[
        [str], tuple[Policy, EagerQuantizationWorkflow]
    ],
    language_action_transformer_batch_factory: Callable[
        [Policy, torch.device], dict[str, dict[str, torch.Tensor]]
    ],
) -> None:
    device = torch.device("cuda")
    policy, qat_workflow = language_action_transformer_qat_policy_factory(qat_preset)
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
    nonzero_gradient_count = _nonzero_gradient_count(module=policy)

    assert torch.isfinite(warmup_loss)
    assert torch.isfinite(loss_output.total_loss)
    assert len(qat_workflow._prepared_module_names) > 0
    assert fake_quantized_linear_count == len(qat_workflow._prepared_module_names)
    assert nonzero_gradient_count > 0


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
        quantize_config=Int4WeightOnlyConfig(group_size=32),
        is_qat=True,
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
    strategy = EagerQuantizationWorkflow(quantize_config=quantize_config, is_qat=True)

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
