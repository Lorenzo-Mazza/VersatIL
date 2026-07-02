"""Tests for versatil.quantization.workflows.eager module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torchao.quantization import (
    Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig,
    Int8DynamicActivationIntxWeightConfig,
    PerGroup,
)

from versatil.post_training_compression.constants import QuantizationWorkflow
from versatil.quantization.constants import QuantizationMode
from versatil.quantization.module_target import EagerQuantizationModuleTarget
from versatil.quantization.workflows.base import BaseQuantizationWorkflow
from versatil.quantization.workflows.eager import EagerQuantizationWorkflow

EAGER_WORKFLOW_MODULE = "versatil.quantization.workflows.eager"


class PolicyWithLinearModules(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(32, 16), nn.Linear(8, 8))


@pytest.fixture
def eager_target_factory() -> Callable[..., EagerQuantizationModuleTarget]:
    """Factory for eager quantization module targets."""

    def factory(
        module_path: str = "",
        quantize_config: MagicMock | Int4WeightOnlyConfig | None = None,
    ) -> EagerQuantizationModuleTarget:
        config = quantize_config if quantize_config is not None else MagicMock(spec=[])
        return EagerQuantizationModuleTarget(
            module_path=module_path,
            quantize_config=config,
        )

    return factory


@pytest.fixture
def policy_with_linear_modules_factory() -> Callable[[], PolicyWithLinearModules]:
    """Factory for small policies with eligible and ineligible Linear modules."""

    def factory() -> PolicyWithLinearModules:
        return PolicyWithLinearModules()

    return factory


@pytest.fixture
def policy_context_factory() -> Callable[..., MagicMock]:
    """Factory for workflow policy contexts."""

    def factory() -> MagicMock:
        context = MagicMock()
        context.policy = MagicMock(spec=nn.Module)
        context.observation_space = MagicMock()
        context.observation_horizon = 2
        context.tokenizer = MagicMock()
        return context

    return factory


@pytest.fixture
def export_mocks_factory() -> Callable[..., dict[str, MagicMock | tuple[MagicMock]]]:
    """Factory for mocked export collaborators."""

    def factory() -> dict[str, MagicMock | tuple[MagicMock]]:
        return {
            "exportable": MagicMock(),
            "example_inputs": (MagicMock(),),
            "exported": MagicMock(spec=nn.Module),
            "quantized": MagicMock(spec=nn.Module),
        }

    return factory


@pytest.mark.unit
class TestEagerQuantizationWorkflow:
    def test_requires_at_least_one_target(self):
        with pytest.raises(
            ValueError,
            match=re.escape("EagerQuantizationWorkflow requires at least one target."),
        ):
            EagerQuantizationWorkflow(targets=[])

    def test_ptq_config_accessible_via_attribute(self):
        config = Int8DynamicActivationInt8WeightConfig()
        target = EagerQuantizationModuleTarget(
            module_path="",
            quantize_config=config,
        )
        workflow = EagerQuantizationWorkflow(targets=[target])

        assert workflow.targets == [target]
        assert isinstance(
            workflow.targets[0].quantize_config,
            Int8DynamicActivationInt8WeightConfig,
        )
        assert isinstance(workflow, BaseQuantizationWorkflow)
        assert workflow.is_qat is False
        assert workflow.quantization_mode == QuantizationMode.EAGER.value

    def test_qat_config_accessible_via_attribute(self):
        config = Int4WeightOnlyConfig(group_size=32)
        target = EagerQuantizationModuleTarget(
            module_path="decoder",
            quantize_config=config,
        )
        workflow = EagerQuantizationWorkflow(
            targets=[target],
            is_qat=True,
            auto_filter_incompatible_linears=False,
        )

        assert workflow.targets == [target]
        assert workflow.auto_filter_incompatible_linears is False
        assert workflow.quantization_mode == QuantizationMode.EAGER.value

    @pytest.mark.parametrize(
        "is_qat",
        [
            False,
            True,
        ],
        ids=["ptq_float_context", "qat_prepared_context"],
    )
    def test_load_policy_context_dispatches_by_qat_flag(
        self,
        eager_target_factory,
        is_qat,
    ):
        workflow = EagerQuantizationWorkflow(
            targets=[eager_target_factory()],
            is_qat=is_qat,
        )
        float_context = MagicMock()
        qat_context = MagicMock()
        expected_context = qat_context if is_qat else float_context

        with (
            patch(
                f"{EAGER_WORKFLOW_MODULE}.load_float_policy_context"
            ) as mock_float_context_loader,
            patch(
                f"{EAGER_WORKFLOW_MODULE}.load_qat_policy_context"
            ) as mock_qat_context_loader,
        ):
            mock_float_context_loader.return_value = float_context
            mock_qat_context_loader.return_value = qat_context

            result = workflow.load_policy_context(
                checkpoint_path="/tmp/checkpoint",
                checkpoint_name="last.ckpt",
            )

        if is_qat:
            mock_qat_context_loader.assert_called_once_with(
                checkpoint_path="/tmp/checkpoint",
                checkpoint_name="last.ckpt",
                quantization=workflow,
            )
            mock_float_context_loader.assert_not_called()
        else:
            mock_float_context_loader.assert_called_once_with(
                checkpoint_path="/tmp/checkpoint",
                checkpoint_name="last.ckpt",
            )
            mock_qat_context_loader.assert_not_called()
        assert result is expected_context

    def test_quantize_applies_ptq_and_exports_context(
        self,
        policy_context_factory,
        export_mocks_factory,
        eager_target_factory,
    ):
        workflow = EagerQuantizationWorkflow(targets=[eager_target_factory()])
        context = policy_context_factory()
        export_mocks = export_mocks_factory()

        with (
            patch.object(EagerQuantizationWorkflow, "_apply_ptq") as mock_apply_ptq,
            patch(
                f"{EAGER_WORKFLOW_MODULE}.build_example_inputs",
                return_value=export_mocks["example_inputs"],
            ) as mock_build_inputs,
            patch(
                f"{EAGER_WORKFLOW_MODULE}.export_policy",
                side_effect=[export_mocks["exported"], export_mocks["quantized"]],
            ) as mock_export,
        ):
            result = workflow.quantize(
                context=context,
                exportable=export_mocks["exportable"],
                calibration_steps=8,
            )

        mock_apply_ptq.assert_called_once_with(model=context.policy)
        mock_build_inputs.assert_called_once_with(
            exportable=export_mocks["exportable"],
            observation_space=context.observation_space,
            observation_horizon=context.observation_horizon,
            tokenizer=context.tokenizer,
        )
        # The float baseline must be exported BEFORE quantize_() mutates the
        # policy, so the report does not compare the quantized model to itself.
        assert mock_export.call_count == 2
        assert mock_export.call_args_list[0] == mock_export.call_args_list[1]
        assert result.float_model is export_mocks["exported"]
        assert result.quantized_model is export_mocks["quantized"]
        assert result.float_model is not result.quantized_model
        assert result.example_inputs is export_mocks["example_inputs"]
        assert result.quantization_workflow == QuantizationWorkflow.EAGER.value

    def test_quantize_converts_qat_model_before_export(
        self,
        policy_context_factory,
        export_mocks_factory,
        eager_target_factory,
    ):
        workflow = EagerQuantizationWorkflow(
            targets=[eager_target_factory()],
            is_qat=True,
        )
        context = policy_context_factory()
        export_mocks = export_mocks_factory()

        with (
            patch.object(workflow, "convert_model") as mock_convert,
            patch(
                f"{EAGER_WORKFLOW_MODULE}.build_example_inputs",
                return_value=export_mocks["example_inputs"],
            ) as mock_build_inputs,
            patch(
                f"{EAGER_WORKFLOW_MODULE}.export_policy",
                side_effect=[export_mocks["exported"], export_mocks["quantized"]],
            ) as mock_export,
        ):
            result = workflow.quantize(
                context=context,
                exportable=export_mocks["exportable"],
                calibration_steps=8,
            )

        mock_convert.assert_called_once_with(model=context.policy)
        mock_build_inputs.assert_called_once_with(
            exportable=export_mocks["exportable"],
            observation_space=context.observation_space,
            observation_horizon=context.observation_horizon,
            tokenizer=context.tokenizer,
        )
        assert mock_export.call_count == 2
        assert result.float_model is export_mocks["exported"]
        assert result.quantized_model is export_mocks["quantized"]
        assert result.quantization_workflow == QuantizationWorkflow.EAGER.value

    def test_ptq_applies_quantize_to_root(
        self,
        policy_with_linear_modules_factory,
        eager_target_factory,
    ):
        model = policy_with_linear_modules_factory()
        target = eager_target_factory(module_path="")
        workflow = EagerQuantizationWorkflow(targets=[target])

        with patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize_mock:
            workflow._apply_ptq(model=model)

        quantize_mock.assert_called_once()
        call_kwargs = quantize_mock.call_args.kwargs
        assert call_kwargs["model"] is model
        assert call_kwargs["config"] is target.quantize_config
        assert call_kwargs["filter_fn"](model.encoder[0], "encoder.0") is True

    def test_ptq_applies_quantize_to_scoped_modules(
        self,
        policy_with_linear_modules_factory,
        eager_target_factory,
    ):
        model = policy_with_linear_modules_factory()
        target = eager_target_factory(module_path="decoder")
        workflow = EagerQuantizationWorkflow(targets=[target])

        with patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize_mock:
            workflow._apply_ptq(model=model)

        call_kwargs = quantize_mock.call_args.kwargs
        assert call_kwargs["model"] is model
        assert call_kwargs["config"] is target.quantize_config
        filter_fn = call_kwargs["filter_fn"]
        assert filter_fn(model.decoder[0], "decoder.0") is True
        assert filter_fn(model.encoder[0], "encoder.0") is False

    def test_ptq_applies_each_target_config(
        self,
        policy_with_linear_modules_factory,
        eager_target_factory,
    ):
        model = policy_with_linear_modules_factory()
        encoder_target = eager_target_factory(module_path="encoder")
        decoder_target = eager_target_factory(module_path="decoder")
        workflow = EagerQuantizationWorkflow(
            targets=[encoder_target, decoder_target],
        )

        with patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize_mock:
            workflow._apply_ptq(model=model)

        assert quantize_mock.call_count == 2
        assert quantize_mock.call_args_list[0].kwargs["config"] is (
            encoder_target.quantize_config
        )
        assert quantize_mock.call_args_list[1].kwargs["config"] is (
            decoder_target.quantize_config
        )

    def test_ptq_filters_group_incompatible_linears(
        self,
        policy_with_linear_modules_factory,
    ):
        model = policy_with_linear_modules_factory()
        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(32),
        )
        target = EagerQuantizationModuleTarget(
            module_path="",
            quantize_config=config,
        )
        workflow = EagerQuantizationWorkflow(targets=[target])

        with patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize_mock:
            workflow._apply_ptq(model=model)

        filter_fn = quantize_mock.call_args.kwargs["filter_fn"]
        assert filter_fn(model.encoder[0], "encoder.0") is True
        assert filter_fn(model.decoder[0], "decoder.0") is True
        assert filter_fn(model.decoder[1], "decoder.1") is False

    def test_ptq_keeps_incompatible_linears_when_filter_disabled(
        self,
        policy_with_linear_modules_factory,
    ):
        model = policy_with_linear_modules_factory()
        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(32),
        )
        target = EagerQuantizationModuleTarget(
            module_path="",
            quantize_config=config,
        )
        workflow = EagerQuantizationWorkflow(
            targets=[target],
            auto_filter_incompatible_linears=False,
        )

        with patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize_mock:
            workflow._apply_ptq(model=model)

        filter_fn = quantize_mock.call_args.kwargs["filter_fn"]
        assert filter_fn(model.decoder[1], "decoder.1") is True

    def test_ptq_does_not_raise_when_no_linear_is_eligible(self):
        model = nn.Sequential(nn.Linear(8, 8))
        target = EagerQuantizationModuleTarget(
            module_path="",
            quantize_config=Int4WeightOnlyConfig(group_size=32),
        )
        workflow = EagerQuantizationWorkflow(targets=[target])

        with patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize_mock:
            workflow._apply_ptq(model=model)

        filter_fn = quantize_mock.call_args.kwargs["filter_fn"]
        assert filter_fn(model[0], "0") is False

    def test_prepare_calls_quantize_with_qat_prepare_config(
        self,
        policy_with_linear_modules_factory,
    ):
        model = policy_with_linear_modules_factory()
        config = Int4WeightOnlyConfig(group_size=32)
        target = EagerQuantizationModuleTarget(
            module_path="decoder",
            quantize_config=config,
        )
        workflow = EagerQuantizationWorkflow(
            targets=[target],
            is_qat=True,
        )

        with patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize_mock:
            workflow.prepare_model(model=model)

        call_kwargs = quantize_mock.call_args.kwargs
        assert call_kwargs["model"] is model
        assert call_kwargs["config"].base_config is config
        assert call_kwargs["config"].step == "prepare"
        filter_fn = call_kwargs["filter_fn"]
        assert filter_fn(model.decoder[0], "decoder.0") is True
        assert filter_fn(model.encoder[0], "encoder.0") is False
        assert filter_fn(model.decoder[1], "decoder.1") is False

    def test_prepare_filters_group_incompatible_linears(
        self,
        policy_with_linear_modules_factory,
    ):
        model = policy_with_linear_modules_factory()
        config = Int8DynamicActivationIntxWeightConfig(
            weight_dtype=torch.int4,
            weight_granularity=PerGroup(32),
        )
        target = EagerQuantizationModuleTarget(
            module_path="",
            quantize_config=config,
        )
        workflow = EagerQuantizationWorkflow(targets=[target], is_qat=True)

        with patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize_mock:
            workflow.prepare_model(model=model)

        filter_fn = quantize_mock.call_args.kwargs["filter_fn"]
        assert filter_fn(model.encoder[0], "encoder.0") is True
        assert filter_fn(model.decoder[0], "decoder.0") is True
        assert filter_fn(model.decoder[1], "decoder.1") is False

    def test_prepare_raises_when_module_path_is_missing(
        self,
        policy_with_linear_modules_factory,
    ):
        model = policy_with_linear_modules_factory()
        target = EagerQuantizationModuleTarget(
            module_path="missing",
            quantize_config=Int4WeightOnlyConfig(group_size=32),
        )
        workflow = EagerQuantizationWorkflow(
            targets=[target],
            is_qat=True,
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Quantization target 'missing' not found in model. "
                "Available top-level modules: ['encoder', 'decoder']."
            ),
        ):
            workflow.prepare_model(model=model)

    def test_prepare_raises_when_no_linear_is_eligible(self):
        model = nn.Sequential(nn.Linear(8, 8))
        target = EagerQuantizationModuleTarget(
            module_path="",
            quantize_config=Int4WeightOnlyConfig(group_size=32),
        )
        workflow = EagerQuantizationWorkflow(
            targets=[target],
            is_qat=True,
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "QAT target '(root)' selected zero eligible nn.Linear modules. "
                "Skipped modules: 0: in_features 8 is not divisible by group_size 32."
            ),
        ):
            workflow.prepare_model(model=model)

    def test_convert_requires_prepare_first(self):
        target = EagerQuantizationModuleTarget(
            module_path="",
            quantize_config=Int4WeightOnlyConfig(group_size=32),
        )
        workflow = EagerQuantizationWorkflow(
            targets=[target],
            is_qat=True,
        )

        with pytest.raises(
            ValueError,
            match=re.escape("QAT convert_model() requires prepare_model() first."),
        ):
            workflow.convert_model(model=MagicMock(spec=nn.Module))

    def test_prepare_requires_qat(self):
        workflow = EagerQuantizationWorkflow(
            targets=[
                EagerQuantizationModuleTarget(
                    module_path="",
                    quantize_config=MagicMock(spec=[]),
                )
            ]
        )

        with pytest.raises(
            ValueError,
            match=re.escape("prepare_model() requires is_qat=True."),
        ):
            workflow.prepare_model(model=MagicMock(spec=nn.Module))

    def test_convert_calls_quantize_with_qat_convert_config(
        self,
        policy_with_linear_modules_factory,
    ):
        model = policy_with_linear_modules_factory()
        config = Int4WeightOnlyConfig(group_size=32)
        target = EagerQuantizationModuleTarget(
            module_path="",
            quantize_config=config,
        )
        workflow = EagerQuantizationWorkflow(targets=[target], is_qat=True)

        with patch(f"{EAGER_WORKFLOW_MODULE}.quantize_"):
            workflow.prepare_model(model=model)
        with patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize_mock:
            workflow.convert_model(model=model)

        call_kwargs = quantize_mock.call_args.kwargs
        assert call_kwargs["model"] is model
        assert call_kwargs["config"].base_config is config
        assert call_kwargs["config"].step == "convert"
        filter_fn = call_kwargs["filter_fn"]
        assert filter_fn(model.encoder[0], "encoder.0") is True
        assert filter_fn(model.decoder[0], "decoder.0") is True
        assert filter_fn(model.decoder[1], "decoder.1") is False
