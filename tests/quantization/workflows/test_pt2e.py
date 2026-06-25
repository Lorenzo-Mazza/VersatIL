"""Tests for versatil.quantization.workflows.pt2e module."""

import re
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import pytest
import torch.nn as nn

from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.post_training_compression.constants import QuantizationWorkflow
from versatil.quantization.constants import QuantizationMode
from versatil.quantization.workflows.base import BaseQuantizationWorkflow
from versatil.quantization.workflows.pt2e import PT2EQuantizationWorkflow

PT2E_WORKFLOW_MODULE = "versatil.quantization.workflows.pt2e"


@pytest.fixture
def compression_target_factory(mock_pt2e_backend_factory):
    """Factory for CompressionTarget with PT2E quantization."""

    def factory(
        module_path: str = "",
        needs_calibration: bool = False,
    ) -> CompressionTarget:
        backend = mock_pt2e_backend_factory(is_dynamic=not needs_calibration)
        workflow = PT2EQuantizationWorkflow(pt2e_backend=backend)
        return CompressionTarget(
            module_path=module_path,
            quantization=workflow,
        )

    return factory


@pytest.fixture
def pt2e_mocks():
    """Patch all external dependencies of PT2E conversion."""
    with (
        patch(f"{PT2E_WORKFLOW_MODULE}.convert_pt2e") as mock_convert,
        patch(f"{PT2E_WORKFLOW_MODULE}.prepare_pt2e") as mock_prepare,
        patch(f"{PT2E_WORKFLOW_MODULE}.ComposableQuantizer") as mock_composer,
    ):
        mock_convert.return_value = MagicMock()
        mock_convert.return_value.graph = MagicMock()
        mock_convert.return_value.graph.__str__ = MagicMock(return_value="")
        yield {
            "convert": mock_convert,
            "prepare": mock_prepare,
            "composer": mock_composer,
        }


@pytest.mark.unit
class TestPT2EQuantizationWorkflow:
    @pytest.mark.parametrize("is_dynamic", [True, False])
    def test_needs_calibration_reflects_dynamic_flag(
        self,
        mock_pt2e_backend_factory,
        is_dynamic,
    ):
        backend = mock_pt2e_backend_factory(is_dynamic=is_dynamic)
        workflow = PT2EQuantizationWorkflow(pt2e_backend=backend)

        assert workflow.needs_calibration == (not is_dynamic)

    def test_backend_accessible_via_property(self, mock_pt2e_backend_factory):
        backend = mock_pt2e_backend_factory(is_dynamic=True)
        workflow = PT2EQuantizationWorkflow(pt2e_backend=backend)

        assert isinstance(workflow, BaseQuantizationWorkflow)
        assert workflow.pt2e_backend.is_dynamic is True
        assert workflow.quantization_mode == QuantizationMode.PT2E.value

    def test_load_policy_context_delegates_to_float_loader(
        self,
        mock_pt2e_backend_factory,
    ):
        workflow = PT2EQuantizationWorkflow(pt2e_backend=mock_pt2e_backend_factory())
        expected_context = MagicMock()

        with patch(
            f"{PT2E_WORKFLOW_MODULE}.load_float_policy_context",
            return_value=expected_context,
        ) as mock_loader:
            result = workflow.load_policy_context(
                checkpoint_path="/tmp/checkpoint",
                checkpoint_name="last.ckpt",
            )

        mock_loader.assert_called_once_with(
            checkpoint_path="/tmp/checkpoint",
            checkpoint_name="last.ckpt",
        )
        assert result is expected_context

    def test_quantize_exports_and_converts_context(
        self,
        mock_pt2e_backend_factory,
        compression_target_factory,
    ):
        workflow = PT2EQuantizationWorkflow(pt2e_backend=mock_pt2e_backend_factory())
        context = MagicMock()
        context.observation_space = MagicMock()
        context.observation_horizon = 2
        context.tokenizer = MagicMock()
        exportable = MagicMock()
        modules = [compression_target_factory(needs_calibration=False)]
        example_inputs = (MagicMock(),)
        exported = MagicMock(spec=nn.Module)
        converted = MagicMock(spec=nn.Module)

        with (
            patch.object(
                PT2EQuantizationWorkflow,
                "_build_calibration",
                return_value=None,
            ) as mock_build_calibration,
            patch(
                f"{PT2E_WORKFLOW_MODULE}.build_example_inputs",
                return_value=example_inputs,
            ) as mock_build_inputs,
            patch(
                f"{PT2E_WORKFLOW_MODULE}.export_policy",
                return_value=exported,
            ) as mock_export,
            patch.object(
                PT2EQuantizationWorkflow,
                "_convert_exported_model",
                return_value=converted,
            ) as mock_convert,
        ):
            result = workflow.quantize(
                context=context,
                exportable=exportable,
                modules=modules,
                calibration_steps=8,
            )

        mock_build_calibration.assert_called_once_with(
            context=context,
            exportable=exportable,
            pt2e_modules=modules,
            calibration_steps=8,
        )
        mock_build_inputs.assert_called_once_with(
            exportable=exportable,
            observation_space=context.observation_space,
            observation_horizon=context.observation_horizon,
            tokenizer=context.tokenizer,
        )
        mock_export.assert_called_once_with(
            exportable=exportable,
            example_inputs=example_inputs,
        )
        mock_convert.assert_called_once_with(
            exported=exported,
            pt2e_modules=modules,
            calibration=None,
        )
        assert result.float_model is exported
        assert result.quantized_model is converted
        assert result.example_inputs is example_inputs
        assert result.quantization_workflow == QuantizationWorkflow.PT2E.value

    def test_quantize_uses_calibration_batch_when_available(
        self,
        mock_pt2e_backend_factory,
        compression_target_factory,
    ):
        workflow = PT2EQuantizationWorkflow(pt2e_backend=mock_pt2e_backend_factory())
        context = MagicMock()
        context.observation_space = MagicMock()
        context.observation_horizon = 2
        context.tokenizer = MagicMock()
        exportable = MagicMock()
        modules = [compression_target_factory(needs_calibration=True)]
        calibration = MagicMock()
        example_inputs = (MagicMock(),)
        calibration.get_single_batch.return_value = example_inputs
        exported = MagicMock(spec=nn.Module)
        converted = MagicMock(spec=nn.Module)

        with (
            patch.object(
                PT2EQuantizationWorkflow,
                "_build_calibration",
                return_value=calibration,
            ) as mock_build_calibration,
            patch(f"{PT2E_WORKFLOW_MODULE}.build_example_inputs") as mock_build_inputs,
            patch(
                f"{PT2E_WORKFLOW_MODULE}.export_policy",
                return_value=exported,
            ) as mock_export,
            patch.object(
                PT2EQuantizationWorkflow,
                "_convert_exported_model",
                return_value=converted,
            ) as mock_convert,
        ):
            result = workflow.quantize(
                context=context,
                exportable=exportable,
                modules=modules,
                calibration_steps=8,
            )

        mock_build_calibration.assert_called_once_with(
            context=context,
            exportable=exportable,
            pt2e_modules=modules,
            calibration_steps=8,
        )
        calibration.get_single_batch.assert_called_once_with()
        mock_build_inputs.assert_not_called()
        mock_export.assert_called_once_with(
            exportable=exportable,
            example_inputs=example_inputs,
        )
        mock_convert.assert_called_once_with(
            exported=exported,
            pt2e_modules=modules,
            calibration=calibration,
        )
        assert result.quantized_model is converted
        assert result.example_inputs is example_inputs

    def test_pt2e_qat_raises_on_init(self, mock_pt2e_backend_factory):
        with pytest.raises(
            NotImplementedError,
            match=re.escape("PT2E QAT configuration is not supported yet."),
        ):
            PT2EQuantizationWorkflow(
                pt2e_backend=mock_pt2e_backend_factory(is_qat=True),
            )

    def test_prepare_model_raises_as_unsupported(self, mock_pt2e_backend_factory):
        workflow = PT2EQuantizationWorkflow(pt2e_backend=mock_pt2e_backend_factory())

        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                "PT2EQuantizationWorkflow does not support QAT preparation."
            ),
        ):
            workflow.prepare_model(model=MagicMock(spec=nn.Module))

    def test_empty_pt2e_modules_returns_exported_unchanged(self):
        exported = MagicMock(spec=nn.Module)

        result = PT2EQuantizationWorkflow._convert_exported_model(
            exported=exported,
            pt2e_modules=[],
            calibration=None,
        )

        assert result is exported

    def test_build_calibration_returns_none_for_dynamic_targets(
        self,
        compression_target_factory,
    ):
        target = compression_target_factory(needs_calibration=False)

        with patch(f"{PT2E_WORKFLOW_MODULE}.get_dataloaders") as mock_dataloaders:
            result = PT2EQuantizationWorkflow._build_calibration(
                context=MagicMock(),
                exportable=MagicMock(),
                pt2e_modules=[target],
                calibration_steps=8,
            )

        mock_dataloaders.assert_not_called()
        assert result is None

    def test_build_calibration_uses_training_dataloader_for_static_targets(
        self,
        compression_target_factory,
    ):
        target = compression_target_factory(needs_calibration=True)
        context = MagicMock()
        context.config = MagicMock()
        exportable = MagicMock()
        exportable.observation_keys = ["left", "depth"]
        train_loader = MagicMock()
        expected_provider = MagicMock()

        with (
            patch(
                f"{PT2E_WORKFLOW_MODULE}.get_dataloaders",
                return_value=(train_loader, None, None, None, None),
            ) as mock_dataloaders,
            patch(
                f"{PT2E_WORKFLOW_MODULE}.CalibrationDataProvider",
                return_value=expected_provider,
            ) as mock_provider,
        ):
            result = PT2EQuantizationWorkflow._build_calibration(
                context=context,
                exportable=exportable,
                pt2e_modules=[target],
                calibration_steps=8,
            )

        mock_dataloaders.assert_called_once_with(config=context.config)
        mock_provider.assert_called_once_with(
            dataloader=train_loader,
            observation_keys=exportable.observation_keys,
            num_calibration_steps=8,
        )
        assert result is expected_provider

    @pytest.mark.parametrize(
        "needs_calibration, has_calibration, expectation",
        [
            (
                True,
                False,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "PT2E static quantization requires calibration data "
                        "but no CalibrationDataProvider was supplied."
                    ),
                ),
            ),
            (True, True, does_not_raise()),
            (False, False, does_not_raise()),
        ],
    )
    def test_pt2e_calibration_validation(
        self,
        compression_target_factory,
        pt2e_mocks,
        needs_calibration,
        has_calibration,
        expectation,
    ):
        target = compression_target_factory(needs_calibration=needs_calibration)
        calibration = MagicMock() if has_calibration else None

        with expectation:
            PT2EQuantizationWorkflow._convert_exported_model(
                exported=MagicMock(spec=nn.Module),
                pt2e_modules=[target],
                calibration=calibration,
            )

    def test_pt2e_uses_composable_quantizer(
        self,
        compression_target_factory,
        pt2e_mocks,
    ):
        target = compression_target_factory(module_path="encoder")

        PT2EQuantizationWorkflow._convert_exported_model(
            exported=MagicMock(spec=nn.Module),
            pt2e_modules=[target],
            calibration=None,
        )

        target.quantization.pt2e_backend.create_quantizer.assert_called_once_with(
            module_path="encoder",
        )
        pt2e_mocks["composer"].assert_called_once()
        pt2e_mocks["prepare"].assert_called_once()
        pt2e_mocks["convert"].assert_called_once()
