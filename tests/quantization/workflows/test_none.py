"""Tests for versatil.quantization.workflows.none module."""

from unittest.mock import MagicMock, patch

import pytest

from versatil.post_training_compression.constants import QuantizationWorkflow
from versatil.quantization.constants import QuantizationMode
from versatil.quantization.workflows.base import BaseQuantizationWorkflow
from versatil.quantization.workflows.none import NoQuantizationWorkflow

NO_WORKFLOW_MODULE = "versatil.quantization.workflows.none"


@pytest.mark.unit
class TestNoQuantizationWorkflow:
    def test_properties_identify_no_quantization(self):
        workflow = NoQuantizationWorkflow()

        assert isinstance(workflow, BaseQuantizationWorkflow)
        assert workflow.quantization_mode == QuantizationMode.NONE.value
        assert workflow.quantization_workflow == QuantizationWorkflow.NONE.value
        assert workflow.is_qat is False

    def test_prepare_model_leaves_model_unchanged(self):
        workflow = NoQuantizationWorkflow()
        model = MagicMock()

        workflow.prepare_model(model=model)

        model.assert_not_called()

    def test_load_policy_context_delegates_to_float_loader(self):
        workflow = NoQuantizationWorkflow()
        expected_context = MagicMock()

        with patch(
            f"{NO_WORKFLOW_MODULE}.load_float_policy_context",
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

    def test_quantize_exports_float_model(self):
        workflow = NoQuantizationWorkflow()
        context = MagicMock()
        context.observation_space = MagicMock()
        context.observation_horizon = 2
        context.tokenizer = MagicMock()
        exportable = MagicMock()
        example_inputs = (MagicMock(),)
        exported = MagicMock()

        with (
            patch(
                f"{NO_WORKFLOW_MODULE}.build_example_inputs",
                return_value=example_inputs,
            ) as mock_build_inputs,
            patch(
                f"{NO_WORKFLOW_MODULE}.export_policy",
                return_value=exported,
            ) as mock_export,
        ):
            result = workflow.quantize(
                context=context,
                exportable=exportable,
                modules=[],
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
        assert result.float_model is exported
        assert result.quantized_model is exported
        assert result.example_inputs is example_inputs
        assert result.quantization_workflow == QuantizationWorkflow.NONE.value
