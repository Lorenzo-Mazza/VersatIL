"""Tests for versatil.quantization.workflows.pt2e module."""

import re
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, call, patch

import pytest
import torch
import torch.nn as nn

from versatil.models.exportable.metadata import PolicyExportMetadata
from versatil.post_training_compression.constants import QuantizationWorkflow
from versatil.quantization.constants import PT2EBackendName, QuantizationMode
from versatil.quantization.module_target import PT2EQuantizationModuleTarget
from versatil.quantization.workflows.base import BaseQuantizationWorkflow
from versatil.quantization.workflows.pt2e import PT2EQuantizationWorkflow

PT2E_WORKFLOW_MODULE = "versatil.quantization.workflows.pt2e"


@pytest.fixture
def pt2e_target_factory(
    mock_pt2e_backend_factory: Callable[..., MagicMock],
) -> Callable[..., MagicMock]:
    def factory(
        module_path: str = "",
        needs_calibration: bool = False,
    ) -> MagicMock:
        backend = mock_pt2e_backend_factory(is_dynamic=not needs_calibration)
        target = MagicMock(spec=PT2EQuantizationModuleTarget)
        target.module_path = module_path
        target.label = module_path or "(root)"
        target.pt2e_backend = backend
        target.needs_calibration = needs_calibration
        return target

    return factory


@pytest.fixture
def pt2e_mocks_factory(
    pt2e_export_metadata_factory: Callable[..., MagicMock],
) -> Iterator[Callable[[], dict[str, MagicMock]]]:
    with (
        patch(f"{PT2E_WORKFLOW_MODULE}.convert_pt2e") as mock_convert,
        patch(f"{PT2E_WORKFLOW_MODULE}.prepare_pt2e") as mock_prepare,
        patch(f"{PT2E_WORKFLOW_MODULE}.ComposableQuantizer") as mock_composer,
        patch(
            f"{PT2E_WORKFLOW_MODULE}.PolicyExportMetadata",
            return_value=pt2e_export_metadata_factory(noise_inputs=()),
        ) as mock_metadata,
    ):
        mock_convert.return_value = MagicMock()
        mock_convert.return_value.graph = MagicMock()
        mock_convert.return_value.graph.__str__ = MagicMock(return_value="")

        def factory() -> dict[str, MagicMock]:
            return {
                "convert": mock_convert,
                "prepare": mock_prepare,
                "composer": mock_composer,
                "metadata": mock_metadata,
            }

        yield factory


@pytest.fixture
def calibration_noise_factory() -> Callable[..., tuple[torch.Tensor, ...]]:
    def factory(batch_size: int, include_step_noise: bool) -> tuple[torch.Tensor, ...]:
        noise_inputs = (
            torch.zeros(batch_size, 4, 3),  # (batch, action_horizon, action_dimension)
        )
        if include_step_noise:
            noise_inputs += (
                torch.zeros(
                    batch_size, 3, 4, 3
                ),  # (batch, steps, action_horizon, action_dimension)
            )
        return noise_inputs

    return factory


@pytest.fixture
def pt2e_export_metadata_factory() -> Callable[..., MagicMock]:
    def factory(noise_inputs: tuple[torch.Tensor, ...]) -> MagicMock:
        export_metadata = MagicMock(spec=PolicyExportMetadata)

        def prepare_inputs(
            observations: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor, ...]:
            return observations + noise_inputs

        export_metadata.prepare_inputs.side_effect = prepare_inputs
        return export_metadata

    return factory


@pytest.mark.unit
class TestPT2EQuantizationWorkflow:
    def test_requires_at_least_one_target(self):
        with pytest.raises(
            ValueError,
            match=re.escape("PT2EQuantizationWorkflow requires at least one target."),
        ):
            PT2EQuantizationWorkflow(targets=[])

    @pytest.mark.parametrize("is_dynamic", [True, False])
    def test_needs_calibration_reflects_dynamic_flag(
        self,
        mock_pt2e_backend_factory,
        is_dynamic,
    ):
        backend = mock_pt2e_backend_factory(is_dynamic=is_dynamic)
        target = PT2EQuantizationModuleTarget(module_path="", pt2e_backend=backend)
        workflow = PT2EQuantizationWorkflow(targets=[target])

        assert workflow.needs_calibration == (not is_dynamic)

    def test_backend_accessible_via_property(self, mock_pt2e_backend_factory):
        backend = mock_pt2e_backend_factory(is_dynamic=True)
        target = PT2EQuantizationModuleTarget(module_path="", pt2e_backend=backend)
        workflow = PT2EQuantizationWorkflow(targets=[target])

        assert isinstance(workflow, BaseQuantizationWorkflow)
        assert workflow.pt2e_backend.is_dynamic is True
        assert workflow.quantization_mode == QuantizationMode.PT2E.value

    def test_backend_names_reflect_targets(self, mock_pt2e_backend_factory) -> None:
        x86_backend = mock_pt2e_backend_factory(is_dynamic=True)
        xnnpack_backend = mock_pt2e_backend_factory(is_dynamic=True)
        xnnpack_backend.name = PT2EBackendName.XNNPACK.value
        targets = [
            PT2EQuantizationModuleTarget(
                module_path="encoder",
                pt2e_backend=x86_backend,
            ),
            PT2EQuantizationModuleTarget(
                module_path="decoder",
                pt2e_backend=xnnpack_backend,
            ),
        ]
        workflow = PT2EQuantizationWorkflow(targets=targets)

        assert workflow.pt2e_backend_names == (
            PT2EBackendName.X86_INDUCTOR.value,
            PT2EBackendName.XNNPACK.value,
        )

    def test_load_policy_context_delegates_to_float_context_loader(
        self,
        mock_pt2e_backend_factory,
    ):
        target = PT2EQuantizationModuleTarget(
            module_path="",
            pt2e_backend=mock_pt2e_backend_factory(),
        )
        workflow = PT2EQuantizationWorkflow(targets=[target])
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
        pt2e_target_factory: Callable[..., PT2EQuantizationModuleTarget],
        calibration_context_factory: Callable[..., MagicMock],
        export_mocks_factory: Callable[[], dict[str, MagicMock | tuple[MagicMock]]],
    ) -> None:
        context = calibration_context_factory(batch_size=2, observation_horizon=2)
        exports = export_mocks_factory()
        exportable = exports["exportable"]
        targets = [pt2e_target_factory(module_path="", needs_calibration=False)]
        workflow = PT2EQuantizationWorkflow(targets=targets)
        example_inputs = exports["example_inputs"]
        exported = exports["exported"]
        converted = exports["quantized"]

        with (
            patch(
                f"{PT2E_WORKFLOW_MODULE}.copy.deepcopy",
                return_value=exports["float_copy"],
            ) as copy_export,
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
                calibration_steps=8,
            )

        mock_build_calibration.assert_called_once_with(
            context=context,
            exportable=exportable,
            targets=targets,
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
            targets=targets,
            calibration=None,
            example_inputs=example_inputs,
            observation_keys=exportable.observation_keys,
            export_metadata=exportable.export_metadata,
        )
        copy_export.assert_called_once_with(exported)
        assert result.float_model is exports["float_copy"]
        assert result.quantized_model is converted
        assert result.example_inputs is example_inputs
        assert result.quantization_workflow == QuantizationWorkflow.PT2E.value

    def test_quantize_exports_with_synthetic_inputs_and_keeps_calibration(
        self,
        pt2e_target_factory: Callable[..., PT2EQuantizationModuleTarget],
        calibration_context_factory: Callable[..., MagicMock],
        export_mocks_factory: Callable[[], dict[str, MagicMock | tuple[MagicMock]]],
    ) -> None:
        context = calibration_context_factory(batch_size=2, observation_horizon=2)
        exports = export_mocks_factory()
        exportable = exports["exportable"]
        targets = [pt2e_target_factory(module_path="", needs_calibration=True)]
        workflow = PT2EQuantizationWorkflow(targets=targets)
        calibration = MagicMock()
        example_inputs = exports["example_inputs"]
        exported = exports["exported"]
        converted = exports["quantized"]

        with (
            patch(
                f"{PT2E_WORKFLOW_MODULE}.copy.deepcopy",
                return_value=exports["float_copy"],
            ) as copy_export,
            patch.object(
                PT2EQuantizationWorkflow,
                "_build_calibration",
                return_value=calibration,
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
                calibration_steps=8,
            )

        mock_build_calibration.assert_called_once_with(
            context=context,
            exportable=exportable,
            targets=targets,
            calibration_steps=8,
        )
        # Export must use synthetic batch>=2 inputs (a raw batch_size=1
        # training batch would hit torch.export 0/1 specialization), while
        # calibration still consumes the real dataloader batches.
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
            targets=targets,
            calibration=calibration,
            example_inputs=example_inputs,
            observation_keys=exportable.observation_keys,
            export_metadata=exportable.export_metadata,
        )
        copy_export.assert_called_once_with(exported)
        assert result.float_model is exports["float_copy"]
        assert result.quantized_model is converted
        assert result.example_inputs is example_inputs

    def test_pt2e_qat_raises_on_init(self, mock_pt2e_backend_factory):
        with pytest.raises(
            NotImplementedError,
            match=re.escape("PT2E QAT configuration is not supported yet."),
        ):
            target = PT2EQuantizationModuleTarget(
                module_path="",
                pt2e_backend=mock_pt2e_backend_factory(is_qat=True),
            )
            PT2EQuantizationWorkflow(
                targets=[target],
            )

    def test_prepare_model_raises_as_unsupported(self, mock_pt2e_backend_factory):
        target = PT2EQuantizationModuleTarget(
            module_path="",
            pt2e_backend=mock_pt2e_backend_factory(),
        )
        workflow = PT2EQuantizationWorkflow(targets=[target])

        with pytest.raises(
            NotImplementedError,
            match=re.escape(
                "PT2EQuantizationWorkflow does not support QAT preparation."
            ),
        ):
            workflow.prepare_model(model=MagicMock(spec=nn.Module))

    def test_empty_targets_returns_exported_unchanged(self) -> None:
        exported = MagicMock(spec=nn.Module)

        result = PT2EQuantizationWorkflow._convert_exported_model(
            exported=exported,
            targets=[],
            calibration=None,
            example_inputs=(),
            observation_keys=[],
        )

        assert result is exported

    @pytest.mark.parametrize("is_dynamic", [True, False])
    def test_observes_weights_with_export_inputs_or_static_calibration_batches(
        self,
        pt2e_target_factory: Callable[..., PT2EQuantizationModuleTarget],
        pt2e_mocks_factory: Callable[[], dict[str, MagicMock]],
        mock_calibration_provider_factory: Callable[..., MagicMock],
        is_dynamic: bool,
    ) -> None:
        pt2e_mocks = pt2e_mocks_factory()
        target = pt2e_target_factory(module_path="", needs_calibration=not is_dynamic)
        example_inputs = (MagicMock(),)
        observation_keys = ["left", "right"]
        calibration = (
            None
            if is_dynamic
            else mock_calibration_provider_factory(
                observation_keys=observation_keys, num_batches=2
            )
        )
        calibration_batches = [] if calibration is None else list(calibration)
        exported = MagicMock(spec=nn.Module)
        PT2EQuantizationWorkflow._convert_exported_model(
            exported=exported,
            targets=[target],
            calibration=calibration,
            example_inputs=example_inputs,
            observation_keys=observation_keys,
        )
        pt2e_mocks["prepare"].assert_called_once_with(
            exported, pt2e_mocks["composer"].return_value
        )
        prepared = pt2e_mocks["prepare"].return_value
        assert prepared.call_args_list == (
            [call(*example_inputs)]
            if is_dynamic
            else [
                call(*(batch[key] for key in observation_keys))
                for batch in calibration_batches
            ]
        )
        pt2e_mocks["convert"].assert_called_once_with(prepared)
        if is_dynamic:
            pt2e_mocks["metadata"].assert_not_called()
        else:
            pt2e_mocks["metadata"].assert_called_once_with()
            assert pt2e_mocks[
                "metadata"
            ].return_value.prepare_inputs.call_args_list == [
                call(observations=tuple(batch[key] for key in observation_keys))
                for batch in calibration_batches
            ]

    @pytest.mark.parametrize("is_dynamic", [True, False])
    @pytest.mark.parametrize("include_step_noise", [True, False])
    def test_calibration_prepares_specified_inputs_and_dynamic_uses_complete_examples(
        self,
        pt2e_target_factory: Callable[..., PT2EQuantizationModuleTarget],
        pt2e_mocks_factory: Callable[[], dict[str, MagicMock]],
        mock_calibration_provider_factory: Callable[..., MagicMock],
        calibration_noise_factory: Callable[..., tuple[torch.Tensor, ...]],
        pt2e_export_metadata_factory: Callable[..., MagicMock],
        is_dynamic: bool,
        include_step_noise: bool,
    ) -> None:
        pt2e_mocks = pt2e_mocks_factory()
        target = pt2e_target_factory(module_path="", needs_calibration=not is_dynamic)
        observation_keys = ["left", "right"]
        calibration = mock_calibration_provider_factory(
            observation_keys=observation_keys, batch_size=2, num_batches=2
        )
        batches = list(calibration)
        noises = calibration_noise_factory(
            batch_size=2, include_step_noise=include_step_noise
        )
        export_metadata = pt2e_export_metadata_factory(noise_inputs=noises)
        example_inputs = tuple(batches[0][key] for key in observation_keys) + noises

        PT2EQuantizationWorkflow._convert_exported_model(
            exported=pt2e_mocks["prepare"].return_value,
            targets=[target],
            calibration=None if is_dynamic else calibration,
            example_inputs=example_inputs,
            observation_keys=observation_keys,
            export_metadata=export_metadata,
        )

        prepared = pt2e_mocks["prepare"].return_value
        if is_dynamic:
            export_metadata.prepare_inputs.assert_not_called()
            prepared.assert_called_once_with(*example_inputs)
        else:
            observation_inputs = [
                tuple(batch[key] for key in observation_keys) for batch in batches
            ]
            assert export_metadata.prepare_inputs.call_args_list == [
                call(observations=inputs) for inputs in observation_inputs
            ]
            assert prepared.call_args_list == [
                call(*(inputs + noises)) for inputs in observation_inputs
            ]
        pt2e_mocks["convert"].assert_called_once_with(prepared)
        pt2e_mocks["metadata"].assert_not_called()

    def test_build_calibration_returns_none_for_dynamic_targets(
        self,
        pt2e_target_factory: Callable[..., MagicMock],
    ) -> None:
        target = pt2e_target_factory(module_path="", needs_calibration=False)

        with patch(f"{PT2E_WORKFLOW_MODULE}.build_calibration_data") as mock_builder:
            result = PT2EQuantizationWorkflow._build_calibration(
                context=MagicMock(),
                exportable=MagicMock(),
                targets=[target],
                calibration_steps=8,
            )

        mock_builder.assert_not_called()
        assert result is None

    def test_build_calibration_delegates_to_shared_data_builder(
        self,
        pt2e_target_factory: Callable[..., MagicMock],
        calibration_context_factory: Callable[..., MagicMock],
    ) -> None:
        target = pt2e_target_factory(module_path="", needs_calibration=True)
        context = calibration_context_factory(batch_size=2, observation_horizon=1)
        exportable = MagicMock()
        exportable.observation_keys = ["left", "depth"]
        with patch(f"{PT2E_WORKFLOW_MODULE}.build_calibration_data") as mock_builder:
            result = PT2EQuantizationWorkflow._build_calibration(
                context=context,
                exportable=exportable,
                targets=[target],
                calibration_steps=8,
            )

        mock_builder.assert_called_once_with(
            context=context,
            observation_keys=exportable.observation_keys,
            num_calibration_steps=8,
            device=torch.device("cpu"),
        )
        assert result is mock_builder.return_value

    @pytest.mark.parametrize("limit", [0, -1])
    def test_static_calibration_rejects_invalid_limit_before_building_data(
        self,
        pt2e_target_factory: Callable[..., PT2EQuantizationModuleTarget],
        calibration_context_factory: Callable[..., MagicMock],
        limit: int,
    ) -> None:
        target = pt2e_target_factory(module_path="", needs_calibration=True)
        context = calibration_context_factory(batch_size=2, observation_horizon=1)
        with (
            patch(f"{PT2E_WORKFLOW_MODULE}.build_calibration_data") as builder,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Static PT2E quantization requires calibration_steps >= 1, "
                    f"got {limit}."
                ),
            ),
        ):
            PT2EQuantizationWorkflow._build_calibration(
                context=context,
                exportable=MagicMock(),
                targets=[target],
                calibration_steps=limit,
            )
        builder.assert_not_called()

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
        pt2e_target_factory: Callable[..., MagicMock],
        pt2e_mocks_factory: Callable[[], dict[str, MagicMock]],
        needs_calibration: bool,
        has_calibration: bool,
        expectation: AbstractContextManager[None],
    ) -> None:
        pt2e_mocks = pt2e_mocks_factory()
        target = pt2e_target_factory(
            module_path="", needs_calibration=needs_calibration
        )
        calibration = MagicMock() if has_calibration else None

        with expectation:
            PT2EQuantizationWorkflow._convert_exported_model(
                exported=MagicMock(spec=nn.Module),
                targets=[target],
                calibration=calibration,
                example_inputs=(MagicMock(),),
                observation_keys=["left"],
            )
        if needs_calibration and not has_calibration:
            pt2e_mocks["prepare"].assert_not_called()
        else:
            pt2e_mocks["convert"].assert_called_once_with(
                pt2e_mocks["prepare"].return_value
            )

    def test_pt2e_uses_composable_quantizer(
        self,
        pt2e_target_factory: Callable[..., MagicMock],
        pt2e_mocks_factory: Callable[[], dict[str, MagicMock]],
    ) -> None:
        pt2e_mocks = pt2e_mocks_factory()
        target = pt2e_target_factory(module_path="encoder", needs_calibration=False)

        PT2EQuantizationWorkflow._convert_exported_model(
            exported=MagicMock(spec=nn.Module),
            targets=[target],
            calibration=None,
            example_inputs=(MagicMock(),),
            observation_keys=["left"],
        )

        target.pt2e_backend.create_quantizer.assert_called_once_with(
            module_path="encoder",
        )
        pt2e_mocks["composer"].assert_called_once()
        pt2e_mocks["prepare"].assert_called_once()
        pt2e_mocks["convert"].assert_called_once()

    def test_pt2e_builds_one_quantizer_per_target(
        self,
        pt2e_target_factory: Callable[..., MagicMock],
        pt2e_mocks_factory: Callable[[], dict[str, MagicMock]],
    ) -> None:
        pt2e_mocks = pt2e_mocks_factory()
        targets = [
            pt2e_target_factory(module_path="encoder", needs_calibration=False),
            pt2e_target_factory(module_path="decoder", needs_calibration=False),
        ]

        PT2EQuantizationWorkflow._convert_exported_model(
            exported=MagicMock(spec=nn.Module),
            targets=targets,
            calibration=None,
            example_inputs=(MagicMock(),),
            observation_keys=["left"],
        )

        targets[0].pt2e_backend.create_quantizer.assert_called_once_with(
            module_path="encoder",
        )
        targets[1].pt2e_backend.create_quantizer.assert_called_once_with(
            module_path="decoder",
        )
        pt2e_mocks["composer"].assert_called_once_with(
            [
                targets[0].pt2e_backend.create_quantizer.return_value,
                targets[1].pt2e_backend.create_quantizer.return_value,
            ],
        )
