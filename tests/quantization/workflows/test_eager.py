"""Tests for versatil.quantization.workflows.eager module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, NonCallableMock, call, patch

import pytest
import torch
import torch.nn as nn
from torchao.core.config import AOBaseConfig

from versatil.models.policy import Policy
from versatil.post_training_compression.constants import (
    QuantizationWorkflow,
)
from versatil.post_training_compression.deployment_backends.base import (
    DeploymentBackend,
)
from versatil.quantization.calibration import CalibrationDataProvider
from versatil.quantization.constants import QuantizationMode
from versatil.quantization.metadata import (
    QuantizationTargetMetadata,
)
from versatil.quantization.module_target import EagerQuantizationModuleTarget
from versatil.quantization.schemas.base import QuantizationSchema
from versatil.quantization.workflows.eager import (
    EagerQuantizationWorkflow,
    _PreparedEagerTarget,
)

EAGER_WORKFLOW_MODULE = "versatil.quantization.workflows.eager"


@pytest.fixture
def schema_target_factory(
    quantization_target_metadata_factory: Callable[..., QuantizationTargetMetadata],
) -> Callable[..., MagicMock]:
    def factory(
        module_path: str,
        needs_calibration: bool,
        needs_preparation: bool = False,
    ) -> MagicMock:
        schema = MagicMock(spec=QuantizationSchema)
        schema.base_config = MagicMock(spec=AOBaseConfig)
        schema.parameters = {}
        schema.needs_calibration = needs_calibration
        schema.preparation_config.return_value = (
            MagicMock(spec=AOBaseConfig)
            if needs_preparation or needs_calibration
            else None
        )
        schema.conversion_config.return_value = MagicMock(spec=AOBaseConfig)
        target = MagicMock(spec=EagerQuantizationModuleTarget)
        target.module_path = module_path
        target.label = module_path or "(root)"
        target.schema = schema
        layer_name = f"{module_path}.projection" if module_path else "projection"
        target.select_modules.return_value = ([layer_name], {})
        target.build_metadata.return_value = quantization_target_metadata_factory(
            module_path=module_path, selected=[]
        )
        return target

    return factory


@pytest.fixture
def prepared_targets_factory() -> Callable[..., list[_PreparedEagerTarget]]:
    def factory(
        targets: list[EagerQuantizationModuleTarget], module_names: list[str]
    ) -> list[_PreparedEagerTarget]:
        return [
            _PreparedEagerTarget(
                target=target,
                module_names={name},
                preparation_config=target.schema.preparation_config.return_value,
                conversion_config=target.schema.conversion_config.return_value,
            )
            for target, name in zip(targets, module_names, strict=True)
        ]

    return factory


@pytest.fixture
def selection_policy_factory() -> Callable[..., MagicMock]:
    def factory(
        modules: tuple[tuple[str, int], ...] = (("projection", 64),),
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        tokenized_actions: bool = False,
    ) -> MagicMock:
        policy = MagicMock(spec=Policy)
        module_map = {}
        for name, in_features in modules:
            layer = MagicMock(spec=nn.Linear)
            layer.in_features = in_features
            layer.out_features = 32
            layer.weight = MagicMock(spec=torch.Tensor)
            layer.weight.device = torch.device(device)
            layer.weight.dtype = dtype
            module_map[name] = layer
            parent = name.rpartition(".")[0]
            if parent:
                module_map.setdefault(parent, MagicMock(spec=nn.Module))
        policy.named_modules.return_value = list(module_map.items())
        policy.get_submodule.side_effect = module_map.__getitem__
        policy.decoder = MagicMock(spec=nn.Module)
        policy.decoder.requires_tokenized_actions = tokenized_actions
        policy.training = False
        policy.device = torch.device(device)
        policy.input_keys = ["observations"]
        return policy

    return factory


@pytest.fixture
def calibrated_policy_factory(
    selection_policy_factory: Callable[..., MagicMock],
) -> Callable[..., MagicMock]:
    def factory(training: bool = False) -> MagicMock:
        policy = selection_policy_factory(
            modules=(("encoder.projection", 64), ("decoder.projection", 64)),
            device="cpu",
            dtype=torch.float32,
        )
        policy.training = training
        return policy

    return factory


@pytest.fixture
def deployment_backend_factory() -> Callable[[], MagicMock]:
    def factory() -> MagicMock:
        return MagicMock(spec=DeploymentBackend)

    return factory


@pytest.fixture
def quantized_conversion_factory() -> Callable[[], Callable]:
    def factory() -> Callable:
        def convert(
            model: nn.Module,
            config: AOBaseConfig,
            filter_fn: Callable[[nn.Module, str], bool],
        ) -> None:
            for name, layer in model.named_modules():
                if filter_fn(layer, name):
                    layer.weight = NonCallableMock(spec=torch.Tensor)

        return convert

    return factory


@pytest.mark.unit
class TestCalibratedModuleWorkflow:
    @pytest.mark.parametrize("needs_calibration", [False, True])
    @pytest.mark.parametrize("is_qat", [False, True])
    def test_builds_data_only_when_schema_requires_observations(
        self,
        schema_target_factory: Callable[..., EagerQuantizationModuleTarget],
        calibration_context_factory: Callable[..., MagicMock],
        needs_calibration: bool,
        is_qat: bool,
    ) -> None:
        context = calibration_context_factory(batch_size=2, observation_horizon=1)
        context.policy.input_keys = ["observations"]
        context.policy.device = torch.device("cpu")
        workflow = EagerQuantizationWorkflow(
            targets=[
                schema_target_factory(
                    module_path="decoder", needs_calibration=needs_calibration
                )
            ],
            is_qat=is_qat,
        )
        with patch(f"{EAGER_WORKFLOW_MODULE}.build_calibration_data") as build:
            result = workflow._build_calibration(context=context, calibration_steps=3)
        if needs_calibration and not is_qat:
            build.assert_called_once_with(
                context=context,
                observation_keys=context.policy.input_keys,
                num_calibration_steps=3,
                device=context.policy.device,
            )
            assert result is build.return_value
        else:
            build.assert_not_called()
            assert result is None

    @pytest.mark.parametrize("steps", [0, -1])
    def test_invalid_calibration_limit_fails_before_dataset_construction(
        self,
        schema_target_factory: Callable[..., EagerQuantizationModuleTarget],
        calibration_context_factory: Callable[..., MagicMock],
        steps: int,
    ) -> None:
        context = calibration_context_factory(batch_size=2, observation_horizon=1)
        workflow = EagerQuantizationWorkflow(
            targets=[
                schema_target_factory(module_path="decoder", needs_calibration=True)
            ],
            is_qat=False,
        )
        with (
            patch(f"{EAGER_WORKFLOW_MODULE}.build_calibration_data") as build,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Calibrated module quantization requires calibration_steps >= 1, "
                    f"got {steps}."
                ),
            ),
        ):
            workflow._build_calibration(context=context, calibration_steps=steps)
        build.assert_not_called()

    def test_prepares_and_calibrates_before_converting_any_target(
        self,
        schema_target_factory: Callable[..., EagerQuantizationModuleTarget],
        prepared_targets_factory: Callable[..., list[_PreparedEagerTarget]],
        calibrated_policy_factory: Callable[..., MagicMock],
    ) -> None:
        direct = schema_target_factory(module_path="encoder", needs_calibration=False)
        observed = schema_target_factory(module_path="decoder", needs_calibration=True)
        workflow = EagerQuantizationWorkflow(targets=[direct, observed], is_qat=False)
        policy = calibrated_policy_factory(training=False)
        calibration = MagicMock(spec=CalibrationDataProvider)
        prepared_targets = prepared_targets_factory(
            targets=workflow.targets,
            module_names=["encoder.projection", "decoder.projection"],
        )
        events = MagicMock()
        with (
            patch.object(workflow, "_resolve_targets", return_value=prepared_targets),
            patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize,
            patch(
                f"{EAGER_WORKFLOW_MODULE}.calibrate_policy", return_value=3
            ) as calibrate,
        ):
            events.attach_mock(quantize, "quantize")
            events.attach_mock(calibrate, "calibrate")
            events.attach_mock(observed.schema.validate_calibration, "validate")
            batches, metadata = workflow._apply_ptq(
                model=policy, calibration=calibration
            )
        assert [event[0] for event in events.mock_calls] == [
            "quantize",
            "calibrate",
            "validate",
            "quantize",
            "quantize",
        ]
        assert [entry.kwargs["config"] for entry in quantize.call_args_list] == [
            observed.schema.preparation_config.return_value,
            direct.schema.conversion_config.return_value,
            observed.schema.conversion_config.return_value,
        ]
        for entry in (quantize.call_args_list[0], quantize.call_args_list[2]):
            assert entry.kwargs["model"] is policy
            select = entry.kwargs["filter_fn"]
            assert select(MagicMock(), "decoder.projection") is True
            assert select(MagicMock(), "encoder.projection") is False
            assert select(MagicMock(), "decoder.created_after_preparation") is False
        calibrate.assert_called_once_with(policy=policy, calibration=calibration)
        observed.schema.validate_calibration.assert_called_once_with(
            model=policy, module_names={"decoder.projection"}
        )
        assert batches == 3
        assert set(metadata[1].weight_representations) == {"decoder.projection"}

    @pytest.mark.parametrize("with_data, training", [(False, False), (True, True)])
    def test_missing_data_or_training_mode_fails_before_preparation(
        self,
        schema_target_factory: Callable[..., EagerQuantizationModuleTarget],
        prepared_targets_factory: Callable[..., list[_PreparedEagerTarget]],
        calibrated_policy_factory: Callable[..., MagicMock],
        with_data: bool,
        training: bool,
    ) -> None:
        workflow = EagerQuantizationWorkflow(
            targets=[
                schema_target_factory(module_path="decoder", needs_calibration=True)
            ],
            is_qat=False,
        )
        policy = calibrated_policy_factory(training=training)
        prepared_targets = prepared_targets_factory(
            targets=workflow.targets, module_names=["decoder.projection"]
        )
        calibration = MagicMock(spec=CalibrationDataProvider) if with_data else None
        message = (
            "Calibrated module quantization requires a Policy in evaluation mode."
            if with_data
            else "The selected quantization schemas require calibration observations."
        )
        with (
            patch.object(workflow, "_resolve_targets", return_value=prepared_targets),
            patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize,
            pytest.raises(ValueError, match=re.escape(message)),
        ):
            workflow._apply_ptq(model=policy, calibration=calibration)
        quantize.assert_not_called()

    def test_unobserved_target_prevents_conversion_of_direct_targets_too(
        self,
        schema_target_factory: Callable[..., EagerQuantizationModuleTarget],
        prepared_targets_factory: Callable[..., list[_PreparedEagerTarget]],
        calibrated_policy_factory: Callable[..., MagicMock],
    ) -> None:
        direct = schema_target_factory(module_path="encoder", needs_calibration=False)
        observed = schema_target_factory(module_path="decoder", needs_calibration=True)
        message = "Decoder did not execute during calibration."
        observed.schema.validate_calibration.side_effect = ValueError(message)
        workflow = EagerQuantizationWorkflow(targets=[direct, observed], is_qat=False)
        policy = calibrated_policy_factory(training=False)
        prepared_targets = prepared_targets_factory(
            targets=workflow.targets,
            module_names=["encoder.projection", "decoder.projection"],
        )
        with (
            patch.object(workflow, "_resolve_targets", return_value=prepared_targets),
            patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize,
            patch(f"{EAGER_WORKFLOW_MODULE}.calibrate_policy", return_value=1),
            pytest.raises(ValueError, match=re.escape(message)),
        ):
            workflow._apply_ptq(
                model=policy, calibration=MagicMock(spec=CalibrationDataProvider)
            )
        quantize.assert_called_once()
        assert (
            quantize.call_args.kwargs["config"]
            is observed.schema.preparation_config.return_value
        )

    def test_qat_resolves_all_targets_before_preparing_any_layer(
        self,
        schema_target_factory: Callable[..., EagerQuantizationModuleTarget],
        calibrated_policy_factory: Callable[..., MagicMock],
    ) -> None:
        workflow = EagerQuantizationWorkflow(
            targets=[
                schema_target_factory(
                    module_path="encoder",
                    needs_calibration=False,
                    needs_preparation=True,
                ),
                schema_target_factory(
                    module_path="decoder",
                    needs_calibration=False,
                    needs_preparation=True,
                ),
            ],
            is_qat=True,
        )
        policy = calibrated_policy_factory(training=True)
        workflow.targets[1].select_modules.side_effect = ValueError(
            "Target 'decoder' selects zero eligible linear modules; skipped modules: {}."
        )
        with (
            patch.object(workflow, "validate_targets") as validate,
            patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Target 'decoder' selects zero eligible linear modules; "
                    "skipped modules: {}."
                ),
            ),
        ):
            workflow.prepare_model(model=policy)
        validate.assert_called_once_with(model=policy)
        quantize.assert_not_called()


@pytest.mark.unit
class TestEagerQuantizationWorkflow:
    @pytest.mark.parametrize("is_qat", [False, True])
    @pytest.mark.parametrize("auto_filter", [False, True])
    def test_stores_configuration(
        self,
        schema_target_factory: Callable[..., MagicMock],
        is_qat: bool,
        auto_filter: bool,
    ) -> None:
        target = schema_target_factory(module_path="decoder", needs_calibration=False)
        workflow = EagerQuantizationWorkflow(
            targets=[target],
            is_qat=is_qat,
            auto_filter_incompatible_linears=auto_filter,
        )
        assert workflow.targets == [target]
        assert workflow.is_qat == is_qat
        assert workflow.auto_filter_incompatible_linears == auto_filter
        assert workflow.quantization_mode == QuantizationMode.EAGER.value

    def test_requires_at_least_one_target(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape("EagerQuantizationWorkflow requires at least one target."),
        ):
            EagerQuantizationWorkflow(targets=[])

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
        schema_target_factory: Callable[..., MagicMock],
        is_qat: bool,
    ) -> None:
        workflow = EagerQuantizationWorkflow(
            targets=[schema_target_factory(module_path="", needs_calibration=False)],
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
        calibration_context_factory: Callable[..., MagicMock],
        export_mocks_factory: Callable,
        schema_target_factory: Callable[..., MagicMock],
        prepared_targets_factory: Callable[..., list[_PreparedEagerTarget]],
        deployment_backend_factory: Callable[[], MagicMock],
        quantization_target_metadata_factory: Callable[..., QuantizationTargetMetadata],
    ) -> None:
        workflow = EagerQuantizationWorkflow(
            targets=[
                schema_target_factory(module_path="decoder", needs_calibration=False)
            ]
        )
        context = calibration_context_factory(batch_size=2, observation_horizon=2)
        backend = deployment_backend_factory()
        export_mocks = export_mocks_factory()
        prepared_targets = prepared_targets_factory(
            targets=workflow.targets, module_names=["decoder.projection"]
        )
        target_metadata = [
            quantization_target_metadata_factory(module_path="decoder", selected=[])
        ]
        events = MagicMock()

        with (
            patch.object(
                workflow, "_resolve_targets", return_value=prepared_targets
            ) as resolve,
            patch.object(
                workflow, "_execute_ptq", return_value=(2, target_metadata)
            ) as execute,
            patch(
                f"{EAGER_WORKFLOW_MODULE}.build_example_inputs",
                return_value=export_mocks["example_inputs"],
            ) as mock_build_inputs,
            patch(
                f"{EAGER_WORKFLOW_MODULE}.export_policy",
                side_effect=[export_mocks["exported"], export_mocks["quantized"]],
            ) as mock_export,
        ):
            events.attach_mock(resolve, "resolve")
            events.attach_mock(mock_build_inputs, "build_inputs")
            events.attach_mock(mock_export, "export")
            events.attach_mock(execute, "execute")
            result = workflow.quantize(
                context=context,
                exportable=export_mocks["exportable"],
                calibration_steps=8,
                deployment_backend=backend,
            )

        assert [event[0] for event in events.mock_calls] == [
            "build_inputs",
            "export",
            "resolve",
            "execute",
            "export",
        ]
        resolve.assert_called_once_with(
            model=context.policy,
            deployment_backend=backend,
        )
        execute.assert_called_once_with(
            model=context.policy, prepared_targets=prepared_targets, calibration=None
        )
        mock_build_inputs.assert_called_once_with(
            exportable=export_mocks["exportable"],
            observation_space=context.observation_space,
            observation_horizon=context.observation_horizon,
            tokenizer=context.tokenizer,
        )
        assert mock_export.call_count == 2
        assert mock_export.call_args_list[0] == mock_export.call_args_list[1]
        assert result.float_model is export_mocks["exported"]
        assert result.quantized_model is export_mocks["quantized"]
        assert result.example_inputs is export_mocks["example_inputs"]
        assert result.quantization_workflow == QuantizationWorkflow.EAGER.value
        assert result.calibration_batches == 2
        assert result.quantization_targets == target_metadata

    def test_quantize_converts_qat_model_before_export(
        self,
        calibration_context_factory: Callable[..., MagicMock],
        export_mocks_factory: Callable,
        schema_target_factory: Callable[..., MagicMock],
        prepared_targets_factory: Callable[..., list[_PreparedEagerTarget]],
        deployment_backend_factory: Callable[[], MagicMock],
        quantization_target_metadata_factory: Callable[..., QuantizationTargetMetadata],
    ) -> None:
        workflow = EagerQuantizationWorkflow(
            targets=[
                schema_target_factory(module_path="decoder", needs_calibration=False)
            ],
            is_qat=True,
        )
        context = calibration_context_factory(batch_size=2, observation_horizon=2)
        backend = deployment_backend_factory()
        export_mocks = export_mocks_factory()
        prepared_targets = prepared_targets_factory(
            targets=workflow.targets, module_names=["decoder.projection"]
        )
        workflow._prepared_targets = prepared_targets
        target_metadata = [
            quantization_target_metadata_factory(module_path="decoder", selected=[])
        ]

        events = MagicMock()
        with (
            patch.object(workflow, "_validate_prepared_targets") as validate,
            patch.object(
                workflow, "_convert_targets", return_value=target_metadata
            ) as mock_convert,
            patch(
                f"{EAGER_WORKFLOW_MODULE}.build_example_inputs",
                return_value=export_mocks["example_inputs"],
            ) as mock_build_inputs,
            patch(
                f"{EAGER_WORKFLOW_MODULE}.export_policy",
                side_effect=[export_mocks["exported"], export_mocks["quantized"]],
            ) as mock_export,
        ):
            events.attach_mock(validate, "validate")
            events.attach_mock(mock_build_inputs, "build_inputs")
            events.attach_mock(mock_export, "export")
            events.attach_mock(mock_convert, "convert")
            result = workflow.quantize(
                context=context,
                exportable=export_mocks["exportable"],
                calibration_steps=8,
                deployment_backend=backend,
            )

        assert [event[0] for event in events.mock_calls] == [
            "validate",
            "build_inputs",
            "export",
            "convert",
            "export",
        ]
        validate.assert_called_once_with(
            model=context.policy, deployment_backend=backend
        )
        mock_convert.assert_called_once_with(
            model=context.policy, prepared_targets=prepared_targets
        )
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
        assert result.calibration_batches == 0
        assert result.quantization_targets == target_metadata


@pytest.mark.unit
class TestEagerTargetResolution:
    @pytest.mark.parametrize("for_conversion", [False, True])
    @pytest.mark.parametrize("with_backend", [False, True])
    def test_delegates_numerical_and_deployment_checks(
        self,
        schema_target_factory: Callable[..., MagicMock],
        selection_policy_factory: Callable[..., MagicMock],
        deployment_backend_factory: Callable[[], MagicMock],
        for_conversion: bool,
        with_backend: bool,
    ) -> None:
        target = schema_target_factory(module_path="decoder", needs_calibration=False)
        model = selection_policy_factory(modules=(("decoder.projection", 64),))
        backend = deployment_backend_factory() if with_backend else None
        workflow = EagerQuantizationWorkflow(targets=[target])
        workflow._validate_target(
            model=model,
            target=target,
            module_names={"decoder.projection"},
            activation_dtype=torch.bfloat16,
            deployment_backend=backend,
            for_conversion=for_conversion,
        )
        target.schema.validate_configuration.assert_called_once_with(
            model=model,
            module_names={"decoder.projection"},
            label="decoder",
            activation_dtype=torch.bfloat16,
            for_conversion=for_conversion,
        )
        if backend is not None:
            backend.validate_eager_target.assert_called_once_with(
                model=model,
                target=target,
                module_names={"decoder.projection"},
                for_conversion=for_conversion,
            )

    @pytest.mark.parametrize("auto_filter", [False, True])
    def test_resolves_selection_and_preparation_conversion_configs_once(
        self,
        schema_target_factory: Callable[..., MagicMock],
        selection_policy_factory: Callable[..., MagicMock],
        deployment_backend_factory: Callable[[], MagicMock],
        auto_filter: bool,
    ) -> None:
        target = schema_target_factory(
            module_path="decoder", needs_calibration=False, needs_preparation=True
        )
        reason = "Weight row width 48 requires divisibility by group_size 32"
        target.select_modules.return_value = (
            ["decoder.projection"],
            {"decoder.head": reason},
        )
        model = selection_policy_factory(modules=(("decoder.projection", 64),))
        backend = deployment_backend_factory()
        workflow = EagerQuantizationWorkflow(
            targets=[target],
            is_qat=True,
            auto_filter_incompatible_linears=auto_filter,
        )
        with (
            patch.object(workflow, "validate_targets") as validate_paths,
            patch(f"{EAGER_WORKFLOW_MODULE}.logger.info") as log,
        ):
            prepared = workflow._resolve_targets(
                model=model,
                deployment_backend=backend,
                activation_dtype=torch.bfloat16,
                for_conversion=False,
            )
        validate_paths.assert_called_once_with(model=model)
        target.select_modules.assert_called_once_with(
            model=model, auto_filter_incompatible=auto_filter
        )
        target.schema.preparation_config.assert_called_once_with(is_qat=True)
        target.schema.conversion_config.assert_called_once_with(is_qat=True)
        assert prepared[0].module_names == {"decoder.projection"}
        assert prepared[0].skipped == {"decoder.head": reason}
        assert prepared[0].preparation_config == (
            target.schema.preparation_config.return_value
        )
        assert (
            prepared[0].conversion_config
            == target.schema.conversion_config.return_value
        )
        log.assert_called_once_with(
            f"Skipping quantization module decoder.head: {reason}"
        )

    @pytest.mark.parametrize("failure_stage", ["selection", "schema", "backend"])
    def test_later_target_failure_prevents_all_preparation_and_conversion(
        self,
        schema_target_factory: Callable[..., MagicMock],
        calibrated_policy_factory: Callable[..., MagicMock],
        deployment_backend_factory: Callable[[], MagicMock],
        failure_stage: str,
    ) -> None:
        targets = [
            schema_target_factory(
                module_path=path, needs_calibration=False, needs_preparation=True
            )
            for path in ("encoder", "decoder")
        ]
        workflow = EagerQuantizationWorkflow(targets=targets, is_qat=False)
        model = calibrated_policy_factory(training=False)
        backend = deployment_backend_factory()
        message = f"Decoder {failure_stage} validation failed."
        if failure_stage == "selection":
            targets[1].select_modules.side_effect = ValueError(message)
        elif failure_stage == "schema":
            targets[1].schema.validate_configuration.side_effect = ValueError(message)
        else:
            backend.validate_eager_target.side_effect = [None, ValueError(message)]
        with (
            patch.object(workflow, "validate_targets"),
            patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize,
            pytest.raises(ValueError, match=re.escape(message)),
        ):
            workflow._apply_ptq(model=model, deployment_backend=backend)
        quantize.assert_not_called()
        for target in targets:
            target.select_modules.assert_called_once_with(
                model=model, auto_filter_incompatible=True
            )

    def test_path_validation_precedes_selection(
        self,
        schema_target_factory: Callable[..., MagicMock],
        calibrated_policy_factory: Callable[..., MagicMock],
    ) -> None:
        target = schema_target_factory(module_path="decoder", needs_calibration=False)
        workflow = EagerQuantizationWorkflow(targets=[target])
        model = calibrated_policy_factory(training=False)
        message = "Quantization target paths overlap."
        with (
            patch.object(workflow, "validate_targets", side_effect=ValueError(message)),
            pytest.raises(ValueError, match=re.escape(message)),
        ):
            workflow._resolve_targets(model=model)
        target.select_modules.assert_not_called()
        target.schema.validate_configuration.assert_not_called()

    def test_captures_all_target_metadata_before_conversion(
        self,
        schema_target_factory: Callable[..., MagicMock],
        prepared_targets_factory: Callable[..., list[_PreparedEagerTarget]],
        calibrated_policy_factory: Callable[..., MagicMock],
        quantized_conversion_factory: Callable[[], Callable],
    ) -> None:
        targets = [
            schema_target_factory(module_path=path, needs_calibration=False)
            for path in ("encoder", "decoder")
        ]
        workflow = EagerQuantizationWorkflow(targets=targets)
        model = calibrated_policy_factory(training=False)
        prepared = prepared_targets_factory(
            targets=targets, module_names=["encoder.projection", "decoder.projection"]
        )
        events = MagicMock()
        events.attach_mock(targets[0].build_metadata, "encoder_metadata")
        events.attach_mock(targets[1].build_metadata, "decoder_metadata")
        with patch(
            f"{EAGER_WORKFLOW_MODULE}.quantize_",
            side_effect=quantized_conversion_factory(),
        ) as quantize:
            events.attach_mock(quantize, "quantize")
            metadata = workflow._convert_targets(model=model, prepared_targets=prepared)
        assert [event[0] for event in events.mock_calls] == [
            "encoder_metadata",
            "decoder_metadata",
            "quantize",
            "quantize",
        ]
        for target, resolved, converted in zip(
            targets, prepared, metadata, strict=True
        ):
            target.build_metadata.assert_called_once_with(
                model=model,
                module_names=resolved.module_names,
                skipped=resolved.skipped,
            )
            assert converted.weight_representations == dict.fromkeys(
                resolved.module_names, "NonCallableMock"
            )
        for conversion, resolved in zip(quantize.call_args_list, prepared, strict=True):
            assert conversion.kwargs["model"] == model
            assert conversion.kwargs["config"] == resolved.conversion_config
            selected_name = next(iter(resolved.module_names))
            assert conversion.kwargs["filter_fn"](model, selected_name) is True
            assert conversion.kwargs["filter_fn"](model, "new_projection") is False


@pytest.mark.unit
class TestEagerQATLifecycle:
    @pytest.mark.parametrize(
        "method_name,is_qat,message",
        [
            ("prepare_model", False, "prepare_model() requires is_qat=True."),
            ("convert_model", False, "convert_model() requires is_qat=True."),
            (
                "_apply_ptq",
                True,
                "_apply_ptq() requires is_qat=False; use prepare_model() and convert_model() for QAT.",
            ),
        ],
    )
    def test_enforces_lifecycle_entry_points(
        self,
        schema_target_factory: Callable[..., MagicMock],
        calibrated_policy_factory: Callable[..., MagicMock],
        method_name: str,
        is_qat: bool,
        message: str,
    ) -> None:
        target = schema_target_factory(module_path="decoder", needs_calibration=False)
        workflow = EagerQuantizationWorkflow(targets=[target], is_qat=is_qat)
        model = calibrated_policy_factory(training=False)
        with (
            patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize,
            pytest.raises(ValueError, match=re.escape(message)),
        ):
            getattr(workflow, method_name)(model=model)
        quantize.assert_not_called()

    def test_conversion_requires_recorded_preparation(
        self,
        schema_target_factory: Callable[..., MagicMock],
        calibrated_policy_factory: Callable[..., MagicMock],
    ) -> None:
        target = schema_target_factory(module_path="decoder", needs_calibration=False)
        workflow = EagerQuantizationWorkflow(targets=[target], is_qat=True)
        model = calibrated_policy_factory(training=False)
        with pytest.raises(
            ValueError,
            match=re.escape("QAT convert_model() requires prepare_model() first."),
        ):
            workflow.convert_model(model=model)
        target.select_modules.assert_not_called()

    def test_conversion_reuses_prepared_names_and_checks_current_placement(
        self,
        schema_target_factory: Callable[..., MagicMock],
        calibrated_policy_factory: Callable[..., MagicMock],
    ) -> None:
        target = schema_target_factory(
            module_path="decoder", needs_calibration=False, needs_preparation=True
        )
        workflow = EagerQuantizationWorkflow(targets=[target], is_qat=True)
        model = calibrated_policy_factory(training=True)
        with (
            patch.object(workflow, "validate_targets"),
            patch(f"{EAGER_WORKFLOW_MODULE}.quantize_") as quantize,
        ):
            workflow.prepare_model(model=model)
            target.select_modules.return_value = (["decoder.new_projection"], {})
            model.training = False
            workflow.convert_model(model=model)
        target.select_modules.assert_called_once_with(
            model=model, auto_filter_incompatible=True
        )
        assert target.schema.validate_configuration.call_args_list == [
            call(
                model=model,
                module_names={"decoder.projection"},
                label="decoder",
                activation_dtype=None,
                for_conversion=for_conversion,
            )
            for for_conversion in (False, True)
        ]
        assert [entry.kwargs["config"] for entry in quantize.call_args_list] == [
            target.schema.preparation_config.return_value,
            target.schema.conversion_config.return_value,
        ]
        for entry in quantize.call_args_list:
            assert entry.kwargs["filter_fn"](model, "decoder.projection") is True
            assert entry.kwargs["filter_fn"](model, "decoder.new_projection") is False
