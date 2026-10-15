"""Tests for versatil.quantization workflow export and reload integration."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
)
from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import (
    Int4TilePackedTo4dTensor,
)
from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor

from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.configs.main import MainConfig
from versatil.configs.paths import get_hydra_configs_dir
from versatil.configs.post_training_compression import (
    ExecutorchXNNPACKBackendConfig,
    PostTrainingCompressorConfig,
)
from versatil.configs.quantization import (
    EagerQuantizationModuleTargetConfig,
    EagerQuantizationWorkflowConfig,
    PT2EQuantizationModuleTargetConfig,
    PT2EQuantizationWorkflowConfig,
    SmoothQuantSchemaConfig,
    X86InductorBackendConfig,
    XNNPACKPT2EBackendConfig,
)
from versatil.data.constants import SampleKey
from versatil.data.episodic_dataset import EpisodicDataset
from versatil.inference.policy_runtime.executorch_adapter import ExecuTorchModuleAdapter
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.exportable.factory import create_exportable_policy
from versatil.models.exportable.metadata import PolicyExportMetadata
from versatil.models.layers.denoising.diffusion_process import SchedulerType
from versatil.models.policy import Policy
from versatil.post_training_compression.constants import (
    CompressionFilename,
    CompressionMetadataKey,
)
from versatil.post_training_compression.deployment_backends.executorch_xnnpack import (
    ExecutorchXNNPACKBackend,
)
from versatil.post_training_compression.export import export_policy
from versatil.post_training_compression.policy_context import PolicyContext
from versatil.post_training_compression.report import QuantizationReport
from versatil.quantization.calibration import CalibrationDataProvider
from versatil.quantization.module_target import (
    EagerQuantizationModuleTarget,
    PT2EQuantizationModuleTarget,
)
from versatil.quantization.pt2e.backends.x86_inductor import X86InductorBackend
from versatil.quantization.schemas.smoothquant import SmoothQuantSchema
from versatil.quantization.workflows.eager import EagerQuantizationWorkflow
from versatil.quantization.workflows.pt2e import PT2EQuantizationWorkflow
from versatil.training.constants import PrecisionType


@pytest.fixture
def calibration_dataset_factory() -> Callable[..., MagicMock]:
    def factory(observations: dict[str, torch.Tensor], repeats: int) -> MagicMock:
        batch_size = next(iter(observations.values())).shape[0]
        samples = [
            {
                SampleKey.OBSERVATION.value: {
                    key: value[
                        index
                    ]  # (batch, horizon, feature_dim) -> (horizon, feature_dim)
                    for key, value in observations.items()
                }
            }
            for index in range(batch_size)
        ] * repeats
        dataset = MagicMock(spec=EpisodicDataset)
        dataset.__len__.return_value = len(samples)
        dataset.__getitem__.side_effect = samples.__getitem__
        return dataset

    return factory


@pytest.fixture
def smoothquant_context_factory() -> Callable[..., PolicyContext]:
    def factory(policy: Policy, checkpoint_path: str) -> PolicyContext:
        config = MainConfig()
        config.experiment.precision = PrecisionType.FP32.value
        config.task.dataloader = DataLoaderConfig(batch_size=2)
        config.task.dataset_schema = MagicMock()
        config.task.dataset_schema.zarr_path = "/dataset.zarr"
        config.task.action_space = policy.action_space
        config.task.prediction_horizon = policy.prediction_horizon
        return PolicyContext(
            policy=policy,
            config=config,
            tokenizer=policy.tokenizer,
            observation_space=policy.observation_space,
            observation_horizon=policy.observation_horizon,
            checkpoint_path=checkpoint_path,
            checkpoint_name="unused.ckpt",
        )

    return factory


@pytest.fixture
def mixed_precision_policy_factory(
    quantization_policy_factory: Callable[..., tuple[Policy, dict[str, torch.Tensor]]],
) -> Callable[..., tuple[Policy, dict[str, torch.Tensor]]]:
    def factory(device: torch.device) -> tuple[Policy, dict[str, torch.Tensor]]:
        policy, observations = quantization_policy_factory(
            family="flow", device=device, dtype=torch.float32, batch_size=3
        )
        policy.decoder.to(dtype=torch.bfloat16)
        policy.decoder.action_heads.to(dtype=torch.float32)
        return policy, observations

    return factory


@pytest.fixture
def compression_recipe_factory() -> Callable[
    [str], EagerQuantizationWorkflowConfig | PT2EQuantizationWorkflowConfig | None
]:
    def factory(
        recipe: str,
    ) -> EagerQuantizationWorkflowConfig | PT2EQuantizationWorkflowConfig | None:
        if recipe == "none":
            return None
        if recipe in ("pt2e_static", "pt2e_dynamic", "pt2e_xnnpack"):
            return PT2EQuantizationWorkflowConfig(
                targets=[
                    PT2EQuantizationModuleTargetConfig(
                        module_path="decoder",
                        pt2e_backend=(
                            XNNPACKPT2EBackendConfig()
                            if recipe == "pt2e_xnnpack"
                            else X86InductorBackendConfig(
                                is_dynamic=recipe == "pt2e_dynamic"
                            )
                        ),
                    )
                ]
            )
        if recipe == "smoothquant":
            target = EagerQuantizationModuleTargetConfig(
                module_path="decoder",
                schema=SmoothQuantSchemaConfig(
                    base_config={
                        "_target_": "torchao.quantization.Int8DynamicActivationInt8WeightConfig",
                        "version": 2,
                        "weight_only_decode": False,
                    }
                ),
            )
        else:
            target = EagerQuantizationModuleTargetConfig(
                module_path="decoder",
                quantize_config={
                    "_target_": "torchao.quantization.Int8WeightOnlyConfig",
                    "version": 2,
                },
            )
        return EagerQuantizationWorkflowConfig(targets=[target])

    return factory


@pytest.mark.integration
@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    "recipe, family, source_device, deployment_device, compile_cpu",
    [
        ("none", "tokens", "cuda", "cpu", False),
        ("none", "flow", "cpu", "cuda", False),
        ("module_int8", "flow", "cuda", "cpu", False),
        ("smoothquant", "flow", "cuda", "cpu", False),
        ("pt2e_static", "flow", "cuda", "cpu", False),
        ("pt2e_dynamic", "diffusion", "cuda", "cpu", False),
        pytest.param(
            "pt2e_xnnpack",
            "flow",
            "cuda",
            "cpu",
            False,
            marks=pytest.mark.requires_executorch,
        ),
        pytest.param(
            "pt2e_static", "flow", "cuda", "cpu", True, marks=pytest.mark.slow
        ),
    ],
)
def test_compression_calibrates_on_source_and_reloads_on_deployment_device(
    quantization_policy_factory: Callable[..., tuple[Policy, dict[str, torch.Tensor]]],
    smoothquant_context_factory: Callable[..., PolicyContext],
    calibration_dataset_factory: Callable[..., MagicMock],
    compression_recipe_factory: Callable,
    recipe: str,
    family: str,
    source_device: str,
    deployment_device: str,
    compile_cpu: bool,
    tmp_path: Path,
) -> None:
    policy, observations = quantization_policy_factory(
        family=family,
        device=torch.device(source_device),
        dtype=torch.float32,
        batch_size=3,
    )
    context = smoothquant_context_factory(policy=policy, checkpoint_path=str(tmp_path))
    config = OmegaConf.structured(
        PostTrainingCompressorConfig(
            checkpoint_path=str(tmp_path),
            output_directory=str(tmp_path / "compressed"),
            device=source_device,
            deployment_device=deployment_device,
            calibration_steps=2,
            generate_report=True,
            quantization=compression_recipe_factory(recipe),
            deployment_backend=(
                ExecutorchXNNPACKBackendConfig(max_batch_size=4)
                if recipe == "pt2e_xnnpack"
                else None
            ),
        )
    )
    compressor = hydra.utils.instantiate(config)
    dataset = calibration_dataset_factory(observations=observations, repeats=2)
    calibration_devices = []
    original_iter = CalibrationDataProvider.__iter__

    def record_calibration(
        provider: CalibrationDataProvider,
    ) -> Iterator[dict[str, torch.Tensor]]:
        for batch in original_iter(provider):
            calibration_devices.extend(tensor.device.type for tensor in batch.values())
            yield batch

    with (
        patch(
            "versatil.quantization.workflows.none.load_float_policy_context",
            return_value=context,
        ),
        patch(
            "versatil.quantization.workflows.eager.load_float_policy_context",
            return_value=context,
        ),
        patch(
            "versatil.quantization.workflows.pt2e.load_float_policy_context",
            return_value=context,
        ),
        patch(
            "versatil.quantization.calibration.EpisodicDataset", return_value=dataset
        ),
        patch.object(CalibrationDataProvider, "__iter__", record_calibration),
        patch.object(
            QuantizationReport,
            "compute_inference_timing",
            return_value={
                "float_milliseconds": 1.0,
                "quantized_milliseconds": 1.0,
                "speedup": 1.0,
            },
        ),
        patch.object(
            compressor.deployment_backend,
            "export",
            wraps=compressor.deployment_backend.export,
        ) as deployment_export,
    ):
        output = Path(compressor.compress(hydra_config=config))

    if recipe in ("smoothquant", "pt2e_static", "pt2e_xnnpack"):
        assert calibration_devices
        assert set(calibration_devices) == {source_device}
    reloaded = (
        ExecuTorchModuleAdapter(
            model_path=str(output / CompressionFilename.EXECUTORCH_MODEL.value)
        )
        if recipe == "pt2e_xnnpack"
        else torch.export.load(
            output / CompressionFilename.COMPRESSED_MODEL.value
        ).module()
    )
    reference = deployment_export.call_args.kwargs["model"]
    example_inputs = deployment_export.call_args.kwargs["example_inputs"]
    assert all(tensor.device.type == deployment_device for tensor in example_inputs)
    for tensor in (*reloaded.parameters(), *reloaded.buffers()):
        assert tensor.device.type == deployment_device
    metadata = json.loads(
        (output / CompressionFilename.COMPRESSION_METADATA.value).read_text()
    )
    export_metadata = PolicyExportMetadata.from_dict(
        metadata=metadata[CompressionMetadataKey.POLICY_EXPORT_METADATA.value]
    )
    if recipe == "pt2e_xnnpack":
        reference_inputs = export_metadata.prepare_inputs(
            observations=tuple(observations[key].cpu() for key in policy.input_keys)
        )  # each (3, T, D), plus (3, H, A) noise
        # Compare matching CPU kernels against an ordinary fixed-batch export.
        reference_program = torch.export.export(reference, reference_inputs)
        reference_path = tmp_path / "cpu_reference.pte"
        reference_path.write_bytes(
            ExecutorchXNNPACKBackend._lower_to_pte_buffer(
                exported_program=reference_program
            )
        )
        reference_runtime = ExecuTorchModuleAdapter(model_path=str(reference_path))
        with torch.no_grad():
            reference_outputs = reference_runtime(
                observation_tensors=reference_inputs
            )  # each (3, H, A)
        for batch_size in (1, 3, 2):
            inputs = tuple(
                tensor[:batch_size] for tensor in reference_inputs
            )  # each (B, T, D) or (B, H, A)
            with torch.no_grad():
                actual = reloaded(observation_tensors=inputs)  # each (B, H, A)
            expected = tuple(
                tensor[:batch_size] for tensor in reference_outputs
            )  # each (B, H, A)
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
        return

    for batch_size in (1, 3):
        inputs = export_metadata.prepare_inputs(
            observations=tuple(
                observations[key][:batch_size].to(device=deployment_device)
                for key in policy.input_keys
            )
        )  # each (3, T, D) -> (B, T, D), plus optional (B, H, A) noise
        with torch.no_grad():
            expected = reference(*inputs)  # actions: (B, H, A); tokens: (B, L)
            actual = reloaded(*inputs)  # actions: (B, H, A); tokens: (B, L)
        torch.testing.assert_close(actual, expected)
    if compile_cpu:
        with (
            compressor.quantization.pt2e_backend.environment_context(),
            torch.no_grad(),
        ):
            compiled = torch.compile(reloaded, backend="inductor", fullgraph=True)
            compiled_output = compiled(*inputs)  # (B, H, A)
        torch.testing.assert_close(compiled_output, actual, rtol=1e-4, atol=1e-5)


@pytest.mark.integration
@pytest.mark.parametrize("recipe", ["none", "smoothquant"])
@pytest.mark.parametrize(
    "source_device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
)
def test_checkpoint_autocast_preserves_mixed_weights_through_compression_and_reload(
    mixed_precision_policy_factory: Callable[
        ..., tuple[Policy, dict[str, torch.Tensor]]
    ],
    smoothquant_context_factory: Callable[..., PolicyContext],
    calibration_dataset_factory: Callable[..., MagicMock],
    compression_recipe_factory: Callable,
    recipe: str,
    source_device: str,
    tmp_path: Path,
) -> None:
    policy, observations = mixed_precision_policy_factory(
        device=torch.device(source_device)
    )
    precision = PrecisionType.BF16_MIXED
    context = smoothquant_context_factory(policy=policy, checkpoint_path=str(tmp_path))
    context.config.experiment.precision = precision.value
    reference = create_exportable_policy(policy=policy)
    source_inputs = reference.export_metadata.prepare_inputs(
        observations=tuple(observations[key] for key in policy.input_keys)
    )  # observations: (3, 1, 3), followed by (3, 2, 3) noise
    with torch.no_grad(), precision.autocast(device_type=source_device):
        reference(*source_inputs)  # each (3, 2, 3)
    original_dtypes = {
        name: parameter.dtype for name, parameter in policy.named_parameters()
    }
    assert set(original_dtypes.values()) == {torch.float32, torch.bfloat16}
    config = OmegaConf.structured(
        PostTrainingCompressorConfig(
            checkpoint_path=str(tmp_path),
            output_directory=str(tmp_path / "compressed"),
            device=source_device,
            deployment_device="cpu",
            precision=None,
            calibration_steps=2,
            generate_report=False,
            quantization=compression_recipe_factory(recipe=recipe),
        )
    )
    compressor = hydra.utils.instantiate(config)
    dataset = calibration_dataset_factory(observations=observations, repeats=2)
    with (
        patch(
            "versatil.quantization.workflows.none.load_float_policy_context",
            return_value=context,
        ),
        patch(
            "versatil.quantization.workflows.eager.load_float_policy_context",
            return_value=context,
        ),
        patch(
            "versatil.quantization.calibration.EpisodicDataset", return_value=dataset
        ),
    ):
        output = Path(compressor.compress(hydra_config=config))
    assert context.config.experiment.precision == precision.value
    metadata = json.loads(
        (output / CompressionFilename.COMPRESSION_METADATA.value).read_text()
    )
    converted_weights = set()
    for target in metadata[CompressionMetadataKey.QUANTIZATION_TARGETS.value] or []:
        for layer in target["selected"]:
            name = f"{layer['name']}.weight"
            assert layer["dtype"] == str(original_dtypes[name])
            converted_weights.add(name)
    assert {
        name: parameter.dtype
        for name, parameter in policy.named_parameters()
        if name not in converted_weights
    } == {
        name: dtype
        for name, dtype in original_dtypes.items()
        if name not in converted_weights
    }
    export_metadata = PolicyExportMetadata.from_dict(
        metadata=metadata[CompressionMetadataKey.POLICY_EXPORT_METADATA.value]
    )
    assert export_metadata.precision is None
    if recipe == "smoothquant":
        assert metadata[CompressionMetadataKey.CALIBRATION_BATCHES.value] == 2
        assert metadata[CompressionMetadataKey.QUANTIZATION_TARGETS.value][0][
            "selected"
        ]
    reloaded = torch.export.load(
        output / CompressionFilename.COMPRESSED_MODEL.value
    ).module()
    if recipe == "none":
        assert {parameter.dtype for parameter in reloaded.parameters()} == {
            torch.float32,
            torch.bfloat16,
        }
    reference = create_exportable_policy(policy=policy)
    for batch_size in (1, 3):
        inputs = tuple(
            tensor[:batch_size].cpu() for tensor in source_inputs
        )  # each (batch, ...)
        with torch.no_grad(), precision.autocast(device_type="cpu"):
            expected = reference(*inputs)  # each (batch, 2, 3)
        with torch.no_grad():
            actual = reloaded(*inputs)  # each (batch, 2, 3)
        torch.testing.assert_close(actual, expected)


@pytest.mark.integration
@pytest.mark.parametrize(
    "family, share_normalizer", [("flow", True), ("tokens", False)]
)
@pytest.mark.parametrize(
    "precision",
    [None, PrecisionType.FP32, PrecisionType.FP16_TRUE, PrecisionType.BF16_TRUE],
)
@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(torch.device("cuda"), marks=pytest.mark.requires_gpu),
    ],
)
def test_compression_precision_preserves_preprocessing_and_reloads_predictions(
    quantization_policy_factory: Callable[..., tuple[Policy, dict[str, torch.Tensor]]],
    smoothquant_context_factory: Callable[..., PolicyContext],
    family: str,
    share_normalizer: bool,
    precision: PrecisionType | None,
    device: torch.device,
    tmp_path: Path,
) -> None:
    policy, observations = quantization_policy_factory(
        family=family, device=device, dtype=torch.float32, batch_size=2
    )
    policy.normalizer.fit(data=observations, dtype=torch.float32, device=device)
    if share_normalizer:
        policy.decoder.set_normalizer(normalizer=policy.normalizer)
    else:
        decoder_observations = {
            key: value * 2 + 0.5  # (batch, observation_horizon, feature_dimension)
            for key, value in observations.items()
        }
        policy.decoder.normalizer.fit(
            data=decoder_observations, dtype=torch.float32, device=device
        )
        policy.decoder.normalizer.to(device=device)
    policy.set_denoising_thresholds(thresholds={policy.input_keys[0]: 0.1234567})
    policy.loss_module = torch.nn.Linear(in_features=3, out_features=1, device=device)
    with torch.no_grad():
        for parameter in policy.loss_module.parameters():
            parameter.fill_(0.1234567)  # (1, 3) or (1,)
    normalizer_state = {
        name: value.detach().clone()  # (...)
        for name, value in policy.normalizer.state_dict().items()
    }
    decoder_normalizer_state = {
        name: value.detach().clone()  # (...)
        for name, value in policy.decoder.normalizer.state_dict().items()
    }
    assert normalizer_state
    if not share_normalizer:
        assert any(
            not torch.equal(value, decoder_normalizer_state[name])
            for name, value in normalizer_state.items()
        )
    loss_state = {
        name: value.detach().clone()  # (1, 3) or (1,)
        for name, value in policy.loss_module.state_dict().items()
    }
    normalized_observations = policy.normalizer.normalize(x=observations)
    decoder_normalized_observations = policy.decoder.normalizer.normalize(
        x=observations
    )
    thresholds = policy.get_denoising_thresholds()
    context = smoothquant_context_factory(policy=policy, checkpoint_path=str(tmp_path))
    config = OmegaConf.structured(
        PostTrainingCompressorConfig(
            checkpoint_path=str(tmp_path),
            output_directory=str(tmp_path / "compressed"),
            device=str(device),
            precision=precision.value if precision is not None else None,
            generate_report=False,
        )
    )
    compressor = hydra.utils.instantiate(config)

    with patch(
        "versatil.quantization.workflows.none.load_float_policy_context",
        return_value=context,
    ) as loader:
        output = Path(compressor.compress(hydra_config=config))

    loader.assert_called_once_with(
        checkpoint_path=str(tmp_path), checkpoint_name="last.ckpt", device=device
    )
    torch.testing.assert_close(
        policy.normalizer.state_dict(), normalizer_state, rtol=0, atol=0
    )
    torch.testing.assert_close(
        policy.decoder.normalizer.state_dict(), decoder_normalizer_state, rtol=0, atol=0
    )
    assert policy.normalizer.dtype == torch.float32
    assert policy.decoder.normalizer.dtype == torch.float32
    torch.testing.assert_close(
        policy.normalizer.normalize(x=observations),
        normalized_observations,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        policy.decoder.normalizer.normalize(x=observations),
        decoder_normalized_observations,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        policy.loss_module.state_dict(), loss_state, rtol=0, atol=0
    )
    saved_normalizer = torch.load(
        output / CompressionFilename.NORMALIZER.value,
        map_location=device,
        weights_only=True,
    )
    torch.testing.assert_close(saved_normalizer, normalizer_state, rtol=0, atol=0)
    assert policy.get_denoising_thresholds() == thresholds
    metadata = json.loads(
        (output / CompressionFilename.COMPRESSION_METADATA.value).read_text()
    )
    export_metadata = PolicyExportMetadata.from_dict(
        metadata=metadata[CompressionMetadataKey.POLICY_EXPORT_METADATA.value]
    )
    assert export_metadata.precision == precision
    assert export_metadata.version == (1 if precision is None else 2)
    reloaded = torch.export.load(
        output / CompressionFilename.COMPRESSED_MODEL.value
    ).module()
    reference = create_exportable_policy(policy=policy)
    for batch_size in (1, 2):
        inputs = export_metadata.prepare_inputs(
            observations=tuple(
                observations[key][:batch_size] for key in reference.observation_keys
            )
        )  # each (2, T, D) -> (B, T, D), followed by optional (B, H, A) noise
        assert all(tensor.device.type == device.type for tensor in inputs)
        expected_dtype = (
            torch.float32 if precision is None else precision.get_model_dtype()
        )
        assert all(
            tensor.dtype == expected_dtype
            for tensor in inputs
            if tensor.is_floating_point()
        )
        with torch.no_grad():
            expected = reference(*inputs)  # actions: (B, H, A); tokens: (B, L)
            actual = reloaded(*inputs)  # actions: (B, H, A); tokens: (B, L)
        torch.testing.assert_close(actual, expected)


@pytest.mark.integration
@pytest.mark.parametrize(
    "compile_model",
    [False, pytest.param(True, marks=pytest.mark.slow)],
    ids=["torch_export", "inductor"],
)
def test_smoothquant_compression_saves_actual_selection_and_calibration_metadata(
    quantization_policy_factory: Callable[..., tuple[Policy, dict[str, torch.Tensor]]],
    smoothquant_context_factory: Callable[..., PolicyContext],
    calibration_dataset_factory: Callable[..., MagicMock],
    compile_model: bool,
    tmp_path: Path,
) -> None:
    policy, observations = quantization_policy_factory(
        family="flow", device=torch.device("cpu"), dtype=torch.float32, batch_size=2
    )
    context = smoothquant_context_factory(policy=policy, checkpoint_path=str(tmp_path))
    with initialize_config_dir(
        config_dir=str(get_hydra_configs_dir()), version_base=None
    ):
        config = compose(
            config_name="end_to_end_ptq/smoothquant_int8",
            overrides=[
                f"checkpoint_path={tmp_path}",
                f"output_directory={tmp_path / 'compressed'}",
                "calibration_steps=2",
            ],
        )
        compressor = hydra.utils.instantiate(config)
    dataset = calibration_dataset_factory(observations=observations, repeats=3)
    with (
        patch(
            "versatil.quantization.workflows.eager.load_float_policy_context",
            return_value=context,
        ),
        patch(
            "versatil.quantization.calibration.EpisodicDataset", return_value=dataset
        ),
        patch.object(
            compressor.deployment_backend,
            "export",
            wraps=compressor.deployment_backend.export,
        ) as deployment_export,
    ):
        output = Path(compressor.compress(hydra_config=config))
    metadata = json.loads(
        (output / CompressionFilename.COMPRESSION_METADATA.value).read_text()
    )
    assert metadata[CompressionMetadataKey.CALIBRATION_BATCHES.value] == 2
    assert metadata[CompressionMetadataKey.DEPLOYMENT_BACKEND.value] == "torch_inductor"
    selected = metadata[CompressionMetadataKey.QUANTIZATION_TARGETS.value][0]
    expected_names = {
        f"decoder.{name}"
        for name, module in policy.decoder.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    assert {layer["name"] for layer in selected["selected"]} == expected_names
    assert selected["weight_representations"] == dict.fromkeys(
        expected_names, "Int8Tensor"
    )
    assert selected["schema"] == (
        "versatil.quantization.schemas.smoothquant.SmoothQuantSchema"
    )
    assert selected["schema_parameters"] == {"alpha": "0.5"}
    assert selected["requires_calibration"] is True
    reloaded = torch.export.load(
        output / CompressionFilename.COMPRESSED_MODEL.value
    ).module()
    exportable = create_exportable_policy(policy=policy)
    inputs = exportable.export_metadata.prepare_inputs(
        observations=tuple(observations[key] for key in policy.input_keys)
    )  # (batch, ...)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        expected = exportable(*inputs)  # (batch, horizon, action_dim)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        actual = reloaded(*inputs)  # (batch, horizon, action_dim)
    torch.testing.assert_close(actual, expected)
    if compile_model:
        # Match execution modes; compilation can change dynamic INT8 rounding.
        compiled_reference = torch.compile(
            deployment_export.call_args.kwargs["model"],
            backend="inductor",
            fullgraph=True,
            options={"fallback_random": True},
        )
        compiled_reloaded = torch.compile(
            reloaded,
            backend="inductor",
            fullgraph=True,
            options={"fallback_random": True},
        )
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            compiled_expected = compiled_reference(
                *inputs
            )  # (batch, horizon, action_dim)
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            compiled_actual = compiled_reloaded(*inputs)  # (batch, horizon, action_dim)
        torch.testing.assert_close(compiled_actual, compiled_expected)


@pytest.mark.integration
def test_smoothquant_calibrates_and_converts_generated_token_linears(
    quantization_policy_factory: Callable[..., tuple[Policy, dict[str, torch.Tensor]]],
) -> None:
    policy, observations = quantization_policy_factory(
        family="tokens", device=torch.device("cpu"), dtype=torch.float32, batch_size=2
    )
    with torch.no_grad():
        policy.predict_from_processed_observation(
            observation=observations
        )  # (batch, length)
    workflow = EagerQuantizationWorkflow(
        targets=[
            EagerQuantizationModuleTarget(
                module_path="decoder",
                schema=SmoothQuantSchema(
                    base_config=Int8DynamicActivationInt8WeightConfig(version=2),
                    alpha=0.5,
                ),
            )
        ],
        is_qat=False,
    )
    calibration = CalibrationDataProvider(
        dataloader=DataLoader(
            [{SampleKey.OBSERVATION.value: observations}], batch_size=None
        ),
        observation_keys=policy.input_keys,
        num_calibration_steps=1,
        device=torch.device("cpu"),
    )
    calibration_batches, targets = workflow._apply_ptq(
        model=policy, calibration=calibration
    )
    assert calibration_batches == 1
    assert set(targets[0].weight_representations.values()) == {"Int8Tensor"}
    with torch.no_grad():
        result = policy.predict_from_processed_observation(
            observation=observations
        )  # (batch, length)
    tokens = result[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
    assert tokens.dtype == torch.int64
    assert tokens.shape == (2, 7)
    assert (tokens >= 0).all()
    assert (tokens < policy.tokenizer.action_tokenizer.vocab_size).all()


@pytest.mark.integration
@pytest.mark.parametrize(
    "preset, device, weight_type",
    [
        ("smoothquant_int8", "cpu", Int8Tensor),
        pytest.param(
            "smoothquant_int4",
            "cuda",
            Int4TilePackedTo4dTensor,
            marks=pytest.mark.requires_gpu,
        ),
    ],
)
@pytest.mark.parametrize(
    "family, scheduler_type",
    [
        ("flow", SchedulerType.DDIM.value),
        ("diffusion", SchedulerType.DDIM.value),
        ("diffusion", SchedulerType.DDPM.value),
    ],
)
def test_smoothquant_preset_calibrates_converts_and_reloads_denoising_policy(
    quantization_policy_factory: Callable[..., tuple[Policy, dict[str, torch.Tensor]]],
    smoothquant_context_factory: Callable[..., PolicyContext],
    calibration_dataset_factory: Callable[..., MagicMock],
    family: str,
    scheduler_type: str,
    preset: str,
    device: str,
    weight_type: type[torch.Tensor],
    tmp_path: Path,
) -> None:
    policy, observations = quantization_policy_factory(
        family=family,
        device=torch.device(device),
        dtype=torch.float32,
        batch_size=3,
        scheduler_type=scheduler_type,
        embedding_dimension=128 if preset == "smoothquant_int4" else 32,
    )
    context = smoothquant_context_factory(policy=policy, checkpoint_path=str(tmp_path))
    with initialize_config_dir(
        config_dir=str(get_hydra_configs_dir()), version_base=None
    ):
        config = compose(
            config_name=f"end_to_end_ptq/{preset}",
            overrides=[
                f"checkpoint_path={tmp_path}",
                f"output_directory={tmp_path / 'compressed'}",
                "calibration_steps=2",
            ],
        )
        compressor = hydra.utils.instantiate(config)
    dataset = calibration_dataset_factory(observations=observations, repeats=3)
    with (
        patch(
            "versatil.quantization.workflows.eager.load_float_policy_context",
            return_value=context,
        ),
        patch(
            "versatil.quantization.calibration.EpisodicDataset", return_value=dataset
        ),
    ):
        output = Path(compressor.compress(hydra_config=config))
    metadata = json.loads(
        (output / CompressionFilename.COMPRESSION_METADATA.value).read_text()
    )
    assert metadata[CompressionMetadataKey.CALIBRATION_BATCHES.value] == 2
    weights = [
        policy.get_submodule(layer["name"]).weight
        for target in metadata[CompressionMetadataKey.QUANTIZATION_TARGETS.value]
        for layer in target["selected"]
    ]
    if preset == "smoothquant_int4":
        assert (
            metadata[CompressionMetadataKey.QUANTIZATION_TARGETS.value][0]["group_size"]
            == 128
        )
    assert weights
    for weight in weights:
        assert isinstance(weight, weight_type)
        assert weight.act_pre_scale.shape == (weight.shape[-1],)
        assert torch.isfinite(weight.act_pre_scale).all()
        assert (weight.act_pre_scale > 0).all()
    export_metadata = PolicyExportMetadata.from_dict(
        metadata=metadata[CompressionMetadataKey.POLICY_EXPORT_METADATA.value]
    )
    assert export_metadata.precision == (
        PrecisionType.BF16_TRUE if preset == "smoothquant_int4" else None
    )
    inputs = export_metadata.prepare_inputs(
        observations=tuple(observations[key] for key in policy.input_keys)
    )  # (batch, ...)
    reloaded = torch.export.load(
        output / CompressionFilename.COMPRESSED_MODEL.value
    ).module()
    exportable = create_exportable_policy(policy=policy)
    for batch_size in (1, 3):
        batch_inputs = tuple(
            tensor[:batch_size] for tensor in inputs
        )  # each (3, ...) -> (batch, ...)
        with torch.no_grad():
            expected = exportable(*batch_inputs)  # each (batch, horizon, action_dim)
            actual = reloaded(*batch_inputs)  # each (batch, horizon, action_dim)
        torch.testing.assert_close(actual, expected)
        for action in actual:
            assert torch.isfinite(action).all()


@pytest.mark.integration
@pytest.mark.parametrize(
    "family, scheduler_type, decoder_architecture",
    [
        ("flow", SchedulerType.DDIM.value, "transformer"),
        ("diffusion", SchedulerType.DDIM.value, "transformer"),
        ("diffusion", SchedulerType.DDPM.value, "transformer"),
        ("tokens", SchedulerType.DDIM.value, "transformer"),
        ("flow", SchedulerType.DDIM.value, "unet"),
        ("diffusion", SchedulerType.DDIM.value, "unet"),
        ("diffusion", SchedulerType.DDPM.value, "unet"),
    ],
)
def test_policy_predictions_survive_export_and_reload(
    quantization_policy_factory: Callable[..., tuple[Policy, dict[str, torch.Tensor]]],
    family: str,
    scheduler_type: str,
    decoder_architecture: str,
    tmp_path: Path,
) -> None:
    policy, observations = quantization_policy_factory(
        family=family,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=2,
        scheduler_type=scheduler_type,
        decoder_architecture=decoder_architecture,
    )
    exportable = create_exportable_policy(policy=policy)
    inputs = exportable.export_metadata.prepare_inputs(
        observations=tuple(observations[key] for key in policy.input_keys)
    )  # (batch, ...)
    exported = export_policy(exportable=exportable, example_inputs=inputs)
    program = torch.export.export(exported, inputs, strict=False)
    artifact_path = tmp_path / "policy.pt2"
    torch.export.save(program, artifact_path)
    reloaded = torch.export.load(artifact_path).module()
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        expected = exportable(
            *inputs
        )  # actions: (batch, horizon, action_dim); tokens: (batch, length)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        actual = reloaded(
            *inputs
        )  # actions: (batch, horizon, action_dim); tokens: (batch, length)
    torch.testing.assert_close(actual, expected)


@pytest.mark.integration
@pytest.mark.parametrize(
    "family, scheduler_type",
    [
        ("flow", SchedulerType.DDIM.value),
        ("diffusion", SchedulerType.DDIM.value),
        ("diffusion", SchedulerType.DDPM.value),
    ],
)
@pytest.mark.parametrize("recipe", ["module_int8", "pt2e_dynamic", "pt2e_static"])
@pytest.mark.parametrize("decoder_architecture", ["transformer", "unet"])
@pytest.mark.filterwarnings("error:must run observer before calling calculate_qparams")
def test_quantized_denoising_policy_survives_export_and_reload(
    quantization_policy_factory: Callable[..., tuple[Policy, dict[str, torch.Tensor]]],
    family: str,
    scheduler_type: str,
    recipe: str,
    decoder_architecture: str,
    tmp_path: Path,
) -> None:
    policy, observations = quantization_policy_factory(
        family=family,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=2,
        scheduler_type=scheduler_type,
        decoder_architecture=decoder_architecture,
    )
    context = PolicyContext(
        policy=policy,
        config=MainConfig(),
        tokenizer=None,
        observation_space=policy.observation_space,
        observation_horizon=1,
        checkpoint_path=str(tmp_path),
        checkpoint_name="unused.ckpt",
    )
    exportable = create_exportable_policy(policy=policy)
    inputs = exportable.export_metadata.prepare_inputs(
        observations=tuple(observations[key] for key in exportable.observation_keys)
    )  # (batch, ...)
    if recipe == "module_int8":
        workflow = EagerQuantizationWorkflow(
            targets=[
                EagerQuantizationModuleTarget(
                    module_path="decoder",
                    quantize_config=Int8WeightOnlyConfig(version=2),
                )
            ]
        )
        result = workflow.quantize(
            context=context, exportable=exportable, calibration_steps=0
        )
        assert result.calibration_batches == 0
        expected_names = {
            f"decoder.{name}"
            for name, module in policy.decoder.named_modules()
            if isinstance(module, torch.nn.Linear)
        }
        assert expected_names
        assert {
            layer.name for layer in result.quantization_targets[0].selected
        } == expected_names
        assert result.quantization_targets[0].weight_representations == dict.fromkeys(
            expected_names, "Int8Tensor"
        )
        if decoder_architecture == "unet":
            convolution_weights = [
                module.weight
                for module in policy.decoder.modules()
                if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))
            ]
            assert convolution_weights
            assert all(
                type(weight) is torch.nn.Parameter and weight.dtype == torch.float32
                for weight in convolution_weights
            )
    else:
        workflow = PT2EQuantizationWorkflow(
            targets=[
                PT2EQuantizationModuleTarget(
                    module_path="decoder",
                    pt2e_backend=X86InductorBackend(
                        is_dynamic=recipe == "pt2e_dynamic"
                    ),
                )
            ]
        )
        calibration = None
        if recipe == "pt2e_static":
            calibration = CalibrationDataProvider(
                dataloader=DataLoader(
                    [{SampleKey.OBSERVATION.value: observations}], batch_size=None
                ),
                observation_keys=exportable.observation_keys,
                num_calibration_steps=1,
            )
        with patch.object(workflow, "_build_calibration", return_value=calibration):
            result = workflow.quantize(
                context=context,
                exportable=exportable,
                calibration_steps=1 if calibration is not None else 0,
            )
        weight_nodes = [
            node
            for node in result.quantized_model.graph.nodes
            if node.target
            == torch.ops.quantized_decomposed.dequantize_per_channel.default
        ]
        assert weight_nodes
        linear_nodes = [
            node
            for node in result.quantized_model.graph.nodes
            if node.target == torch.ops.aten.linear.default
        ]
        assert linear_nodes
        assert all(
            node.args[1].target
            == torch.ops.quantized_decomposed.dequantize_per_channel.default
            for node in linear_nodes
        )
        if decoder_architecture == "unet":
            convolution_nodes = [
                node
                for node in result.quantized_model.graph.nodes
                if node.target
                in (
                    torch.ops.aten.conv1d.default,
                    torch.ops.aten.conv_transpose1d.default,
                )
            ]
            assert convolution_nodes
            assert all(node.args[1].op == "get_attr" for node in convolution_nodes)
            assert all(
                result.quantized_model.get_parameter(node.args[1].target).dtype
                == torch.float32
                for node in convolution_nodes
            )
        for node in weight_nodes:
            quantized_weight = result.quantized_model.get_buffer(node.args[0].target)
            scales = result.quantized_model.get_buffer(node.args[1].target)
            assert scales.shape == (quantized_weight.shape[0],)
            assert torch.isfinite(scales).all()
            assert (scales > 0).all()
    program = torch.export.export(result.quantized_model, inputs, strict=False)
    artifact_path = tmp_path / "quantized_policy.pt2"
    torch.export.save(program, artifact_path)
    reloaded = torch.export.load(artifact_path).module()
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        expected = result.quantized_model(*inputs)  # (batch, horizon, action_dim)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        actual = reloaded(*inputs)  # (batch, horizon, action_dim)
    torch.testing.assert_close(actual, expected)
    for action in actual:
        assert torch.isfinite(action).all()
