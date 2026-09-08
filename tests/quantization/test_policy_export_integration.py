"""Tests for versatil.quantization workflow export and reload integration."""

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader
from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    Int8WeightOnlyConfig,
)
from torchao.quantization.quantize_.workflows.int8.int8_tensor import Int8Tensor

from versatil.configs.data.dataloader import DataLoaderConfig
from versatil.configs.main import MainConfig
from versatil.configs.paths import get_hydra_configs_dir
from versatil.data.constants import SampleKey
from versatil.data.episodic_dataset import EpisodicDataset
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.exportable.factory import create_exportable_policy
from versatil.models.layers.denoising.diffusion_process import SchedulerType
from versatil.models.policy import Policy
from versatil.post_training_compression.constants import (
    CompressionFilename,
    CompressionMetadataKey,
)
from versatil.post_training_compression.export import export_policy
from versatil.post_training_compression.policy_context import PolicyContext
from versatil.quantization.calibration import CalibrationDataProvider
from versatil.quantization.module_target import (
    EagerQuantizationModuleTarget,
    PT2EQuantizationModuleTarget,
)
from versatil.quantization.pt2e.backends.x86_inductor import X86InductorBackend
from versatil.quantization.schemas.smoothquant import SmoothQuantSchema
from versatil.quantization.workflows.eager import EagerQuantizationWorkflow
from versatil.quantization.workflows.pt2e import PT2EQuantizationWorkflow


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
    tmp_path: Path,
) -> None:
    policy, observations = quantization_policy_factory(
        family=family,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=2,
        scheduler_type=scheduler_type,
    )
    context = smoothquant_context_factory(policy=policy, checkpoint_path=str(tmp_path))
    with initialize_config_dir(
        config_dir=str(get_hydra_configs_dir()), version_base=None
    ):
        config = compose(config_name="end_to_end_ptq/smoothquant_int8")
        workflow = hydra.utils.instantiate(config.quantization)
    exportable = create_exportable_policy(policy=policy)
    dataset = calibration_dataset_factory(observations=observations, repeats=3)
    with patch(
        "versatil.quantization.calibration.EpisodicDataset", return_value=dataset
    ):
        result = workflow.quantize(
            context=context, exportable=exportable, calibration_steps=2
        )
    assert result.calibration_batches == 2
    weights = [
        module.weight
        for module in policy.decoder.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    assert weights
    for weight in weights:
        assert isinstance(weight, Int8Tensor)
        assert weight.act_pre_scale.shape == (weight.shape[-1],)
        assert torch.isfinite(weight.act_pre_scale).all()
        assert (weight.act_pre_scale > 0).all()
    inputs = exportable.export_metadata.prepare_inputs(
        observations=tuple(observations[key] for key in exportable.observation_keys)
    )  # (batch, ...)
    program = torch.export.export(result.quantized_model, inputs, strict=False)
    artifact = tmp_path / "smoothquant.pt2"
    torch.export.save(program, artifact)
    reloaded = torch.export.load(artifact).module()
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        expected = exportable(*inputs)  # (batch, horizon, action_dim)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        actual = reloaded(*inputs)  # (batch, horizon, action_dim)
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
