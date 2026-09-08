"""Tests for versatil.endpoints.post_training_compress module."""

import gc
import logging
import os
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import hydra
import numpy as np
import pytest
import torch
import torch._inductor.config as inductor_config
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from tokenizers import Tokenizer as HuggingFaceTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch._dynamo.utils import counters
from torchao.quantization import (
    Int8DynamicActivationInt8WeightConfig,
    quantize_,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer.composable_quantizer import (
    ComposableQuantizer,
)
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)
from transformers import PreTrainedTokenizerFast
from tso_robotics_sockets import CompressionType

import versatil.configs  # noqa: F401
from tests.conftest import get_test_device
from tests.endpoints.conftest import (
    HYDRA_CONFIG_DIR,
    build_tiny_overrides,
    get_free_port,
    resolve_dataset_type,
    start_mock_observation_server,
)
from versatil.configs.post_training_compression import (
    PostTrainingCompressorConfig,
    PreparationConfig,
)
from versatil.configs.quantization import (
    EagerQuantizationModuleTargetConfig,
    EagerQuantizationWorkflowConfig,
    PT2EQuantizationModuleTargetConfig,
    PT2EQuantizationWorkflowConfig,
    X86InductorBackendConfig,
)
from versatil.data.constants import Cameras, ObsKey, ProprioKey
from versatil.data.dataloader import get_dataloaders
from versatil.data.processing.transform import (
    normalize_observation,
    tokenize_observation,
)
from versatil.inference.inference_client import InferenceClient
from versatil.inference.policy_runtime.compressed_runtime import CompressedPolicyRuntime
from versatil.inference.policy_runtime.float_runtime import FloatPolicyRuntime
from versatil.inference.socket_transport import (
    SocketActionTransport,
    SocketObservationTransport,
)
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.factory.autoregressive_vla import (
    AutoregressiveVLADecoder,
)
from versatil.models.exportable.base import ExportablePolicy
from versatil.models.exportable.factory import create_exportable_policy
from versatil.post_training_compression.compressor import PostTrainingCompressor
from versatil.post_training_compression.constants import (
    CompressionFilename,
    QuantizationWorkflow,
)
from versatil.post_training_compression.export import (
    build_example_inputs,
    export_policy,
)
from versatil.post_training_compression.preparation import (
    fuse_all_conv_batchnorm_pairs,
    prepare_batchnorms_for_quantization,
)
from versatil.post_training_compression.pruning import (
    StructuredPruner,
    UnstructuredPruner,
)
from versatil.post_training_compression.serialization import save_compressed_model
from versatil.quantization.calibration import CalibrationDataProvider
from versatil.quantization.module_target import PT2EQuantizationModuleTarget
from versatil.quantization.pt2e.backends.x86_inductor import X86InductorBackend
from versatil.quantization.workflows.pt2e import PT2EQuantizationWorkflow
from versatil.workspace import Workspace

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_EPISODES = 3
TIMESTEPS_PER_EPISODE = 15

TRAINING_DEVICE = get_test_device()

COMMON_OVERRIDES = [
    "task.dataloader.batch_size=2",
    "task.dataloader.num_workers=1",
    "task.dataloader.val_ratio=0.0",
    "training.num_epochs=1",
    "experiment.use_wandb=false",
    "experiment.name=ptq_test",
    f"experiment.device={TRAINING_DEVICE.type}",
]

PTQ_CONFIG_DIR = Path(HYDRA_CONFIG_DIR) / "end_to_end_ptq"
PTQ_CONFIG_NAMES = [
    f"end_to_end_ptq/{p.stem}"
    for p in sorted(PTQ_CONFIG_DIR.glob("*.yaml"))
    if "example" not in p.stem
]
PTQ_X86_CONFIG_NAME = "end_to_end_ptq/unstructured_prune_x86"
PTQ_EAGER_XNNPACK_CONFIG_NAME = "end_to_end_ptq/eager_xnnpack"
PTQ_PT2E_XNNPACK_CONFIG_NAME = "end_to_end_ptq/pt2e_xnnpack"

PTQ_TEST_CONFIGS = [
    "end_to_end_training_runs/libero_lerobot/bcat_language",
    "end_to_end_training_runs/libero_lerobot/bcat",
    "end_to_end_training_runs/libero_lerobot/act",
    "end_to_end_training_runs/libero_lerobot/flow_dit_cross_attention",
]

GLOBAL_PT2E_PARAMS = [
    pytest.param(config_name, id=config_name.split("/")[-1])
    for config_name in PTQ_TEST_CONFIGS
]

LEROBOT_METADATA_PATCH = patch(
    "versatil.data.raw.schemas.lerobot.LeRobotDatasetMetadataV30.__init__",
    lambda self, dataset_path: setattr(self, "dataset_path", dataset_path),
)


@pytest.fixture(autouse=True, scope="session")
def _configure_inductor():
    """Set inductor config once per session."""
    original_freezing = os.environ.get("TORCHINDUCTOR_FREEZING")
    original_cpp_wrapper = inductor_config.cpp_wrapper
    original_freezing_setting = inductor_config.freezing
    os.environ["TORCHINDUCTOR_FREEZING"] = "1"
    inductor_config.cpp_wrapper = True
    yield
    if original_freezing is None:
        os.environ.pop("TORCHINDUCTOR_FREEZING", None)
    else:
        os.environ["TORCHINDUCTOR_FREEZING"] = original_freezing
    inductor_config.cpp_wrapper = original_cpp_wrapper
    inductor_config.freezing = original_freezing_setting


@pytest.fixture
def trained_checkpoint(
    tmp_path: Path, synthetic_zarr_factory: Callable[..., str]
) -> Callable[..., Path]:
    """Train a configured policy and return its checkpoint directory."""

    def factory(
        config_name: str = PTQ_TEST_CONFIGS[0],
        extra_overrides: list[str] | None = None,
        action_values: dict[str, list[float]] | None = None,
        configure: Callable[[DictConfig], None] | None = None,
    ) -> Path:
        dataset_type = resolve_dataset_type(config_name)
        zarr_path = str(tmp_path / "data.zarr")
        checkpoint_dir = str(tmp_path / "checkpoints")

        synthetic_zarr_factory(
            dataset_type=dataset_type,
            zarr_path=zarr_path,
            image_height=IMAGE_HEIGHT,
            image_width=IMAGE_WIDTH,
            num_episodes=NUM_EPISODES,
            timesteps_per_episode=TIMESTEPS_PER_EPISODE,
            action_values=action_values,
        )

        decoder_overrides = build_tiny_overrides(config_name)
        all_overrides = (
            COMMON_OVERRIDES
            + decoder_overrides
            + [
                f"experiment.checkpoint_folder={checkpoint_dir}",
                f"task.dataset_schema.zarr_path={zarr_path}",
            ]
            + (extra_overrides or [])
        )

        with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
            yaml_config = compose(
                config_name=config_name,
                overrides=all_overrides,
            )
            if configure is not None:
                configure(yaml_config)
            with LEROBOT_METADATA_PATCH:
                config = hydra.utils.instantiate(yaml_config)

        config.policy.to(torch.device("cpu"))

        with patch("versatil.workspace.HydraConfig") as mock_hydra:
            mock_hydra.get.return_value = MagicMock()
            mock_hydra.get.return_value.job.config_name = "test_ptq"
            workspace = Workspace(config, original_yaml_config=yaml_config)
            workspace.run()

        output_dir = Path(checkpoint_dir) / "test_ptq" / "ptq_test"
        assert (output_dir / "last.ckpt").exists()
        del workspace
        gc.collect()
        return output_dir

    return factory


@pytest.fixture
def rgb_policy_configuration() -> Callable[[DictConfig], None]:
    """Keep camera observations for the RGB-only policy regression."""

    def configure(config: DictConfig) -> None:
        metadata = config.task.observation_space.observations_metadata
        config.task.observation_space.observations_metadata = {
            Cameras.AGENTVIEW.value: metadata["${cameras:AGENTVIEW}"],
            Cameras.EYE_IN_HAND.value: metadata["${cameras:EYE_IN_HAND}"],
        }

    return configure


@pytest.fixture
def trained_binned_checkpoint(
    trained_checkpoint: Callable[..., Path],
) -> Callable[[], Path]:
    """Build a factory that trains a proprioceptive GPT on fixed action chunks."""

    def configure(config: DictConfig) -> None:
        observation_metadata = config.task.observation_space.observations_metadata
        action_metadata = config.task.action_space.actions_metadata
        config.task.observation_space.observations_metadata = {
            ProprioKey.EE_POS.value: observation_metadata["${proprio_key:EE_POS}"]
        }
        config.task.action_space.actions_metadata = {
            ProprioKey.EE_POS_ACTION.value: action_metadata[
                "${proprio_key:EE_POS_ACTION}"
            ]
        }
        config.policy.encoding_pipeline.encoders = {}

    def factory() -> Path:
        tokenizer = "task.dataloader.tokenization.action_tokenizer"
        return trained_checkpoint(
            config_name="end_to_end_training_runs/libero_lerobot/gpt_transformer",
            extra_overrides=[
                "task/observation_space=libero_rgb_proprio_lerobot",
                "task/dataloader/tokenization=action_fast",
                "task.dataloader.tokenization.observation_tokenizer=null",
                f"{tokenizer}.action_discretizer.type=${{action_discretizer:BINNED}}",
                f"{tokenizer}.action_discretizer.num_bins=16",
                f"{tokenizer}.max_token_len=7",
                f"policy.decoder.input_keys=[{ProprioKey.EE_POS.value}]",
                "policy.decoder.embedding_dimension=32",
                "policy.decoder.number_of_heads=2",
                "policy.decoder.number_of_key_value_heads=2",
                "policy.decoder.number_of_layers=1",
                "policy.decoder.feedforward_dimension=64",
                "policy.decoder.max_seq_len=32",
                "policy.decoder.dropout_rate=0.0",
                "policy.decoder.attention_dropout=0.0",
                "policy.decoder.temperature=1.0",
                "policy.decoder.learnable_temperature=false",
                "policy.loss.loss_modules.token_loss.label_smoothing=0.0",
                "task.prediction_horizon=2",
                "task.dataloader.num_workers=0",
                "training.num_epochs=5",
                "training.optimizer.lr=0.01",
                "training.optimizer.param_groups=[]",
                "training.use_ema=false",
                "experiment.device=cpu",
            ],
            action_values={ProprioKey.EE_POS_ACTION.value: [0.25, -0.5, 0.75]},
            configure=configure,
        )

    return factory


@pytest.fixture
def binned_policy_observation_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Build a factory for batched three-dimensional end-effector observations."""

    def factory(batch_size: int) -> dict[str, torch.Tensor]:
        return {
            ProprioKey.EE_POS.value: torch.from_numpy(
                rng.standard_normal(size=(batch_size, 1, 3)).astype(np.float32)
            )  # (batch, observation_horizon, position_dim)
        }

    return factory


@pytest.fixture
def local_language_tokenizer_factory(tmp_path: Path) -> Callable[..., Path]:
    """Save a local language tokenizer for OpenVLA observation and action tokens."""

    def factory(vocabulary_size: int) -> Path:
        vocabulary = {
            "[PAD]": 0,
            "[EOS]": 1,
            "[UNK]": 2,
            "[BOS]": 3,
            "pick": 4,
            "up": 5,
            "object": 6,
        }
        vocabulary.update(
            {
                f"token_{index}": index
                for index in range(len(vocabulary), vocabulary_size)
            }
        )
        tokenizer = HuggingFaceTokenizer(WordLevel(vocab=vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        language_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            pad_token="[PAD]",
        )
        tokenizer_directory = tmp_path / "local_language_tokenizer"
        language_tokenizer.save_pretrained(tokenizer_directory)
        return tokenizer_directory

    return factory


@pytest.fixture
def trained_openvla_checkpoint(
    trained_checkpoint: Callable[..., Path],
    tiny_prismatic_configuration_factory: Callable[..., Path],
    local_language_tokenizer_factory: Callable[..., Path],
) -> Callable[[], Path]:
    """Train a small OpenVLA policy with local image-language and action vocabularies."""

    def configure(config: DictConfig) -> None:
        metadata = config.task.observation_space.observations_metadata
        config.task.observation_space.observations_metadata = {
            Cameras.AGENTVIEW.value: metadata["${cameras:AGENTVIEW}"],
            ObsKey.LANGUAGE.value: metadata["${obs_key:LANGUAGE}"],
        }
        camera_metadata = config.task.observation_space.observations_metadata[
            Cameras.AGENTVIEW.value
        ]
        camera_metadata.image_height = IMAGE_HEIGHT
        camera_metadata.image_width = IMAGE_WIDTH
        action_metadata = config.task.action_space.actions_metadata
        config.task.action_space.actions_metadata = {
            ProprioKey.EE_POS_ACTION.value: action_metadata[
                "${proprio_key:EE_POS_ACTION}"
            ]
        }

    def factory() -> Path:
        model_directory = tiny_prismatic_configuration_factory(
            hidden_dimension=32, vocabulary_size=128, max_text_length=4
        )
        tokenizer_directory = local_language_tokenizer_factory(vocabulary_size=128)
        observation_tokenizer = "task.dataloader.tokenization.observation_tokenizer"
        action_tokenizer = "task.dataloader.tokenization.action_tokenizer"
        backbone = "policy.decoder.vlm_backbone"
        return trained_checkpoint(
            config_name="end_to_end_training_runs/libero_lerobot/openvla",
            extra_overrides=[
                f"{observation_tokenizer}.tokenizer_model={tokenizer_directory}",
                f"{observation_tokenizer}.max_token_len=4",
                f"{observation_tokenizer}.prompt_template='{{instruction}}'",
                f"{action_tokenizer}.token_id_mapping.language_tokenizer_model={tokenizer_directory}",
                f"{action_tokenizer}.action_discretizer.num_bins=16",
                f"{action_tokenizer}.max_token_len=4",
                f"{backbone}.model_name={model_directory}",
                f"{backbone}.pretrained=false",
                f"{backbone}.frozen=false",
                f"{backbone}.input_keys=[{Cameras.AGENTVIEW.value}]",
                f"{backbone}.lora_config=null",
                f"{backbone}.gradient_checkpointing=false",
                "policy.decoder.max_seq_len=32",
                "task.prediction_horizon=1",
                "task.dataloader.num_workers=0",
                "training.num_epochs=5",
                "training.gradient_accumulate_every=1",
                "training.optimizer.lr=0.01",
                "training.optimizer.param_groups=[]",
                "training.lr_warmup_steps=0",
                "training.use_ema=false",
                "experiment.device=cpu",
                "experiment.precision=${precision:FP32}",
            ],
            action_values={ProprioKey.EE_POS_ACTION.value: [0.25, -0.5, 0.75]},
            configure=configure,
        )

    return factory


@pytest.fixture
def openvla_observation_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor | list[list[str]]]]:
    """Create a camera observation and language instruction for each batch item."""

    def factory(batch_size: int) -> dict[str, torch.Tensor | list[list[str]]]:
        return {
            Cameras.AGENTVIEW.value: torch.from_numpy(
                rng.uniform(0, 1, size=(batch_size, 1, 3, 32, 32)).astype(np.float32)
            ),  # (batch, observation_horizon, channels, height, width)
            ObsKey.LANGUAGE.value: [["pick up object"] for _ in range(batch_size)],
        }

    return factory


@pytest.fixture
def compression_pipeline(trained_checkpoint):
    """Load policy, create calibration and exportable. Cleans up after test."""
    created = []

    def factory(
        config_name: str = PTQ_TEST_CONFIGS[0],
    ) -> tuple[FloatPolicyRuntime, CalibrationDataProvider, ExportablePolicy]:
        output_dir = trained_checkpoint(config_name=config_name)
        with LEROBOT_METADATA_PATCH:
            policy_loader = FloatPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=str(output_dir),
                checkpoint_name="last.ckpt",
            )

        exportable = create_exportable_policy(policy=policy_loader.policy)

        with LEROBOT_METADATA_PATCH:
            train_loader, _, _, _, _ = get_dataloaders(config=policy_loader.config)

        calibration = CalibrationDataProvider(
            dataloader=train_loader,
            observation_keys=exportable.observation_keys,
            num_calibration_steps=3,
        )
        created.append(policy_loader)
        return policy_loader, calibration, exportable

    yield factory

    for obj in created:
        del obj
    gc.collect()


def _get_float_outputs(
    exportable: ExportablePolicy,
    example_inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Run float model and return outputs for divergence comparison."""
    with torch.no_grad():
        return exportable(*example_inputs)


def _save_and_verify_inference(
    compressed_model: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    exportable: ExportablePolicy,
    policy: torch.nn.Module,
    output_dir: Path,
    tmp_path: Path,
    float_outputs: tuple[torch.Tensor, ...],
    quantization_workflow: str,
    expect_divergence: bool = True,
) -> None:
    """Save compressed model, verify files exist, verify inference, check divergence."""
    compressed_dir = str(tmp_path / "compressed")

    with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
        ptq_config = compose(
            config_name="end_to_end_ptq/unstructured_prune_x86.yaml",
            overrides=[f"checkpoint_path={str(output_dir)}"],
        )

    pt2e_backend_config = None
    if quantization_workflow == QuantizationWorkflow.PT2E.value:
        pt2e_backend_config = OmegaConf.to_container(
            ptq_config.quantization.targets[0].pt2e_backend,
            resolve=True,
        )

    save_compressed_model(
        converted_model=compressed_model,
        example_inputs=example_inputs,
        save_directory=compressed_dir,
        input_keys=exportable.observation_keys,
        output_keys=exportable.action_keys,
        normalizer=policy.normalizer,
        training_checkpoint_path=str(output_dir),
        quantization_config=ptq_config,
        quantization_workflow=quantization_workflow,
        pt2e_backend_config=pt2e_backend_config,
        export_metadata=exportable.export_metadata,
    )

    assert (Path(compressed_dir) / "compressed_policy.pt2").exists()
    assert (Path(compressed_dir) / "normalizer.pt").exists()
    assert (Path(compressed_dir) / "compression_metadata.json").exists()

    # Verify compressed model produces finite outputs
    with torch.no_grad():
        compressed_outputs = compressed_model(*example_inputs)
    assert all(t.isfinite().all() for t in compressed_outputs)
    if expect_divergence:
        outputs_changed = any(
            not torch.equal(compressed, original)
            for compressed, original in zip(
                compressed_outputs, float_outputs, strict=True
            )
        )
        assert outputs_changed, (
            "Compressed outputs identical to float — compression may have failed"
        )

    # Verify compressed inference via mock server
    with LEROBOT_METADATA_PATCH:
        compressed_runtime = CompressedPolicyRuntime(
            device=torch.device("cpu"),
            checkpoint_path=compressed_dir,
        )

    assert compressed_runtime.input_keys == exportable.observation_keys
    assert compressed_runtime.output_keys == exportable.action_keys

    port = get_free_port()
    server = start_mock_observation_server(
        observation_space=compressed_runtime.observation_space,
        port=port,
    )
    try:
        client = InferenceClient(
            policy_runtime=compressed_runtime,
            observation_transport=SocketObservationTransport(
                server_address="127.0.0.1",
                server_port=port,
            ),
            action_transport=SocketActionTransport(
                server_address="127.0.0.1",
                server_port=port,
            ),
            compression_type=CompressionType.RAW.value,
        )
        status = client.step()
        assert status == "continue"
        action_metadata = client.action_postprocessor.build_action_metadata()
        assert len(action_metadata) > 0
    finally:
        server.stop()


def _prepare_backbones(policy: torch.nn.Module) -> None:
    """Apply BN preparation and fusion to all encoder backbones."""
    prepare_batchnorms_for_quantization(policy)
    fuse_all_conv_batchnorm_pairs(policy)


def _build_backbone_quantizers(
    policy: torch.nn.Module,
) -> list[X86InductorQuantizer]:
    """Create per-backbone static PT2E quantizers."""
    quantizers = []
    for name, encoder in policy.encoding_pipeline.encoders.items():
        if hasattr(encoder, "backbone"):
            quantizer = X86InductorQuantizer()
            quantizer.set_module_name_qconfig(
                f"encoding_pipeline.encoders.{name}.backbone",
                get_default_x86_inductor_quantization_config(is_dynamic=False),
            )
            quantizers.append(quantizer)
    assert len(quantizers) > 0
    return quantizers


def _prune_backbones(policy: torch.nn.Module) -> None:
    """Prune all encoder backbones and verify sparsity."""
    for _, encoder in policy.encoding_pipeline.encoders.items():
        if hasattr(encoder, "backbone"):
            prepare_batchnorms_for_quantization(encoder.backbone)
            fuse_all_conv_batchnorm_pairs(encoder.backbone)
            pruner = UnstructuredPruner(amount=0.3)
            _, zeroed = pruner.prune(module=encoder.backbone)
            assert zeroed > 0


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("config_name", GLOBAL_PT2E_PARAMS)
class TestGlobalPT2EQuantization:
    def test_global_pt2e_quantization(
        self, config_name, tmp_path, compression_pipeline
    ):
        policy_loader, calibration, exportable = compression_pipeline(
            config_name=config_name,
        )
        policy = policy_loader.policy
        output_dir = Path(policy_loader.checkpoint_path)

        _prepare_backbones(policy)
        observation = next(iter(calibration))
        example_inputs = exportable.export_metadata.prepare_inputs(
            observations=tuple(observation[key] for key in exportable.observation_keys)
        )  # (batch, ...)
        float_outputs = _get_float_outputs(
            exportable=exportable,
            example_inputs=example_inputs,
        )

        exported = export_policy(exportable=exportable, example_inputs=example_inputs)

        quantizer = X86InductorBackend(is_dynamic=False).create_quantizer(
            module_path=""
        )
        prepared = prepare_pt2e(exported, quantizer)

        with torch.no_grad():
            for observation in calibration:
                inputs = exportable.export_metadata.prepare_inputs(
                    observations=tuple(
                        observation[key] for key in exportable.observation_keys
                    )
                )  # (batch, ...)
                prepared(*inputs)  # (batch, horizon, action_dim)

        quantized = convert_pt2e(prepared)

        graph_str = str(quantized.graph)
        assert any(
            keyword in graph_str
            for keyword in ["quantize_per_tensor", "dequantize_per_tensor"]
        ), "Quantized graph has no quantize/dequantize ops"

        _save_and_verify_inference(
            compressed_model=quantized,
            example_inputs=example_inputs,
            exportable=exportable,
            policy=policy,
            output_dir=output_dir,
            tmp_path=tmp_path,
            float_outputs=float_outputs,
            quantization_workflow=QuantizationWorkflow.PT2E.value,
        )


@pytest.mark.slow
@pytest.mark.integration
class TestPerModulePT2EWithPruning:
    @pytest.mark.parametrize("apply_pruning", [False, True])
    def test_pt2e_backbones_with_optional_pruning(
        self,
        apply_pruning,
        tmp_path,
        compression_pipeline,
    ):
        policy_loader, calibration, exportable = compression_pipeline()
        policy = policy_loader.policy
        output_dir = Path(policy_loader.checkpoint_path)

        if apply_pruning:
            _prune_backbones(policy)
        else:
            _prepare_backbones(policy)

        observation = next(iter(calibration))
        example_inputs = exportable.export_metadata.prepare_inputs(
            observations=tuple(observation[key] for key in exportable.observation_keys)
        )  # (batch, ...)
        float_outputs = _get_float_outputs(
            exportable=exportable,
            example_inputs=example_inputs,
        )

        exported = export_policy(exportable=exportable, example_inputs=example_inputs)

        quantizers = _build_backbone_quantizers(policy)
        composed = ComposableQuantizer(quantizers)
        prepared = prepare_pt2e(exported, composed)

        with torch.no_grad():
            for observation in calibration:
                inputs = exportable.export_metadata.prepare_inputs(
                    observations=tuple(
                        observation[key] for key in exportable.observation_keys
                    )
                )  # (batch, ...)
                prepared(*inputs)  # (batch, horizon, action_dim)

        converted = convert_pt2e(prepared)

        static_ops = str(converted.graph).count("quantize_per_tensor")
        assert static_ops > 0

        _save_and_verify_inference(
            compressed_model=converted,
            example_inputs=example_inputs,
            exportable=exportable,
            policy=policy,
            output_dir=output_dir,
            tmp_path=tmp_path,
            float_outputs=float_outputs,
            quantization_workflow=QuantizationWorkflow.PT2E.value,
        )


@pytest.mark.slow
@pytest.mark.integration
class TestGlobalEagerPTQDynamic:
    @pytest.mark.parametrize(
        "embedding_dimension, expect_divergence",
        [
            (16, False),
            (32, True),
        ],
        ids=["skip_small_layers", "quantize_large_layers"],
    )
    def test_eager_before_export(
        self,
        embedding_dimension,
        expect_divergence,
        tmp_path,
        trained_checkpoint,
    ):
        output_dir = trained_checkpoint(
            config_name=PTQ_TEST_CONFIGS[0],
            extra_overrides=[
                f"policy.decoder.embedding_dimension={embedding_dimension}",
            ],
        )
        with LEROBOT_METADATA_PATCH:
            policy_loader = FloatPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=str(output_dir),
                checkpoint_name="last.ckpt",
            )
        policy = policy_loader.policy
        exportable = create_exportable_policy(policy=policy)

        with LEROBOT_METADATA_PATCH:
            train_loader, _, _, _, _ = get_dataloaders(config=policy_loader.config)
        calibration = CalibrationDataProvider(
            dataloader=train_loader,
            observation_keys=exportable.observation_keys,
            num_calibration_steps=3,
        )

        observation = next(iter(calibration))
        example_inputs = exportable.export_metadata.prepare_inputs(
            observations=tuple(observation[key] for key in exportable.observation_keys)
        )  # (batch, ...)
        float_outputs = _get_float_outputs(
            exportable=exportable,
            example_inputs=example_inputs,
        )

        # quantize_() must run on eager model before export
        quantize_(
            policy,
            Int8DynamicActivationInt8WeightConfig(),
        )

        exported = export_policy(exportable=exportable, example_inputs=example_inputs)

        _save_and_verify_inference(
            compressed_model=exported,
            example_inputs=example_inputs,
            exportable=exportable,
            policy=policy,
            output_dir=output_dir,
            tmp_path=tmp_path,
            float_outputs=float_outputs,
            quantization_workflow=QuantizationWorkflow.EAGER.value,
            expect_divergence=expect_divergence,
        )


@pytest.mark.slow
@pytest.mark.integration
class TestCompiledTraining:
    def test_training_with_torch_compile(self, trained_checkpoint, caplog):
        with caplog.at_level(logging.INFO):
            output_dir = trained_checkpoint(
                extra_overrides=[
                    "training.compile=true",
                    "training.compile_mode=${compile_mode:DEFAULT}",
                ],
            )

        assert (output_dir / "last.ckpt").exists()
        assert "Compiling policy with torch.compile" in caplog.text


@pytest.mark.slow
@pytest.mark.integration
class TestBuildExampleInputsFallback:
    def test_build_example_inputs_with_real_policy(self, compression_pipeline):
        policy_loader, _, exportable = compression_pipeline()

        example_inputs = build_example_inputs(
            exportable=exportable,
            observation_space=policy_loader.observation_space,
            observation_horizon=policy_loader.config.task.observation_horizon,
            tokenizer=policy_loader.tokenizer,
        )

        assert len(example_inputs) == len(exportable.observation_keys)
        assert all(isinstance(t, torch.Tensor) for t in example_inputs)
        assert all(t.shape[0] == 2 for t in example_inputs)


@pytest.mark.slow
@pytest.mark.integration
class TestGlobalFallbackPipeline:
    def test_prepare_prune_export_on_full_policy(self, tmp_path, compression_pipeline):
        policy_loader, calibration, exportable = compression_pipeline()
        policy = policy_loader.policy

        prepare_batchnorms_for_quantization(policy)
        fuse_all_conv_batchnorm_pairs(policy)

        pruner = UnstructuredPruner(amount=0.3)
        _, zeroed = pruner.prune(module=policy)
        assert zeroed > 0

        observation = next(iter(calibration))
        example_inputs = exportable.export_metadata.prepare_inputs(
            observations=tuple(observation[key] for key in exportable.observation_keys)
        )  # (batch, ...)
        exported = export_policy(exportable=exportable, example_inputs=example_inputs)

        with torch.no_grad():
            outputs = exported(*example_inputs)
        assert all(t.isfinite().all() for t in outputs)

    def test_prepare_structured_and_unstructured_then_export(
        self, tmp_path, compression_pipeline
    ):
        policy_loader, calibration, exportable = compression_pipeline()
        policy = policy_loader.policy

        prepare_batchnorms_for_quantization(policy)
        fuse_all_conv_batchnorm_pairs(policy)

        StructuredPruner(amount=0.2).prune(module=policy)
        _, zeroed = UnstructuredPruner(amount=0.3).prune(module=policy)
        assert zeroed > 0

        observation = next(iter(calibration))
        example_inputs = exportable.export_metadata.prepare_inputs(
            observations=tuple(observation[key] for key in exportable.observation_keys)
        )  # (batch, ...)
        exported = export_policy(exportable=exportable, example_inputs=example_inputs)

        with torch.no_grad():
            outputs = exported(*example_inputs)
        assert all(t.isfinite().all() for t in outputs)


@pytest.mark.slow
@pytest.mark.integration
class TestCompressorEndToEnd:
    @pytest.mark.parametrize("quantized", [False, True], ids=["float", "int8"])
    def test_trained_openvla_checkpoint_compression_reconstructs_actions(
        self,
        trained_openvla_checkpoint: Callable[[], Path],
        openvla_observation_factory: Callable[
            ..., dict[str, torch.Tensor | list[list[str]]]
        ],
        quantized: bool,
        tmp_path: Path,
    ) -> None:
        checkpoint_directory = trained_openvla_checkpoint()
        with LEROBOT_METADATA_PATCH:
            reference = FloatPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=str(checkpoint_directory),
                checkpoint_name="last.ckpt",
                compile_model=False,
            )
        assert isinstance(reference.policy.decoder, AutoregressiveVLADecoder)
        quantization_config = None
        if quantized:
            quantization_config = EagerQuantizationWorkflowConfig(
                targets=[
                    EagerQuantizationModuleTargetConfig(
                        module_path="decoder.vlm_backbone.language_model",
                        quantize_config={
                            "_target_": "torchao.quantization.Int8WeightOnlyConfig"
                        },
                    )
                ],
                is_qat=False,
            )
            quantization = hydra.utils.instantiate(quantization_config)
            calibration_batches, targets = quantization._apply_ptq(
                model=reference.policy
            )
            assert calibration_batches == 0
            assert targets[0].selected
            assert set(targets[0].weight_representations.values()) == {"Int8Tensor"}
        observations = openvla_observation_factory(batch_size=2)
        expected_actions = reference.run_inference(obs_dict=observations)
        processed = normalize_observation(
            observation=observations,
            normalizer=reference.policy.normalizer,
            observation_space=reference.observation_space,
        )  # camera: (batch, horizon, channels, height, width)
        processed = tokenize_observation(
            observation=processed,
            obs_tokenizer=reference.tokenizer.observation_tokenizer,
            batched=True,
        )  # tokens and padding: (batch, horizon, text_length)
        with torch.no_grad():
            expected_tokens = reference.policy.predict_from_processed_observation(
                observation=processed
            )[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]  # (batch, token_length)
        hydra_config = OmegaConf.structured(
            PostTrainingCompressorConfig(
                checkpoint_path=str(checkpoint_directory),
                checkpoint_name="last.ckpt",
                modules=[],
                preparation=PreparationConfig(
                    replace_frozen_batchnorm=False, fuse_conv_batchnorm=False
                ),
                quantization=quantization_config,
                calibration_steps=0,
                output_directory=str(tmp_path / "compressed_openvla"),
            )
        )
        with LEROBOT_METADATA_PATCH:
            compressor = hydra.utils.instantiate(hydra_config)
            output = compressor.compress(hydra_config=hydra_config)
            runtime = CompressedPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=output,
                compile_model=False,
            )
        with torch.no_grad():
            actual_tokens = runtime._compressed_model(
                *[processed[key] for key in runtime.input_keys]
            )[0]  # (batch, maximum_token_length)
        torch.testing.assert_close(actual_tokens, expected_tokens)
        torch.testing.assert_close(
            runtime.run_inference(obs_dict=observations), expected_actions
        )
        for batch_size in [1, 3]:
            observations = openvla_observation_factory(batch_size=batch_size)
            torch.testing.assert_close(
                runtime.run_inference(obs_dict=observations),
                reference.run_inference(obs_dict=observations),
            )

    @pytest.mark.parametrize(
        "workflow",
        [
            QuantizationWorkflow.NONE,
            QuantizationWorkflow.EAGER,
            QuantizationWorkflow.PT2E,
        ],
        ids=["float", "int8", "pt2e_dynamic"],
    )
    def test_binned_checkpoint_compression_reconstructs_trained_actions(
        self,
        trained_binned_checkpoint: Callable[[], Path],
        binned_policy_observation_factory: Callable[..., dict[str, torch.Tensor]],
        workflow: QuantizationWorkflow,
        tmp_path: Path,
    ) -> None:
        checkpoint_directory = trained_binned_checkpoint()
        with LEROBOT_METADATA_PATCH:
            reference = FloatPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=str(checkpoint_directory),
                checkpoint_name="last.ckpt",
                compile_model=False,
            )
        observations = binned_policy_observation_factory(batch_size=2)
        normalized_observations = normalize_observation(
            observation=observations,
            normalizer=reference.policy.normalizer,
            observation_space=reference.observation_space,
        )  # each observation: (batch, observation_horizon, position_dim)
        with torch.no_grad():
            predictions = reference.policy.predict_from_processed_observation(
                observation=normalized_observations
            )  # (batch, token_length)
        tokens = predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
        assert tokens.shape == (2, 7)
        expected_tokens = torch.tensor(
            [[8, 8, 8, 8, 8, 8, 16], [8, 8, 8, 8, 8, 8, 16]], dtype=torch.long
        )  # (batch, token_length)
        torch.testing.assert_close(tokens, expected_tokens)
        quantization_config = None
        if workflow == QuantizationWorkflow.EAGER:
            quantization_config = EagerQuantizationWorkflowConfig(
                targets=[
                    EagerQuantizationModuleTargetConfig(
                        module_path="decoder",
                        quantize_config={
                            "_target_": "torchao.quantization.Int8WeightOnlyConfig"
                        },
                    )
                ],
                is_qat=False,
            )
            quantization = hydra.utils.instantiate(quantization_config)
            calibration_batches, targets = quantization._apply_ptq(
                model=reference.policy
            )
            assert calibration_batches == 0
            assert targets[0].selected
            assert set(targets[0].weight_representations.values()) == {"Int8Tensor"}
        elif workflow == QuantizationWorkflow.PT2E:
            quantization_config = PT2EQuantizationWorkflowConfig(
                targets=[
                    PT2EQuantizationModuleTargetConfig(
                        module_path="decoder",
                        pt2e_backend=X86InductorBackendConfig(is_dynamic=True),
                    )
                ]
            )
        expected_actions = reference.run_inference(obs_dict=observations)
        trained_action = torch.tensor(
            [0.3125, -0.4375, 0.8125], dtype=torch.float32
        ).reshape(1, 1, 3)  # (position_dim,) -> (1, 1, position_dim)
        torch.testing.assert_close(
            expected_actions[ProprioKey.EE_POS_ACTION.value],
            trained_action.expand(
                2, 2, 3
            ),  # (1, 1, position_dim) -> (batch, horizon, position_dim)
        )
        compressed_directory = tmp_path / "compressed_binned"
        hydra_config = OmegaConf.structured(
            PostTrainingCompressorConfig(
                checkpoint_path=str(checkpoint_directory),
                checkpoint_name="last.ckpt",
                modules=[],
                preparation=PreparationConfig(
                    replace_frozen_batchnorm=False, fuse_conv_batchnorm=False
                ),
                quantization=quantization_config,
                calibration_steps=0,
                output_directory=str(compressed_directory),
            )
        )
        with LEROBOT_METADATA_PATCH:
            compressor = hydra.utils.instantiate(hydra_config)
            output = compressor.compress(hydra_config=hydra_config)
            runtime = CompressedPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=output,
                compile_model=workflow == QuantizationWorkflow.PT2E,
            )
        lowered_linears = counters["inductor"]["qlinear_unary_lower_count"]
        actual_actions = runtime.run_inference(obs_dict=observations)
        torch.testing.assert_close(actual_actions, expected_actions)
        if workflow == QuantizationWorkflow.PT2E:
            assert counters["inductor"]["qlinear_unary_lower_count"] > lowered_linears
        assert runtime.output_keys == [DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
        for batch_size in [1, 3]:
            batch = binned_policy_observation_factory(batch_size=batch_size)
            torch.testing.assert_close(
                runtime.run_inference(obs_dict=batch),
                reference.run_inference(obs_dict=batch),
            )

    def test_compress_full_pipeline_with_pt2e(self, tmp_path, trained_checkpoint):
        output_dir = trained_checkpoint()
        compressed_dir = str(tmp_path / "compressed_output")

        with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
            hydra_config = compose(
                config_name=PTQ_X86_CONFIG_NAME,
                overrides=[f"checkpoint_path={str(output_dir)}"],
            )
        compressor = PostTrainingCompressor(
            checkpoint_path=str(output_dir),
            checkpoint_name="last.ckpt",
            modules=[],
            preparation=PreparationConfig(
                replace_frozen_batchnorm=True,
                fuse_conv_batchnorm=True,
            ),
            pruning=[UnstructuredPruner(amount=0.3)],
            quantization=PT2EQuantizationWorkflow(
                targets=[
                    PT2EQuantizationModuleTarget(
                        module_path="",
                        pt2e_backend=X86InductorBackend(is_dynamic=False),
                    )
                ],
            ),
            calibration_steps=3,
            output_directory=compressed_dir,
        )

        with LEROBOT_METADATA_PATCH:
            result = compressor.compress(hydra_config=hydra_config)

        assert result == compressed_dir
        assert (
            Path(compressed_dir) / CompressionFilename.COMPRESSED_MODEL.value
        ).exists()
        assert (Path(compressed_dir) / CompressionFilename.NORMALIZER.value).exists()
        assert (
            Path(compressed_dir) / CompressionFilename.COMPRESSION_METADATA.value
        ).exists()

    def test_compress_without_quantization(
        self,
        tmp_path: Path,
        trained_checkpoint: Callable[..., Path],
        rgb_policy_configuration: Callable[[DictConfig], None],
    ) -> None:
        output_dir = trained_checkpoint(
            config_name=PTQ_TEST_CONFIGS[1],
            extra_overrides=[
                "task.dataloader.tokenization.tokenize_observations=false"
            ],
            configure=rgb_policy_configuration,
        )
        compressed_dir = str(tmp_path / "compressed_no_quant")

        with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
            hydra_config = compose(
                config_name=PTQ_X86_CONFIG_NAME,
                overrides=[f"checkpoint_path={str(output_dir)}"],
            )
        compressor = PostTrainingCompressor(
            checkpoint_path=str(output_dir),
            checkpoint_name="last.ckpt",
            modules=[],
            preparation=PreparationConfig(
                replace_frozen_batchnorm=True,
                fuse_conv_batchnorm=True,
            ),
            pruning=[UnstructuredPruner(amount=0.3)],
            output_directory=compressed_dir,
        )

        with LEROBOT_METADATA_PATCH:
            result = compressor.compress(hydra_config=hydra_config)

        assert result == compressed_dir
        assert (
            Path(compressed_dir) / CompressionFilename.COMPRESSED_MODEL.value
        ).exists()
        with LEROBOT_METADATA_PATCH:
            reference = FloatPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=str(output_dir),
                checkpoint_name="last.ckpt",
                compile_model=False,
            )
            runtime = CompressedPolicyRuntime(
                device=torch.device("cpu"),
                checkpoint_path=result,
                compile_model=False,
            )
        compressor._prepare_and_prune(
            policy=reference.policy, modules=compressor.resolve_modules()
        )
        exportable = create_exportable_policy(policy=reference.policy)
        inputs = build_example_inputs(
            exportable=exportable,
            observation_space=reference.observation_space,
            observation_horizon=reference.observation_horizon,
            tokenizer=reference.tokenizer,
        )
        for batch_size in [1, 2]:
            observations = {
                key: tensor[:batch_size]  # (2, horizon, ...) -> (batch, horizon, ...)
                for key, tensor in zip(exportable.observation_keys, inputs, strict=True)
            }
            torch.testing.assert_close(
                runtime.run_inference(obs_dict=observations),
                reference.run_inference(obs_dict=observations),
            )

    def test_compress_generates_timestamped_directory(
        self, tmp_path, trained_checkpoint
    ):
        output_dir = trained_checkpoint()

        with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
            hydra_config = compose(
                config_name=PTQ_X86_CONFIG_NAME,
                overrides=[f"checkpoint_path={str(output_dir)}"],
            )
        compressor = PostTrainingCompressor(
            checkpoint_path=str(output_dir),
            checkpoint_name="last.ckpt",
            modules=[],
            preparation=PreparationConfig(),
        )

        with LEROBOT_METADATA_PATCH:
            result = compressor.compress(hydra_config=hydra_config)

        assert str(output_dir / "compressed") in result
        assert (Path(result) / CompressionFilename.COMPRESSED_MODEL.value).exists()

    @pytest.mark.requires_executorch
    def test_compress_full_pipeline_with_eager_xnnpack(
        self,
        tmp_path: Path,
        trained_checkpoint: Callable[..., Path],
    ) -> None:
        output_dir = trained_checkpoint(
            extra_overrides=["policy.decoder.embedding_dimension=32"],
        )
        compressed_dir = str(tmp_path / "compressed_eager_xnnpack")

        with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
            hydra_config = compose(
                config_name=PTQ_EAGER_XNNPACK_CONFIG_NAME,
                overrides=[
                    f"checkpoint_path={str(output_dir)}",
                    f"output_directory={compressed_dir}",
                ],
            )
            compressor = hydra.utils.instantiate(hydra_config)

        with LEROBOT_METADATA_PATCH:
            result = compressor.compress(hydra_config=hydra_config)

        assert result == compressed_dir
        assert (
            Path(compressed_dir) / CompressionFilename.EXECUTORCH_MODEL.value
        ).exists()
        assert (Path(compressed_dir) / CompressionFilename.NORMALIZER.value).exists()
        assert (
            Path(compressed_dir) / CompressionFilename.COMPRESSION_METADATA.value
        ).exists()

    @pytest.mark.requires_executorch
    def test_compress_full_pipeline_with_pt2e_xnnpack(
        self,
        tmp_path: Path,
        trained_checkpoint: Callable[..., Path],
    ) -> None:
        output_dir = trained_checkpoint(
            extra_overrides=["policy.decoder.embedding_dimension=32"],
        )
        compressed_dir = str(tmp_path / "compressed_pt2e_xnnpack")

        with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
            hydra_config = compose(
                config_name=PTQ_PT2E_XNNPACK_CONFIG_NAME,
                overrides=[
                    f"checkpoint_path={str(output_dir)}",
                    f"output_directory={compressed_dir}",
                ],
            )
            compressor = hydra.utils.instantiate(hydra_config)

        with LEROBOT_METADATA_PATCH:
            result = compressor.compress(hydra_config=hydra_config)

        assert result == compressed_dir
        assert (
            Path(compressed_dir) / CompressionFilename.EXECUTORCH_MODEL.value
        ).exists()
        assert (Path(compressed_dir) / CompressionFilename.NORMALIZER.value).exists()
        assert (
            Path(compressed_dir) / CompressionFilename.COMPRESSION_METADATA.value
        ).exists()
