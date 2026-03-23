"""Tests for versatil.endpoints.post_training_compress module."""

import gc
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import hydra
import pytest
import torch
import torch._inductor.config as inductor_config
from hydra import compose, initialize_config_dir
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from torchao.quantization.pt2e.quantizer.composable_quantizer import (
    ComposableQuantizer,
)
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
    X86InductorQuantizer,
    get_default_x86_inductor_quantization_config,
)
from tso_robotics_sockets import CompressionType

import versatil.configs  # noqa: F401
from tests.endpoints.conftest import (
    HYDRA_CONFIG_DIR,
    build_tiny_overrides,
    get_free_port,
    resolve_dataset_type,
    start_mock_observation_server,
)
from versatil.data.dataloader import get_dataloaders
from versatil.inference.inference_client import InferenceClient
from versatil.inference.policy_loading import CompressedPolicyLoader, PolicyLoader
from versatil.inference.socket_transport import (
    SocketActionTransport,
    SocketObservationTransport,
)
from versatil.models.exportable_policy import ExportablePolicy
from versatil.post_training_compression.export import export_policy
from versatil.post_training_compression.preparation import (
    fuse_all_conv_batchnorm_pairs,
    prepare_batchnorms_for_quantization,
)
from versatil.post_training_compression.pruning import UnstructuredPruner
from versatil.post_training_compression.serialization import save_compressed_model
from versatil.quantization.calibration import CalibrationDataProvider
from versatil.quantization.torch_patches import patch_get_source_partitions
from versatil.workspace import Workspace

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
NUM_EPISODES = 3
TIMESTEPS_PER_EPISODE = 15

COMMON_OVERRIDES = [
    "task.dataloader.batch_size=2",
    "task.dataloader.image_height=32",
    "task.dataloader.image_width=32",
    "task.dataloader.num_workers=1",
    "task.dataloader.val_ratio=0.0",
    "training.num_epochs=1",
    "experiment.use_wandb=false",
    "experiment.name=ptq_test",
    "experiment.device=cpu",
]

PTQ_TEST_CONFIGS = [
    "end_to_end_training_runs/libero_lerobot/action_transformer_language",
    "end_to_end_training_runs/libero_lerobot/action_transformer",
    "end_to_end_training_runs/libero_lerobot/act",
    "end_to_end_training_runs/libero_lerobot/flow_dit_cross_attention",
]

LEROBOT_METADATA_PATCH = patch(
    "versatil.data.raw.schemas.lerobot.LeRobotDatasetMetadataV30.__init__",
    lambda self, dataset_path: setattr(self, "dataset_path", dataset_path),
)


@pytest.fixture(autouse=True, scope="session")
def _configure_inductor():
    """Set inductor config and patch source partitions once per session."""
    original_freezing = os.environ.get("TORCHINDUCTOR_FREEZING")
    original_cpp_wrapper = inductor_config.cpp_wrapper
    os.environ["TORCHINDUCTOR_FREEZING"] = "1"
    inductor_config.cpp_wrapper = True
    patch_get_source_partitions()
    yield
    if original_freezing is None:
        os.environ.pop("TORCHINDUCTOR_FREEZING", None)
    else:
        os.environ["TORCHINDUCTOR_FREEZING"] = original_freezing
    inductor_config.cpp_wrapper = original_cpp_wrapper


@pytest.fixture
def trained_checkpoint(tmp_path, synthetic_zarr_factory):
    """Train 1 epoch and yield checkpoint directory. Cleans up after."""

    def factory(
        config_name: str = PTQ_TEST_CONFIGS[0],
        extra_overrides: list[str] | None = None,
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
def compression_pipeline(trained_checkpoint):
    """Load policy, create calibration and exportable. Cleans up after test."""
    created = []

    def factory(
        config_name: str = PTQ_TEST_CONFIGS[0],
    ) -> tuple[PolicyLoader, CalibrationDataProvider, ExportablePolicy]:
        output_dir = trained_checkpoint(config_name=config_name)
        with LEROBOT_METADATA_PATCH:
            policy_loader = PolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=str(output_dir),
                checkpoint_name="last.ckpt",
                precision="32",
            )

        exportable = ExportablePolicy.from_policy(policy_loader.policy)

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
    expect_divergence: bool = True,
) -> None:
    """Save compressed model, verify files exist, verify inference, check divergence."""
    compressed_dir = str(tmp_path / "compressed")

    with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base=None):
        ptq_config = compose(
            config_name="end_to_end_ptq/x86_ptq",
            overrides=[f"checkpoint_path={str(output_dir)}"],
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
            for compressed, original in zip(compressed_outputs, float_outputs)
        )
        assert outputs_changed, (
            "Compressed outputs identical to float — compression may have failed"
        )

    # Verify compressed inference via mock server
    with LEROBOT_METADATA_PATCH:
        compressed_loader = CompressedPolicyLoader(
            device=torch.device("cpu"),
            checkpoint_path=compressed_dir,
        )

    assert compressed_loader.input_keys == exportable.observation_keys
    assert compressed_loader.output_keys == exportable.action_keys

    port = get_free_port()
    server = start_mock_observation_server(
        observation_space=compressed_loader.observation_space,
        port=port,
    )
    try:
        client = InferenceClient(
            policy_loader=compressed_loader,
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
@pytest.mark.parametrize(
    "config_name",
    PTQ_TEST_CONFIGS,
    ids=[c.split("/")[-1] for c in PTQ_TEST_CONFIGS],
)
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
        example_inputs = calibration.get_single_batch()
        float_outputs = _get_float_outputs(
            exportable=exportable,
            example_inputs=example_inputs,
        )

        exported = export_policy(exportable=exportable, example_inputs=example_inputs)

        quantizer = X86InductorQuantizer()
        quantizer.set_global(get_default_x86_inductor_quantization_config())
        prepared = prepare_pt2e(exported, quantizer)

        with torch.no_grad():
            for batch in calibration:
                prepared(*batch)

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
        )


@pytest.mark.slow
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

        example_inputs = calibration.get_single_batch()
        float_outputs = _get_float_outputs(
            exportable=exportable,
            example_inputs=example_inputs,
        )

        exported = export_policy(exportable=exportable, example_inputs=example_inputs)

        quantizers = _build_backbone_quantizers(policy)
        composed = ComposableQuantizer(quantizers)
        prepared = prepare_pt2e(exported, composed)

        with torch.no_grad():
            for batch in calibration:
                prepared(*batch)

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
        )


@pytest.mark.slow
class TestGlobalQuantizeApiDynamic:
    @pytest.mark.parametrize(
        "embedding_dimension, expect_divergence",
        [
            (16, False),
            (32, True),
        ],
        ids=["skip_small_layers", "quantize_large_layers"],
    )
    def test_quantize_api_before_export(
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
            policy_loader = PolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=str(output_dir),
                checkpoint_name="last.ckpt",
                precision="32",
            )
        policy = policy_loader.policy
        exportable = ExportablePolicy.from_policy(policy)

        with LEROBOT_METADATA_PATCH:
            train_loader, _, _, _, _ = get_dataloaders(config=policy_loader.config)
        calibration = CalibrationDataProvider(
            dataloader=train_loader,
            observation_keys=exportable.observation_keys,
            num_calibration_steps=3,
        )

        example_inputs = calibration.get_single_batch()
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
            expect_divergence=expect_divergence,
        )
