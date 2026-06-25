"""This module compresses a trained policy through a sequence of workflows (fusion, pruning, quantization) and produces a compatible deployment artifact."""

import logging
from datetime import datetime
from pathlib import Path

import torch.nn as nn
from omegaconf import DictConfig

from versatil.configs.post_training_compression import PreparationConfig
from versatil.models.exportable_policy import ExportablePolicy
from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.post_training_compression.constants import CompressionBackendName
from versatil.post_training_compression.deployment_backends.base import (
    DeploymentBackend,
)
from versatil.post_training_compression.deployment_backends.torch_inductor import (
    TorchInductorBackend,
)
from versatil.post_training_compression.preparation import (
    fuse_all_conv_batchnorm_pairs,
    prepare_batchnorms_for_quantization,
)
from versatil.post_training_compression.pruning.base import BasePruner
from versatil.post_training_compression.report import QuantizationReport
from versatil.post_training_compression.serialization import save_compressed_model
from versatil.quantization.constants import QuantizationMode
from versatil.quantization.workflows.base import BaseQuantizationWorkflow
from versatil.quantization.workflows.eager import EagerQuantizationWorkflow
from versatil.quantization.workflows.none import NoQuantizationWorkflow
from versatil.quantization.workflows.pt2e import PT2EQuantizationWorkflow
from versatil.training.constants import CheckpointFilename


class PostTrainingCompressor:
    """Post-training compression pipeline for a trained policy.

    Orchestrates the full compression flow: load → validate →
    prepare → prune → export → quantize → save.
    """

    def __init__(
        self,
        checkpoint_path: str,
        modules: list[CompressionTarget],
        preparation: PreparationConfig,
        device: str = "cpu",
        calibration_steps: int = 32,
        checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value,
        output_directory: str | None = None,
        generate_report: bool = False,
        pruning: list[BasePruner] | None = None,
        quantization: BaseQuantizationWorkflow | None = None,
        backend: DeploymentBackend | None = None,
    ) -> None:
        """Initialize the compression pipeline.

        Args:
            checkpoint_path: Path to the training checkpoint directory.
            modules: Per-module compression schemes (empty = global).
            preparation: Global preparation settings.
            device: Device for policy loading. Export and
                calibration always run on CPU due to
                torch.export device constraints. The compressed
                output is saved on CPU regardless of this setting.
            calibration_steps: Number of calibration batches for
                static quantization.
            checkpoint_name: Checkpoint filename inside the directory.
            output_directory: Where to save compressed output.
                Defaults to checkpoint_path/compressed/<timestamp>.
            generate_report: Whether to generate a quantization
                report after saving. Disabled by default since
                it runs additional forward passes for benchmarking.
            pruning: Global pruning strategies (inherited by modules).
            quantization: Global quantization workflow (inherited by
                modules). ``None`` exports the float model without
                quantization.
            backend: Deployment backend that owns artifact format and
                lowering. Defaults to torch inductor.
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_name
        self.output_directory = output_directory
        self.generate_report = generate_report
        self.device = device
        self.calibration_steps = calibration_steps
        self.modules = modules
        self.preparation = preparation
        self.pruning: list[BasePruner] = pruning or []
        self.quantization = quantization
        self.backend = backend or TorchInductorBackend()

    def compress(self, hydra_config: DictConfig) -> str:
        """Run the full compression pipeline.

        Args:
            hydra_config: Raw Hydra config for serialization
                into the compressed checkpoint directory.

        Returns:
            Path to the saved compressed model directory.
        """
        modules = self.resolve_modules()
        quantization_workflow = self._resolve_quantization_workflow(modules=modules)
        self._validate_backend_compatibility(
            backend_name=self.backend.name,
            mode=quantization_workflow.quantization_mode,
        )
        context = quantization_workflow.load_policy_context(
            checkpoint_path=self.checkpoint_path,
            checkpoint_name=self.checkpoint_name,
        )
        policy = context.policy
        self.validate(policy=policy, modules=modules)
        self._prepare_and_prune(policy=policy, modules=modules)
        exportable = ExportablePolicy.from_policy(policy)
        logging.info("Input keys: %s", exportable.observation_keys)
        logging.info("Output keys: %s", exportable.action_keys)
        quantized = quantization_workflow.quantize(
            context=context,
            exportable=exportable,
            modules=modules,
            calibration_steps=self.calibration_steps,
        )
        deployment_artifact = self.backend.export(
            model=quantized.quantized_model,
            example_inputs=quantized.example_inputs,
        )
        output_directory = self._resolve_output_directory()
        save_compressed_model(
            converted_model=deployment_artifact.converted_model,
            example_inputs=deployment_artifact.example_inputs,
            save_directory=output_directory,
            input_keys=policy.input_keys,
            output_keys=policy.output_keys,
            normalizer=policy.normalizer,
            training_checkpoint_path=self.checkpoint_path,
            quantization_config=hydra_config,
            quantization_workflow=quantized.quantization_workflow,
            model_filename=deployment_artifact.model_filename,
            artifact_format=deployment_artifact.artifact_format.value,
            backend_name=deployment_artifact.backend_name,
            model_bytes=deployment_artifact.model_bytes,
        )
        logging.info("Compressed model saved to %s", output_directory)
        if self.generate_report:
            report = QuantizationReport(
                float_model=quantized.float_model,
                quantized_model=quantized.quantized_model,
                example_inputs=quantized.example_inputs,
                action_keys=policy.output_keys,
                quantization_workflow=quantized.quantization_workflow,
            )
            logging.info("\n%s", report.generate_report())
        return output_directory

    def resolve_modules(self) -> list[CompressionTarget]:
        """Return the compression targets for this run.

        Supports two configuration modes: per-module (explicit
        ``modules`` list targeting specific submodules) and global
        (``modules`` is empty, applying the top-level preparation,
        pruning, and quantization to the entire policy).

        Returns:
            Non-empty list of CompressionTarget instances.
        """
        if self.modules:
            return self.modules
        return [
            CompressionTarget(
                module_path="",
                preparation=self.preparation,
                pruning=self.pruning,
                quantization=self.quantization,
            )
        ]

    def validate(self, policy: nn.Module, modules: list[CompressionTarget]) -> None:
        """Validate module paths and quantization workflow compatibility.

        Args:
            policy: The loaded policy model.
            modules: Resolved compression targets from resolve_modules().

        Raises:
            ValueError: If a module_path doesn't match a submodule,
                or if PT2E and eager quantization workflows are both present.
        """
        has_pt2e = any(
            isinstance(m.quantization, PT2EQuantizationWorkflow) for m in modules
        )
        has_eager = any(
            isinstance(m.quantization, EagerQuantizationWorkflow) for m in modules
        )
        if has_pt2e and has_eager:
            raise ValueError(
                "PT2E and eager quantization workflows cannot be combined. "
                "PT2E operates on the exported FX graph while "
                "eager quantization requires nn.Module submodules. "
                "Use one workflow per compression run."
            )
        for module in modules:
            if module.module_path == "":
                continue
            try:
                policy.get_submodule(module.module_path)
            except AttributeError as error:
                available = list(dict(policy.named_children()).keys())
                raise ValueError(
                    f"Module path '{module.module_path}' not found in "
                    f"policy. Available top-level modules: {available}"
                ) from error

    def _prepare_and_prune(
        self,
        policy: nn.Module,
        modules: list[CompressionTarget],
    ) -> None:
        """Apply BN preparation, fusion, and pruning per module."""
        for module in modules:
            submodule = (
                policy
                if module.module_path == ""
                else policy.get_submodule(module.module_path)
            )
            label = module.module_path or "(root)"
            logging.info("Processing module %s", label)
            if module.preparation is not None:
                if module.preparation.replace_frozen_batchnorm:
                    count = prepare_batchnorms_for_quantization(submodule)
                    logging.info("Prepared %d BatchNorm modules in %s", count, label)
                if module.preparation.fuse_conv_batchnorm:
                    count = fuse_all_conv_batchnorm_pairs(submodule)
                    logging.info("Fused %d Conv+BN pairs in %s", count, label)
            for pruner in module.pruning:
                total, zeroed = pruner.prune(module=submodule)
                logging.info(
                    "Pruned %s with %s: %d/%d zeroed (%.1f%%)",
                    label,
                    type(pruner).__name__,
                    zeroed,
                    total,
                    100.0 * zeroed / total if total > 0 else 0.0,
                )

    @staticmethod
    def _resolve_quantization_workflow(
        modules: list[CompressionTarget],
    ) -> BaseQuantizationWorkflow:
        """Return the configured workflow, defaulting to no quantization."""
        quantization_modes = {
            module.quantization.quantization_mode
            for module in modules
            if module.quantization is not None
        }
        if len(quantization_modes) > 1:
            ordered_modes = sorted(quantization_modes)
            raise ValueError(
                "Compression targets cannot mix quantization modes. "
                f"Configured modes: {ordered_modes}."
            )
        for module in modules:
            if module.quantization is not None:
                return module.quantization
        return NoQuantizationWorkflow()

    @staticmethod
    def _validate_backend_compatibility(backend_name: str, mode: str) -> None:
        """Validate quantization workflow and deployment backend compatibility."""
        compatibility = {
            CompressionBackendName.TORCH_INDUCTOR.value: (
                QuantizationMode.NONE.value,
                QuantizationMode.PT2E.value,
                QuantizationMode.EAGER.value,
            ),
            CompressionBackendName.EXECUTORCH_XNNPACK.value: (
                QuantizationMode.NONE.value,
                QuantizationMode.PT2E.value,
                QuantizationMode.EAGER.value,
            ),
        }
        supported_modes = compatibility.get(backend_name)
        if supported_modes is None:
            raise ValueError(f"Unknown deployment backend '{backend_name}'.")
        if mode in supported_modes:
            return
        raise ValueError(
            f"Deployment backend {backend_name} supports quantization modes "
            f"{list(supported_modes)}, got '{mode}'."
        )

    def _resolve_output_directory(self) -> str:
        """Resolve the output directory for the compressed model.

        Uses explicit output_directory if set, otherwise creates
        a timestamped subdirectory under checkpoint_path/compressed/.

        Returns:
            Absolute path to the output directory.
        """
        if self.output_directory is not None:
            return self.output_directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(Path(self.checkpoint_path) / "compressed" / timestamp)
