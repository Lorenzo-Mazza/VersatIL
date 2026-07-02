"""Post-training compression orchestration."""

import logging
from datetime import datetime
from pathlib import Path

import torch.nn as nn
from omegaconf import DictConfig

from versatil.configs.post_training_compression import PreparationConfig
from versatil.models.exportable_policy import ExportablePolicy
from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.post_training_compression.constants import DeploymentBackendName
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
from versatil.quantization.constants import PT2EBackendName, QuantizationMode
from versatil.quantization.workflows.base import BaseQuantizationWorkflow
from versatil.quantization.workflows.none import NoQuantizationWorkflow
from versatil.training.constants import CheckpointFilename


class PostTrainingCompressor:
    """Post-training compression pipeline for a trained policy.

    Orchestrates loading, validation, preparation, pruning, quantization,
    deployment export, and serialization.
    """

    def __init__(
        self,
        checkpoint_path: str,
        modules: list[CompressionTarget],
        preparation: PreparationConfig,
        calibration_steps: int = 32,
        checkpoint_name: str = CheckpointFilename.DEFAULT_CHECKPOINT.value,
        output_directory: str | None = None,
        generate_report: bool = False,
        pruning: list[BasePruner] | None = None,
        quantization: BaseQuantizationWorkflow | None = None,
        deployment_backend: DeploymentBackend | None = None,
    ) -> None:
        """Initialize the compression pipeline.

        Args:
            checkpoint_path: Path to the training checkpoint directory.
            modules: Per-module compression schemes (empty = global).
            preparation: Global preparation settings.
            calibration_steps: Number of calibration batches for
                static quantization.
            checkpoint_name: Checkpoint filename inside the directory.
            output_directory: Where to save compressed output.
                Defaults to checkpoint_path/compressed/<timestamp>.
            generate_report: Whether to generate a quantization
                report after saving. Disabled by default since
                it runs additional forward passes for benchmarking.
            pruning: Global pruning strategies (inherited by modules).
            quantization: Quantization workflow. ``None`` exports the
                float model without quantization.
            deployment_backend: Deployment backend that owns artifact format and
                lowering. Defaults to torch inductor.
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_name
        self.output_directory = output_directory
        self.generate_report = generate_report
        self.calibration_steps = calibration_steps
        self.modules = modules
        self.preparation = preparation
        self.pruning: list[BasePruner] = pruning or []
        self.quantization = quantization
        self.deployment_backend = deployment_backend or TorchInductorBackend()

    def compress(self, hydra_config: DictConfig) -> str:
        """Run the full compression pipeline.

        Args:
            hydra_config: Raw Hydra config for serialization
                into the compressed checkpoint directory.

        Returns:
            Path to the saved compressed model directory.
        """
        modules = self.resolve_modules()
        quantization_workflow = self._resolve_quantization_workflow()
        self._validate_deployment_backend_compatibility(
            deployment_backend_name=self.deployment_backend.name,
            mode=quantization_workflow.quantization_mode,
            pt2e_backend_names=quantization_workflow.pt2e_backend_names,
        )
        context = quantization_workflow.load_policy_context(
            checkpoint_path=self.checkpoint_path,
            checkpoint_name=self.checkpoint_name,
        )
        policy = context.policy
        self.validate(policy=policy, modules=modules)
        quantization_workflow.validate_targets(model=policy)
        self._prepare_and_prune(policy=policy, modules=modules)
        exportable = ExportablePolicy.from_policy(policy)
        logging.info("Input keys: %s", exportable.observation_keys)
        logging.info("Output keys: %s", exportable.action_keys)
        quantized = quantization_workflow.quantize(
            context=context,
            exportable=exportable,
            calibration_steps=self.calibration_steps,
        )
        deployment_artifact = self.deployment_backend.export(
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
            denoising_thresholds=policy.get_denoising_thresholds(),
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
        (``modules`` is empty, applying the top-level preparation and
        pruning to the entire policy).

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
            )
        ]

    def validate(self, policy: nn.Module, modules: list[CompressionTarget]) -> None:
        """Validate compression target module paths and overlaps.

        Args:
            policy: The loaded policy model.
            modules: Resolved compression targets from resolve_modules().

        Raises:
            ValueError: If a module_path doesn't match a submodule, or if two
                targets overlap and would compound pruning on the same module.
        """
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
        for index, module in enumerate(modules):
            for other in modules[index + 1 :]:
                if module.overlaps(other=other):
                    raise ValueError(
                        "Compression targets overlap: "
                        f"'{module.module_path}' and '{other.module_path}'."
                    )

    def _prepare_and_prune(
        self,
        policy: nn.Module,
        modules: list[CompressionTarget],
    ) -> None:
        """Apply BN preparation, fusion, and pruning per module.

        Args:
            policy: Loaded policy model to mutate in-place.
            modules: Resolved preparation and pruning targets.
        """
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

    def _resolve_quantization_workflow(self) -> BaseQuantizationWorkflow:
        """Return the configured workflow, defaulting to no quantization.

        Returns:
            Configured quantization workflow, or ``NoQuantizationWorkflow`` when
            compression was configured without quantization.
        """
        return self.quantization or NoQuantizationWorkflow()

    @staticmethod
    def _validate_deployment_backend_compatibility(
        deployment_backend_name: str,
        mode: str,
        pt2e_backend_names: tuple[str, ...] = (),
    ) -> None:
        """Validate quantization workflow and deployment backend compatibility.

        Args:
            deployment_backend_name: Deployment backend identifier.
            mode: Quantization mode selected by the workflow.
            pt2e_backend_names: PT2E backend identifiers used by the workflow.

        Raises:
            ValueError: If the backend is unknown or does not support ``mode``.
        """
        compatibility = {
            DeploymentBackendName.TORCH_INDUCTOR.value: (
                QuantizationMode.NONE.value,
                QuantizationMode.PT2E.value,
                QuantizationMode.EAGER.value,
            ),
            DeploymentBackendName.EXECUTORCH_XNNPACK.value: (
                QuantizationMode.NONE.value,
                QuantizationMode.PT2E.value,
                QuantizationMode.EAGER.value,
            ),
        }
        supported_modes = compatibility.get(deployment_backend_name)
        if supported_modes is None:
            raise ValueError(f"Unknown deployment backend '{deployment_backend_name}'.")
        if mode in supported_modes:
            if mode != QuantizationMode.PT2E.value:
                return
            PostTrainingCompressor._validate_pt2e_backend_compatibility(
                deployment_backend_name=deployment_backend_name,
                pt2e_backend_names=pt2e_backend_names,
            )
            return
        raise ValueError(
            f"Deployment backend {deployment_backend_name} supports quantization modes "
            f"{list(supported_modes)}, got '{mode}'."
        )

    @staticmethod
    def _validate_pt2e_backend_compatibility(
        deployment_backend_name: str,
        pt2e_backend_names: tuple[str, ...],
    ) -> None:
        """Validate concrete PT2E backend support for a deployment backend.

        Args:
            deployment_backend_name: Deployment backend identifier.
            pt2e_backend_names: PT2E backend identifiers used by the workflow.

        Raises:
            ValueError: If any PT2E backend is incompatible with deployment.
        """
        compatibility = {
            DeploymentBackendName.TORCH_INDUCTOR.value: (
                PT2EBackendName.X86_INDUCTOR.value,
            ),
            DeploymentBackendName.EXECUTORCH_XNNPACK.value: (
                PT2EBackendName.XNNPACK.value,
            ),
        }
        supported_backends = compatibility[deployment_backend_name]
        unsupported_backends = [
            name for name in pt2e_backend_names if name not in supported_backends
        ]
        if not unsupported_backends:
            return
        raise ValueError(
            f"Deployment backend {deployment_backend_name} supports PT2E backends "
            f"{list(supported_backends)}, got {list(pt2e_backend_names)}."
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
