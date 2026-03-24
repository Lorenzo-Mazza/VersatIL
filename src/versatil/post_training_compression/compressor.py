"""Post-training compressor for a trained policy."""

import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import DictConfig

from versatil.configs.post_training_compression import PreparationConfig
from versatil.data.dataloader import get_dataloaders
from versatil.inference.policy_loading.float_loader import PolicyLoader
from versatil.models.exportable_policy import ExportablePolicy
from versatil.post_training_compression.compression_target import CompressionTarget
from versatil.post_training_compression.constants import QuantizationStrategy
from versatil.post_training_compression.export import (
    build_example_inputs,
    export_policy,
)
from versatil.post_training_compression.preparation import (
    fuse_all_conv_batchnorm_pairs,
    prepare_batchnorms_for_quantization,
)
from versatil.post_training_compression.pruning.base import BasePruner
from versatil.post_training_compression.report import QuantizationReport
from versatil.post_training_compression.serialization import save_compressed_model
from versatil.quantization.calibration import CalibrationDataProvider
from versatil.quantization.quantize import (
    apply_pt2e_quantization,
    apply_quantize_api,
)
from versatil.quantization.strategies import (
    PT2EStrategy,
    QuantizeApiStrategy,
)
from versatil.training.constants import CheckpointFilename, PrecisionType


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
        quantization: PT2EStrategy | QuantizeApiStrategy | None = None,
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
            quantization: Global quantization strategy (inherited by
                modules).
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

    def compress(self, hydra_config: DictConfig) -> str:
        """Run the full compression pipeline.

        Args:
            hydra_config: Raw Hydra config for serialization
                into the compressed checkpoint directory.

        Returns:
            Path to the saved compressed model directory.
        """
        policy_loader = self._load_policy()
        policy = policy_loader.policy
        modules = self.resolve_modules()
        self.validate(policy=policy)
        self._prepare_and_prune(policy=policy, modules=modules)
        exportable = ExportablePolicy.from_policy(policy)
        logging.info("Input keys: %s", exportable.observation_keys)
        logging.info("Output keys: %s", exportable.action_keys)
        exported, converted, example_inputs, strategy = self._export_and_quantize(
            policy=policy,
            policy_loader=policy_loader,
            exportable=exportable,
            modules=modules,
        )
        output_directory = self._resolve_output_directory()
        save_compressed_model(
            converted_model=converted,
            example_inputs=example_inputs,
            save_directory=output_directory,
            input_keys=policy.input_keys,
            output_keys=policy.output_keys,
            normalizer=policy.normalizer,
            training_checkpoint_path=self.checkpoint_path,
            quantization_config=hydra_config,
            quantization_strategy=strategy,
        )
        logging.info("Compressed model saved to %s", output_directory)
        if self.generate_report:
            report = QuantizationReport(
                float_model=exported,
                quantized_model=converted,
                example_inputs=example_inputs,
                action_keys=policy.output_keys,
                quantization_strategy=strategy,
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

    def validate(self, policy: nn.Module) -> None:
        """Validate module paths and strategy compatibility.

        Args:
            policy: The loaded policy model.

        Raises:
            ValueError: If a module_path doesn't match a submodule,
                or if PT2E and quantize_() strategies are both present.
        """
        has_pt2e = any(isinstance(m.quantization, PT2EStrategy) for m in self.modules)
        has_quantize_api = any(
            isinstance(m.quantization, QuantizeApiStrategy) for m in self.modules
        )
        if has_pt2e and has_quantize_api:
            raise ValueError(
                "PT2E and quantize_() strategies cannot be combined. "
                "PT2E operates on the exported FX graph while "
                "quantize_() requires eager nn.Module submodules. "
                "Use one strategy per compression run."
            )
        for module in self.modules:
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

    def _load_policy(self) -> PolicyLoader:
        """Load policy from checkpoint."""
        logging.info("Loading policy from %s", self.checkpoint_path)
        return PolicyLoader(
            device=torch.device("cpu"),
            checkpoint_path=self.checkpoint_path,
            checkpoint_name=self.checkpoint_name,
            precision=PrecisionType.FP32.value,
        )

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

    def _export_and_quantize(
        self,
        policy: nn.Module,
        policy_loader: PolicyLoader,
        exportable: ExportablePolicy,
        modules: list[CompressionTarget],
    ) -> tuple[nn.Module, nn.Module, tuple[torch.Tensor, ...], str]:
        """Export, calibrate, and quantize the policy.

        Args:
            policy: The prepared policy.
            policy_loader: Loader with config access for dataloader.
            exportable: Positional-tensor wrapper for export.
            modules: Resolved compression modules.

        Returns:
            Tuple of (float_exported, quantized_converted,
            example_inputs, strategy_name).
        """
        pt2e_modules = [m for m in modules if isinstance(m.quantization, PT2EStrategy)]
        quantize_api_modules = [
            m for m in modules if isinstance(m.quantization, QuantizeApiStrategy)
        ]
        if quantize_api_modules:
            apply_quantize_api(
                model=policy,
                quantize_api_modules=quantize_api_modules,
            )

        needs_calibration = any(m.quantization.needs_calibration for m in pt2e_modules)
        calibration = None
        if needs_calibration:
            train_loader, _, _, _, _ = get_dataloaders(config=policy_loader.config)
            calibration = CalibrationDataProvider(
                dataloader=train_loader,
                observation_keys=exportable.observation_keys,
                num_calibration_steps=self.calibration_steps,
            )
        example_inputs = (
            calibration.get_single_batch()
            if calibration is not None
            else build_example_inputs(
                exportable=exportable,
                observation_space=policy_loader.observation_space,
                dataloader_config=policy_loader.config.task.dataloader,
                tokenizer=policy_loader.tokenizer,
            )
        )
        logging.info("Exporting model...")
        exported = export_policy(exportable=exportable, example_inputs=example_inputs)
        if pt2e_modules:
            converted = apply_pt2e_quantization(
                exported=exported,
                pt2e_modules=pt2e_modules,
                calibration=calibration,
            )
        else:
            converted = exported
        strategy = (
            QuantizationStrategy.QUANTIZE_API.value
            if quantize_api_modules
            else QuantizationStrategy.PT2E.value
        )
        return exported, converted, example_inputs, strategy

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
