"""Hydra-based post-training compression endpoint.

Thin orchestrator: load policy → validate → prepare → prune →
export → PT2E → quantize_() → save .pt2.
"""

import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from versatil.configs.post_training_compression import PreparationConfig
from versatil.data.dataloader import get_dataloaders
from versatil.inference.policy_loading import PolicyLoader
from versatil.models.exportable_policy import ExportablePolicy
from versatil.post_training_compression.compressor import PostTrainingCompressor
from versatil.post_training_compression.export import (
    build_example_inputs,
    export_policy,
)
from versatil.post_training_compression.preparation import (
    fuse_all_conv_batchnorm_pairs,
    prepare_batchnorms_for_quantization,
)
from versatil.post_training_compression.serialization import save_compressed_model
from versatil.quantization.calibration import CalibrationDataProvider
from versatil.quantization.quantize import (
    apply_pt2e_quantization,
    apply_quantize_api,
)
from versatil.quantization.strategies import PT2EStrategy, QuantizeApiStrategy
from versatil.training.constants import PrecisionType

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "hydra_configs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _apply_preparation(
    submodule: torch.nn.Module,
    preparation: PreparationConfig,
    module_path: str,
) -> None:
    """Apply BN replacement and conv+BN fusion to a submodule.

    Args:
        submodule: The target submodule.
        preparation: PreparationConfig with boolean flags.
        module_path: Dotted path for logging.
    """
    if preparation.replace_frozen_batchnorm:
        count = prepare_batchnorms_for_quantization(submodule)
        logger.info("Prepared %d BatchNorm modules in %s", count, module_path)
    if preparation.fuse_conv_batchnorm:
        count = fuse_all_conv_batchnorm_pairs(submodule)
        logger.info("Fused %d Conv+BN pairs in %s", count, module_path)


@hydra.main(
    version_base=None,
    config_path=str(EXPERIMENTS_DIR),
    config_name="end_to_end_ptq/x86_ptq",
)
def main(config: DictConfig) -> None:
    """Post-training compression endpoint."""
    logger.info("Post-Training Compression")
    logger.info(OmegaConf.to_yaml(config))
    compressor: PostTrainingCompressor = hydra.utils.instantiate(config)
    logger.info("Loading policy from %s", compressor.checkpoint_path)
    policy_loader = PolicyLoader(
        device=torch.device(compressor.device),
        checkpoint_path=compressor.checkpoint_path,
        checkpoint_name=compressor.checkpoint_name,
        precision=PrecisionType.FP32.value,
    )
    policy = policy_loader.policy
    modules = compressor.modules
    compressor.validate(policy=policy)

    for module in modules:
        submodule = (
            policy
            if module.module_path == ""
            else policy.get_submodule(module.module_path)
        )
        label = module.module_path or "(root)"
        logger.info("Processing module %s", label)
        if module.preparation is not None:
            _apply_preparation(
                submodule=submodule,
                preparation=module.preparation,
                module_path=label,
            )
        if module.pruning is not None:
            total, zeroed = module.pruning.prune(module=submodule)
            logger.info(
                "Pruned %s: %d/%d zeroed (%.1f%%)",
                label,
                zeroed,
                total,
                100.0 * zeroed / total if total > 0 else 0.0,
            )

    exportable = ExportablePolicy.from_policy(policy)
    logger.info("Input keys: %s", exportable.observation_keys)
    logger.info("Output keys: %s", exportable.action_keys)
    pt2e_modules = [
        module for module in modules if isinstance(module.quantization, PT2EStrategy)
    ]
    quantize_api_modules = [
        module
        for module in modules
        if isinstance(module.quantization, QuantizeApiStrategy)
    ]
    # quantize_() must run on eager model before export
    if quantize_api_modules:
        apply_quantize_api(
            model=policy,
            quantize_api_modules=quantize_api_modules,
        )
    needs_calibration = any(
        module.quantization.needs_calibration for module in pt2e_modules
    )
    calibration = None
    if needs_calibration:
        train_loader, _, _, _, _ = get_dataloaders(config=policy_loader.config)  # type: ignore[arg-type]
        calibration = CalibrationDataProvider(
            dataloader=train_loader,
            observation_keys=exportable.observation_keys,
            num_calibration_steps=compressor.calibration_steps,
        )
    example_inputs = (
        calibration.get_single_batch()
        if calibration is not None
        else build_example_inputs(policy=policy, exportable=exportable)
    )
    logger.info("Exporting model...")
    exported = export_policy(exportable=exportable, example_inputs=example_inputs)
    if pt2e_modules:
        converted = apply_pt2e_quantization(
            exported=exported,
            pt2e_modules=pt2e_modules,
            calibration=calibration,
        )
    else:
        converted = exported
    output_directory = compressor.output_directory
    if output_directory is None:
        output_directory = str(Path(compressor.checkpoint_path) / "compressed")
    save_compressed_model(
        converted_model=converted,
        example_inputs=example_inputs,
        save_directory=output_directory,
        input_keys=policy.input_keys,
        output_keys=policy.output_keys,
        normalizer=policy.normalizer,
        training_checkpoint_path=compressor.checkpoint_path,
        quantization_config=config,
    )
    logger.info("Compressed model saved to %s", output_directory)


if __name__ == "__main__":
    main()
