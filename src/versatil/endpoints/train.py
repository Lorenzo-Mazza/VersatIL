"""Hydra-based training endpoint for all policies."""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig
from versatil.configs.validator import validate_config
from versatil.workspace import Workspace

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "hydra_configs"


@hydra.main(version_base=None, config_path=str(EXPERIMENTS_DIR), config_name="main")
def main(config: DictConfig) -> None:
    """Main training function.

    Args:
        config: Hydra configuration (DictConfig that will be converted to MainConfig)
    """
    if not config:
        raise ValueError(
            "No configuration specified! You must provide --config-name.\n"
            "\nExample: python -m src.versatil.endpoints.train --config-name act_bowel_retraction"
        )
    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(config))
    logger.info("=" * 80)
    # Handle distributed training environment variables
    # These are set by SLURM or other job schedulers
    if "WORLD_SIZE" in os.environ:
        config.experiment.distributed = True
        logger.info(
            f"Distributed training detected (WORLD_SIZE={os.environ['WORLD_SIZE']})"
        )

    instantiated_config = hydra.utils.instantiate(config)
    validate_config(instantiated_config)
    workspace = Workspace(instantiated_config, original_yaml_config=config)
    if instantiated_config.experiment.resume_from is not None:
        checkpoint_path = Path(instantiated_config.experiment.resume_from)
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            workspace.load_checkpoint(str(checkpoint_path))
        else:
            logger.warning(
                f"Checkpoint not found: {checkpoint_path}. Starting from scratch."
            )
    workspace.run()
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
