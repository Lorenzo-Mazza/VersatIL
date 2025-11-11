"""Hydra-based training endpoint for all policies."""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from refactoring.workspace import Workspace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


@hydra.main(version_base=None, config_path=str(EXPERIMENTS_DIR), config_name=None)
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration (DictConfig that will be converted to MainConfig)
    """
    if not cfg:
        raise ValueError(
            "No configuration specified! You must provide --config-name.\n"
            "\nExample: python -m src.refactoring.endpoints.train --config-name act_bowel_retraction"
        )

    logger.info("=" * 80)
    logger.info("Training Configuration")
    logger.info("=" * 80)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    # Work with the DictConfig directly
    # Note: The YAML structure doesn't exactly match MainConfig structure
    # (e.g., YAML has task.dataset but Python has task.dataset_schema)
    # So we use the config as-is rather than strict validation
    config: DictConfig = cfg

    # Handle distributed training environment variables
    # These are set by SLURM or other job schedulers
    if "WORLD_SIZE" in os.environ:
        config.experiment.distributed = True
        logger.info(f"Distributed training detected (WORLD_SIZE={os.environ['WORLD_SIZE']})")

    workspace = Workspace(config)
    if config.experiment.resume_from is not None:
        checkpoint_path = Path(config.experiment.resume_from)
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            workspace.load_checkpoint(str(checkpoint_path))
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}. Starting from scratch.")

    workspace.run()

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
