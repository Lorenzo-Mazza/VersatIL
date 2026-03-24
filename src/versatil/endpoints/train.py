"""Hydra-based training endpoint for all policies."""

import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from versatil.common.logging import override_log_format
from versatil.validation import validate_experiment
from versatil.workspace import Workspace

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "hydra_configs"


@hydra.main(version_base=None, config_path=str(EXPERIMENTS_DIR), config_name="main")
def main(config: DictConfig) -> None:
    """Main training function.

    Args:
        config: Hydra configuration (DictConfig that will be converted to MainConfig)
    """
    override_log_format()
    if not config:
        raise ValueError(
            "No configuration specified! You must provide --config-name.\n"
            "\nExample: python -m src.versatil.endpoints.train --config-name act_bowel_retraction"
        )
    logging.info("=" * 80)
    logging.info("Training Configuration")
    logging.info("=" * 80)
    logging.info(OmegaConf.to_yaml(config))
    logging.info("=" * 80)
    # Handle distributed training environment variables
    # These are set by SLURM or other job schedulers
    if "WORLD_SIZE" in os.environ:
        config.experiment.distributed = True
        logging.info(
            f"Distributed training detected (WORLD_SIZE={os.environ['WORLD_SIZE']})"
        )

    instantiated_config = hydra.utils.instantiate(config)
    validate_experiment(instantiated_config)
    workspace = Workspace(instantiated_config, original_yaml_config=config)
    workspace.run()
    logging.info("Training completed successfully!")


if __name__ == "__main__":
    main()
