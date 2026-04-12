"""Hydra-based post-training compression endpoint.

Note:
    The full workflow is the following: load policy → validate → fuse layers → prune →
    export → quantize → save .pt2 compressed checkpoint.
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import versatil.common.argparse_compat  # noqa: F401

from versatil.common.logging import override_log_format
from versatil.post_training_compression.compressor import PostTrainingCompressor

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "hydra_configs"


@hydra.main(
    version_base=None,
    config_path=str(EXPERIMENTS_DIR),
    config_name="end_to_end_ptq/unstructured_prune_x86.yaml",
)
def main(config: DictConfig) -> None:
    """Post-training compression endpoint."""
    override_log_format()
    logging.info("Post-Training Compression")
    logging.info(OmegaConf.to_yaml(config))
    compressor: PostTrainingCompressor = hydra.utils.instantiate(config)
    compressor.compress(hydra_config=config)


if __name__ == "__main__":
    main()
