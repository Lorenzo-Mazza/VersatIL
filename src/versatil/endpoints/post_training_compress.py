"""Hydra-based post-training compression endpoint.

Note:
    The full workflow is the following: load policy → validate → fuse layers → prune →
    export → quantize → save the compressed checkpoint as a Torch Export ``.pt2``
    or ExecuTorch ``.pte`` artifact, depending on the deployment backend.
"""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from versatil.common.logging import override_log_format
from versatil.configs.paths import get_hydra_configs_dir
from versatil.post_training_compression.compressor import PostTrainingCompressor

EXPERIMENTS_DIR = get_hydra_configs_dir()


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
