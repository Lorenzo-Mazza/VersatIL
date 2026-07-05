"""Hydra-based endpoint for policy explainability insights."""

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from versatil.common.logging import override_log_format
from versatil.configs.paths import get_hydra_configs_dir
from versatil.explainability.runner import ExplainabilityRunner

EXPERIMENTS_DIR = get_hydra_configs_dir()


@hydra.main(
    version_base=None,
    config_path=str(EXPERIMENTS_DIR),
    config_name="end_to_end_explain/default.yaml",
)
def main(config: DictConfig) -> None:
    """Run the explainability endpoint.

    Args:
        config: Hydra configuration for the explainability runner.
    """
    override_log_format()
    logging.info("Explainability endpoint")
    logging.info(OmegaConf.to_yaml(config))
    runner: ExplainabilityRunner = hydra.utils.instantiate(config)
    runner.run()


if __name__ == "__main__":
    main()
