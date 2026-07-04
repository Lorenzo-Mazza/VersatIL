"""Root, experiment, and endpoint config registrations."""

from hydra.core.config_store import ConfigStore

from versatil.configs import (
    DeploymentConfig,
    ExperimentConfig,
    ExplainabilityConfig,
    MainConfig,
    PostTrainingCompressorConfig,
)


def register(cs: ConfigStore) -> None:
    """Store this domain's config nodes.

    Args:
        cs: The global Hydra config store.
    """
    cs.store(name="config", node=MainConfig)
    cs.store(group="experiment", name="base", node=ExperimentConfig)
    cs.store(
        name="post_training_compression",
        node=PostTrainingCompressorConfig,
    )
    cs.store(
        name="explainability",
        node=ExplainabilityConfig,
    )
    cs.store(
        name="deployment",
        node=DeploymentConfig,
    )
