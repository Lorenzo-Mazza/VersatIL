"""Training and optimizer config registrations."""

from hydra.core.config_store import ConfigStore

from versatil.configs import (
    AdamConfig,
    AdamWConfig,
    OptimizerConfig,
    ParameterGroupConfig,
    SGDConfig,
    TrainingConfig,
)


def register(cs: ConfigStore) -> None:
    """Store this domain's config nodes.

    Args:
        cs: The global Hydra config store.
    """
    cs.store(group="training", name="base", node=TrainingConfig)
    cs.store(group="training/optimizer", name="base", node=OptimizerConfig)
    cs.store(group="training/optimizer", name="adamw_schema", node=AdamWConfig)
    cs.store(group="training/optimizer", name="adam_schema", node=AdamConfig)
    cs.store(group="training/optimizer", name="sgd_schema", node=SGDConfig)
    cs.store(
        group="training/optimizer/parameter_group",
        name="base",
        node=ParameterGroupConfig,
    )
