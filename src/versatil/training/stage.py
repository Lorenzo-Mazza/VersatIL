"""Runtime training-stage object instantiated from ``TrainingStageConfig``."""

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf


class TrainingStage:
    """Training configuration for multi-stage training.

    Note:
        A stage is active for ``start_epoch <= epoch < end_epoch`` when
        ``end_epoch`` is set. Without ``end_epoch``, the stage remains active
        until the next stage starts, or indefinitely when it is the final stage.
    """

    def __init__(
        self,
        name: str,
        start_epoch: int,
        end_epoch: int | None = None,
        trainable_groups: list[str] | None = None,
        frozen_groups: list[str] | None = None,
        group_lrs: dict[str, float] | None = None,
        group_weight_decays: dict[str, float] | None = None,
        loss_weights: dict[str, Any] | None = None,
        eval_frozen_modules: bool = True,
    ) -> None:
        """Build a training stage and validate self-contained invariants."""
        self.name = name
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.trainable_groups = list(trainable_groups or [])
        self.frozen_groups = list(frozen_groups or [])
        self.group_lrs = _plain_mapping(group_lrs)
        self.group_weight_decays = _plain_mapping(group_weight_decays)
        self.loss_weights = _plain_mapping(loss_weights)
        self.eval_frozen_modules = eval_frozen_modules
        if self.end_epoch is not None and self.end_epoch <= self.start_epoch:
            raise ValueError(
                f"TrainingStage '{self.name}' end_epoch must be greater than "
                f"start_epoch; got {self.end_epoch} <= {self.start_epoch}."
            )
        conflicting_groups = set(self.trainable_groups) & set(self.frozen_groups)
        if conflicting_groups:
            raise ValueError(
                f"Training stage '{self.name}' lists groups in both "
                f"trainable_groups and frozen_groups: {sorted(conflicting_groups)}."
            )
        for group_name, value in self.group_weight_decays.items():
            if not isinstance(value, float):
                raise ValueError(
                    "TrainingStage.group_weight_decays values must be floats; "
                    f"got {type(value).__name__} for group '{group_name}'."
                )

    def is_active_at(
        self, current_epoch: int, next_stage: "TrainingStage | None" = None
    ) -> bool:
        """Return whether this stage should be applied at ``current_epoch``.

        Note:
            Uses ``next_stage.start_epoch`` as the exclusive upper bound when
            this stage's ``end_epoch`` is ``None`` and another stage follows.
        """
        if current_epoch < self.start_epoch:
            return False
        effective_end_epoch = (
            self.end_epoch
            if self.end_epoch is not None
            else (next_stage.start_epoch if next_stage is not None else None)
        )
        return effective_end_epoch is None or current_epoch < effective_end_epoch


def _plain_mapping(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert Hydra mappings to recursive plain dictionaries.

    Args:
        mapping: Optional mapping provided directly or by Hydra.

    Returns:
        Recursive plain dictionary with resolved OmegaConf values.

    Raises:
        TypeError: If OmegaConf conversion does not produce a mapping.
    """
    if mapping is None:
        return {}
    if isinstance(mapping, DictConfig):
        container = OmegaConf.to_container(mapping, resolve=True)
    else:
        container = mapping
    if not isinstance(container, Mapping):
        raise TypeError(
            "TrainingStage mapping fields must be dictionaries; "
            f"got {type(container).__name__}."
        )
    plain: dict[str, Any] = {}
    for key, value in container.items():
        if isinstance(value, (DictConfig, Mapping)):
            plain[key] = _plain_mapping(value)
        else:
            plain[key] = value
    return plain
