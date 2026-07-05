"""Resumable early stopping callback."""

from typing import Any

from pytorch_lightning.callbacks import EarlyStopping


class ResumableEarlyStopping(EarlyStopping):
    """EarlyStopping that ignores checkpoint state, always using config values.

    Lightning restores early-stopping counters from the checkpoint, so a run
    that already stopped once could never be resumed past that point. Dropping
    the restored state lets resumed runs restart early stopping from the
    configured values.
    """

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Ignore restored state so resumed runs never stop on stale counters."""
