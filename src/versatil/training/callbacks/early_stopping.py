"""Resumable early stopping callback."""

from pytorch_lightning.callbacks import EarlyStopping


class ResumableEarlyStopping(EarlyStopping):
    """EarlyStopping that ignores checkpoint state, always using config values.

    Note: this allows to resume training beyond an initial early stopping state, which is
     otherwise not possible to overwrite from Lightning.
    """

    def load_state_dict(self, state_dict):
        pass
