"""Tests for versatil.training.callbacks.early_stopping module."""

import pytest
from pytorch_lightning.callbacks import EarlyStopping

from versatil.training.callbacks.early_stopping import ResumableEarlyStopping


@pytest.mark.unit
def test_load_state_dict_preserves_fresh_state_unlike_base_early_stopping():
    exhausted_state = {
        "wait_count": 7,
        "stopped_epoch": 42,
        "best_score": 0.1,
        "patience": 7,
    }

    base_callback = EarlyStopping(monitor="val_loss", patience=7)
    base_callback.load_state_dict(exhausted_state)
    assert base_callback.wait_count == 7
    assert base_callback.stopped_epoch == 42

    resumable_callback = ResumableEarlyStopping(monitor="val_loss", patience=7)
    initial_wait_count = resumable_callback.wait_count
    initial_stopped_epoch = resumable_callback.stopped_epoch
    initial_best_score = resumable_callback.best_score

    resumable_callback.load_state_dict(exhausted_state)

    assert resumable_callback.wait_count == initial_wait_count
    assert resumable_callback.stopped_epoch == initial_stopped_epoch
    assert resumable_callback.best_score == initial_best_score
