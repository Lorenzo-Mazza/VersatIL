"""Raw data package test fixtures: resizer helpers."""

import albumentations as A
import pytest


@pytest.fixture
def noop_resizer() -> A.NoOp:
    """No-op resizer for extract_episode calls that do not need resizing."""
    return A.NoOp()
