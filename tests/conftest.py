"""Root test fixtures shared across the entire test suite."""
import versatil  # noqa: F401 — triggers dotenv loading and cache directory setup

import numpy as np
import pytest
import torch


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed RNG for data generators. Fresh instance per test for isolation."""
    return np.random.default_rng(42)


@pytest.fixture
def device() -> torch.device:
    """Get available device (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size() -> int:
    """Default batch size for tests."""
    return 2


@pytest.fixture
def temporal_length() -> int:
    """Default temporal sequence length."""
    return 2


@pytest.fixture
def image_size() -> tuple[int, int]:
    """Default image size (height, width)."""
    return 224, 224