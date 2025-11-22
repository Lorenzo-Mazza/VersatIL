"""Shared fixtures for tokenize tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


@pytest.fixture
def normalized_proprio_data():
    """Generate normalized proprioceptive data for testing binning tokenizers.

    Returns:
        Dict with proprio keys mapped to normalized arrays (N=100, D=7)
    """
    np.random.seed(42)
    return {
        "proprio_robot_frame": np.random.randn(100, 7).astype(np.float32) * 0.5,
        "proprio_camera_frame": np.random.randn(100, 7).astype(np.float32) * 0.5,
    }


@pytest.fixture
def language_instructions():
    """Generate sample language instructions."""
    return [
        "Pick up the red block",
        "Place the block on the table",
        "Grasp the needle",
        "Insert the needle into the tissue",
        "Retract the needle",
    ]


@pytest.fixture
def normalized_action_chunks():
    """Generate normalized action chunks for FAST tokenizer testing.

    Returns:
        Array of shape (N_chunks=10, T=5, D=7)
    """
    np.random.seed(42)
    return np.random.randn(10, 5, 7).astype(np.float32) * 0.5


@pytest.fixture
def observation_dict_with_language(language_instructions, normalized_proprio_data):
    """Create observation dict with language and proprio.

    Returns batch of observations with temporal dimension.
    """
    batch_size = 5
    obs_horizon = 2

    # Language: list of strings (one per sample in batch)
    lang = language_instructions[:batch_size]

    # Proprio: (B, T, D)
    proprio_robot = torch.from_numpy(
        normalized_proprio_data["proprio_robot_frame"][:batch_size * obs_horizon]
        .reshape(batch_size, obs_horizon, -1)
    )

    return {
        "language_instruction": lang,
        "proprio_robot_frame": proprio_robot,
    }


@pytest.fixture
def simple_language_tokenizer_model():
    """Return a small, fast language model for testing."""
    return "google/bert_uncased_L-2_H-128_A-2"  # Tiny BERT for fast tests