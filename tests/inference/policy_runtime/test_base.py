"""Tests for versatil.inference.policy_runtime.base module."""

import logging
from unittest.mock import patch

import pytest

from versatil.inference.policy_runtime.base import initialize_inference_seed

BASE_RUNTIME_MODULE = "versatil.inference.policy_runtime.base"


@pytest.mark.unit
@pytest.mark.parametrize("cuda_available", [True, False])
@pytest.mark.parametrize(
    "configured_seed, expected_seed",
    [
        (None, 8_675_309),
        (123, 123),
    ],
)
def test_initialize_inference_seed_resolves_and_applies_seed(
    caplog: pytest.LogCaptureFixture,
    configured_seed: int | None,
    expected_seed: int,
    cuda_available: bool,
) -> None:
    with (
        patch(
            f"{BASE_RUNTIME_MODULE}.secrets.randbits",
            return_value=expected_seed,
        ) as mock_random_seed,
        patch(f"{BASE_RUNTIME_MODULE}.torch.manual_seed") as mock_manual_seed,
        patch(
            f"{BASE_RUNTIME_MODULE}.torch.cuda.is_available",
            return_value=cuda_available,
        ),
        patch(
            f"{BASE_RUNTIME_MODULE}.torch.cuda.manual_seed_all"
        ) as mock_cuda_manual_seed,
        caplog.at_level(logging.INFO),
    ):
        resolved_seed = initialize_inference_seed(seed=configured_seed)

    assert resolved_seed == expected_seed
    mock_manual_seed.assert_called_once_with(seed=expected_seed)
    if configured_seed is None:
        mock_random_seed.assert_called_once_with(63)
    else:
        mock_random_seed.assert_not_called()
    if cuda_available:
        mock_cuda_manual_seed.assert_called_once_with(seed=expected_seed)
    else:
        mock_cuda_manual_seed.assert_not_called()
    assert f"Inference random seed: {expected_seed}" in caplog.messages
