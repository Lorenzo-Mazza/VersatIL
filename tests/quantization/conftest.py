"""Shared fixtures for quantization tests."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.quantization.calibration import CalibrationDataProvider


@pytest.fixture
def mock_calibration_provider_factory(
    rng: np.random.Generator,
) -> Callable[..., MagicMock]:
    """Factory for mock CalibrationDataProvider with deterministic batches."""

    def factory(
        batch_size: int = 2,
        input_dimension: int = 4,
        num_batches: int = 3,
    ) -> MagicMock:
        provider = MagicMock(spec=CalibrationDataProvider)
        provider.device = torch.device("cpu")
        batches = []
        for _ in range(num_batches):
            data = rng.standard_normal((batch_size, input_dimension)).astype(np.float32)
            batches.append((torch.from_numpy(data),))
        provider.__iter__ = MagicMock(return_value=iter(batches))
        single_data = rng.standard_normal((batch_size, input_dimension)).astype(
            np.float32
        )
        provider.get_single_batch.return_value = (torch.from_numpy(single_data),)
        return provider

    return factory
