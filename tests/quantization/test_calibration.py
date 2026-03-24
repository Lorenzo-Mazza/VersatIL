"""Tests for versatil.quantization.calibration module."""

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.quantization.calibration import CalibrationDataProvider


@pytest.fixture
def observation_batch_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, dict[str, torch.Tensor]]]:
    def factory(
        batch_size: int = 2,
        feature_dimension: int = 4,
        observation_keys: list[str] | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        if observation_keys is None:
            observation_keys = ["left", "right"]
        observation = {}
        for key in observation_keys:
            data = rng.standard_normal((batch_size, feature_dimension)).astype(
                np.float32
            )
            observation[key] = torch.from_numpy(data)
        return {SampleKey.OBSERVATION.value: observation}

    return factory


@pytest.fixture
def mock_dataloader_factory(
    observation_batch_factory: Callable,
) -> Callable[..., MagicMock]:
    def factory(
        num_batches: int = 5,
        batch_size: int = 2,
        feature_dimension: int = 4,
        observation_keys: list[str] | None = None,
    ) -> MagicMock:
        batches = [
            observation_batch_factory(
                batch_size=batch_size,
                feature_dimension=feature_dimension,
                observation_keys=observation_keys,
            )
            for _ in range(num_batches)
        ]
        dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        dataloader.__iter__ = MagicMock(return_value=iter(batches))
        return dataloader

    return factory


@pytest.fixture
def calibration_provider_factory(
    mock_dataloader_factory: Callable,
) -> Callable[..., CalibrationDataProvider]:
    def factory(
        num_batches: int = 5,
        batch_size: int = 2,
        feature_dimension: int = 4,
        observation_keys: list[str] | None = None,
        num_calibration_steps: int = 128,
    ) -> CalibrationDataProvider:
        if observation_keys is None:
            observation_keys = ["left", "right"]
        dataloader = mock_dataloader_factory(
            num_batches=num_batches,
            batch_size=batch_size,
            feature_dimension=feature_dimension,
            observation_keys=observation_keys,
        )
        return CalibrationDataProvider(
            dataloader=dataloader,
            observation_keys=observation_keys,
            num_calibration_steps=num_calibration_steps,
            device=torch.device("cpu"),
        )

    return factory


@pytest.mark.unit
class TestCalibrationDataProviderIteration:
    def test_yields_correct_number_of_steps(
        self,
        calibration_provider_factory,
    ):
        provider = calibration_provider_factory(
            num_batches=5,
            num_calibration_steps=3,
        )

        batches = list(provider)

        assert len(batches) == 3

    def test_stops_at_dataloader_exhaustion_when_fewer_than_max_steps(
        self,
        calibration_provider_factory,
    ):
        provider = calibration_provider_factory(
            num_batches=2,
            num_calibration_steps=10,
        )

        batches = list(provider)

        assert len(batches) == 2

    def test_yields_tuples_matching_observation_keys_order(
        self,
        rng: np.random.Generator,
    ):
        keys = ["beta", "alpha"]
        beta_data = rng.standard_normal((2, 5)).astype(np.float32)
        alpha_data = rng.standard_normal((2, 3)).astype(np.float32)
        observation = {
            "beta": torch.from_numpy(beta_data),
            "alpha": torch.from_numpy(alpha_data),
        }
        batch = {SampleKey.OBSERVATION.value: observation}
        dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        dataloader.__iter__ = MagicMock(return_value=iter([batch]))

        provider = CalibrationDataProvider(
            dataloader=dataloader,
            observation_keys=keys,
            num_calibration_steps=1,
            device=torch.device("cpu"),
        )

        result = next(iter(provider))

        assert len(result) == 2
        # First element corresponds to "beta" (first in keys), shape (2, 5)
        assert result[0].shape == (2, 5)
        assert torch.equal(result[0], observation["beta"])
        # Second element corresponds to "alpha" (second in keys), shape (2, 3)
        assert result[1].shape == (2, 3)
        assert torch.equal(result[1], observation["alpha"])

    @pytest.mark.parametrize(
        "batch_size, feature_dimension",
        [(2, 4), (3, 8)],
    )
    def test_yielded_tensors_have_expected_shapes(
        self,
        calibration_provider_factory,
        batch_size,
        feature_dimension,
    ):
        keys = ["left"]
        provider = calibration_provider_factory(
            observation_keys=keys,
            num_batches=1,
            num_calibration_steps=1,
            batch_size=batch_size,
            feature_dimension=feature_dimension,
        )

        batch = next(iter(provider))

        assert batch[0].shape == (batch_size, feature_dimension)


@pytest.mark.unit
class TestCalibrationDataProviderSingleBatch:
    def test_get_single_batch_returns_first_batch(
        self,
        rng: np.random.Generator,
    ):
        keys = ["left", "right"]
        batch_size = 2
        feature_dimension = 4

        left_data = rng.standard_normal((batch_size, feature_dimension)).astype(
            np.float32
        )
        right_data = rng.standard_normal((batch_size, feature_dimension)).astype(
            np.float32
        )
        first_observation = {
            "left": torch.from_numpy(left_data),
            "right": torch.from_numpy(right_data),
        }
        first_batch = {SampleKey.OBSERVATION.value: first_observation}

        second_observation = {
            "left": torch.zeros(batch_size, feature_dimension),
            "right": torch.zeros(batch_size, feature_dimension),
        }
        second_batch = {SampleKey.OBSERVATION.value: second_observation}

        dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        dataloader.__iter__ = MagicMock(return_value=iter([first_batch, second_batch]))

        provider = CalibrationDataProvider(
            dataloader=dataloader,
            observation_keys=keys,
            num_calibration_steps=2,
            device=torch.device("cpu"),
        )

        single = provider.get_single_batch()

        assert len(single) == 2
        # Verify values match the first batch, not the second
        assert torch.equal(single[0], first_observation["left"])
        assert torch.equal(single[1], first_observation["right"])

    def test_get_single_batch_raises_on_empty_dataloader(self):
        dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        dataloader.__iter__ = MagicMock(return_value=iter([]))

        provider = CalibrationDataProvider(
            dataloader=dataloader,
            observation_keys=["left"],
            num_calibration_steps=0,
            device=torch.device("cpu"),
        )

        with pytest.raises(StopIteration):
            provider.get_single_batch()


@pytest.mark.unit
class TestCalibrationDataProviderDevice:
    def test_defaults_to_cuda_when_available(self):
        dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        provider = CalibrationDataProvider(
            dataloader=dataloader,
            observation_keys=["left"],
            num_calibration_steps=1,
        )

        if torch.cuda.is_available():
            assert provider.device.type == "cuda"
        else:
            assert provider.device.type == "cpu"

    def test_explicit_device_overrides_default(self):
        dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        provider = CalibrationDataProvider(
            dataloader=dataloader,
            observation_keys=["left"],
            num_calibration_steps=1,
            device=torch.device("cpu"),
        )

        assert provider.device.type == "cpu"

    def test_yielded_tensors_on_specified_device(
        self,
        rng: np.random.Generator,
    ):
        observation = {
            "left": torch.from_numpy(rng.standard_normal((2, 4)).astype(np.float32)),
        }
        batch = {SampleKey.OBSERVATION.value: observation}
        dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        dataloader.__iter__ = MagicMock(return_value=iter([batch]))

        provider = CalibrationDataProvider(
            dataloader=dataloader,
            observation_keys=["left"],
            num_calibration_steps=1,
            device=torch.device("cpu"),
        )

        result = next(iter(provider))

        assert result[0].device.type == "cpu"

    @pytest.mark.requires_gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_yielded_tensors_on_cuda(
        self,
        rng: np.random.Generator,
    ):
        observation = {
            "left": torch.from_numpy(rng.standard_normal((2, 4)).astype(np.float32)),
        }
        batch = {SampleKey.OBSERVATION.value: observation}
        dataloader = MagicMock(spec=torch.utils.data.DataLoader)
        dataloader.__iter__ = MagicMock(return_value=iter([batch]))

        provider = CalibrationDataProvider(
            dataloader=dataloader,
            observation_keys=["left"],
            num_calibration_steps=1,
            device=torch.device("cuda"),
        )

        result = next(iter(provider))

        assert result[0].device.type == "cuda"
