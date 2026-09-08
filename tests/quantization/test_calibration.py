"""Tests for versatil.quantization.calibration module."""

import re
from collections.abc import Callable, Iterator
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torchao.prototype.smoothquant.api import SmoothQuantConfig
from torchao.prototype.smoothquant.core import SmoothQuantObservedLinear
from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
from torchao.quantization.quantize_.common.quantization_step import QuantizationStep

from versatil.data.constants import SampleKey
from versatil.models.layers.denoising.diffusion_process import SchedulerType
from versatil.models.policy import Policy
from versatil.quantization.calibration import (
    CalibrationDataProvider,
    build_calibration_data,
    calibrate_policy,
)

CALIBRATION_MODULE = "versatil.quantization.calibration"


@pytest.fixture
def observation_batch_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, dict[str, torch.Tensor]]]:
    def factory(
        observation_keys: list[str],
        batch_size: int = 2,
        feature_dimension: int = 4,
        dtype: torch.dtype = torch.float32,
    ) -> dict[str, dict[str, torch.Tensor]]:
        observation = {}
        for key in observation_keys:
            data = rng.integers(low=0, high=8, size=(batch_size, feature_dimension))
            observation[key] = torch.from_numpy(data).to(
                dtype=dtype
            )  # (batch_size, feature_dimension)
        return {SampleKey.OBSERVATION.value: observation}

    return factory


@pytest.fixture
def mock_dataloader_factory() -> Callable[..., MagicMock]:
    def factory(batches: list[dict[str, dict[str, torch.Tensor]]]) -> MagicMock:
        dataloader = MagicMock(spec=DataLoader)
        dataloader.consumed_batches = 0

        def iterate() -> Iterator[dict[str, dict[str, torch.Tensor]]]:
            for batch in batches:
                dataloader.consumed_batches += 1
                yield batch

        dataloader.__iter__.side_effect = iterate
        return dataloader

    return factory


@pytest.mark.unit
class TestCalibratePolicy:
    @pytest.mark.parametrize("num_batches", [1, 3])
    def test_runs_processed_prediction_without_gradients_for_every_batch(
        self,
        mock_policy_factory: Callable[..., MagicMock],
        mock_calibration_provider_factory: Callable[..., MagicMock],
        num_batches: int,
    ) -> None:
        policy = mock_policy_factory()
        policy.training = False
        calibration = mock_calibration_provider_factory(
            observation_keys=["observations"], num_batches=num_batches
        )
        gradients_enabled = []
        predict = policy.predict_from_processed_observation
        predict.side_effect = lambda observation: gradients_enabled.append(
            torch.is_grad_enabled()
        )
        count = calibrate_policy(policy=policy, calibration=calibration)
        assert count == num_batches
        assert gradients_enabled == [False] * num_batches
        assert predict.call_count == num_batches
        for entry, observation in zip(predict.call_args_list, calibration, strict=True):
            assert entry.kwargs == {"observation": observation}
        policy.forward.assert_not_called()
        policy.predict_action.assert_not_called()
        policy.train.assert_not_called()

    def test_training_mode_is_rejected_before_consuming_data(
        self, mock_policy_factory: Callable[..., MagicMock]
    ) -> None:
        policy = mock_policy_factory()
        policy.training = True
        calibration = MagicMock(spec=CalibrationDataProvider)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Policy calibration requires evaluation mode. Call policy.eval() first."
            ),
        ):
            calibrate_policy(policy=policy, calibration=calibration)
        calibration.__iter__.assert_not_called()
        policy.predict_from_processed_observation.assert_not_called()

    def test_empty_calibration_is_rejected(
        self, mock_policy_factory: Callable[..., MagicMock]
    ) -> None:
        policy = mock_policy_factory()
        policy.training = False
        calibration = MagicMock(spec=CalibrationDataProvider)
        calibration.__iter__.return_value = iter([])
        with pytest.raises(
            ValueError,
            match=re.escape("Calibration data yielded no observation batches."),
        ):
            calibrate_policy(policy=policy, calibration=calibration)
        policy.predict_from_processed_observation.assert_not_called()


@pytest.mark.unit
class TestCalibrationDataProvider:
    @pytest.mark.parametrize("available_batches, limit", [(5, 2), (2, 5)])
    def test_consumes_only_requested_batches(
        self,
        observation_batch_factory: Callable,
        mock_dataloader_factory: Callable,
        available_batches: int,
        limit: int,
    ) -> None:
        batch = observation_batch_factory(observation_keys=["left"])
        dataloader = mock_dataloader_factory(batches=[batch] * available_batches)
        provider = CalibrationDataProvider(
            dataloader=dataloader,
            observation_keys=["left"],
            num_calibration_steps=limit,
            device=torch.device("cpu"),
        )

        result = list(provider)

        assert len(result) == min(available_batches, limit)
        assert dataloader.consumed_batches == min(available_batches, limit)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.int64, torch.bool])
    def test_selects_named_inputs_and_preserves_values_and_dtypes(
        self,
        observation_batch_factory: Callable,
        mock_dataloader_factory: Callable,
        dtype: torch.dtype,
    ) -> None:
        batch = observation_batch_factory(
            observation_keys=["alpha", "beta", "unused"], dtype=dtype
        )
        keys = ["beta", "alpha"]
        provider = CalibrationDataProvider(
            dataloader=mock_dataloader_factory(batches=[batch]),
            observation_keys=keys,
            num_calibration_steps=1,
            device=torch.device("cpu"),
        )
        keys.clear()

        result = next(iter(provider))

        assert set(result) == {"alpha", "beta"}
        for key, tensor in result.items():
            torch.testing.assert_close(tensor, batch[SampleKey.OBSERVATION.value][key])
            assert tensor.dtype == dtype
            assert tensor.device.type == "cpu"
        assert "unused" in batch[SampleKey.OBSERVATION.value]

    def test_reiterates_dataloader_for_each_calibration_pass(
        self,
        observation_batch_factory: Callable,
        mock_dataloader_factory: Callable,
    ) -> None:
        batch = observation_batch_factory(observation_keys=["left"])
        dataloader = mock_dataloader_factory(batches=[batch])
        provider = CalibrationDataProvider(
            dataloader=dataloader,
            observation_keys=["left"],
            num_calibration_steps=1,
            device=torch.device("cpu"),
        )

        first = list(provider)
        second = list(provider)

        torch.testing.assert_close(first, second)
        assert dataloader.consumed_batches == 2

    @pytest.mark.parametrize("limit", [0, -1, 1])
    def test_requires_positive_batch_limit(
        self, mock_dataloader_factory: Callable, limit: int
    ) -> None:
        expectation = (
            does_not_raise()
            if limit > 0
            else pytest.raises(
                ValueError,
                match=re.escape(
                    f"num_calibration_steps must be positive, got {limit}."
                ),
            )
        )
        with expectation:
            CalibrationDataProvider(
                dataloader=mock_dataloader_factory(batches=[]),
                observation_keys=["left"],
                num_calibration_steps=limit,
                device=torch.device("cpu"),
            )

    def test_rejects_empty_dataloader(self, mock_dataloader_factory: Callable) -> None:
        provider = CalibrationDataProvider(
            dataloader=mock_dataloader_factory(batches=[]),
            observation_keys=["left"],
            num_calibration_steps=1,
            device=torch.device("cpu"),
        )
        with pytest.raises(
            ValueError,
            match=re.escape("Calibration dataloader yielded no observation batches."),
        ):
            list(provider)

    def test_identifies_missing_model_inputs(
        self,
        observation_batch_factory: Callable,
        mock_dataloader_factory: Callable,
    ) -> None:
        batch = observation_batch_factory(observation_keys=["left"])
        provider = CalibrationDataProvider(
            dataloader=mock_dataloader_factory(batches=[batch]),
            observation_keys=["left", "right"],
            num_calibration_steps=1,
            device=torch.device("cpu"),
        )
        with pytest.raises(
            ValueError,
            match=re.escape("Calibration batch is missing observations: ['right']."),
        ):
            list(provider)


@pytest.mark.unit
class TestBuildCalibrationData:
    @pytest.mark.parametrize("batch_size, observation_horizon", [(2, 1), (4, 3)])
    def test_uses_checkpoint_preprocessing_without_augmentation_or_shuffle(
        self,
        calibration_context_factory: Callable[..., MagicMock],
        batch_size: int,
        observation_horizon: int,
    ) -> None:
        context = calibration_context_factory(
            batch_size=batch_size, observation_horizon=observation_horizon
        )
        with (
            patch(f"{CALIBRATION_MODULE}.EpisodicDataset") as dataset_factory,
            patch(f"{CALIBRATION_MODULE}.DataLoader") as loader_factory,
            patch(f"{CALIBRATION_MODULE}.CalibrationDataProvider") as provider_factory,
        ):
            result = build_calibration_data(
                context=context,
                observation_keys=["left", "right"],
                num_calibration_steps=8,
                device=torch.device("cpu"),
            )

        dataset_factory.assert_called_once_with(
            zarr_path="/dataset.zarr",
            action_space=context.config.task.action_space,
            observation_space=context.observation_space,
            dataloader_config=context.config.task.dataloader,
            pred_horizon=4,
            obs_horizon=observation_horizon,
            train=True,
            seed=42,
            augment_images=False,
        )
        dataset = dataset_factory.return_value
        dataset.set_normalizer.assert_called_once_with(
            normalizer=context.policy.normalizer
        )
        dataset.set_tokenizer.assert_called_once_with(tokenizer=context.tokenizer)
        loader_factory.assert_called_once_with(
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        provider_factory.assert_called_once_with(
            dataloader=loader_factory.return_value,
            observation_keys=["left", "right"],
            num_calibration_steps=8,
            device=torch.device("cpu"),
        )
        assert result is provider_factory.return_value

    @pytest.mark.parametrize("limit", [0, -1])
    def test_rejects_invalid_limit_before_opening_dataset(
        self, calibration_context_factory: Callable, limit: int
    ) -> None:
        context = calibration_context_factory(batch_size=2, observation_horizon=1)
        with (
            patch(f"{CALIBRATION_MODULE}.EpisodicDataset") as dataset_factory,
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"num_calibration_steps must be positive, got {limit}."
                ),
            ),
        ):
            build_calibration_data(
                context=context,
                observation_keys=["left"],
                num_calibration_steps=limit,
                device=torch.device("cpu"),
            )
        dataset_factory.assert_not_called()


@pytest.mark.integration
@pytest.mark.requires_gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_calibration_moves_observations_to_cuda(
    observation_batch_factory: Callable,
) -> None:
    batch = observation_batch_factory(observation_keys=["left"], dtype=torch.int64)
    provider = CalibrationDataProvider(
        dataloader=DataLoader([batch], batch_size=None),
        observation_keys=["left"],
        num_calibration_steps=1,
        device=torch.device("cuda"),
    )

    result = next(iter(provider))

    assert result["left"].device.type == "cuda"
    torch.testing.assert_close(
        result["left"].cpu(), batch[SampleKey.OBSERVATION.value]["left"]
    )  # (batch_size, feature_dimension)


@pytest.mark.integration
@pytest.mark.parametrize(
    "family, scheduler_type, minimum_calls",
    [
        ("flow", SchedulerType.DDIM.value, 3),
        ("diffusion", SchedulerType.DDIM.value, 3),
        ("diffusion", SchedulerType.DDPM.value, 3),
        ("tokens", SchedulerType.DDIM.value, 6),
    ],
)
def test_module_observers_see_denoising_steps_and_generated_tokens(
    quantization_policy_factory: Callable[..., tuple[Policy, dict[str, torch.Tensor]]],
    family: str,
    scheduler_type: str,
    minimum_calls: int,
) -> None:
    policy, observation = quantization_policy_factory(
        family=family,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=2,
        scheduler_type=scheduler_type,
    )
    provider = CalibrationDataProvider(
        dataloader=DataLoader(
            [{SampleKey.OBSERVATION.value: observation}], batch_size=None
        ),
        observation_keys=policy.input_keys,
        num_calibration_steps=1,
        device=torch.device("cpu"),
    )
    # Creating a DataLoader iterator draws a seed; fetch data before comparing noise.
    processed_observation = next(iter(provider))
    with torch.no_grad():
        policy.predict_from_processed_observation(
            observation=processed_observation
        )  # actions: (batch, horizon, action_dim); tokens: (batch, token_length)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        expected = policy.predict_from_processed_observation(
            observation=processed_observation
        )  # actions: (batch, horizon, action_dim); tokens: (batch, token_length)
    # Observer installation initializes temporary weights before retaining the originals.
    with torch.random.fork_rng(devices=[]):
        quantize_(
            model=policy.decoder,
            config=SmoothQuantConfig(
                base_config=Int8DynamicActivationInt8WeightConfig(version=2),
                step=QuantizationStep.PREPARE,
                alpha=0.5,
            ),
        )
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        actual = policy.predict_from_processed_observation(
            observation=processed_observation
        )  # actions: (batch, horizon, action_dim); tokens: (batch, token_length)

    torch.testing.assert_close(actual, expected)
    observers = [
        module.obs
        for module in policy.decoder.modules()
        if isinstance(module, SmoothQuantObservedLinear)
    ]
    assert max(len(observer.inputs) for observer in observers) >= minimum_calls
    for observer in observers:
        for inputs in observer.inputs:
            assert torch.isfinite(inputs).all()
            assert not inputs.requires_grad
