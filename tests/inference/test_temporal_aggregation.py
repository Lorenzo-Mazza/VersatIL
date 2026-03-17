"""Tests for versatil.inference.temporal_aggregation module."""

from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.inference.temporal_aggregation import TemporalAggregator


@pytest.fixture
def temporal_aggregator_factory(
    device: torch.device,
) -> Callable[..., TemporalAggregator]:
    def factory(
        action_keys_to_dimensions: dict[str, int] | None = None,
        prediction_horizon: int = 4,
        max_timesteps: int = 20,
        exponential_decay: float = 0.01,
        favor_more_recent: bool = True,
    ) -> TemporalAggregator:
        if action_keys_to_dimensions is None:
            action_keys_to_dimensions = {"position": 3, "gripper": 1}
        return TemporalAggregator(
            device=device,
            action_keys_to_dimensions=action_keys_to_dimensions,
            prediction_horizon=prediction_horizon,
            max_timesteps=max_timesteps,
            exponential_decay=exponential_decay,
            favor_more_recent=favor_more_recent,
        )

    return factory


@pytest.fixture
def prediction_factory(
    rng: np.random.Generator,
    device: torch.device,
) -> Callable[..., dict[str, torch.Tensor]]:
    def factory(
        action_keys_to_dimensions: dict[str, int] | None = None,
        prediction_horizon: int = 4,
    ) -> dict[str, torch.Tensor]:
        if action_keys_to_dimensions is None:
            action_keys_to_dimensions = {"position": 3, "gripper": 1}
        predictions = {}
        for key, dimension in action_keys_to_dimensions.items():
            data = rng.standard_normal((prediction_horizon, dimension)).astype(
                np.float32
            )
            predictions[key] = torch.from_numpy(data).to(device)
        return predictions

    return factory


@pytest.mark.unit
class TestTemporalAggregatorInitialization:
    @pytest.mark.parametrize("prediction_horizon", [2, 8])
    @pytest.mark.parametrize("max_timesteps", [10, 50])
    @pytest.mark.parametrize("exponential_decay", [0.01, 0.1])
    @pytest.mark.parametrize("favor_more_recent", [True, False])
    def test_stores_configuration(
        self,
        temporal_aggregator_factory,
        prediction_horizon,
        max_timesteps,
        exponential_decay,
        favor_more_recent,
    ):
        aggregator = temporal_aggregator_factory(
            prediction_horizon=prediction_horizon,
            max_timesteps=max_timesteps,
            exponential_decay=exponential_decay,
            favor_more_recent=favor_more_recent,
        )
        assert aggregator.prediction_horizon == prediction_horizon
        assert aggregator.max_timesteps == max_timesteps
        assert aggregator.exponential_decay == exponential_decay
        assert aggregator.favor_more_recent == favor_more_recent

    def test_initializes_zero_populated_mask(self, temporal_aggregator_factory):
        aggregator = temporal_aggregator_factory(max_timesteps=10, prediction_horizon=4)
        assert aggregator.populated_mask.shape == (10, 14)
        assert not aggregator.populated_mask.any()

    def test_initializes_zero_action_histories(self, temporal_aggregator_factory):
        keys_to_dims = {"position": 3, "gripper": 1}
        aggregator = temporal_aggregator_factory(
            action_keys_to_dimensions=keys_to_dims,
            max_timesteps=10,
            prediction_horizon=4,
        )
        assert aggregator.action_histories["position"].shape == (10, 14, 3)
        assert aggregator.action_histories["gripper"].shape == (10, 14, 1)
        assert (aggregator.action_histories["position"] == 0).all()
        assert (aggregator.action_histories["gripper"] == 0).all()

    def test_initial_timestep_is_zero(self, temporal_aggregator_factory):
        aggregator = temporal_aggregator_factory()
        assert aggregator.timestep == 0


@pytest.mark.unit
class TestTemporalAggregatorStoreAndAverage:
    def test_first_step_returns_first_predicted_action(
        self,
        temporal_aggregator_factory,
        device,
    ):
        aggregator = temporal_aggregator_factory(
            action_keys_to_dimensions={"position": 2},
            prediction_horizon=3,
        )
        predictions = {
            "position": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device
            )
        }
        result = aggregator.store_and_average(current_predictions=predictions)
        # At timestep 0, only one prediction contributes — no averaging
        torch.testing.assert_close(
            result["position"],
            torch.tensor([1.0, 2.0], device=device),
        )

    def test_advances_timestep(self, temporal_aggregator_factory, prediction_factory):
        aggregator = temporal_aggregator_factory()
        assert aggregator.timestep == 0
        aggregator.store_and_average(current_predictions=prediction_factory())
        assert aggregator.timestep == 1
        aggregator.store_and_average(current_predictions=prediction_factory())
        assert aggregator.timestep == 2

    def test_returns_all_action_keys(self, temporal_aggregator_factory, device):
        keys_to_dims = {"position": 3, "gripper": 1}
        aggregator = temporal_aggregator_factory(
            action_keys_to_dimensions=keys_to_dims,
            prediction_horizon=2,
        )
        predictions = {
            "position": torch.zeros(2, 3, device=device),
            "gripper": torch.zeros(2, 1, device=device),
        }
        result = aggregator.store_and_average(current_predictions=predictions)
        assert set(result.keys()) == {"position", "gripper"}
        assert result["position"].shape == (3,)
        assert result["gripper"].shape == (1,)

    def test_overlapping_predictions_are_averaged(
        self, temporal_aggregator_factory, device
    ):
        aggregator = temporal_aggregator_factory(
            action_keys_to_dimensions={"action": 1},
            prediction_horizon=3,
            exponential_decay=0.0,  # Uniform weights when decay=0
        )
        # Step 0: predict [10, 20, 30] for timesteps 0,1,2
        predictions_0 = {
            "action": torch.tensor([[10.0], [20.0], [30.0]], device=device)
        }
        aggregator.store_and_average(current_predictions=predictions_0)
        # Step 1: predict [21, 31, 40] for timesteps 1,2,3
        # Timestep 1 has predictions: 20 (from step 0) and 21 (from step 1)
        # With uniform weights (decay=0), average = (20 + 21) / 2 = 20.5
        predictions_1 = {
            "action": torch.tensor([[21.0], [31.0], [40.0]], device=device)
        }
        result = aggregator.store_and_average(current_predictions=predictions_1)
        torch.testing.assert_close(
            result["action"],
            torch.tensor([20.5], device=device),
        )

    def test_boundary_at_max_timesteps(self, temporal_aggregator_factory, device):
        aggregator = temporal_aggregator_factory(
            action_keys_to_dimensions={"action": 1},
            prediction_horizon=3,
            max_timesteps=5,
            exponential_decay=0.0,
        )
        results = []
        for step in range(4):
            predictions = {
                "action": torch.full((3, 1), fill_value=float(step), device=device)
            }
            result = aggregator.store_and_average(current_predictions=predictions)
            results.append(result["action"])
        assert aggregator.timestep == 4
        # At step 3, timestep column 3 has contributions from steps 1,2,3
        # (predictions from step 1 cover timesteps 1-3,
        #  step 2 covers 2-4, step 3 covers 3-5)
        # With uniform weights: (1 + 2 + 3) / 3 = 2.0
        torch.testing.assert_close(
            results[3],
            torch.tensor([2.0], device=device),
        )

    def test_favor_more_recent_weights_newer_predictions_higher(
        self, temporal_aggregator_factory, device
    ):
        aggregator_favor_recent = temporal_aggregator_factory(
            action_keys_to_dimensions={"action": 1},
            prediction_horizon=3,
            exponential_decay=1.0,  # Strong decay to make the effect clear
            favor_more_recent=True,
        )
        aggregator_favor_older = temporal_aggregator_factory(
            action_keys_to_dimensions={"action": 1},
            prediction_horizon=3,
            exponential_decay=1.0,
            favor_more_recent=False,
        )
        predictions_0 = {"action": torch.tensor([[0.0], [100.0], [0.0]], device=device)}
        predictions_1 = {"action": torch.tensor([[200.0], [0.0], [0.0]], device=device)}
        # For timestep 1: step 0 predicted 100, step 1 predicted 200
        # favor_more_recent=True: newer (step 1) gets higher weight
        # favor_more_recent=False: older (step 0) gets higher weight
        aggregator_favor_recent.store_and_average(current_predictions=predictions_0)
        result_recent = aggregator_favor_recent.store_and_average(
            current_predictions=predictions_1
        )
        aggregator_favor_older.store_and_average(current_predictions=predictions_0)
        result_older = aggregator_favor_older.store_and_average(
            current_predictions=predictions_1
        )
        # With favor_more_recent, result should be closer to 200 (newer)
        # With favor_older, result should be closer to 100 (older)
        assert result_recent["action"].item() > result_older["action"].item()


@pytest.mark.unit
class TestTemporalAggregatorReset:
    def test_reset_zeroes_timestep(
        self, temporal_aggregator_factory, prediction_factory
    ):
        aggregator = temporal_aggregator_factory()
        aggregator.store_and_average(current_predictions=prediction_factory())
        aggregator.store_and_average(current_predictions=prediction_factory())
        assert aggregator.timestep == 2
        aggregator.reset()
        assert aggregator.timestep == 0

    def test_reset_clears_populated_mask(
        self, temporal_aggregator_factory, prediction_factory
    ):
        aggregator = temporal_aggregator_factory()
        aggregator.store_and_average(current_predictions=prediction_factory())
        assert aggregator.populated_mask.any()
        aggregator.reset()
        assert not aggregator.populated_mask.any()

    def test_reset_clears_action_histories(
        self, temporal_aggregator_factory, prediction_factory
    ):
        aggregator = temporal_aggregator_factory()
        aggregator.store_and_average(current_predictions=prediction_factory())
        aggregator.reset()
        for tensor in aggregator.action_histories.values():
            assert (tensor == 0).all()

    def test_reset_allows_reuse_with_correct_results(
        self, temporal_aggregator_factory, device
    ):
        aggregator = temporal_aggregator_factory(
            action_keys_to_dimensions={"action": 1},
            prediction_horizon=2,
        )
        # First episode
        old_predictions = {"action": torch.tensor([[999.0], [999.0]], device=device)}
        aggregator.store_and_average(current_predictions=old_predictions)
        aggregator.reset()

        # Second episode — should not be contaminated by first
        new_predictions = {"action": torch.tensor([[1.0], [2.0]], device=device)}
        result = aggregator.store_and_average(current_predictions=new_predictions)
        torch.testing.assert_close(
            result["action"],
            torch.tensor([1.0], device=device),
        )


@pytest.mark.unit
class TestComputeExponentialWeights:
    @pytest.mark.parametrize("num_predictions", [1, 3, 10])
    def test_weights_sum_to_one(self, temporal_aggregator_factory, num_predictions):
        aggregator = temporal_aggregator_factory(exponential_decay=0.05)
        weights = aggregator._compute_exponential_weights(
            num_predictions=num_predictions
        )
        torch.testing.assert_close(
            weights.sum(),
            torch.tensor(1.0, device=weights.device),
        )

    def test_single_prediction_returns_weight_one(self, temporal_aggregator_factory):
        aggregator = temporal_aggregator_factory()
        weights = aggregator._compute_exponential_weights(num_predictions=1)
        torch.testing.assert_close(
            weights,
            torch.tensor([[1.0]], device=weights.device),
        )

    def test_output_shape(self, temporal_aggregator_factory):
        aggregator = temporal_aggregator_factory()
        weights = aggregator._compute_exponential_weights(num_predictions=7)
        assert weights.shape == (7, 1)

    def test_favor_more_recent_puts_highest_weight_last(
        self, temporal_aggregator_factory
    ):
        aggregator = temporal_aggregator_factory(
            exponential_decay=0.5, favor_more_recent=True
        )
        weights = aggregator._compute_exponential_weights(num_predictions=5)
        # favor_more_recent reverses indices so most recent (last index)
        # gets exp(0) = 1.0 (highest), oldest gets exp(-0.5*4)
        assert weights[-1].item() > weights[0].item()

    def test_favor_older_puts_highest_weight_first(self, temporal_aggregator_factory):
        aggregator = temporal_aggregator_factory(
            exponential_decay=0.5, favor_more_recent=False
        )
        weights = aggregator._compute_exponential_weights(num_predictions=5)
        # Without reversal, index 0 gets exp(0) = 1.0 (highest)
        assert weights[0].item() > weights[-1].item()

    def test_zero_decay_produces_uniform_weights(self, temporal_aggregator_factory):
        aggregator = temporal_aggregator_factory(exponential_decay=0.0)
        weights = aggregator._compute_exponential_weights(num_predictions=4)
        expected = torch.full((4, 1), fill_value=0.25, device=weights.device)
        torch.testing.assert_close(weights, expected)

    def test_weights_with_zero_predictions_returns_empty(
        self, temporal_aggregator_factory
    ):
        aggregator = temporal_aggregator_factory()
        weights = aggregator._compute_exponential_weights(num_predictions=0)
        assert weights.shape == (0, 1)
        assert weights.dtype == torch.float32

    def test_exact_weight_values_for_known_decay(self, temporal_aggregator_factory):
        # num_predictions=3, decay=1.0, favor_more_recent=True
        # indices [0,1,2] reversed to [2,1,0]
        # raw weights: exp(-2), exp(-1), exp(0)
        # normalized by sum
        aggregator = temporal_aggregator_factory(
            exponential_decay=1.0, favor_more_recent=True
        )
        weights = aggregator._compute_exponential_weights(num_predictions=3)
        raw = np.exp(-1.0 * np.array([2.0, 1.0, 0.0]))
        expected = raw / raw.sum()
        torch.testing.assert_close(
            weights,
            torch.tensor(
                expected, dtype=torch.float32, device=weights.device
            ).unsqueeze(dim=1),
        )
