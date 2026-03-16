"""Temporal aggregation for action sequences."""

import numpy as np
import torch


class TemporalAggregator:
    """Exponential-weighted temporal averaging of overlapping action predictions.

    Accumulates action chunks over time and returns a weighted average
    for the current step. Works with any set of action keys and dimensions.
    """

    def __init__(
        self,
        device: torch.device,
        action_keys_to_dimensions: dict[str, int],
        prediction_horizon: int,
        max_timesteps: int = 10000,
        exponential_decay: float = 0.01,
        favor_more_recent: bool = True,
    ):
        """Initialize temporal aggregator.

        Args:
            device: Torch device for tensors.
            action_keys_to_dimensions: Mapping from action key to dimension.
            prediction_horizon: Number of future steps predicted per inference.
            max_timesteps: Maximum episode length.
            exponential_decay: Decay factor for exponential weighting.
            favor_more_recent: Whether to weight newer predictions more heavily.
        """
        self.device = device
        self.action_keys_to_dimensions = action_keys_to_dimensions
        self.prediction_horizon = prediction_horizon
        self.max_timesteps = max_timesteps
        self.exponential_decay = exponential_decay
        self.favor_more_recent = favor_more_recent
        self.timestep = 0
        total_length = self.max_timesteps + self.prediction_horizon
        self.populated_mask = torch.zeros(
            [self.max_timesteps, total_length],
            dtype=torch.bool,
        ).to(self.device)
        self.action_histories: dict[str, torch.Tensor] = {}
        for key, dimension in self.action_keys_to_dimensions.items():
            self.action_histories[key] = torch.zeros(
                [self.max_timesteps, total_length, dimension]
            ).to(self.device)

    def store_and_average(
        self, current_predictions: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Store current predictions and return averaged action for this timestep.

        Args:
            current_predictions: Dict mapping action key to predicted tensor
                of shape (prediction_horizon, dimension).

        Returns:
            Dict mapping action key to averaged tensor of shape (dimension,).
        """
        horizon_slice = slice(
            self.timestep, self.timestep + self.prediction_horizon
        )
        self.populated_mask[[self.timestep], horizon_slice] = True
        for key, predictions in current_predictions.items():
            self.action_histories[key][
                [self.timestep], horizon_slice
            ] = predictions.float()

        actions_populated = self.populated_mask[:, self.timestep]
        num_populated = int(actions_populated.sum().item())
        exponential_weights = self._compute_exponential_weights(num_populated)
        averaged = {}
        for key in current_predictions:
            actions_for_step = self.action_histories[key][:, self.timestep][
                actions_populated
            ]
            averaged[key] = (actions_for_step * exponential_weights).sum(dim=0)

        self.timestep += 1
        return averaged

    def reset(self) -> None:
        """Reset all state for a new episode."""
        self.timestep = 0
        self.populated_mask.zero_()
        for tensor in self.action_histories.values():
            tensor.zero_()

    def _compute_exponential_weights(
        self, num_predictions: int
    ) -> torch.Tensor:
        """Compute normalized exponential weights.

        Args:
            num_predictions: Number of overlapping predictions to weight.

        Returns:
            Tensor of shape (num_predictions, 1).
        """
        if num_predictions <= 0:
            return torch.empty(0, 1, device=self.device, dtype=torch.float32)
        indices = np.arange(num_predictions)
        if self.favor_more_recent:
            indices = indices[::-1]
        weights = np.exp(-self.exponential_decay * indices)
        weights = weights / weights.sum()
        return (
            torch.from_numpy(weights).to(self.device).float().unsqueeze(dim=1)
        )