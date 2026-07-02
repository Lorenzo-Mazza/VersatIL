"""Action processing module for episodic dataset.

Handles all action-related computations including:
- Computing actions from observations
- Orientation delta computation
- Action denoising
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

from versatil.data.constants import (
    ActionComputationMethod,
    GripperType,
    OrientationRepresentation,
    ProprioceptiveType,
)
from versatil.data.metadata import (
    ActionMetadata,
    ObservationMetadata,
    OnTheFlyActionMetadata,
    PositionObservationMetadata,
)
from versatil.data.task import ActionSpace


class ActionProcessor:
    """Computes actions from robot observations with denoising support."""

    def __init__(
        self,
        action_space: ActionSpace,
    ):
        """Initialize action processor.

        Args:
            action_space: TaskSpace action space configuration (contains all action-related settings)
        """
        self.action_space: ActionSpace = action_space
        self.denoise_actions: bool = action_space.denoise_actions
        self.denoising_percentile: float = action_space.denoising_percentile
        self.denoising_thresholds: dict[str, float] = {}
        self._denoising_thresholds_computed: bool = False
        self.dataset_magnitudes: dict[str, np.ndarray] = {}

    @property
    def requires_denoising_setup(self) -> bool:
        """Check if denoising is enabled but thresholds haven't been computed.

        Returns:
            True if denoise_actions is enabled but thresholds haven't been computed yet.
        """
        return self.denoise_actions and not self._denoising_thresholds_computed

    def compute_denoising_threshold(
        self,
        obs_data: np.ndarray,
        key: str,
        meta: ObservationMetadata,
        episode_ends: np.ndarray,
    ) -> None:
        """Compute denoising thresholds from observation deltas of proprioceptive positions."""
        if not isinstance(meta, PositionObservationMetadata):
            logging.warning(
                "Denoising threshold computation only supported for position observations."
            )
            return
        # Mask out cross-episode transitions
        cross_indices = episode_ends[:-1] - 1
        valid_mask = np.ones(len(obs_data) - 1, dtype=bool)
        valid_mask[cross_indices] = False
        deltas = obs_data[1:] - obs_data[:-1]  # (T-1, dim)
        deltas = deltas[valid_mask]
        norms = np.linalg.norm(deltas, axis=1)
        non_zero = norms[norms > 0]
        self.dataset_magnitudes[key] = norms
        self.denoising_thresholds[key] = np.percentile(
            non_zero, self.denoising_percentile
        )
        self._denoising_thresholds_computed = True

    def compute_sample_actions(
        self,
        padded_data: dict[str, np.ndarray],
        action_slice_start: int,
        action_slice_end: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, ActionMetadata]]:
        """Compute actions from a sampled sequence of the zarr replay buffer.

        Precomputed actions are extracted directly from the buffer.
        On-the-fly actions are computed from current & next observations.
        Additionally, denoising is applied to on-the-fly actions if enabled.
        """
        action_data = {}
        action_meta = {}
        for key, meta in self.action_space.actions_metadata.items():
            if meta.is_precomputed:
                precomputed_actions = padded_data[key]
                processed_action = precomputed_actions[
                    action_slice_start:action_slice_end
                ]
            else:
                if not isinstance(meta, OnTheFlyActionMetadata):
                    raise TypeError(
                        f"Action '{key}' is not precomputed, so its metadata must "
                        f"be OnTheFlyActionMetadata, got {type(meta).__name__}"
                    )
                obs_for_action = padded_data[key]
                next_obs = obs_for_action[action_slice_start + 1 : action_slice_end + 1]
                current_obs = obs_for_action[action_slice_start:action_slice_end]
                if (
                    self.denoise_actions
                    and meta.action_type == ProprioceptiveType.POSITION.value
                ):
                    next_obs, current_obs = self.apply_delta_denoising(
                        key=key, next_values=next_obs, current_values=current_obs
                    )
                processed_action = self.compute_action_on_the_fly(
                    current_obs=current_obs, next_obs=next_obs, metadata=meta
                )

            action_data[key] = processed_action
            action_meta[key] = meta

        return action_data, action_meta

    def log_movement_distribution(self) -> None:
        """Log movement (observation delta) distribution stats."""
        for key, norms in self.dataset_magnitudes.items():
            logging.info(
                f"{key} movement stats: "
                f"mean={norms.mean():.6f}, std={norms.std():.6f}, "
                f"min={norms.min():.6f}, max={norms.max():.6f}, "
                f"p5={np.percentile(norms, 5):.6f}, p50={np.percentile(norms, 50):.6f}, "
                f"p95={np.percentile(norms, 95):.6f}"
            )

            if key in self.denoising_thresholds:
                threshold = self.denoising_thresholds[key]
                num_below = np.sum(norms < threshold)
                pct_below = 100.0 * num_below / len(norms)
                logging.info(
                    f"{key} denoising: threshold={threshold:.6f}, "
                    f"{num_below}/{len(norms)} ({pct_below:.1f}%) movements zeroed"
                )

    def compute_action_on_the_fly(
        self,
        current_obs: np.ndarray,
        next_obs: np.ndarray,
        metadata: OnTheFlyActionMetadata,
    ) -> np.ndarray:
        """Compute action from current and next observations."""
        match metadata.action_type:
            case ProprioceptiveType.POSITION.value:
                return self.compute_position_action_from_observation(
                    current_position=current_obs,
                    next_position=next_obs,
                    method=metadata.computation_method,
                )
            case ProprioceptiveType.ORIENTATION.value:
                return self.compute_orientation_action_from_observation(
                    current_orientation=current_obs,
                    next_orientation=next_obs,
                    method=metadata.computation_method,
                    representation=metadata.source_metadata.orientation_representation,
                )
            case ProprioceptiveType.GRIPPER.value:
                return self.compute_gripper_action_from_observation(
                    current_gripper=current_obs,
                    next_gripper=next_obs,
                    method=metadata.computation_method,
                    gripper_type=metadata.source_metadata.gripper_type,
                )
            case _:
                raise ValueError(
                    f"Unsupported action type for on-the-fly computation: {metadata.action_type}"
                )

    @staticmethod
    def compute_position_action_from_observation(
        current_position: np.ndarray,
        next_position: np.ndarray,
        method: str = ActionComputationMethod.DELTA.value,
    ) -> np.ndarray:
        """Compute position action from current and next positions."""
        match method:
            case ActionComputationMethod.NEXT_TIMESTEP.value:
                return next_position
            case ActionComputationMethod.DELTA.value:
                return next_position - current_position
            case _:
                raise ValueError(
                    f"Unsupported position action computation method: {method}"
                )

    @staticmethod
    def compute_gripper_action_from_observation(
        current_gripper: np.ndarray,
        next_gripper: np.ndarray,
        method: str = ActionComputationMethod.DELTA.value,
        gripper_type: str = GripperType.BINARY.value,
    ) -> np.ndarray:
        """Compute gripper action from current and next gripper states."""
        if gripper_type == GripperType.BINARY.value:
            if method != ActionComputationMethod.NEXT_TIMESTEP.value:
                raise ValueError(
                    "Delta not supported for binary grippers; use NEXT_TIMESTEP"
                )
            return next_gripper
        if method == ActionComputationMethod.NEXT_TIMESTEP.value:
            return next_gripper
        elif method == ActionComputationMethod.DELTA.value:
            return next_gripper - current_gripper
        else:
            raise ValueError(f"Unsupported method: {method}")

    def compute_orientation_action_from_observation(
        self,
        current_orientation: np.ndarray,
        next_orientation: np.ndarray,
        method: str = ActionComputationMethod.DELTA.value,
        representation: str = OrientationRepresentation.QUATERNION.value,
    ) -> np.ndarray:
        """Compute orientation action from current and next orientations."""
        if method == ActionComputationMethod.NEXT_TIMESTEP.value:
            return next_orientation
        if method != ActionComputationMethod.DELTA.value:
            raise ValueError(
                f"Unsupported method '{method}' for representation '{representation}'"
            )
        if representation == OrientationRepresentation.ROLL.value:
            return self._compute_roll_deltas(current_orientation, next_orientation)
        elif representation == OrientationRepresentation.QUATERNION.value:
            return self._compute_quaternion_deltas(
                current_orientation, next_orientation
            )
        elif representation == OrientationRepresentation.EULER.value:
            return self._compute_euler_deltas(current_orientation, next_orientation)
        else:
            raise ValueError(
                f"Unsupported orientation representation: {representation}"
            )

    @staticmethod
    def _compute_quaternion_deltas(
        curr_ori: np.ndarray, next_ori: np.ndarray
    ) -> np.ndarray:
        """Compute quaternion deltas (w, x, y, z format)."""
        quat_order_from = [1, 2, 3, 0]  # to (x,y,z,w)
        curr_rot = scipy.spatial.transform.Rotation.from_quat(
            curr_ori[:, quat_order_from]
        )
        next_rot = scipy.spatial.transform.Rotation.from_quat(
            next_ori[:, quat_order_from]
        )
        rel_rot = next_rot * curr_rot.inv()
        quat_order_to = [3, 0, 1, 2]  # back to (w,x,y,z)
        return rel_rot.as_quat()[:, quat_order_to]

    @staticmethod
    def _compute_roll_deltas(curr_ori: np.ndarray, next_ori: np.ndarray) -> np.ndarray:
        """Compute roll angle deltas wrapped to [-pi, pi].

        Wrapping keeps deltas continuous when the stored angle crosses the
        +pi/-pi boundary; the raw difference would spike by ~2*pi there and
        corrupt both the training label and the normalization statistics.

        Args:
            curr_ori: Current roll angles (N, 1)
            next_ori: Next roll angles (N, 1)

        Returns:
            Roll deltas (N, 1) in [-pi, pi]
        """
        delta = next_ori - curr_ori
        return np.arctan2(np.sin(delta), np.cos(delta))

    @staticmethod
    def _compute_euler_deltas(curr_ori: np.ndarray, next_ori: np.ndarray) -> np.ndarray:
        """Compute euler angle deltas."""
        curr_rot = scipy.spatial.transform.Rotation.from_euler("xyz", curr_ori)
        next_rot = scipy.spatial.transform.Rotation.from_euler("xyz", next_ori)
        rel_rot = next_rot * curr_rot.inv()
        return rel_rot.as_euler("xyz")

    def apply_delta_denoising(
        self, next_values: np.ndarray, current_values: np.ndarray, key: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply denoising threshold to the magnitudes of delta movements between a quantity at time t and t+1.

        Raises:
            RuntimeError: If denoising thresholds haven't been computed.
        """
        if self.requires_denoising_setup:
            raise RuntimeError(
                "Denoising is enabled but thresholds have not been computed."
            )
        if key in self.denoising_thresholds:
            next_values = next_values.copy()
            diffs = next_values - current_values
            norms = np.linalg.norm(diffs, axis=1)
            mask = norms < self.denoising_thresholds[key]
            next_values[mask] = current_values[mask]
        return next_values, current_values

    def plot_action_magnitude_distribution(self) -> plt.Figure | None:
        """Plot magnitude distributions of the actions before/after denoising.

        Returns:
            The matplotlib figure, or None if no data available.
        """
        if not self.denoising_thresholds:
            logging.warning("No denoising data available to plot")
            return None

        sns.set_theme(style="whitegrid", palette="muted")
        num_plots = len(self.denoising_thresholds)
        fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]
        for plot_idx, key in enumerate(self.denoising_thresholds):
            ax = axes[plot_idx]
            norms = self.dataset_magnitudes[key]
            threshold = self.denoising_thresholds[key]
            log_norms = np.log10(norms + 1e-10)
            sns.histplot(
                log_norms, bins=100, alpha=0.6, label="Raw", color="steelblue", ax=ax
            )
            if threshold > 0:
                log_threshold = np.log10(threshold)
                denoised = np.where(norms < threshold, 0.0, norms)
                denoised_nonzero = denoised[denoised > 0]
                if len(denoised_nonzero) > 0:
                    sns.histplot(
                        np.log10(denoised_nonzero),
                        bins=100,
                        alpha=0.6,
                        label="Denoised",
                        color="coral",
                        ax=ax,
                    )
                ax.axvline(
                    log_threshold,
                    color="crimson",
                    linestyle="--",
                    linewidth=2,
                    label=f"Threshold: {threshold:.2e}",
                )
            ax.set_xlabel(f"log10({key} Observation Deltas Magnitude)", fontsize=11)
            ax.set_ylabel("Count", fontsize=11)
            ax.set_title(
                f"{key} Observation Deltas Distribution", fontsize=13, fontweight="bold"
            )
            ax.legend(frameon=True, fancybox=True)

        plt.tight_layout()
        sns.reset_defaults()
        return fig
