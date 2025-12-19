"""Action processing module for episodic dataset.

Handles all action-related computations including:
- Computing actions from observations
- Orientation delta computation
- Action denoising
"""

import logging

import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

from refactoring.data.task import ActionSpace
from refactoring.data.constants import (
    GRIPPER_ACTION_KEY,
    ORIENTATION_ACTION_KEY,
    POSITION_ACTION_KEY,
    GripperType,
    OrientationRepresentation,
)


class ActionProcessor:
    """Computes actions from robot observations with denoising support."""

    def __init__(self, action_space: ActionSpace):
        """Initialize action processor.

        Args:
            action_space: TaskSpace action space configuration (contains all action-related settings)
        """
        self.action_space = action_space
        self.predict_in_camera_frame = action_space.predict_in_camera_frame
        self.deltas_as_actions = action_space.deltas_as_actions
        self.denoise_actions = action_space.denoise_actions
        self.has_position = action_space.has_position
        self.has_orientation = action_space.has_orientation
        self.has_gripper = action_space.has_gripper
        self.position_dim = action_space.position_dim if self.has_position else 0
        self.orientation_dim = action_space.orientation_dim if self.has_orientation else 0
        self.gripper_dim = action_space.gripper_dim if self.has_gripper else 0
        self.denoising_percentile = action_space.denoising_percentile
        self.action_denoising_threshold = 0.0
        self.orientation_denoising_threshold = 0.0
        self._denoising_thresholds_computed = False
        self._position_norms: np.ndarray | None = None
        self._orientation_angles: np.ndarray | None = None


    @property
    def requires_denoising_setup(self) -> bool:
        """Check if denoising is enabled but thresholds haven't been computed.

        Returns:
            True if denoise_actions is enabled but thresholds haven't been computed yet.
        """
        return self.denoise_actions and not self._denoising_thresholds_computed


    def compute_denoising_thresholds(
        self, curr_obs: np.ndarray, next_obs: np.ndarray
    ) -> None:
        """Compute and store denoising thresholds from all training data.

        This must be called before compute_actions_from_observations when
        denoise_actions is enabled. Should be called once during dataset
        initialization with all training samples.

        Args:
            curr_obs: All current observations from training data (N, obs_dim)
            next_obs: All next observations from training data (N, obs_dim)
        """
        if not self.denoise_actions:
            self._denoising_thresholds_computed = True
            return

        if self.has_position:
            curr_pos = curr_obs[:, :self.position_dim]
            next_pos = next_obs[:, :self.position_dim]
            self.compute_action_denoising_threshold(next_pos, curr_pos)

        if self.has_orientation:
            pos_end = self.position_dim
            ori_end = pos_end + self.orientation_dim
            curr_ori = curr_obs[:, pos_end:ori_end]
            next_ori = next_obs[:, pos_end:ori_end]
            self.compute_orientation_denoising_threshold(next_ori, curr_ori)

        self._denoising_thresholds_computed = True


    def compute_actions_from_observations(
        self,
        curr_obs: np.ndarray,
        next_obs: np.ndarray,
        curr_gripper: np.ndarray | None = None,
        next_gripper: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute action dictionary from current and next observations.

        Args:
            curr_obs: Current observations (N, obs_dim)
            next_obs: Next observations (N, obs_dim)
            curr_gripper: Current gripper states (N, gripper_dim) - optional
            next_gripper: Next gripper states (N, gripper_dim) - optional

        Returns:
            Dictionary with action arrays for each modality
        """
        actions = {}
        if self.has_position:
            next_pos = next_obs[:, : self.position_dim]
            curr_pos = curr_obs[:, : self.position_dim]
            if self.denoise_actions:
                next_pos, curr_pos = self.apply_position_denoising(
                    next_pos, curr_pos
                )
            actions[POSITION_ACTION_KEY] = (
                (next_pos - curr_pos) if self.deltas_as_actions else next_pos
            )

        if self.has_orientation:
            pos_end = self.position_dim
            ori_end = pos_end + self.orientation_dim
            next_ori = next_obs[:, pos_end:ori_end]
            curr_ori = curr_obs[:, pos_end:ori_end]
            if self.denoise_actions:
                next_ori, curr_ori = self.apply_orientation_denoising(
                    next_ori, curr_ori
                )
            actions[ORIENTATION_ACTION_KEY] = (
                self._compute_orientation_deltas(curr_ori, next_ori)
                if self.deltas_as_actions
                else next_ori
            )
        if self.has_gripper and next_gripper is not None:
            actions[GRIPPER_ACTION_KEY] = self.compute_gripper_actions(
                curr_gripper, next_gripper
            )
        return actions

    def compute_gripper_actions(
        self,
        curr_gripper: np.ndarray | None,
        next_gripper: np.ndarray,
    ) -> np.ndarray:
        """Compute gripper actions from gripper states.

        For most cases, gripper action is simply the next timestep's state.
        This works for both binary (0/1) and continuous (0.0-1.0) grippers.

        If deltas_as_actions is True and gripper is continuous, compute deltas.
        Binary grippers always use next state (not deltas).

        Args:
            curr_gripper: Current gripper states (N, gripper_dim)
            next_gripper: Next gripper states (N, gripper_dim)

        Returns:
            Gripper actions (N, gripper_dim)
        """

        gripper_type = self.action_space.gripper_type

        if gripper_type == GripperType.BINARY.value:
            # Binary gripper: action is next state (open/close command)
            return next_gripper
        elif gripper_type == GripperType.CONTINUOUS.value:
            # Continuous gripper: can use deltas if requested
            if self.deltas_as_actions and curr_gripper is not None:
                return next_gripper - curr_gripper  # type: ignore[no-any-return]
            else:
                return next_gripper
        else:
            raise ValueError(f"Unsupported gripper type: {gripper_type}")


    def _compute_orientation_deltas(
        self, curr_ori: np.ndarray, next_ori: np.ndarray
    ) -> np.ndarray:
        """Compute orientation deltas based on representation type."""
        ori_repr = self.action_space.orientation_repr
        if ori_repr == OrientationRepresentation.ROLL.value:
            return self._compute_roll_deltas(curr_ori, next_ori)
        elif ori_repr == OrientationRepresentation.QUATERNION.value:
            return self._compute_quaternion_deltas(curr_ori, next_ori)
        elif ori_repr == OrientationRepresentation.EULER.value:
            return self._compute_euler_deltas(curr_ori, next_ori)

        else:
            raise ValueError(f"Unsupported orientation representation: {ori_repr}")


    def _compute_quaternion_deltas(
            self, curr_ori: np.ndarray, next_ori: np.ndarray
    ) -> np.ndarray:
        """Compute quaternion deltas (w, action_embedding, y, z format)."""
        quat_order_from = [1, 2, 3, 0]  # to (action_embedding,y,z,w)
        curr_rot = scipy.spatial.transform.Rotation.from_quat(curr_ori[:, quat_order_from])
        next_rot = scipy.spatial.transform.Rotation.from_quat(next_ori[:, quat_order_from])
        rel_rot = next_rot * curr_rot.inv()
        quat_order_to = [3, 0, 1, 2]  # back to (w,action_embedding,y,z)
        return rel_rot.as_quat()[:, quat_order_to]  # type: ignore[no-any-return]


    def _compute_roll_deltas(
            self, curr_ori: np.ndarray, next_ori: np.ndarray
    ) -> np.ndarray:
        """Compute roll angle deltas (simple angle difference in radians).

        Args:
            curr_ori: Current roll angles (N, 1)
            next_ori: Next roll angles (N, 1)

        Returns:
            Roll deltas (N, 1)
        """
        return next_ori - curr_ori  # type: ignore[no-any-return]


    def _compute_euler_deltas(
        self, curr_ori: np.ndarray, next_ori: np.ndarray
    ) -> np.ndarray:
        """Compute euler angle deltas."""
        curr_rot = scipy.spatial.transform.Rotation.from_euler("xyz", curr_ori)
        next_rot = scipy.spatial.transform.Rotation.from_euler("xyz", next_ori)
        rel_rot = next_rot * curr_rot.inv()
        return rel_rot.as_euler("xyz")  # type: ignore[no-any-return]


    def apply_position_denoising(
        self, next_pos: np.ndarray, curr_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply denoising threshold to position data.

        Raises:
            RuntimeError: If denoising thresholds haven't been computed.
        """
        if self.requires_denoising_setup:
            raise RuntimeError(
                "Denoising is enabled but thresholds have not been computed. "
                "Call compute_denoising_thresholds() with all training data first."
            )

        if self.action_denoising_threshold > 0:
            diffs = next_pos - curr_pos
            norms = np.linalg.norm(diffs, axis=1)
            mask = norms < self.action_denoising_threshold
            next_pos[mask] = curr_pos[mask]

        return next_pos, curr_pos


    def compute_action_denoising_threshold(
        self, all_next_pos: np.ndarray, all_curr_pos: np.ndarray
    ) -> None:
        """Compute and store the action denoising threshold from all training data.

        This should be called once during dataset initialization with all training samples.

        Args:
            all_next_pos: All next positions from training data (N, position_dim)
            all_curr_pos: All current positions from training data (N, position_dim)
        """
        diffs = all_next_pos - all_curr_pos
        norms = np.linalg.norm(diffs, axis=1)
        self._position_norms = norms
        non_zero_norms = norms[norms > 0]

        logging.info(
            f"Raw delta position action stats: "
            f"mean={norms.mean():.6f}, std={norms.std():.6f}, "
            f"min={norms.min():.6f}, max={norms.max():.6f}, "
            f"p5={np.percentile(norms, 5):.6f}, p50={np.percentile(norms, 50):.6f}, p95={np.percentile(norms, 95):.6f}"
        )

        if len(non_zero_norms) > 0:
            self.action_denoising_threshold = np.percentile(non_zero_norms, self.denoising_percentile)
            num_below_threshold = np.sum(norms < self.action_denoising_threshold)
            pct_below_threshold = 100.0 * num_below_threshold / len(norms)
            logging.info(
                f"Computed delta positional action threshold ({self.denoising_percentile}th percentile): "
                f"{self.action_denoising_threshold:.6f}. "
                f"{num_below_threshold}/{len(norms)} ({pct_below_threshold:.1f}%) delta actions will be zeroed."
            )
            denoised_norms = np.where(norms < self.action_denoising_threshold, 0.0, norms)
            logging.info(
                f"Delta position action stats after denoising: "
                f"mean={denoised_norms.mean():.6f}, std={denoised_norms.std():.6f}, "
                f"min={denoised_norms.min():.6f}, max={denoised_norms.max():.6f}, "
                f"p5={np.percentile(denoised_norms, 5):.6f}, p50={np.percentile(denoised_norms, 50):.6f}, p95={np.percentile(denoised_norms, 95):.6f}"
            )
        else:
            self.action_denoising_threshold = 0.0


    def apply_orientation_denoising(
        self, next_ori: np.ndarray, curr_ori: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply denoising threshold to orientation data.

        Raises:
            RuntimeError: If denoising thresholds haven't been computed.
        """
        if self.requires_denoising_setup:
            raise RuntimeError(
                "Denoising is enabled but thresholds have not been computed. "
                "Call compute_denoising_thresholds() with all training data first."
            )

        if self.orientation_denoising_threshold > 0:
            angles = self._compute_orientation_magnitudes(curr_ori, next_ori)
            mask = angles < self.orientation_denoising_threshold
            next_ori[mask] = curr_ori[mask]

        return next_ori, curr_ori

    def compute_orientation_denoising_threshold(
        self, all_next_ori: np.ndarray, all_curr_ori: np.ndarray
    ) -> None:
        """Compute and store the orientation denoising threshold from all training data.

        This should be called once during dataset initialization with all training samples.

        Args:
            all_next_ori: All next orientations from training data (N, orientation_dim)
            all_curr_ori: All current orientations from training data (N, orientation_dim)
        """
        angles = self._compute_orientation_magnitudes(all_curr_ori, all_next_ori)
        self._orientation_angles = angles
        non_zero_angles = angles[angles > 0]

        logging.info(
            f"Raw orientation action stats: "
            f"mean={angles.mean():.6f}, std={angles.std():.6f}, "
            f"min={angles.min():.6f}, max={angles.max():.6f}, "
            f"p5={np.percentile(angles, 5):.6f}, p50={np.percentile(angles, 50):.6f}, p95={np.percentile(angles, 95):.6f}"
        )

        if len(non_zero_angles) > 0:
            self.orientation_denoising_threshold = np.percentile(non_zero_angles, self.denoising_percentile)
            num_below_threshold = np.sum(angles < self.orientation_denoising_threshold)
            pct_below_threshold = 100.0 * num_below_threshold / len(angles)
            logging.info(
                f"Computed orientation threshold ({self.denoising_percentile}th percentile): "
                f"{self.orientation_denoising_threshold:.6f}. "
                f"{num_below_threshold}/{len(angles)} ({pct_below_threshold:.1f}%) actions will be zeroed."
            )
            denoised_angles = np.where(angles < self.orientation_denoising_threshold, 0.0, angles)
            logging.info(
                f"Orientation action stats after denoising: "
                f"mean={denoised_angles.mean():.6f}, std={denoised_angles.std():.6f}, "
                f"min={denoised_angles.min():.6f}, max={denoised_angles.max():.6f}, "
                f"p5={np.percentile(denoised_angles, 5):.6f}, p50={np.percentile(denoised_angles, 50):.6f}, p95={np.percentile(denoised_angles, 95):.6f}"
            )
        else:
            self.orientation_denoising_threshold = 0.0


    def _compute_orientation_magnitudes(
            self, curr_ori: np.ndarray, next_ori: np.ndarray
    ) -> np.ndarray:
        """Compute angular distances between orientations."""
        ori_repr = self.action_space.orientation_repr
        if ori_repr == OrientationRepresentation.ROLL.value:
            # Roll only representation, simply subtract values
            rel_rotation = np.abs(next_ori[:, 0] - curr_ori[:, 0])
        elif ori_repr == OrientationRepresentation.QUATERNION.value:
            quat_order = [1, 2, 3, 0]  # to (action_embedding,y,z,w)
            curr_rot = scipy.spatial.transform.Rotation.from_quat(
                curr_ori[:, quat_order]
            )
            next_rot = scipy.spatial.transform.Rotation.from_quat(
                next_ori[:, quat_order]
            )
            rel_rot = next_rot * curr_rot.inv()
            rel_rotation = rel_rot.magnitude()
        elif ori_repr == OrientationRepresentation.EULER.value:
            curr_rot = scipy.spatial.transform.Rotation.from_euler("xyz", curr_ori)
            next_rot = scipy.spatial.transform.Rotation.from_euler("xyz", next_ori)
            rel_rot = next_rot * curr_rot.inv()
            rel_rotation = rel_rot.magnitude()
        else:
            raise ValueError(f"Unsupported orientation representation: {ori_repr}")
        return rel_rotation  # type: ignore[no-any-return]

    def rotate_actions(
        self, action_dict: dict[str, np.ndarray], R: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Rotate actions by rotation matrix R (for augmentation).

        Args:
            action_dict: Dictionary of action arrays
            R: 3x3 rotation matrix

        Returns:
            Dictionary with rotated actions
        """
        rotated = {}

        if POSITION_ACTION_KEY in action_dict:
            rotated[POSITION_ACTION_KEY] = (
                R @ action_dict[POSITION_ACTION_KEY].T
            ).T

        if ORIENTATION_ACTION_KEY in action_dict:
            rotated[ORIENTATION_ACTION_KEY] = self._rotate_orientations(
                action_dict[ORIENTATION_ACTION_KEY], R
            )

        if GRIPPER_ACTION_KEY in action_dict:
            rotated[GRIPPER_ACTION_KEY] = action_dict[GRIPPER_ACTION_KEY]

        return rotated


    def _rotate_orientations(
            self, orientations: np.ndarray, R: np.ndarray
    ) -> np.ndarray:
        """Rotate orientations by rotation matrix R."""
        ori_repr = self.action_space.orientation_repr
        R_rot = scipy.spatial.transform.Rotation.from_matrix(R)
        if ori_repr == OrientationRepresentation.QUATERNION.value:
            quat_order_from = [1, 2, 3, 0]  # to (action_embedding,y,z,w)
            rot = scipy.spatial.transform.Rotation.from_quat(orientations[:, quat_order_from])
            if self.deltas_as_actions:
                rotated = R_rot * rot * R_rot.inv()
            else:
                rotated = R_rot * rot
            quat_order_to = [3, 0, 1, 2]  # back to (w,action_embedding,y,z)
            return rotated.as_quat()[:, quat_order_to]  # type: ignore[no-any-return]
        elif ori_repr == OrientationRepresentation.EULER.value:
            rot = scipy.spatial.transform.Rotation.from_euler("xyz", orientations)
            if self.deltas_as_actions:
                rotated = R_rot * rot * R_rot.inv()
            else:
                rotated = R_rot * rot
            return rotated.as_euler("xyz")  # type: ignore[no-any-return]
        elif ori_repr == OrientationRepresentation.ROLL.value:
            if self.deltas_as_actions:
                return orientations  # Roll deltas not rotated
            else:
                # Extract Z-axis rotation from R and add to roll
                euler_angles = R_rot.as_euler("xyz")
                z_rotation = euler_angles[2]  # Roll around Z-axis
                return orientations + z_rotation  # type: ignore[no-any-return]
        else:
            raise ValueError(f"Unsupported orientation representation: {ori_repr}")


    def plot_action_delta_distribution(self, output_path: str) -> None:
        """Plot position and orientation delta distributions before/after denoising.

        Args:
            output_path: Path to save the plot.
        """
        has_pos = self._position_norms is not None
        has_ori = self._orientation_angles is not None

        if not has_pos and not has_ori:
            logging.warning("No denoising data available to plot")
            return

        sns.set_theme(style="whitegrid", palette="muted")
        num_plots = int(has_pos) + int(has_ori)
        fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5))
        if num_plots == 1:
            axes = [axes]

        plot_idx = 0

        if has_pos:
            ax = axes[plot_idx]
            norms = self._position_norms
            threshold = self.action_denoising_threshold

            sns.histplot(norms, bins=100, alpha=0.6, label="Raw", color="steelblue", ax=ax, log_scale=(False, True))
            if threshold > 0:
                denoised = np.where(norms < threshold, 0.0, norms)
                sns.histplot(denoised[denoised > 0], bins=100, alpha=0.6, label="Denoised", color="coral", ax=ax, log_scale=(False, True))
                ax.axvline(threshold, color="crimson", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.4f}")

            ax.set_xlabel("Position Delta Norm", fontsize=11)
            ax.set_ylabel("Count (log)", fontsize=11)
            ax.set_title("Position Delta Distribution", fontsize=13, fontweight="bold")
            ax.legend(frameon=True, fancybox=True)
            plot_idx += 1

        if has_ori:
            ax = axes[plot_idx]
            angles = self._orientation_angles
            threshold = self.orientation_denoising_threshold

            sns.histplot(angles, bins=100, alpha=0.6, label="Raw", color="steelblue", ax=ax, log_scale=(False, True))
            if threshold > 0:
                denoised = np.where(angles < threshold, 0.0, angles)
                sns.histplot(denoised[denoised > 0], bins=100, alpha=0.6, label="Denoised", color="coral", ax=ax, log_scale=(False, True))
                ax.axvline(threshold, color="crimson", linestyle="--", linewidth=2, label=f"Threshold: {threshold:.4f}")

            ax.set_xlabel("Orientation Delta (rad)", fontsize=11)
            ax.set_ylabel("Count (log)", fontsize=11)
            ax.set_title("Orientation Delta Distribution", fontsize=13, fontweight="bold")
            ax.legend(frameon=True, fancybox=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        sns.reset_defaults()
        logging.info(f"Saved denoising distribution plot to {output_path}")

