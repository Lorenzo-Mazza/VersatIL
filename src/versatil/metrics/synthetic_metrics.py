"""Evaluation metrics for synthetic multimodal benchmark tasks."""

import numpy as np


def compute_mode_coverage(
    generated_trajectories: np.ndarray,
    expert_trajectories: np.ndarray,
    expert_mode_ids: np.ndarray,
    num_modes: int,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float | dict[int, int]]:
    """Measure how many expert modes the generated trajectories cover.

    Each valid generated trajectory is assigned to the nearest expert
    mode by mean L2 distance over the full trajectory. Invalid
    trajectories (e.g. collided with an obstacle) are excluded entirely
    from mode assignment, coverage, and entropy statistics.

    Args:
        generated_trajectories: Predicted Cartesian trajectories (x, y).
            Shape (num_rollouts, num_timesteps, 2).
        expert_trajectories: Expert demonstration trajectories (x, y).
            Shape (num_expert, num_timesteps, 2).
        expert_mode_ids: Ground-truth mode label per expert trajectory.
            Shape (num_expert,).
        num_modes: Total number of distinct behavioral modes.
        valid_mask: Optional boolean mask of shape (num_rollouts,).
            False entries are skipped. Defaults to all-True.

    Returns:
        Dictionary with:
            "mode_coverage": fraction of modes hit (0.0 to 1.0).
            "mode_entropy_ratio": normalized Shannon entropy of the mode
                assignment distribution (1.0 = uniform, 0.0 = single mode).
            "per_mode_count": dict mapping mode index to assignment count.
    """
    mode_centroids = _compute_mode_centroids(
        expert_trajectories=expert_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=num_modes,
    )
    per_mode_count = dict.fromkeys(range(num_modes), 0)
    for rollout_index, trajectory in enumerate(generated_trajectories):
        if valid_mask is not None and not valid_mask[rollout_index]:
            continue
        distances = np.array(
            [
                np.mean(np.linalg.norm(trajectory - centroid, axis=-1))
                for centroid in mode_centroids
            ]
        )
        assigned_mode = int(np.argmin(distances))
        per_mode_count[assigned_mode] += 1
    covered_modes = sum(1 for count in per_mode_count.values() if count > 0)
    mode_coverage = covered_modes / num_modes
    total_assignments = sum(per_mode_count.values())
    if total_assignments > 0:
        probabilities = np.array(
            [
                per_mode_count[mode_index] / total_assignments
                for mode_index in range(num_modes)
            ]
        )
        nonzero_probabilities = probabilities[probabilities > 0]
        entropy = -np.sum(nonzero_probabilities * np.log(nonzero_probabilities))
        max_entropy = np.log(num_modes)
        mode_entropy_ratio = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        mode_entropy_ratio = 0.0

    return {
        "mode_coverage": float(mode_coverage),
        "mode_entropy_ratio": mode_entropy_ratio,
        "per_mode_count": per_mode_count,
    }


def compute_goal_success_rate(
    generated_trajectories: np.ndarray,
    goal: np.ndarray,
    threshold: float = 0.05,
) -> float:
    """Compute fraction of trajectories whose final position is within threshold of the goal.

    Args:
        generated_trajectories: Predicted Cartesian trajectories (x, y).
            Shape (num_rollouts, num_timesteps, 2).
        goal: Goal Cartesian position (x, y). Shape (2,).
        threshold: Euclidean distance threshold for success.

    Returns:
        Success rate in [0.0, 1.0].
    """
    final_positions = generated_trajectories[:, -1, :]
    distances = np.linalg.norm(final_positions - goal, axis=-1)
    return float(np.mean(distances < threshold))


def compute_success_rate(
    generated_trajectories: np.ndarray,
    obstacles: list[tuple[float, float, float, float]],
    mode_endpoints: np.ndarray,
    goal_threshold: float = 0.1,
    min_path_length: float = 0.0,
) -> dict[str, float]:
    """Fraction of trajectories that avoid obstacles, actually move, and reach an expert endpoint.

    A trajectory is counted successful only when all three conditions hold:
        1. No trajectory point lies inside any obstacle rectangle.
        2. Cumulative path length >= ``min_path_length`` — rejects
           stationary policies that trivially satisfy the endpoint check
           on closed-loop tasks (circle) where start == endpoint.
        3. Its final position is within ``goal_threshold`` of at least one
           expert endpoint (per-mode mean of the final step).

    Args:
        generated_trajectories: Predicted Cartesian trajectories (x, y).
            Shape (num_rollouts, num_timesteps, 2).
        obstacles: List of (x_min, y_min, x_max, y_max) rectangles. Empty
            list disables the collision check.
        mode_endpoints: Expert endpoint per mode, shape (num_modes, 2).
        goal_threshold: Euclidean distance threshold for reaching an endpoint.
        min_path_length: Minimum cumulative path length a trajectory must
            travel to count as a real attempt. 0.0 disables the check.

    Returns:
        Dictionary with:
            "success_rate": overall fraction satisfying all conditions.
            "collision_rate": fraction that collided with an obstacle.
            "endpoint_reach_rate": fraction that reached any endpoint.
            "path_length_rate": fraction with path length >= min_path_length.
    """
    success_masks = compute_success_masks(
        generated_trajectories=generated_trajectories,
        obstacles=obstacles,
        mode_endpoints=mode_endpoints,
        goal_threshold=goal_threshold,
        min_path_length=min_path_length,
    )
    return compute_success_rates_from_masks(success_masks=success_masks)


def compute_success_masks(
    generated_trajectories: np.ndarray,
    obstacles: list[tuple[float, float, float, float]],
    mode_endpoints: np.ndarray,
    goal_threshold: float = 0.1,
    min_path_length: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compute per-rollout masks for synthetic success conditions.

    Args:
        generated_trajectories: Predicted Cartesian trajectories (x, y).
            Shape (num_rollouts, num_timesteps, 2).
        obstacles: List of (x_min, y_min, x_max, y_max) rectangles. Empty
            list disables the collision check.
        mode_endpoints: Expert endpoint per mode, shape (num_modes, 2).
        goal_threshold: Euclidean distance threshold for reaching an endpoint.
        min_path_length: Minimum cumulative path length a trajectory must
            travel to count as a real attempt. 0.0 disables the check.

    Returns:
        Dictionary with:
            "collision_mask": True where the rollout enters an obstacle.
            "endpoint_reach_mask": True where the rollout reaches any endpoint.
            "path_length_mask": True where path length >= min_path_length.
            "success_mask": True where all success conditions hold.
    """
    collision_mask = collides_with_obstacles(
        trajectories=generated_trajectories, obstacles=obstacles
    )
    final_positions = generated_trajectories[:, -1, :]  # (num_rollouts, 2)
    distances = np.linalg.norm(
        final_positions[:, None, :] - mode_endpoints[None, :, :], axis=-1
    )  # (num_rollouts, num_modes)
    reach_mask = (distances < goal_threshold).any(axis=-1)  # (num_rollouts,)
    step_lengths = np.linalg.norm(
        np.diff(generated_trajectories, axis=1), axis=-1
    )  # (num_rollouts, num_timesteps-1)
    path_lengths = step_lengths.sum(axis=-1)  # (num_rollouts,)
    path_length_mask = path_lengths >= min_path_length
    success_mask = reach_mask & ~collision_mask & path_length_mask
    return {
        "collision_mask": collision_mask,
        "endpoint_reach_mask": reach_mask,
        "path_length_mask": path_length_mask,
        "success_mask": success_mask,
    }


def compute_success_rates_from_masks(
    success_masks: dict[str, np.ndarray],
) -> dict[str, float]:
    """Aggregate synthetic success-condition masks into rollout rates.

    Args:
        success_masks: Dictionary produced by ``compute_success_masks``.

    Returns:
        Dictionary with:
            "success_rate": overall fraction satisfying all conditions.
            "collision_rate": fraction that collided with an obstacle.
            "endpoint_reach_rate": fraction that reached any endpoint.
            "path_length_rate": fraction with path length >= min_path_length.
    """
    success_mask = success_masks["success_mask"]
    collision_mask = success_masks["collision_mask"]
    reach_mask = success_masks["endpoint_reach_mask"]
    path_length_mask = success_masks["path_length_mask"]
    num_trajectories = success_mask.shape[0]
    collision_rate = float(np.mean(collision_mask)) if num_trajectories else 0.0
    reach_rate = float(np.mean(reach_mask)) if num_trajectories else 0.0
    path_length_rate = float(np.mean(path_length_mask)) if num_trajectories else 0.0
    success_rate = float(np.mean(success_mask)) if num_trajectories else 0.0
    return {
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "endpoint_reach_rate": reach_rate,
        "path_length_rate": path_length_rate,
    }


def collides_with_obstacles(
    trajectories: np.ndarray,
    obstacles: list[tuple[float, float, float, float]],
) -> np.ndarray:
    """Per-trajectory boolean mask for any-point-inside-any-obstacle.

    Args:
        trajectories: Cartesian (x, y) points, shape (num_rollouts, num_timesteps, 2).
        obstacles: List of (x_min, y_min, x_max, y_max) axis-aligned rectangles.

    Returns:
        Boolean array of shape (num_rollouts,); True where the trajectory
        enters at least one obstacle. All False when ``obstacles`` is empty.
    """
    num_trajectories = trajectories.shape[0]
    if not obstacles:
        return np.zeros(num_trajectories, dtype=bool)
    collided = np.zeros(num_trajectories, dtype=bool)
    for x_min, y_min, x_max, y_max in obstacles:
        inside_x = (trajectories[:, :, 0] >= x_min) & (trajectories[:, :, 0] <= x_max)
        inside_y = (trajectories[:, :, 1] >= y_min) & (trajectories[:, :, 1] <= y_max)
        inside = (inside_x & inside_y).any(axis=-1)  # (num_rollouts,)
        collided |= inside
    return collided


def compute_mode_endpoints(
    expert_trajectories: np.ndarray,
    expert_mode_ids: np.ndarray,
    num_modes: int,
) -> np.ndarray:
    """Compute mean final position per expert mode.

    Args:
        expert_trajectories: Expert trajectories, shape (num_expert, num_timesteps, 2).
        expert_mode_ids: Mode label per expert, shape (num_expert,).
        num_modes: Total number of modes.

    Returns:
        Array of shape (num_modes, 2) with the per-mode mean final position.

    Raises:
        ValueError: If any mode in [0, num_modes) has no expert trajectories.
    """
    endpoints = np.zeros((num_modes, 2), dtype=expert_trajectories.dtype)
    for mode_index in range(num_modes):
        mask = expert_mode_ids == mode_index
        if not np.any(mask):
            raise ValueError(
                f"No expert trajectories for mode {mode_index}; "
                f"expected all {num_modes} modes represented in expert data."
            )
        endpoints[mode_index] = expert_trajectories[mask, -1, :].mean(axis=0)
    return endpoints


def _compute_mode_centroids(
    expert_trajectories: np.ndarray,
    expert_mode_ids: np.ndarray,
    num_modes: int,
) -> list[np.ndarray]:
    """Compute mean trajectory per mode from expert demonstrations.

    Args:
        expert_trajectories: Expert trajectories, shape (num_expert, num_timesteps, 2).
        expert_mode_ids: Mode label per expert, shape (num_expert,).
        num_modes: Total number of modes.

    Returns:
        List of centroid trajectories, one per mode. Shape of each: (num_timesteps, 2).
    """
    centroids = []
    for mode_index in range(num_modes):
        mode_mask = expert_mode_ids == mode_index
        if np.any(mode_mask):
            centroids.append(np.mean(expert_trajectories[mode_mask], axis=0))
        else:
            centroids.append(np.zeros_like(expert_trajectories[0]))
    return centroids
