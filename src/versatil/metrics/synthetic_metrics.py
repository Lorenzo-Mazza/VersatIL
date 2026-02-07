"""Evaluation metrics for synthetic multimodal benchmark tasks."""

import numpy as np


def compute_mode_coverage(
    generated_trajectories: np.ndarray,
    expert_trajectories: np.ndarray,
    expert_mode_ids: np.ndarray,
    num_modes: int,
) -> dict[str, float | dict[int, int]]:
    """Measure how many expert modes the generated trajectories cover.

    Each generated trajectory is assigned to the nearest expert mode by
    mean L2 distance over the full trajectory. Mode coverage is the
    fraction of modes that receive at least one assignment.

    Args:
        generated_trajectories: Predicted Cartesian trajectories (x, y).
            Shape (num_rollouts, num_timesteps, 2).
        expert_trajectories: Expert demonstration trajectories (x, y).
            Shape (num_expert, num_timesteps, 2).
        expert_mode_ids: Ground-truth mode label per expert trajectory.
            Shape (num_expert,).
        num_modes: Total number of distinct behavioral modes.

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
    per_mode_count = {mode_index: 0 for mode_index in range(num_modes)}
    for trajectory in generated_trajectories:
        distances = np.array([
            np.mean(np.linalg.norm(trajectory - centroid, axis=-1))
            for centroid in mode_centroids
        ])
        assigned_mode = int(np.argmin(distances))
        per_mode_count[assigned_mode] += 1
    covered_modes = sum(1 for count in per_mode_count.values() if count > 0)
    mode_coverage = covered_modes / num_modes
    total_assignments = sum(per_mode_count.values())
    if total_assignments > 0:
        probabilities = np.array([
            per_mode_count[mode_index] / total_assignments
            for mode_index in range(num_modes)
        ])
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