"""Rollout and evaluation utilities for synthetic benchmark policies.

Provides open-loop and closed-loop rollout functions that load a trained
policy checkpoint, generate position trajectories via the policy, and
evaluate them against expert demonstrations using mode coverage and goal
success metrics.
"""

import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from versatil.configs import MainConfig
from versatil.data.constants import Cameras, ProprioKey, SyntheticObsKey
from versatil.data.synthetic.constants import (
    CIRCLE_CONTEXT_COLORS,
    CORRIDOR_DEFAULT_NUM_STYLES,
    DEFAULT_IMAGE_SIZE,
    MULTIPATH_DEFAULT_NOISE_STD,
    MULTIPATH_DEFAULT_NUM_MODES,
    MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
)
from versatil.data.synthetic.generators import generate_task_episodes
from versatil.data.synthetic.renderer import render_frame
from versatil.data.synthetic.task_layout import get_task_layout
from versatil.data.synthetic.visualization import (
    plot_trajectories_2d,
    save_rollouts_gif,
)
from versatil.metrics.synthetic_metrics import (
    compute_mode_coverage,
    compute_mode_endpoints,
    compute_success_masks,
    compute_success_rates_from_masks,
)
from versatil.models.policy import Policy
from versatil.training.lightning_policy import LightningPolicy


def load_policy_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
    checkpoint_name: str = "last.ckpt",
) -> tuple[Policy, MainConfig]:
    """Load a trained policy and config from a checkpoint directory.

    Follows the same loading pattern as the production inference clients:
    instantiate config, create policy, load weights via LightningPolicy.

    Args:
        checkpoint_path: Directory containing config.yaml and the .ckpt file.
        device: Torch device string.
        checkpoint_name: Checkpoint filename inside checkpoint_path.

    Returns:
        Tuple of (policy in eval mode, resolved MainConfig).
    """
    config_file = os.path.join(checkpoint_path, "config.yaml")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config not found at {config_file}")
    config: MainConfig = hydra.utils.instantiate(OmegaConf.load(config_file))
    policy: Policy = config.policy
    policy.to(device).eval()
    checkpoint_file = os.path.join(checkpoint_path, checkpoint_name)
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    lightning_module = LightningPolicy(policy=policy, training_config=config.training)
    lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)
    return policy, config


def run_rollouts(
    policy: Policy,
    task_name: str,
    num_rollouts: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    context_mode: int | None = None,
    temporal_aggregation: bool = False,
    exponential_decay: float = 0.01,
    output_dir: str | None = None,
) -> np.ndarray:
    """Re-render and re-predict at each timestep with temporal aggregation.

    At each step, the policy predicts a full action trajectory. With
    temporal aggregation enabled, overlapping predictions for the current
    timestep are averaged with exponential weights favoring more recent
    queries. Without it, only the first predicted action is used.

    Policy-input frames intentionally render without any goal marker.
    Per-mode expert goals are still drawn in the user-facing rollout
    visualization via ``layout.goals``.

    Args:
        policy: Trained policy in eval mode.
        task_name: SyntheticTaskName.value string.
        num_rollouts: Number of independent rollouts.
        image_size: Side length for rendered images.
        context_mode: Context mode index for conditional tasks (None to omit).
        temporal_aggregation: Average overlapping action predictions with
            exponential weighting. Matches the production inference clients.
        exponential_decay: Decay factor for temporal aggregation weights.
            Smaller values produce more uniform weighting across queries.
        output_dir: Optional directory for saving PNG + GIF visualizations
            of the rollout trajectories. If None, no visualizations are saved.

    Returns:
        Position trajectories, shape (num_rollouts, prediction_horizon + 1, 2).
    """
    layout = get_task_layout(task_name=task_name)
    start = layout.start
    obstacles = layout.obstacles
    num_modes = layout.num_modes
    prediction_horizon = policy.prediction_horizon
    observation_keys = set(policy.observation_space.observations_metadata.keys())
    action_key = ProprioKey.SYNTHETIC_POSITION_ACTION.value
    context_vector = None
    context_color = None
    if context_mode is not None:
        context_vector = np.zeros(num_modes, dtype=np.float32)
        context_vector[context_mode] = 1.0
        context_color = CIRCLE_CONTEXT_COLORS.get(context_mode)

    obs_horizon = policy.observation_horizon
    all_trajectories = np.zeros(
        (num_rollouts, prediction_horizon + 1, 2), dtype=np.float32
    )

    for rollout_index in range(num_rollouts):
        positions = all_trajectories[rollout_index]
        positions[0] = start.copy()

        if temporal_aggregation:
            action_buffer = np.zeros(
                (prediction_horizon, prediction_horizon, 2), dtype=np.float32
            )
            populated_mask = np.zeros(
                (prediction_horizon, prediction_horizon), dtype=bool
            )
            for step in range(prediction_horizon):
                trail = positions[: step + 1]
                available_history = step + 1
                if available_history < obs_horizon:
                    positions[step + 1] = positions[step].copy()
                    continue
                history_start = step + 1 - obs_horizon
                history = positions[history_start : step + 1]
                observation = _prepare_observation(
                    position_history=history,
                    obstacles=obstacles,
                    image_size=image_size,
                    observation_keys=observation_keys,
                    trail=trail,
                    context_vector=context_vector,
                    context_color=context_color,
                )
                with torch.no_grad():
                    actions = policy.predict_action(obs_dict=observation)
                action_deltas = actions[action_key].squeeze(0).cpu().numpy()
                remaining = min(prediction_horizon - step, prediction_horizon)
                action_buffer[step, step : step + remaining] = action_deltas[:remaining]
                populated_mask[step, step : step + remaining] = True
                valid_queries = populated_mask[:, step]
                candidate_actions = action_buffer[valid_queries, step]
                num_candidates = len(candidate_actions)
                indices = np.arange(num_candidates)[::-1]
                weights = np.exp(-exponential_decay * indices)
                weights = weights / weights.sum()
                selected_action = (candidate_actions * weights[:, np.newaxis]).sum(
                    axis=0
                )
                positions[step + 1] = np.clip(
                    positions[step] + selected_action, 0.0, 1.0
                )
        else:
            # Execute full chunk without replanning
            step = 0
            while step < prediction_horizon:
                available_history = step + 1
                if available_history < obs_horizon:
                    positions[step + 1] = positions[step].copy()
                    step += 1
                    continue
                trail = positions[: step + 1]
                history_start = step + 1 - obs_horizon
                history = positions[history_start : step + 1]
                observation = _prepare_observation(
                    position_history=history,
                    obstacles=obstacles,
                    image_size=image_size,
                    observation_keys=observation_keys,
                    trail=trail,
                    context_vector=context_vector,
                    context_color=context_color,
                )
                with torch.no_grad():
                    actions = policy.predict_action(obs_dict=observation)
                action_deltas = actions[action_key].squeeze(0).cpu().numpy()
                # Execute all remaining actions in this chunk
                remaining = min(prediction_horizon - step, prediction_horizon)
                for chunk_offset in range(remaining):
                    positions[step + 1] = np.clip(
                        positions[step] + action_deltas[chunk_offset], 0.0, 1.0
                    )
                    step += 1

    if output_dir is not None:
        name_prefix = "rollout_temporal_agg" if temporal_aggregation else "rollout"
        _save_rollout_visualizations(
            trajectories=all_trajectories,
            task_name=task_name,
            output_dir=output_dir,
            name_prefix=name_prefix,
            image_size=image_size,
        )
    return all_trajectories


def _save_rollout_visualizations(
    trajectories: np.ndarray,
    task_name: str,
    output_dir: str,
    name_prefix: str,
    image_size: int,
) -> None:
    """Save rollout trajectories as a static PNG and an animated GIF.

    Args:
        trajectories: Rollout trajectories, shape (num_rollouts,
            num_timesteps, 2), values in [0, 1].
        task_name: SyntheticTaskName.value string.
        output_dir: Destination directory (created if missing).
        name_prefix: Filename prefix identifying the rollout type, e.g.
            ``"open_loop"`` or ``"closed_loop_temporal_agg"``.
        image_size: Side length of each GIF frame in pixels.
    """
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    png_path = output_dir_path / f"{name_prefix}_{task_name}.png"
    gif_path = output_dir_path / f"{name_prefix}_{task_name}.gif"
    figure = plot_trajectories_2d(
        trajectories=trajectories,
        task_name=task_name,
        output_path=str(png_path),
    )
    plt.close(figure)
    save_rollouts_gif(
        trajectories=trajectories,
        task_name=task_name,
        output_path=str(gif_path),
        image_size=image_size,
    )


def evaluate_rollouts(
    rollout_trajectories: np.ndarray,
    task_name: str,
    num_expert_episodes: int = 1000,
    expert_seed: int = 42,
    trajectory_length: int = MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
    noise_std: float = MULTIPATH_DEFAULT_NOISE_STD,
    num_modes: int = MULTIPATH_DEFAULT_NUM_MODES,
    num_styles: int = CORRIDOR_DEFAULT_NUM_STYLES,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> dict[str, float | dict[int, int]]:
    """Evaluate rollout trajectories against regenerated expert demonstrations.

    Generates expert data with the given parameters, then computes
    mode coverage and obstacle-aware success rate. The "reach" threshold
    used by the synthetic success metrics is derived from the expert endpoint
    spread (mean plus five standard deviations of distances from the per-mode
    endpoint mean), so it scales with the task's intrinsic noise.

    Args:
        rollout_trajectories: Generated position trajectories.
            Shape (num_rollouts, num_timesteps, 2).
        task_name: SyntheticTaskName.value string.
        num_expert_episodes: Number of expert episodes for reference.
        expert_seed: Seed for reproducible expert generation.
        trajectory_length: Expert trajectory length.
        noise_std: Expert noise standard deviation.
        num_modes: Number of modes for expert generation.
        num_styles: Number of styles (trajectory_style task only).
        image_size: Image size for expert generation.

    Returns:
        Dict with raw mode coverage metrics, valid mode coverage metrics,
        success_rate, collision_rate, endpoint_reach_rate, and path_length_rate.
    """
    expert_episodes = generate_task_episodes(
        task_name=task_name,
        num_episodes=num_expert_episodes,
        seed=expert_seed,
        image_size=image_size,
        num_modes=num_modes,
        trajectory_length=trajectory_length,
        noise_std=noise_std,
        num_styles=num_styles,
    )

    expert_trajectories = np.array([episode["position"] for episode in expert_episodes])
    expert_mode_ids = np.array(
        [int(episode["mode_id"][0, 0]) for episode in expert_episodes]
    )
    layout = get_task_layout(
        task_name=task_name,
        num_modes=num_modes,
        num_styles=num_styles,
        noise_std=noise_std,
    )
    rollout_length = rollout_trajectories.shape[1]
    expert_length = expert_trajectories.shape[1]
    comparison_length = min(rollout_length, expert_length)
    rollout_comparison_trajectories = rollout_trajectories[:, :comparison_length, :]
    expert_comparison_trajectories = expert_trajectories[:, :comparison_length, :]
    mode_endpoints = compute_mode_endpoints(
        expert_trajectories=expert_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=layout.num_modes,
    )
    reach_threshold = _expert_endpoint_reach_threshold(
        expert_trajectories=expert_trajectories,
        expert_mode_ids=expert_mode_ids,
        mode_endpoints=mode_endpoints,
    )
    min_path_length = 0.5 * _expert_mean_path_length(
        expert_trajectories=expert_trajectories
    )
    success_masks = compute_success_masks(
        generated_trajectories=rollout_trajectories,
        obstacles=layout.obstacles,
        mode_endpoints=mode_endpoints,
        goal_threshold=reach_threshold,
        min_path_length=min_path_length,
    )
    collision_free_mask = ~success_masks["collision_mask"]
    coverage_results = compute_mode_coverage(
        generated_trajectories=rollout_comparison_trajectories,
        expert_trajectories=expert_comparison_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=layout.num_modes,
        valid_mask=collision_free_mask,
    )
    valid_coverage_results = compute_mode_coverage(
        generated_trajectories=rollout_comparison_trajectories,
        expert_trajectories=expert_comparison_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=layout.num_modes,
        valid_mask=success_masks["success_mask"],
    )
    success_stats = compute_success_rates_from_masks(success_masks=success_masks)
    results = dict(coverage_results)
    results["valid_mode_coverage"] = valid_coverage_results["mode_coverage"]
    results["valid_mode_entropy_ratio"] = valid_coverage_results["mode_entropy_ratio"]
    results.update(success_stats)
    return results


def _expert_mean_path_length(expert_trajectories: np.ndarray) -> float:
    """Mean cumulative Euclidean path length across expert trajectories."""
    step_lengths = np.linalg.norm(
        np.diff(expert_trajectories, axis=1), axis=-1
    )  # (num_expert, num_timesteps-1)
    per_trajectory = step_lengths.sum(axis=-1)  # (num_expert,)
    return float(per_trajectory.mean())


def _expert_endpoint_reach_threshold(
    expert_trajectories: np.ndarray,
    expert_mode_ids: np.ndarray,
    mode_endpoints: np.ndarray,
    min_threshold: float = 0.1,
) -> float:
    """Expert-derived "close enough" radius around each mode endpoint.

    Uses mean + 5·std of expert final-position distances to their own
    mode mean, floored at ``min_threshold``. The floor prevents overly strict
    thresholds on closed-loop tasks where all endpoints cluster at the
    start.
    """
    final_positions = expert_trajectories[:, -1, :]  # (num_expert, 2)
    distances = np.linalg.norm(
        final_positions - mode_endpoints[expert_mode_ids], axis=-1
    )  # (num_expert,)
    return float(max(distances.mean() + 5.0 * distances.std(), min_threshold))


def _prepare_observation(
    position_history: np.ndarray,
    obstacles: list[tuple[float, float, float, float]],
    image_size: int,
    observation_keys: set[str],
    trail: np.ndarray | None = None,
    context_vector: np.ndarray | None = None,
    context_color: tuple[int, int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Build an observation dict suitable for Policy.predict_action().

    Renders one frame per history timestep with progressive trails, and
    stacks all modalities along the temporal dimension to match the
    (B, T, ...) convention used during training.
    Args:
        position_history: Last obs_horizon Cartesian positions (x, y)
            in [0, 1]. Shape (obs_horizon, 2).
        obstacles: Obstacle rectangles for rendering.
        image_size: Side length of rendered square images.
        observation_keys: Set of keys the policy's observation space requires.
        trail: Full trail up to current step for rendering. Shape (N, 2)
            or None. Each history frame renders the trail up to its timestep.
        context_vector: One-hot context for conditional tasks, or None.

    Returns:
        Dict mapping observation keys to batched torch tensors
        with shape (B=1, T=obs_horizon, ...).
    """
    observation = {}
    obs_horizon = len(position_history)

    image_key = Cameras.AGENTVIEW.value
    if image_key in observation_keys:
        frames = []
        for timestep_index in range(obs_horizon):
            # Each history frame gets the trail up to that timestep
            if trail is not None:
                trail_end = len(trail) - obs_horizon + timestep_index + 1
                frame_trail = trail[:trail_end] if trail_end > 1 else None
            else:
                frame_trail = None
            frame = render_frame(
                position=position_history[timestep_index],
                obstacles=obstacles,
                image_size=image_size,
                trail=frame_trail,
                context_color=context_color,
            )
            # uint8 (H, W, C) -> float32 (C, H, W)
            frame_tensor = torch.from_numpy(frame).float() / 255.0
            frames.append(frame_tensor.permute(2, 0, 1))
        # (T, C, H, W) -> (B=1, T, C, H, W)
        observation[image_key] = torch.stack(frames).unsqueeze(0)

    position_key = ProprioKey.SYNTHETIC_POSITION.value
    if position_key in observation_keys:
        # (T, D) -> (B=1, T, D)
        observation[position_key] = (
            torch.from_numpy(position_history.copy()).float().unsqueeze(0)
        )
    context_key = SyntheticObsKey.CONTEXT.value
    if context_key in observation_keys and context_vector is not None:
        # (C,) -> (B=1, T, C) by tiling across obs_horizon
        context_tiled = np.tile(context_vector, (obs_horizon, 1))
        observation[context_key] = torch.from_numpy(context_tiled).float().unsqueeze(0)

    return observation
