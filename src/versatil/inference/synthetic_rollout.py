"""Rollout and evaluation utilities for synthetic benchmark policies.

Provides open-loop and closed-loop rollout functions that load a trained
policy checkpoint, generate position trajectories via the policy, and
evaluate them against expert demonstrations using mode coverage and goal
success metrics.
"""

import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from versatil.configs import MainConfig
from versatil.data.constants import Cameras, ProprioKey, SyntheticObsKey
from versatil.data.synthetic.constants import (
    DEFAULT_IMAGE_SIZE,
    MULTIPATH_DEFAULT_NOISE_STD,
    MULTIPATH_DEFAULT_NUM_MODES,
    MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
    STYLE_DEFAULT_NUM_STYLES,
    SyntheticTaskName,
)
from versatil.data.synthetic.generators import generate_task_episodes
from versatil.data.synthetic.renderer import render_frame
from versatil.data.synthetic.task_layout import (
    SyntheticTaskLayout,
    get_task_layout,
)
from versatil.data.synthetic.visualization import (
    plot_trajectories_2d,
    save_rollouts_gif,
)
from versatil.metrics.synthetic_metrics import (
    compute_goal_success_rate,
    compute_mode_coverage,
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
    temporal_aggregation: bool = True,
    exponential_decay: float = 0.01,
    output_dir: str | None = None,
) -> np.ndarray:
    """Re-render and re-predict at each timestep with temporal aggregation.

    At each step, the policy predicts a full action trajectory. With
    temporal aggregation enabled, overlapping predictions for the current
    timestep are averaged with exponential weights favoring more recent
    queries. Without it, only the first predicted action is used.

    Note:
        For tasks with mode-dependent goals (sequential_decision,
        shared_prefix), the rendered goal is an approximate center
        since the true endpoint is unknown at inference time.

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
    render_goal = _get_render_goal(layout=layout, task_name=task_name)
    obstacles = layout.obstacles
    num_modes = layout.num_modes
    prediction_horizon = policy.prediction_horizon
    observation_keys = set(policy.observation_space.observations_metadata.keys())
    action_key = ProprioKey.SYNTHETIC_POSITION_ACTION.value
    context_vector = None
    if context_mode is not None:
        context_vector = np.zeros(num_modes, dtype=np.float32)
        context_vector[context_mode] = 1.0

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
                    goal=render_goal,
                    image_size=image_size,
                    observation_keys=observation_keys,
                    trail=trail,
                    context_vector=context_vector,
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
                    goal=render_goal,
                    image_size=image_size,
                    observation_keys=observation_keys,
                    trail=trail,
                    context_vector=context_vector,
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
    plot_trajectories_2d(
        trajectories=trajectories,
        task_name=task_name,
        output_path=str(png_path),
    )
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
    num_styles: int = STYLE_DEFAULT_NUM_STYLES,
    image_size: int = DEFAULT_IMAGE_SIZE,
    goal_threshold: float = 0.05,
) -> dict[str, float | dict[int, int]]:
    """Evaluate rollout trajectories against regenerated expert demonstrations.

    Generates expert data with the given parameters, then computes
    mode coverage and (when applicable) goal success rate.

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
        goal_threshold: Euclidean distance threshold for goal success.

    Returns:
        Dict with mode_coverage, mode_entropy_ratio, per_mode_count,
        and goal_success_rate (when the task has a fixed goal).
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
    layout = get_task_layout(task_name=task_name)
    rollout_length = rollout_trajectories.shape[1]
    expert_length = expert_trajectories.shape[1]
    comparison_length = min(rollout_length, expert_length)
    coverage_results = compute_mode_coverage(
        generated_trajectories=rollout_trajectories[:, :comparison_length, :],
        expert_trajectories=expert_trajectories[:, :comparison_length, :],
        expert_mode_ids=expert_mode_ids,
        num_modes=layout.num_modes,
    )
    results = dict(coverage_results)
    if layout.goal is not None:
        results["goal_success_rate"] = compute_goal_success_rate(
            generated_trajectories=rollout_trajectories,
            goal=layout.goal,
            threshold=goal_threshold,
        )
    return results


def _get_render_goal(
    layout: SyntheticTaskLayout,
    task_name: str,
) -> np.ndarray:
    """Return a goal position for rendering, falling back to an approximate
    center for tasks whose true goal depends on the selected mode.

    Args:
        layout: Task layout from ``get_task_layout``.
        task_name: SyntheticTaskName.value string, used to select the
            fallback location for mode-dependent tasks.

    Returns:
        Cartesian (x, y) goal position for rendering. Shape (2,), float32.

    Raises:
        ValueError: If ``task_name`` has a mode-dependent goal but is not
            a recognized task handled by this helper.
    """
    if layout.goal is not None:
        return layout.goal.copy()
    match task_name:
        case SyntheticTaskName.SEQUENTIAL_DECISION.value:
            return np.array([0.5, 0.95], dtype=np.float32)
        case SyntheticTaskName.SHARED_PREFIX.value:
            return np.array([1.0, 0.5], dtype=np.float32)
        case _:
            raise ValueError(
                f"Task {task_name} has mode-dependent goal but no render fallback"
            )


def _prepare_observation(
    position_history: np.ndarray,
    obstacles: list[tuple[float, float, float, float]],
    goal: np.ndarray,
    image_size: int,
    observation_keys: set[str],
    trail: np.ndarray | None = None,
    context_vector: np.ndarray | None = None,
) -> dict[str, torch.Tensor]:
    """Build an observation dict suitable for Policy.predict_action().

    Renders one frame per history timestep with progressive trails, and
    stacks all modalities along the temporal dimension to match the
    (B, T, ...) convention used during training.

    Args:
        position_history: Last obs_horizon Cartesian positions (x, y)
            in [0, 1]. Shape (obs_horizon, 2).
        obstacles: Obstacle rectangles for rendering.
        goal: Goal position for rendering. Shape (2,).
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
                goal=goal,
                image_size=image_size,
                trail=frame_trail,
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
