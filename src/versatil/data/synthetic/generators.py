"""Trajectory generators for synthetic multimodal benchmark tasks.

Each task produces episodes with controlled multimodality in [0, 1]x[0, 1]
Cartesian space. Actions are fixed delta positions: action[t] = position[t+1] - position[t].
"""

import numpy as np

from versatil.data.synthetic.constants import (
    CIRCLE_CENTER_BOTTOM,
    CIRCLE_CENTER_TOP,
    CIRCLE_CONTEXT_COLORS,
    CIRCLE_DEFAULT_NUM_MODES,
    CIRCLE_OBSTACLES,
    CIRCLE_RADIUS,
    CORRIDOR_DEFAULT_NUM_STYLES,
    CORRIDOR_GAP_HEIGHT,
    CORRIDOR_GOAL,
    CORRIDOR_START,
    CORRIDOR_WALL_X1,
    CORRIDOR_WALL_X2,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_EPISODES,
    DEFAULT_SEED,
    MULTIPATH_DEFAULT_NOISE_STD,
    MULTIPATH_DEFAULT_NUM_MODES,
    MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
    RADIAL_CENTER,
    RADIAL_RADIUS,
    SEQUENTIAL_ENDPOINT_Y,
    SEQUENTIAL_FIRST_BRANCH_X_DELTA,
    SEQUENTIAL_FORK_TRANSITION_OFFSET,
    SEQUENTIAL_FORK_Y_1,
    SEQUENTIAL_FORK_Y_2,
    SEQUENTIAL_NUM_COMPOUND_MODES,
    SEQUENTIAL_OBSTACLES,
    SEQUENTIAL_SECOND_BRANCH_X_DELTA,
    SEQUENTIAL_START,
    SyntheticTaskName,
)
from versatil.data.synthetic.renderer import render_episode


def generate_task_episodes(
    task_name: str = SyntheticTaskName.CIRCLE.value,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    seed: int = DEFAULT_SEED,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_modes: int = MULTIPATH_DEFAULT_NUM_MODES,
    trajectory_length: int = MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
    noise_std: float = MULTIPATH_DEFAULT_NOISE_STD,
    num_styles: int = CORRIDOR_DEFAULT_NUM_STYLES,
    mode_weights: list[float] | None = None,
) -> list[dict[str, np.ndarray]]:
    """Generate synthetic episodes for a given task.

    Args:
        task_name: SyntheticTaskName.value string identifying which
            multimodal navigation task to generate.
        num_episodes: Total number of episodes to generate, balanced
            equally across all behavioral modes.
        seed: Random seed for reproducible generation.
        image_size: Side length in pixels of the rendered top-down RGB
            images (square).
        num_modes: Number of distinct behavioral modes for tasks that
            accept a variable mode count (radial, corridor_navigation).
        trajectory_length: Number of timesteps per episode.
        noise_std: Standard deviation of Gaussian noise added to
            trajectory positions for intra-mode variance.
        num_styles: Number of sinusoidal trajectory styles per corridor
            (corridor_navigation task only).
        mode_weights: Relative weights per mode for imbalanced generation.
            None for uniform distribution across modes.

    Returns:
        List of episode dicts. Each dict contains:
            "image": rendered top-down RGB, shape (T, image_size, image_size, 3), uint8
            "position": Cartesian (x, y) states, shape (T, 2), float32
            "action": delta (dx, dy) commands, shape (T, 2), float32
            "mode_id": ground-truth mode label, shape (T, 1), uint8
            "context": conditioning context vector, shape (T, C), float32
    """
    random_generator = np.random.default_rng(seed)
    match task_name:
        case SyntheticTaskName.CIRCLE.value:
            return _generate_circle(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
                mode_weights=mode_weights,
            )
        case SyntheticTaskName.CONDITIONAL_CIRCLE.value:
            return _generate_conditional_circle(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
                mode_weights=mode_weights,
            )
        case SyntheticTaskName.SEQUENTIAL_DECISION.value:
            return _generate_sequential_decision(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
                mode_weights=mode_weights,
            )
        case SyntheticTaskName.RADIAL.value:
            return _generate_radial(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                num_modes=num_modes,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
                mode_weights=mode_weights,
            )
        case SyntheticTaskName.CORRIDOR_NAVIGATION.value:
            return _generate_corridor_navigation(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                num_modes=num_modes,
                num_styles=num_styles,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
                mode_weights=mode_weights,
            )
        case _:
            raise ValueError(f"Unknown synthetic task: {task_name}")


def _generate_circle(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    trajectory_length: int,
    noise_std: float,
    mode_weights: list[float] | None,
) -> list[dict[str, np.ndarray]]:
    """Traverse one of two tangent circles as a closed loop.

    Mode 0 = bottom circle, Mode 1 = top circle. The trajectory starts at
    the tangent point (0.5, 0.5) and traces a full clockwise loop around
    the selected circle, returning to the start.
    """
    return _generate_circle_episodes(
        num_episodes=num_episodes,
        random_generator=random_generator,
        image_size=image_size,
        trajectory_length=trajectory_length,
        noise_std=noise_std,
        mode_weights=mode_weights,
        use_context=False,
    )


def _generate_conditional_circle(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    trajectory_length: int,
    noise_std: float,
    mode_weights: list[float] | None,
) -> list[dict[str, np.ndarray]]:
    """Same layout as circle but with a one-hot context signal per mode.

    Tests whether models exploit the context to resolve ambiguity.
    When conditioned on context, each mode becomes unimodal.
    """
    return _generate_circle_episodes(
        num_episodes=num_episodes,
        random_generator=random_generator,
        image_size=image_size,
        trajectory_length=trajectory_length,
        noise_std=noise_std,
        mode_weights=mode_weights,
        use_context=True,
    )


def _generate_circle_episodes(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    trajectory_length: int,
    noise_std: float,
    mode_weights: list[float] | None,
    use_context: bool,
) -> list[dict[str, np.ndarray]]:
    """Shared implementation for circle and conditional_circle tasks.

    Args:
        num_episodes: Total episodes to generate.
        random_generator: NumPy random generator.
        image_size: Side length of rendered images.
        trajectory_length: Timesteps per episode.
        noise_std: Gaussian noise standard deviation.
        mode_weights: Per-mode weights or None for uniform.
        use_context: When True, set one-hot context and render context color.
    """
    num_modes = CIRCLE_DEFAULT_NUM_MODES
    episodes = []
    episodes_per_mode = _resolve_mode_counts(
        total_episodes=num_episodes,
        num_modes=num_modes,
        mode_weights=mode_weights,
    )
    centers = {0: CIRCLE_CENTER_BOTTOM, 1: CIRCLE_CENTER_TOP}

    for mode_index in range(num_modes):
        center = centers[mode_index]
        context_color = CIRCLE_CONTEXT_COLORS[mode_index] if use_context else None
        for _ in range(episodes_per_mode[mode_index]):
            positions = _parametric_circle(
                center=center,
                radius=CIRCLE_RADIUS,
                num_points=trajectory_length,
                clockwise=True,
            )
            positions = _add_noise_and_clamp(
                trajectory=positions,
                noise_std=noise_std,
                random_generator=random_generator,
            )
            actions = _compute_actions(positions)
            images = render_episode(
                positions=positions,
                obstacles=CIRCLE_OBSTACLES,
                image_size=image_size,
                context_color=context_color,
            )
            if use_context:
                context_vector = np.zeros(num_modes, dtype=np.float32)
                context_vector[mode_index] = 1.0
                context = np.tile(context_vector, (trajectory_length, 1))
            else:
                context = np.zeros((trajectory_length, num_modes), dtype=np.float32)
            mode_label = np.full((trajectory_length, 1), mode_index, dtype=np.uint8)
            episodes.append(
                {
                    "image": images,
                    "position": positions,
                    "action": actions,
                    "mode_id": mode_label,
                    "context": context,
                }
            )
    random_generator.shuffle(episodes)
    return episodes


def _generate_sequential_decision(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    trajectory_length: int,
    noise_std: float,
    mode_weights: list[float] | None,
) -> list[dict[str, np.ndarray]]:
    """Navigate upward from (0.5, 0) with two sequential left/right forks.

    First fork at y=0.4, second at y=0.7. Produces 4 compound modes
    (LL, LR, RL, RR) with obstacles at each fork point. Tests whether
    the model represents hierarchical sequential mode structure.
    """
    compound_modes = SEQUENTIAL_NUM_COMPOUND_MODES
    episodes = []
    episodes_per_mode = _resolve_mode_counts(
        total_episodes=num_episodes,
        num_modes=compound_modes,
        mode_weights=mode_weights,
    )
    mode_definitions = [
        ("left", "left"),
        ("left", "right"),
        ("right", "left"),
        ("right", "right"),
    ]
    start_x = float(SEQUENTIAL_START[0])
    start_y = float(SEQUENTIAL_START[1])

    for mode_index, (first_choice, second_choice) in enumerate(mode_definitions):
        first_x_delta = (
            -SEQUENTIAL_FIRST_BRANCH_X_DELTA
            if first_choice == "left"
            else SEQUENTIAL_FIRST_BRANCH_X_DELTA
        )
        second_x_delta = (
            -SEQUENTIAL_SECOND_BRANCH_X_DELTA
            if second_choice == "left"
            else SEQUENTIAL_SECOND_BRANCH_X_DELTA
        )
        waypoints = [
            (start_x, start_y),
            (start_x, SEQUENTIAL_FORK_Y_1),
            (
                start_x + first_x_delta,
                SEQUENTIAL_FORK_Y_1 + SEQUENTIAL_FORK_TRANSITION_OFFSET,
            ),
            (start_x + first_x_delta, SEQUENTIAL_FORK_Y_2),
            (
                start_x + first_x_delta + second_x_delta,
                SEQUENTIAL_FORK_Y_2 + SEQUENTIAL_FORK_TRANSITION_OFFSET,
            ),
            (start_x + first_x_delta + second_x_delta, SEQUENTIAL_ENDPOINT_Y),
        ]
        goal = np.array(waypoints[-1], dtype=np.float32)
        for _ in range(episodes_per_mode[mode_index]):
            positions = _interpolate_waypoints(
                waypoints=waypoints, num_points=trajectory_length
            )
            positions = _add_noise_and_clamp(
                trajectory=positions,
                noise_std=noise_std,
                random_generator=random_generator,
            )
            actions = _compute_actions(positions)
            images = render_episode(
                positions=positions,
                obstacles=SEQUENTIAL_OBSTACLES,
                goal=goal,
                image_size=image_size,
            )
            context = np.zeros((trajectory_length, compound_modes), dtype=np.float32)
            mode_label = np.full((trajectory_length, 1), mode_index, dtype=np.uint8)
            episodes.append(
                {
                    "image": images,
                    "position": positions,
                    "action": actions,
                    "mode_id": mode_label,
                    "context": context,
                }
            )
    random_generator.shuffle(episodes)
    return episodes


def _generate_radial(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    num_modes: int,
    trajectory_length: int,
    noise_std: float,
    mode_weights: list[float] | None,
) -> list[dict[str, np.ndarray]]:
    """K straight-line trajectories from center to K evenly-spaced points on a circle.

    Mode i travels to angle 2*pi*i/K at radius 0.4 from center.
    Obstacles are placed dynamically between each adjacent pair of radii.
    BC failure: mean action is zero displacement.
    """
    episodes = []
    episodes_per_mode = _resolve_mode_counts(
        total_episodes=num_episodes,
        num_modes=num_modes,
        mode_weights=mode_weights,
    )
    obstacles = _generate_radial_obstacles(num_modes=num_modes)

    for mode_index in range(num_modes):
        angle = 2.0 * np.pi * mode_index / num_modes
        endpoint_x = float(RADIAL_CENTER[0]) + RADIAL_RADIUS * np.cos(angle)
        endpoint_y = float(RADIAL_CENTER[1]) + RADIAL_RADIUS * np.sin(angle)
        goal = np.array([endpoint_x, endpoint_y], dtype=np.float32)
        waypoints = [
            (float(RADIAL_CENTER[0]), float(RADIAL_CENTER[1])),
            (endpoint_x, endpoint_y),
        ]
        for _ in range(episodes_per_mode[mode_index]):
            positions = _interpolate_waypoints(
                waypoints=waypoints, num_points=trajectory_length
            )
            positions = _add_noise_and_clamp(
                trajectory=positions,
                noise_std=noise_std,
                random_generator=random_generator,
            )
            actions = _compute_actions(positions)
            images = render_episode(
                positions=positions,
                obstacles=obstacles,
                goal=goal,
                image_size=image_size,
            )
            context = np.zeros((trajectory_length, num_modes), dtype=np.float32)
            mode_label = np.full((trajectory_length, 1), mode_index, dtype=np.uint8)
            episodes.append(
                {
                    "image": images,
                    "position": positions,
                    "action": actions,
                    "mode_id": mode_label,
                    "context": context,
                }
            )
    random_generator.shuffle(episodes)
    return episodes


def _generate_corridor_navigation(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    num_modes: int,
    num_styles: int,
    trajectory_length: int,
    noise_std: float,
    mode_weights: list[float] | None,
) -> list[dict[str, np.ndarray]]:
    """Navigate through one of K gaps in a vertical wall, with S style variations.

    A vertical wall at x in [0.45, 0.55] has K gaps. Each gap defines
    a corridor mode. S sinusoidal style variations per corridor produce
    K*S total modes. Trajectory goes start -> gap center -> goal.

    K must be even so that no gap falls at y=0.5, ensuring the BC
    mean (which aims straight at y=0.5) always collides with the wall.
    """
    if num_modes % 2 != 0:
        raise ValueError(
            f"corridor_navigation requires even num_modes so no gap "
            f"falls at y=0.5 (BC must collide), got {num_modes}"
        )
    total_modes = num_modes * num_styles
    episodes = []
    episodes_per_mode = _resolve_mode_counts(
        total_episodes=num_episodes,
        num_modes=total_modes,
        mode_weights=mode_weights,
    )
    gap_centers = _compute_corridor_gap_centers(num_gaps=num_modes)
    obstacles = _generate_corridor_obstacles(gap_centers=gap_centers)
    wall_center_x = (CORRIDOR_WALL_X1 + CORRIDOR_WALL_X2) / 2.0

    for gap_index in range(num_modes):
        gap_y = gap_centers[gap_index]
        for style_index in range(num_styles):
            flat_mode_index = gap_index * num_styles + style_index
            waypoints = [
                (float(CORRIDOR_START[0]), float(CORRIDOR_START[1])),
                (wall_center_x, gap_y),
                (float(CORRIDOR_GOAL[0]), float(CORRIDOR_GOAL[1])),
            ]
            for _ in range(episodes_per_mode[flat_mode_index]):
                positions = _interpolate_waypoints(
                    waypoints=waypoints, num_points=trajectory_length
                )
                if num_styles > 1:
                    positions = _apply_sinusoidal_style(
                        positions=positions,
                        style_index=style_index,
                        num_styles=num_styles,
                    )
                positions = _add_noise_and_clamp(
                    trajectory=positions,
                    noise_std=noise_std,
                    random_generator=random_generator,
                )
                actions = _compute_actions(positions)
                images = render_episode(
                    positions=positions,
                    obstacles=obstacles,
                    goal=CORRIDOR_GOAL,
                    image_size=image_size,
                )
                context = np.zeros((trajectory_length, total_modes), dtype=np.float32)
                mode_label = np.full(
                    (trajectory_length, 1), flat_mode_index, dtype=np.uint8
                )
                episodes.append(
                    {
                        "image": images,
                        "position": positions,
                        "action": actions,
                        "mode_id": mode_label,
                        "context": context,
                    }
                )
    random_generator.shuffle(episodes)
    return episodes


def _parametric_circle(
    center: np.ndarray,
    radius: float,
    num_points: int,
    clockwise: bool,
) -> np.ndarray:
    """Generate positions along a parametric circle.

    Starts at the point on the circle closest to (0.5, 0.5) and traces
    a full loop. For the bottom circle this is the top of the circle,
    for the top circle this is the bottom.

    Args:
        center: Circle center (x, y). Shape (2,).
        radius: Circle radius in [0, 1] space.
        num_points: Number of trajectory positions.
        clockwise: Traverse clockwise if True, counter-clockwise otherwise.

    Returns:
        Cartesian positions, shape (num_points, 2), dtype float32.
    """
    start_angle = np.arctan2(0.5 - float(center[1]), 0.5 - float(center[0]))
    direction = -1.0 if clockwise else 1.0
    theta = start_angle + direction * np.linspace(
        0.0, 2.0 * np.pi, num_points, endpoint=False, dtype=np.float32
    )
    x_positions = float(center[0]) + radius * np.cos(theta)
    y_positions = float(center[1]) + radius * np.sin(theta)
    return np.stack(
        [x_positions.astype(np.float32), y_positions.astype(np.float32)],
        axis=-1,
    )


def _generate_radial_obstacles(
    num_modes: int,
) -> list[tuple[float, float, float, float]]:
    """Generate obstacle rectangles between each adjacent pair of radii.

    Places a small rectangle at the midpoint angle between consecutive
    radii, at half the radial distance from center.

    Args:
        num_modes: Number of radial modes (K).

    Returns:
        List of (x_min, y_min, x_max, y_max) obstacle rectangles.
    """
    obstacles: list[tuple[float, float, float, float]] = []
    # Scale obstacle size with angular separation to avoid clipping trajectories at large K
    angular_gap = 2.0 * np.pi / num_modes
    max_half_size = 0.04
    obstacle_half_width = min(
        max_half_size, 0.25 * RADIAL_RADIUS * np.sin(angular_gap / 2)
    )
    obstacle_half_height = obstacle_half_width
    midpoint_radius = RADIAL_RADIUS * 0.5

    for mode_index in range(num_modes):
        angle_a = 2.0 * np.pi * mode_index / num_modes
        angle_b = 2.0 * np.pi * (mode_index + 1) / num_modes
        midpoint_angle = (angle_a + angle_b) / 2.0
        center_x = float(RADIAL_CENTER[0]) + midpoint_radius * np.cos(midpoint_angle)
        center_y = float(RADIAL_CENTER[1]) + midpoint_radius * np.sin(midpoint_angle)
        obstacles.append(
            (
                center_x - obstacle_half_width,
                center_y - obstacle_half_height,
                center_x + obstacle_half_width,
                center_y + obstacle_half_height,
            )
        )
    return obstacles


def _compute_corridor_gap_centers(
    num_gaps: int,
) -> list[float]:
    """Compute the y-coordinates of gap centers in the corridor wall.

    Gaps are evenly distributed across the wall height, excluding the
    top and bottom edges.

    Args:
        num_gaps: Number of gaps (K).

    Returns:
        List of y-coordinates for each gap center.
    """
    return [(index + 1) / (num_gaps + 1) for index in range(num_gaps)]


def _generate_corridor_obstacles(
    gap_centers: list[float],
) -> list[tuple[float, float, float, float]]:
    """Generate wall segments between adjacent gaps in the corridor.

    Creates K-1 wall segments between K gaps. No wall segments at
    the top/bottom edges of the unit square.

    Args:
        gap_centers: Sorted y-coordinates of gap centers.

    Returns:
        List of (x_min, y_min, x_max, y_max) wall segment rectangles.
    """
    obstacles: list[tuple[float, float, float, float]] = []
    half_gap = CORRIDOR_GAP_HEIGHT / 2.0

    for index in range(len(gap_centers) - 1):
        wall_y_min = gap_centers[index] + half_gap
        wall_y_max = gap_centers[index + 1] - half_gap
        if wall_y_max > wall_y_min:
            obstacles.append(
                (CORRIDOR_WALL_X1, wall_y_min, CORRIDOR_WALL_X2, wall_y_max)
            )
    return obstacles


def _apply_sinusoidal_style(
    positions: np.ndarray,
    style_index: int,
    num_styles: int,
) -> np.ndarray:
    """Add sinusoidal y-displacement to produce trajectory style variations.

    Each style uses a different frequency to create visually distinct
    curved trajectories through the same corridor.

    Args:
        positions: Base trajectory positions, shape (num_steps, 2).
        style_index: Index of the sinusoidal style (0-based).
        num_styles: Total number of styles for amplitude scaling.

    Returns:
        Modified positions with sinusoidal y-displacement, shape (num_steps, 2).
    """
    num_steps = len(positions)
    normalized_time = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    envelope = 4.0 * normalized_time * (1.0 - normalized_time)
    frequency = 2.0 * (style_index + 1)
    # Amplitude must not exceed gap half-height to avoid pushing through walls
    max_amplitude = CORRIDOR_GAP_HEIGHT / 2.0 * 0.8
    amplitude = min(0.06 / num_styles, max_amplitude)
    y_offset = amplitude * np.sin(frequency * np.pi * normalized_time) * envelope
    modified = positions.copy()
    modified[:, 1] += y_offset.astype(np.float32)
    return modified


def _interpolate_waypoints(
    waypoints: list[tuple[float, float]],
    num_points: int,
) -> np.ndarray:
    """Linearly interpolate between ordered waypoints to produce a trajectory.

    Distributes num_points evenly along the piecewise-linear path defined
    by the waypoint sequence.

    Args:
        waypoints: Ordered Cartesian waypoints [(x0, y0), (x1, y1), ...].
        num_points: Total number of trajectory positions to produce.

    Returns:
        Cartesian positions of shape (num_points, 2), dtype float32.
    """
    waypoint_array = np.array(waypoints, dtype=np.float32)
    segment_lengths = np.linalg.norm(np.diff(waypoint_array, axis=0), axis=1)
    cumulative_distance = np.concatenate([np.array([0.0]), np.cumsum(segment_lengths)])
    total_distance = cumulative_distance[-1]

    uniform_distances = np.linspace(0.0, total_distance, num_points)
    interpolated_x = np.interp(
        uniform_distances, cumulative_distance, waypoint_array[:, 0]
    )
    interpolated_y = np.interp(
        uniform_distances, cumulative_distance, waypoint_array[:, 1]
    )
    return np.stack([interpolated_x, interpolated_y], axis=-1).astype(np.float32)


def _add_noise_and_clamp(
    trajectory: np.ndarray,
    noise_std: float,
    random_generator: np.random.Generator,
) -> np.ndarray:
    """Add isotropic Gaussian noise and clamp to [0, 1].

    Args:
        trajectory: Cartesian positions (x, y) of shape (num_steps, 2).
        noise_std: Standard deviation of the additive Gaussian noise.
        random_generator: NumPy random generator for reproducibility.

    Returns:
        Noisy positions clamped to [0, 1], shape (num_steps, 2), float32.
    """
    noise = random_generator.normal(0.0, noise_std, size=trajectory.shape).astype(
        np.float32
    )
    noisy_trajectory = trajectory + noise
    return np.clip(noisy_trajectory, 0.0, 1.0)


def _compute_actions(
    positions: np.ndarray,
) -> np.ndarray:
    """Compute delta-position actions from consecutive Cartesian positions.

    action[t] = position[t+1] - position[t], with the last action set to zeros.

    Args:
        positions: Cartesian positions (x, y) of shape (num_steps, 2).

    Returns:
        Delta actions (dx, dy) of shape (num_steps, 2), dtype float32.
    """
    actions = np.zeros_like(positions)
    actions[:-1] = positions[1:] - positions[:-1]
    return actions


def _balanced_mode_counts(
    total_episodes: int,
    num_modes: int,
) -> list[int]:
    """Distribute episodes as evenly as possible across modes.

    Any remainder from integer division is distributed one extra episode
    to the first modes.

    Args:
        total_episodes: Total number of episodes to distribute.
        num_modes: Number of behavioral modes.

    Returns:
        List of episode counts per mode, summing to total_episodes.
    """
    base_count = total_episodes // num_modes
    remainder = total_episodes % num_modes
    counts = [
        base_count + (1 if index < remainder else 0) for index in range(num_modes)
    ]
    return counts


def _weighted_mode_counts(
    total_episodes: int,
    mode_weights: list[float],
) -> list[int]:
    """Distribute episodes according to relative mode weights.

    Weights are normalized to sum to 1. Rounding remainders are assigned
    to modes with the largest fractional parts.

    Args:
        total_episodes: Total number of episodes to distribute.
        mode_weights: Relative weight per mode (must be positive).

    Returns:
        List of episode counts per mode, summing to total_episodes.
    """
    weight_sum = sum(mode_weights)
    normalized = [weight / weight_sum for weight in mode_weights]
    fractional_counts = [total_episodes * weight for weight in normalized]
    base_counts = [int(count) for count in fractional_counts]
    remainders = [
        fractional - base for fractional, base in zip(fractional_counts, base_counts)
    ]
    deficit = total_episodes - sum(base_counts)
    sorted_indices = sorted(
        range(len(remainders)), key=lambda i: remainders[i], reverse=True
    )
    for rank in range(deficit):
        base_counts[sorted_indices[rank]] += 1
    return base_counts


def _resolve_mode_counts(
    total_episodes: int,
    num_modes: int,
    mode_weights: list[float] | None,
) -> list[int]:
    """Dispatch to balanced or weighted episode distribution.

    Args:
        total_episodes: Total number of episodes.
        num_modes: Number of behavioral modes.
        mode_weights: Relative weights or None for uniform.

    Returns:
        List of episode counts per mode.
    """
    if mode_weights is None:
        return _balanced_mode_counts(total_episodes=total_episodes, num_modes=num_modes)
    if len(mode_weights) != num_modes:
        raise ValueError(
            f"mode_weights length ({len(mode_weights)}) must match "
            f"num_modes ({num_modes})"
        )
    return _weighted_mode_counts(
        total_episodes=total_episodes, mode_weights=mode_weights
    )
