"""Trajectory generators for synthetic multimodal benchmark tasks.

Each task produces episodes with controlled multimodality in [0, 1]x[0, 1]
Cartesian space. Actions are fixed delta positions: action[t] = position[t+1] - position[t].
"""

import numpy as np

from versatil.data.synthetic.constants import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NUM_EPISODES,
    DEFAULT_SEED,
    MULTIPATH_CONTEXT_COLORS,
    MULTIPATH_DEFAULT_NOISE_STD,
    MULTIPATH_DEFAULT_NUM_MODES,
    MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
    MULTIPATH_GOAL,
    MULTIPATH_MIN_TRAJECTORY_LENGTH,
    MULTIPATH_OBSTACLES,
    MULTIPATH_WAYPOINTS,
    SEQUENTIAL_BRANCH_X_DELTA,
    SEQUENTIAL_DEFAULT_NOISE_STD,
    SEQUENTIAL_DEFAULT_TRAJECTORY_LENGTH,
    SEQUENTIAL_INTERSECTION_Y_1,
    SEQUENTIAL_INTERSECTION_Y_2,
    SEQUENTIAL_MIN_TRAJECTORY_LENGTH,
    SEQUENTIAL_START,
    SHARED_PREFIX_DECISION_POINT_X,
    SHARED_PREFIX_DEFAULT_NOISE_STD,
    SHARED_PREFIX_DEFAULT_TRAJECTORY_LENGTH,
    SHARED_PREFIX_ENDPOINTS,
    SHARED_PREFIX_MIN_TRAJECTORY_LENGTH,
    SHARED_PREFIX_SHARED_STEPS,
    SHARED_PREFIX_START,
    STYLE_DEFAULT_NOISE_STD,
    STYLE_DEFAULT_NUM_STYLES,
    STYLE_DEFAULT_TRAJECTORY_LENGTH,
    STYLE_GOAL,
    STYLE_MIN_TRAJECTORY_LENGTH,
    STYLE_START,
    SyntheticTaskName,
)
from versatil.data.synthetic.renderer import render_episode


def generate_task_episodes(
    task_name: str = SyntheticTaskName.MULTI_PATH_NAVIGATION.value,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    seed: int = DEFAULT_SEED,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_modes: int = MULTIPATH_DEFAULT_NUM_MODES,
    trajectory_length: int = MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
    noise_std: float = MULTIPATH_DEFAULT_NOISE_STD,
    num_styles: int = STYLE_DEFAULT_NUM_STYLES,
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
            use discrete path choices (multi_path, conditional, shared_prefix).
        trajectory_length: Number of timesteps per episode.
        noise_std: Standard deviation of Gaussian noise added to
            trajectory positions for intra-mode variance.
        num_styles: Number of trajectory styles for the trajectory_style task.

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
        case SyntheticTaskName.MULTI_PATH_NAVIGATION.value:
            return _generate_multi_path_navigation(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                num_modes=num_modes,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
            )
        case SyntheticTaskName.CONDITIONAL_NAVIGATION.value:
            return _generate_conditional_navigation(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                num_modes=num_modes,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
            )
        case SyntheticTaskName.TRAJECTORY_STYLE.value:
            return _generate_trajectory_style(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                num_styles=num_styles,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
            )
        case SyntheticTaskName.SEQUENTIAL_DECISION.value:
            return _generate_sequential_decision(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
            )
        case SyntheticTaskName.SHARED_PREFIX.value:
            return _generate_shared_prefix(
                num_episodes=num_episodes,
                random_generator=random_generator,
                image_size=image_size,
                num_modes=num_modes,
                trajectory_length=trajectory_length,
                noise_std=noise_std,
            )
        case _:
            raise ValueError(f"Unknown synthetic task: {task_name}")


def _generate_multi_path_navigation(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    num_modes: int,
    trajectory_length: int,
    noise_std: float,
) -> list[dict[str, np.ndarray]]:
    """Navigate from (0,0) to (0.95, 0.95) through one of N corridors.

    Two rectangular obstacles divide the space into N distinct passages.
    Each episode uniformly selects one corridor, producing N behavioral
    modes that are geometrically separated. The model receives no
    information about which corridor to take, so a unimodal policy will
    average across paths and collide with obstacles.
    """
    if trajectory_length < MULTIPATH_MIN_TRAJECTORY_LENGTH:
        raise ValueError(
            f"multi_path_navigation requires trajectory_length >= {MULTIPATH_MIN_TRAJECTORY_LENGTH}, "
            f"got {trajectory_length}"
        )
    episodes = []
    episodes_per_mode = _balanced_mode_counts(num_episodes, num_modes)
    for mode_index in range(num_modes):
        waypoints = MULTIPATH_WAYPOINTS[mode_index]
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
                obstacles=MULTIPATH_OBSTACLES,
                goal=MULTIPATH_GOAL,
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


def _generate_conditional_navigation(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    num_modes: int,
    trajectory_length: int,
    noise_std: float,
) -> list[dict[str, np.ndarray]]:
    """Navigate the same multi-path layout, but with a context signal.

    Identical obstacle layout and corridors as multi_path_navigation, but
    a discrete context variable c in {0, ..., N-1} is provided as a one-hot
    observation that deterministically maps to a specific corridor. When
    the model conditions on context, the task becomes unimodal per context
    value, and the latent space should collapse. Compare with/without
    context to measure latent space utility.
    """
    if trajectory_length < MULTIPATH_MIN_TRAJECTORY_LENGTH:
        raise ValueError(
            f"conditional_navigation requires trajectory_length >= {MULTIPATH_MIN_TRAJECTORY_LENGTH}, "
            f"got {trajectory_length}"
        )
    episodes = []
    episodes_per_mode = _balanced_mode_counts(num_episodes, num_modes)
    for mode_index in range(num_modes):
        waypoints = MULTIPATH_WAYPOINTS[mode_index]
        context_color = MULTIPATH_CONTEXT_COLORS.get(mode_index)
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
                obstacles=MULTIPATH_OBSTACLES,
                goal=MULTIPATH_GOAL,
                image_size=image_size,
                context_color=context_color,
            )
            context_vector = np.zeros(num_modes, dtype=np.float32)
            context_vector[mode_index] = 1.0
            context = np.tile(context_vector, (trajectory_length, 1))  # (T, num_modes)
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


def _generate_trajectory_style(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    num_styles: int,
    trajectory_length: int,
    noise_std: float,
) -> list[dict[str, np.ndarray]]:
    """Move from (0, 0.5) to (1.0, 0.5) using different movement styles.

    All trajectories share the same start and goal, so modes differ not
    in where the agent goes but how it gets there: straight line, arc
    curving up, arc curving down, or zigzag. This tests whether the
    latent space captures continuous style variation rather than
    discrete geometric path choices. Harder than multi-path because
    modes overlap spatially and differ primarily in velocity profiles.
    """
    noise_std = (
        STYLE_DEFAULT_NOISE_STD
        if noise_std == MULTIPATH_DEFAULT_NOISE_STD
        else noise_std
    )
    trajectory_length = (
        STYLE_DEFAULT_TRAJECTORY_LENGTH
        if trajectory_length == MULTIPATH_DEFAULT_TRAJECTORY_LENGTH
        else trajectory_length
    )
    if trajectory_length < STYLE_MIN_TRAJECTORY_LENGTH:
        raise ValueError(
            f"trajectory_style requires trajectory_length >= {STYLE_MIN_TRAJECTORY_LENGTH}, "
            f"got {trajectory_length}"
        )
    episodes = []
    episodes_per_style = _balanced_mode_counts(num_episodes, num_styles)
    normalized_time = np.linspace(0.0, 1.0, trajectory_length, dtype=np.float32)
    for style_index in range(num_styles):
        for _ in range(episodes_per_style[style_index]):
            x_positions = STYLE_START[0] + normalized_time * (
                STYLE_GOAL[0] - STYLE_START[0]
            )

            if style_index == 0:
                # Fast & straight
                y_positions = np.full(
                    trajectory_length, STYLE_START[1], dtype=np.float32
                )
            elif style_index == 1:
                # Curved up
                y_positions = STYLE_START[1] + 0.3 * np.sin(np.pi * normalized_time)
            elif style_index == 2:
                # Curved down
                y_positions = STYLE_START[1] - 0.3 * np.sin(np.pi * normalized_time)
            else:
                # Zigzag with envelope
                envelope = 4.0 * normalized_time * (1.0 - normalized_time)
                y_positions = (
                    STYLE_START[1]
                    + 0.15 * np.sin(8.0 * np.pi * normalized_time) * envelope
                )

            positions = np.stack([x_positions, y_positions.astype(np.float32)], axis=-1)
            # Noise on y-axis only
            y_noise = random_generator.normal(
                0.0, noise_std, size=trajectory_length
            ).astype(np.float32)
            positions[:, 1] += y_noise
            positions = np.clip(positions, 0.0, 1.0)
            actions = _compute_actions(positions)
            images = render_episode(
                positions=positions,
                obstacles=[],
                goal=STYLE_GOAL,
                image_size=image_size,
            )
            context = np.zeros((trajectory_length, num_styles), dtype=np.float32)
            mode_label = np.full((trajectory_length, 1), style_index, dtype=np.uint8)
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
) -> list[dict[str, np.ndarray]]:
    """Navigate upward from (0.5, 0) with two sequential left/right choices.

    Starting from (0.5, 0), the agent approaches two sequential
    intersection points (at y~0.3 and y~0.55) where it independently
    chooses left or right. This creates 4 compound modes (LL, LR, RL, RR)
    with distinct endpoints. Tests whether the model can represent
    hierarchical/sequential mode structure rather than flat mode enumeration.
    """
    trajectory_length = (
        SEQUENTIAL_DEFAULT_TRAJECTORY_LENGTH
        if trajectory_length == MULTIPATH_DEFAULT_TRAJECTORY_LENGTH
        else trajectory_length
    )
    noise_std = (
        SEQUENTIAL_DEFAULT_NOISE_STD
        if noise_std == MULTIPATH_DEFAULT_NOISE_STD
        else noise_std
    )
    if trajectory_length < SEQUENTIAL_MIN_TRAJECTORY_LENGTH:
        raise ValueError(
            f"sequential_decision requires trajectory_length >= {SEQUENTIAL_MIN_TRAJECTORY_LENGTH}, "
            f"got {trajectory_length}"
        )
    compound_modes = 4
    episodes = []
    episodes_per_mode = _balanced_mode_counts(num_episodes, compound_modes)
    # Segment lengths: shared approach (20) + branch 1 (20) + branch 2 (20) = 60
    mode_definitions = [
        ("left", "left"),  # Mode 0: LL
        ("left", "right"),  # Mode 1: LR
        ("right", "left"),  # Mode 2: RL
        ("right", "right"),  # Mode 3: RR
    ]
    start_x = SEQUENTIAL_START[0]
    start_y = SEQUENTIAL_START[1]
    for mode_index, (first_choice, second_choice) in enumerate(mode_definitions):
        first_x_delta = (
            -SEQUENTIAL_BRANCH_X_DELTA
            if first_choice == "left"
            else SEQUENTIAL_BRANCH_X_DELTA
        )
        second_x_delta = (
            -SEQUENTIAL_BRANCH_X_DELTA
            if second_choice == "left"
            else SEQUENTIAL_BRANCH_X_DELTA
        )
        waypoints = [
            (start_x, start_y),
            (start_x, SEQUENTIAL_INTERSECTION_Y_1),
            (start_x + first_x_delta, SEQUENTIAL_INTERSECTION_Y_1 + 0.05),
            (start_x + first_x_delta, SEQUENTIAL_INTERSECTION_Y_2),
            (
                start_x + first_x_delta + second_x_delta,
                SEQUENTIAL_INTERSECTION_Y_2 + 0.05,
            ),
            (start_x + first_x_delta + second_x_delta, 0.95),
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
                obstacles=[],
                goal=np.array(waypoints[-1], dtype=np.float32),
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


def _generate_shared_prefix(
    num_episodes: int,
    random_generator: np.random.Generator,
    image_size: int,
    num_modes: int,
    trajectory_length: int,
    noise_std: float,
) -> list[dict[str, np.ndarray]]:
    """Move right from (0, 0.5) with a shared prefix then diverge into N paths.

    All trajectories share an identical horizontal prefix from x=0 to
    x=0.5 (first 30 steps), then diverge at the decision point into
    N directions (up-right, straight-right, down-right). Mode averaging at the decision point
    causes a unimodal policy to produce a trajectory that goes nowhere
    useful after the prefix, catastrophically failing.
    """
    trajectory_length = (
        SHARED_PREFIX_DEFAULT_TRAJECTORY_LENGTH
        if trajectory_length == MULTIPATH_DEFAULT_TRAJECTORY_LENGTH
        else trajectory_length
    )
    noise_std = (
        SHARED_PREFIX_DEFAULT_NOISE_STD
        if noise_std == MULTIPATH_DEFAULT_NOISE_STD
        else noise_std
    )
    if trajectory_length < SHARED_PREFIX_MIN_TRAJECTORY_LENGTH:
        raise ValueError(
            f"shared_prefix requires trajectory_length >= {SHARED_PREFIX_MIN_TRAJECTORY_LENGTH}, "
            f"got {trajectory_length}"
        )
    episodes = []
    episodes_per_mode = _balanced_mode_counts(num_episodes, num_modes)
    shared_steps = SHARED_PREFIX_SHARED_STEPS
    divergent_steps = trajectory_length - shared_steps
    # Shared prefix: straight right from x=0 to x=0.5 at y=0.5
    shared_x = np.linspace(
        SHARED_PREFIX_START[0],
        SHARED_PREFIX_DECISION_POINT_X,
        shared_steps,
        dtype=np.float32,
    )
    shared_y = np.full(shared_steps, SHARED_PREFIX_START[1], dtype=np.float32)
    shared_positions = np.stack([shared_x, shared_y], axis=-1)
    for mode_index in range(num_modes):
        endpoint_x, endpoint_y = SHARED_PREFIX_ENDPOINTS[mode_index]
        # Divergent segment: from decision point to endpoint
        divergent_x = np.linspace(
            SHARED_PREFIX_DECISION_POINT_X,
            endpoint_x,
            divergent_steps,
            dtype=np.float32,
        )
        divergent_y = np.linspace(
            SHARED_PREFIX_START[1],
            endpoint_y,
            divergent_steps,
            dtype=np.float32,
        )
        divergent_positions = np.stack([divergent_x, divergent_y], axis=-1)
        for _ in range(episodes_per_mode[mode_index]):
            positions = np.concatenate(
                [shared_positions, divergent_positions], axis=0
            ).copy()
            positions = _add_noise_and_clamp(
                trajectory=positions,
                noise_std=noise_std,
                random_generator=random_generator,
            )
            actions = _compute_actions(positions)
            goal = np.array([endpoint_x, endpoint_y], dtype=np.float32)
            images = render_episode(
                positions=positions,
                obstacles=[],
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
