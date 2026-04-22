"""Tests for versatil.data.synthetic.generators module."""

import re
from collections.abc import Callable
from unittest.mock import patch

import numpy as np
import pytest

from versatil.data.synthetic.constants import (
    CIRCLE_CENTER_BOTTOM,
    CIRCLE_CENTER_TOP,
    CIRCLE_CONTEXT_COLORS,
    CIRCLE_DEFAULT_NUM_MODES,
    CIRCLE_RADIUS,
    CORRIDOR_WALL_X1,
    CORRIDOR_WALL_X2,
    MAX_TRAJECTORY_RETRIES,
    RADIAL_CENTER,
    RADIAL_RADIUS,
    SEQUENTIAL_ENDPOINT_Y,
    SEQUENTIAL_FIRST_BRANCH_X_DELTA,
    SEQUENTIAL_NUM_COMPOUND_MODES,
    SEQUENTIAL_OBSTACLES,
    SEQUENTIAL_SECOND_BRANCH_X_DELTA,
    SEQUENTIAL_START,
    SyntheticTaskName,
)
from versatil.data.synthetic.generators import (
    _add_noise_and_clamp,
    _apply_sinusoidal_style,
    _balanced_mode_counts,
    _compute_actions,
    _compute_corridor_gap_centers,
    _compute_corridor_gap_height,
    _generate_corridor_obstacles,
    _generate_radial_obstacles,
    _interpolate_waypoints,
    _parametric_circle,
    _resolve_mode_counts,
    _sample_noisy_trajectory_no_collision,
    _trajectory_collides,
    _weighted_mode_counts,
    generate_task_episodes,
)

EPISODE_KEYS = {"image", "position", "action", "mode_id", "context"}


@pytest.fixture
def fake_render_episode_factory() -> Callable[..., Callable[..., np.ndarray]]:
    def factory(image_size: int = 8) -> Callable[..., np.ndarray]:
        def fake_render(
            positions: np.ndarray,
            obstacles: list[tuple[float, float, float, float]],
            goal: np.ndarray | None = None,
            image_size: int = image_size,
            show_trail: bool = True,
            context_color: tuple[int, int, int] | None = None,
        ) -> np.ndarray:
            num_timesteps = len(positions)
            return np.zeros((num_timesteps, image_size, image_size, 3), dtype=np.uint8)

        return fake_render

    return factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "waypoints, num_points",
    [
        ([(0.0, 0.0), (1.0, 1.0)], 5),
        ([(0.0, 0.0), (0.5, 0.5), (1.0, 0.0)], 11),
        ([(0.2, 0.3), (0.8, 0.7)], 3),
    ],
)
def test_interpolate_waypoints_returns_shape_endpoints_and_dtype(
    waypoints: list[tuple[float, float]], num_points: int
):
    result = _interpolate_waypoints(waypoints=waypoints, num_points=num_points)

    assert result.shape == (num_points, 2)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[0], waypoints[0], atol=1e-6)
    np.testing.assert_allclose(result[-1], waypoints[-1], atol=1e-6)


@pytest.mark.unit
def test_add_noise_and_clamp_zero_std_returns_clamped_input(
    trajectory_factory: Callable[..., np.ndarray],
    rng: np.random.Generator,
):
    trajectory = trajectory_factory(num_points=10)

    result = _add_noise_and_clamp(
        trajectory=trajectory,
        noise_std=0.0,
        random_generator=rng,
    )

    np.testing.assert_array_equal(result, trajectory)


@pytest.mark.unit
def test_add_noise_and_clamp_nonzero_std_differs_and_stays_in_unit_square(
    trajectory_factory: Callable[..., np.ndarray],
    rng: np.random.Generator,
):
    trajectory = trajectory_factory(num_points=20, fill_value=0.5)

    result = _add_noise_and_clamp(
        trajectory=trajectory,
        noise_std=0.1,
        random_generator=rng,
    )

    assert not np.array_equal(result, trajectory)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


@pytest.mark.unit
def test_add_noise_and_clamp_clips_out_of_range_trajectory(
    rng: np.random.Generator,
):
    trajectory = np.full((5, 2), 1.5, dtype=np.float32)

    result = _add_noise_and_clamp(
        trajectory=trajectory,
        noise_std=0.0,
        random_generator=rng,
    )

    np.testing.assert_array_equal(result, np.ones_like(trajectory))


@pytest.mark.unit
def test_compute_actions_produces_position_deltas_with_zero_tail():
    positions = np.array([[0.0, 0.0], [0.1, 0.2], [0.4, 0.6]], dtype=np.float32)

    actions = _compute_actions(positions=positions)

    assert actions.shape == positions.shape
    assert actions.dtype == positions.dtype
    np.testing.assert_allclose(actions[:-1], positions[1:] - positions[:-1], atol=1e-6)
    np.testing.assert_array_equal(actions[-1], np.zeros(2, dtype=np.float32))


@pytest.mark.unit
@pytest.mark.parametrize(
    "total_episodes, num_modes, expected",
    [
        (9, 3, [3, 3, 3]),
        (10, 3, [4, 3, 3]),
        (7, 2, [4, 3]),
        (0, 4, [0, 0, 0, 0]),
    ],
)
def test_balanced_mode_counts_distributes_with_remainder_to_first_modes(
    total_episodes: int, num_modes: int, expected: list[int]
):
    counts = _balanced_mode_counts(total_episodes=total_episodes, num_modes=num_modes)

    assert counts == expected
    assert sum(counts) == total_episodes


@pytest.mark.unit
def test_weighted_mode_counts_distributes_proportionally():
    counts = _weighted_mode_counts(total_episodes=100, mode_weights=[0.7, 0.2, 0.1])

    assert counts == [70, 20, 10]


@pytest.mark.unit
@pytest.mark.parametrize(
    "total_episodes, mode_weights",
    [
        (100, [0.7, 0.2, 0.1]),
        (13, [0.5, 0.5]),
        (7, [1.0, 1.0, 1.0]),
    ],
)
def test_weighted_mode_counts_sums_to_total(
    total_episodes: int, mode_weights: list[float]
):
    counts = _weighted_mode_counts(
        total_episodes=total_episodes, mode_weights=mode_weights
    )

    assert sum(counts) == total_episodes


@pytest.mark.unit
def test_resolve_mode_counts_uses_balanced_when_weights_none():
    counts = _resolve_mode_counts(total_episodes=10, num_modes=3, mode_weights=None)

    assert counts == _balanced_mode_counts(total_episodes=10, num_modes=3)


@pytest.mark.unit
def test_resolve_mode_counts_uses_weighted_when_weights_provided():
    weights = [0.6, 0.4]
    counts = _resolve_mode_counts(total_episodes=10, num_modes=2, mode_weights=weights)

    assert counts == _weighted_mode_counts(total_episodes=10, mode_weights=weights)


@pytest.mark.unit
def test_resolve_mode_counts_raises_on_length_mismatch():
    with pytest.raises(
        ValueError,
        match=re.escape("mode_weights length (3) must match num_modes (2)"),
    ):
        _resolve_mode_counts(
            total_episodes=10, num_modes=2, mode_weights=[0.5, 0.3, 0.2]
        )


@pytest.mark.unit
@pytest.mark.parametrize("num_episodes", [4, 6, 10])
def test_generate_circle_episode_shape_and_keys(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    num_episodes: int,
):
    trajectory_length = 10
    image_size = 8
    num_modes = CIRCLE_DEFAULT_NUM_MODES

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(image_size=image_size),
    ):
        episodes = generate_task_episodes(
            task_name=SyntheticTaskName.CIRCLE.value,
            num_episodes=num_episodes,
            seed=42,
            image_size=image_size,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )

    assert len(episodes) == num_episodes
    for episode in episodes:
        assert set(episode.keys()) == EPISODE_KEYS
        assert episode["image"].shape == (
            trajectory_length,
            image_size,
            image_size,
            3,
        )
        assert episode["image"].dtype == np.uint8
        assert episode["position"].shape == (trajectory_length, 2)
        assert episode["position"].dtype == np.float32
        assert episode["action"].shape == (trajectory_length, 2)
        assert episode["action"].dtype == np.float32
        assert episode["mode_id"].shape == (trajectory_length, 1)
        assert episode["mode_id"].dtype == np.uint8
        assert episode["context"].shape == (trajectory_length, num_modes)
        assert episode["context"].dtype == np.float32


@pytest.mark.unit
def test_generate_circle_mode_balance(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    num_episodes = 10
    num_modes = CIRCLE_DEFAULT_NUM_MODES

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = generate_task_episodes(
            task_name=SyntheticTaskName.CIRCLE.value,
            num_episodes=num_episodes,
            seed=42,
            image_size=8,
            trajectory_length=10,
            noise_std=0.0,
        )

    mode_ids = [int(ep["mode_id"][0, 0]) for ep in episodes]
    assert set(mode_ids) == {0, 1}
    assert mode_ids.count(0) == num_episodes // num_modes
    assert mode_ids.count(1) == num_episodes // num_modes


@pytest.mark.unit
def test_generate_circle_trajectory_is_circular(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    num_modes = CIRCLE_DEFAULT_NUM_MODES

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = generate_task_episodes(
            task_name=SyntheticTaskName.CIRCLE.value,
            num_episodes=num_modes,
            seed=42,
            image_size=8,
            trajectory_length=60,
            noise_std=0.0,
        )

    centers = {0: CIRCLE_CENTER_BOTTOM, 1: CIRCLE_CENTER_TOP}
    for episode in episodes:
        mode_index = int(episode["mode_id"][0, 0])
        center = centers[mode_index]
        distances = np.linalg.norm(episode["position"] - center[np.newaxis, :], axis=1)
        np.testing.assert_allclose(distances, CIRCLE_RADIUS, atol=1e-5)


@pytest.mark.unit
def test_generate_circle_context_is_zeros(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    num_modes = CIRCLE_DEFAULT_NUM_MODES
    trajectory_length = 10

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = generate_task_episodes(
            task_name=SyntheticTaskName.CIRCLE.value,
            num_episodes=num_modes,
            seed=42,
            image_size=8,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )

    for episode in episodes:
        np.testing.assert_array_equal(
            episode["context"],
            np.zeros((trajectory_length, num_modes), dtype=np.float32),
        )


@pytest.mark.unit
def test_generate_circle_renders_without_goal(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ) as mock_render:
        generate_task_episodes(
            task_name=SyntheticTaskName.CIRCLE.value,
            num_episodes=2,
            seed=42,
            image_size=8,
            trajectory_length=10,
            noise_std=0.0,
        )

    for call in mock_render.call_args_list:
        assert "goal" not in call.kwargs


@pytest.mark.unit
def test_generate_conditional_circle_context_is_one_hot(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    num_modes = CIRCLE_DEFAULT_NUM_MODES
    num_episodes = num_modes * 2
    trajectory_length = 10

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = generate_task_episodes(
            task_name=SyntheticTaskName.CONDITIONAL_CIRCLE.value,
            num_episodes=num_episodes,
            seed=42,
            image_size=8,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )

    assert len(episodes) == num_episodes
    for episode in episodes:
        mode_index = int(episode["mode_id"][0, 0])
        expected_context = np.zeros((trajectory_length, num_modes), dtype=np.float32)
        expected_context[:, mode_index] = 1.0
        np.testing.assert_array_equal(episode["context"], expected_context)


@pytest.mark.unit
def test_generate_conditional_circle_passes_context_color_to_render(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    num_modes = CIRCLE_DEFAULT_NUM_MODES
    num_episodes = num_modes

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ) as mock_render:
        generate_task_episodes(
            task_name=SyntheticTaskName.CONDITIONAL_CIRCLE.value,
            num_episodes=num_episodes,
            seed=42,
            image_size=8,
            trajectory_length=10,
            noise_std=0.0,
        )

    assert mock_render.call_count == num_episodes
    render_context_colors = [
        call.kwargs["context_color"] for call in mock_render.call_args_list
    ]
    expected_colors = [CIRCLE_CONTEXT_COLORS[index] for index in range(num_modes)]
    assert sorted(render_context_colors) == sorted(expected_colors)


@pytest.mark.unit
def test_generate_sequential_decision_renders_without_goal(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    num_episodes = SEQUENTIAL_NUM_COMPOUND_MODES

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ) as mock_render:
        generate_task_episodes(
            task_name=SyntheticTaskName.SEQUENTIAL_DECISION.value,
            num_episodes=num_episodes,
            seed=42,
            image_size=8,
            trajectory_length=60,
            noise_std=0.0,
        )

    assert mock_render.call_count == num_episodes
    for call in mock_render.call_args_list:
        assert "goal" not in call.kwargs
        assert call.kwargs["obstacles"] == SEQUENTIAL_OBSTACLES


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode_index, expected_endpoint_x",
    [
        (
            0,
            float(SEQUENTIAL_START[0])
            - SEQUENTIAL_FIRST_BRANCH_X_DELTA
            - SEQUENTIAL_SECOND_BRANCH_X_DELTA,
        ),
        (
            1,
            float(SEQUENTIAL_START[0])
            - SEQUENTIAL_FIRST_BRANCH_X_DELTA
            + SEQUENTIAL_SECOND_BRANCH_X_DELTA,
        ),
        (
            2,
            float(SEQUENTIAL_START[0])
            + SEQUENTIAL_FIRST_BRANCH_X_DELTA
            - SEQUENTIAL_SECOND_BRANCH_X_DELTA,
        ),
        (
            3,
            float(SEQUENTIAL_START[0])
            + SEQUENTIAL_FIRST_BRANCH_X_DELTA
            + SEQUENTIAL_SECOND_BRANCH_X_DELTA,
        ),
    ],
)
def test_generate_sequential_decision_endpoints_differ(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    mode_index: int,
    expected_endpoint_x: float,
):
    num_episodes = SEQUENTIAL_NUM_COMPOUND_MODES

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = generate_task_episodes(
            task_name=SyntheticTaskName.SEQUENTIAL_DECISION.value,
            num_episodes=num_episodes,
            seed=42,
            image_size=8,
            trajectory_length=60,
            noise_std=0.0,
        )

    matching_episode = next(
        ep for ep in episodes if int(ep["mode_id"][0, 0]) == mode_index
    )
    final_position = matching_episode["position"][-1]
    np.testing.assert_allclose(final_position[0], expected_endpoint_x, atol=1e-4)
    np.testing.assert_allclose(final_position[1], SEQUENTIAL_ENDPOINT_Y, atol=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize("num_modes", [2, 4, 8])
def test_generate_radial_mode_count_matches_k(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    num_modes: int,
):
    num_episodes = num_modes * 2

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = generate_task_episodes(
            task_name=SyntheticTaskName.RADIAL.value,
            num_episodes=num_episodes,
            seed=42,
            image_size=8,
            num_modes=num_modes,
            trajectory_length=30,
            noise_std=0.0,
        )

    mode_ids = {int(ep["mode_id"][0, 0]) for ep in episodes}
    assert mode_ids == set(range(num_modes))


@pytest.mark.unit
def test_generate_radial_trajectory_is_straight_line(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    num_modes = 4
    trajectory_length = 30

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = generate_task_episodes(
            task_name=SyntheticTaskName.RADIAL.value,
            num_episodes=num_modes,
            seed=42,
            image_size=8,
            num_modes=num_modes,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )

    for episode in episodes:
        positions = episode["position"]
        mode_index = int(episode["mode_id"][0, 0])
        angle = 2.0 * np.pi * mode_index / num_modes
        expected_endpoint = np.array(
            [
                float(RADIAL_CENTER[0]) + RADIAL_RADIUS * np.cos(angle),
                float(RADIAL_CENTER[1]) + RADIAL_RADIUS * np.sin(angle),
            ],
            dtype=np.float32,
        )
        direction = expected_endpoint - RADIAL_CENTER
        direction_normalized = direction / np.linalg.norm(direction)
        offsets = positions - RADIAL_CENTER[np.newaxis, :]
        norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        unit_offsets = offsets / norms
        np.testing.assert_allclose(
            unit_offsets[1:],
            np.tile(direction_normalized, (trajectory_length - 1, 1)),
            atol=1e-5,
        )


@pytest.mark.unit
def test_generate_corridor_mode_count_matches_k_times_s(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    num_modes = 4
    num_styles = 2
    total_modes = num_modes * num_styles
    num_episodes = total_modes * 2

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = generate_task_episodes(
            task_name=SyntheticTaskName.CORRIDOR_NAVIGATION.value,
            num_episodes=num_episodes,
            seed=42,
            image_size=8,
            num_modes=num_modes,
            num_styles=num_styles,
            trajectory_length=30,
            noise_std=0.0,
        )

    mode_ids = {int(ep["mode_id"][0, 0]) for ep in episodes}
    assert mode_ids == set(range(total_modes))


@pytest.mark.unit
def test_generate_corridor_even_k_enforced():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "corridor_navigation requires even num_modes so no gap "
            "falls at y=0.5 (BC must collide), got 3"
        ),
    ):
        generate_task_episodes(
            task_name=SyntheticTaskName.CORRIDOR_NAVIGATION.value,
            num_episodes=6,
            seed=42,
            image_size=8,
            num_modes=3,
            trajectory_length=30,
            noise_std=0.0,
        )


@pytest.mark.unit
def test_generate_radial_renders_with_dynamic_obstacles(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    num_modes = 4

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ) as mock_render:
        generate_task_episodes(
            task_name=SyntheticTaskName.RADIAL.value,
            num_episodes=num_modes,
            seed=42,
            image_size=8,
            num_modes=num_modes,
            trajectory_length=10,
            noise_std=0.0,
        )

    for call in mock_render.call_args_list:
        obstacles = call.kwargs["obstacles"]
        assert len(obstacles) == num_modes
        assert "goal" not in call.kwargs


@pytest.mark.unit
def test_generate_corridor_renders_with_dynamic_obstacles(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    num_modes = 4

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ) as mock_render:
        generate_task_episodes(
            task_name=SyntheticTaskName.CORRIDOR_NAVIGATION.value,
            num_episodes=num_modes,
            seed=42,
            image_size=8,
            num_modes=num_modes,
            num_styles=1,
            trajectory_length=10,
            noise_std=0.0,
        )

    for call in mock_render.call_args_list:
        obstacles = call.kwargs["obstacles"]
        assert len(obstacles) == num_modes - 1
        assert "goal" not in call.kwargs


@pytest.mark.unit
def test_generate_corridor_trajectory_passes_through_gap(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    num_modes = 4
    num_styles = 1

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = generate_task_episodes(
            task_name=SyntheticTaskName.CORRIDOR_NAVIGATION.value,
            num_episodes=num_modes,
            seed=42,
            image_size=8,
            num_modes=num_modes,
            num_styles=num_styles,
            trajectory_length=60,
            noise_std=0.0,
        )

    gap_centers = _compute_corridor_gap_centers(num_gaps=num_modes)
    half_gap = _compute_corridor_gap_height(num_gaps=num_modes) / 2.0
    for episode in episodes:
        positions = episode["position"]
        in_wall_region = (positions[:, 0] >= CORRIDOR_WALL_X1) & (
            positions[:, 0] <= CORRIDOR_WALL_X2
        )
        assert in_wall_region.any(), "trajectory must cross the wall region"
        wall_y_values = positions[in_wall_region, 1]
        in_any_gap = any(
            np.any((wall_y_values >= gc - half_gap) & (wall_y_values <= gc + half_gap))
            for gc in gap_centers
        )
        assert in_any_gap, "trajectory must pass through a gap, not through a wall"


@pytest.mark.unit
def test_generate_corridor_styles_produce_different_trajectories(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    num_modes = 2
    num_styles = 3
    total_modes = num_modes * num_styles

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = generate_task_episodes(
            task_name=SyntheticTaskName.CORRIDOR_NAVIGATION.value,
            num_episodes=total_modes,
            seed=42,
            image_size=8,
            num_modes=num_modes,
            num_styles=num_styles,
            trajectory_length=60,
            noise_std=0.0,
        )

    corridor_0_episodes = [
        ep for ep in episodes if int(ep["mode_id"][0, 0]) < num_styles
    ]
    assert len(corridor_0_episodes) == num_styles
    for index in range(1, len(corridor_0_episodes)):
        assert not np.array_equal(
            corridor_0_episodes[0]["position"],
            corridor_0_episodes[index]["position"],
        )


@pytest.mark.unit
def test_parametric_circle_distance_from_center():
    center = np.array([0.5, 0.7], dtype=np.float32)
    radius = 0.2
    positions = _parametric_circle(
        center=center, radius=radius, num_points=60, clockwise=True
    )

    assert positions.shape == (60, 2)
    distances = np.linalg.norm(positions - center[np.newaxis, :], axis=1)
    np.testing.assert_allclose(distances, radius, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("num_gaps", [2, 4, 6])
def test_compute_corridor_gap_centers_evenly_spaced(num_gaps: int):
    gap_centers = _compute_corridor_gap_centers(num_gaps=num_gaps)

    assert len(gap_centers) == num_gaps
    for center in gap_centers:
        assert center != 0.5
    spacings = [
        gap_centers[index + 1] - gap_centers[index]
        for index in range(len(gap_centers) - 1)
    ]
    np.testing.assert_allclose(spacings, spacings[0], atol=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize("num_gaps", [2, 3, 5, 10])
def test_compute_corridor_gap_height_symmetric_half_split(num_gaps: int):
    height = _compute_corridor_gap_height(num_gaps=num_gaps)

    assert height == pytest.approx(1.0 / (num_gaps + 1) / 2.0)


@pytest.mark.unit
def test_trajectory_collides_returns_true_when_inside_obstacle():
    obstacles = [(0.2, 0.2, 0.4, 0.4)]
    trajectory = np.array(
        [[0.1, 0.1], [0.3, 0.3], [0.9, 0.9]],
        dtype=np.float32,
    )

    assert _trajectory_collides(trajectory=trajectory, obstacles=obstacles) is True


@pytest.mark.unit
def test_trajectory_collides_returns_false_when_outside():
    obstacles = [(0.2, 0.2, 0.4, 0.4), (0.6, 0.6, 0.8, 0.8)]
    trajectory = np.array(
        [[0.0, 0.0], [0.1, 0.5], [0.5, 0.1], [1.0, 1.0]],
        dtype=np.float32,
    )

    assert _trajectory_collides(trajectory=trajectory, obstacles=obstacles) is False


@pytest.mark.unit
def test_trajectory_collides_returns_false_with_no_obstacles():
    trajectory = np.array(
        [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]],
        dtype=np.float32,
    )

    assert _trajectory_collides(trajectory=trajectory, obstacles=[]) is False


@pytest.mark.unit
def test_sample_noisy_trajectory_no_collision_returns_clean_trajectory(
    rng: np.random.Generator,
):
    base_trajectory = np.array(
        [[0.05, 0.05], [0.05, 0.5], [0.05, 0.95]],
        dtype=np.float32,
    )
    obstacles = [(0.5, 0.5, 0.9, 0.9)]

    result = _sample_noisy_trajectory_no_collision(
        base_trajectory=base_trajectory,
        obstacles=obstacles,
        noise_std=0.01,
        random_generator=rng,
    )

    assert result.shape == base_trajectory.shape
    assert _trajectory_collides(trajectory=result, obstacles=obstacles) is False


@pytest.mark.unit
def test_sample_noisy_trajectory_no_collision_raises_when_geometry_too_tight(
    rng: np.random.Generator,
):
    base_trajectory = np.array(
        [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        dtype=np.float32,
    )
    obstacles = [(0.0, 0.0, 1.0, 1.0)]

    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"Failed to generate a collision-free trajectory after "
            f"{MAX_TRAJECTORY_RETRIES} attempts (obstacle geometry too tight)."
        ),
    ):
        _sample_noisy_trajectory_no_collision(
            base_trajectory=base_trajectory,
            obstacles=obstacles,
            noise_std=0.01,
            random_generator=rng,
        )


@pytest.mark.unit
@pytest.mark.parametrize("num_modes", [4, 8, 16])
def test_generate_radial_obstacles_count_matches_num_modes(num_modes: int):
    obstacles = _generate_radial_obstacles(num_modes=num_modes, noise_std=0.005)

    assert len(obstacles) == num_modes


@pytest.mark.unit
@pytest.mark.parametrize("num_modes", [4, 8])
def test_generate_radial_obstacles_returns_well_formed_rectangles(num_modes: int):
    obstacles = _generate_radial_obstacles(num_modes=num_modes, noise_std=0.005)

    for x_min, y_min, x_max, y_max in obstacles:
        assert x_min < x_max
        assert y_min < y_max


@pytest.mark.unit
def test_generate_radial_obstacles_returns_empty_when_noise_too_large():
    obstacles = _generate_radial_obstacles(num_modes=4, noise_std=10.0)

    assert obstacles == []


@pytest.mark.unit
@pytest.mark.parametrize("num_modes", [4, 8, 16])
def test_generate_radial_obstacles_inscribed_square_bound(num_modes: int):
    noise_std = 0.005
    midpoint_radius = RADIAL_RADIUS * 0.5
    expected_half_width = (
        midpoint_radius * np.sin(np.pi / num_modes) - 3.0 * noise_std
    ) / np.sqrt(2.0)

    obstacles = _generate_radial_obstacles(num_modes=num_modes, noise_std=noise_std)

    for x_min, _, x_max, _ in obstacles:
        actual_half_width = (x_max - x_min) / 2.0
        np.testing.assert_allclose(actual_half_width, expected_half_width, atol=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("num_gaps", [2, 4, 6])
def test_generate_corridor_obstacles_count_is_gaps_minus_one(num_gaps: int):
    gap_centers = _compute_corridor_gap_centers(num_gaps=num_gaps)
    obstacles = _generate_corridor_obstacles(gap_centers=gap_centers)

    assert len(obstacles) == num_gaps - 1


@pytest.mark.unit
def test_apply_sinusoidal_style_modifies_y_axis():
    positions = np.stack([np.linspace(0.0, 1.0, 30), np.full(30, 0.5)], axis=-1).astype(
        np.float32
    )

    modified = _apply_sinusoidal_style(
        positions=positions,
        style_index=0,
        num_styles=2,
        gap_height=0.2,
    )

    assert modified.shape == positions.shape
    np.testing.assert_array_equal(modified[:, 0], positions[:, 0])
    assert not np.array_equal(modified[:, 1], positions[:, 1])


@pytest.mark.unit
@pytest.mark.parametrize("gap_height", [0.05, 0.1, 0.25])
@pytest.mark.parametrize("num_styles", [1, 2, 4])
def test_apply_sinusoidal_style_amplitude_matches_quarter_gap_scaling(
    gap_height: float,
    num_styles: int,
):
    positions = np.stack([np.linspace(0.0, 1.0, 60), np.full(60, 0.5)], axis=-1).astype(
        np.float32
    )

    modified = _apply_sinusoidal_style(
        positions=positions,
        style_index=0,
        num_styles=num_styles,
        gap_height=gap_height,
    )

    expected_amplitude = min(gap_height / (4.0 * num_styles), gap_height / 2.0)
    max_displacement = float(np.max(np.abs(modified[:, 1] - positions[:, 1])))
    assert max_displacement <= expected_amplitude + 1e-6


@pytest.mark.unit
@pytest.mark.parametrize(
    "task_name, expected_target",
    [
        (
            SyntheticTaskName.CIRCLE.value,
            "_generate_circle",
        ),
        (
            SyntheticTaskName.CONDITIONAL_CIRCLE.value,
            "_generate_conditional_circle",
        ),
        (
            SyntheticTaskName.SEQUENTIAL_DECISION.value,
            "_generate_sequential_decision",
        ),
        (
            SyntheticTaskName.RADIAL.value,
            "_generate_radial",
        ),
        (
            SyntheticTaskName.CORRIDOR_NAVIGATION.value,
            "_generate_corridor_navigation",
        ),
    ],
)
def test_generate_task_episodes_dispatches_to_correct_generator(
    task_name: str,
    expected_target: str,
):
    dispatch_targets = [
        "_generate_circle",
        "_generate_conditional_circle",
        "_generate_sequential_decision",
        "_generate_radial",
        "_generate_corridor_navigation",
    ]
    mock_return_value: list[dict[str, np.ndarray]] = []

    with (
        patch(
            f"versatil.data.synthetic.generators.{dispatch_targets[0]}",
            return_value=mock_return_value,
        ) as mock_circle,
        patch(
            f"versatil.data.synthetic.generators.{dispatch_targets[1]}",
            return_value=mock_return_value,
        ) as mock_conditional_circle,
        patch(
            f"versatil.data.synthetic.generators.{dispatch_targets[2]}",
            return_value=mock_return_value,
        ) as mock_sequential,
        patch(
            f"versatil.data.synthetic.generators.{dispatch_targets[3]}",
            return_value=mock_return_value,
        ) as mock_radial,
        patch(
            f"versatil.data.synthetic.generators.{dispatch_targets[4]}",
            return_value=mock_return_value,
        ) as mock_corridor,
    ):
        result = generate_task_episodes(
            task_name=task_name,
            num_episodes=6,
            seed=7,
            image_size=8,
            num_modes=4,
            trajectory_length=10,
            noise_std=0.01,
            num_styles=2,
        )

    assert result is mock_return_value
    mocks_by_target = {
        dispatch_targets[0]: mock_circle,
        dispatch_targets[1]: mock_conditional_circle,
        dispatch_targets[2]: mock_sequential,
        dispatch_targets[3]: mock_radial,
        dispatch_targets[4]: mock_corridor,
    }
    for target, mock in mocks_by_target.items():
        if target == expected_target:
            assert mock.call_count == 1
            call_kwargs = mock.call_args.kwargs
            assert call_kwargs["num_episodes"] == 6
            assert call_kwargs["image_size"] == 8
            assert call_kwargs["trajectory_length"] == 10
            assert call_kwargs["noise_std"] == 0.01
            assert call_kwargs["mode_weights"] is None
            if target in ("_generate_radial", "_generate_corridor_navigation"):
                assert call_kwargs["num_modes"] == 4
            if target == "_generate_corridor_navigation":
                assert call_kwargs["num_styles"] == 2
        else:
            assert mock.call_count == 0


@pytest.mark.unit
def test_generate_task_episodes_raises_for_unknown_task():
    unknown_name = "not_a_task"

    with pytest.raises(
        ValueError,
        match=re.escape(f"Unknown synthetic task: {unknown_name}"),
    ):
        generate_task_episodes(
            task_name=unknown_name,
            num_episodes=1,
            seed=7,
            image_size=8,
            num_modes=3,
            trajectory_length=10,
            noise_std=0.01,
            num_styles=4,
        )


@pytest.mark.unit
def test_generate_task_episodes_passes_mode_weights(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
):
    weights = [0.8, 0.2]
    num_episodes = 10

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = generate_task_episodes(
            task_name=SyntheticTaskName.CIRCLE.value,
            num_episodes=num_episodes,
            seed=42,
            image_size=8,
            trajectory_length=10,
            noise_std=0.0,
            mode_weights=weights,
        )

    mode_counts = [
        sum(1 for ep in episodes if int(ep["mode_id"][0, 0]) == mode)
        for mode in range(CIRCLE_DEFAULT_NUM_MODES)
    ]
    assert mode_counts == [8, 2]
    assert sum(mode_counts) == num_episodes
