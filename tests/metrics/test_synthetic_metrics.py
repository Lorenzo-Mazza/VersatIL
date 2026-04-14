"""Tests for versatil.metrics.synthetic_metrics module."""

import re
from collections.abc import Callable

import numpy as np
import pytest

from versatil.metrics.synthetic_metrics import (
    _compute_mode_centroids,
    collides_with_obstacles,
    compute_goal_success_rate,
    compute_mode_coverage,
    compute_mode_endpoints,
    compute_success_rate,
)

NUM_TIMESTEPS = 10
POSITION_DIMENSION = 2
NUM_MODES = 3
MODE_CENTERS: list[np.ndarray] = [
    np.array([0.1, 0.1], dtype=np.float32),
    np.array([0.5, 0.5], dtype=np.float32),
    np.array([0.9, 0.9], dtype=np.float32),
]


@pytest.fixture
def position_factory() -> Callable[..., np.ndarray]:
    def factory(x: float, y: float) -> np.ndarray:
        return np.array([x, y], dtype=np.float32)

    return factory


@pytest.fixture
def trajectory_factory(
    rng: np.random.Generator,
) -> Callable[..., np.ndarray]:
    def factory(
        num_trajectories: int = 5,
        num_timesteps: int = NUM_TIMESTEPS,
        center: np.ndarray | None = None,
        noise_scale: float = 0.001,
    ) -> np.ndarray:
        noise = (
            rng.standard_normal(
                (num_trajectories, num_timesteps, POSITION_DIMENSION)
            ).astype(np.float32)
            * noise_scale
        )
        if center is not None:
            return noise + center[np.newaxis, np.newaxis, :]
        return noise

    return factory


@pytest.fixture
def expert_data_factory(
    trajectory_factory: Callable[..., np.ndarray],
) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    def factory(
        episodes_per_mode: int = 10,
        num_modes: int = NUM_MODES,
    ) -> tuple[np.ndarray, np.ndarray]:
        all_trajectories = []
        all_mode_ids = []
        for mode_index in range(num_modes):
            trajectories = trajectory_factory(
                num_trajectories=episodes_per_mode,
                center=MODE_CENTERS[mode_index],
            )
            all_trajectories.append(trajectories)
            all_mode_ids.extend([mode_index] * episodes_per_mode)
        expert_trajectories = np.concatenate(all_trajectories, axis=0)
        expert_mode_ids = np.array(all_mode_ids, dtype=np.int64)
        return expert_trajectories, expert_mode_ids

    return factory


@pytest.fixture
def multi_mode_generated_factory(
    trajectory_factory: Callable[..., np.ndarray],
) -> Callable[..., np.ndarray]:
    def factory(per_mode_counts: list[int]) -> np.ndarray:
        parts = [
            trajectory_factory(num_trajectories=count, center=MODE_CENTERS[mode_index])
            for mode_index, count in enumerate(per_mode_counts)
            if count > 0
        ]
        if not parts:
            return np.zeros((0, NUM_TIMESTEPS, POSITION_DIMENSION), dtype=np.float32)
        return np.concatenate(parts, axis=0)

    return factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "per_mode_counts, expected_coverage",
    [
        ([2, 2, 2], 1.0),
        ([6, 0, 0], 1.0 / NUM_MODES),
        ([3, 3, 0], 2.0 / NUM_MODES),
    ],
)
def test_compute_mode_coverage_matches_covered_mode_fraction(
    expert_data_factory: Callable[..., tuple[np.ndarray, np.ndarray]],
    multi_mode_generated_factory: Callable[..., np.ndarray],
    per_mode_counts: list[int],
    expected_coverage: float,
):
    expert_trajectories, expert_mode_ids = expert_data_factory()
    generated = multi_mode_generated_factory(per_mode_counts=per_mode_counts)

    result = compute_mode_coverage(
        generated_trajectories=generated,
        expert_trajectories=expert_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=NUM_MODES,
    )

    assert result["mode_coverage"] == pytest.approx(expected_coverage)


@pytest.mark.unit
def test_compute_mode_coverage_per_mode_count_sums_to_num_generated(
    expert_data_factory: Callable[..., tuple[np.ndarray, np.ndarray]],
    multi_mode_generated_factory: Callable[..., np.ndarray],
):
    per_mode_counts = [3, 3, 3]
    expert_trajectories, expert_mode_ids = expert_data_factory()
    generated = multi_mode_generated_factory(per_mode_counts=per_mode_counts)

    result = compute_mode_coverage(
        generated_trajectories=generated,
        expert_trajectories=expert_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=NUM_MODES,
    )

    for mode_index, expected_count in enumerate(per_mode_counts):
        assert result["per_mode_count"][mode_index] == expected_count


@pytest.mark.unit
@pytest.mark.parametrize(
    "per_mode_counts, expected_entropy_ratio",
    [
        ([10, 10, 10], 1.0),
        ([10, 0, 0], 0.0),
    ],
)
def test_compute_mode_coverage_entropy_ratio_reflects_mode_distribution(
    expert_data_factory: Callable[..., tuple[np.ndarray, np.ndarray]],
    multi_mode_generated_factory: Callable[..., np.ndarray],
    per_mode_counts: list[int],
    expected_entropy_ratio: float,
):
    expert_trajectories, expert_mode_ids = expert_data_factory()
    generated = multi_mode_generated_factory(per_mode_counts=per_mode_counts)

    result = compute_mode_coverage(
        generated_trajectories=generated,
        expert_trajectories=expert_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=NUM_MODES,
    )

    assert result["mode_entropy_ratio"] == pytest.approx(
        expected_entropy_ratio, abs=0.01
    )


@pytest.mark.unit
def test_compute_mode_coverage_empty_generated_returns_zero_coverage_and_entropy(
    expert_data_factory: Callable[..., tuple[np.ndarray, np.ndarray]],
    multi_mode_generated_factory: Callable[..., np.ndarray],
):
    expert_trajectories, expert_mode_ids = expert_data_factory()
    generated = multi_mode_generated_factory(per_mode_counts=[0, 0, 0])

    result = compute_mode_coverage(
        generated_trajectories=generated,
        expert_trajectories=expert_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=NUM_MODES,
    )

    assert result["mode_coverage"] == 0.0
    assert result["mode_entropy_ratio"] == 0.0


@pytest.mark.unit
def test_compute_mode_coverage_single_mode_returns_full_coverage_zero_entropy(
    trajectory_factory: Callable[..., np.ndarray],
):
    expert_trajectories = trajectory_factory(num_trajectories=3, center=MODE_CENTERS[0])
    expert_mode_ids = np.zeros(3, dtype=np.int64)
    generated = trajectory_factory(num_trajectories=2, center=MODE_CENTERS[0])

    result = compute_mode_coverage(
        generated_trajectories=generated,
        expert_trajectories=expert_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=1,
    )

    assert result["mode_coverage"] == 1.0
    assert result["mode_entropy_ratio"] == 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "center_xy, expected_rate",
    [
        ((1.0, 1.0), 1.0),
        ((0.0, 0.0), 0.0),
    ],
)
def test_compute_goal_success_rate_uniform_trajectories(
    position_factory: Callable[..., np.ndarray],
    trajectory_factory: Callable[..., np.ndarray],
    center_xy: tuple[float, float],
    expected_rate: float,
):
    goal = position_factory(x=1.0, y=1.0)
    center = position_factory(x=center_xy[0], y=center_xy[1])
    trajectories = trajectory_factory(
        num_trajectories=5, center=center, noise_scale=0.0
    )

    rate = compute_goal_success_rate(
        generated_trajectories=trajectories,
        goal=goal,
        threshold=0.05,
    )

    assert rate == expected_rate


@pytest.mark.unit
def test_compute_goal_success_rate_partial(
    position_factory: Callable[..., np.ndarray],
    trajectory_factory: Callable[..., np.ndarray],
):
    goal = position_factory(x=1.0, y=1.0)
    far_center = position_factory(x=0.0, y=0.0)
    at_goal = trajectory_factory(num_trajectories=2, center=goal, noise_scale=0.0)
    far_away = trajectory_factory(
        num_trajectories=3, center=far_center, noise_scale=0.0
    )
    trajectories = np.concatenate([at_goal, far_away], axis=0)

    rate = compute_goal_success_rate(
        generated_trajectories=trajectories,
        goal=goal,
        threshold=0.05,
    )

    assert rate == pytest.approx(2.0 / 5.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "threshold, expected_rate",
    [
        (0.01, 0.0),
        (0.2, 1.0),
    ],
)
def test_compute_goal_success_rate_threshold_sensitivity(
    position_factory: Callable[..., np.ndarray],
    trajectory_factory: Callable[..., np.ndarray],
    threshold: float,
    expected_rate: float,
):
    goal = position_factory(x=1.0, y=1.0)
    # Place all final positions at L2 distance 0.1 from the goal so the
    # outcome changes as the threshold crosses 0.1.
    near_goal = position_factory(x=1.0 + 0.1 / np.sqrt(2), y=1.0 + 0.1 / np.sqrt(2))
    trajectories = trajectory_factory(
        num_trajectories=4, center=near_goal, noise_scale=0.0
    )

    rate = compute_goal_success_rate(
        generated_trajectories=trajectories,
        goal=goal,
        threshold=threshold,
    )

    assert rate == pytest.approx(expected_rate)


@pytest.mark.unit
def test_compute_mode_coverage_excludes_invalid_trajectories(
    expert_data_factory: Callable[..., tuple[np.ndarray, np.ndarray]],
    multi_mode_generated_factory: Callable[..., np.ndarray],
):
    # Generate 2 trajectories per mode across 3 modes (6 total). Mark the
    # trajectories for mode 2 as invalid — coverage should drop to 2/3 and
    # per_mode_count[2] should be 0.
    per_mode_counts = [2, 2, 2]
    expert_trajectories, expert_mode_ids = expert_data_factory()
    generated = multi_mode_generated_factory(per_mode_counts=per_mode_counts)
    valid_mask = np.array([True, True, True, True, False, False])

    result = compute_mode_coverage(
        generated_trajectories=generated,
        expert_trajectories=expert_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=NUM_MODES,
        valid_mask=valid_mask,
    )

    assert result["per_mode_count"][2] == 0
    assert result["mode_coverage"] == pytest.approx(2.0 / NUM_MODES)


@pytest.mark.unit
def test_compute_mode_coverage_all_invalid_returns_zero(
    expert_data_factory: Callable[..., tuple[np.ndarray, np.ndarray]],
    multi_mode_generated_factory: Callable[..., np.ndarray],
):
    per_mode_counts = [2, 2, 2]
    expert_trajectories, expert_mode_ids = expert_data_factory()
    generated = multi_mode_generated_factory(per_mode_counts=per_mode_counts)
    valid_mask = np.zeros(sum(per_mode_counts), dtype=bool)

    result = compute_mode_coverage(
        generated_trajectories=generated,
        expert_trajectories=expert_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=NUM_MODES,
        valid_mask=valid_mask,
    )

    assert result["mode_coverage"] == 0.0
    assert result["mode_entropy_ratio"] == 0.0


@pytest.mark.unit
def test_collides_with_obstacles_detects_trajectory_inside_rectangle(
    trajectory_factory: Callable[..., np.ndarray],
    position_factory: Callable[..., np.ndarray],
):
    inside_center = position_factory(x=0.5, y=0.5)
    outside_center = position_factory(x=0.1, y=0.1)
    inside = trajectory_factory(
        num_trajectories=1, center=inside_center, noise_scale=0.0
    )
    outside = trajectory_factory(
        num_trajectories=1, center=outside_center, noise_scale=0.0
    )
    trajectories = np.concatenate([inside, outside], axis=0)
    obstacles = [(0.4, 0.4, 0.6, 0.6)]

    collided = collides_with_obstacles(trajectories=trajectories, obstacles=obstacles)

    assert collided.tolist() == [True, False]


@pytest.mark.unit
def test_collides_with_obstacles_no_obstacles_returns_all_false(
    trajectory_factory: Callable[..., np.ndarray],
    position_factory: Callable[..., np.ndarray],
):
    center = position_factory(x=0.5, y=0.5)
    trajectories = trajectory_factory(
        num_trajectories=3, center=center, noise_scale=0.0
    )

    collided = collides_with_obstacles(trajectories=trajectories, obstacles=[])

    assert collided.tolist() == [False, False, False]


@pytest.mark.unit
def test_compute_success_rate_no_obstacles_and_reached_endpoint(
    position_factory: Callable[..., np.ndarray],
    trajectory_factory: Callable[..., np.ndarray],
):
    endpoint = position_factory(x=0.9, y=0.9)
    trajectories = trajectory_factory(
        num_trajectories=4, center=endpoint, noise_scale=0.0
    )
    mode_endpoints = np.array([endpoint], dtype=np.float32)

    stats = compute_success_rate(
        generated_trajectories=trajectories,
        obstacles=[],
        mode_endpoints=mode_endpoints,
        goal_threshold=0.1,
    )

    assert stats["success_rate"] == pytest.approx(1.0)
    assert stats["collision_rate"] == pytest.approx(0.0)
    assert stats["endpoint_reach_rate"] == pytest.approx(1.0)


@pytest.mark.unit
def test_compute_success_rate_collision_blocks_success(
    position_factory: Callable[..., np.ndarray],
    trajectory_factory: Callable[..., np.ndarray],
):
    obstacle = (0.4, 0.4, 0.6, 0.6)
    endpoint = position_factory(x=0.5, y=0.5)  # Inside obstacle
    trajectories = trajectory_factory(
        num_trajectories=3, center=endpoint, noise_scale=0.0
    )
    mode_endpoints = np.array([endpoint], dtype=np.float32)

    stats = compute_success_rate(
        generated_trajectories=trajectories,
        obstacles=[obstacle],
        mode_endpoints=mode_endpoints,
        goal_threshold=0.1,
    )

    assert stats["collision_rate"] == pytest.approx(1.0)
    assert stats["endpoint_reach_rate"] == pytest.approx(1.0)
    assert stats["success_rate"] == pytest.approx(0.0)


@pytest.mark.unit
def test_compute_success_rate_missing_endpoint_blocks_success(
    position_factory: Callable[..., np.ndarray],
    trajectory_factory: Callable[..., np.ndarray],
):
    endpoint = position_factory(x=0.9, y=0.9)
    far_center = position_factory(x=0.0, y=0.0)
    trajectories = trajectory_factory(
        num_trajectories=2, center=far_center, noise_scale=0.0
    )
    mode_endpoints = np.array([endpoint], dtype=np.float32)

    stats = compute_success_rate(
        generated_trajectories=trajectories,
        obstacles=[],
        mode_endpoints=mode_endpoints,
        goal_threshold=0.1,
    )

    assert stats["endpoint_reach_rate"] == pytest.approx(0.0)
    assert stats["success_rate"] == pytest.approx(0.0)


@pytest.mark.unit
def test_compute_success_rate_accepts_any_mode_endpoint(
    position_factory: Callable[..., np.ndarray],
    trajectory_factory: Callable[..., np.ndarray],
):
    endpoint_a = position_factory(x=0.1, y=0.1)
    endpoint_b = position_factory(x=0.9, y=0.9)
    trajectories = trajectory_factory(
        num_trajectories=4, center=endpoint_b, noise_scale=0.0
    )
    mode_endpoints = np.stack([endpoint_a, endpoint_b], axis=0)

    stats = compute_success_rate(
        generated_trajectories=trajectories,
        obstacles=[],
        mode_endpoints=mode_endpoints,
        goal_threshold=0.1,
    )

    assert stats["success_rate"] == pytest.approx(1.0)


@pytest.mark.unit
def test_compute_success_rate_stationary_trajectory_fails_path_length_check(
    position_factory: Callable[..., np.ndarray],
):
    # Simulate a "closed loop" task: start == endpoint at (0.5, 0.5).
    # A policy that stays still trivially reaches the endpoint but has zero
    # path length, so it must NOT be counted as success.
    start = position_factory(x=0.5, y=0.5)
    trajectories = np.broadcast_to(start, (3, 10, 2)).astype(np.float32).copy()
    mode_endpoints = np.array([start], dtype=np.float32)

    stats = compute_success_rate(
        generated_trajectories=trajectories,
        obstacles=[],
        mode_endpoints=mode_endpoints,
        goal_threshold=0.1,
        min_path_length=0.5,
    )

    assert stats["endpoint_reach_rate"] == pytest.approx(1.0)
    assert stats["path_length_rate"] == pytest.approx(0.0)
    assert stats["success_rate"] == pytest.approx(0.0)


@pytest.mark.unit
def test_compute_success_rate_moving_trajectory_passes_path_length_check(
    position_factory: Callable[..., np.ndarray],
):
    # Trajectory traces a straight line from (0.0, 0.5) to (1.0, 0.5) in 10
    # steps — path length == 1.0 — then final point matches the endpoint.
    num_steps = 10
    trajectory = np.zeros((1, num_steps, 2), dtype=np.float32)
    trajectory[0, :, 0] = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
    trajectory[0, :, 1] = 0.5
    endpoint = position_factory(x=1.0, y=0.5)
    mode_endpoints = np.array([endpoint], dtype=np.float32)

    stats = compute_success_rate(
        generated_trajectories=trajectory,
        obstacles=[],
        mode_endpoints=mode_endpoints,
        goal_threshold=0.05,
        min_path_length=0.5,
    )

    assert stats["path_length_rate"] == pytest.approx(1.0)
    assert stats["success_rate"] == pytest.approx(1.0)


@pytest.mark.unit
def test_compute_success_rate_default_min_path_length_is_zero(
    position_factory: Callable[..., np.ndarray],
):
    # Default min_path_length=0 keeps backward-compatible behavior: a
    # stationary trajectory at the endpoint counts as success.
    start = position_factory(x=0.5, y=0.5)
    trajectories = np.broadcast_to(start, (2, 5, 2)).astype(np.float32).copy()
    mode_endpoints = np.array([start], dtype=np.float32)

    stats = compute_success_rate(
        generated_trajectories=trajectories,
        obstacles=[],
        mode_endpoints=mode_endpoints,
        goal_threshold=0.1,
    )

    assert stats["path_length_rate"] == pytest.approx(1.0)
    assert stats["success_rate"] == pytest.approx(1.0)


@pytest.mark.unit
def test_compute_mode_endpoints_returns_mean_final_position_per_mode(
    trajectory_factory: Callable[..., np.ndarray],
):
    mode_0 = trajectory_factory(num_trajectories=4, center=MODE_CENTERS[0])
    mode_1 = trajectory_factory(num_trajectories=4, center=MODE_CENTERS[1])
    expert_trajectories = np.concatenate([mode_0, mode_1], axis=0)
    expert_mode_ids = np.concatenate(
        [np.zeros(4, dtype=np.int64), np.ones(4, dtype=np.int64)]
    )

    endpoints = compute_mode_endpoints(
        expert_trajectories=expert_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=2,
    )

    assert endpoints.shape == (2, 2)
    np.testing.assert_allclose(endpoints[0], mode_0[:, -1, :].mean(axis=0))
    np.testing.assert_allclose(endpoints[1], mode_1[:, -1, :].mean(axis=0))


@pytest.mark.unit
def test_compute_mode_endpoints_missing_mode_raises(
    trajectory_factory: Callable[..., np.ndarray],
):
    expert_trajectories = trajectory_factory(num_trajectories=4, center=MODE_CENTERS[0])
    expert_mode_ids = np.zeros(4, dtype=np.int64)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "No expert trajectories for mode 1; "
            "expected all 2 modes represented in expert data."
        ),
    ):
        compute_mode_endpoints(
            expert_trajectories=expert_trajectories,
            expert_mode_ids=expert_mode_ids,
            num_modes=2,
        )


@pytest.mark.unit
def test_compute_mode_centroids_returns_mean_trajectory_per_mode(
    position_factory: Callable[..., np.ndarray],
    trajectory_factory: Callable[..., np.ndarray],
):
    center = position_factory(x=0.3, y=0.7)
    mode_trajectories = trajectory_factory(
        num_trajectories=2, center=center, noise_scale=0.0
    )
    expert_mode_ids = np.zeros(2, dtype=np.int64)

    centroids = _compute_mode_centroids(
        expert_trajectories=mode_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=1,
    )

    np.testing.assert_allclose(centroids[0], np.mean(mode_trajectories, axis=0))


@pytest.mark.unit
def test_compute_mode_centroids_empty_mode_returns_zeros(
    trajectory_factory: Callable[..., np.ndarray],
):
    expert_trajectories = trajectory_factory(num_trajectories=3, center=MODE_CENTERS[0])
    expert_mode_ids = np.zeros(3, dtype=np.int64)

    centroids = _compute_mode_centroids(
        expert_trajectories=expert_trajectories,
        expert_mode_ids=expert_mode_ids,
        num_modes=2,
    )

    np.testing.assert_array_equal(
        centroids[1],
        np.zeros((NUM_TIMESTEPS, POSITION_DIMENSION), dtype=np.float32),
    )
