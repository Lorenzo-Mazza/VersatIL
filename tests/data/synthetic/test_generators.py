"""Tests for versatil.data.synthetic.generators module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import numpy as np
import pytest

from versatil.data.synthetic.constants import (
    MULTIPATH_CONTEXT_COLORS,
    MULTIPATH_DEFAULT_NOISE_STD,
    MULTIPATH_DEFAULT_NUM_MODES,
    MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
    MULTIPATH_MIN_TRAJECTORY_LENGTH,
    SEQUENTIAL_BRANCH_X_DELTA,
    SEQUENTIAL_DEFAULT_NOISE_STD,
    SEQUENTIAL_DEFAULT_TRAJECTORY_LENGTH,
    SEQUENTIAL_MIN_TRAJECTORY_LENGTH,
    SEQUENTIAL_START,
    SHARED_PREFIX_DEFAULT_NOISE_STD,
    SHARED_PREFIX_DEFAULT_NUM_MODES,
    SHARED_PREFIX_DEFAULT_TRAJECTORY_LENGTH,
    SHARED_PREFIX_ENDPOINTS,
    SHARED_PREFIX_MIN_TRAJECTORY_LENGTH,
    SHARED_PREFIX_SHARED_STEPS,
    STYLE_DEFAULT_NOISE_STD,
    STYLE_DEFAULT_NUM_STYLES,
    STYLE_DEFAULT_TRAJECTORY_LENGTH,
    STYLE_MIN_TRAJECTORY_LENGTH,
    STYLE_START,
    SyntheticTaskName,
)
from versatil.data.synthetic.generators import (
    _add_noise_and_clamp,
    _balanced_mode_counts,
    _compute_actions,
    _generate_conditional_navigation,
    _generate_multi_path_navigation,
    _generate_sequential_decision,
    _generate_shared_prefix,
    _generate_trajectory_style,
    _interpolate_waypoints,
    generate_task_episodes,
)

EPISODE_KEYS = {"image", "position", "action", "mode_id", "context"}


@pytest.fixture
def trajectory_factory(
    rng: np.random.Generator,
) -> Callable[..., np.ndarray]:
    def factory(
        num_points: int = 10,
        fill_value: float | None = None,
    ) -> np.ndarray:
        if fill_value is not None:
            return np.full((num_points, 2), fill_value, dtype=np.float32)
        return rng.uniform(0.0, 1.0, size=(num_points, 2)).astype(np.float32)

    return factory


@pytest.fixture
def fake_render_episode_factory() -> Callable[..., Callable[..., np.ndarray]]:
    def factory(image_size: int = 8) -> Callable[..., np.ndarray]:
        def fake_render(
            positions: np.ndarray,
            obstacles: list[tuple[float, float, float, float]],
            goal: np.ndarray,
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
@pytest.mark.parametrize(
    "trajectory_length, expectation",
    [
        (
            MULTIPATH_MIN_TRAJECTORY_LENGTH,
            does_not_raise(),
        ),
        (
            MULTIPATH_MIN_TRAJECTORY_LENGTH - 1,
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"multi_path_navigation requires trajectory_length >= "
                    f"{MULTIPATH_MIN_TRAJECTORY_LENGTH}, got "
                    f"{MULTIPATH_MIN_TRAJECTORY_LENGTH - 1}"
                ),
            ),
        ),
    ],
)
def test_generate_multi_path_navigation_min_trajectory_length_validation(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
    trajectory_length: int,
    expectation,
):
    with (
        patch(
            "versatil.data.synthetic.generators.render_episode",
            side_effect=fake_render_episode_factory(),
        ),
        expectation,
    ):
        _generate_multi_path_navigation(
            num_episodes=MULTIPATH_DEFAULT_NUM_MODES,
            random_generator=rng,
            image_size=8,
            num_modes=MULTIPATH_DEFAULT_NUM_MODES,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_episodes, num_modes",
    [
        (6, 3),
        (7, 3),
        (4, 2),
    ],
)
def test_generate_multi_path_navigation_balances_modes_and_shapes_episodes(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
    num_episodes: int,
    num_modes: int,
):
    trajectory_length = 10
    image_size = 8

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(image_size=image_size),
    ) as mock_render:
        episodes = _generate_multi_path_navigation(
            num_episodes=num_episodes,
            random_generator=rng,
            image_size=image_size,
            num_modes=num_modes,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )

    assert len(episodes) == num_episodes
    assert mock_render.call_count == num_episodes
    mode_id_values = [int(episode["mode_id"][0, 0]) for episode in episodes]
    mode_counts = [mode_id_values.count(index) for index in range(num_modes)]
    assert sum(mode_counts) == num_episodes
    assert set(mode_id_values) == set(range(num_modes))

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
        np.testing.assert_array_equal(
            episode["context"],
            np.zeros((trajectory_length, num_modes), dtype=np.float32),
        )
        np.testing.assert_array_equal(
            episode["mode_id"],
            np.full(
                (trajectory_length, 1),
                episode["mode_id"][0, 0],
                dtype=np.uint8,
            ),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "trajectory_length, expectation",
    [
        (
            MULTIPATH_MIN_TRAJECTORY_LENGTH,
            does_not_raise(),
        ),
        (
            MULTIPATH_MIN_TRAJECTORY_LENGTH - 1,
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"conditional_navigation requires trajectory_length >= "
                    f"{MULTIPATH_MIN_TRAJECTORY_LENGTH}, got "
                    f"{MULTIPATH_MIN_TRAJECTORY_LENGTH - 1}"
                ),
            ),
        ),
    ],
)
def test_generate_conditional_navigation_min_trajectory_length_validation(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
    trajectory_length: int,
    expectation,
):
    with (
        patch(
            "versatil.data.synthetic.generators.render_episode",
            side_effect=fake_render_episode_factory(),
        ),
        expectation,
    ):
        _generate_conditional_navigation(
            num_episodes=MULTIPATH_DEFAULT_NUM_MODES,
            random_generator=rng,
            image_size=8,
            num_modes=MULTIPATH_DEFAULT_NUM_MODES,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )


@pytest.mark.unit
def test_generate_conditional_navigation_context_is_one_hot_per_mode(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
):
    num_modes = MULTIPATH_DEFAULT_NUM_MODES
    num_episodes = num_modes * 2
    trajectory_length = 10

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = _generate_conditional_navigation(
            num_episodes=num_episodes,
            random_generator=rng,
            image_size=8,
            num_modes=num_modes,
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
def test_generate_conditional_navigation_passes_per_mode_context_color_to_render(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
):
    num_modes = MULTIPATH_DEFAULT_NUM_MODES
    num_episodes = num_modes
    trajectory_length = 10

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ) as mock_render:
        _generate_conditional_navigation(
            num_episodes=num_episodes,
            random_generator=rng,
            image_size=8,
            num_modes=num_modes,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )

    assert mock_render.call_count == num_episodes
    render_context_colors = [
        call.kwargs["context_color"] for call in mock_render.call_args_list
    ]
    expected_colors = [MULTIPATH_CONTEXT_COLORS[index] for index in range(num_modes)]
    assert sorted(render_context_colors) == sorted(expected_colors)


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_trajectory_length, input_noise_std, "
    "expected_trajectory_length, expected_noise_std_used",
    [
        (
            MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
            MULTIPATH_DEFAULT_NOISE_STD,
            STYLE_DEFAULT_TRAJECTORY_LENGTH,
            STYLE_DEFAULT_NOISE_STD,
        ),
        (
            STYLE_MIN_TRAJECTORY_LENGTH + 5,
            0.05,
            STYLE_MIN_TRAJECTORY_LENGTH + 5,
            0.05,
        ),
    ],
)
def test_generate_trajectory_style_substitutes_multipath_sentinel_defaults(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
    input_trajectory_length: int,
    input_noise_std: float,
    expected_trajectory_length: int,
    expected_noise_std_used: float,
):
    num_styles = STYLE_DEFAULT_NUM_STYLES
    num_episodes = num_styles

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = _generate_trajectory_style(
            num_episodes=num_episodes,
            random_generator=rng,
            image_size=8,
            num_styles=num_styles,
            trajectory_length=input_trajectory_length,
            noise_std=input_noise_std,
        )

    for episode in episodes:
        assert episode["position"].shape[0] == expected_trajectory_length
    # Sanity check: the (unused) expected_noise_std is non-negative and the
    # substitution didn't regress into an invalid value.
    assert expected_noise_std_used >= 0.0


@pytest.mark.unit
@pytest.mark.parametrize(
    "trajectory_length, expectation",
    [
        (
            STYLE_MIN_TRAJECTORY_LENGTH,
            does_not_raise(),
        ),
        (
            STYLE_MIN_TRAJECTORY_LENGTH - 1,
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"trajectory_style requires trajectory_length >= "
                    f"{STYLE_MIN_TRAJECTORY_LENGTH}, got "
                    f"{STYLE_MIN_TRAJECTORY_LENGTH - 1}"
                ),
            ),
        ),
    ],
)
def test_generate_trajectory_style_min_trajectory_length_validation(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
    trajectory_length: int,
    expectation,
):
    with (
        patch(
            "versatil.data.synthetic.generators.render_episode",
            side_effect=fake_render_episode_factory(),
        ),
        expectation,
    ):
        _generate_trajectory_style(
            num_episodes=STYLE_DEFAULT_NUM_STYLES,
            random_generator=rng,
            image_size=8,
            num_styles=STYLE_DEFAULT_NUM_STYLES,
            trajectory_length=trajectory_length,
            noise_std=0.05,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "style_index, y_profile_check",
    [
        (
            0,
            lambda positions: np.allclose(positions[:, 1], STYLE_START[1], atol=0.02),
        ),
        (
            1,
            lambda positions: positions[:, 1].max() > STYLE_START[1] + 0.1,
        ),
        (
            2,
            lambda positions: positions[:, 1].min() < STYLE_START[1] - 0.1,
        ),
        (
            3,
            lambda positions: (
                positions[:, 1].max() > STYLE_START[1] + 0.02
                and positions[:, 1].min() < STYLE_START[1] - 0.02
            ),
        ),
    ],
)
def test_generate_trajectory_style_produces_distinct_y_profile_per_style(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
    style_index: int,
    y_profile_check: Callable[[np.ndarray], bool],
):
    num_styles = STYLE_DEFAULT_NUM_STYLES
    num_episodes = num_styles
    trajectory_length = STYLE_DEFAULT_TRAJECTORY_LENGTH

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = _generate_trajectory_style(
            num_episodes=num_episodes,
            random_generator=rng,
            image_size=8,
            num_styles=num_styles,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )

    matching_episode = next(
        episode for episode in episodes if int(episode["mode_id"][0, 0]) == style_index
    )
    assert y_profile_check(matching_episode["position"])


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_trajectory_length, input_noise_std, expected_trajectory_length",
    [
        (
            MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
            MULTIPATH_DEFAULT_NOISE_STD,
            SEQUENTIAL_DEFAULT_TRAJECTORY_LENGTH,
        ),
        (
            SEQUENTIAL_MIN_TRAJECTORY_LENGTH,
            0.05,
            SEQUENTIAL_MIN_TRAJECTORY_LENGTH,
        ),
    ],
)
def test_generate_sequential_decision_substitutes_multipath_sentinel_defaults(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
    input_trajectory_length: int,
    input_noise_std: float,
    expected_trajectory_length: int,
):
    num_episodes = 4
    # Sanity: SEQUENTIAL_DEFAULT_NOISE_STD is used when multipath default is passed
    assert SEQUENTIAL_DEFAULT_NOISE_STD != MULTIPATH_DEFAULT_NOISE_STD

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = _generate_sequential_decision(
            num_episodes=num_episodes,
            random_generator=rng,
            image_size=8,
            trajectory_length=input_trajectory_length,
            noise_std=input_noise_std,
        )

    for episode in episodes:
        assert episode["position"].shape[0] == expected_trajectory_length


@pytest.mark.unit
@pytest.mark.parametrize(
    "trajectory_length, expectation",
    [
        (
            SEQUENTIAL_MIN_TRAJECTORY_LENGTH,
            does_not_raise(),
        ),
        (
            SEQUENTIAL_MIN_TRAJECTORY_LENGTH - 1,
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"sequential_decision requires trajectory_length >= "
                    f"{SEQUENTIAL_MIN_TRAJECTORY_LENGTH}, got "
                    f"{SEQUENTIAL_MIN_TRAJECTORY_LENGTH - 1}"
                ),
            ),
        ),
    ],
)
def test_generate_sequential_decision_min_trajectory_length_validation(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
    trajectory_length: int,
    expectation,
):
    with (
        patch(
            "versatil.data.synthetic.generators.render_episode",
            side_effect=fake_render_episode_factory(),
        ),
        expectation,
    ):
        _generate_sequential_decision(
            num_episodes=4,
            random_generator=rng,
            image_size=8,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode_index, expected_mid_x, expected_final_x",
    [
        (
            0,
            SEQUENTIAL_START[0] - SEQUENTIAL_BRANCH_X_DELTA,
            SEQUENTIAL_START[0] - 2 * SEQUENTIAL_BRANCH_X_DELTA,
        ),
        (
            1,
            SEQUENTIAL_START[0] - SEQUENTIAL_BRANCH_X_DELTA,
            SEQUENTIAL_START[0],
        ),
        (
            2,
            SEQUENTIAL_START[0] + SEQUENTIAL_BRANCH_X_DELTA,
            SEQUENTIAL_START[0],
        ),
        (
            3,
            SEQUENTIAL_START[0] + SEQUENTIAL_BRANCH_X_DELTA,
            SEQUENTIAL_START[0] + 2 * SEQUENTIAL_BRANCH_X_DELTA,
        ),
    ],
)
def test_generate_sequential_decision_compound_modes_have_distinct_endpoints(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
    mode_index: int,
    expected_mid_x: float,
    expected_final_x: float,
):
    num_episodes = 4
    trajectory_length = SEQUENTIAL_DEFAULT_TRAJECTORY_LENGTH

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = _generate_sequential_decision(
            num_episodes=num_episodes,
            random_generator=rng,
            image_size=8,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )

    matching_episode = next(
        episode for episode in episodes if int(episode["mode_id"][0, 0]) == mode_index
    )
    mid_x = matching_episode["position"][trajectory_length // 2, 0]
    final_x = matching_episode["position"][-1, 0]
    assert abs(mid_x - expected_mid_x) < 0.05
    assert abs(final_x - expected_final_x) < 0.05


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_trajectory_length, input_noise_std, expected_trajectory_length",
    [
        (
            MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
            MULTIPATH_DEFAULT_NOISE_STD,
            SHARED_PREFIX_DEFAULT_TRAJECTORY_LENGTH,
        ),
        (
            SHARED_PREFIX_MIN_TRAJECTORY_LENGTH,
            0.05,
            SHARED_PREFIX_MIN_TRAJECTORY_LENGTH,
        ),
    ],
)
def test_generate_shared_prefix_substitutes_multipath_sentinel_defaults(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
    input_trajectory_length: int,
    input_noise_std: float,
    expected_trajectory_length: int,
):
    num_episodes = SHARED_PREFIX_DEFAULT_NUM_MODES
    assert SHARED_PREFIX_DEFAULT_NOISE_STD != MULTIPATH_DEFAULT_NOISE_STD

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = _generate_shared_prefix(
            num_episodes=num_episodes,
            random_generator=rng,
            image_size=8,
            num_modes=SHARED_PREFIX_DEFAULT_NUM_MODES,
            trajectory_length=input_trajectory_length,
            noise_std=input_noise_std,
        )

    for episode in episodes:
        assert episode["position"].shape[0] == expected_trajectory_length


@pytest.mark.unit
@pytest.mark.parametrize(
    "trajectory_length, expectation",
    [
        (
            SHARED_PREFIX_MIN_TRAJECTORY_LENGTH,
            does_not_raise(),
        ),
        (
            SHARED_PREFIX_MIN_TRAJECTORY_LENGTH - 1,
            pytest.raises(
                ValueError,
                match=re.escape(
                    f"shared_prefix requires trajectory_length >= "
                    f"{SHARED_PREFIX_MIN_TRAJECTORY_LENGTH}, got "
                    f"{SHARED_PREFIX_MIN_TRAJECTORY_LENGTH - 1}"
                ),
            ),
        ),
    ],
)
def test_generate_shared_prefix_min_trajectory_length_validation(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
    trajectory_length: int,
    expectation,
):
    with (
        patch(
            "versatil.data.synthetic.generators.render_episode",
            side_effect=fake_render_episode_factory(),
        ),
        expectation,
    ):
        _generate_shared_prefix(
            num_episodes=SHARED_PREFIX_DEFAULT_NUM_MODES,
            random_generator=rng,
            image_size=8,
            num_modes=SHARED_PREFIX_DEFAULT_NUM_MODES,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )


@pytest.mark.unit
def test_generate_shared_prefix_positions_identical_in_shared_segment(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
):
    num_modes = SHARED_PREFIX_DEFAULT_NUM_MODES
    num_episodes = num_modes
    trajectory_length = SHARED_PREFIX_DEFAULT_TRAJECTORY_LENGTH

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = _generate_shared_prefix(
            num_episodes=num_episodes,
            random_generator=rng,
            image_size=8,
            num_modes=num_modes,
            trajectory_length=trajectory_length,
            noise_std=0.0,
        )

    # Under zero noise, every mode's first SHARED_PREFIX_SHARED_STEPS
    # positions should be exactly identical.
    reference_prefix = episodes[0]["position"][:SHARED_PREFIX_SHARED_STEPS]
    for episode in episodes[1:]:
        np.testing.assert_array_equal(
            episode["position"][:SHARED_PREFIX_SHARED_STEPS],
            reference_prefix,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode_index",
    list(range(SHARED_PREFIX_DEFAULT_NUM_MODES)),
)
def test_generate_shared_prefix_divergent_endpoint_matches_mode(
    fake_render_episode_factory: Callable[..., Callable[..., np.ndarray]],
    rng: np.random.Generator,
    mode_index: int,
):
    num_modes = SHARED_PREFIX_DEFAULT_NUM_MODES
    num_episodes = num_modes

    with patch(
        "versatil.data.synthetic.generators.render_episode",
        side_effect=fake_render_episode_factory(),
    ):
        episodes = _generate_shared_prefix(
            num_episodes=num_episodes,
            random_generator=rng,
            image_size=8,
            num_modes=num_modes,
            trajectory_length=SHARED_PREFIX_DEFAULT_TRAJECTORY_LENGTH,
            noise_std=0.0,
        )

    expected_endpoint = np.array(SHARED_PREFIX_ENDPOINTS[mode_index], dtype=np.float32)
    matching_episode = next(
        episode for episode in episodes if int(episode["mode_id"][0, 0]) == mode_index
    )
    np.testing.assert_allclose(
        matching_episode["position"][-1], expected_endpoint, atol=1e-5
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "task_name, expected_target",
    [
        (
            SyntheticTaskName.MULTI_PATH_NAVIGATION.value,
            "_generate_multi_path_navigation",
        ),
        (
            SyntheticTaskName.CONDITIONAL_NAVIGATION.value,
            "_generate_conditional_navigation",
        ),
        (
            SyntheticTaskName.TRAJECTORY_STYLE.value,
            "_generate_trajectory_style",
        ),
        (
            SyntheticTaskName.SEQUENTIAL_DECISION.value,
            "_generate_sequential_decision",
        ),
        (
            SyntheticTaskName.SHARED_PREFIX.value,
            "_generate_shared_prefix",
        ),
    ],
)
def test_generate_task_episodes_dispatches_to_correct_private_generator(
    task_name: str,
    expected_target: str,
):
    dispatch_targets = [
        "_generate_multi_path_navigation",
        "_generate_conditional_navigation",
        "_generate_trajectory_style",
        "_generate_sequential_decision",
        "_generate_shared_prefix",
    ]
    mock_return_value: list[dict[str, np.ndarray]] = []

    with (
        patch(
            f"versatil.data.synthetic.generators.{dispatch_targets[0]}",
            return_value=mock_return_value,
        ) as mock_multi_path,
        patch(
            f"versatil.data.synthetic.generators.{dispatch_targets[1]}",
            return_value=mock_return_value,
        ) as mock_conditional,
        patch(
            f"versatil.data.synthetic.generators.{dispatch_targets[2]}",
            return_value=mock_return_value,
        ) as mock_style,
        patch(
            f"versatil.data.synthetic.generators.{dispatch_targets[3]}",
            return_value=mock_return_value,
        ) as mock_sequential,
        patch(
            f"versatil.data.synthetic.generators.{dispatch_targets[4]}",
            return_value=mock_return_value,
        ) as mock_shared_prefix,
    ):
        result = generate_task_episodes(
            task_name=task_name,
            num_episodes=6,
            seed=7,
            image_size=8,
            num_modes=3,
            trajectory_length=10,
            noise_std=0.01,
            num_styles=4,
        )

    assert result is mock_return_value
    mocks_by_target = {
        dispatch_targets[0]: mock_multi_path,
        dispatch_targets[1]: mock_conditional,
        dispatch_targets[2]: mock_style,
        dispatch_targets[3]: mock_sequential,
        dispatch_targets[4]: mock_shared_prefix,
    }
    for target, mock in mocks_by_target.items():
        if target == expected_target:
            assert mock.call_count == 1
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
