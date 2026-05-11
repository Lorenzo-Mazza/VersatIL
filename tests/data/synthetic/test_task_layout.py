"""Tests for versatil.data.synthetic.task_layout module."""

import re
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import numpy as np
import pytest

from versatil.data.synthetic.constants import (
    CIRCLE_DEFAULT_NUM_MODES,
    CIRCLE_OBSTACLES,
    CIRCLE_START,
    CORRIDOR_DEFAULT_NUM_MODES,
    CORRIDOR_DEFAULT_NUM_STYLES,
    CORRIDOR_GOAL,
    CORRIDOR_START,
    MULTIPATH_DEFAULT_NOISE_STD,
    RADIAL_CENTER,
    RADIAL_DEFAULT_NUM_MODES,
    RADIAL_RADIUS,
    SEQUENTIAL_ENDPOINT_Y,
    SEQUENTIAL_FIRST_BRANCH_X_DELTA,
    SEQUENTIAL_NUM_COMPOUND_MODES,
    SEQUENTIAL_OBSTACLES,
    SEQUENTIAL_SECOND_BRANCH_X_DELTA,
    SEQUENTIAL_START,
    SyntheticTaskName,
)
from versatil.data.synthetic.task_layout import get_task_layout


def _expected_sequential_goals() -> np.ndarray:
    start_x = float(SEQUENTIAL_START[0])
    return np.array(
        [
            (
                start_x
                + first_sign * SEQUENTIAL_FIRST_BRANCH_X_DELTA
                + second_sign * SEQUENTIAL_SECOND_BRANCH_X_DELTA,
                SEQUENTIAL_ENDPOINT_Y,
            )
            for first_sign in (-1.0, 1.0)
            for second_sign in (-1.0, 1.0)
        ],
        dtype=np.float32,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "task_name, expected_start, expected_goals, expected_obstacles, expected_num_modes",
    [
        (
            SyntheticTaskName.CIRCLE.value,
            CIRCLE_START,
            None,
            CIRCLE_OBSTACLES,
            CIRCLE_DEFAULT_NUM_MODES,
        ),
        (
            SyntheticTaskName.CONDITIONAL_CIRCLE.value,
            CIRCLE_START,
            None,
            CIRCLE_OBSTACLES,
            CIRCLE_DEFAULT_NUM_MODES,
        ),
        (
            SyntheticTaskName.SEQUENTIAL_DECISION.value,
            SEQUENTIAL_START,
            _expected_sequential_goals(),
            SEQUENTIAL_OBSTACLES,
            SEQUENTIAL_NUM_COMPOUND_MODES,
        ),
    ],
)
def test_get_task_layout_fixed_mode_tasks(
    task_name: str,
    expected_start: np.ndarray,
    expected_goals: np.ndarray | None,
    expected_obstacles: list[tuple[float, float, float, float]],
    expected_num_modes: int,
):
    layout = get_task_layout(task_name=task_name)

    np.testing.assert_array_equal(layout.start, expected_start)
    if expected_goals is None:
        assert layout.goals is None
    else:
        np.testing.assert_allclose(layout.goals, expected_goals)
    assert layout.obstacles == expected_obstacles
    assert layout.num_modes == expected_num_modes


@pytest.mark.unit
@pytest.mark.parametrize("num_modes", [4, 8, 16])
def test_get_task_layout_radial_uses_dynamic_obstacles(num_modes: int):
    layout = get_task_layout(
        task_name=SyntheticTaskName.RADIAL.value, num_modes=num_modes
    )

    np.testing.assert_array_equal(layout.start, RADIAL_CENTER)
    angles = 2.0 * np.pi * np.arange(num_modes) / num_modes
    expected_goals = np.stack(
        [
            RADIAL_CENTER[0] + RADIAL_RADIUS * np.cos(angles),
            RADIAL_CENTER[1] + RADIAL_RADIUS * np.sin(angles),
        ],
        axis=-1,
    ).astype(np.float32)
    np.testing.assert_allclose(layout.goals, expected_goals)
    assert len(layout.obstacles) == num_modes
    assert layout.num_modes == num_modes


@pytest.mark.unit
def test_get_task_layout_radial_defaults_to_eight_modes():
    layout = get_task_layout(task_name=SyntheticTaskName.RADIAL.value)

    assert layout.num_modes == RADIAL_DEFAULT_NUM_MODES
    assert len(layout.obstacles) == RADIAL_DEFAULT_NUM_MODES


@pytest.mark.unit
def test_radial_layout_passes_noise_std_to_obstacles():
    num_modes = 6
    noise_std = 0.004
    sentinel_obstacles: list[tuple[float, float, float, float]] = [(0.1, 0.1, 0.2, 0.2)]
    with patch(
        "versatil.data.synthetic.task_layout._generate_radial_obstacles",
        return_value=sentinel_obstacles,
    ) as mock_generate:
        layout = get_task_layout(
            task_name=SyntheticTaskName.RADIAL.value,
            num_modes=num_modes,
            noise_std=noise_std,
        )

    mock_generate.assert_called_once_with(num_modes=num_modes, noise_std=noise_std)
    assert layout.obstacles == sentinel_obstacles


@pytest.mark.unit
def test_radial_layout_obstacle_size_shrinks_with_higher_noise():
    low_noise_layout = get_task_layout(
        task_name=SyntheticTaskName.RADIAL.value,
        num_modes=4,
        noise_std=0.001,
    )
    high_noise_layout = get_task_layout(
        task_name=SyntheticTaskName.RADIAL.value,
        num_modes=4,
        noise_std=0.05,
    )

    assert len(low_noise_layout.obstacles) == 4
    low_noise_widths = [
        x_max - x_min for x_min, _, x_max, _ in low_noise_layout.obstacles
    ]
    low_noise_heights = [
        y_max - y_min for _, y_min, _, y_max in low_noise_layout.obstacles
    ]
    if len(high_noise_layout.obstacles) == 0:
        assert all(width > 0.0 for width in low_noise_widths)
        assert all(height > 0.0 for height in low_noise_heights)
    else:
        high_noise_widths = [
            x_max - x_min for x_min, _, x_max, _ in high_noise_layout.obstacles
        ]
        high_noise_heights = [
            y_max - y_min for _, y_min, _, y_max in high_noise_layout.obstacles
        ]
        for low_width, high_width in zip(low_noise_widths, high_noise_widths):
            assert high_width < low_width
        for low_height, high_height in zip(low_noise_heights, high_noise_heights):
            assert high_height < low_height


@pytest.mark.unit
def test_get_task_layout_default_noise_std_matches_constant():
    with patch(
        "versatil.data.synthetic.task_layout._generate_radial_obstacles",
        return_value=[],
    ) as mock_generate:
        get_task_layout(task_name=SyntheticTaskName.RADIAL.value, num_modes=4)

    mock_generate.assert_called_once_with(
        num_modes=4, noise_std=MULTIPATH_DEFAULT_NOISE_STD
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_modes, num_styles, expected_total_modes, expected_num_obstacles",
    [
        (4, 1, 4, 3),
        (2, 3, 6, 1),
        (6, 2, 12, 5),
    ],
)
def test_get_task_layout_corridor_uses_dynamic_obstacles(
    num_modes: int,
    num_styles: int,
    expected_total_modes: int,
    expected_num_obstacles: int,
):
    layout = get_task_layout(
        task_name=SyntheticTaskName.CORRIDOR_NAVIGATION.value,
        num_modes=num_modes,
        num_styles=num_styles,
    )

    np.testing.assert_array_equal(layout.start, CORRIDOR_START)
    np.testing.assert_array_equal(layout.goals, CORRIDOR_GOAL[np.newaxis, :])
    assert layout.num_modes == expected_total_modes
    assert len(layout.obstacles) == expected_num_obstacles


@pytest.mark.unit
def test_get_task_layout_corridor_defaults():
    layout = get_task_layout(
        task_name=SyntheticTaskName.CORRIDOR_NAVIGATION.value,
    )

    expected_total = CORRIDOR_DEFAULT_NUM_MODES * CORRIDOR_DEFAULT_NUM_STYLES
    assert layout.num_modes == expected_total
    assert len(layout.obstacles) == CORRIDOR_DEFAULT_NUM_MODES - 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "task_name, expectation",
    [
        (SyntheticTaskName.CIRCLE.value, does_not_raise()),
        (
            "nonexistent_task",
            pytest.raises(
                ValueError,
                match=re.escape("Unknown synthetic task: nonexistent_task"),
            ),
        ),
    ],
)
def test_get_task_layout_unknown_task_raises(
    task_name: str, expectation: AbstractContextManager
):
    with expectation:
        get_task_layout(task_name=task_name)
