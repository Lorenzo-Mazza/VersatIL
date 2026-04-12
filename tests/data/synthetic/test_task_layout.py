"""Tests for versatil.data.synthetic.task_layout module."""

import re
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise

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
    RADIAL_CENTER,
    RADIAL_DEFAULT_NUM_MODES,
    SEQUENTIAL_NUM_COMPOUND_MODES,
    SEQUENTIAL_OBSTACLES,
    SEQUENTIAL_START,
    SyntheticTaskName,
)
from versatil.data.synthetic.task_layout import get_task_layout


@pytest.mark.unit
@pytest.mark.parametrize(
    "task_name, expected_start, expected_goal, expected_obstacles, expected_num_modes",
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
            None,
            SEQUENTIAL_OBSTACLES,
            SEQUENTIAL_NUM_COMPOUND_MODES,
        ),
    ],
)
def test_get_task_layout_fixed_mode_tasks(
    task_name: str,
    expected_start: np.ndarray,
    expected_goal: np.ndarray | None,
    expected_obstacles: list[tuple[float, float, float, float]],
    expected_num_modes: int,
):
    layout = get_task_layout(task_name=task_name)

    np.testing.assert_array_equal(layout.start, expected_start)
    if expected_goal is None:
        assert layout.goal is None
    else:
        np.testing.assert_array_equal(layout.goal, expected_goal)
    assert layout.obstacles == expected_obstacles
    assert layout.num_modes == expected_num_modes


@pytest.mark.unit
@pytest.mark.parametrize("num_modes", [4, 8, 16])
def test_get_task_layout_radial_uses_dynamic_obstacles(num_modes: int):
    layout = get_task_layout(
        task_name=SyntheticTaskName.RADIAL.value, num_modes=num_modes
    )

    np.testing.assert_array_equal(layout.start, RADIAL_CENTER)
    assert layout.goal is None
    assert len(layout.obstacles) == num_modes
    assert layout.num_modes == num_modes


@pytest.mark.unit
def test_get_task_layout_radial_defaults_to_eight_modes():
    layout = get_task_layout(task_name=SyntheticTaskName.RADIAL.value)

    assert layout.num_modes == RADIAL_DEFAULT_NUM_MODES
    assert len(layout.obstacles) == RADIAL_DEFAULT_NUM_MODES


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
    np.testing.assert_array_equal(layout.goal, CORRIDOR_GOAL)
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
