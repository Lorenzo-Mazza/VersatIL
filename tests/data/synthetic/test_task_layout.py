"""Tests for versatil.data.synthetic.task_layout module."""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from versatil.data.synthetic.constants import (
    MULTIPATH_DEFAULT_NUM_MODES,
    MULTIPATH_GOAL,
    MULTIPATH_OBSTACLES,
    MULTIPATH_START,
    SEQUENTIAL_START,
    SHARED_PREFIX_DEFAULT_NUM_MODES,
    SHARED_PREFIX_START,
    STYLE_DEFAULT_NUM_STYLES,
    STYLE_GOAL,
    STYLE_START,
    SyntheticTaskName,
)
from versatil.data.synthetic.task_layout import (
    SEQUENTIAL_NUM_COMPOUND_MODES,
    get_task_layout,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "task_name, expected_start, expected_goal, expected_obstacles, expected_num_modes",
    [
        (
            SyntheticTaskName.MULTI_PATH_NAVIGATION.value,
            MULTIPATH_START,
            MULTIPATH_GOAL,
            MULTIPATH_OBSTACLES,
            MULTIPATH_DEFAULT_NUM_MODES,
        ),
        (
            SyntheticTaskName.CONDITIONAL_NAVIGATION.value,
            MULTIPATH_START,
            MULTIPATH_GOAL,
            MULTIPATH_OBSTACLES,
            MULTIPATH_DEFAULT_NUM_MODES,
        ),
        (
            SyntheticTaskName.TRAJECTORY_STYLE.value,
            STYLE_START,
            STYLE_GOAL,
            [],
            STYLE_DEFAULT_NUM_STYLES,
        ),
        (
            SyntheticTaskName.SEQUENTIAL_DECISION.value,
            SEQUENTIAL_START,
            None,
            [],
            SEQUENTIAL_NUM_COMPOUND_MODES,
        ),
        (
            SyntheticTaskName.SHARED_PREFIX.value,
            SHARED_PREFIX_START,
            None,
            [],
            SHARED_PREFIX_DEFAULT_NUM_MODES,
        ),
    ],
)
def test_get_task_layout_matches_source_constants_per_task(
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
@pytest.mark.parametrize(
    "task_name, expectation",
    [
        (
            SyntheticTaskName.MULTI_PATH_NAVIGATION.value,
            does_not_raise(),
        ),
        (
            "nonexistent_task",
            pytest.raises(
                ValueError,
                match=re.escape("Unknown synthetic task: nonexistent_task"),
            ),
        ),
    ],
)
def test_get_task_layout_unknown_task_raises(task_name, expectation):
    with expectation:
        get_task_layout(task_name=task_name)
