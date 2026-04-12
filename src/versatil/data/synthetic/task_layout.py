"""Task layout metadata for synthetic benchmark visualization and rendering."""

from dataclasses import dataclass

import numpy as np

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

SEQUENTIAL_NUM_COMPOUND_MODES = 4


TASK_DISPLAY_NAMES: dict[str, str] = {
    SyntheticTaskName.MULTI_PATH_NAVIGATION.value: "Multi-Path Navigation",
    SyntheticTaskName.CONDITIONAL_NAVIGATION.value: "Conditional Navigation",
    SyntheticTaskName.TRAJECTORY_STYLE.value: "Trajectory Style",
    SyntheticTaskName.SEQUENTIAL_DECISION.value: "Sequential Decision",
    SyntheticTaskName.SHARED_PREFIX.value: "Shared Prefix",
}


@dataclass(frozen=True)
class SyntheticTaskLayout:
    """Task-specific layout data for rendering and visualization.

    Attributes:
        start: Start position in [0, 1]x[0, 1] Cartesian space. Shape (2,).
        goal: Goal position, or None for tasks with mode-dependent goals.
        obstacles: List of (x_min, y_min, x_max, y_max) rectangles.
        num_modes: Number of behavioral modes for this task.
    """

    start: np.ndarray
    goal: np.ndarray | None
    obstacles: list[tuple[float, float, float, float]]
    num_modes: int


def get_task_layout(task_name: str) -> SyntheticTaskLayout:
    """Return the layout data (start, goal, obstacles, num_modes) for a task.

    Args:
        task_name: SyntheticTaskName.value string.

    Returns:
        SyntheticTaskLayout with the task-specific geometry.

    Raises:
        ValueError: If task_name is not a recognized synthetic task.
    """
    match task_name:
        case SyntheticTaskName.MULTI_PATH_NAVIGATION.value:
            return SyntheticTaskLayout(
                start=MULTIPATH_START,
                goal=MULTIPATH_GOAL,
                obstacles=MULTIPATH_OBSTACLES,
                num_modes=MULTIPATH_DEFAULT_NUM_MODES,
            )
        case SyntheticTaskName.CONDITIONAL_NAVIGATION.value:
            return SyntheticTaskLayout(
                start=MULTIPATH_START,
                goal=MULTIPATH_GOAL,
                obstacles=MULTIPATH_OBSTACLES,
                num_modes=MULTIPATH_DEFAULT_NUM_MODES,
            )
        case SyntheticTaskName.TRAJECTORY_STYLE.value:
            return SyntheticTaskLayout(
                start=STYLE_START,
                goal=STYLE_GOAL,
                obstacles=[],
                num_modes=STYLE_DEFAULT_NUM_STYLES,
            )
        case SyntheticTaskName.SEQUENTIAL_DECISION.value:
            return SyntheticTaskLayout(
                start=SEQUENTIAL_START,
                goal=None,
                obstacles=[],
                num_modes=SEQUENTIAL_NUM_COMPOUND_MODES,
            )
        case SyntheticTaskName.SHARED_PREFIX.value:
            return SyntheticTaskLayout(
                start=SHARED_PREFIX_START,
                goal=None,
                obstacles=[],
                num_modes=SHARED_PREFIX_DEFAULT_NUM_MODES,
            )
        case _:
            raise ValueError(f"Unknown synthetic task: {task_name}")
