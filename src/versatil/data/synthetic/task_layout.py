"""Task layout metadata for synthetic benchmark visualization and rendering."""

from dataclasses import dataclass

import numpy as np

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
    SEQUENTIAL_NUM_COMPOUND_MODES,
    SEQUENTIAL_OBSTACLES,
    SEQUENTIAL_START,
    SyntheticTaskName,
)
from versatil.data.synthetic.generators import (
    _compute_corridor_gap_centers,
    _generate_corridor_obstacles,
    _generate_radial_obstacles,
)

TASK_DISPLAY_NAMES: dict[str, str] = {
    SyntheticTaskName.CIRCLE.value: "Circle",
    SyntheticTaskName.CONDITIONAL_CIRCLE.value: "Conditional Circle",
    SyntheticTaskName.SEQUENTIAL_DECISION.value: "Sequential Decision",
    SyntheticTaskName.RADIAL.value: "Radial",
    SyntheticTaskName.CORRIDOR_NAVIGATION.value: "Corridor Navigation",
}


@dataclass(frozen=True)
class SyntheticTaskLayout:
    """Task-specific layout data for rendering and visualization.

    Attributes:
        start: Start position in [0, 1]x[0, 1] Cartesian space. Shape (2,).
        goal: Goal position, or None for tasks with no fixed goal.
        obstacles: List of (x_min, y_min, x_max, y_max) rectangles.
        num_modes: Number of behavioral modes for this task.
    """

    start: np.ndarray
    goal: np.ndarray | None
    obstacles: list[tuple[float, float, float, float]]
    num_modes: int


def get_task_layout(
    task_name: str,
    num_modes: int | None = None,
    num_styles: int | None = None,
    noise_std: float = MULTIPATH_DEFAULT_NOISE_STD,
) -> SyntheticTaskLayout:
    """Return the layout data (start, goal, obstacles, num_modes) for a task.

    For radial and corridor tasks, obstacles depend on the number of modes
    and are generated dynamically.

    Args:
        task_name: SyntheticTaskName.value string.
        num_modes: Number of modes for tasks with variable mode count
            (radial, corridor_navigation). Uses task defaults when None.
        num_styles: Number of styles per corridor for corridor_navigation.
            Uses task default when None.
        noise_std: Trajectory noise std. Passed to obstacle sizing so a
            3-sigma noise margin is kept between trajectory and obstacle.

    Returns:
        SyntheticTaskLayout with the task-specific geometry.

    Raises:
        ValueError: If task_name is not a recognized synthetic task.
    """
    match task_name:
        case SyntheticTaskName.CIRCLE.value:
            return SyntheticTaskLayout(
                start=CIRCLE_START,
                goal=None,
                obstacles=CIRCLE_OBSTACLES,
                num_modes=CIRCLE_DEFAULT_NUM_MODES,
            )
        case SyntheticTaskName.CONDITIONAL_CIRCLE.value:
            return SyntheticTaskLayout(
                start=CIRCLE_START,
                goal=None,
                obstacles=CIRCLE_OBSTACLES,
                num_modes=CIRCLE_DEFAULT_NUM_MODES,
            )
        case SyntheticTaskName.SEQUENTIAL_DECISION.value:
            return SyntheticTaskLayout(
                start=SEQUENTIAL_START,
                goal=None,
                obstacles=SEQUENTIAL_OBSTACLES,
                num_modes=SEQUENTIAL_NUM_COMPOUND_MODES,
            )
        case SyntheticTaskName.RADIAL.value:
            resolved_modes = (
                num_modes if num_modes is not None else RADIAL_DEFAULT_NUM_MODES
            )
            return SyntheticTaskLayout(
                start=RADIAL_CENTER,
                goal=None,
                obstacles=_generate_radial_obstacles(
                    num_modes=resolved_modes, noise_std=noise_std
                ),
                num_modes=resolved_modes,
            )
        case SyntheticTaskName.CORRIDOR_NAVIGATION.value:
            resolved_modes = (
                num_modes if num_modes is not None else CORRIDOR_DEFAULT_NUM_MODES
            )
            resolved_styles = (
                num_styles if num_styles is not None else CORRIDOR_DEFAULT_NUM_STYLES
            )
            gap_centers = _compute_corridor_gap_centers(num_gaps=resolved_modes)
            return SyntheticTaskLayout(
                start=CORRIDOR_START,
                goal=CORRIDOR_GOAL,
                obstacles=_generate_corridor_obstacles(gap_centers=gap_centers),
                num_modes=resolved_modes * resolved_styles,
            )
        case _:
            raise ValueError(f"Unknown synthetic task: {task_name}")
