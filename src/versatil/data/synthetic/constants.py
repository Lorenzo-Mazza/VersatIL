"""Constants for synthetic multimodal benchmark tasks."""

import enum

import numpy as np


class SyntheticTaskName(enum.StrEnum):
    """Enum for synthetic benchmark task names."""

    MULTI_PATH_NAVIGATION = "multi_path_navigation"
    CONDITIONAL_NAVIGATION = "conditional_navigation"
    TRAJECTORY_STYLE = "trajectory_style"
    SEQUENTIAL_DECISION = "sequential_decision"
    SHARED_PREFIX = "shared_prefix"


DEFAULT_IMAGE_SIZE = 64
DEFAULT_SEED = 42
DEFAULT_NUM_EPISODES = 1000

#: Task 1 & 2: Multi-path, Conditional Navigation

MULTIPATH_DEFAULT_NUM_MODES = 3
MULTIPATH_DEFAULT_TRAJECTORY_LENGTH = 60
MULTIPATH_DEFAULT_NOISE_STD = 0.01

MULTIPATH_MIN_TRAJECTORY_LENGTH = 2
MULTIPATH_START = np.array([0.0, 0.0], dtype=np.float32)
MULTIPATH_GOAL = np.array([0.95, 0.95], dtype=np.float32)

MULTIPATH_WAYPOINTS: dict[int, list[tuple[float, float]]] = {
    0: [(0.0, 0.0), (0.85, 0.08), (0.9, 0.85), (0.95, 0.95)],  # Path A: right→up
    1: [(0.0, 0.0), (0.05, 0.88), (0.85, 0.92), (0.95, 0.95)],  # Path B: up→right
    2: [(0.0, 0.0), (0.2, 0.42), (0.75, 0.52), (0.95, 0.95)],  # Path C: diagonal gap
}

MULTIPATH_OBSTACLES: list[tuple[float, float, float, float]] = [
    (0.25, 0.15, 0.75, 0.40),  # Obstacle 1
    (0.15, 0.55, 0.65, 0.80),  # Obstacle 2
]

MULTIPATH_CONTEXT_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 0, 0),  # Red for path A
    1: (0, 0, 255),  # Blue for path B
    2: (255, 255, 0),  # Yellow for path C
}

#: Task 3: Trajectory Style

STYLE_DEFAULT_NUM_STYLES = 4
STYLE_DEFAULT_TRAJECTORY_LENGTH = 60
STYLE_DEFAULT_NOISE_STD = 0.015

STYLE_START = np.array([0.0, 0.5], dtype=np.float32)
STYLE_GOAL = np.array([1.0, 0.5], dtype=np.float32)

STYLE_MIN_TRAJECTORY_LENGTH = 2

# Task 4: Sequential Decision

SEQUENTIAL_DEFAULT_TRAJECTORY_LENGTH = 60
SEQUENTIAL_DEFAULT_NOISE_STD = 0.012

SEQUENTIAL_MIN_TRAJECTORY_LENGTH = 41
SEQUENTIAL_START = np.array([0.5, 0.0], dtype=np.float32)

SEQUENTIAL_INTERSECTION_Y_1 = 0.3
SEQUENTIAL_INTERSECTION_Y_2 = 0.55
SEQUENTIAL_BRANCH_X_DELTA = 0.15

# Task 5: Shared Prefix

SHARED_PREFIX_DEFAULT_NUM_MODES = 3
SHARED_PREFIX_DEFAULT_TRAJECTORY_LENGTH = 60
SHARED_PREFIX_DEFAULT_NOISE_STD = 0.012

SHARED_PREFIX_MIN_TRAJECTORY_LENGTH = 31
SHARED_PREFIX_START = np.array([0.0, 0.5], dtype=np.float32)
SHARED_PREFIX_DECISION_POINT_X = 0.5
SHARED_PREFIX_SHARED_STEPS = 30

SHARED_PREFIX_ENDPOINTS: dict[int, tuple[float, float]] = {
    0: (1.0, 0.85),  # Up-right
    1: (1.0, 0.5),  # Straight-right
    2: (1.0, 0.15),  # Down-right
}
