"""Tests for versatil.data.synthetic.renderer module."""

from collections.abc import Callable

import cv2
import numpy as np
import pytest

from versatil.data.synthetic.renderer import (
    AGENT_COLOR,
    BACKGROUND_COLOR,
    CONTEXT_INDICATOR_ORIGIN,
    CONTEXT_INDICATOR_SIZE_RATIO,
    GOAL_COLOR,
    OBSTACLE_COLOR,
    _cartesian_to_pixel,
    render_episode,
    render_frame,
)

TEST_IMAGE_SIZE = 64
TEST_CONTEXT_COLOR = (255, 0, 0)
INDICATOR_INSIDE_PIXEL = (
    CONTEXT_INDICATOR_ORIGIN
    + max(4, int(CONTEXT_INDICATOR_SIZE_RATIO * TEST_IMAGE_SIZE)) // 2
)


@pytest.fixture
def position_factory(
    rng: np.random.Generator,
) -> Callable[..., np.ndarray]:
    def factory(
        x: float | None = None,
        y: float | None = None,
    ) -> np.ndarray:
        if x is None:
            x = float(rng.uniform(0.0, 1.0))
        if y is None:
            y = float(rng.uniform(0.0, 1.0))
        return np.array([x, y], dtype=np.float32)

    return factory


@pytest.fixture
def obstacle_factory() -> Callable[..., list[tuple[float, float, float, float]]]:
    def factory(
        obstacles: list[tuple[float, float, float, float]] | None = None,
    ) -> list[tuple[float, float, float, float]]:
        if obstacles is None:
            return [(0.2, 0.2, 0.5, 0.5)]
        return obstacles

    return factory


@pytest.fixture
def positions_factory(
    rng: np.random.Generator,
) -> Callable[..., np.ndarray]:
    def factory(
        num_timesteps: int = 3,
        values: list[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        if values is not None:
            return np.array(values, dtype=np.float32)
        return rng.uniform(0.0, 1.0, size=(num_timesteps, 2)).astype(np.float32)

    return factory


@pytest.fixture
def trail_factory(
    rng: np.random.Generator,
) -> Callable[..., np.ndarray]:
    def factory(
        num_points: int = 3,
        values: list[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        if values is not None:
            return np.array(values, dtype=np.float32)
        return rng.uniform(0.0, 1.0, size=(num_points, 2)).astype(np.float32)

    return factory


@pytest.mark.unit
@pytest.mark.parametrize("render_image_size", [32, 64])
def test_render_frame_output_shape_and_dtype(
    position_factory: Callable[..., np.ndarray],
    obstacle_factory: Callable[..., list[tuple[float, float, float, float]]],
    render_image_size: int,
):
    position = position_factory(x=0.5, y=0.5)
    goal = position_factory(x=0.9, y=0.9)
    frame = render_frame(
        position=position,
        obstacles=obstacle_factory(),
        goal=goal,
        image_size=render_image_size,
    )
    assert frame.shape == (render_image_size, render_image_size, 3)
    assert frame.dtype == np.uint8


@pytest.mark.unit
def test_render_frame_agent_pixel_matches_agent_color(
    position_factory: Callable[..., np.ndarray],
):
    position = position_factory(x=0.5, y=0.5)
    goal = position_factory(x=0.9, y=0.9)
    frame = render_frame(
        position=position,
        obstacles=[],
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
    )
    agent_column, agent_row = _cartesian_to_pixel(
        position=position, image_size=TEST_IMAGE_SIZE
    )
    assert tuple(frame[agent_row, agent_column]) == tuple(AGENT_COLOR)


@pytest.mark.unit
def test_render_frame_goal_pixel_matches_goal_color(
    position_factory: Callable[..., np.ndarray],
):
    goal = position_factory(x=0.1, y=0.1)
    position = position_factory(x=0.9, y=0.9)
    frame = render_frame(
        position=position,
        obstacles=[],
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
    )
    goal_column, goal_row = _cartesian_to_pixel(
        position=goal, image_size=TEST_IMAGE_SIZE
    )
    assert tuple(frame[goal_row, goal_column]) == tuple(GOAL_COLOR)


@pytest.mark.unit
def test_render_frame_no_goal_marker_when_goal_is_none(
    position_factory: Callable[..., np.ndarray],
):
    position = position_factory(x=0.5, y=0.5)
    frame_with_goal = render_frame(
        position=position,
        obstacles=[],
        goal=position_factory(x=0.1, y=0.1),
        image_size=TEST_IMAGE_SIZE,
    )
    frame_without_goal = render_frame(
        position=position,
        obstacles=[],
        goal=None,
        image_size=TEST_IMAGE_SIZE,
    )
    goal_column, goal_row = _cartesian_to_pixel(
        position=position_factory(x=0.1, y=0.1), image_size=TEST_IMAGE_SIZE
    )
    assert tuple(frame_with_goal[goal_row, goal_column]) == tuple(GOAL_COLOR)
    assert tuple(frame_without_goal[goal_row, goal_column]) == tuple(BACKGROUND_COLOR)


@pytest.mark.unit
def test_render_frame_obstacle_region_filled(
    position_factory: Callable[..., np.ndarray],
    obstacle_factory: Callable[..., list[tuple[float, float, float, float]]],
):
    position = position_factory(x=0.0, y=0.0)
    goal = position_factory(x=0.0, y=1.0)
    obstacles = obstacle_factory(obstacles=[(0.4, 0.4, 0.6, 0.6)])
    frame = render_frame(
        position=position,
        obstacles=obstacles,
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
    )
    obstacle_center = position_factory(x=0.5, y=0.5)
    obstacle_column, obstacle_row = _cartesian_to_pixel(
        position=obstacle_center, image_size=TEST_IMAGE_SIZE
    )
    assert tuple(frame[obstacle_row, obstacle_column]) == tuple(OBSTACLE_COLOR)


@pytest.mark.unit
@pytest.mark.parametrize(
    "context_color, expected_pixel_is_context",
    [
        (TEST_CONTEXT_COLOR, True),
        (None, False),
    ],
)
def test_render_frame_context_indicator_drawn_when_color_provided(
    position_factory: Callable[..., np.ndarray],
    context_color: tuple[int, int, int] | None,
    expected_pixel_is_context: bool,
):
    position = position_factory(x=0.5, y=0.5)
    goal = position_factory(x=0.9, y=0.9)
    frame = render_frame(
        position=position,
        obstacles=[],
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
        context_color=context_color,
    )
    pixel = tuple(frame[INDICATOR_INSIDE_PIXEL, INDICATOR_INSIDE_PIXEL])
    if expected_pixel_is_context:
        assert pixel == context_color
    else:
        assert pixel == tuple(BACKGROUND_COLOR)


@pytest.mark.unit
def test_render_frame_trail_drawn_when_multi_point(
    position_factory: Callable[..., np.ndarray],
    trail_factory: Callable[..., np.ndarray],
):
    position = position_factory(x=0.8, y=0.8)
    goal = position_factory(x=0.95, y=0.95)
    trail = trail_factory(
        values=[(0.1, 0.1), (0.4, 0.4), (0.8, 0.8)],
    )
    frame_with_trail = render_frame(
        position=position,
        obstacles=[],
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
        trail=trail,
    )
    frame_without_trail = render_frame(
        position=position,
        obstacles=[],
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
        trail=None,
    )
    assert not np.array_equal(frame_with_trail, frame_without_trail)


@pytest.mark.unit
@pytest.mark.parametrize(
    "trail_values",
    [
        None,
        [(0.5, 0.5)],
    ],
)
def test_render_frame_trail_skipped_when_single_or_none(
    position_factory: Callable[..., np.ndarray],
    trail_factory: Callable[..., np.ndarray],
    trail_values: list[tuple[float, float]] | None,
):
    position = position_factory(x=0.5, y=0.5)
    goal = position_factory(x=0.9, y=0.9)
    trail = None if trail_values is None else trail_factory(values=trail_values)
    frame_with_trail = render_frame(
        position=position,
        obstacles=[],
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
        trail=trail,
    )
    frame_without_trail = render_frame(
        position=position,
        obstacles=[],
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
        trail=None,
    )
    np.testing.assert_array_equal(frame_with_trail, frame_without_trail)


@pytest.mark.unit
def test_render_episode_output_shape(
    position_factory: Callable[..., np.ndarray],
    positions_factory: Callable[..., np.ndarray],
):
    num_timesteps = 3
    render_image_size = 32
    positions = positions_factory(
        values=[(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)],
    )
    goal = position_factory(x=0.9, y=0.9)
    images = render_episode(
        positions=positions,
        obstacles=[],
        goal=goal,
        image_size=render_image_size,
    )
    assert images.shape == (
        num_timesteps,
        render_image_size,
        render_image_size,
        3,
    )


@pytest.mark.unit
def test_render_episode_each_frame_has_agent(
    position_factory: Callable[..., np.ndarray],
    positions_factory: Callable[..., np.ndarray],
):
    positions = positions_factory(
        values=[(0.2, 0.2), (0.5, 0.5), (0.8, 0.8)],
    )
    goal = position_factory(x=0.95, y=0.95)
    images = render_episode(
        positions=positions,
        obstacles=[],
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
    )
    for timestep in range(len(positions)):
        agent_column, agent_row = _cartesian_to_pixel(
            position=positions[timestep], image_size=TEST_IMAGE_SIZE
        )
        assert tuple(images[timestep, agent_row, agent_column]) == tuple(AGENT_COLOR)


@pytest.mark.unit
def test_render_episode_trail_grows_over_time(
    position_factory: Callable[..., np.ndarray],
    positions_factory: Callable[..., np.ndarray],
):
    positions = positions_factory(
        values=[(0.1, 0.1), (0.3, 0.3), (0.6, 0.6)],
    )
    goal = position_factory(x=0.95, y=0.95)
    images = render_episode(
        positions=positions,
        obstacles=[],
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
        show_trail=True,
    )
    background = np.array(BACKGROUND_COLOR, dtype=np.uint8)
    non_background_frame_0 = np.sum(~np.all(images[0] == background, axis=-1))
    non_background_frame_2 = np.sum(~np.all(images[2] == background, axis=-1))
    assert non_background_frame_2 > non_background_frame_0


@pytest.mark.unit
def test_render_episode_trail_disabled_when_show_trail_false(
    position_factory: Callable[..., np.ndarray],
    positions_factory: Callable[..., np.ndarray],
):
    positions = positions_factory(
        values=[(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)],
    )
    goal = position_factory(x=0.95, y=0.95)
    images_with_trail = render_episode(
        positions=positions,
        obstacles=[],
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
        show_trail=True,
    )
    images_without_trail = render_episode(
        positions=positions,
        obstacles=[],
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
        show_trail=False,
    )
    assert not np.array_equal(images_with_trail[2], images_without_trail[2])


@pytest.mark.unit
def test_render_episode_passes_context_color_to_frames(
    position_factory: Callable[..., np.ndarray],
    positions_factory: Callable[..., np.ndarray],
    obstacle_factory: Callable[..., list[tuple[float, float, float, float]]],
):
    context_color = TEST_CONTEXT_COLOR
    positions = positions_factory(
        values=[(0.5, 0.5), (0.6, 0.6), (0.7, 0.7)],
    )
    goal = position_factory(x=1.0, y=1.0)
    images = render_episode(
        positions=positions,
        obstacles=obstacle_factory(),
        goal=goal,
        image_size=TEST_IMAGE_SIZE,
        context_color=context_color,
    )
    for timestep in range(len(positions)):
        assert (
            tuple(images[timestep, INDICATOR_INSIDE_PIXEL, INDICATOR_INSIDE_PIXEL])
            == context_color
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "x, y, expected_column, expected_row",
    [
        (0.0, 0.0, 0, 63),
        (1.0, 1.0, 63, 0),
        (0.5, 0.5, 31, 31),
    ],
)
def test_cartesian_to_pixel_corner_mappings(
    position_factory: Callable[..., np.ndarray],
    x: float,
    y: float,
    expected_column: int,
    expected_row: int,
):
    position = position_factory(x=x, y=y)
    column, row = _cartesian_to_pixel(position=position, image_size=TEST_IMAGE_SIZE)
    assert column == expected_column
    assert row == expected_row


@pytest.mark.unit
@pytest.mark.parametrize(
    "x, y, expected_column, expected_row",
    [
        (-0.1, 1.5, 0, 0),
        (2.0, -0.5, 63, 63),
        (-5.0, 0.5, 0, 31),
    ],
)
def test_cartesian_to_pixel_clamps_out_of_range(
    position_factory: Callable[..., np.ndarray],
    x: float,
    y: float,
    expected_column: int,
    expected_row: int,
):
    position = position_factory(x=x, y=y)
    column, row = _cartesian_to_pixel(position=position, image_size=TEST_IMAGE_SIZE)
    assert column == expected_column
    assert row == expected_row


def _count_obstacle_blobs(frame: np.ndarray) -> int:
    mask = np.all(frame == np.array(OBSTACLE_COLOR, dtype=np.uint8), axis=-1)
    num_components, _ = cv2.connectedComponents(mask.astype(np.uint8))
    return num_components - 1


@pytest.mark.integration
@pytest.mark.parametrize("num_obstacles", [1, 2, 3, 5, 7, 9])
def test_render_frame_draws_all_obstacles_without_hardcoding(
    position_factory: Callable[..., np.ndarray],
    num_obstacles: int,
):
    image_size = 256
    x_min = 0.1
    x_max = 0.9
    obstacle_half_width = 0.02
    gap_spacing = (x_max - x_min) / (num_obstacles + 1)
    obstacles = [
        (
            x_min + (index + 1) * gap_spacing - obstacle_half_width,
            0.4,
            x_min + (index + 1) * gap_spacing + obstacle_half_width,
            0.6,
        )
        for index in range(num_obstacles)
    ]
    frame = render_frame(
        position=position_factory(x=0.0, y=0.0),
        obstacles=obstacles,
        goal=None,
        image_size=image_size,
    )
    assert _count_obstacle_blobs(frame=frame) == num_obstacles
