"""2D top-down image renderer for synthetic benchmark tasks."""

import cv2
import numpy as np

BACKGROUND_COLOR = (240, 240, 240)
AGENT_COLOR = (66, 133, 244)
GOAL_COLOR = (52, 168, 83)
OBSTACLE_COLOR = (80, 80, 80)
TRAIL_COLOR = (180, 180, 180)
AGENT_RADIUS_RATIO = 0.047
GOAL_RADIUS_RATIO = 0.047
CONTEXT_INDICATOR_SIZE_RATIO = 0.094
CONTEXT_INDICATOR_ORIGIN = 2
TRAIL_THICKNESS = 1


def render_frame(
    position: np.ndarray,
    obstacles: list[tuple[float, float, float, float]],
    goal: np.ndarray,
    image_size: int = 64,
    trail: np.ndarray | None = None,
    context_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Render a single top-down 2D frame as an RGB image.

    Draws obstacles, goal marker, agent trail, agent marker, and an
    optional context indicator onto a square canvas.

    Args:
        position: Cartesian agent position (x, y) in [0, 1]x[0, 1].
            Shape (2,).
        obstacles: Axis-aligned rectangles as (x_min, y_min, x_max, y_max)
            in [0, 1] coordinates.
        goal: Cartesian goal position (x, y) in [0, 1]x[0, 1]. Shape (2,).
        image_size: Side length of the square output image in pixels.
        trail: Past Cartesian positions (x, y) for rendering the agent
            trail. Shape (num_past_steps, 2). None to disable trail.
        context_color: RGB color tuple for the context indicator square
            drawn in the top-left corner. None to disable.

    Returns:
        RGB image of shape (image_size, image_size, 3), dtype uint8.
    """
    image = np.full((image_size, image_size, 3), BACKGROUND_COLOR, dtype=np.uint8)
    agent_radius = max(2, int(AGENT_RADIUS_RATIO * image_size))
    goal_radius = max(2, int(GOAL_RADIUS_RATIO * image_size))
    indicator_size = max(4, int(CONTEXT_INDICATOR_SIZE_RATIO * image_size))
    for x_min, y_min, x_max, y_max in obstacles:
        top_left = _cartesian_to_pixel(np.array([x_min, y_min]), image_size)
        bottom_right = _cartesian_to_pixel(np.array([x_max, y_max]), image_size)
        cv2.rectangle(image, top_left, bottom_right, OBSTACLE_COLOR, thickness=-1)
    goal_pixel = _cartesian_to_pixel(goal, image_size)
    cv2.circle(image, goal_pixel, goal_radius, GOAL_COLOR, thickness=-1)
    if trail is not None and len(trail) > 1:
        trail_pixels = np.array(
            [_cartesian_to_pixel(point, image_size) for point in trail],
            dtype=np.int32,
        )
        cv2.polylines(
            image,
            [trail_pixels],
            isClosed=False,
            color=TRAIL_COLOR,
            thickness=TRAIL_THICKNESS,
        )
    agent_pixel = _cartesian_to_pixel(position, image_size)
    cv2.circle(image, agent_pixel, agent_radius, AGENT_COLOR, thickness=-1)
    if context_color is not None:
        cv2.rectangle(
            image,
            (CONTEXT_INDICATOR_ORIGIN, CONTEXT_INDICATOR_ORIGIN),
            (
                CONTEXT_INDICATOR_ORIGIN + indicator_size,
                CONTEXT_INDICATOR_ORIGIN + indicator_size,
            ),
            context_color,
            thickness=-1,
        )
    return image


def render_episode(
    positions: np.ndarray,
    obstacles: list[tuple[float, float, float, float]],
    goal: np.ndarray,
    image_size: int = 64,
    show_trail: bool = True,
    context_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Render all timestep frames for a full episode.

    Args:
        positions: Cartesian agent positions (x, y) over time.
            Shape (num_timesteps, 2), values in [0, 1].
        obstacles: Axis-aligned rectangles as (x_min, y_min, x_max, y_max)
            in [0, 1] coordinates.
        goal: Cartesian goal position (x, y) in [0, 1]x[0, 1]. Shape (2,).
        image_size: Side length of the square output image in pixels.
        show_trail: Whether to render the trajectory trail up to each frame.
        context_color: RGB color tuple for the context indicator square.
            None to disable.

    Returns:
        RGB images of shape (num_timesteps, image_size, image_size, 3),
        dtype uint8.
    """
    num_timesteps = len(positions)
    images = np.empty((num_timesteps, image_size, image_size, 3), dtype=np.uint8)
    for timestep in range(num_timesteps):
        trail = positions[: timestep + 1] if show_trail else None
        images[timestep] = render_frame(
            position=positions[timestep],
            obstacles=obstacles,
            goal=goal,
            image_size=image_size,
            trail=trail,
            context_color=context_color,
        )
    return images


def _cartesian_to_pixel(
    position: np.ndarray,
    image_size: int,
) -> tuple[int, int]:
    """Convert Cartesian [0, 1] coordinates to pixel coordinates.

    The y-axis is flipped so that y=0 maps to the bottom of the image
    and y=1 maps to the top, matching standard Cartesian convention.

    Args:
        position: Cartesian position (x, y) in [0, 1]x[0, 1]. Shape (2,).
        image_size: Side length of the square image in pixels.

    Returns:
        Pixel coordinates (column, row) for OpenCV drawing functions.
    """
    column = int(np.clip(position[0] * (image_size - 1), 0, image_size - 1))
    row = int(np.clip((1.0 - position[1]) * (image_size - 1), 0, image_size - 1))
    return (column, row)
