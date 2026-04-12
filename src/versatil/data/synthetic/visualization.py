"""Visualization utilities for synthetic benchmark trajectories."""

from pathlib import Path

import cv2
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

from versatil.data.synthetic.constants import (
    DEFAULT_IMAGE_SIZE,
)
from versatil.data.synthetic.renderer import (
    AGENT_RADIUS_RATIO,
    BACKGROUND_COLOR,
    GOAL_COLOR,
    OBSTACLE_COLOR,
    TRAIL_COLOR,
    TRAIL_THICKNESS,
    _cartesian_to_pixel,
)
from versatil.data.synthetic.task_layout import (
    TASK_DISPLAY_NAMES,
    SyntheticTaskLayout,
    get_task_layout,
)

FONT_PATH = Path(__file__).parent / "assets" / "RobotoSerif.ttf"

PLOT_MODE_COLORS: dict[int, str] = {
    0: "#5B9BD5",
    1: "#E06B6B",
    2: "#7FAE6A",
    3: "#D4A55C",
}
PLOT_AGENT_MARKER_COLOR = "#C1272D"
PLOT_OBSTACLE_COLOR = "#BFBFBF"
PLOT_GOAL_COLOR = "#8FD68F"
PLOT_BACKGROUND_COLOR = "#FFFFFF"
PLOT_BORDER_COLOR = "#B0B0B0"
PLOT_LEGEND_TRAJECTORY_COLOR = "#5B9BD5"
PLOT_GOAL_SIZE = 0.05
PLOT_TRAJECTORY_ALPHA = 0.55
PLOT_TRAJECTORY_LINEWIDTH = 2.5
PLOT_EXPERT_ALPHA = 0.12
PLOT_EXPERT_LINEWIDTH = 2.0
PLOT_AGENT_MARKER_SIZE = 55


def _apply_plot_theme() -> None:
    """Register the bundled Roboto Serif font and apply the seaborn paper theme."""
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
    sns.set_theme(
        style="white",
        context="paper",
        font="Roboto Serif",
        rc={
            "font.family": "serif",
            "font.serif": ["Roboto Serif", "DejaVu Serif"],
            "axes.titlesize": 14,
            "axes.labelsize": 10,
        },
    )


def _draw_task_background(
    axes: plt.Axes,
    layout: SyntheticTaskLayout,
) -> None:
    """Draw the unit square, obstacles, and goal marker onto a matplotlib axes.

    Args:
        axes: Matplotlib axes to draw onto.
        layout: Task layout providing obstacles and (optionally) the goal.
    """
    axes.set_facecolor(PLOT_BACKGROUND_COLOR)
    axes.set_xlim(-0.02, 1.02)
    axes.set_ylim(-0.02, 1.02)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color(PLOT_BORDER_COLOR)
        spine.set_linestyle((0, (2, 2)))
        spine.set_linewidth(0.8)
    for x_min, y_min, x_max, y_max in layout.obstacles:
        axes.add_patch(
            Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor=PLOT_OBSTACLE_COLOR,
                edgecolor="none",
                zorder=1,
            )
        )
    if layout.goal is not None:
        axes.add_patch(
            Rectangle(
                (
                    layout.goal[0] - PLOT_GOAL_SIZE / 2,
                    layout.goal[1] - PLOT_GOAL_SIZE / 2,
                ),
                PLOT_GOAL_SIZE,
                PLOT_GOAL_SIZE,
                facecolor=PLOT_GOAL_COLOR,
                edgecolor="none",
                zorder=2,
            )
        )


def _build_legend_handles(
    layout: SyntheticTaskLayout,
) -> list[Line2D | Patch]:
    """Build legend handles (agent, trajectory, goal, obstacle) for a task.

    Args:
        layout: Task layout used to conditionally include goal and obstacle
            entries based on task geometry.

    Returns:
        List of matplotlib legend handles ready to pass to ``axes.legend``.
    """
    handles: list[Line2D | Patch] = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=PLOT_AGENT_MARKER_COLOR,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=9,
            label="Agent",
        ),
        Line2D(
            [0],
            [0],
            color=PLOT_LEGEND_TRAJECTORY_COLOR,
            linewidth=PLOT_TRAJECTORY_LINEWIDTH,
            alpha=0.75,
            label="Trajectory",
        ),
    ]
    if layout.goal is not None:
        handles.append(Patch(facecolor=PLOT_GOAL_COLOR, edgecolor="none", label="Goal"))
    if len(layout.obstacles) > 0:
        handles.append(
            Patch(
                facecolor=PLOT_OBSTACLE_COLOR,
                edgecolor="none",
                label="Obstacle",
            )
        )
    return handles


def plot_trajectories_2d(
    trajectories: np.ndarray,
    task_name: str,
    output_path: str | None = None,
    mode_ids: np.ndarray | None = None,
    expert_trajectories: np.ndarray | None = None,
    expert_mode_ids: np.ndarray | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot 2D trajectories overlaid on the task layout.

    Always returns the figure. Additionally saves to disk when
    ``output_path`` is provided.

    Args:
        trajectories: Cartesian trajectories, shape (num_trajectories,
            num_timesteps, 2), values in [0, 1].
        task_name: SyntheticTaskName.value string.
        output_path: Optional PNG path. Saves to disk when provided.
        mode_ids: Optional per-trajectory mode index for coloring.
            Shape (num_trajectories,).
        expert_trajectories: Optional faint background trajectories.
            Shape (num_experts, num_timesteps, 2).
        expert_mode_ids: Optional per-expert mode index for coloring.
        title: Optional plot title. Defaults to the task display name.

    Returns:
        The matplotlib Figure. Caller is responsible for closing it.
    """
    _apply_plot_theme()
    layout = get_task_layout(task_name=task_name)
    figure, axes = plt.subplots(figsize=(6, 6), dpi=150)
    _draw_task_background(axes=axes, layout=layout)

    if expert_trajectories is not None:
        for index in range(len(expert_trajectories)):
            mode_index = (
                int(expert_mode_ids[index]) if expert_mode_ids is not None else 0
            )
            color = PLOT_MODE_COLORS[mode_index % len(PLOT_MODE_COLORS)]
            axes.plot(
                expert_trajectories[index, :, 0],
                expert_trajectories[index, :, 1],
                color=color,
                alpha=PLOT_EXPERT_ALPHA,
                linewidth=PLOT_EXPERT_LINEWIDTH,
                zorder=3,
            )

    for index in range(len(trajectories)):
        mode_index = int(mode_ids[index]) if mode_ids is not None else 0
        color = PLOT_MODE_COLORS[mode_index % len(PLOT_MODE_COLORS)]
        axes.plot(
            trajectories[index, :, 0],
            trajectories[index, :, 1],
            color=color,
            alpha=PLOT_TRAJECTORY_ALPHA,
            linewidth=PLOT_TRAJECTORY_LINEWIDTH,
            zorder=4,
        )

    if len(trajectories) > 0:
        final_positions = trajectories[:, -1, :]
        axes.scatter(
            final_positions[:, 0],
            final_positions[:, 1],
            s=PLOT_AGENT_MARKER_SIZE,
            color=PLOT_AGENT_MARKER_COLOR,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )

    axes.legend(
        handles=_build_legend_handles(layout=layout),
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        handlelength=1.8,
        edgecolor=PLOT_BORDER_COLOR,
    )
    plot_title = title if title is not None else TASK_DISPLAY_NAMES[task_name]
    axes.set_title(plot_title, pad=12)
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
            facecolor=PLOT_BACKGROUND_COLOR,
        )
    return figure


def _render_multi_agent_frame(
    positions: np.ndarray,
    trails: np.ndarray | None,
    mode_ids: np.ndarray | None,
    layout: SyntheticTaskLayout,
    image_size: int,
) -> np.ndarray:
    """Render one frame with multiple agents drawn on the same canvas.

    Used by ``save_rollouts_gif`` to show all rollouts progressing in parallel.

    Args:
        positions: Current agent positions, shape (num_agents, 2), in [0, 1].
        trails: Past positions per agent, shape (num_agents, num_past_steps, 2),
            or None to disable trails.
        mode_ids: Optional per-agent mode index for color lookup.
            Shape (num_agents,).
        layout: Task layout providing obstacles and goal.
        image_size: Side length of the output square image in pixels.

    Returns:
        RGB image of shape (image_size, image_size, 3), dtype uint8.
    """
    image = np.full((image_size, image_size, 3), BACKGROUND_COLOR, dtype=np.uint8)
    agent_radius = max(2, int(AGENT_RADIUS_RATIO * image_size))
    goal_radius = max(2, int(AGENT_RADIUS_RATIO * image_size))
    for x_min, y_min, x_max, y_max in layout.obstacles:
        top_left = _cartesian_to_pixel(np.array([x_min, y_min]), image_size)
        bottom_right = _cartesian_to_pixel(np.array([x_max, y_max]), image_size)
        cv2.rectangle(image, top_left, bottom_right, OBSTACLE_COLOR, thickness=-1)
    if layout.goal is not None:
        goal_pixel = _cartesian_to_pixel(layout.goal, image_size)
        cv2.circle(image, goal_pixel, goal_radius, GOAL_COLOR, thickness=-1)
    num_agents = positions.shape[0]
    if trails is not None:
        for agent_index in range(num_agents):
            trail = trails[agent_index]
            if len(trail) <= 1:
                continue
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
    for agent_index in range(num_agents):
        agent_pixel = _cartesian_to_pixel(positions[agent_index], image_size)
        mode_index = int(mode_ids[agent_index]) if mode_ids is not None else 0
        agent_color = _mode_color_bgr(mode_index=mode_index)
        cv2.circle(image, agent_pixel, agent_radius, agent_color, thickness=-1)
    return image


def _mode_color_bgr(mode_index: int) -> tuple[int, int, int]:
    """Return the BGR color tuple for a given mode index.

    BGR ordering is required for OpenCV drawing functions.

    Args:
        mode_index: Mode index, wrapped modulo the palette size.

    Returns:
        (blue, green, red) tuple with values in [0, 255].
    """
    palette: dict[int, tuple[int, int, int]] = {
        0: (213, 155, 91),  # blue  (#5B9BD5 in BGR)
        1: (107, 107, 224),  # red   (#E06B6B in BGR)
        2: (106, 174, 127),  # green (#7FAE6A in BGR)
        3: (92, 165, 212),  # gold  (#D4A55C in BGR)
    }
    return palette[mode_index % len(palette)]


def save_rollouts_gif(
    trajectories: np.ndarray,
    task_name: str,
    output_path: str,
    mode_ids: np.ndarray | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    frames_per_second: int = 30,
) -> None:
    """Save an animated GIF showing all rollout trajectories evolving in parallel.

    At each frame, every rollout's current position is drawn on the same
    canvas along with the trail up to that timestep. Colors encode mode_id.

    Args:
        trajectories: Rollout trajectories, shape (num_rollouts,
            num_timesteps, 2), values in [0, 1].
        task_name: SyntheticTaskName.value string.
        output_path: Destination GIF path.
        mode_ids: Optional per-rollout mode index for agent coloring.
            Shape (num_rollouts,).
        image_size: Side length of each rendered frame in pixels.
        frames_per_second: GIF playback rate.
    """
    if task_name not in TASK_DISPLAY_NAMES:
        raise ValueError(f"Unknown synthetic task: {task_name}")
    layout = get_task_layout(task_name=task_name)
    num_timesteps = trajectories.shape[1]
    duration_milliseconds = int(round(1000 / frames_per_second))
    frames: list[Image.Image] = []
    for timestep in range(num_timesteps):
        positions_at_timestep = trajectories[:, timestep, :]
        trails_up_to_timestep = trajectories[:, : timestep + 1, :]
        frame_array = _render_multi_agent_frame(
            positions=positions_at_timestep,
            trails=trails_up_to_timestep,
            mode_ids=mode_ids,
            layout=layout,
            image_size=image_size,
        )
        frames.append(Image.fromarray(frame_array))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_milliseconds,
        loop=0,
        optimize=False,
    )
