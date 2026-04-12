"""Tests for versatil.data.synthetic.visualization module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from versatil.data.synthetic.constants import SyntheticTaskName
from versatil.data.synthetic.task_layout import SyntheticTaskLayout
from versatil.data.synthetic.visualization import (
    _apply_plot_theme,
    _build_legend_handles,
    _draw_task_background,
    _mode_color_bgr,
    _render_multi_agent_frame,
    plot_trajectories_2d,
    save_rollouts_gif,
)


@pytest.fixture
def rollout_trajectory_factory(
    rng: np.random.Generator,
) -> Callable[..., np.ndarray]:
    def factory(
        num_trajectories: int = 3,
        num_timesteps: int = 5,
    ) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=(num_trajectories, num_timesteps, 2)).astype(
            np.float32
        )

    return factory


@pytest.fixture
def rollout_mode_id_factory() -> Callable[..., np.ndarray]:
    def factory(num_trajectories: int = 3, num_modes: int = 3) -> np.ndarray:
        return np.array(
            [index % num_modes for index in range(num_trajectories)],
            dtype=np.int64,
        )

    return factory


@pytest.fixture
def mock_layout_factory() -> Callable[..., MagicMock]:
    def factory(
        num_obstacles: int = 0,
        has_goal: bool = False,
        num_modes: int = 3,
    ) -> MagicMock:
        layout = MagicMock(spec=SyntheticTaskLayout)
        layout.obstacles = [
            (0.1 * index, 0.2, 0.1 * index + 0.05, 0.3)
            for index in range(num_obstacles)
        ]
        layout.goal = np.array([0.5, 0.5], dtype=np.float32) if has_goal else None
        layout.start = np.array([0.0, 0.0], dtype=np.float32)
        layout.num_modes = num_modes
        return layout

    return factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "font_exists, expected_addfont_calls",
    [(True, 1), (False, 0)],
)
def test_apply_plot_theme_registers_font_conditionally_and_sets_theme(
    font_exists: bool, expected_addfont_calls: int
):
    mock_font_path = MagicMock()
    mock_font_path.exists.return_value = font_exists
    mock_font_path.__str__.return_value = "/fake/font/path.ttf"
    with (
        patch(
            "versatil.data.synthetic.visualization.FONT_PATH",
            mock_font_path,
        ),
        patch(
            "versatil.data.synthetic.visualization.font_manager.fontManager.addfont"
        ) as mock_addfont,
        patch("versatil.data.synthetic.visualization.sns.set_theme") as mock_set_theme,
    ):
        _apply_plot_theme()

    assert mock_addfont.call_count == expected_addfont_calls
    mock_set_theme.assert_called_once_with(
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_obstacles, has_goal, expected_patch_count",
    [
        (0, False, 0),
        (0, True, 1),
        (2, False, 2),
        (2, True, 3),
    ],
)
def test_draw_task_background_adds_expected_patches_and_styles_axes(
    mock_layout_factory: Callable[..., MagicMock],
    num_obstacles: int,
    has_goal: bool,
    expected_patch_count: int,
):
    layout = mock_layout_factory(num_obstacles=num_obstacles, has_goal=has_goal)
    figure, axes = plt.subplots()
    _draw_task_background(axes=axes, layout=layout)

    assert len(axes.patches) == expected_patch_count
    assert axes.get_xlim() == (-0.02, 1.02)
    assert axes.get_ylim() == (-0.02, 1.02)
    assert axes.get_aspect() == 1.0
    assert len(axes.get_xticks()) == 0
    assert len(axes.get_yticks()) == 0
    plt.close(figure)


@pytest.mark.unit
@pytest.mark.parametrize(
    "has_goal, has_obstacles, expected_labels",
    [
        (False, False, ["Agent", "Trajectory"]),
        (True, False, ["Agent", "Trajectory", "Goal"]),
        (False, True, ["Agent", "Trajectory", "Obstacle"]),
        (True, True, ["Agent", "Trajectory", "Goal", "Obstacle"]),
    ],
)
def test_build_legend_handles_composition_matches_layout(
    mock_layout_factory: Callable[..., MagicMock],
    has_goal: bool,
    has_obstacles: bool,
    expected_labels: list[str],
):
    layout = mock_layout_factory(
        num_obstacles=1 if has_obstacles else 0,
        has_goal=has_goal,
    )

    handles = _build_legend_handles(layout=layout)

    assert [handle.get_label() for handle in handles] == expected_labels


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode_index, expected_color",
    [
        (0, (213, 155, 91)),
        (1, (107, 107, 224)),
        (2, (106, 174, 127)),
        (3, (92, 165, 212)),
        (4, (213, 155, 91)),
        (7, (92, 165, 212)),
    ],
)
def test_mode_color_bgr_returns_palette_color_with_modulo_wrap(
    mode_index: int, expected_color: tuple[int, int, int]
):
    assert _mode_color_bgr(mode_index=mode_index) == expected_color


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_obstacles, has_goal, trails_kind, num_agents, expected_polyline_calls",
    [
        (0, False, "none", 2, 0),
        (0, True, "single_point", 2, 0),
        (2, False, "multi_point", 3, 3),
        (2, True, "multi_point", 1, 1),
    ],
)
def test_render_multi_agent_frame_draws_expected_cv2_shapes(
    mock_layout_factory: Callable[..., MagicMock],
    num_obstacles: int,
    has_goal: bool,
    trails_kind: str,
    num_agents: int,
    expected_polyline_calls: int,
):
    layout = mock_layout_factory(num_obstacles=num_obstacles, has_goal=has_goal)
    image_size = 32
    positions = np.tile(np.array([0.5, 0.5], dtype=np.float32), (num_agents, 1))
    if trails_kind == "none":
        trails = None
    elif trails_kind == "single_point":
        trails = positions[:, np.newaxis, :]
    else:
        trails = np.tile(
            np.array([[0.1, 0.1], [0.5, 0.5]], dtype=np.float32),
            (num_agents, 1, 1),
        )

    with (
        patch("versatil.data.synthetic.visualization.cv2.rectangle") as mock_rectangle,
        patch("versatil.data.synthetic.visualization.cv2.circle") as mock_circle,
        patch("versatil.data.synthetic.visualization.cv2.polylines") as mock_polylines,
    ):
        image = _render_multi_agent_frame(
            positions=positions,
            trails=trails,
            mode_ids=None,
            layout=layout,
            image_size=image_size,
        )

    assert image.shape == (image_size, image_size, 3)
    assert image.dtype == np.uint8
    assert mock_rectangle.call_count == num_obstacles
    expected_circle_calls = (1 if has_goal else 0) + num_agents
    assert mock_circle.call_count == expected_circle_calls
    assert mock_polylines.call_count == expected_polyline_calls


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_trajectories, num_experts, custom_title, expected_scatter_calls",
    [
        (3, 0, None, 1),
        (3, 4, "Custom", 1),
        (0, 0, None, 0),
        (0, 2, None, 0),
    ],
)
def test_plot_trajectories_2d_delegates_drawing_and_writes_output(
    mock_layout_factory: Callable[..., MagicMock],
    rollout_trajectory_factory: Callable[..., np.ndarray],
    rollout_mode_id_factory: Callable[..., np.ndarray],
    tmp_path: Path,
    num_trajectories: int,
    num_experts: int,
    custom_title: str | None,
    expected_scatter_calls: int,
):
    layout = mock_layout_factory(num_obstacles=2, has_goal=True)
    trajectories = rollout_trajectory_factory(num_trajectories=num_trajectories)
    mode_ids = (
        rollout_mode_id_factory(num_trajectories=num_trajectories)
        if num_trajectories > 0
        else None
    )
    expert_trajectories = (
        rollout_trajectory_factory(num_trajectories=num_experts)
        if num_experts > 0
        else None
    )
    expert_mode_ids = (
        rollout_mode_id_factory(num_trajectories=num_experts)
        if num_experts > 0
        else None
    )
    output_path = tmp_path / "out.png"
    task_name = SyntheticTaskName.MULTI_PATH_NAVIGATION.value

    mock_figure = MagicMock(spec=plt.Figure)
    mock_axes = MagicMock(spec=plt.Axes)

    with (
        patch("versatil.data.synthetic.visualization._apply_plot_theme") as mock_theme,
        patch(
            "versatil.data.synthetic.visualization.get_task_layout",
            return_value=layout,
        ) as mock_get_layout,
        patch(
            "versatil.data.synthetic.visualization._draw_task_background"
        ) as mock_draw_bg,
        patch(
            "versatil.data.synthetic.visualization._build_legend_handles",
            return_value=["fake_handle"],
        ) as mock_build_legend,
        patch(
            "versatil.data.synthetic.visualization.plt.subplots",
            return_value=(mock_figure, mock_axes),
        ),
        patch("versatil.data.synthetic.visualization.plt.tight_layout"),
        patch("versatil.data.synthetic.visualization.plt.savefig") as mock_savefig,
        patch("versatil.data.synthetic.visualization.plt.close") as mock_close,
    ):
        plot_trajectories_2d(
            trajectories=trajectories,
            task_name=task_name,
            output_path=str(output_path),
            mode_ids=mode_ids,
            expert_trajectories=expert_trajectories,
            expert_mode_ids=expert_mode_ids,
            title=custom_title,
        )

    mock_theme.assert_called_once()
    mock_get_layout.assert_called_once_with(task_name=task_name)
    mock_draw_bg.assert_called_once_with(axes=mock_axes, layout=layout)
    mock_build_legend.assert_called_once_with(layout=layout)

    assert mock_axes.plot.call_count == num_trajectories + num_experts
    assert mock_axes.scatter.call_count == expected_scatter_calls

    mock_axes.legend.assert_called_once()
    assert mock_axes.legend.call_args.kwargs["handles"] == ["fake_handle"]

    expected_title = (
        custom_title if custom_title is not None else "Multi-Path Navigation"
    )
    mock_axes.set_title.assert_called_once_with(expected_title, pad=12)
    mock_savefig.assert_called_once()
    assert mock_savefig.call_args.args[0] == str(output_path)
    mock_close.assert_called_once_with(mock_figure)


@pytest.mark.unit
@pytest.mark.parametrize(
    "task_name, expectation",
    [
        (
            SyntheticTaskName.MULTI_PATH_NAVIGATION.value,
            does_not_raise(),
        ),
        (
            "not_a_task",
            pytest.raises(
                ValueError,
                match=re.escape("Unknown synthetic task: not_a_task"),
            ),
        ),
    ],
)
def test_plot_trajectories_2d_validates_task_name(
    rollout_trajectory_factory: Callable[..., np.ndarray],
    tmp_path: Path,
    task_name: str,
    expectation,
):
    output_path = tmp_path / "out.png"

    with expectation:
        plot_trajectories_2d(
            trajectories=rollout_trajectory_factory(),
            task_name=task_name,
            output_path=str(output_path),
        )
    plt.close("all")


@pytest.mark.unit
@pytest.mark.parametrize(
    "num_timesteps, frames_per_second, expected_duration_ms",
    [
        (3, 30, 33),
        (5, 20, 50),
    ],
)
def test_save_rollouts_gif_renders_frame_per_timestep_and_saves(
    mock_layout_factory: Callable[..., MagicMock],
    rollout_trajectory_factory: Callable[..., np.ndarray],
    rollout_mode_id_factory: Callable[..., np.ndarray],
    tmp_path: Path,
    num_timesteps: int,
    frames_per_second: int,
    expected_duration_ms: int,
):
    layout = mock_layout_factory(num_obstacles=1, has_goal=True)
    trajectories = rollout_trajectory_factory(
        num_trajectories=2, num_timesteps=num_timesteps
    )
    mode_ids = rollout_mode_id_factory(num_trajectories=2)
    output_path = tmp_path / "rollouts.gif"
    task_name = SyntheticTaskName.MULTI_PATH_NAVIGATION.value
    image_size = 16

    fake_frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    fake_image = MagicMock()

    with (
        patch(
            "versatil.data.synthetic.visualization.get_task_layout",
            return_value=layout,
        ) as mock_get_layout,
        patch(
            "versatil.data.synthetic.visualization._render_multi_agent_frame",
            return_value=fake_frame,
        ) as mock_render,
        patch(
            "versatil.data.synthetic.visualization.Image.fromarray",
            return_value=fake_image,
        ) as mock_fromarray,
    ):
        save_rollouts_gif(
            trajectories=trajectories,
            task_name=task_name,
            output_path=str(output_path),
            mode_ids=mode_ids,
            image_size=image_size,
            frames_per_second=frames_per_second,
        )

    mock_get_layout.assert_called_once_with(task_name=task_name)
    assert mock_render.call_count == num_timesteps
    for timestep, call in enumerate(mock_render.call_args_list):
        positions_arg = call.kwargs["positions"]
        trails_arg = call.kwargs["trails"]
        np.testing.assert_array_equal(positions_arg, trajectories[:, timestep, :])
        np.testing.assert_array_equal(trails_arg, trajectories[:, : timestep + 1, :])
        assert call.kwargs["layout"] is layout
        assert call.kwargs["image_size"] == image_size
        assert call.kwargs["mode_ids"] is mode_ids
    assert mock_fromarray.call_count == num_timesteps

    fake_image.save.assert_called_once()
    save_kwargs = fake_image.save.call_args.kwargs
    assert fake_image.save.call_args.args[0] == str(output_path)
    assert save_kwargs["save_all"] is True
    assert save_kwargs["duration"] == expected_duration_ms
    assert save_kwargs["loop"] == 0
    assert len(save_kwargs["append_images"]) == num_timesteps - 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "task_name, expectation",
    [
        (
            SyntheticTaskName.MULTI_PATH_NAVIGATION.value,
            does_not_raise(),
        ),
        (
            "not_a_task",
            pytest.raises(
                ValueError,
                match=re.escape("Unknown synthetic task: not_a_task"),
            ),
        ),
    ],
)
def test_save_rollouts_gif_validates_task_name(
    mock_layout_factory: Callable[..., MagicMock],
    rollout_trajectory_factory: Callable[..., np.ndarray],
    tmp_path: Path,
    task_name: str,
    expectation,
):
    layout = mock_layout_factory(num_obstacles=0, has_goal=True)
    fake_frame = np.zeros((16, 16, 3), dtype=np.uint8)

    with (
        patch(
            "versatil.data.synthetic.visualization.get_task_layout",
            return_value=layout,
        ),
        patch(
            "versatil.data.synthetic.visualization._render_multi_agent_frame",
            return_value=fake_frame,
        ),
        patch(
            "versatil.data.synthetic.visualization.Image.fromarray",
            return_value=MagicMock(),
        ),
        expectation,
    ):
        save_rollouts_gif(
            trajectories=rollout_trajectory_factory(num_timesteps=3),
            task_name=task_name,
            output_path=str(tmp_path / "out.gif"),
            image_size=16,
            frames_per_second=30,
        )
