"""Tests for versatil.inference.synthetic_rollout module."""

import re
from collections.abc import Callable
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.constants import Cameras, ProprioKey, SyntheticObsKey
from versatil.data.synthetic.constants import CIRCLE_CONTEXT_COLORS, SyntheticTaskName
from versatil.data.synthetic.task_layout import SyntheticTaskLayout
from versatil.inference.synthetic_rollout import (
    _get_render_goal,
    _prepare_observation,
    _save_rollout_visualizations,
    evaluate_rollouts,
    load_policy_from_checkpoint,
    run_rollouts,
)


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


@pytest.fixture
def position_history_factory(
    rng: np.random.Generator,
) -> Callable[..., np.ndarray]:
    def factory(obs_horizon: int = 1) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=(obs_horizon, 2)).astype(np.float32)

    return factory


@pytest.fixture
def rollout_trajectory_factory(
    rng: np.random.Generator,
) -> Callable[..., np.ndarray]:
    def factory(
        num_rollouts: int = 2,
        num_timesteps: int = 5,
    ) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=(num_rollouts, num_timesteps, 2)).astype(
            np.float32
        )

    return factory


@pytest.fixture
def mock_expert_episodes_factory(
    rng: np.random.Generator,
) -> Callable[..., list[dict[str, np.ndarray]]]:
    def factory(
        num_episodes: int = 4,
        num_timesteps: int = 5,
        num_modes: int = 3,
    ) -> list[dict[str, np.ndarray]]:
        episodes = []
        for episode_index in range(num_episodes):
            mode_index = episode_index % num_modes
            episodes.append(
                {
                    "position": rng.uniform(0.0, 1.0, size=(num_timesteps, 2)).astype(
                        np.float32
                    ),
                    "mode_id": np.full((num_timesteps, 1), mode_index, dtype=np.int64),
                }
            )
        return episodes

    return factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "has_goal, task_name, expected_goal, expectation",
    [
        (
            True,
            SyntheticTaskName.CIRCLE.value,
            np.array([0.5, 0.5], dtype=np.float32),
            does_not_raise(),
        ),
        (
            False,
            SyntheticTaskName.SEQUENTIAL_DECISION.value,
            np.array([0.5, 0.95], dtype=np.float32),
            does_not_raise(),
        ),
        (
            False,
            SyntheticTaskName.RADIAL.value,
            np.array([0.5, 0.5], dtype=np.float32),
            does_not_raise(),
        ),
        (
            False,
            SyntheticTaskName.CIRCLE.value,
            None,
            does_not_raise(),
        ),
        (
            False,
            SyntheticTaskName.CONDITIONAL_CIRCLE.value,
            None,
            does_not_raise(),
        ),
    ],
)
def test_get_render_goal_returns_layout_goal_or_task_fallback(
    mock_layout_factory: Callable[..., MagicMock],
    has_goal: bool,
    task_name: str,
    expected_goal: np.ndarray | None,
    expectation: AbstractContextManager,
):
    layout = mock_layout_factory(has_goal=has_goal)

    with expectation:
        result = _get_render_goal(layout=layout, task_name=task_name)
        np.testing.assert_array_equal(result, expected_goal)


@pytest.mark.unit
@pytest.mark.parametrize("obs_horizon", [1, 3])
@pytest.mark.parametrize(
    "has_image, has_position, has_context, context_vector_provided",
    [
        (True, True, False, False),
        (True, False, True, True),
        (False, True, True, False),
        (True, True, True, True),
        (False, False, False, False),
    ],
)
def test_prepare_observation_assembles_keys_per_observation_space(
    position_history_factory: Callable[..., np.ndarray],
    obs_horizon: int,
    has_image: bool,
    has_position: bool,
    has_context: bool,
    context_vector_provided: bool,
):
    observation_keys: set[str] = set()
    if has_image:
        observation_keys.add(Cameras.AGENTVIEW.value)
    if has_position:
        observation_keys.add(ProprioKey.SYNTHETIC_POSITION.value)
    if has_context:
        observation_keys.add(SyntheticObsKey.CONTEXT.value)

    position_history = position_history_factory(obs_horizon=obs_horizon)
    obstacles: list[tuple[float, float, float, float]] = []
    goal = np.array([1.0, 1.0], dtype=np.float32)
    image_size = 16
    num_context_channels = 3
    context_vector = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32) if context_vector_provided else None
    )

    fake_frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    with patch(
        "versatil.inference.synthetic_rollout.render_frame",
        return_value=fake_frame,
    ) as mock_render_frame:
        observation = _prepare_observation(
            position_history=position_history,
            obstacles=obstacles,
            goal=goal,
            image_size=image_size,
            observation_keys=observation_keys,
            context_vector=context_vector,
        )

    if has_image:
        assert mock_render_frame.call_count == obs_horizon
        image_tensor = observation[Cameras.AGENTVIEW.value]
        assert image_tensor.shape == (1, obs_horizon, 3, image_size, image_size)
        assert image_tensor.dtype == torch.float32
    else:
        assert mock_render_frame.call_count == 0
        assert Cameras.AGENTVIEW.value not in observation

    if has_position:
        position_tensor = observation[ProprioKey.SYNTHETIC_POSITION.value]
        assert position_tensor.shape == (1, obs_horizon, 2)
        assert position_tensor.dtype == torch.float32
        np.testing.assert_array_equal(position_tensor[0].numpy(), position_history)
    else:
        assert ProprioKey.SYNTHETIC_POSITION.value not in observation

    if has_context and context_vector_provided:
        context_tensor = observation[SyntheticObsKey.CONTEXT.value]
        assert context_tensor.shape == (1, obs_horizon, num_context_channels)
        assert context_tensor.dtype == torch.float32
        for timestep in range(obs_horizon):
            np.testing.assert_array_equal(
                context_tensor[0, timestep].numpy(), context_vector
            )
    else:
        assert SyntheticObsKey.CONTEXT.value not in observation


@pytest.mark.unit
@pytest.mark.parametrize(
    "context_color",
    [None, (255, 0, 0), (0, 0, 255)],
)
def test_prepare_observation_forwards_context_color_to_render_frame(
    position_history_factory: Callable[..., np.ndarray],
    context_color: tuple[int, int, int] | None,
):
    position_history = position_history_factory(obs_horizon=2)
    observation_keys = {Cameras.AGENTVIEW.value}
    fake_frame = np.zeros((16, 16, 3), dtype=np.uint8)

    with patch(
        "versatil.inference.synthetic_rollout.render_frame",
        return_value=fake_frame,
    ) as mock_render:
        _prepare_observation(
            position_history=position_history,
            obstacles=[],
            goal=None,
            image_size=16,
            observation_keys=observation_keys,
            context_color=context_color,
        )

    for call in mock_render.call_args_list:
        assert call.kwargs["context_color"] == context_color


@pytest.mark.unit
def test_prepare_observation_passes_none_goal_to_render_frame(
    position_history_factory: Callable[..., np.ndarray],
):
    position_history = position_history_factory(obs_horizon=1)
    observation_keys = {Cameras.AGENTVIEW.value}
    fake_frame = np.zeros((16, 16, 3), dtype=np.uint8)

    with patch(
        "versatil.inference.synthetic_rollout.render_frame",
        return_value=fake_frame,
    ) as mock_render:
        _prepare_observation(
            position_history=position_history,
            obstacles=[],
            goal=None,
            image_size=16,
            observation_keys=observation_keys,
        )

    assert mock_render.call_args.kwargs["goal"] is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "name_prefix, subdir",
    [
        ("rollout", ""),
        ("rollout_temporal_agg", "nested/viz"),
    ],
)
def test_save_rollout_visualizations_delegates_to_png_and_gif(
    rollout_trajectory_factory: Callable[..., np.ndarray],
    tmp_path: Path,
    name_prefix: str,
    subdir: str,
):
    trajectories = rollout_trajectory_factory()
    task_name = SyntheticTaskName.CIRCLE.value
    image_size = 32
    output_dir = tmp_path / subdir if subdir else tmp_path

    with (
        patch("versatil.inference.synthetic_rollout.plot_trajectories_2d") as mock_plot,
        patch(
            "versatil.inference.synthetic_rollout.save_rollouts_gif"
        ) as mock_save_gif,
    ):
        _save_rollout_visualizations(
            trajectories=trajectories,
            task_name=task_name,
            output_dir=str(output_dir),
            name_prefix=name_prefix,
            image_size=image_size,
        )

    assert output_dir.exists()
    expected_png = str(output_dir / f"{name_prefix}_{task_name}.png")
    expected_gif = str(output_dir / f"{name_prefix}_{task_name}.gif")
    mock_plot.assert_called_once_with(
        trajectories=trajectories,
        task_name=task_name,
        output_path=expected_png,
    )
    mock_save_gif.assert_called_once_with(
        trajectories=trajectories,
        task_name=task_name,
        output_path=expected_gif,
        image_size=image_size,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "has_goal, rollout_length, expert_length, expected_comparison_length",
    [
        (True, 5, 5, 5),
        (True, 8, 5, 5),
        (False, 4, 6, 4),
        (False, 5, 5, 5),
    ],
)
def test_evaluate_rollouts_truncates_and_calls_metrics_with_layout(
    rollout_trajectory_factory: Callable[..., np.ndarray],
    mock_expert_episodes_factory: Callable[..., list[dict[str, np.ndarray]]],
    mock_layout_factory: Callable[..., MagicMock],
    has_goal: bool,
    rollout_length: int,
    expert_length: int,
    expected_comparison_length: int,
):
    num_rollouts = 3
    num_expert_episodes = 4
    num_modes = 3
    rollout_trajectories = rollout_trajectory_factory(
        num_rollouts=num_rollouts,
        num_timesteps=rollout_length,
    )
    expert_episodes = mock_expert_episodes_factory(
        num_episodes=num_expert_episodes,
        num_timesteps=expert_length,
        num_modes=num_modes,
    )
    layout = mock_layout_factory(has_goal=has_goal, num_modes=num_modes)
    task_name = SyntheticTaskName.CIRCLE.value
    coverage_metrics = {
        "mode_coverage": 0.8,
        "mode_entropy_ratio": 0.7,
        "per_mode_count": {0: 1, 1: 1, 2: 1},
    }
    success_stats = {
        "success_rate": 0.7,
        "collision_rate": 0.1,
        "endpoint_reach_rate": 0.8,
    }
    mode_endpoints = np.zeros((num_modes, 2), dtype=np.float32)

    with (
        patch(
            "versatil.inference.synthetic_rollout.generate_task_episodes",
            return_value=expert_episodes,
        ) as mock_generate,
        patch(
            "versatil.inference.synthetic_rollout.get_task_layout",
            return_value=layout,
        ) as mock_get_layout,
        patch(
            "versatil.inference.synthetic_rollout.compute_mode_coverage",
            return_value=coverage_metrics,
        ) as mock_mode_coverage,
        patch(
            "versatil.inference.synthetic_rollout.compute_mode_endpoints",
            return_value=mode_endpoints,
        ),
        patch(
            "versatil.inference.synthetic_rollout.compute_success_rate",
            return_value=success_stats,
        ) as mock_success,
    ):
        results = evaluate_rollouts(
            rollout_trajectories=rollout_trajectories,
            task_name=task_name,
            num_expert_episodes=num_expert_episodes,
            expert_seed=7,
            trajectory_length=expert_length,
            noise_std=0.02,
            num_modes=num_modes,
            num_styles=4,
            image_size=32,
        )

    mock_generate.assert_called_once_with(
        task_name=task_name,
        num_episodes=num_expert_episodes,
        seed=7,
        image_size=32,
        num_modes=num_modes,
        trajectory_length=expert_length,
        noise_std=0.02,
        num_styles=4,
    )
    mock_get_layout.assert_called_once_with(
        task_name=task_name, num_modes=num_modes, num_styles=4, noise_std=0.02
    )

    mock_mode_coverage.assert_called_once()
    coverage_kwargs = mock_mode_coverage.call_args.kwargs
    assert coverage_kwargs["num_modes"] == num_modes
    assert coverage_kwargs["generated_trajectories"].shape == (
        num_rollouts,
        expected_comparison_length,
        2,
    )
    assert coverage_kwargs["expert_trajectories"].shape == (
        num_expert_episodes,
        expected_comparison_length,
        2,
    )
    assert coverage_kwargs["expert_mode_ids"].shape == (num_expert_episodes,)
    # valid_mask has one entry per rollout and excludes collided trajectories
    assert coverage_kwargs["valid_mask"].shape == (num_rollouts,)
    assert coverage_kwargs["valid_mask"].dtype == bool
    np.testing.assert_array_equal(
        coverage_kwargs["expert_mode_ids"],
        np.array(
            [index % num_modes for index in range(num_expert_episodes)],
            dtype=np.int64,
        ),
    )

    for metric_key, metric_value in coverage_metrics.items():
        assert results[metric_key] == metric_value

    assert "goal_success_rate" not in results

    mock_success.assert_called_once()
    for stat_key, stat_value in success_stats.items():
        assert results[stat_key] == stat_value


@pytest.mark.unit
@pytest.mark.parametrize(
    "create_config, create_checkpoint, expectation",
    [
        (
            False,
            False,
            pytest.raises(FileNotFoundError, match=re.escape("Config not found at ")),
        ),
        (
            True,
            False,
            pytest.raises(
                FileNotFoundError, match=re.escape("Checkpoint not found at ")
            ),
        ),
        (True, True, does_not_raise()),
    ],
)
def test_load_policy_from_checkpoint_validates_files_and_loads_weights(
    mock_policy_factory: Callable[..., MagicMock],
    tmp_path: Path,
    create_config: bool,
    create_checkpoint: bool,
    expectation: AbstractContextManager,
):
    if create_config:
        (tmp_path / "config.yaml").write_text("placeholder: true\n")
    if create_checkpoint:
        (tmp_path / "last.ckpt").write_bytes(b"placeholder")
    mock_policy = mock_policy_factory()
    mock_config = MagicMock()
    mock_config.policy = mock_policy
    fake_checkpoint = {"state_dict": {"layer.weight": MagicMock()}}
    mock_lightning_instance = MagicMock()

    with (
        patch(
            "versatil.inference.synthetic_rollout.OmegaConf.load",
            return_value=MagicMock(),
        ),
        patch(
            "versatil.inference.synthetic_rollout.hydra.utils.instantiate",
            return_value=mock_config,
        ),
        patch(
            "versatil.inference.synthetic_rollout.torch.load",
            return_value=fake_checkpoint,
        ) as mock_torch_load,
        patch(
            "versatil.inference.synthetic_rollout.LightningPolicy",
            return_value=mock_lightning_instance,
        ) as mock_lightning_cls,
        expectation,
    ):
        policy, config = load_policy_from_checkpoint(
            checkpoint_path=str(tmp_path),
            device="cpu",
            checkpoint_name="last.ckpt",
        )

    if create_config and create_checkpoint:
        assert policy is mock_policy
        assert config is mock_config
        mock_policy.to.assert_called_once_with("cpu")
        mock_policy.to.return_value.eval.assert_called_once()
        mock_torch_load.assert_called_once_with(
            str(tmp_path / "last.ckpt"),
            map_location="cpu",
            weights_only=False,
        )
        mock_lightning_cls.assert_called_once_with(
            policy=mock_policy, training_config=mock_config.training
        )
        mock_lightning_instance.load_state_dict.assert_called_once_with(
            fake_checkpoint["state_dict"], strict=False
        )


@pytest.fixture
def constant_action_chunk_factory() -> Callable[..., dict[str, torch.Tensor]]:
    def factory(
        prediction_horizon: int = 4,
        delta: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        action_tensor = torch.full(
            (1, prediction_horizon, 2), delta, dtype=torch.float32
        )
        return {ProprioKey.SYNTHETIC_POSITION_ACTION.value: action_tensor}

    return factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "temporal_aggregation, obs_horizon, prediction_horizon, "
    "expected_query_count_per_rollout",
    [
        (True, 1, 4, 4),
        (True, 2, 4, 3),
        (False, 1, 4, 1),
        (False, 2, 4, 1),
    ],
)
def test_run_rollouts_queries_policy_per_temporal_mode_and_obs_horizon(
    mock_policy_factory: Callable[..., MagicMock],
    mock_layout_factory: Callable[..., MagicMock],
    constant_action_chunk_factory: Callable[..., dict[str, torch.Tensor]],
    temporal_aggregation: bool,
    obs_horizon: int,
    prediction_horizon: int,
    expected_query_count_per_rollout: int,
):
    num_rollouts = 2
    layout = mock_layout_factory(has_goal=True, num_modes=3)
    mock_policy = mock_policy_factory(
        prediction_horizon=prediction_horizon,
        observation_horizon=obs_horizon,
        observations_metadata={
            Cameras.AGENTVIEW.value: MagicMock(),
            ProprioKey.SYNTHETIC_POSITION.value: MagicMock(),
        },
        predict_action_return=constant_action_chunk_factory(
            prediction_horizon=prediction_horizon
        ),
    )

    with (
        patch(
            "versatil.inference.synthetic_rollout.get_task_layout",
            return_value=layout,
        ),
        patch(
            "versatil.inference.synthetic_rollout._get_render_goal",
            return_value=np.array([1.0, 1.0], dtype=np.float32),
        ),
        patch(
            "versatil.inference.synthetic_rollout._prepare_observation",
            return_value={},
        ),
    ):
        trajectories = run_rollouts(
            policy=mock_policy,
            task_name=SyntheticTaskName.CIRCLE.value,
            num_rollouts=num_rollouts,
            image_size=16,
            temporal_aggregation=temporal_aggregation,
        )

    assert (
        mock_policy.predict_action.call_count
        == expected_query_count_per_rollout * num_rollouts
    )
    assert trajectories.shape == (num_rollouts, prediction_horizon + 1, 2)
    assert trajectories.dtype == np.float32


@pytest.mark.unit
@pytest.mark.parametrize(
    "context_mode, num_modes, expected_context_vector",
    [
        (None, 3, None),
        (0, 3, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        (2, 3, np.array([0.0, 0.0, 1.0], dtype=np.float32)),
    ],
)
def test_run_rollouts_builds_one_hot_context_vector(
    mock_policy_factory: Callable[..., MagicMock],
    mock_layout_factory: Callable[..., MagicMock],
    constant_action_chunk_factory: Callable[..., dict[str, torch.Tensor]],
    context_mode: int | None,
    num_modes: int,
    expected_context_vector: np.ndarray | None,
):
    prediction_horizon = 3
    layout = mock_layout_factory(has_goal=True, num_modes=num_modes)
    mock_policy = mock_policy_factory(
        prediction_horizon=prediction_horizon,
        observation_horizon=1,
        observations_metadata={
            ProprioKey.SYNTHETIC_POSITION.value: MagicMock(),
        },
        predict_action_return=constant_action_chunk_factory(
            prediction_horizon=prediction_horizon
        ),
    )

    with (
        patch(
            "versatil.inference.synthetic_rollout.get_task_layout",
            return_value=layout,
        ),
        patch(
            "versatil.inference.synthetic_rollout._get_render_goal",
            return_value=np.array([1.0, 1.0], dtype=np.float32),
        ),
        patch(
            "versatil.inference.synthetic_rollout._prepare_observation",
            return_value={},
        ) as mock_prepare,
    ):
        run_rollouts(
            policy=mock_policy,
            task_name=SyntheticTaskName.CONDITIONAL_CIRCLE.value,
            num_rollouts=1,
            image_size=16,
            context_mode=context_mode,
            temporal_aggregation=False,
        )

    first_call_kwargs = mock_prepare.call_args_list[0].kwargs
    context_vector_arg = first_call_kwargs["context_vector"]
    context_color_arg = first_call_kwargs["context_color"]
    if expected_context_vector is None:
        assert context_vector_arg is None
        assert context_color_arg is None
    else:
        np.testing.assert_array_equal(context_vector_arg, expected_context_vector)
        # context_color is None for modes without a registered color
        expected_color = CIRCLE_CONTEXT_COLORS.get(context_mode)
        assert context_color_arg == expected_color


@pytest.mark.unit
@pytest.mark.parametrize(
    "output_dir_set, temporal_aggregation, expected_name_prefix",
    [
        (False, True, None),
        (False, False, None),
        (True, True, "rollout_temporal_agg"),
        (True, False, "rollout"),
    ],
)
def test_run_rollouts_saves_visualizations_only_when_output_dir_set(
    mock_policy_factory: Callable[..., MagicMock],
    mock_layout_factory: Callable[..., MagicMock],
    constant_action_chunk_factory: Callable[..., dict[str, torch.Tensor]],
    tmp_path: Path,
    output_dir_set: bool,
    temporal_aggregation: bool,
    expected_name_prefix: str | None,
):
    prediction_horizon = 3
    image_size = 16
    layout = mock_layout_factory(has_goal=True, num_modes=3)
    mock_policy = mock_policy_factory(
        prediction_horizon=prediction_horizon,
        observation_horizon=1,
        observations_metadata={
            ProprioKey.SYNTHETIC_POSITION.value: MagicMock(),
        },
        predict_action_return=constant_action_chunk_factory(
            prediction_horizon=prediction_horizon
        ),
    )
    output_dir = str(tmp_path) if output_dir_set else None
    task_name = SyntheticTaskName.CIRCLE.value

    with (
        patch(
            "versatil.inference.synthetic_rollout.get_task_layout",
            return_value=layout,
        ),
        patch(
            "versatil.inference.synthetic_rollout._get_render_goal",
            return_value=np.array([1.0, 1.0], dtype=np.float32),
        ),
        patch(
            "versatil.inference.synthetic_rollout._prepare_observation",
            return_value={},
        ),
        patch(
            "versatil.inference.synthetic_rollout._save_rollout_visualizations"
        ) as mock_save_viz,
    ):
        run_rollouts(
            policy=mock_policy,
            task_name=task_name,
            num_rollouts=1,
            image_size=image_size,
            temporal_aggregation=temporal_aggregation,
            output_dir=output_dir,
        )

    if expected_name_prefix is None:
        mock_save_viz.assert_not_called()
    else:
        mock_save_viz.assert_called_once()
        viz_kwargs = mock_save_viz.call_args.kwargs
        assert viz_kwargs["task_name"] == task_name
        assert viz_kwargs["output_dir"] == output_dir
        assert viz_kwargs["name_prefix"] == expected_name_prefix
        assert viz_kwargs["image_size"] == image_size


@pytest.mark.unit
@pytest.mark.parametrize("temporal_aggregation", [True, False])
def test_run_rollouts_integrates_constant_action_deltas_into_positions(
    mock_policy_factory: Callable[..., MagicMock],
    mock_layout_factory: Callable[..., MagicMock],
    constant_action_chunk_factory: Callable[..., dict[str, torch.Tensor]],
    temporal_aggregation: bool,
):
    prediction_horizon = 4
    delta = 0.1
    start = np.array([0.0, 0.0], dtype=np.float32)
    layout = mock_layout_factory(has_goal=True, num_modes=3)
    layout.start = start
    mock_policy = mock_policy_factory(
        prediction_horizon=prediction_horizon,
        observation_horizon=1,
        observations_metadata={
            ProprioKey.SYNTHETIC_POSITION.value: MagicMock(),
        },
        predict_action_return=constant_action_chunk_factory(
            prediction_horizon=prediction_horizon, delta=delta
        ),
    )

    with (
        patch(
            "versatil.inference.synthetic_rollout.get_task_layout",
            return_value=layout,
        ),
        patch(
            "versatil.inference.synthetic_rollout._get_render_goal",
            return_value=np.array([1.0, 1.0], dtype=np.float32),
        ),
        patch(
            "versatil.inference.synthetic_rollout._prepare_observation",
            return_value={},
        ),
    ):
        trajectories = run_rollouts(
            policy=mock_policy,
            task_name=SyntheticTaskName.CIRCLE.value,
            num_rollouts=1,
            image_size=16,
            temporal_aggregation=temporal_aggregation,
        )

    np.testing.assert_array_equal(trajectories[0, 0], start)
    for step in range(prediction_horizon):
        expected_position = np.clip(start + delta * (step + 1), 0.0, 1.0)
        np.testing.assert_allclose(
            trajectories[0, step + 1], expected_position, atol=1e-6
        )
