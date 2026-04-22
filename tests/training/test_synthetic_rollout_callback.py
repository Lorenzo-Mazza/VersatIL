"""Tests for versatil.training.synthetic_rollout_callback module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from versatil.data.constants import SyntheticObsKey
from versatil.data.synthetic.constants import SyntheticTaskName
from versatil.training.synthetic_rollout_callback import SyntheticRolloutCallback


@pytest.fixture
def callback_factory() -> Callable[..., SyntheticRolloutCallback]:
    def factory(
        task_name: str = SyntheticTaskName.CIRCLE.value,
        num_modes: int = 2,
        num_styles: int = 1,
        trajectory_length: int = 60,
        noise_std: float = 0.008,
        num_rollouts: int = 10,
        image_size: int = 32,
        log_every_n_epochs: int = 1,
    ) -> SyntheticRolloutCallback:
        return SyntheticRolloutCallback(
            task_name=task_name,
            num_modes=num_modes,
            num_styles=num_styles,
            trajectory_length=trajectory_length,
            noise_std=noise_std,
            num_rollouts=num_rollouts,
            image_size=image_size,
            log_every_n_epochs=log_every_n_epochs,
        )

    return factory


@pytest.fixture
def mock_trainer_factory() -> Callable[..., MagicMock]:
    def factory(
        current_epoch: int = 0,
        has_logger: bool = True,
        max_epochs: int = 2000,
    ) -> MagicMock:
        trainer = MagicMock()
        trainer.current_epoch = current_epoch
        trainer.max_epochs = max_epochs
        if not has_logger:
            trainer.logger = None
        return trainer

    return factory


@pytest.fixture
def mock_pl_module_factory() -> Callable[..., MagicMock]:
    def factory(training: bool = True) -> MagicMock:
        pl_module = MagicMock()
        pl_module.policy.training = training
        return pl_module

    return factory


@pytest.fixture
def fake_trajectories_factory(
    rng: np.random.Generator,
) -> Callable[..., np.ndarray]:
    def factory(
        num_rollouts: int = 10,
        num_timesteps: int = 10,
    ) -> np.ndarray:
        return rng.uniform(0.0, 1.0, size=(num_rollouts, num_timesteps, 2)).astype(
            np.float32
        )

    return factory


@pytest.fixture
def fake_results_factory() -> Callable[..., dict]:
    def factory(
        mode_coverage: float = 0.67,
        entropy_ratio: float = 0.85,
        per_mode_count: dict[int, int] | None = None,
        success_rate: float = 0.75,
        collision_rate: float = 0.1,
        endpoint_reach_rate: float = 0.85,
        path_length_rate: float = 0.9,
    ) -> dict:
        return {
            "mode_coverage": mode_coverage,
            "mode_entropy_ratio": entropy_ratio,
            "per_mode_count": per_mode_count
            if per_mode_count is not None
            else {0: 4, 1: 3, 2: 3},
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "endpoint_reach_rate": endpoint_reach_rate,
            "path_length_rate": path_length_rate,
        }

    return factory


def _patch_callback_dependencies(
    fake_trajectories: np.ndarray,
    fake_results: dict,
):
    """Context manager that patches run_rollouts, evaluate_rollouts, and wandb/plt."""
    return (
        patch(
            "versatil.training.synthetic_rollout_callback.run_rollouts",
            return_value=fake_trajectories,
        ),
        patch(
            "versatil.training.synthetic_rollout_callback.evaluate_rollouts",
            return_value=fake_results,
        ),
        patch(
            "versatil.training.synthetic_rollout_callback.plot_trajectories_2d",
            return_value=MagicMock(),
        ),
        patch("versatil.training.synthetic_rollout_callback.plt.close"),
        patch("versatil.training.synthetic_rollout_callback.Image.open"),
        patch("versatil.training.synthetic_rollout_callback.wandb.Image"),
    )


@pytest.mark.unit
@pytest.mark.parametrize("num_rollouts", [10, 50])
@pytest.mark.parametrize("image_size", [32, 64])
def test_stores_configuration(
    callback_factory: Callable[..., SyntheticRolloutCallback],
    num_rollouts: int,
    image_size: int,
):
    task_name = SyntheticTaskName.CORRIDOR_NAVIGATION.value
    log_every_n_epochs = 3
    callback = callback_factory(
        task_name=task_name,
        num_rollouts=num_rollouts,
        image_size=image_size,
        log_every_n_epochs=log_every_n_epochs,
    )
    assert callback.task_name == task_name
    assert callback.num_rollouts == num_rollouts
    assert callback.image_size == image_size
    assert callback.log_every_n_epochs == log_every_n_epochs


@pytest.mark.unit
@pytest.mark.parametrize(
    "current_epoch, log_every_n_epochs, max_epochs, should_run",
    [
        (0, 1, 100, True),
        (1, 1, 100, True),
        (3, 5, 100, False),
        (5, 5, 100, True),
        (7, 3, 100, False),
        (9, 3, 100, True),
        (1999, 100, 2000, True),
        (1998, 100, 2000, False),
    ],
)
def test_epoch_gating(
    callback_factory: Callable[..., SyntheticRolloutCallback],
    mock_trainer_factory: Callable[..., MagicMock],
    mock_pl_module_factory: Callable[..., MagicMock],
    fake_trajectories_factory: Callable[..., np.ndarray],
    fake_results_factory: Callable[..., dict],
    current_epoch: int,
    log_every_n_epochs: int,
    max_epochs: int,
    should_run: bool,
):
    callback = callback_factory(log_every_n_epochs=log_every_n_epochs)
    trainer = mock_trainer_factory(
        current_epoch=current_epoch, has_logger=False, max_epochs=max_epochs
    )
    pl_module = mock_pl_module_factory()

    with (
        patch(
            "versatil.training.synthetic_rollout_callback.run_rollouts",
            return_value=fake_trajectories_factory(),
        ) as mock_run,
        patch(
            "versatil.training.synthetic_rollout_callback.evaluate_rollouts",
            return_value=fake_results_factory(),
        ),
    ):
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    if should_run:
        mock_run.assert_called_once()
    else:
        mock_run.assert_not_called()


@pytest.mark.unit
def test_calls_run_rollouts_with_correct_args(
    callback_factory: Callable[..., SyntheticRolloutCallback],
    mock_trainer_factory: Callable[..., MagicMock],
    mock_pl_module_factory: Callable[..., MagicMock],
    fake_trajectories_factory: Callable[..., np.ndarray],
    fake_results_factory: Callable[..., dict],
):
    task_name = SyntheticTaskName.CORRIDOR_NAVIGATION.value
    num_rollouts = 25
    image_size = 48
    callback = callback_factory(
        task_name=task_name,
        num_rollouts=num_rollouts,
        image_size=image_size,
    )
    trainer = mock_trainer_factory(has_logger=False)
    pl_module = mock_pl_module_factory()

    patches = _patch_callback_dependencies(
        fake_trajectories=fake_trajectories_factory(num_rollouts=num_rollouts),
        fake_results=fake_results_factory(),
    )
    with (
        patches[0] as mock_run,
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
    ):
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    mock_run.assert_called_once_with(
        policy=pl_module.policy,
        task_name=task_name,
        num_rollouts=num_rollouts,
        image_size=image_size,
        context_mode=None,
        temporal_aggregation=False,
    )


@pytest.mark.unit
def test_calls_run_rollouts_once_per_mode_for_conditional_task(
    callback_factory: Callable[..., SyntheticRolloutCallback],
    mock_trainer_factory: Callable[..., MagicMock],
    mock_pl_module_factory: Callable[..., MagicMock],
    fake_trajectories_factory: Callable[..., np.ndarray],
    fake_results_factory: Callable[..., dict],
):
    task_name = SyntheticTaskName.CONDITIONAL_CIRCLE.value
    num_rollouts = 10
    image_size = 48
    num_modes = 2
    callback = callback_factory(
        task_name=task_name,
        num_rollouts=num_rollouts,
        image_size=image_size,
    )
    trainer = mock_trainer_factory(has_logger=False)
    pl_module = mock_pl_module_factory()
    pl_module.policy.observation_space.observations_metadata = {
        SyntheticObsKey.CONTEXT.value: MagicMock(),
    }

    fake_layout = MagicMock()
    fake_layout.num_modes = num_modes

    patches = _patch_callback_dependencies(
        fake_trajectories=fake_trajectories_factory(num_rollouts=num_rollouts),
        fake_results=fake_results_factory(),
    )
    with (
        patches[0] as mock_run,
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patch(
            "versatil.training.synthetic_rollout_callback.get_task_layout",
            return_value=fake_layout,
        ),
    ):
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    assert mock_run.call_count == num_modes
    context_modes_passed = [
        call.kwargs["context_mode"] for call in mock_run.call_args_list
    ]
    assert context_modes_passed == list(range(num_modes))


@pytest.mark.unit
def test_calls_evaluate_rollouts_with_correct_args(
    callback_factory: Callable[..., SyntheticRolloutCallback],
    mock_trainer_factory: Callable[..., MagicMock],
    mock_pl_module_factory: Callable[..., MagicMock],
    fake_trajectories_factory: Callable[..., np.ndarray],
    fake_results_factory: Callable[..., dict],
):
    task_name = SyntheticTaskName.SEQUENTIAL_DECISION.value
    image_size = 48
    fake_trajectories = fake_trajectories_factory()
    callback = callback_factory(task_name=task_name, image_size=image_size)
    trainer = mock_trainer_factory(has_logger=False)
    pl_module = mock_pl_module_factory()

    patches = _patch_callback_dependencies(
        fake_trajectories=fake_trajectories,
        fake_results=fake_results_factory(),
    )
    with (
        patches[0],
        patches[1] as mock_eval,
        patches[2],
        patches[3],
        patches[4],
        patches[5],
    ):
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    mock_eval.assert_called_once_with(
        rollout_trajectories=fake_trajectories,
        task_name=task_name,
        image_size=image_size,
        num_modes=callback.num_modes,
        num_styles=callback.num_styles,
        trajectory_length=callback.trajectory_length,
        noise_std=callback.noise_std,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode_coverage, entropy_ratio, per_mode_count",
    [
        (1.0, 1.0, {0: 5, 1: 5, 2: 5}),
        (0.33, 0.0, {0: 10, 1: 0, 2: 0}),
    ],
)
def test_logs_coverage_metrics_to_wandb(
    callback_factory: Callable[..., SyntheticRolloutCallback],
    mock_trainer_factory: Callable[..., MagicMock],
    mock_pl_module_factory: Callable[..., MagicMock],
    fake_trajectories_factory: Callable[..., np.ndarray],
    mode_coverage: float,
    entropy_ratio: float,
    per_mode_count: dict[int, int],
):
    callback = callback_factory()
    trainer = mock_trainer_factory(current_epoch=5)
    pl_module = mock_pl_module_factory()

    fake_results = {
        "mode_coverage": mode_coverage,
        "mode_entropy_ratio": entropy_ratio,
        "per_mode_count": per_mode_count,
        "success_rate": 0.5,
        "collision_rate": 0.1,
        "endpoint_reach_rate": 0.6,
        "path_length_rate": 0.7,
    }
    patches = _patch_callback_dependencies(
        fake_trajectories=fake_trajectories_factory(),
        fake_results=fake_results,
    )
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    logged = trainer.logger.log_metrics.call_args.args[0]
    assert logged["synthetic/mode_coverage"] == mode_coverage
    assert logged["synthetic/mode_entropy_ratio"] == entropy_ratio
    for mode_index, count in per_mode_count.items():
        assert logged[f"synthetic/mode_{mode_index}_count"] == count
    assert trainer.logger.log_metrics.call_args.kwargs["step"] == 5


@pytest.mark.unit
def test_logs_trajectory_plot_as_wandb_image(
    callback_factory: Callable[..., SyntheticRolloutCallback],
    mock_trainer_factory: Callable[..., MagicMock],
    mock_pl_module_factory: Callable[..., MagicMock],
    fake_trajectories_factory: Callable[..., np.ndarray],
    fake_results_factory: Callable[..., dict],
):
    callback = callback_factory()
    trainer = mock_trainer_factory()
    pl_module = mock_pl_module_factory()
    mock_figure = MagicMock()

    with (
        patch(
            "versatil.training.synthetic_rollout_callback.run_rollouts",
            return_value=fake_trajectories_factory(),
        ),
        patch(
            "versatil.training.synthetic_rollout_callback.evaluate_rollouts",
            return_value=fake_results_factory(),
        ),
        patch(
            "versatil.training.synthetic_rollout_callback.plot_trajectories_2d",
            return_value=mock_figure,
        ) as mock_plot,
        patch("versatil.training.synthetic_rollout_callback.plt.close") as mock_close,
        patch("versatil.training.synthetic_rollout_callback.Image.open"),
        patch(
            "versatil.training.synthetic_rollout_callback.wandb.Image"
        ) as mock_wandb_image,
    ):
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    # Two calls: _log_training_data (with mode_ids/title) + rollout plot
    rollout_calls = [c for c in mock_plot.call_args_list if "mode_ids" not in c.kwargs]
    assert len(rollout_calls) == 1
    assert rollout_calls[0].kwargs["task_name"] == callback.task_name
    assert mock_figure.savefig.call_count == 2
    assert mock_close.call_count == 2
    assert mock_wandb_image.call_count == 2
    logged = trainer.logger.log_metrics.call_args.args[0]
    assert "synthetic/rollout_trajectories" in logged


@pytest.mark.unit
def test_skips_plotting_and_logging_when_no_logger(
    callback_factory: Callable[..., SyntheticRolloutCallback],
    mock_trainer_factory: Callable[..., MagicMock],
    mock_pl_module_factory: Callable[..., MagicMock],
    fake_trajectories_factory: Callable[..., np.ndarray],
    fake_results_factory: Callable[..., dict],
):
    callback = callback_factory()
    trainer = mock_trainer_factory(has_logger=False)
    pl_module = mock_pl_module_factory()

    with (
        patch(
            "versatil.training.synthetic_rollout_callback.run_rollouts",
            return_value=fake_trajectories_factory(),
        ),
        patch(
            "versatil.training.synthetic_rollout_callback.evaluate_rollouts",
            return_value=fake_results_factory(),
        ),
        patch(
            "versatil.training.synthetic_rollout_callback.plot_trajectories_2d",
        ) as mock_plot,
    ):
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    mock_plot.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "was_training, expect_train_called",
    [
        (True, True),
        (False, False),
    ],
)
def test_restores_training_mode_only_if_was_training(
    callback_factory: Callable[..., SyntheticRolloutCallback],
    mock_trainer_factory: Callable[..., MagicMock],
    mock_pl_module_factory: Callable[..., MagicMock],
    fake_trajectories_factory: Callable[..., np.ndarray],
    fake_results_factory: Callable[..., dict],
    was_training: bool,
    expect_train_called: bool,
):
    callback = callback_factory()
    trainer = mock_trainer_factory(has_logger=False)
    pl_module = mock_pl_module_factory(training=was_training)

    with (
        patch(
            "versatil.training.synthetic_rollout_callback.run_rollouts",
            return_value=fake_trajectories_factory(),
        ),
        patch(
            "versatil.training.synthetic_rollout_callback.evaluate_rollouts",
            return_value=fake_results_factory(),
        ),
    ):
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    pl_module.policy.eval.assert_called_once()
    if expect_train_called:
        pl_module.policy.train.assert_called_once()
    else:
        pl_module.policy.train.assert_not_called()


@pytest.mark.unit
def test_policy_eval_called_inside_no_grad(
    callback_factory: Callable[..., SyntheticRolloutCallback],
    mock_trainer_factory: Callable[..., MagicMock],
    mock_pl_module_factory: Callable[..., MagicMock],
    fake_trajectories_factory: Callable[..., np.ndarray],
    fake_results_factory: Callable[..., dict],
):
    callback = callback_factory()
    trainer = mock_trainer_factory(has_logger=False)
    pl_module = mock_pl_module_factory()

    with (
        patch(
            "versatil.training.synthetic_rollout_callback.run_rollouts",
            return_value=fake_trajectories_factory(),
        ) as mock_run,
        patch(
            "versatil.training.synthetic_rollout_callback.evaluate_rollouts",
            return_value=fake_results_factory(),
        ),
    ):
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    # eval() must be called before run_rollouts
    pl_module.policy.eval.assert_called_once()
    mock_run.assert_called_once()


@pytest.mark.unit
def test_logs_training_data_on_first_epoch_only(
    callback_factory: Callable[..., SyntheticRolloutCallback],
    mock_trainer_factory: Callable[..., MagicMock],
    mock_pl_module_factory: Callable[..., MagicMock],
    fake_trajectories_factory: Callable[..., np.ndarray],
    fake_results_factory: Callable[..., dict],
):
    callback = callback_factory()
    trainer = mock_trainer_factory(current_epoch=0)
    pl_module = mock_pl_module_factory()

    with (
        patch(
            "versatil.training.synthetic_rollout_callback.run_rollouts",
            return_value=fake_trajectories_factory(),
        ),
        patch(
            "versatil.training.synthetic_rollout_callback.evaluate_rollouts",
            return_value=fake_results_factory(),
        ),
        patch(
            "versatil.training.synthetic_rollout_callback.generate_task_episodes",
            return_value=[
                {"position": np.zeros((10, 2)), "mode_id": np.zeros((10, 1))}
            ],
        ) as mock_generate,
        patch(
            "versatil.training.synthetic_rollout_callback.plot_trajectories_2d",
            return_value=MagicMock(),
        ),
        patch("versatil.training.synthetic_rollout_callback.plt.close"),
        patch("versatil.training.synthetic_rollout_callback.Image.open"),
        patch("versatil.training.synthetic_rollout_callback.wandb.Image"),
    ):
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)
        first_generate_count = mock_generate.call_count

        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)
        second_generate_count = mock_generate.call_count

    assert first_generate_count == 1
    assert second_generate_count == 1


@pytest.mark.unit
def test_training_data_plot_receives_mode_ids_and_title(
    callback_factory: Callable[..., SyntheticRolloutCallback],
    mock_trainer_factory: Callable[..., MagicMock],
    mock_pl_module_factory: Callable[..., MagicMock],
    fake_trajectories_factory: Callable[..., np.ndarray],
    fake_results_factory: Callable[..., dict],
):
    callback = callback_factory()
    trainer = mock_trainer_factory(current_epoch=0)
    pl_module = mock_pl_module_factory()

    with (
        patch(
            "versatil.training.synthetic_rollout_callback.run_rollouts",
            return_value=fake_trajectories_factory(),
        ),
        patch(
            "versatil.training.synthetic_rollout_callback.evaluate_rollouts",
            return_value=fake_results_factory(),
        ),
        patch(
            "versatil.training.synthetic_rollout_callback.generate_task_episodes",
            return_value=[
                {"position": np.zeros((10, 2)), "mode_id": np.zeros((10, 1))}
            ],
        ),
        patch(
            "versatil.training.synthetic_rollout_callback.plot_trajectories_2d",
            return_value=MagicMock(),
        ) as mock_plot,
        patch("versatil.training.synthetic_rollout_callback.plt.close"),
        patch("versatil.training.synthetic_rollout_callback.Image.open"),
        patch("versatil.training.synthetic_rollout_callback.wandb.Image"),
    ):
        callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    training_data_call = next(
        call for call in mock_plot.call_args_list if "mode_ids" in call.kwargs
    )
    assert training_data_call.kwargs["title"] == "Training Data"
    assert training_data_call.kwargs["mode_ids"] is not None
