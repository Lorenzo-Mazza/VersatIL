"""Callback for evaluating synthetic benchmark policies during training."""

import io
import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from PIL import Image
from pytorch_lightning.callbacks import Callback

from versatil.data.synthetic.constants import (
    CORRIDOR_DEFAULT_NUM_STYLES,
    MULTIPATH_DEFAULT_NOISE_STD,
    MULTIPATH_DEFAULT_NUM_MODES,
    MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
)
from versatil.data.synthetic.generators import generate_task_episodes
from versatil.data.synthetic.visualization import plot_trajectories_2d
from versatil.inference.synthetic_rollout import evaluate_rollouts, run_rollouts
from versatil.models.policy import Policy


class SyntheticRolloutCallback(Callback):
    """Run rollouts and log mode coverage metrics at the end of each training epoch.

    Puts the policy in eval mode, generates trajectories via closed-loop
    rollout, computes mode coverage and goal success against regenerated
    expert demonstrations, and logs metrics + trajectory plots to wandb.

    Args:
        task_name: SyntheticTaskName.value string.
        num_rollouts: Number of rollout trajectories per evaluation.
        image_size: Side length for rendered observation images.
        log_every_n_epochs: Evaluate every N epochs.
    """

    def __init__(
        self,
        task_name: str,
        num_rollouts: int = 50,
        image_size: int = 64,
        log_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.task_name = task_name
        self.num_rollouts = num_rollouts
        self.image_size = image_size
        self.log_every_n_epochs = log_every_n_epochs
        self._training_data_logged = False

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Run rollouts, compute metrics, log to wandb and console."""
        if trainer.logger is not None and not self._training_data_logged:
            self._log_training_data(trainer=trainer)
            self._training_data_logged = True

        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return

        policy: Policy = pl_module.policy
        was_training = policy.training
        policy.eval()

        with torch.no_grad():
            trajectories = run_rollouts(
                policy=policy,
                task_name=self.task_name,
                num_rollouts=self.num_rollouts,
                image_size=self.image_size,
                temporal_aggregation=False,  # open-loop: single prediction, no replanning
            )

        results = evaluate_rollouts(
            rollout_trajectories=trajectories,
            task_name=self.task_name,
            image_size=self.image_size,
        )

        epoch = trainer.current_epoch
        mode_coverage = results["mode_coverage"]
        entropy_ratio = results["mode_entropy_ratio"]
        per_mode = results["per_mode_count"]
        goal_success = results.get("goal_success_rate")

        log_parts = [
            f"epoch {epoch}",
            f"mode_coverage={mode_coverage:.2f}",
            f"entropy={entropy_ratio:.2f}",
            f"per_mode={per_mode}",
        ]
        if goal_success is not None:
            log_parts.append(f"goal_success={goal_success:.2f}")
        logging.info(f"Synthetic rollout: {', '.join(log_parts)}")

        if trainer.logger is not None:
            metrics: dict[str, float | wandb.Image] = {
                "synthetic/mode_coverage": mode_coverage,
                "synthetic/mode_entropy_ratio": entropy_ratio,
            }
            if goal_success is not None:
                metrics["synthetic/goal_success_rate"] = goal_success
            for mode_index, count in per_mode.items():
                metrics[f"synthetic/mode_{mode_index}_count"] = count

            metrics["synthetic/rollout_trajectories"] = _figure_to_wandb_image(
                plot_trajectories_2d(
                    trajectories=trajectories,
                    task_name=self.task_name,
                )
            )

            trainer.logger.log_metrics(metrics, step=epoch)

        if was_training:
            policy.train()

    def _log_training_data(self, trainer: pl.Trainer) -> None:
        """Log training data trajectories to wandb on the first epoch."""
        episodes = generate_task_episodes(
            task_name=self.task_name,
            num_episodes=100,
            seed=0,
            image_size=self.image_size,
            num_modes=MULTIPATH_DEFAULT_NUM_MODES,
            trajectory_length=MULTIPATH_DEFAULT_TRAJECTORY_LENGTH,
            noise_std=MULTIPATH_DEFAULT_NOISE_STD,
            num_styles=CORRIDOR_DEFAULT_NUM_STYLES,
        )
        trajectories = np.array([episode["position"] for episode in episodes])
        mode_ids = np.array([int(episode["mode_id"][0, 0]) for episode in episodes])
        figure = plot_trajectories_2d(
            trajectories=trajectories,
            task_name=self.task_name,
            mode_ids=mode_ids,
            title="Training Data",
        )
        trainer.logger.log_metrics(
            {"synthetic/training_data": _figure_to_wandb_image(figure)},
            step=0,
        )


def _figure_to_wandb_image(figure: plt.Figure) -> wandb.Image:
    """Convert a matplotlib figure to a wandb.Image and close it."""
    buf = io.BytesIO()
    figure.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    image = wandb.Image(Image.open(buf))
    plt.close(figure)
    return image
