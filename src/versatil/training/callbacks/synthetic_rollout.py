"""Callback for evaluating synthetic benchmark policies during training."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback

from versatil.data.constants import SyntheticObsKey
from versatil.data.synthetic.generators import generate_task_episodes
from versatil.data.synthetic.task_layout import get_task_layout
from versatil.data.synthetic.visualization import plot_trajectories_2d
from versatil.inference.synthetic_rollout import evaluate_rollouts, run_rollouts
from versatil.models.policy import Policy
from versatil.training.callbacks.wandb_figure import figure_to_wandb_image


class SyntheticRolloutCallback(Callback):
    """Run rollouts and log mode coverage metrics at the end of each training epoch.

    Puts the policy in eval mode, generates trajectories via closed-loop
    rollout, computes mode coverage and goal success against regenerated
    expert demonstrations, and logs metrics + trajectory plots to wandb.

    Args:
        task_name: SyntheticTaskName.value string.
        num_modes: Number of behavioral modes to generate for expert
            reference. Must match the training dataset.
        num_styles: Number of sinusoidal styles per corridor gap. Ignored
            by tasks that do not use styles.
        trajectory_length: Length of generated expert and rollout
            trajectories.
        noise_std: Standard deviation of expert trajectory noise.
        num_rollouts: Number of rollout trajectories per evaluation.
        image_size: Side length for rendered observation images.
        log_every_n_epochs: Evaluate every N epochs.
    """

    def __init__(
        self,
        task_name: str,
        num_modes: int,
        num_styles: int,
        trajectory_length: int,
        noise_std: float,
        num_rollouts: int = 50,
        image_size: int = 64,
        log_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.task_name = task_name
        self.num_modes = num_modes
        self.num_styles = num_styles
        self.trajectory_length = trajectory_length
        self.noise_std = noise_std
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

        is_log_epoch = trainer.current_epoch % self.log_every_n_epochs == 0
        is_last_epoch = trainer.current_epoch == trainer.max_epochs - 1
        if not is_log_epoch and not is_last_epoch:
            return

        policy: Policy = pl_module.policy
        was_training = policy.training
        policy.eval()

        context_modes = self._resolve_context_modes(policy=policy)
        with torch.no_grad():
            per_mode_trajectories = [
                run_rollouts(
                    policy=policy,
                    task_name=self.task_name,
                    num_rollouts=self.num_rollouts,
                    image_size=self.image_size,
                    context_mode=mode,
                    temporal_aggregation=False,  # open-loop
                )
                for mode in context_modes
            ]
        trajectories = (
            per_mode_trajectories[0]
            if len(per_mode_trajectories) == 1
            else np.concatenate(per_mode_trajectories, axis=0)
        )

        results = evaluate_rollouts(
            rollout_trajectories=trajectories,
            task_name=self.task_name,
            image_size=self.image_size,
            num_modes=self.num_modes,
            num_styles=self.num_styles,
            trajectory_length=self.trajectory_length,
            noise_std=self.noise_std,
        )

        epoch = trainer.current_epoch
        mode_coverage = results["mode_coverage"]
        entropy_ratio = results["mode_entropy_ratio"]
        per_mode = results["per_mode_count"]
        success_rate = results["success_rate"]
        collision_rate = results["collision_rate"]
        endpoint_reach_rate = results["endpoint_reach_rate"]
        path_length_rate = results["path_length_rate"]

        log_parts = [
            f"epoch {epoch}",
            f"success={success_rate:.2f}",
            f"collision={collision_rate:.2f}",
            f"endpoint_reach={endpoint_reach_rate:.2f}",
            f"path_length={path_length_rate:.2f}",
            f"mode_coverage={mode_coverage:.2f}",
            f"entropy={entropy_ratio:.2f}",
            f"per_mode={per_mode}",
        ]
        logging.info(f"Synthetic rollout: {', '.join(log_parts)}")

        if trainer.logger is not None:
            metrics: dict[str, float | wandb.Image] = {
                "synthetic/success_rate": success_rate,
                "synthetic/collision_rate": collision_rate,
                "synthetic/endpoint_reach_rate": endpoint_reach_rate,
                "synthetic/path_length_rate": path_length_rate,
                "synthetic/mode_coverage": mode_coverage,
                "synthetic/mode_entropy_ratio": entropy_ratio,
            }
            for mode_index, count in per_mode.items():
                metrics[f"synthetic/mode_{mode_index}_count"] = count

            rollout_figure = plot_trajectories_2d(
                trajectories=trajectories,
                task_name=self.task_name,
                num_modes=self.num_modes,
                num_styles=self.num_styles,
                noise_std=self.noise_std,
            )
            metrics["synthetic/rollout_trajectories"] = figure_to_wandb_image(
                rollout_figure, dpi=150
            )
            plt.close(rollout_figure)

            trainer.logger.log_metrics(metrics, step=epoch)

        if was_training:
            policy.train()

    def _resolve_context_modes(self, policy: Policy) -> list[int | None]:
        """Determine which context modes to roll out.

        Returns [None] for non-conditional policies. For policies that consume
        the CONTEXT observation, returns one entry per layout mode so every
        mode gets its own rollout batch.
        """
        has_context = (
            SyntheticObsKey.CONTEXT.value
            in policy.observation_space.observations_metadata
        )
        if not has_context:
            return [None]
        layout = get_task_layout(
            task_name=self.task_name,
            num_modes=self.num_modes,
            num_styles=self.num_styles,
            noise_std=self.noise_std,
        )
        return list(range(layout.num_modes))

    def _log_training_data(self, trainer: pl.Trainer) -> None:
        """Log training data trajectories to wandb on the first epoch."""
        episodes = generate_task_episodes(
            task_name=self.task_name,
            num_episodes=100,
            seed=0,
            image_size=self.image_size,
            num_modes=self.num_modes,
            trajectory_length=self.trajectory_length,
            noise_std=self.noise_std,
            num_styles=self.num_styles,
        )
        trajectories = np.array([episode["position"] for episode in episodes])
        mode_ids = np.array([int(episode["mode_id"][0, 0]) for episode in episodes])
        figure = plot_trajectories_2d(
            trajectories=trajectories,
            task_name=self.task_name,
            mode_ids=mode_ids,
            title="Training Data",
            num_modes=self.num_modes,
            num_styles=self.num_styles,
            noise_std=self.noise_std,
        )
        trainer.logger.log_metrics(
            {"synthetic/training_data": figure_to_wandb_image(figure, dpi=150)},
            step=0,
        )
        plt.close(figure)
