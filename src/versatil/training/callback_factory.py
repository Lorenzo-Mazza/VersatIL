"""Builders for the training callback stack."""

import logging
from pathlib import Path

from pytorch_lightning.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
    TQDMProgressBar,
)

from versatil.configs import ExperimentConfig, TrainingConfig
from versatil.training.callbacks.early_stopping import ResumableEarlyStopping
from versatil.training.callbacks.ema import EMACallback
from versatil.training.callbacks.gradient_norm import GradientNormCallback
from versatil.training.callbacks.reduce_lr_on_plateau import ReduceLROnPlateauCallback
from versatil.training.callbacks.training_stage import TrainingStageCallback


def build_training_callbacks(
    experiment_config: ExperimentConfig,
    training_config: TrainingConfig,
    output_dir: Path,
    has_validation: bool,
) -> list[Callback]:
    """Build the workspace's training callback stack.

    Args:
        experiment_config: Experiment settings (checkpointing cadence).
        training_config: Training settings (EMA, SWA, stages, LR control).
        output_dir: Directory receiving checkpoints.
        has_validation: Whether a validation loader exists.

    Returns:
        Callbacks in registration order; ordering matters for callbacks that
        read state set by earlier ones (e.g. training stages before prior
        target standardization).
    """
    callbacks: list[Callback] = [TQDMProgressBar(refresh_rate=1)]
    callbacks.extend(_build_ema_callbacks(training_config=training_config))
    callbacks.extend(
        _build_checkpoint_callbacks(
            experiment_config=experiment_config,
            output_dir=output_dir,
            has_validation=has_validation,
        )
    )
    callbacks.extend(
        _build_early_stopping_callbacks(
            training_config=training_config, has_validation=has_validation
        )
    )
    callbacks.append(GradientNormCallback(log_every_n_steps=50))
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.extend(_build_swa_callbacks(training_config=training_config))
    callbacks.extend(
        _build_stage_callbacks(
            training_config=training_config, has_validation=has_validation
        )
    )
    return callbacks


def _build_ema_callbacks(training_config: TrainingConfig) -> list[Callback]:
    """Build the EMA callback when enabled."""
    if not training_config.use_ema:
        return []
    logging.info(f"Added EMA callback (power={training_config.ema_power})")
    return [EMACallback(power=training_config.ema_power)]


def _build_checkpoint_callbacks(
    experiment_config: ExperimentConfig,
    output_dir: Path,
    has_validation: bool,
) -> list[Callback]:
    """Build best- and latest-checkpoint callbacks when checkpointing is on."""
    if not experiment_config.save_checkpoints:
        logging.info("Skipping ModelCheckpoint callbacks (save_checkpoints=False)")
        return []
    # training_step logs "train_loss" with on_step=False, so Lightning never
    # creates a "train_loss_epoch" variant; the plain key already holds the
    # epoch aggregate.
    monitor = "val_loss" if has_validation else "train_loss"
    best_checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename="best-{epoch:02d}-{" + monitor + ":.4f}",
        monitor=monitor,
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False,
    )
    logging.info(f"Added ModelCheckpoint callback (top-k=3, monitor={monitor})")
    latest_checkpoint = ModelCheckpoint(
        dirpath=output_dir,
        filename="latest-{epoch:02d}",
        monitor="epoch",
        mode="max",
        save_top_k=-1,
        every_n_epochs=experiment_config.checkpoint_every,
        save_last=True,
        verbose=True,
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
    )
    logging.info(
        f"Added latest checkpoint callback "
        f"(every {experiment_config.checkpoint_every} epochs)"
    )
    return [best_checkpoint, latest_checkpoint]


def _build_early_stopping_callbacks(
    training_config: TrainingConfig,
    has_validation: bool,
) -> list[Callback]:
    """Build early stopping when validation and a patience are configured."""
    patience = training_config.early_stopping_patience
    if not has_validation:
        logging.info("Skipping EarlyStopping callback (no validation data)")
        return []
    if patience is None:
        logging.info("Skipping EarlyStopping callback (early_stopping_patience=None)")
        return []
    logging.info(f"Added EarlyStopping callback (patience={patience})")
    return [
        ResumableEarlyStopping(
            monitor="val_loss", mode="min", patience=patience, verbose=True
        )
    ]


def _build_swa_callbacks(training_config: TrainingConfig) -> list[Callback]:
    """Build stochastic weight averaging when enabled."""
    if training_config.swa_lrs is None:
        return []
    swa_epoch_start = int(training_config.swa_epoch_start * training_config.num_epochs)
    logging.info(
        f"Added SWA callback (learning_rate={training_config.swa_lrs}, "
        f"start_epoch={swa_epoch_start}, "
        f"annealing_epochs={training_config.swa_annealing_epochs})"
    )
    return [
        StochasticWeightAveraging(
            swa_lrs=training_config.swa_lrs,
            swa_epoch_start=swa_epoch_start,
            annealing_epochs=training_config.swa_annealing_epochs,
        )
    ]


def _build_stage_callbacks(
    training_config: TrainingConfig,
    has_validation: bool,
) -> list[Callback]:
    """Build training-stage and LR-plateau callbacks.

    Raises:
        ValueError: If training stages are combined with reduce_lr_on_plateau.
    """
    callbacks: list[Callback] = []
    training_stages = training_config.stages
    if training_stages and training_config.reduce_lr_on_plateau:
        raise ValueError("training.stages does not support reduce_lr_on_plateau in v1.")
    if training_stages:
        callbacks.append(
            TrainingStageCallback(
                stages=training_stages,
                learning_rate_schedule_active=(training_config.lr_schedule is not None),
            )
        )
        logging.info(f"Added TrainingStage callback ({len(training_stages)} stages)")
    if training_config.reduce_lr_on_plateau:
        monitor = "val_loss" if has_validation else "train_loss"
        callbacks.append(
            ReduceLROnPlateauCallback(
                monitor=monitor,
                patience=training_config.reduce_lr_patience,
                cooldown=training_config.reduce_lr_cooldown,
            )
        )
        logging.info(
            f"Added ReduceLROnPlateau callback (monitor={monitor}, "
            f"patience={training_config.reduce_lr_patience})"
        )
    return callbacks
