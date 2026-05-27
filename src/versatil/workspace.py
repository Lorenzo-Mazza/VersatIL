"""Workspace for training and evaluating policies using PyTorch Lightning."""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.tuner import Tuner
from torch.utils import data

import wandb
from versatil.common.tensor_ops import to_device
from versatil.configs import MainConfig
from versatil.data.dataloader import get_dataloaders
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.tokenization import Tokenizer
from versatil.models.policy import Policy
from versatil.training.callbacks.early_stopping import ResumableEarlyStopping
from versatil.training.callbacks.ema import EMACallback
from versatil.training.callbacks.gradient_norm import GradientNormCallback
from versatil.training.callbacks.provider import CallbackProvider
from versatil.training.callbacks.reduce_lr_on_plateau import ReduceLROnPlateauCallback
from versatil.training.callbacks.training_stage import TrainingStageCallback
from versatil.training.constants import PrecisionType
from versatil.training.lightning_policy import LightningPolicy


class Workspace:
    """Single workspace for training any policy using PyTorch Lightning.

    This workspace handles:
    - Data loading and normalization
    - Policy instantiation and wrapping with Lightning
    - Trainer setup with callbacks (EMA, checkpointing, confusion matrix)
    - Distributed training via DDP
    - WandB logging
    - Checkpointing with save_top_k and save_last
    """

    def __init__(self, config: DictConfig, original_yaml_config: DictConfig):
        """Initialize workspace.

        Args:
            config: Main configuration containing all settings, already instantiated
            original_yaml_config: Original YAML config before instantiation, for saving
        """
        self.config: MainConfig = config
        self.original_yaml_config = original_yaml_config
        hydra_cfg = HydraConfig.get()
        main_config_name = (
            hydra_cfg.job.config_name if hydra_cfg.job.config_name else "experiment"
        )
        additional_exp_name = config.experiment.name
        sweep_suffix = self._get_multirun_suffix(hydra_cfg)
        self.exp_name = f"{main_config_name}/{additional_exp_name}{sweep_suffix}"
        self.config.experiment.name = self.exp_name
        self.original_yaml_config.experiment.name = self.exp_name
        self.output_dir = (
            Path(config.experiment.checkpoint_folder)
            / main_config_name
            / f"{additional_exp_name}{sweep_suffix}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._set_seed()
        self.policy: Policy | None = None
        self.lightning_policy: LightningPolicy | None = None
        self.trainer: pl.Trainer | None = None
        self.train_loader: data.DataLoader | None = None
        self.val_loader: data.DataLoader | None = None
        self.normalizer: LinearNormalizer | None = None
        self.tokenizer: Tokenizer | None = None
        self.logger = None
        self.gripper_class_weights: torch.Tensor | None = None
        logging.info(f"Workspace initialized for experiment: {self.exp_name}")
        logging.info(f"Output directory: {self.output_dir}")
        self.save_config()

    @staticmethod
    def _get_multirun_suffix(hydra_cfg: DictConfig) -> str:
        """Build a unique suffix for Hydra multirun jobs.

        Uses the job number to keep paths short while ensuring uniqueness.

        Args:
            hydra_cfg: HydraConfig for the current job.

        Returns:
            Empty string for single runs, "/job{num}" for multiruns.
        """
        if hydra_cfg.mode != RunMode.MULTIRUN:
            return ""
        return f"/job{hydra_cfg.job.num}"

    def save_config(self):
        """Save configuration to YAML file in output directory.

        The config is saved as 'config.yaml' and is required for inference
        and model explanation. This method should be called after workspace
        initialization to ensure the config is available for later use.
        """
        config_path = self.output_dir / "config.yaml"
        # Resolve all interpolations before saving
        resolved_config = OmegaConf.to_container(
            self.original_yaml_config, resolve=True
        )
        resolved_config_dict = OmegaConf.create(resolved_config)
        OmegaConf.save(resolved_config_dict, config_path)
        logging.info(f"Config saved to {config_path}")

    def run(self):
        """Run the complete training workflow."""
        self.logger = self._create_logger()
        self._setup_data()
        self._setup_policy()
        self.lightning_policy._train_dataloader = self.train_loader
        self.lightning_policy._val_dataloader = self.val_loader
        self._setup_trainer()
        resume_checkpoint_path = None
        if self.config.experiment.resume_from is not None:
            checkpoint_path = Path(self.config.experiment.resume_from)
            if checkpoint_path.exists():
                logging.info(f"Resuming from checkpoint: {checkpoint_path}")
                resume_checkpoint_path = str(checkpoint_path)
            else:
                logging.warning(
                    f"Checkpoint not found: {checkpoint_path}. Starting from scratch."
                )

        self._tune_hyperparameters()

        logging.info("Starting training...")
        if self.trainer is None:
            raise RuntimeError("Trainer should be initialized before training.")
        self.trainer.fit(
            model=self.lightning_policy,
            ckpt_path=resume_checkpoint_path,
            weights_only=False,
        )
        logging.info(f"Training completed. Best checkpoint saved to {self.output_dir}")

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        seed = self.config.experiment.seed
        torch.manual_seed(seed)
        np.random.default_rng(seed)
        np.random.seed(seed)  # Legacy: required by some third-party libraries
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _setup_data(self):
        """Setup dataloaders, normalizer, and tokenizer."""
        logging.info("Setting up dataloaders...")
        (
            self.train_loader,
            self.val_loader,
            self.normalizer,
            self.tokenizer,
            gripper_class_weights,
        ) = get_dataloaders(self.config)

        if gripper_class_weights is not None:
            self.gripper_class_weights = torch.tensor(
                [gripper_class_weights],
                device=self.config.experiment.device,
            )
            logging.info(f"Using gripper class weights: {gripper_class_weights:.4f}")
        else:
            self.gripper_class_weights = None

        if self.tokenizer is not None:
            logging.info("Tokenizer initialized for discrete action tokenization")
            tokenizer_path = self.output_dir / "tokenizer"
            self.tokenizer.save_pretrained(tokenizer_path)
            logging.info(f"Tokenizer saved to {tokenizer_path}")

        logging.info(f"Train dataset size: {len(self.train_loader.dataset)} samples")
        if self.val_loader is not None:
            logging.info(f"Val dataset size: {len(self.val_loader.dataset)} samples")
        else:
            logging.info("Validation disabled (val_ratio=0)")
        action_processor = self.train_loader.dataset.action_processor
        fig = action_processor.plot_action_magnitude_distribution()
        if fig is not None:
            plot_path = self.output_dir / "action_deltas_distribution.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
            logging.info(f"Saved action delta distribution plot to {plot_path}")
            if self.logger is not None:
                self.logger.experiment.log(
                    {"action_delta_distribution": wandb.Image(fig)}
                )
            plt.close(fig)
        self.denoising_thresholds = action_processor.denoising_thresholds

    def _setup_policy(self):
        """Instantiate policy and wrap with Lightning."""
        logging.info("Instantiating policy...")
        self.policy: Policy = self.config.policy
        pipeline_dtype = PrecisionType(
            str(self.config.experiment.precision)
        ).get_model_dtype()
        self.policy.encoding_pipeline.set_output_dtype(pipeline_dtype)
        self.policy.set_normalizer(self.normalizer)
        self.policy.set_tokenizer(self.tokenizer)
        self.policy.set_denoising_thresholds(self.denoising_thresholds)
        self.policy.set_gripper_class_weights(self.gripper_class_weights)
        # Calculate total training steps for learning-rate scheduling
        # Steps per epoch = len(train_loader) // gradient_accumulate_every
        # Total steps = steps_per_epoch * num_epochs
        steps_per_epoch = (
            len(self.train_loader) // self.config.training.gradient_accumulate_every
        )
        total_training_steps = steps_per_epoch * self.config.training.num_epochs
        self.lightning_policy = LightningPolicy(
            policy=self.policy,
            training_config=self.config.training,
            total_training_steps=total_training_steps,
        )
        self._initialize_lazy_modules()
        if self.config.training.compile:
            logging.info(
                "Compiling policy with torch.compile (mode=%s)...",
                self.config.training.compile_mode,
            )
            self.policy = torch.compile(
                self.policy, mode=self.config.training.compile_mode
            )
            self.lightning_policy.policy = self.policy
        logging.info(f"Policy created: {self.policy.__class__.__name__}")
        logging.info(f"Total training steps: {total_training_steps}")

    def _setup_trainer(self):
        """Setup PyTorch Lightning trainer with callbacks and logger."""
        callbacks = self._create_callbacks()
        strategy = self._create_strategy()
        gradient_clip_val = None
        if self.config.training.clip_gradient_norm:
            gradient_clip_val = self.config.training.clip_max_norm

        if self.config.experiment.float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(
                self.config.experiment.float32_matmul_precision
            )
            logging.info(
                f"Set float32 matmul precision to '{self.config.experiment.float32_matmul_precision}'"
            )
        val_every = (
            self.config.experiment.val_every if self.val_loader is not None else 0
        )
        limit_val_batches = 1.0 if self.val_loader is not None else 0
        log_every_n_steps = self._get_log_every_n_steps()

        self.trainer = pl.Trainer(
            max_epochs=self.config.training.num_epochs,
            accelerator="gpu" if "cuda" in self.config.experiment.device else "cpu",
            devices="auto" if self.config.experiment.distributed else 1,
            strategy=strategy,
            logger=self.logger,
            callbacks=callbacks,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=self.config.training.gradient_accumulate_every,
            check_val_every_n_epoch=val_every,
            limit_val_batches=limit_val_batches,
            log_every_n_steps=log_every_n_steps,
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=self.config.experiment.save_checkpoints,
            deterministic=False,  # For performance
            precision=self.config.experiment.precision,
        )

        logging.info(f"Trainer created with {len(callbacks)} callbacks")

    def _get_log_every_n_steps(self) -> int:
        """Choose a logging interval that stays visible on small datasets."""
        if self.train_loader is None:
            return 50
        return max(1, min(50, len(self.train_loader)))

    def _create_callbacks(self):
        """Create training callbacks.

        Returns:
            List of Lightning callbacks
        """
        callbacks = [TQDMProgressBar(refresh_rate=1)]
        logging.info("Added TQDM progress bar callback (refresh every batch)")
        has_validation = self.val_loader is not None

        if self.config.training.use_ema:
            ema_callback = EMACallback(
                power=self.config.training.ema_power,
            )
            callbacks.append(ema_callback)
            logging.info(f"Added EMA callback (power={self.config.training.ema_power})")

        if self.config.experiment.save_checkpoints:
            if has_validation:
                checkpoint_callback_best = ModelCheckpoint(
                    dirpath=self.output_dir,
                    filename="best-{epoch:02d}-{val_loss:.4f}",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=3,
                    save_last=True,
                    verbose=True,
                    auto_insert_metric_name=False,
                )
            else:
                checkpoint_callback_best = ModelCheckpoint(
                    dirpath=self.output_dir,
                    filename="best-{epoch:02d}-{train_loss_epoch:.4f}",
                    monitor="train_loss_epoch",
                    mode="min",
                    save_top_k=3,
                    save_last=True,
                    verbose=True,
                    auto_insert_metric_name=False,
                )
            callbacks.append(checkpoint_callback_best)
            logging.info(
                f"Added ModelCheckpoint callback (top-k=3, monitor={checkpoint_callback_best.monitor})"
            )

            checkpoint_callback_latest = ModelCheckpoint(
                dirpath=self.output_dir,
                filename="latest-{epoch:02d}",
                monitor="epoch",
                mode="max",
                save_top_k=-1,
                every_n_epochs=self.config.experiment.checkpoint_every,
                save_last=True,
                verbose=True,
                auto_insert_metric_name=False,
                save_on_train_epoch_end=not has_validation,
            )
            callbacks.append(checkpoint_callback_latest)
            logging.info(
                f"Added latest checkpoint callback (every {self.config.experiment.checkpoint_every} epochs)"
            )
        else:
            logging.info("Skipping ModelCheckpoint callbacks (save_checkpoints=False)")

        early_stopping_patience = self.config.training.early_stopping_patience
        if not has_validation:
            logging.info("Skipping EarlyStopping callback (no validation data)")
        elif early_stopping_patience is None:
            logging.info(
                "Skipping EarlyStopping callback (early_stopping_patience=None)"
            )
        else:
            early_stopping_callback = ResumableEarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=early_stopping_patience,
                verbose=True,
            )
            callbacks.append(early_stopping_callback)
            logging.info(
                f"Added EarlyStopping callback (patience={early_stopping_patience})"
            )

        gradient_norm_callback = GradientNormCallback(log_every_n_steps=50)
        callbacks.append(gradient_norm_callback)
        logging.info("Added GradientNorm callback (log every 50 steps)")

        lr_monitor_callback = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor_callback)
        logging.info("Added LearningRateMonitor callback (logging per step)")

        if self.config.training.swa_lrs is not None:
            # Calculate start epoch based on fraction of total epochs
            swa_epoch_start = int(
                self.config.training.swa_epoch_start * self.config.training.num_epochs
            )
            swa_callback = StochasticWeightAveraging(
                swa_lrs=self.config.training.swa_lrs,
                swa_epoch_start=swa_epoch_start,
                annealing_epochs=self.config.training.swa_annealing_epochs,
            )
            callbacks.append(swa_callback)
            logging.info(
                f"Added SWA callback (learning_rate={self.config.training.swa_lrs}, "
                f"start_epoch={swa_epoch_start}, annealing_epochs={self.config.training.swa_annealing_epochs})"
            )

        training_stages = self.config.training.stages
        if training_stages and self.config.training.reduce_lr_on_plateau:
            raise ValueError(
                "training.stages does not support reduce_lr_on_plateau in v1."
            )
        if training_stages:
            training_stage_callback = TrainingStageCallback(
                stages=training_stages,
                learning_rate_schedule_active=(
                    self.config.training.lr_schedule is not None
                ),
            )
            callbacks.append(training_stage_callback)
            logging.info(
                f"Added TrainingStage callback ({len(training_stages)} stages)"
            )

        if self.config.training.reduce_lr_on_plateau:
            monitor = "val_loss" if has_validation else "train_loss"
            reduce_lr_callback = ReduceLROnPlateauCallback(
                monitor=monitor,
                patience=self.config.training.reduce_lr_patience,
                cooldown=self.config.training.reduce_lr_cooldown,
            )
            callbacks.append(reduce_lr_callback)
            logging.info(
                f"Added ReduceLROnPlateau callback (monitor={monitor}, patience={self.config.training.reduce_lr_patience})"
            )

        component_callbacks = self._collect_component_callbacks()
        callbacks.extend(component_callbacks)

        return callbacks

    def _collect_component_callbacks(self) -> list:
        """Collect callbacks declared by policy components via CallbackProvider protocol.

        Iterates over the decoder, algorithm, and loss modules, calling
        ``get_callbacks()`` on any that implement the protocol. Deduplicates
        by callback class to avoid duplicates when multiple components declare
        the same callback type.

        Returns:
            List of deduplicated Lightning callbacks from all components.
        """
        components = [
            self.policy.decoder,
            self.policy.algorithm,
            *self.policy.loss_module.loss_modules.values(),
            self.config.task.dataset_schema,
        ]
        seen_types: set[type] = set()
        collected: list = []
        for component in components:
            if not isinstance(component, CallbackProvider):
                continue
            for callback in component.get_callbacks(
                experiment_config=self.config.experiment
            ):
                if type(callback) not in seen_types:
                    seen_types.add(type(callback))
                    collected.append(callback)
                    logging.info(
                        f"Added {type(callback).__name__} from "
                        f"{type(component).__name__}"
                    )
        return collected

    def _create_logger(self) -> WandbLogger | None:
        """Create WandB logger if enabled.

        Returns:
            WandbLogger if enabled, None otherwise
        """
        if not self.config.experiment.use_wandb:
            logging.info("WandB logging disabled")
            return None

        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            logging.warning("WANDB_API_KEY not set, disabling wandb logging")
            self.config.experiment.use_wandb = False
            return None
        # Close any prior wandb run lingering from a previous multirun job in
        # the same process; otherwise WandbLogger silently reuses it.
        if wandb.run is not None:
            wandb.finish()
        wandb_logger = WandbLogger(
            project=self.config.experiment.wandb_project,
            entity=self.config.experiment.wandb_entity,
            name=self.exp_name,
            save_dir=self.output_dir,
            log_model=False,  # We handle checkpointing ourselves
        )
        wandb_logger.log_hyperparams(
            OmegaConf.to_container(self.original_yaml_config, resolve=True)
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        logging.info(f"WandB logger created for experiment: {self.exp_name}")
        return wandb_logger

    def _create_strategy(self):
        """Create distributed training strategy.

        Returns:
            DDP strategy if distributed training enabled, else "auto"
        """
        if not self.config.experiment.distributed:
            return "auto"

        # Setup DDP strategy for SLURM environments
        # Environment variables parsed by Lightning:
        # - WORLD_SIZE: Total processes
        # - SLURM_PROCID: Global rank
        # - SLURM_GPUS_ON_NODE: GPUs per node
        # - SLURM_CPUS_PER_TASK: Workers per GPU

        strategy = DDPStrategy(
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        logging.info("Using DDP strategy for distributed training")
        logging.info(f"World size: {os.environ.get('WORLD_SIZE', 'N/A')}")
        logging.info(f"Rank: {os.environ.get('SLURM_PROCID', 'N/A')}")

        return strategy

    def _initialize_lazy_modules(self):
        """Initialize lazy modules by doing a dummy forward pass."""
        if self.train_loader is None or len(self.train_loader) == 0:
            raise RuntimeError(
                "Train loader is not initialized or empty, cannot initialize lazy modules"
            )
        data_iter = iter(self.train_loader)
        batch = next(data_iter)
        device = torch.device(self.config.experiment.device)
        batch = to_device(batch, device)
        self.lightning_policy.to(device)
        self.lightning_policy.train()
        with (
            torch.enable_grad(),
            torch.autocast(
                device_type=device.type,
                dtype=PrecisionType(
                    str(self.config.experiment.precision)
                ).get_model_dtype(),
            ),
        ):
            _ = self.lightning_policy.training_step(batch, 0)
        # Reset metrics polluted by the dummy forward pass
        self.lightning_policy.train_metrics.reset()
        logging.info("Lazy modules initialized successfully")

    def _tune_hyperparameters(self):
        """Run hyperparameter tuning if enabled.

        Tunes learning rate and/or batch size using PyTorch Lightning Tuner.
        Updates the config and dataloaders with tuned values.
        """
        if not self.config.training.tune_lr:
            return

        if self.trainer is None:
            raise RuntimeError("Trainer must be initialized before tuning.")
        if self.lightning_policy is None:
            raise RuntimeError("Lightning policy must be initialized before tuning.")
        if self.config.experiment.distributed:
            logging.warning(
                "Hyperparameter tuning not supported with distributed training. Skipping..."
            )
            return
        original_callbacks = self.trainer.callbacks.copy()
        self.trainer.callbacks = [
            cb
            for cb in self.trainer.callbacks
            if not isinstance(cb, StochasticWeightAveraging)
        ]
        tuner = Tuner(self.trainer)

        if self.config.training.tune_lr:
            logging.info("Running learning rate tuning...")
            self.lightning_policy.lr = self.config.training.optimizer.lr
            lr_finder_results = tuner.lr_find(
                model=self.lightning_policy,
                min_lr=1e-8,
                max_lr=1.0,
                num_training=100,
            )

            suggested_learning_rate = lr_finder_results.suggestion()
            logging.info(f"Suggested learning rate: {suggested_learning_rate}")
            self.config.training.optimizer.lr = suggested_learning_rate
            self.original_yaml_config.training.optimizer.lr = suggested_learning_rate
            logging.info(
                f"Updated config with learning rate: {suggested_learning_rate}"
            )

        self.trainer.callbacks = original_callbacks
        if self.config.training.tune_lr:
            self.save_config()
            logging.info("Saved updated config with tuned hyperparameters")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.config.experiment.device,
            weights_only=False,
        )
        if self.policy is None:
            raise RuntimeError("Policy must be initialized before loading checkpoint")
        if self.lightning_policy is None:
            raise RuntimeError(
                "LightningPolicy must be initialized before loading checkpoint"
            )
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint format not recognized")

        self.lightning_policy.load_state_dict(checkpoint["state_dict"])
        logging.info("Checkpoint loaded successfully")

        # We need to load explicitly the tokenizer because it's not a torch.nn.Module , differently from the normalizer.
        tokenizer_path = self.output_dir / "tokenizer"
        if tokenizer_path.exists():
            device = torch.device(self.config.experiment.device)
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_path, device=device)
            self.policy.set_tokenizer(self.tokenizer)
            logging.info(f"Tokenizer loaded from {tokenizer_path}")
        else:
            self.tokenizer = None

    def predict(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Predict actions from observations.

        Args:
            obs_dict: Dictionary of observation tensors

        Returns:
            Predicted actions
        """
        if self.lightning_policy is None:
            raise RuntimeError("Policy not initialized. Call run() first.")
        if self.policy is None:
            raise RuntimeError("Policy must be initialized. Call run() first.")
        policy = self.policy
        if (
            self.config.training.use_ema
            and self.trainer is not None
            and hasattr(self.trainer, "callbacks")
        ):
            for callback in self.trainer.callbacks:
                if isinstance(callback, EMACallback) and callback.ema_model is not None:
                    policy = callback.ema_model
                    break
        policy.eval()
        with torch.no_grad():
            return policy.predict_action(obs_dict)
