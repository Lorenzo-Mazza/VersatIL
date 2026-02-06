# mypy: ignore-errors
"""Workspace for training and evaluating policies using PyTorch Lightning."""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.tuner import Tuner
from torch.utils import data

from versatil.common.tensor_ops import to_device
from versatil.configs import MainConfig
from versatil.data.dataloader import get_dataloaders
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.tokenization import Tokenizer
from versatil.metrics import MoELoss
from versatil.models.decoding.algorithm import VariationalAlgorithm
from versatil.models.decoding.decoders.factory.free_action_transformer import (
    FreeActionTransformer,
)
from versatil.models.decoding.decoders.factory.phase_act import PhaseACT
from versatil.models.policy import Policy
from versatil.training.callbacks import (
    ConfusionMatrixCallback,
    EMACallback,
    GradientNormCallback,
    ReduceLROnPlateauCallback,
    ExpertUsageCallback,
    LatentVisualizationCallback,
    ResumableEarlyStopping,
)
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
        self.exp_name = f"{main_config_name}/{additional_exp_name}"
        self.config.experiment.name = self.exp_name
        self.original_yaml_config.experiment.name = self.exp_name
        self.output_dir = (
            Path(config.experiment.checkpoint_folder)
            / main_config_name
            / additional_exp_name
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
        assert self.trainer is not None, "Trainer should be initialized"
        self.trainer.fit(model=self.lightning_policy, ckpt_path=resume_checkpoint_path)
        logging.info(f"Training completed. Best checkpoint saved to {self.output_dir}")

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.config.experiment.seed)
        np.random.seed(self.config.experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.experiment.seed)

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
        logging.info(f"Val dataset size: {len(self.val_loader.dataset)} samples")
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
        self.policy.set_normalizer(self.normalizer)
        self.policy.set_tokenizer(self.tokenizer)
        self.policy.set_denoising_thresholds(self.denoising_thresholds)
        self.policy.set_gripper_class_weights(self.gripper_class_weights)
        # Calculate total training steps for LR scheduling
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
        self.trainer = pl.Trainer(
            max_epochs=self.config.training.num_epochs,
            accelerator="gpu" if "cuda" in self.config.experiment.device else "cpu",
            devices="auto" if self.config.experiment.distributed else 1,
            strategy=strategy,
            logger=self.logger,
            callbacks=callbacks,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=self.config.training.gradient_accumulate_every,
            check_val_every_n_epoch=self.config.experiment.val_every,
            log_every_n_steps=50,
            enable_progress_bar=True,
            enable_model_summary=True,
            deterministic=False,  # For performance
            precision=self.config.experiment.precision,
        )

        logging.info(f"Trainer created with {len(callbacks)} callbacks")

    def _create_callbacks(self):
        """Create training callbacks.

        Returns:
            List of Lightning callbacks
        """
        callbacks = []

        if self.config.training.use_ema:
            ema_callback = EMACallback(
                power=self.config.training.ema_power,
            )
            callbacks.append(ema_callback)
            logging.info(f"Added EMA callback (power={self.config.training.ema_power})")

        checkpoint_callback_best = ModelCheckpoint(
            dirpath=self.output_dir,
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,  # Keep top 3 best models
            save_last=False,
            verbose=True,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback_best)
        logging.info(f"Added ModelCheckpoint callback (top-k=3)")

        checkpoint_callback_latest = ModelCheckpoint(
            dirpath=self.output_dir,
            filename="latest-{epoch:02d}",
            save_top_k=1,  # Save only last
            every_n_epochs=self.config.experiment.checkpoint_every,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback_latest)
        logging.info(
            f"Added latest checkpoint callback (every {self.config.experiment.checkpoint_every} epochs)"
        )
        early_stopping_callback = ResumableEarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=self.config.training.early_stopping_patience,
            verbose=True,
        )
        callbacks.append(early_stopping_callback)
        logging.info("Added EarlyStopping callback (patience=10)")

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
                f"Added SWA callback (lr={self.config.training.swa_lrs}, "
                f"start_epoch={swa_epoch_start}, annealing_epochs={self.config.training.swa_annealing_epochs})"
            )

        if isinstance(self.policy.decoder, PhaseACT):
            cm_callback = ConfusionMatrixCallback(
                log_every_n_epochs=self.config.experiment.val_every,
            )
            callbacks.append(cm_callback)
            logging.info("Added ConfusionMatrix callback for phase classification")

        if isinstance(self.policy.algorithm, VariationalAlgorithm) or isinstance(
            self.policy.decoder, FreeActionTransformer
        ):
            latent_vis_callback = LatentVisualizationCallback(
                log_every_n_epochs=self.config.experiment.val_every,
            )
            callbacks.append(latent_vis_callback)
            logging.info("Added LatentVisualization callback for variational algorithm")

        if self.config.training.reduce_lr_on_plateau:
            reduce_lr_callback = ReduceLROnPlateauCallback(
                patience=self.config.training.reduce_lr_patience,
                cooldown=self.config.training.reduce_lr_cooldown,
            )
            callbacks.append(reduce_lr_callback)
            logging.info(
                f"Added ReduceLROnPlateau callback (patience={self.config.training.reduce_lr_patience})"
            )

        if any(
            isinstance(module, MoELoss)
            for module in self.policy.loss_module.loss_modules.values()
        ):
            expert_usage_callback = ExpertUsageCallback(log_every_n_epochs=1)
            callbacks.append(expert_usage_callback)
            logging.info("Added ExpertUsage callback for MoE loss")

        return callbacks

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
        self.lightning_policy.eval()
        with torch.no_grad():
            _ = self.lightning_policy.training_step(batch, 0)
        self.lightning_policy.train()
        logging.info("Lazy modules initialized successfully")

    def _tune_hyperparameters(self):
        """Run hyperparameter tuning if enabled.

        Tunes learning rate and/or batch size using PyTorch Lightning Tuner.
        Updates the config and dataloaders with tuned values.
        """
        if not self.config.training.tune_lr:
            return

        assert self.trainer is not None, "Trainer must be initialized"
        assert self.lightning_policy is not None, "Lightning policy must be initialized"
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

            suggested_lr = lr_finder_results.suggestion()
            logging.info(f"Suggested learning rate: {suggested_lr}")
            self.config.training.optimizer.lr = suggested_lr
            self.original_yaml_config.training.optimizer.lr = suggested_lr
            logging.info(f"Updated config with learning rate: {suggested_lr}")

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
            raise RuntimeError("LightningPolicy must be initialized before loading checkpoint")
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

    def predict(self, obs_dict):
        """Predict actions from observations.

        Args:
            obs_dict: Dictionary of observation tensors

        Returns:
            Predicted actions
        """
        if self.lightning_policy is None:
            raise RuntimeError("Policy not initialized. Call run() first.")
        assert self.policy is not None, "Policy must be initialized"
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
