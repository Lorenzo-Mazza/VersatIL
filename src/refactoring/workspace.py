# mypy: ignore-errors
"""Workspace for training and evaluating policies using PyTorch Lightning."""

import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from torch.utils import data

from refactoring.configs.main import MainConfig
from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.data.dataloader import get_dataloaders
from refactoring.data.normalize.normalizer import LinearNormalizer
from refactoring.data.tokenize import Tokenizer
from refactoring.models.policy import Policy
from refactoring.training.callbacks import ConfusionMatrixCallback, EMACallback
from refactoring.training.lightning_policy import LightningPolicy


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

    def __init__(self, config: MainConfig):
        """Initialize workspace.

        Args:
            config: Main configuration containing all settings
        """
        self.config = config
        self._ensure_configs_are_dataclasses()
        self.exp_name = config.experiment.name
        self.output_dir = Path(config.experiment.checkpoint_folder) / self.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._set_seed()
        self.policy: Policy | None = None
        self.lightning_policy: LightningPolicy | None = None
        self.trainer: pl.Trainer | None = None
        self.train_loader: data.DataLoader | None = None
        self.val_loader: data.DataLoader | None = None
        self.normalizer: LinearNormalizer | None = None
        self.tokenizer: Tokenizer | None = None

        self.gripper_class_weights: torch.Tensor | None = None
        logging.info(f"Workspace initialized for experiment: {self.exp_name}")
        logging.info(f"Output directory: {self.output_dir}")
        self.save_config()

    def _ensure_configs_are_dataclasses(self):
        """Convert OmegaConf DictConfigs to dataclass instances where needed.

        This ensures that configs with methods (ActionSpace, ObservationSpace)
        are actual dataclass instances, not OmegaConf DictConfigs, so their
        methods can be called.

        This is necessary because:
        - ActionSpace has get_total_action_dim() and get_required_zarr_keys() methods
        - ObservationSpace has get_required_zarr_keys() method
        """
        if OmegaConf.is_config(self.config.task.action_space):
            config_dict = OmegaConf.to_container(self.config.task.action_space, resolve=True)
            self.config.task.action_space = ActionSpace(**config_dict)
        if OmegaConf.is_config(self.config.task.observation_space):
            config_dict = OmegaConf.to_container(self.config.task.observation_space, resolve=True)
            self.config.task.observation_space = ObservationSpace(**config_dict)

    def save_config(self):
        """Save configuration to YAML file in output directory.

        The config is saved as 'config.yaml' and is required for inference
        and model explanation. This method should be called after workspace
        initialization to ensure the config is available for later use.
        """
        config_path = self.output_dir / "config.yaml"
        OmegaConf.save(self.config, config_path)
        logging.info(f"Config saved to {config_path}")


    def run(self):
        """Run the complete training workflow."""
        self._setup_data()
        self._setup_policy()
        self._setup_trainer()
        logging.info("Starting training...")
        assert self.trainer is not None, "Trainer should be initialized"
        self.trainer.fit(
            model=self.lightning_policy,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )
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

        # Store gripper class weights if needed
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

    def _setup_policy(self):
        """Instantiate policy and wrap with Lightning."""
        logging.info("Instantiating policy...")
        # Config normalization already done in __init__, so observation_space and action_space are dataclasses
        self.policy: Policy = instantiate(self.config.policy)
        self.policy.set_normalizer(self.normalizer)
        self.policy.set_tokenizer(self.tokenizer)

        # Calculate total training steps for LR scheduling
        # Steps per epoch = len(train_loader) // gradient_accumulate_every
        # Total steps = steps_per_epoch * num_epochs
        steps_per_epoch = len(self.train_loader) // self.config.training.gradient_accumulate_every
        total_training_steps = steps_per_epoch * self.config.training.num_epochs

        # Wrap with Lightning
        self.lightning_policy = LightningPolicy(
            policy=self.policy,
            training_config=self.config.training,
            total_training_steps=total_training_steps,
        )
        logging.info(f"Policy created: {self.policy.__class__.__name__}")
        logging.info(f"Total training steps: {total_training_steps}")

    def _setup_trainer(self):
        """Setup PyTorch Lightning trainer with callbacks and logger."""
        callbacks = self._create_callbacks()
        logger = self._create_logger()
        strategy = self._create_strategy()

        # Gradient clipping
        gradient_clip_val = None
        if self.config.training.clip_gradient_norm:
            gradient_clip_val = self.config.training.clip_max_norm

        # Set float32 matmul precision for Tensor Cores if configured
        if self.config.experiment.float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(self.config.experiment.float32_matmul_precision)
            logging.info(
                f"Set float32 matmul precision to '{self.config.experiment.float32_matmul_precision}'"
            )

        # Create trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config.training.num_epochs,
            accelerator="gpu" if "cuda" in self.config.experiment.device else "cpu",
            devices="auto" if self.config.experiment.distributed else 1,
            strategy=strategy,
            logger=logger,
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

        # Callback for exponential moving average of model weights
        if self.config.training.use_ema:
            ema_callback = EMACallback(
                power=self.config.training.ema_power,
            )
            callbacks.append(ema_callback)
            logging.info(f"Added EMA callback (power={self.config.training.ema_power})")

        # Model checkpointing - save top-k best models based on val_loss
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
        logging.info("Added ModelCheckpoint callback (top-k=3)")

        # Save latest checkpoint periodically
        checkpoint_callback_latest = ModelCheckpoint(
            dirpath=self.output_dir,
            filename="latest-{epoch:02d}",
            save_top_k=-1,  # Save all
            every_n_epochs=self.config.experiment.checkpoint_every,
            save_last=True,
            verbose=True,
            auto_insert_metric_name=False,
        )
        callbacks.append(checkpoint_callback_latest)
        logging.info(f"Added latest checkpoint callback (every {self.config.experiment.checkpoint_every} epochs)")

        # Confusion matrix callback for phase models
        if self.config.task.action_space.task_has_phases:
            cm_callback = ConfusionMatrixCallback(
                log_every_n_epochs=self.config.experiment.val_every,
            )
            callbacks.append(cm_callback)
            logging.info("Added ConfusionMatrix callback for phase classification")

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
        wandb_logger.log_hyperparams(self.config)
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

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.experiment.device)
        assert self.policy is not None, "Policy must be initialized before loading checkpoint"
        assert self.lightning_policy is not None, "LightningPolicy must be initialized"
        if "state_dict" in checkpoint:
            # Lightning checkpoint format
            self.lightning_policy.load_state_dict(checkpoint["state_dict"])
        else:
            raise ValueError("Checkpoint format not recognized")
        logging.info("Checkpoint loaded successfully")

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
        # Use EMA model if available
        if self.config.training.use_ema and self.trainer is not None and hasattr(self.trainer, "callbacks"):
            for callback in self.trainer.callbacks:
                if isinstance(callback, EMACallback) and callback.ema_model is not None:
                    policy = callback.ema_model
                    break
        policy.eval()
        with torch.no_grad():
            return policy.predict_action(obs_dict)
