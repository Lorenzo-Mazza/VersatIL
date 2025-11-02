"""Workspace for training and evaluating Imitation Learning policies, supports both distributed and single-node training."""
import copy
import os
import time
from typing import Optional

import torch
import wandb
import numpy as np
from pathlib import Path
from diffusers import DDPMScheduler, DDIMScheduler
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from legacy_config import ACTConfig, PhaseACTConfig
from model.act_policy import ACTPolicy
from model.common.lr_scheduler import get_scheduler
from model.diffusion_policy import DiffusionPolicy
from model.flow_matching_policy import FlowMatchingPolicy
from model.diffusion.ema_model import EMAModel
from dataset.dataloader import get_dataloaders
from legacy_config import FlowMatchingConfig, DiffusionConfig, PolicyConfig, save_config
from legacy_constants import DiffusionScheduler
from model.phase_act_policy import PhaseACTPolicy
from pytorch_utils import dict_apply, optimizer_to
from abc import ABC, abstractmethod
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname


def setup_wandb(config, exp_name):
    """Initialize wandb with API key."""
    api_key = os.environ.get('WANDB_API_KEY')
    if not api_key:
        return None
    wandb.login(key=api_key)
    return wandb.init(project="threading-diffusion", entity="nct-dresden", config=vars(config), name=exp_name)


class TaskWorkspace(ABC):
    def __init__(self, config: PolicyConfig, is_inference:bool = False):
        self.config = config
        self.exp_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{config.exp_name}"
        self.output_dir = Path(f"{config.checkpoint_dir}/{self.exp_name}")
        if not is_inference:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.set_seed()
        self._setup_distributed_training()
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.policy = None
        self.optimizer = None
        self.lr_scheduler = None
        self._setup_policy()
        self._setup_optimizer()
        self._setup_ema()
        if self.distributed_training:
            self.policy = DDP(self.policy, device_ids=[self.local_rank])



    @abstractmethod
    def _setup_policy(self):
        raise NotImplementedError("Subclasses must implement this.")

    def _setup_distributed_training(self):
        self.distributed_training = self.config.distributed_training
        if self.distributed_training:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ["SLURM_PROCID"])
            self.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
            self.local_rank = self.rank - self.gpus_per_node * (self.rank // self.gpus_per_node)
            #self.rank = int(os.environ['RANK'])
            #self.local_rank = int(os.environ['LOCAL_RANK'])
            dist.init_process_group(backend='nccl', world_size=self.world_size, rank=self.rank)
            torch.cuda.set_device(self.local_rank)
            self.config.device = torch.device(f'cuda:{self.local_rank}')
            print(f"Rank {self.rank} of {self.world_size} on {gethostname()}, local_rank {self.local_rank}", flush=True)
            self.config.num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
        else:
            self.world_size = 1
            self.rank = 0
            self.local_rank = 0

    def _setup_optimizer(self):
        if self.config.backbone_has_separate_lr:
            param_dicts = [
                {"params": [p for n, p in self.policy.model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.policy.model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.config.lr_backbone,
                },
            ]
            self.optimizer = torch.optim.AdamW(param_dicts, lr=self.config.learning_rate,
                                          weight_decay=self.config.weight_decay)
        else:
            self.optimizer = torch.optim.AdamW(
                self.policy.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=self.config.betas,
                eps=self.config.eps,
            )
        optimizer_to(self.optimizer, torch.device(self.config.device))

    def _setup_ema(self):
        self.ema_model = None
        self.ema = None
        if self.config.use_ema:
            self.ema_model = copy.deepcopy(self.policy)
            self.ema = EMAModel(model=self.ema_model, power=self.config.ema_power)
            self.ema_model.to(torch.device(self.config.device))

    def _setup_lr_scheduler(self, num_training_steps: Optional[int] = None):
        if self.config.lr_schedule is None:
            self.lr_scheduler = None
            return
        else:
            self.lr_scheduler = get_scheduler(
                self.config.lr_schedule,
                optimizer=self.optimizer,
                num_warmup_steps=self.config.lr_warmup_steps,
                num_training_steps=num_training_steps,
                last_epoch=self.global_step - 1
            )

    def run(self):
        if self.config.use_wandb and self.rank == 0:
            result = setup_wandb(self.config, self.exp_name)
            if result is not None:
                wandb.define_metric("*", step_metric="epoch")
                wandb.watch(self.policy.module if self.distributed_training else self.policy)
            else:
                print("WANDB_API_KEY not set, wandb logging disabled.")
                self.config.use_wandb = False

        train_loader, val_loader, normalizer, gripper_positive_class_weights = get_dataloaders(self.config)
        if gripper_positive_class_weights is not None:
            gripper_positive_class_weights = torch.tensor([gripper_positive_class_weights], device=self.config.device)
        self.gripper_positive_class_weights = gripper_positive_class_weights
        if self.distributed_training:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_loader.dataset, num_replicas=self.world_size, rank=self.rank
            )
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=self.config.batch_size,
                sampler=train_sampler,
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=True
            )

        self.policy.set_normalizer(normalizer)
        if self.config.use_ema:
            self.ema_model.set_normalizer(normalizer)

        self._setup_lr_scheduler(num_training_steps=(len(train_loader) * self.config.num_epochs) // self.config.gradient_accumulate_every)

        if self.rank == 0:
            print(f"Starting training for {self.config.num_epochs} epochs")
            save_config(self.config, self.output_dir)

        patience = 5
        early_stopping = 0
        train_history = []
        validation_history = []
        train_epochs = []
        val_epochs = []

        while self.epoch < self.config.num_epochs:
            if self.distributed_training:
                train_loader.sampler.set_epoch(self.epoch)
            avg_train_metrics = self._train_epoch(train_loader)
            if self.distributed_training:
                loss_tensor = torch.tensor([avg_train_metrics.loss], device=self.config.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_train_metrics.loss = loss_tensor.item() / self.world_size
            train_history.append(avg_train_metrics.to_dict())
            train_epochs.append(self.epoch)
            metrics = {'epoch': self.epoch, 'train_loss': avg_train_metrics.loss,
                       'lr': self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else self.optimizer.param_groups[0]['lr']
                       }
            metrics.update({f'train_{k}': v for k, v in avg_train_metrics.to_dict().items() if k != 'loss'})
            if self.rank == 0:
                print(f"Epoch {self.epoch}: train_loss={avg_train_metrics.loss:.4f}")

            do_val = self.epoch % self.config.val_every == 0
            if do_val and self.rank == 0:
                avg_val_metrics = self._validate(val_loader)
                validation_history.append(avg_val_metrics.to_dict())
                val_epochs.append(self.epoch)
                print(f"val_loss={avg_val_metrics.loss:.4f}")
                metrics.update({'val_loss': avg_val_metrics.loss})
                metrics.update({f'val_{k}': v for k, v in avg_val_metrics.to_dict().items() if k != 'loss'})
                # Add confusion matrix if available (for PhaseACT models)
                if hasattr(self.last_train_metrics_full, 'get_wandb_confusion_matrix'):
                    #cm_plot = self.last_val_metrics_full.get_wandb_confusion_matrix()
                    cm_plot = self.last_train_metrics_full.get_seaborn_confusion_matrix()
                    if cm_plot and self.config.use_wandb:
                        metrics['train_phase_confusion_matrix'] = cm_plot

                if hasattr(self.last_val_metrics_full, 'get_wandb_confusion_matrix'):
                    #cm_plot = self.last_val_metrics_full.get_wandb_confusion_matrix()
                    cm_plot = self.last_val_metrics_full.get_seaborn_confusion_matrix()
                    if cm_plot and self.config.use_wandb:
                        metrics['val_phase_confusion_matrix'] = cm_plot

                if avg_val_metrics.loss < self.best_val_loss:
                    self.best_val_loss = avg_val_metrics.loss
                    self.save_checkpoint('best.pt')
                    metrics['best_val_loss'] = avg_val_metrics.loss
                    early_stopping = 0
                else:
                    early_stopping += 1

            if self.config.use_wandb and self.rank == 0:
                wandb.log(metrics)

            if self.distributed_training:
                stop_tensor = torch.tensor([1 if do_val and early_stopping >= patience else 0], device=self.config.device, dtype=torch.int)
                dist.broadcast(stop_tensor, src=0)
                should_stop = stop_tensor.item() == 1
                best_val_tensor = torch.tensor([self.best_val_loss if do_val else 0.0], device=self.config.device)
                dist.broadcast(best_val_tensor, src=0)
                self.best_val_loss = best_val_tensor.item()
            else:
                should_stop = do_val and early_stopping >= patience

            if self.epoch % self.config.checkpoint_every == 0 and self.rank == 0:
                self.save_checkpoint('latest.pt')
            if self.epoch % self.config.plot_every == 0 and self.epoch > 0 and self.rank == 0:
                self.plot_history(train_history, validation_history, train_epochs, val_epochs)
            if should_stop:
                if self.rank == 0:
                    print(f"Early stopping after {early_stopping} epochs without validation loss improving.")
                    print(f"Best validation loss achieved: {self.best_val_loss:.4f}")
                break
            self.epoch += 1
        if self.config.use_wandb and self.rank == 0:
            wandb.finish()
        if self.distributed_training:
            dist.destroy_process_group()

    def _train_epoch(self, train_loader):
        policy = self.policy.module if self.distributed_training else self.policy
        policy.train()

        epoch_metrics = None
        num_batches = 0

        with tqdm(train_loader, desc=f"Epoch {self.epoch}", disable=self.rank != 0) as pbar:
            for batch in pbar:
                batch = dict_apply(batch, lambda x: x.to(self.config.device, non_blocking=True))
                metrics = policy.compute_loss(batch, is_train=True, gripper_positive_class_weight=self.gripper_positive_class_weights)
                loss = metrics.loss / self.config.gradient_accumulate_every
                loss.backward()


                if self.global_step % self.config.gradient_accumulate_every == 0:
                    if self.config.clip_gradient_norm:
                        # This returns the norm BEFORE clipping
                        original_grad_norm = clip_grad_norm_(policy.parameters(), max_norm=1.0)
                        if self.config.use_wandb and self.rank == 0:
                            clipped_grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=float('inf'))
                            wandb.log({
                                "pre_clip_grad_norm": original_grad_norm,
                                "post_clip_grad_norm": clipped_grad_norm, 
                                "global_step": self.global_step
                            })
                    self.optimizer.step()
                    self.optimizer.zero_grad()


                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                if self.config.use_ema:
                    self.ema.step(policy)

                if epoch_metrics is None:
                    epoch_metrics = metrics
                else:
                    epoch_metrics.accumulate(metrics)
                num_batches += 1
                avg_metrics = epoch_metrics.average(num_batches)
                pbar.set_postfix({'loss': f'{avg_metrics.loss:.4f}'})
                self.global_step += 1
        
        self.last_train_metrics_full = epoch_metrics  

        return epoch_metrics.average(num_batches)

    @torch.no_grad()
    def _validate(self, val_loader):
        policy = self.ema_model if self.ema_model is not None else (self.policy.module if self.distributed_training else self.policy)
        policy.eval()

        epoch_metrics = None
        num_batches = 0

        for i, batch in enumerate(val_loader):
            batch = dict_apply(batch, lambda x: x.to(self.config.device, non_blocking=True))
            metrics = policy.compute_loss(batch, is_train=False, gripper_positive_class_weight=self.gripper_positive_class_weights)
            if epoch_metrics is None:
                epoch_metrics = metrics
            else:
                epoch_metrics.accumulate(metrics)
            num_batches += 1
            
            if i % 10 == 0:
                torch.cuda.empty_cache()


        avg_metrics = epoch_metrics.average(num_batches)
        self.last_val_metrics_full = epoch_metrics  
        return avg_metrics


    def save_checkpoint(self, filename):
        if self.rank != 0:
            return
        policy_state = (self.ema_model if self.ema_model is not None else (self.policy.module if self.distributed_training else self.policy)).state_dict()
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': policy_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        if self.lr_scheduler is not None:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, self.output_dir / filename)
        print(f"Saved checkpoint: {filename}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        (self.ema_model if self.ema_model is not None else self.policy).load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])


    @torch.no_grad()
    def predict_actions(self, obs_dict) -> torch.Tensor:
        policy = self.ema_model if self.ema_model is not None else (self.policy.module if self.distributed_training else self.policy)
        policy.eval()
        obs_dict = {k: v.to(self.config.device) for k, v in obs_dict.items()}
        return policy.predict_action(obs_dict, device=self.config.device)


    def plot_history(self, train_history, validation_history, train_epochs, val_epochs):
        # save training curves
        plot_dict = {}
        for key in train_history[0]:
            plot_path = self.output_dir / f'train_val_{key}_seed_{self.config.seed}.png'
            plt.figure()
            train_values = [summary[key] for summary in train_history]
            val_values = [summary[key] for summary in validation_history]
            plt.plot(train_epochs, train_values, label='train')
            plt.plot(val_epochs, val_values, label='validation')
            plt.tight_layout()
            plt.legend()
            plt.title(key)
            plt.savefig(plot_path)
            plot_dict[f"plot_{key}"] = wandb.Image(plot_path)
        if self.config.use_wandb:
            wandb.log({**plot_dict, 'epoch': self.epoch})
        print(f'Saved plots to {self.output_dir}')

    def set_seed(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)


class DiffusionWorkspace(TaskWorkspace):
    def __init__(self, config: DiffusionConfig, is_inference: bool = False):
        super().__init__(config,  is_inference=is_inference)

    def _setup_policy(self):
        match self.config.diffusion_scheduler:
            case DiffusionScheduler.DDIM.value:
                self.noise_scheduler = DDIMScheduler(
                    num_train_timesteps=self.config.num_train_timesteps,
                    beta_schedule=self.config.beta_schedule,
                    clip_sample=self.config.clip_sample,
                    beta_start=self.config.beta_start,
                    beta_end=self.config.beta_end,
                    steps_offset=self.config.steps_offset,
                    set_alpha_to_one=self.config.set_alpha_to_one,
                    prediction_type=self.config.prediction_type
                )
            case DiffusionScheduler.DDPM.value:
                self.noise_scheduler = DDPMScheduler(
                    num_train_timesteps=self.config.num_train_timesteps,
                    beta_schedule=self.config.beta_schedule,
                    clip_sample=self.config.clip_sample,
                    beta_start=self.config.beta_start,
                    beta_end=self.config.beta_end,
                    variance_type=self.config.scheduler_variance_type,
                    prediction_type=self.config.prediction_type
                )
            case _:
                raise ValueError("Invalid diffusion scheduler. Supported types are DDIM and DDPM.")

        self.policy = DiffusionPolicy(
            shape_meta=self.config.shape_meta,
            architecture=self.config.action_model_architecture,
            noise_scheduler=self.noise_scheduler,
            horizon=self.config.pred_horizon,
            n_action_steps=self.config.action_horizon,
            n_obs_steps=self.config.obs_horizon,
            down_dims=self.config.down_dims,
            use_group_norm=True,
            obs_as_global_cond=True,
            crop_size=(int(self.config.image_height), int(self.config.image_width)), # 95% of image size
            random_crop=False,
            imagenet_norm=False,
            num_inference_steps=self.config.num_inference_steps,
            camera_names=self.config.camera_names,
            backbone=self.config.backbone,
            pretrained_backbone=self.config.pretrained_backbone,
            depth_fusion_strategy=self.config.depth_fusion,
            predict_gripper_action=self.config.predict_gripper_action,
            freeze_dformer=self.config.freeze_dformer,
            dformer_checkpoint_path=self.config.dformer_checkpoint_path
        )
        self.policy.to(self.config.device)

class FlowMatchingWorkspace(TaskWorkspace):
    def __init__(self, config: FlowMatchingConfig, is_inference:bool = False):
        super().__init__(config,  is_inference=is_inference)

    def _setup_policy(self):
        self.policy = FlowMatchingPolicy(
            shape_meta=self.config.shape_meta,
            architecture=self.config.action_model_architecture,
            horizon=self.config.pred_horizon,
            n_action_steps=self.config.action_horizon,
            n_obs_steps=self.config.obs_horizon,
            down_dims=self.config.down_dims,
            use_group_norm=True,
            obs_as_global_cond=True,
            crop_size=(int(0.95*self.config.image_height), int(0.95*self.config.image_width)), # 95% of image size
            random_crop=self.config.random_crop,
            imagenet_norm=False,
            num_inference_steps=self.config.num_inference_steps,
            camera_names=self.config.camera_names,
            backbone=self.config.backbone,
            pretrained_backbone=self.config.pretrained_backbone,
            depth_fusion_strategy=self.config.depth_fusion,
            sigma=self.config.sigma,
            predict_gripper_action=self.config.predict_gripper_action,
            freeze_dformer=self.config.freeze_dformer,
            dformer_checkpoint_path=self.config.dformer_checkpoint_path
        )
        self.policy.to(self.config.device)


class ACTWorkspace(TaskWorkspace):
    def __init__(self, config: ACTConfig, is_inference:bool = False):
        super().__init__(config, is_inference=is_inference)

    def _setup_policy(self):
        self.policy = ACTPolicy(self.config)
        self.policy.to(self.config.device)


class PhaseACTWorkspace(TaskWorkspace):
    def __init__(self, config: PhaseACTConfig, is_inference:bool = False):
        super().__init__(config, is_inference=is_inference)

    def _setup_policy(self):
        self.policy = PhaseACTPolicy(self.config)
        self.policy.to(self.config.device)
