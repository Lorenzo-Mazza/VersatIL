#!/usr/bin/env python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname

import time
import wandb

from legacy_config import DiffusionConfig
from workspace import DiffusionWorkspace
from dataset import get_dataloaders
from model.common.lr_scheduler import get_scheduler


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main():
    """Main function"""
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(
        f"Hello from rank {rank} of {world_size} on {gethostname()} where there are"
        f" {gpus_per_node} allocated GPUs per node.",
        flush=True,
    )

    setup(rank, world_size)
    if rank == 0:
        print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)

    config = DiffusionConfig()
    config.device = torch.device(f"cuda:{local_rank}")
    config.num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])

    if config.use_wandb and local_rank == 0:
        exp_name = f"distributed_experiment_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.login(key="984304f2dd99bd4b4a6e7710e995662d8cc2f504")
        wandb.init(project="threading-diffusion", config=vars(config), name=exp_name)

    workspace = DiffusionWorkspace(config)

    train_loader, val_loader, normalizer = get_dataloaders(config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_loader.dataset, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
        pin_memory=True,
    )
    workspace.policy.set_normalizer(normalizer)
    if workspace.ema_model is not None:
        workspace.ema_model.set_normalizer(normalizer)

    workspace.policy.to(local_rank)
    workspace.ema_model.to(local_rank)
    workspace.policy = DDP(workspace.policy, device_ids=[local_rank])

    workspace.optimizer = torch.optim.AdamW(
        workspace.policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )
    workspace.lr_scheduler = get_scheduler(
        config.lr_schedule,
        optimizer=workspace.optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_loader) * config.num_epochs,
        last_epoch=-1,
    )

    print(f"Starting training for {config.num_epochs} epochs")
    patience = 500
    early_stopping = 0
    for epoch in range(config.num_epochs):
        train_loss = workspace._train_epoch(train_loader)

        if epoch % config.val_every == 0:
            if rank == 0:
                val_loss = workspace._validate(val_loader)
                # Average train loss across nodes for logging
                train_loss_tensor = torch.tensor(train_loss, device="cuda")
                dist.all_reduce(train_loss_tensor)
                train_loss = train_loss_tensor.item() / world_size
                metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                if val_loss < workspace.best_val_loss:
                    workspace.best_val_loss = val_loss
                    workspace.save_checkpoint("best.pt")
                    metrics["best_val_loss"] = val_loss
                    early_stopping = 0
                else:
                    early_stopping += 1

                if config.use_wandb:
                    wandb.log(metrics)

        if epoch % config.checkpoint_every == 0 and rank == 0:
            workspace.save_checkpoint("latest.pt")
        if early_stopping >= patience:
            print(
                f"Early stopping after {early_stopping} epochs without validation loss improving."
            )
            print(f"Best validation loss achieved: {workspace.best_val_loss:.4f}")
            break
        workspace.epoch += 1

    if config.use_wandb:
        wandb.finish()

    cleanup()


if __name__ == "__main__":
    main()
