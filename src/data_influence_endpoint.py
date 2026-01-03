import os
from typing import Dict, Optional

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from legacy_config import load_config
from constants import (
    PolicyType,
    InfluenceMethod,
    ACTION_KEY,
    OBSERVATION_KEY,
    IS_PAD_KEY,
    ROBOT_STATE_KEY,
)
from dataset.dataloader import get_dataloaders, EpisodicDataset
from pytorch_utils import dict_apply
from workspace import DiffusionWorkspace, FlowMatchingWorkspace, ACTWorkspace
from kfac import (
    KFAC,
)  # Assuming installed for Hessian approx; fallback to identity if not


class DataInfluenceEstimator:
    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        method: str = InfluenceMethod.INFLUENCE_FUNCTIONS.value,
        val_subsample: int = 1000,
        train_subsample: Optional[int] = None,
        use_kfac: bool = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.config = load_config(checkpoint_path)
        self.config.device = device
        self.method = method
        self.val_subsample = val_subsample
        self.train_subsample = train_subsample
        self.use_kfac = use_kfac
        self._load_model()
        self.save_dir = os.path.join(self.checkpoint_path, "influences")
        os.makedirs(self.save_dir, exist_ok=True)
        self.influence_scores = {}  # episode_idx -> score

    def _load_model(self) -> DiffusionWorkspace | FlowMatchingWorkspace | ACTWorkspace:
        if self.config.policy_name == PolicyType.DIFFUSION_POLICY.value:
            workspace = DiffusionWorkspace(config=self.config, is_inference=True)
            workspace.policy = (
                workspace.ema_model if self.config.use_ema else workspace.policy
            )
        elif self.config.policy_name == PolicyType.FLOW_MATCHING.value:
            workspace = FlowMatchingWorkspace(config=self.config, is_inference=True)
            workspace.policy = (
                workspace.ema_model if self.config.use_ema else workspace.policy
            )
        elif self.config.policy_name == PolicyType.ACT.value:
            workspace = ACTWorkspace(config=self.config, is_inference=True)
        else:
            raise ValueError(f"Unknown policy type: {self.config.policy_name}")
        workspace.load_checkpoint(
            path=f"{self.checkpoint_path}/best.pt"
        )  # Or /latest.pt
        self.model = workspace
        return workspace

    def _subsample_loader(self, loader: DataLoader, subsample_size: int) -> DataLoader:
        dataset_len = (
            len(loader.dataset)
            if hasattr(loader.dataset, "__len__")
            else loader.batch_size * len(loader)
        )
        if subsample_size >= dataset_len:
            return loader
        indices = torch.randperm(dataset_len)[:subsample_size].tolist()
        subset = Subset(loader.dataset, indices)
        return DataLoader(
            subset,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
        )

    def _episode_to_batch(
        self, episode_data: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        len_episode = len(episode_data[ACTION_KEY])
        batch = {
            OBSERVATION_KEY: {},
            ACTION_KEY: torch.from_numpy(episode_data[ACTION_KEY])
            .unsqueeze(0)
            .to(self.config.device),
            IS_PAD_KEY: torch.zeros((1, len_episode), dtype=torch.bool).to(
                self.config.device
            ),
        }
        for cam in self.config.camera_names:
            batch[OBSERVATION_KEY][cam] = (
                torch.from_numpy(episode_data[cam]).unsqueeze(0).to(self.config.device)
            )
        if ROBOT_STATE_KEY in episode_data:
            batch[OBSERVATION_KEY][ROBOT_STATE_KEY] = (
                torch.from_numpy(episode_data[ROBOT_STATE_KEY])
                .unsqueeze(0)
                .to(self.config.device)
            )
        return batch

    def _flatten_gradients(self, model: torch.nn.Module) -> torch.Tensor:
        return torch.cat(
            [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        )

    def _get_average_gradient(
        self, loader: DataLoader, negate: bool = False
    ) -> torch.Tensor:
        avg_grad = None
        count = 0
        for batch in loader:
            batch = dict_apply(
                batch, lambda x: x.to(self.config.device, non_blocking=True)
            )
            metrics = self.model.policy.compute_loss(batch, is_train=False)
            loss = metrics.loss
            self.model.policy.zero_grad()
            loss.backward()
            grad = self._flatten_gradients(self.model.policy)
            if avg_grad is None:
                avg_grad = grad.clone()
            else:
                avg_grad += grad
            count += 1
        avg_grad = avg_grad / count
        return -avg_grad if negate else avg_grad

    def compute_influences(self):
        # Recreate loaders
        train_loader, val_loader, normalizer = get_dataloaders(self.config)
        self.model.policy.set_normalizer(normalizer)
        self.model.policy.eval()

        if self.val_subsample:
            val_loader = self._subsample_loader(val_loader, self.val_subsample)
        if self.train_subsample:
            train_loader = self._subsample_loader(train_loader, self.train_subsample)

        kfac = KFAC(self.model.policy) if self.use_kfac else None

        # grad_J ≈ avg on val: -∇_θ val_loss
        grad_J = self._get_average_gradient(val_loader, negate=True)
        dataset: EpisodicDataset = train_loader.dataset

        selected_episodes = np.nonzero(dataset.sampler.episode_mask)[0]
        for episode_idx in selected_episodes:
            episode_data = dataset.replay_buffer.get_episode(episode_idx)
            batch = self._episode_to_batch(episode_data)
            metrics = self.model.policy.compute_loss(batch, is_train=False)
            loss = metrics.loss
            self.model.policy.zero_grad()
            loss.backward()
            grad_traj = self._flatten_gradients(self.model.policy)

            if kfac:
                h_inv_grad = kfac.inverse_times(grad_traj)
            else:
                h_inv_grad = grad_traj
            score = -torch.dot(grad_J, h_inv_grad).item()

            self.influence_scores[episode_idx] = score

        # Save sorted scores
        sorted_scores = dict(
            sorted(self.influence_scores.items(), key=lambda x: x[1], reverse=True)
        )
        np.savez(self.save_dir / f"{self.method}_scores.npz", **sorted_scores)
        print(
            f"Saved scores to {self.save_dir}. Top episodes: {list(sorted_scores.items())[:5]}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute data influences post-training"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/mnt/cluster/workspaces/mazzalore/iros/ACT_checkpoints/20250819_181703_needle_driving_1.0_fixed_angle_no_proprio_rf_deltas/",
        help="Checkpoint path for the model",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=InfluenceMethod.INFLUENCE_FUNCTIONS.value,
        help="Influence method (default: influence_functions)",
    )
    parser.add_argument(
        "--val-subsample",
        type=int,
        default=1000,
        help="Subsample validation data (default: 1000)",
    )
    parser.add_argument(
        "--train-subsample",
        type=int,
        default=None,
        help="Subsample training data (optional)",
    )
    parser.add_argument(
        "--use-kfac",
        action="store_true",
        help="Use KFAC for Hessian approx (default: False)",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    estimator = DataInfluenceEstimator(
        device=device,
        checkpoint_path=args.checkpoint_path,
        method=args.method,
        val_subsample=args.val_subsample,
        train_subsample=args.train_subsample,
        use_kfac=args.use_kfac,
    )
    estimator.compute_influences()


if __name__ == "__main__":
    main()
