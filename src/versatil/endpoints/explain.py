"""Endpoint for model explainability and visualization.

This module provides functionality to generate visual explanations for model predictions
using various interpretability techniques (GradCAM, saliency maps, integrated gradients).
"""

import argparse
import logging
import os
from pathlib import Path

import cv2
import hydra.utils
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from versatil.data.constants import Cameras
from versatil.data.processing.image_processor import ImageProcessor
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.explain.constants import ExplanationType
from versatil.explain.explainer import show_cam_on_image
from versatil.training.lightning_policy import LightningPolicy


class ModelExplainer:
    """Generates visual explanations for model predictions."""

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        data_path: str | None = None,
        explanation_types: tuple[str, ...] = (
            ExplanationType.GRADCAM_PLUS_PLUS.value,
            ExplanationType.SALIENCY_MAP.value,
            ExplanationType.INTEGRATED_GRADIENT.value,
        ),
        explanation_frequency: int = 50,
    ):
        """Initialize explainer.

        Args:
            device: Device to run on
            checkpoint_path: Path to model checkpoint directory
            data_path: Path to data for explanation
            explanation_types: Tuple of explanation types to generate
            explanation_frequency: Generate explanations every N timesteps
        """
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.device = device
        self.explanation_types = explanation_types
        self.frequency = explanation_frequency
        self._load_model()
        self._setup_paths()
        self._setup_transforms()
        self.timestep = 0

    def _load_model(self):
        """Load model and config from checkpoint."""
        # Load config first
        config_path = os.path.join(self.checkpoint_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                f"Expected 'config.yaml' in checkpoint directory."
            )
        config = hydra.utils.instantiate(OmegaConf.load(config_path))
        self.config = config
        logging.info(msg=f"Loading config from {config_path}")
        checkpoint_file = os.path.join(self.checkpoint_path, "latest.ckpt")
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(self.checkpoint_path, "last.ckpt")

        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_file}. "
                f"Expected 'latest.ckpt' or 'last.ckpt'"
            )

        logging.info(msg=f"Loading model from {checkpoint_file}")
        self.lightning_model = LightningPolicy.load_from_checkpoint(
            checkpoint_file,
            map_location=self.device,
        )
        self.lightning_model.eval()
        self.policy = self.lightning_model.policy
        self.observation_space = self.policy.observation_space
        self.action_space = self.policy.action_space
        self.dataset_schema = self.config.task.dataset_schema

    def _ensure_configs_are_dataclasses(self):
        """Convert OmegaConf DictConfigs to dataclass instances where needed."""
        if OmegaConf.is_config(self.config.task.action_space):
            config_dict = OmegaConf.to_container(
                self.config.task.action_space, resolve=True
            )
            self.config.task.action_space = ActionSpace(**config_dict)
        if OmegaConf.is_config(self.config.task.observation_space):
            config_dict = OmegaConf.to_container(
                self.config.task.observation_space, resolve=True
            )
            self.config.task.observation_space = ObservationSpace(**config_dict)

    def _setup_paths(self):
        """Setup save directories for heatmaps."""
        self.save_dir = os.path.join(self.checkpoint_path, "heatmaps")
        os.makedirs(self.save_dir, exist_ok=True)

    def _setup_transforms(self):
        """Setup image processing using observation space camera metadata."""
        cameras = [cam.value for cam in Cameras if cam != Cameras.DEPTH]
        if Cameras.DEPTH.value in self.observation_space.camera_keys:
            cameras.append(Cameras.DEPTH.value)
        self.camera_names = cameras

        self.image_processor = ImageProcessor(
            camera_metadata=self.config.task.observation_space.cameras,
            train=False,
        )
        self.observation_horizon = self.config.task.observation_horizon

    def get_observation(self) -> dict[str, torch.Tensor]:
        """Load and preprocess observations from dataset using schema.

        Returns:
            Dictionary of observation tensors
        """
        if self.dataset is None:
            raise RuntimeError("Dataset must be loaded before getting observations")
        df = self.dataset.iloc[self.timestep : self.timestep + self.observation_horizon]
        obs = {}

        for cam in self.camera_names:
            if cam == Cameras.DEPTH.value:
                # Get left image path to compute depth path
                left_col = self.dataset_schema.get_image_path_column(Cameras.LEFT.value)
                if left_col not in df.columns:
                    continue
                left_paths = df[left_col].tolist()
                depth_paths = [
                    self.dataset_schema.compute_depth_path(path) for path in left_paths
                ]

                disp = [
                    cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    for path in depth_paths
                ]
                depth = [np.where(img > 0, 1.0 / img, 0.0) for img in disp]
                resized = [self.transform(image=img)["depth"] for img in depth]
            else:
                # Use schema to get correct column name
                col = self.dataset_schema.get_image_path_column(cam)
                if col not in df.columns:
                    continue
                rgb = []
                for path in df[col]:
                    img = cv2.imread(path, cv2.IMREAD_COLOR)
                    if img is not None:
                        rgb.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                resized = [self.transform(image=img)["image"] / 255.0 for img in rgb]

            obs[cam] = torch.stack(resized)

        return obs

    def explain_prediction(
        self,
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        """Generate explanations for current observation.

        Returns:
            Tuple of (heatmaps, observation) where heatmaps is keyed by explanation type then camera
        """
        observation = self.get_observation()

        for key in observation:
            observation[key] = observation[key].unsqueeze(0).to(self.device)

        heatmaps: dict[str, dict[str, torch.Tensor]] = {}

        logging.info(msg=f"Generating explanations for timestep {self.timestep}")
        logging.info(msg=f"Explanation types: {self.explanation_types}")

        return heatmaps, observation

    def explain_episodes(self):
        """Generate explanations for all episodes in data path."""
        if self.data_path is None:
            raise ValueError("data_path must be set to explain episodes")

        episodes_paths = [
            p
            for p in Path(self.data_path).glob("*/")
            if p.is_dir() and (p / self.dataset_schema.dataset_filename).exists()
        ]

        logging.info(msg=f"Found {len(episodes_paths)} episodes to explain")

        for path in episodes_paths:
            logging.info(msg=f"\nProcessing episode: {path.name}")
            self.dataset = pd.read_csv(path / self.dataset_schema.dataset_filename)
            self.timestep = 0

            while self.timestep < len(self.dataset) - 1:
                heatmaps, observation = self.explain_prediction()

                for explanation_type in self.explanation_types:
                    if explanation_type not in heatmaps:
                        continue

                    current_heatmaps = heatmaps[explanation_type]
                    for cam in current_heatmaps:
                        heatmap_np = current_heatmaps[cam].cpu().numpy()

                        if self.observation_horizon > 1:
                            for i in range(self.observation_horizon):
                                self._save_heatmap(
                                    heatmap_np[i],
                                    observation[cam][0, i],
                                    path.name,
                                    cam,
                                    explanation_type,
                                    sub_idx=i,
                                )
                        else:
                            heatmap_np = heatmap_np[0]
                            image = observation[cam][0, 0]
                            self._save_heatmap(
                                heatmap_np,
                                image,
                                path.name,
                                cam,
                                explanation_type,
                            )

                self.timestep += self.frequency

    def _save_heatmap(
        self,
        heatmap: np.ndarray,
        image: torch.Tensor,
        episode_name: str,
        camera: str,
        explanation_type: str,
        sub_idx: int | None = None,
    ):
        """Save heatmap visualization.

        Args:
            heatmap: Heatmap array
            image: Original image tensor
            episode_name: Name of episode
            camera: Camera name
            explanation_type: Type of explanation
            sub_idx: Optional sub-index for temporal sequences
        """
        channel = image.shape[0]
        if channel == 3:
            img = image.permute(1, 2, 0).cpu().numpy()
        else:
            normalized = (image[0] / image[0].max()).cpu().numpy()
            img = np.repeat(normalized[:, :, np.newaxis], 3, axis=2)

        cam_img = show_cam_on_image(img, heatmap, use_rgb=True, image_weight=0.5)
        cam_img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)

        folder_path = os.path.join(self.save_dir, episode_name)
        os.makedirs(folder_path, exist_ok=True)

        if sub_idx is not None:
            filename = f"timestep_{self.timestep}_sub_{sub_idx}_{explanation_type}_{camera}.png"
        else:
            filename = f"timestep_{self.timestep}_{explanation_type}_{camera}.png"

        save_path = os.path.join(folder_path, filename)
        cv2.imwrite(save_path, cam_img_bgr)
        logging.info(msg=f"Saved heatmap: {save_path}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate model explanations")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input data directory",
    )
    parser.add_argument(
        "--explain-frequency",
        type=int,
        default=50,
        help="Generate explanations every N timesteps",
    )
    parser.add_argument(
        "--explanation-types",
        type=str,
        default=f"{ExplanationType.GRADCAM_PLUS_PLUS.value},{ExplanationType.SALIENCY_MAP.value}",
        help="Comma-separated list of explanation types",
    )
    args = parser.parse_args()
    args.explanation_types = tuple(s.strip() for s in args.explanation_types.split(","))
    return args


def main():
    """Main entry point."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    explainer = ModelExplainer(
        checkpoint_path=args.checkpoint_path,
        data_path=args.input_path,
        device=device,
        explanation_types=args.explanation_types,
        explanation_frequency=args.explain_frequency,
    )

    explainer.explain_episodes()


if __name__ == "__main__":
    main()
