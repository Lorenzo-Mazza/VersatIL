import argparse
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from constants import ROBOT_STATE_KEY, Cameras, ExplanationType, PolicyType

from dataset.dataloader import EPISODE_FILENAME
from dataset.preprocess import (
    CAMERA_FRAME_KINEMATICS_COLS,
    LEFT_IMAGE_PATH_KEY,
    RECTIFIED_LEFT_IMAGE_PATH_KEY,
    RECTIFIED_RIGHT_IMAGE_PATH_KEY,
    RIGHT_IMAGE_PATH_KEY,
    ROBOT_FRAME_KINEMATICS_COLS,
)
from legacy_config import load_config
from model.explainer import show_cam_on_image
from workspace import (
    ACTWorkspace,
    DiffusionWorkspace,
    FlowMatchingWorkspace,
    PhaseACTWorkspace,
)


class ModelExplainer:
    def __init__(self,
                 device: torch.device,
                 checkpoint_path: str,
                 data_path: str = None,
                 explanation_types: tuple[str, ...] = (ExplanationType.GRADCAM_PLUS_PLUS.value, ExplanationType.SALIENCY_MAP.value,
                                                       ExplanationType.INTEGRATED_GRADIENT.value),
                 explanation_frequency: int = 50
                 ):
        self.checkpoint_path = checkpoint_path
        self.data_path = data_path
        self.config = load_config(checkpoint_path)
        self.config.device = device
        self._load_model()
        self.data_path = data_path
        self.save_dir = os.path.join(self.checkpoint_path, "heatmaps")
        os.makedirs(self.save_dir, exist_ok=True)
        self.dataset = None
        self.explanation_types = explanation_types

        additional_targets = {"right_image": "image"}
        if self.config.use_depth:
            additional_targets["depth"] = "mask"

        self.transform = A.Compose([
            A.Resize(height=self.config.image_height, width=self.config.image_width),
            ToTensorV2(), ],
            additional_targets=additional_targets,
        )
        self.observation_horizon = self.config.obs_horizon
        self.timestep = 0
        self.frequency = explanation_frequency


    def _load_model(self) -> DiffusionWorkspace | FlowMatchingWorkspace | ACTWorkspace:

        if self.config.policy_name == PolicyType.DIFFUSION_POLICY.value:
            workspace = DiffusionWorkspace(config=self.config, is_inference=True)
            workspace.policy = workspace.ema_model if self.config.use_ema else workspace.policy
        elif self.config.policy_name == PolicyType.FLOW_MATCHING.value:
            workspace = FlowMatchingWorkspace(config=self.config, is_inference=True)
            workspace.policy = workspace.ema_model if self.config.use_ema else workspace.policy
        elif self.config.policy_name == PolicyType.ACT.value:
            workspace = ACTWorkspace(config=self.config, is_inference=True)
        elif self.config.policy_name == PolicyType.PHASE_ACT.value:
            workspace = PhaseACTWorkspace(config=self.config, is_inference=True)
        else:
            raise ValueError(f"Unknown policy type: {self.config.policy_name}")
        workspace.load_checkpoint(path=f"{self.checkpoint_path}/latest.pt")
        self.model = workspace
        return workspace


    def get_observation(self) -> dict:
        df = self.dataset.iloc[self.timestep:self.timestep + self.observation_horizon]
        obs = {}
        if self.config.state_dim>0 :
            obs_parts = []
            if self.config.obs_robot_frame:
                obs_parts.append(df[ROBOT_FRAME_KINEMATICS_COLS][:])
            if self.config.obs_camera_frame:
                obs_parts.append(df[CAMERA_FRAME_KINEMATICS_COLS][:])
            robot_state_data = np.concatenate(obs_parts, axis=1)
            obs[ROBOT_STATE_KEY] = torch.from_numpy(robot_state_data)


        for cam in self.config.camera_names:
            if cam == Cameras.DEPTH.value:
                col = Cameras.DEPTH.value
                df[col] = df[LEFT_IMAGE_PATH_KEY].apply(lambda x: x.replace("framesLeft", "stereoDepth").replace(".png", ".tiff"))
            else:
                col = {
                    Cameras.LEFT.value: LEFT_IMAGE_PATH_KEY if not self.config.use_rectified else RECTIFIED_LEFT_IMAGE_PATH_KEY,
                    Cameras.RIGHT.value: RIGHT_IMAGE_PATH_KEY if not self.config.use_rectified else RECTIFIED_RIGHT_IMAGE_PATH_KEY
                }[cam]
            if cam == Cameras.DEPTH.value:
                disp = [cv2.imread(img , cv2.IMREAD_UNCHANGED).astype(np.float32) for img in df[col]]
                depth = [np.where(img > 0, 1.0 / img, 0.0) for img in disp]
                resized = [self.transform(image=img)['depth'] for img in depth]
            else:
                rgb = [ cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img in df[col]]
                resized = [self.transform(image=img)['image'] / 255.0 for img in rgb]
            obs[cam] = torch.stack(resized)
        return obs


    def explain_prediction(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        observation = self.get_observation()
        left_tensor = observation[Cameras.LEFT.value].unsqueeze(0)# Shape: (1, self.observation_horizon, 3, H, W)
        right_tensor = observation[Cameras.RIGHT.value].unsqueeze(0)
        depth_tensor = None
        if self.config.use_depth:
            depth_tensor = observation[Cameras.DEPTH.value].unsqueeze(0)
        obs_dict = {
            Cameras.LEFT.value: left_tensor,
            Cameras.RIGHT.value: right_tensor,
        }
        if self.config.use_depth:
            obs_dict[Cameras.DEPTH.value] = depth_tensor
        if self.config.state_dim>0:
            obs_dict[ROBOT_STATE_KEY] = observation[ROBOT_STATE_KEY].unsqueeze(0)

        heatmaps = self.model.policy.explain_predictions(obs_dict=obs_dict, explanation_types=self.explanation_types)
        return heatmaps, observation


    def explain_episodes(self):
        episodes_paths = [p for p in Path(self.data_path).glob("*/") if p.is_dir() and (p / EPISODE_FILENAME).exists()]
        for path in episodes_paths:
            self.dataset = pd.read_csv(path / EPISODE_FILENAME)
            while self.timestep < len(self.dataset) - 1:
                heatmaps, observation = self.explain_prediction()
                for explanation_type in self.explanation_types:
                    current_heatmaps = heatmaps[explanation_type]
                    for cam in current_heatmaps:
                        heatmap_np = current_heatmaps[cam].cpu().numpy()
                        if self.observation_horizon > 1:
                            for i in range(self.observation_horizon):
                                heatmap_single = heatmap_np[i]
                                obs_single = observation[cam][i]
                                channel = obs_single.shape[0]
                                if channel == 3:
                                    img = obs_single.permute(1, 2, 0).cpu().numpy()
                                else:
                                    normalized = (obs_single[0] / obs_single[0].max()).cpu().numpy()
                                    img = np.repeat(normalized[:, :, np.newaxis], 3, axis=2)
                                cam_img = show_cam_on_image(img, heatmap_single, use_rgb=True, image_weight=0.5)
                                cam_img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
                                folder_name = path.name
                                folder_path = os.path.join(self.save_dir, folder_name)
                                os.makedirs(folder_path, exist_ok=True)
                                save_path = os.path.join(folder_path, f"timestep_{self.timestep}_sub_{i}_{explanation_type}_{cam}.png")
                                cv2.imwrite(save_path, cam_img_bgr)
                                print(f"Saved heatmap for timestep {self.timestep} sub {i} in {save_path}")
                        else:
                            heatmap_np = heatmap_np[0]  # Remove the observation sequence dimension
                            image = observation[cam][0]
                            channel = image.shape[0]
                            if channel == 3:
                                img = image.permute(1, 2, 0).cpu().numpy()
                            else:
                                normalized = (image[0] / image.max()).cpu().numpy()
                                img = np.repeat(normalized[:, :, np.newaxis], 3, axis=2)
                            cam_img = show_cam_on_image(img, heatmap_np, use_rgb=True, image_weight=0.5)
                            cam_img_bgr = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
                            folder_name = path.name
                            folder_path = os.path.join(self.save_dir, folder_name)
                            os.makedirs(folder_path, exist_ok=True)
                            save_path = os.path.join(folder_path, f"timestep_{self.timestep}_{explanation_type}_{cam}.png")
                            cv2.imwrite(save_path, cam_img_bgr)
                            print(f"Saved heatmap for timestep {self.timestep} in {save_path}")
                self.timestep += self.frequency
            self.timestep = 0  # Reset timestep for the next episode

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the model explainer")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/mnt/cluster/workspaces/mazzalore/iros/ACT_checkpoints/20250819_181703_needle_driving_1.0_fixed_angle_no_proprio_rf_deltas/",
        #default="/mnt/cluster/workspaces/mazzalore/iros/DIFFUSION_POLICY_checkpoints/20250816_193428_needle_driving_1.0_fixed_angle_no_proprio_cf_deltas",
        help="Checkpoint path for the model"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default="/mnt/cluster/datasets/threading_il/inference_v5",
        help="Path to the inference data"
    )
    parser.add_argument(
        "--explain-frequency",
        type=int,
        default=50,
        help="Explain every N timesteps (default: 50)"
    )

    parser.add_argument(
        "explanation_types",
        type=str,
        default=f"{ExplanationType.GRADCAM_PLUS_PLUS.value}, {ExplanationType.SALIENCY_MAP.value}, {ExplanationType.ABLATION_CAM.value}", #{ExplanationType.INTEGRATED_GRADIENT.value}",
        nargs='?',
    )
    args = parser.parse_args()
    args.explanation_types = tuple(s.strip() for s in args.explanation_types.split(','))
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explainer = ModelExplainer(
        checkpoint_path=args.checkpoint_path,
        data_path=args.input_path,
        device=device,
        explanation_types=args.explanation_types,
        explanation_frequency=args.explain_frequency
    )
    explainer.explain_episodes()


if __name__ == "__main__":
    main()
