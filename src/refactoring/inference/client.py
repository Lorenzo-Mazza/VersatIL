# mypy: ignore-errors
"""Inference client for real-time model deployment.

This module provides the InferenceClient class that interfaces with the
imitation_learning_toolkit's AbstractModelClient for real-time robot control.
"""
import logging
import os
import time

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from imitation_learning_toolkit.sockets.model_client import AbstractModelClient, Action
from omegaconf import OmegaConf

from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.data.constants import (
    GRIPPER_ACTION_KEY,
    POSITION_ACTION_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    Cameras,
)
from refactoring.training.lightning_policy import LightningPolicy


class InferenceClient(AbstractModelClient):
    """Client for real-time inference with trained policies."""

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        model_server_address: str = "localhost",
        model_server_port: int = 5555,
        temporal_agg: bool = True,
        update_rate_hz: float | None = None,
        **kwargs,
    ):
        """Initialize inference client.

        Args:
            device: Device to run inference on
            checkpoint_path: Path to checkpoint directory
            model_server_address: Address of the model server controlling the robot
            model_server_port: Port of the model server
            temporal_agg: Whether to use temporal aggregation for actions
            update_rate_hz: Update frequency in Hz (overrides checkpoint config)
            **kwargs: Additional arguments passed to AbstractModelClient
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.temporal_agg = temporal_agg

        self._load_model()

        observation_horizon = self.policy.decoder.observation_horizon
        prediction_horizon = self.policy.prediction_horizon

        image_height = self.config.task.dataloader.image_height
        image_width = self.config.task.dataloader.image_width

        obs_space = self.policy.observation_space
        action_space = self.policy.action_space

        use_depth = Cameras.DEPTH.value in obs_space.camera_keys
        predict_gripper = action_space.has_gripper
        predict_in_camera_frame = action_space.predict_in_camera_frame
        predicts_delta = action_space.deltas_as_actions
        obs_robot_frame = obs_space.use_proprio_base_frame
        obs_camera_frame = obs_space.use_proprio_camera_frame

        action_dim = action_space.get_total_action_dim()

        if update_rate_hz is None:
            update_rate_hz = 10.0

        super().__init__(
            model_server_address=model_server_address,
            model_server_port=model_server_port,
            observation_buffer_size=observation_horizon,
            request_depth=use_depth,
            request_rectified_images=False,
            request_gripper_state=predict_gripper,
            predicts_in_camera_frame=predict_in_camera_frame,
            predicts_delta=predicts_delta,
            obs_robot_frame=obs_robot_frame,
            obs_camera_frame=obs_camera_frame,
            device=device,
            update_rate_hz=update_rate_hz,
            enable_logging=False,
        )

        self.image_height = image_height
        self.image_width = image_width
        self.action_dim = action_dim
        self.action_horizon = prediction_horizon

        additional_targets = {"right_image": "image"}
        if use_depth:
            additional_targets["depth"] = "mask"

        self.transform = A.Compose(
            [
                A.Resize(height=self.image_height, width=self.image_width),
                ToTensorV2(),
            ],
            additional_targets=additional_targets,
        )

        self.max_timesteps = 10000
        position_dim = action_space.position_dim
        self.all_time_actions = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.action_horizon, position_dim]
        ).to(self.device)
        self.timestep = 0
        self.current_all_actions = None

    def _ensure_configs_are_dataclasses(self):
        """Convert OmegaConf DictConfigs to dataclass instances where needed.

        This ensures that configs with methods (ActionSpace, ObservationSpace)
        are actual dataclass instances, not OmegaConf DictConfigs, so their
        methods can be called.

        This is necessary because:
        - ActionSpace has get_total_action_dim() and get_required_zarr_keys() methods
        - ObservationSpace has get_required_zarr_keys() method
        - OmegaConf DictConfigs don't have these methods
        """
        # Convert ActionSpace if it's an OmegaConf DictConfig
        if OmegaConf.is_config(self.config.task.action_space):
            config_dict = OmegaConf.to_container(self.config.task.action_space, resolve=True)
            self.config.task.action_space = ActionSpace(**config_dict)

        # Convert ObservationSpace if it's an OmegaConf DictConfig
        if OmegaConf.is_config(self.config.task.observation_space):
            config_dict = OmegaConf.to_container(self.config.task.observation_space, resolve=True)
            self.config.task.observation_space = ObservationSpace(**config_dict)

    def _load_model(self) -> LightningPolicy:
        """Load model and config from checkpoint.

        Returns:
            Loaded LightningPolicy model
        """
        config_path = os.path.join(self.checkpoint_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                f"Expected 'config.yaml' in checkpoint directory."
            )

        print(f"Loading config from {config_path}")
        self.config = OmegaConf.load(config_path)
        self._ensure_configs_are_dataclasses()

        checkpoint_file = os.path.join(self.checkpoint_path, "latest.ckpt")
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(self.checkpoint_path, "last.ckpt")

        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_file}. "
                f"Expected 'latest.ckpt' or 'last.ckpt'"
            )

        print(f"Loading model from {checkpoint_file}")
        self.model = LightningPolicy.load_from_checkpoint(
            checkpoint_file,
            map_location=self.device,
        )
        self.model.eval()
        self.policy = self.model.policy
        return self.model

    def get_actions_from_model(self) -> list[Action]:
        """Compute next actions using the trained policy model.

        Returns:
            List of Action objects
        """
        total_start_time = time.time()
        print(f"\n=== TIMESTEP {self.timestep} - Starting get_actions_from_model ===")

        preprocessing_start_time = time.time()
        print(f"[TIMING] Input preprocessing started at: {preprocessing_start_time:.6f}")

        if self.obs_camera_frame and self.obs_robot_frame:
            state_dim = 6
        elif self.obs_camera_frame or self.obs_robot_frame:
            state_dim = 3
        else:
            state_dim = 0

        if state_dim > 0:
            last_states = self.robot_state_buffer[-self.observation_buffer_size :]
            qpos = np.array([state[:state_dim] for state in last_states])
            qpos_tensor = torch.tensor(qpos, dtype=torch.float32).unsqueeze(0)
        else:
            qpos_tensor = None

        left_img_list = self.left_image_buffer[-self.observation_buffer_size :]
        right_img_list = self.right_image_buffer[-self.observation_buffer_size :]

        depth_processing_start = time.time()
        depth_imgs = None
        if self.request_depth:
            depth_img_list = self.depth_buffer[-self.observation_buffer_size :]
            transformed = [
                self.transform(image=left_np, right_image=right_np, depth=depth_np)
                for left_np, right_np, depth_np in zip(left_img_list, right_img_list, depth_img_list)
            ]
            depth_tensors = [t["depth"] for t in transformed]
            depth_imgs = torch.stack(depth_tensors).unsqueeze(0).unsqueeze(-3)
            max_depth = 9.352702140808105
            depth_imgs = torch.clamp(depth_imgs, min=0.0, max=max_depth)
        else:
            transformed = [
                self.transform(image=left_np, right_image=right_np)
                for left_np, right_np in zip(left_img_list, right_img_list)
            ]

        print(f"[TIMING] Depth plus RGB transform took: {time.time() - depth_processing_start:.6f} seconds")
        rgb_processing_start = time.time()
        left_tensors = [t["image"] / 255.0 for t in transformed]
        right_tensors = [t["right_image"] / 255.0 for t in transformed]
        left_imgs = torch.stack(left_tensors).unsqueeze(0)
        right_imgs = torch.stack(right_tensors).unsqueeze(0)
        print(f"[TIMING] RGB processing took: {time.time() - rgb_processing_start:.6f} seconds")

        obs_dict = {
            Cameras.LEFT.value: left_imgs,
            Cameras.RIGHT.value: right_imgs,
        }

        if state_dim > 0:
            if self.obs_robot_frame and self.obs_camera_frame:
                obs_dict[PROPRIO_OBS_ROBOT_FRAME_KEY] = qpos_tensor[:, :, :3]
                obs_dict[PROPRIO_OBS_CAMERA_FRAME_KEY] = qpos_tensor[:, :, 3:]
            elif self.obs_robot_frame:
                obs_dict[PROPRIO_OBS_ROBOT_FRAME_KEY] = qpos_tensor
            elif self.obs_camera_frame:
                obs_dict[PROPRIO_OBS_CAMERA_FRAME_KEY] = qpos_tensor

        if self.request_depth:
            obs_dict[Cameras.DEPTH.value] = depth_imgs

        if self.predicts_in_camera_frame and self.obs_camera_frame and self.obs_robot_frame:
            current_robot_position = self.robot_state_buffer[-1][3:6]
        else:
            current_robot_position = self.robot_state_buffer[-1][:3]

        preprocessing_end_time = time.time()
        preprocessing_duration = preprocessing_end_time - preprocessing_start_time
        print(f"[TIMING] Input preprocessing completed in: {preprocessing_duration:.6f} seconds")

        inference_start_time = time.time()
        print(f"[TIMING] Model inference started at: {inference_start_time:.6f}")

        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict=obs_dict)

        self.current_all_actions = action_dict[POSITION_ACTION_KEY]

        inference_end_time = time.time()
        inference_duration = inference_end_time - inference_start_time
        print(f"[TIMING] Model inference completed in: {inference_duration:.6f} seconds")

        postprocessing_start_time = time.time()
        print(f"[TIMING] Post-processing started at: {postprocessing_start_time:.6f}")

        if self.temporal_agg:
            temporal_agg_start = time.time()
            raw_action = self.get_exponential_averaged_action()
            print(f"[TIMING] Temporal aggregation took: {time.time() - temporal_agg_start:.6f} seconds")

            raw_action = raw_action.cpu().detach().numpy()
            raw_position_action = raw_action[:3]

            if self.policy.action_space.has_gripper:
                raw_gripper_tensor = action_dict[GRIPPER_ACTION_KEY][0, 0]
                raw_gripper_action = raw_gripper_tensor.cpu().detach().numpy()
            else:
                raw_gripper_action = None

            if not self.predicts_delta:
                raw_position_action = raw_position_action - current_robot_position

            # Roll = 0.0 for now
            robot_action = np.concatenate((raw_position_action[:3], [0.0]))
            if self.policy.action_space.has_gripper:
                assert raw_gripper_action is not None
                gripper_action = raw_gripper_action > 0.5
            else:
                gripper_action = None
            actions = [Action(robot_action=robot_action, gripper_action=gripper_action)]
        else:
            actions = []
            for i in range(self.action_horizon):
                raw_action = self.current_all_actions[0, i].cpu().detach().numpy()
                raw_position_action = raw_action[:3]

                if self.policy.action_space.has_gripper:
                    raw_gripper_tensor = action_dict[GRIPPER_ACTION_KEY][0, i]
                    raw_gripper_action = raw_gripper_tensor.cpu().detach().numpy()
                else:
                    raw_gripper_action = None

                if not self.predicts_delta:
                    raw_position_action = raw_position_action - current_robot_position

                robot_action = np.concatenate((raw_position_action[:3], [0.0]))
                if self.policy.action_space.has_gripper:
                    assert raw_gripper_action is not None
                    gripper_action = raw_gripper_action > 0.5
                else:
                    gripper_action = None
                actions.append(Action(robot_action=robot_action, gripper_action=gripper_action))

        postprocessing_end_time = time.time()
        postprocessing_duration = postprocessing_end_time - postprocessing_start_time
        print(f"[TIMING] Post-processing completed in: {postprocessing_duration:.6f} seconds")

        self.timestep += 1

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        print(f"\n[TIMING SUMMARY] Timestep {self.timestep - 1}:")
        print(f"  - Preprocessing: {preprocessing_duration:.6f}s ({preprocessing_duration/total_duration*100:.1f}%)")
        print(f"  - Model inference: {inference_duration:.6f}s ({inference_duration/total_duration*100:.1f}%)")
        print(f"  - Post-processing: {postprocessing_duration:.6f}s ({postprocessing_duration/total_duration*100:.1f}%)")
        print(f"  - TOTAL: {total_duration:.6f}s")
        print(f"  - Effective FPS: {1.0/total_duration:.2f}")
        print(f"=== TIMESTEP {self.timestep - 1} COMPLETE ===\n")

        if self.enable_logging:
            logging.log(level=logging.INFO, msg=f"{actions=}")
        print(actions)
        return actions

    def get_exponential_averaged_action(self) -> torch.Tensor:
        """Average exponentially the actions predicted for the current timestep.

        Returns:
            Exponentially averaged action tensor
        """
        self.all_time_actions[
            [self.timestep], self.timestep : self.timestep + self.action_horizon
        ] = self.current_all_actions
        actions_for_curr_step = self.all_time_actions[:, self.timestep]
        actions_populated = torch.all(actions_for_curr_step != 0, dim=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0)
        return raw_action
