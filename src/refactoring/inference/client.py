# mypy: ignore-errors
"""Inference client for real-time model deployment.

This module provides the InferenceClient class that interfaces with the
imitation_learning_toolkit's AbstractModelClient for real-time robot control.

Note:
    The Inference Client uses as convention delta actions for position and orientation.
    This means that regardless of how the policy was trained (predicting absolute or delta actions),
    the client will convert absolute predictions to deltas before sending commands to the Policy Server.
"""
import logging
import os
import time

from refactoring.models.policy import Policy
from refactoring.training.constants import PrecisionType, MAP_PRECISION_TO_DTYPE

logging.basicConfig(level=logging.INFO)

import albumentations as A
import hydra
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from imitation_learning_toolkit.sockets.model_client import AbstractModelClient, Action
from omegaconf import OmegaConf

from refactoring.configs import MainConfig
from refactoring.data.action_processor import ActionProcessor
from refactoring.data.constants import (
    GRIPPER_ACTION_KEY,
    POSITION_ACTION_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    Cameras, ORIENTATION_ACTION_KEY, GripperType, LANGUAGE_KEY,
)
from refactoring.data.tokenization.tokenizer import Tokenizer
from refactoring.training.lightning_policy import LightningPolicy


class InferenceClient(AbstractModelClient):
    """Client for real-time inference with trained policies."""

    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        checkpoint_name: str = "last.ckpt",
        model_server_address: str = "127.0.0.1",
        model_server_port: int = 5555,
        temporal_agg: bool = True,
        favor_more_recent: bool = True,
        exponential_decay: float = 0.01,
        update_rate_hz: float | None = None,
        timing_log: bool = False,
        precision: str = PrecisionType.BF16_MIXED.value,
        **kwargs,
    ):
        """Initialize inference client.

        Args:
            device: Device to run inference on
            checkpoint_path: Path to checkpoint directory
            checkpoint_name: Name of the checkpoint file (default: "latest.ckpt")
            model_server_address: Address of the model server controlling the robot
            model_server_port: Port of the model server
            temporal_agg: Whether to use temporal aggregation for actions
            favor_more_recent: Whether to favor more recent actions in temporal aggregation
            exponential_decay: Exponential decay factor for temporal aggregation
            update_rate_hz: Update frequency in Hz (overrides checkpoint config)
            timing_log: Whether to log timing information
            precision: Precision type for model inference
            **kwargs: Additional arguments passed to AbstractModelClient
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_name
        self.device = device
        self.temporal_agg = temporal_agg
        self.favor_more_recent = favor_more_recent
        self.exponential_decay = exponential_decay
        self.tokenizer = None
        self.timing_log = timing_log
        self.precision = precision
        logging.info("Loading policy and config...")
        self._load_model()
        logging.info("Policy and config loaded successfully.")
        self.observation_horizon = self.policy.decoder.observation_horizon
        self.prediction_horizon = self.policy.prediction_horizon
        self.image_height = self.config.task.dataloader.image_height
        self.image_width = self.config.task.dataloader.image_width
        self.action_dim = self.policy.action_space.get_total_action_dim()
        self.use_depth = Cameras.DEPTH.value in self.policy.observation_space.camera_keys

        obs_space = self.policy.observation_space
        action_space = self.policy.action_space
        if update_rate_hz is None:
            update_rate_hz = 10.0
        
        super().__init__(
            model_server_address=model_server_address,
            model_server_port=model_server_port,
            observation_buffer_size=self.observation_horizon,
            request_depth=self.use_depth,
            request_rectified_images=True,
            request_gripper_state=action_space.has_gripper,
            request_language_instruction=obs_space.use_language,
            predicts_in_camera_frame=action_space.predict_in_camera_frame,
            predicts_delta=action_space.deltas_as_actions,
            obs_robot_frame=obs_space.use_proprio_base_frame,
            obs_camera_frame=obs_space.use_proprio_camera_frame,
            device=str(device),
            update_rate_hz=update_rate_hz,
            enable_logging=False,
        )

        # TODO: integrate the use of action processor to compute orientation actions, currently unused
        self.action_processor = ActionProcessor(self.policy.action_space)
        additional_targets = {"right_image": "image"}
        if self.use_depth:
            additional_targets["depth"] = "mask"

        self.image_transform = A.Compose(
            [
                A.Resize(height=self.image_height, width=self.image_width),
                ToTensorV2(),
            ],
            additional_targets=additional_targets,
        )
        self.max_timesteps = 10000
        self.has_position = action_space.has_position
        self.has_orientation = action_space.has_orientation
        self.has_gripper = action_space.has_gripper
        self.position_dim = action_space.position_dim if self.has_position else 0
        self.orientation_dim = action_space.orientation_dim if self.has_orientation else 0
        self.gripper_type = action_space.gripper_type if action_space.has_gripper else None
        self.gripper_dim = action_space.gripper_dim if action_space.has_gripper else 0
        if not self.has_position:
            raise ValueError("InferenceClient currently requires position actions.")

        self.all_time_position_actions = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.prediction_horizon, self.position_dim]
        ).to(self.device)
        self.all_time_populated_mask = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.prediction_horizon], dtype=torch.bool
        ).to(self.device)

        if self.has_orientation:
            # TODO: Extend to other orientation representations in the future.
            if self.orientation_dim != 1 or not self.has_position:
                raise NotImplementedError(
                    "Currently only 1D orientation (roll) with position is supported for the inference policy client."
                )

            self.all_time_orientations = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.prediction_horizon, self.orientation_dim]
            ).to(self.device)
        if self.policy.action_space.has_gripper:
            self.all_time_grippers = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.prediction_horizon, self.gripper_dim]
            ).to(self.device)

        self.timestep = 0
        self.current_all_position_actions = None
        self.current_all_orientations = None
        self.current_all_grippers = None


    def _load_model(self) -> None:
        """Load config and policy from checkpoint."""
        config_path = os.path.join(self.checkpoint_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Config file not found at {config_path}. "
                f"Expected 'config.yaml' in checkpoint directory."
            )
        logging.info(f"Loading config from {config_path}")
        config = hydra.utils.instantiate(OmegaConf.load(config_path))
        self.config: MainConfig = config
        checkpoint_file = os.path.join(self.checkpoint_path, self.checkpoint_name)
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_file}. "
                f"Expected {self.checkpoint_name} in checkpoint directory."
            )
        logging.info(f"Loading model and tokenizer from {checkpoint_file}")
        tokenizer_path = os.path.join(self.checkpoint_path, "tokenizer")
        if os.path.exists(tokenizer_path):
            self.tokenizer = Tokenizer.from_pretrained(tokenizer_path, device=self.device)
            logging.info(f"Tokenizer loaded from {tokenizer_path}")
        else:
            self.tokenizer = None

        self.policy: Policy = self.config.policy
        if self.tokenizer is not None:
            self.tokenizer.to(self.device)
            self.policy.set_tokenizer(self.tokenizer)
            logging.info("Resized policy layers via set_tokenizer (obs/action vocab)")

        self.policy.to(self.device).eval()
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        lightning_module = LightningPolicy(policy=self.policy, training_config=self.config.training)
        lightning_module.load_state_dict(checkpoint['state_dict'], strict=False)

        if Cameras.DEPTH.value in self.policy.observation_space.camera_keys:
            depth_stats = self.policy.normalizer[Cameras.DEPTH.value].params_dict['input_stats']
            self.depth_min = float(depth_stats['min'].item())
            self.depth_max = float(depth_stats['max'].item())
            logging.info(f"Depth clipping range from normalizer: [{self.depth_min:.4f}, {self.depth_max:.4f}]")
        else:
            self.depth_min = None
            self.depth_max = None

        action_space = self.policy.action_space
        if action_space.has_position:
            if hasattr(self.policy, 'position_delta_threshold'):
                self.position_delta_threshold = float(self.policy.position_delta_threshold.item())
                logging.info(f"Position delta denoising threshold: {self.position_delta_threshold:.6f}")
            else:
                self.position_delta_threshold = 0.0
                logging.warning("Policy missing position_delta_threshold, denoising disabled for position")
        else:
            self.position_delta_threshold = 0.0

        if action_space.has_orientation:
            if hasattr(self.policy, 'orientation_delta_threshold'):
                self.orientation_delta_threshold = float(self.policy.orientation_delta_threshold.item())
                logging.info(f"Orientation delta denoising threshold: {self.orientation_delta_threshold:.6f}")
            else:
                self.orientation_delta_threshold = 0.0
                logging.warning("Policy missing orientation_delta_threshold, denoising disabled for orientation")
        else:
            self.orientation_delta_threshold = 0.0

        logging.info("Model and config successfully loaded.")

    def get_actions_from_model(self) -> list[Action]:
        """Compute next actions using the trained policy model.

        Returns:
            List of Action objects
        """
        total_start_time, total_end_time = None, None
        preprocessing_start_time, preprocessing_end_time = None, None
        inference_start_time, inference_end_time = None, None
        postprocessing_start_time, postprocessing_end_time, preprocessing_duration = None, None, None
        depth_processing_start, rgb_processing_start = None, None
        inference_duration, postprocessing_duration = None, None

        if self.timing_log:
            total_start_time = time.time()
            logging.info(f"\n=== TIMESTEP {self.timestep} - Starting get_actions_from_model ===")
            preprocessing_start_time = time.time()
            logging.info(f"[TIMING] Input preprocessing started at: {preprocessing_start_time:.6f}")

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

        if self.timing_log:
            depth_processing_start = time.time()
        depth_imgs = None
        if self.request_depth:
            depth_img_list = self.depth_buffer[-self.observation_buffer_size :]
            transformed = [
                self.image_transform(image=left_np, right_image=right_np, depth=depth_np)
                for left_np, right_np, depth_np in zip(left_img_list, right_img_list, depth_img_list)
            ]
            depth_tensors = [t["depth"] for t in transformed]
            depth_imgs = torch.stack(depth_tensors).unsqueeze(0).unsqueeze(-3)
            if self.depth_min is not None and self.depth_max is not None:
                depth_imgs = torch.clamp(depth_imgs, min=self.depth_min, max=self.depth_max)
        else:
            transformed = [
                self.image_transform(image=left_np, right_image=right_np)
                for left_np, right_np in zip(left_img_list, right_img_list)
            ]

        if self.timing_log:
            logging.info(f"[TIMING] Depth plus RGB transform took: {time.time() - depth_processing_start:.6f} seconds")
            rgb_processing_start = time.time()

        left_tensors = [t["image"] / 255.0 for t in transformed]
        right_tensors = [t["right_image"] / 255.0 for t in transformed]
        left_imgs = torch.stack(left_tensors).unsqueeze(0)
        right_imgs = torch.stack(right_tensors).unsqueeze(0)

        if self.timing_log:
            logging.info(f"[TIMING] RGB processing took: {time.time() - rgb_processing_start:.6f} seconds")

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

        if self.request_language_instruction:
            language_instruction = self.language_instruction_buffer[-self.observation_buffer_size :]
            obs_dict[LANGUAGE_KEY] = language_instruction


        if self.timing_log:
            preprocessing_end_time = time.time()
            preprocessing_duration = preprocessing_end_time - preprocessing_start_time
            logging.info(f"[TIMING] Input preprocessing completed in: {preprocessing_duration:.6f} seconds")

        current_roll = 0.0
        if self.predicts_in_camera_frame and self.obs_camera_frame and self.obs_robot_frame:
            current_robot_position = self.robot_state_buffer[-1][3:6] # Camera frame is the 3:6 dims if both frames are used
            if self.has_orientation:
                current_roll = self.robot_state_buffer[-1][6] if len(self.robot_state_buffer[-1]) > 6 else 0.0
        else:
            current_robot_position = self.robot_state_buffer[-1][:3]
            if self.has_orientation:
                current_roll = self.robot_state_buffer[-1][4] if len(self.robot_state_buffer[-1]) > 4 else 0.0

        if self.timing_log:
            inference_start_time = time.time()
            logging.info(f"[TIMING] Model inference started at: {inference_start_time:.6f}")

        with torch.autocast(device_type=str(self.device), dtype=MAP_PRECISION_TO_DTYPE[self.precision]):
            with torch.no_grad():
                action_dict = self.policy.predict_action(obs_dict=obs_dict)

        if self.has_position:
            self.current_all_position_actions = action_dict[POSITION_ACTION_KEY]
        else:
            self.current_all_position_actions = None
        if self.has_orientation:
            self.current_all_orientations = action_dict[ORIENTATION_ACTION_KEY]
        else:
            self.current_all_orientations = None
        if self.policy.action_space.has_gripper:
            self.current_all_grippers = action_dict[GRIPPER_ACTION_KEY]
        else:
            self.current_all_grippers = None

        if self.timing_log:
            inference_end_time = time.time()
            inference_duration = inference_end_time - inference_start_time
            logging.info(f"[TIMING] Model inference completed in: {inference_duration:.6f} seconds")

            postprocessing_start_time = time.time()
            logging.info(f"[TIMING] Post-processing started at: {postprocessing_start_time:.6f}")


        if self.temporal_agg:
            averaged_actions = self.get_exponential_averaged_actions()
            raw_position = averaged_actions[POSITION_ACTION_KEY]
            raw_orientation = averaged_actions.get(ORIENTATION_ACTION_KEY, None)
            raw_gripper = averaged_actions.get(GRIPPER_ACTION_KEY, None)
            robot_action, gripper_action = self._postprocess_action_tensors(
                raw_position_tensor=raw_position, raw_orientation_tensor=raw_orientation, raw_gripper_tensor=raw_gripper,
                current_robot_position=current_robot_position, current_roll=current_roll
            )
            actions = [Action(robot_action=robot_action, gripper_action=gripper_action)]
        else:
            actions = []
            for i in range(self.prediction_horizon):
                raw_position = self.current_all_position_actions[0, i]
                raw_orientation = self.current_all_orientations[0, i] if self.has_orientation else None
                raw_gripper = self.current_all_grippers[0, i] if self.has_gripper else None
                robot_action, gripper_action = self._postprocess_action_tensors(
                    raw_position_tensor=raw_position, raw_orientation_tensor=raw_orientation, raw_gripper_tensor=raw_gripper,
                    current_robot_position=current_robot_position, current_roll=current_roll
                )
                actions.append(Action(robot_action=robot_action, gripper_action=gripper_action))

        if self.timing_log:
            postprocessing_end_time = time.time()
            postprocessing_duration = postprocessing_end_time - postprocessing_start_time
            logging.info(f"[TIMING] Post-processing completed in: {postprocessing_duration:.6f} seconds")

        self.timestep += 1

        if self.timing_log:
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time

            logging.info(f"\n[TIMING SUMMARY] Timestep {self.timestep - 1}:")
            logging.info(f"  - Preprocessing: {preprocessing_duration:.6f}s ({preprocessing_duration/total_duration*100:.1f}%)")
            logging.info(f"  - Model inference: {inference_duration:.6f}s ({inference_duration/total_duration*100:.1f}%)")
            logging.info(f"  - Post-processing: {postprocessing_duration:.6f}s ({postprocessing_duration/total_duration*100:.1f}%)")
            logging.info(f"  - TOTAL: {total_duration:.6f}s")
            logging.info(f"  - Effective FPS: {1.0/total_duration:.2f}")
            logging.info(f"=== TIMESTEP {self.timestep - 1} COMPLETE ===\n")

        if self.enable_logging:
            logging.log(level=logging.INFO, msg=f"{actions=}")
        print(actions)
        return actions


    def get_exponential_averaged_actions(self) -> dict[str, torch.Tensor]:
        """Average exponentially the actions predicted for the current timestep.

        Returns:
            Exponentially averaged action tensors in a dictionary indexed by action keys (position, orientation, gripper).
        """
        averaged = {}
        self.all_time_position_actions[
            [self.timestep], self.timestep : self.timestep + self.prediction_horizon
        ] = self.current_all_position_actions
        self.all_time_populated_mask[
            [self.timestep], self.timestep : self.timestep + self.prediction_horizon
        ] = True
        # Use mask to filter populated timesteps
        actions_populated = self.all_time_populated_mask[:, self.timestep]
        actions_for_curr_step_pos = self.all_time_position_actions[:, self.timestep][actions_populated]
        indices = np.arange(len(actions_for_curr_step_pos))
        if self.favor_more_recent:
            indices = indices[::-1]  # Newest first
        exp_weights = np.exp(-self.exponential_decay * indices)
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights_t = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
        averaged_pos = (actions_for_curr_step_pos * exp_weights_t).sum(dim=0)
        averaged[POSITION_ACTION_KEY] = averaged_pos

        if self.has_orientation:
            self.all_time_orientations[[self.timestep], self.timestep: self.timestep + self.prediction_horizon] = self.current_all_orientations
            actions_for_curr_step_ori = self.all_time_orientations[:, self.timestep][actions_populated]
            indices = np.arange(len(actions_for_curr_step_ori))
            if self.favor_more_recent:
                indices = indices[::-1]
            exp_weights = np.exp(-self.exponential_decay * indices)
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights_t = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            averaged_ori = (actions_for_curr_step_ori * exp_weights_t).sum(dim=0)
            averaged[ORIENTATION_ACTION_KEY] = averaged_ori

        if self.policy.action_space.has_gripper:
            self.all_time_grippers[
            [self.timestep], self.timestep: self.timestep + self.prediction_horizon
            ] = self.current_all_grippers
            actions_for_curr_step_grip = self.all_time_grippers[:, self.timestep][actions_populated]
            indices = np.arange(len(actions_for_curr_step_grip))
            if self.favor_more_recent:
                indices = indices[::-1]
            exp_weights = np.exp(-self.exponential_decay * indices)
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights_t = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
            averaged_grip = (actions_for_curr_step_grip * exp_weights_t).sum(dim=0)
            averaged[GRIPPER_ACTION_KEY] = averaged_grip

        return averaged


    def _postprocess_action_tensors(
            self,
            raw_position_tensor: torch.Tensor,
            raw_orientation_tensor: torch.Tensor | None,
            raw_gripper_tensor: torch.Tensor | None,
            current_robot_position: np.ndarray,
            current_roll: float,
    ) -> tuple[np.ndarray, np.ndarray | None | bool]:
        """Post-process raw action tensors for one step.

        Converts absolute predictions to deltas, then applies norm-based denoising
        to filter out noisy small-magnitude deltas.
        """
        position_action = raw_position_tensor.cpu().detach().numpy()[:3]
        if not self.predicts_delta:
            position_action = position_action - current_robot_position

        if self.position_delta_threshold > 0:
            position_norm = np.linalg.norm(position_action)
            if position_norm < self.position_delta_threshold:
                position_action = np.zeros_like(position_action)

        if raw_gripper_tensor is not None:
            raw_gripper_action = raw_gripper_tensor.cpu().detach().numpy()
            if self.gripper_type == GripperType.BINARY.value:
                gripper_action = raw_gripper_action > 0.5
            else:
                gripper_action = raw_gripper_action
        else:
            gripper_action = None

        if self.has_orientation:
            assert raw_orientation_tensor is not None
            orientation_action = raw_orientation_tensor.cpu().detach().numpy()
            # TODO: Here we only handle 3D position + roll (1D orientation). Extend for other representations in the future.
            assert self.orientation_dim == 1
            assert self.has_position
            if not self.predicts_delta:
                orientation_action = orientation_action[0] - current_roll

            if self.orientation_delta_threshold > 0:
                orientation_magnitude = np.abs(orientation_action)
                if orientation_magnitude < self.orientation_delta_threshold:
                    orientation_action = 0.0

            robot_action = np.concatenate((position_action, [orientation_action[0] if isinstance(orientation_action, np.ndarray) else orientation_action]))
        else:
            robot_action = np.concatenate((position_action, [0.0]))  # Roll = 0.0 if no orientation predicted

        return robot_action, gripper_action

