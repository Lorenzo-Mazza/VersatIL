"""Inference client for real-time model deployment on the TSO robot testbed.

This module provides the TSOPolicyClient class that interfaces with the
imitation_learning_toolkit's AbstractModelClient for real-time robot control.

Note:
    The TSO Policy Server uses as convention delta actions for position and orientation.
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
from refactoring.data.constants import (
    ActionComputationMethod,
    BinaryGripperRange,
    Cameras,
    CoordinateSystem,
    GripperType,
    ObsKey,
    ProprioKey,
)
from refactoring.data.metadata import OnTheFlyActionMetadata
from refactoring.data.task import ActionSpace, ObservationSpace
from refactoring.data.tokenization.tokenizer import Tokenizer
from refactoring.training.lightning_policy import LightningPolicy


class TSOPolicyClient(AbstractModelClient):
    """Client for real-time inference with trained policies on the TSO robot testbed."""

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
        obs_space: ObservationSpace = self.policy.observation_space
        action_space: ActionSpace = self.policy.action_space
        self.action_dim = action_space.get_total_action_dim()
        self._setup_position_action(action_space)
        self._setup_orientation_action(action_space)
        self._setup_gripper_action(action_space)
        self._setup_observations(obs_space)
        self._setup_denoising_thresholds()
        if update_rate_hz is None:
            update_rate_hz = 10.0

        super().__init__(
            model_server_address=model_server_address,
            model_server_port=model_server_port,
            observation_buffer_size=self.observation_horizon,
            request_depth=self.use_depth,
            request_rectified_images=True,
            request_gripper_state=self.has_gripper,
            request_language_instruction=self.use_language,
            predicts_in_camera_frame=(self.position_frame == CoordinateSystem.CAMERA.value),
            predicts_delta=self.predicts_delta,
            obs_robot_frame=self.use_proprio_robot_frame,
            obs_camera_frame=self.use_proprio_camera_frame,
            device=str(device),
            update_rate_hz=update_rate_hz,
            enable_logging=False,
        )
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

        if self.has_orientation and self.orientation_dim != 1:
            raise NotImplementedError(
                "Only 1D orientation (roll) is currently supported for TSO InferenceClient"
            )
        self.all_time_position_actions = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.prediction_horizon, self.position_dim]
        ).to(self.device)
        self.all_time_populated_mask = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.prediction_horizon], dtype=torch.bool
        ).to(self.device)

        if self.has_orientation:
            self.all_time_orientations = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.prediction_horizon, self.orientation_dim]
            ).to(self.device)

        if self.has_gripper:
            self.all_time_grippers = torch.zeros(
                [self.max_timesteps, self.max_timesteps + self.prediction_horizon, self.gripper_dim]
            ).to(self.device)

        self.timestep = 0
        self.current_all_position_actions = None
        self.current_all_orientations = None
        self.current_all_grippers = None


    def _setup_position_action(self, action_space: ActionSpace) -> None:
        """Setup position action key and metadata from ActionSpace."""
        position_camera_key = ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value
        position_robot_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        if position_camera_key in action_space.actions_metadata:
            self.position_key = position_camera_key
        elif position_robot_key in action_space.actions_metadata:
            self.position_key = position_robot_key
        else:
            raise ValueError(
                "TSO InferenceClient requires position actions. "
                f"Expected key '{ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value}' or "
                f"'{ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value}' in action_space.actions_metadata."
                f" Got keys: {list(action_space.actions_metadata.keys())}"
            )
        self.has_position = True
        pos_meta = action_space.actions_metadata[self.position_key]
        if isinstance(pos_meta, OnTheFlyActionMetadata):
            self.predicts_delta = pos_meta.computation_method == ActionComputationMethod.DELTA.value
            self.position_frame = pos_meta.source_metadata.frame
            self.position_dim = pos_meta.prediction_dimension
        else:
            raise ValueError("TSO InferenceClient only supports OnTheFlyActionMetadata for position actions.")


    def _setup_orientation_action(self, action_space: ActionSpace) -> None:
        """Setup orientation action key and metadata from ActionSpace."""
        orientation_camera_key = ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_ORI.value
        orientation_robot_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_ORI.value
        if orientation_camera_key in action_space.actions_metadata:
            self.orientation_key = orientation_camera_key
            self.has_orientation = True
        elif orientation_robot_key in action_space.actions_metadata:
            self.orientation_key = orientation_robot_key
            self.has_orientation = True
        else:
            self.orientation_key = None
            self.has_orientation = False
            self.orientation_dim = 0
            self.orientation_frame = None
            self.orientation_representation = None
            return
        ori_meta = action_space.actions_metadata[self.orientation_key]
        if isinstance(ori_meta, OnTheFlyActionMetadata):
            self.orientation_representation = ori_meta.source_metadata.orientation_representation
            self.orientation_frame = ori_meta.source_metadata.frame
        else:
            self.orientation_representation = ori_meta.orientation_representation
            self.orientation_frame = ori_meta.frame
        self.orientation_dim = ori_meta.prediction_dimension


    def _setup_gripper_action(self, action_space: ActionSpace) -> None:
        """Setup gripper action key and metadata from ActionSpace."""
        gripper_key = ProprioKey.GRIPPER_STATE.value
        if gripper_key in action_space.actions_metadata:
            self.gripper_key = gripper_key
            self.has_gripper = True
            gripper_meta = action_space.actions_metadata[gripper_key]
            if isinstance(gripper_meta, OnTheFlyActionMetadata):
                self.gripper_type = gripper_meta.source_metadata.gripper_type
                self.binary_gripper_range = gripper_meta.source_metadata.binary_gripper_range
            else:
                self.gripper_type = gripper_meta.gripper_type
                self.binary_gripper_range = gripper_meta.binary_gripper_range
            self.gripper_dim = gripper_meta.prediction_dimension
        else:
            self.gripper_key = None
            self.has_gripper = False
            self.gripper_type = None
            self.binary_gripper_range = None
            self.gripper_dim = 0
        if self.gripper_type == GripperType.BINARY.value and self.binary_gripper_range is None:
            logging.warning("Gripper binary range is not set. Assuming {0,1}.")
            self.binary_gripper_range = BinaryGripperRange.ZERO_ONE.value


    def _setup_observations(self, obs_space: ObservationSpace) -> None:
        """Setup observation keys from ObservationSpace metadata."""
        position_camera_key = ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value
        position_robot_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        self.use_depth = Cameras.DEPTH.value in obs_space.cameras
        self.use_language = ObsKey.LANGUAGE.value in obs_space.observations_metadata
        self.use_proprio_camera_frame = position_camera_key in obs_space.observations_metadata
        self.use_proprio_robot_frame = position_robot_key in obs_space.observations_metadata


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
        self._validate_checkpoint_loading(checkpoint['state_dict'], lightning_module)

        if Cameras.DEPTH.value in self.policy.observation_space.cameras:
            depth_stats = self.policy.normalizer[Cameras.DEPTH.value].params_dict['input_stats']
            self.depth_min = float(depth_stats['min'].item())
            self.depth_max = float(depth_stats['max'].item())
            logging.info(f"Depth clipping range from normalizer: [{self.depth_min:.4f}, {self.depth_max:.4f}]")
        else:
            self.depth_min = None
            self.depth_max = None
        logging.info("Model and config successfully loaded.")


    def _validate_checkpoint_loading(
        self,
        checkpoint_state_dict: dict[str, torch.Tensor],
        lightning_module: LightningPolicy,
    ) -> None:
        """Validate that critical checkpoint components were properly loaded.

        This catches issues with lazy-initialized modules where checkpoint weights
        might be silently ignored if `strict=False` and the module's internal
        dictionaries are empty at load time.

        Raises:
            RuntimeError: If critical components failed to load from checkpoint.
        """
        model_state = lightning_module.state_dict()
        checkpoint_keys = set(checkpoint_state_dict.keys())
        model_keys = set(model_state.keys())
        critical_prefixes = [
            'policy.decoder.',
            'policy.encoding_pipeline.',
            'policy.normalizer.',
        ]
        errors = []
        warnings = []
        for prefix in critical_prefixes:
            ckpt_count = len([k for k in checkpoint_keys if k.startswith(prefix)])
            model_count = len([k for k in model_keys if k.startswith(prefix)])
            if ckpt_count > 0 and model_count == 0:
                errors.append(
                    f"CRITICAL: Checkpoint has {ckpt_count} keys for '{prefix}' but model has NONE! "
                    f"This indicates lazy-initialized layers failed to load."
                )
            elif ckpt_count > 0 and model_count < ckpt_count:
                matched = len([k for k in checkpoint_keys if k.startswith(prefix) and k in model_keys])
                if matched < ckpt_count:
                    warnings.append(
                        f"WARNING: Checkpoint has {ckpt_count} keys for '{prefix}' but model only has {model_count}. "
                        f"Matched: {matched}/{ckpt_count}"
                    )
        lazy_module_prefixes = [
            ('policy.decoder.architecture.feature_projection.linear_projections.', 'FeatureProjection linear'),
            ('policy.decoder.architecture.feature_projection.spatial_projections.', 'FeatureProjection spatial'),
            ('policy.decoder.architecture.camera_embeddings.embeddings.', 'DynamicFeatureEmbedding'),
        ]
        for ckpt_prefix, module_name in lazy_module_prefixes:
            ckpt_keys_for_module = [k for k in checkpoint_keys if k.startswith(ckpt_prefix)]
            model_keys_for_module = [k for k in model_keys if k.startswith(ckpt_prefix)]

            if len(ckpt_keys_for_module) > 0 and len(model_keys_for_module) == 0:
                errors.append(
                    f"CRITICAL: {module_name} failed to load! "
                    f"Checkpoint has {len(ckpt_keys_for_module)} keys but model has NONE. "
                    f"Example keys: {ckpt_keys_for_module[:3]}"
                )
        sample_keys = [k for k in checkpoint_keys if k in model_keys][:5]
        for key in sample_keys:
            ckpt_val = checkpoint_state_dict[key]
            model_val = model_state[key]
            if not torch.allclose(ckpt_val.to(model_val.device), model_val, atol=1e-6):
                errors.append(
                    f"CRITICAL: Weight mismatch for '{key}'! "
                    f"Checkpoint and model values differ after load_state_dict."
                )
        for warning in warnings:
            logging.warning(warning)
        if errors:
            for error in errors:
                logging.error(error)
            raise RuntimeError(
                f"Checkpoint loading validation failed with {len(errors)} critical error(s). "
                f"The model will NOT produce correct outputs. "
                f"First error: {errors[0]}"
            )


    def _setup_denoising_thresholds(self) -> None:
        """Setup denoising thresholds from policy.denoising_thresholds (DictOfTensorMixin)."""
        denoising_thresholds = self.policy.denoising_thresholds.params_dict
        if self.position_key in denoising_thresholds:
            self.position_delta_threshold = float(denoising_thresholds[self.position_key].item())
            logging.info(f"Position delta denoising threshold [{self.position_key}]: {self.position_delta_threshold:.6f}")
        else:
            self.position_delta_threshold = 0.0
            logging.info("No position denoising threshold found, denoising disabled for position")

        if self.orientation_key and self.orientation_key in denoising_thresholds:
            self.orientation_delta_threshold = float(denoising_thresholds[self.orientation_key].item())
            logging.info(f"Orientation delta denoising threshold [{self.orientation_key}]: {self.orientation_delta_threshold:.6f}")
        else:
            self.orientation_delta_threshold = 0.0
            if self.has_orientation:
                logging.info("No orientation denoising threshold found, denoising disabled for orientation")


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
            position_robot_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
            position_camera_key = ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value
            if self.obs_robot_frame and self.obs_camera_frame:
                obs_dict[position_robot_key] = qpos_tensor[:, :, :3]
                obs_dict[position_camera_key] = qpos_tensor[:, :, 3:]
            elif self.obs_robot_frame:
                obs_dict[position_robot_key] = qpos_tensor
            elif self.obs_camera_frame:
                obs_dict[position_camera_key] = qpos_tensor

        if self.request_depth:
            obs_dict[Cameras.DEPTH.value] = depth_imgs

        if self.request_language_instruction:
            language_instruction = self.language_instruction_buffer[-self.observation_buffer_size :]
            obs_dict[ObsKey.LANGUAGE.value] = language_instruction


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
            self.current_all_position_actions = action_dict[self.position_key]
        else:
            self.current_all_position_actions = None
        if self.has_orientation:
            self.current_all_orientations = action_dict[self.orientation_key]
        else:
            self.current_all_orientations = None
        if self.has_gripper:
            self.current_all_grippers = action_dict[self.gripper_key]
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
            raw_position = averaged_actions[self.position_key]
            raw_orientation = averaged_actions.get(self.orientation_key, None) if self.has_orientation else None
            raw_gripper = averaged_actions.get(self.gripper_key, None) if self.has_gripper else None
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
        ] = self.current_all_position_actions.float()
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
        exp_weights_t = torch.from_numpy(exp_weights).to(self.device).float().unsqueeze(dim=1)
        averaged_pos = (actions_for_curr_step_pos * exp_weights_t).sum(dim=0)
        averaged[self.position_key] = averaged_pos

        if self.has_orientation:
            self.all_time_orientations[[self.timestep], self.timestep: self.timestep + self.prediction_horizon
            ] = self.current_all_orientations.float()
            actions_for_curr_step_ori = self.all_time_orientations[:, self.timestep][actions_populated]
            indices = np.arange(len(actions_for_curr_step_ori))
            if self.favor_more_recent:
                indices = indices[::-1]
            exp_weights = np.exp(-self.exponential_decay * indices)
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights_t = torch.from_numpy(exp_weights).to(self.device).float().unsqueeze(dim=1)
            averaged_ori = (actions_for_curr_step_ori * exp_weights_t).sum(dim=0)
            averaged[self.orientation_key] = averaged_ori

        if self.has_gripper:
            self.all_time_grippers[
            [self.timestep], self.timestep: self.timestep + self.prediction_horizon
            ] = self.current_all_grippers.float()
            actions_for_curr_step_grip = self.all_time_grippers[:, self.timestep][actions_populated]
            indices = np.arange(len(actions_for_curr_step_grip))
            if self.favor_more_recent:
                indices = indices[::-1]
            exp_weights = np.exp(-self.exponential_decay * indices)
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights_t = torch.from_numpy(exp_weights).to(self.device).float().unsqueeze(dim=1)
            averaged_grip = (actions_for_curr_step_grip * exp_weights_t).sum(dim=0)
            averaged[self.gripper_key] = averaged_grip

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
        position_action = raw_position_tensor.cpu().detach().float().numpy()[:3]
        if not self.predicts_delta:
            position_action = position_action - current_robot_position

        if self.position_delta_threshold > 0:
            position_norm = np.linalg.norm(position_action)
            if position_norm < self.position_delta_threshold:
                position_action = np.zeros_like(position_action)

        if raw_gripper_tensor is not None:
            raw_gripper_action = raw_gripper_tensor.cpu().detach().float().numpy()
            if self.gripper_type == GripperType.BINARY.value:
                if self.binary_gripper_range == BinaryGripperRange.ZERO_ONE.value:
                    gripper_action = raw_gripper_action > 0.5  # Model outputs [0, 1]
                else: 
                    gripper_action = raw_gripper_action > 0.0  # Model outputs [-1, 1]
            else:
                gripper_action = raw_gripper_action
        else:
            gripper_action = None

        if self.has_orientation:
            assert raw_orientation_tensor is not None
            orientation_action = raw_orientation_tensor.cpu().detach().float().numpy()
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

