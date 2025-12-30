"""Inference client for LIBERO simulation environments.

This module provides the LiberoClient class for running trained policies
in the LIBERO simulation benchmark via ZMQ communication.

Note:
    LIBERO uses precomputed delta actions in robot base frame.
    Actions are 7D: position delta (3) + orientation delta euler (3) + gripper (1).
    Gripper values are in range [-1, 1] where -1 is open and 1 is closed.
"""
import logging
import os
import time
from dataclasses import dataclass
import enum 
import albumentations as A
import hydra
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from imitation_learning_toolkit.sockets.client import SocketClient
from imitation_learning_toolkit.sockets.compression import CompressionType, decompress_array
from omegaconf import OmegaConf

from refactoring.configs import MainConfig
from refactoring.data.constants import (
    BinaryGripperRange,
    Cameras,
    GripperType,
    ObsKey,
    ProprioKey,
)
from refactoring.data.metadata import (
    GripperActionMetadata,
    OrientationActionMetadata,
    PositionActionMetadata,
)
from refactoring.data.task import ActionSpace, ObservationSpace
from refactoring.data.tokenization.tokenizer import Tokenizer
from refactoring.models.policy import Policy
from refactoring.training.constants import MAP_PRECISION_TO_DTYPE, PrecisionType
from refactoring.training.lightning_policy import LightningPolicy

logging.basicConfig(level=logging.INFO)


class LiberoRequestKeys(str, enum.Enum):
    """JSON keys for client requests to LIBERO server."""
    ROUTE_NAME = "route_name"
    ROBOT_ACTION = "robot_action"
    REQUEST_AGENTVIEW = "request_agentview_rgb"
    REQUEST_EYE_IN_HAND = "request_eye_in_hand_rgb"
    REQUEST_EE_POS = "request_ee_pos"
    REQUEST_EE_ORI = "request_ee_ori"
    REQUEST_GRIPPER_STATES = "request_gripper_states"
    REQUEST_LANGUAGE_INSTRUCTION = "request_language_instruction"
    COMPRESSION_TYPE = "compression_type"


class LiberoResponseKeys(str, enum.Enum):
    """JSON keys for server responses from LIBERO server."""
    STATUS = "status"
    AGENTVIEW_RGB = "agentview_rgb"
    EYE_IN_HAND_RGB = "eye_in_hand_rgb"
    EE_POS = "ee_pos"
    EE_ORI = "ee_ori"
    GRIPPER_STATES = "gripper_states"
    LANGUAGE_INSTRUCTION = "language_instruction"
    DONE = "done"
    SUCCESS = "success"


class LiberoRoutes(str, enum.Enum):
    """Route names for LIBERO server."""
    GET_OBSERVATION = "get_observation"
    SEND_ACTION = "send_action"


class LiberoStatus(str, enum.Enum):
    """Status values from LIBERO server."""
    FINISHED = "FINISHED"
    ERROR = "ERROR"


@dataclass
class LiberoObservation:
    """Data class for LIBERO observation data from the server."""
    agentview_rgb: np.ndarray
    eye_in_hand_rgb: np.ndarray
    ee_pos: np.ndarray | None
    ee_ori: np.ndarray | None
    gripper_states: np.ndarray | None
    language_instruction: str | None


@dataclass
class LiberoAction:
    """Data class for LIBERO action data.

    Attributes:
        robot_action: 7D action array [pos_delta(3), ori_delta(3), gripper(1)]
        gripper_action: Gripper action as boolean for open/close
    """
    robot_action: np.ndarray
    gripper_action: bool | None


class LiberoClient(SocketClient):
    """Client for running inference with trained policies in LIBERO simulation.

    Uses ZMQ for communication with LIBERO server. Has LIBERO-specific
    buffer names (agentview_buffer, eye_in_hand_buffer) and action format.
    """

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
        update_rate_hz: float = 10.0,
        enable_logging: bool = False,
        compression_type: str = CompressionType.RAW.value,
        precision: str = PrecisionType.BF16_MIXED.value,
    ):
        """Initialize LIBERO inference client.

        Args:
            device: Device to run inference on
            checkpoint_path: Path to checkpoint directory
            checkpoint_name: Name of the checkpoint file
            model_server_address: Address of the LIBERO model server
            model_server_port: Port of the LIBERO model server
            temporal_agg: Whether to use temporal aggregation for actions
            favor_more_recent: Whether to favor more recent actions in temporal aggregation
            exponential_decay: Exponential decay factor for temporal aggregation
            update_rate_hz: Update frequency in Hz
            enable_logging: Enable logging of server responses and actions
            compression_type: Compression type for image data
            precision: Precision type for model inference
        """
        super().__init__(server_address=model_server_address, server_port=model_server_port)

        self.checkpoint_path = checkpoint_path
        self.checkpoint_name = checkpoint_name
        self.device = device
        self.temporal_agg = temporal_agg
        self.favor_more_recent = favor_more_recent
        self.exponential_decay = exponential_decay
        self.update_rate_hz = update_rate_hz
        self.enable_logging = enable_logging
        self.compression_type = compression_type
        self.tokenizer = None
        self.precision = precision

        if self.enable_logging:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

        logging.info("Loading policy and config...")
        self._load_model()
        logging.info("Policy and config loaded successfully.")

        self.observation_horizon = self.policy.decoder.observation_horizon
        self.prediction_horizon = self.policy.prediction_horizon
        self.image_height = self.config.task.dataloader.image_height
        self.image_width = self.config.task.dataloader.image_width
        self.observation_buffer_size = self.observation_horizon

        obs_space: ObservationSpace = self.policy.observation_space
        action_space: ActionSpace = self.policy.action_space
        self.action_dim = action_space.get_total_action_dim()

        self._setup_position_action(action_space)
        self._setup_orientation_action(action_space)
        self._setup_gripper_action(action_space)
        self._setup_observations(obs_space)
        self.agentview_buffer: list[np.ndarray] = []
        self.eye_in_hand_buffer: list[np.ndarray] = []
        self.ee_pos_buffer: list[np.ndarray] = []
        self.ee_ori_buffer: list[np.ndarray] = []
        self.gripper_states_buffer: list[np.ndarray] = []
        self.language_instruction_buffer: list[str] = []
        self.buffer_data_counter = 0
        additional_targets = {Cameras.EYE_IN_HAND.value: "image"}
        self.image_transform = A.Compose(
            [
                A.Resize(height=self.image_height, width=self.image_width),
                ToTensorV2(),
            ],
            additional_targets=additional_targets,
        )
        self.max_timesteps = 10000
        self.all_time_position_actions = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.prediction_horizon, self.position_dim]
        ).to(self.device)
        self.all_time_populated_mask = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.prediction_horizon], dtype=torch.bool
        ).to(self.device)
        self.all_time_orientations = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.prediction_horizon, self.orientation_dim]
        ).to(self.device)
        self.all_time_grippers = torch.zeros(
            [self.max_timesteps, self.max_timesteps + self.prediction_horizon, self.gripper_dim]
        ).to(self.device)
        self.timestep = 0
        self.current_all_position_actions = None
        self.current_all_orientations = None
        self.current_all_grippers = None


    def _load_model(self) -> Policy:
        """Load config and policy from checkpoint."""
        config_path = os.path.join(self.checkpoint_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}.")
        logging.info(f"Loading config from {config_path}")
        config = hydra.utils.instantiate(OmegaConf.load(config_path))
        self.config: MainConfig = config
        checkpoint_file = os.path.join(self.checkpoint_path, self.checkpoint_name)
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_file}.")
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

        self.policy.to(self.device).eval()
        checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        lightning_module = LightningPolicy(policy=self.policy, training_config=self.config.training)

        # Debug: Check for key mismatches during loading
        model_keys = set(lightning_module.state_dict().keys())
        checkpoint_keys = set(checkpoint['state_dict'].keys())
        missing_keys = model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_keys
        if missing_keys:
            logging.warning(f"Missing keys in checkpoint: {list(missing_keys)[:10]}... (total: {len(missing_keys)})")
        if unexpected_keys:
            logging.warning(f"Unexpected keys in checkpoint: {list(unexpected_keys)[:10]}... (total: {len(unexpected_keys)})")

        lightning_module.load_state_dict(checkpoint['state_dict'], strict=False)
        logging.info("Model and config successfully loaded.")
        return self.policy


    def _setup_position_action(self, action_space: ActionSpace) -> None:
        """Setup position action - LIBERO requires 3D position."""
        position_key = ProprioKey.EE_POS_ACTION.value
        if position_key not in action_space.actions_metadata:
            raise ValueError(f"LiberoClient requires position actions with key '{position_key}'.")
        self.position_key = position_key
        pos_meta = action_space.actions_metadata[position_key]
        if not isinstance(pos_meta, PositionActionMetadata):
            raise ValueError(f"Expected PositionActionMetadata for {position_key}")
        self.position_dim = pos_meta.prediction_dimension
        self.position_frame = pos_meta.frame
        if self.position_dim != 3:
            raise ValueError(f"LIBERO requires 3D position actions, got {self.position_dim}D")


    def _setup_orientation_action(self, action_space: ActionSpace) -> None:
        """Setup orientation action - LIBERO requires 3D euler orientation."""
        orientation_key = ProprioKey.EE_ORI_ACTION.value
        if orientation_key not in action_space.actions_metadata:
            raise ValueError(f"LiberoClient requires orientation actions with key '{orientation_key}'.")
        self.orientation_key = orientation_key
        ori_meta = action_space.actions_metadata[orientation_key]
        if not isinstance(ori_meta, OrientationActionMetadata):
            raise ValueError(f"Expected OrientationActionMetadata for {orientation_key}")
        self.orientation_dim = ori_meta.prediction_dimension
        self.orientation_frame = ori_meta.frame
        self.orientation_representation = ori_meta.orientation_representation
        if self.orientation_dim != 3:
            raise ValueError(f"LIBERO requires 3D orientation actions (euler), got {self.orientation_dim}D")


    def _setup_gripper_action(self, action_space: ActionSpace) -> None:
        """Setup gripper action - LIBERO requires 1D gripper."""
        gripper_key = ProprioKey.GRIPPER_STATE_ACTION.value
        if gripper_key not in action_space.actions_metadata:
            raise ValueError(f"LiberoClient requires gripper actions with key '{gripper_key}'.")
        self.gripper_key = gripper_key
        gripper_meta = action_space.actions_metadata[gripper_key]
        if not isinstance(gripper_meta, GripperActionMetadata):
            raise ValueError(f"Expected GripperActionMetadata for {gripper_key}")
        self.gripper_dim = gripper_meta.prediction_dimension
        self.gripper_type = gripper_meta.gripper_type
        self.binary_gripper_range = gripper_meta.binary_gripper_range
        if self.gripper_dim != 1:
            raise ValueError(f"LIBERO requires 1D gripper actions, got {self.gripper_dim}D")


    def _setup_observations(self, obs_space: ObservationSpace) -> None:
        """Setup observation keys from ObservationSpace metadata."""
        self.use_agentview = Cameras.AGENTVIEW.value in obs_space.cameras
        self.use_eye_in_hand = Cameras.EYE_IN_HAND.value in obs_space.cameras
        self.use_language = ObsKey.LANGUAGE.value in obs_space.observations_metadata
        if not self.use_agentview and not self.use_eye_in_hand:
            raise ValueError("LiberoClient requires at least one camera (agentview_rgb or eye_in_hand_rgb).")


    def get_observation(self) -> tuple[LiberoObservation, bool, bool]:
        """Request and process an observation from the LIBERO server.

        Returns:
            Tuple of (observation, done, success)
        """
        response = self.send_request(route_name=LiberoRoutes.GET_OBSERVATION.value, dict_data={
            LiberoRequestKeys.REQUEST_AGENTVIEW.value: self.use_agentview,
            LiberoRequestKeys.REQUEST_EYE_IN_HAND.value: self.use_eye_in_hand,
            LiberoRequestKeys.REQUEST_EE_POS.value: True,
            LiberoRequestKeys.REQUEST_EE_ORI.value: True,
            LiberoRequestKeys.REQUEST_GRIPPER_STATES.value: True,
            LiberoRequestKeys.REQUEST_LANGUAGE_INSTRUCTION.value: self.use_language,
            LiberoRequestKeys.COMPRESSION_TYPE.value: self.compression_type,
        })
        if self.enable_logging:
            # Log useful info only, not base64 images
            logging.info(f"Obs received - ee_pos: {response.get('ee_pos')}, ee_ori: {response.get('ee_ori')}, gripper: {response.get('gripper_states')}, language: {response.get('language_instruction')}")
        if LiberoResponseKeys.STATUS.value not in response:
            raise RuntimeError("Server response missing 'status' key")
        if response[LiberoResponseKeys.STATUS.value] != LiberoStatus.FINISHED.value:
            raise RuntimeError(f"Unexpected server status: {response[LiberoResponseKeys.STATUS.value]}")

        agentview_rgb = None
        if LiberoResponseKeys.AGENTVIEW_RGB.value in response:
            agentview_rgb = decompress_array(response[LiberoResponseKeys.AGENTVIEW_RGB.value], self.compression_type)
            if self.enable_logging:
                logging.info(f"Agentview shape: {agentview_rgb.shape}, dtype: {agentview_rgb.dtype}, range: [{agentview_rgb.min()}, {agentview_rgb.max()}]")
        eye_in_hand_rgb = None
        if LiberoResponseKeys.EYE_IN_HAND_RGB.value in response:
            eye_in_hand_rgb = decompress_array(response[LiberoResponseKeys.EYE_IN_HAND_RGB.value], self.compression_type)
            if self.enable_logging:
                logging.info(f"Eye-in-hand shape: {eye_in_hand_rgb.shape}, dtype: {eye_in_hand_rgb.dtype}, range: [{eye_in_hand_rgb.min()}, {eye_in_hand_rgb.max()}]")
        ee_pos = None
        if LiberoResponseKeys.EE_POS.value in response:
            ee_pos = np.array(response[LiberoResponseKeys.EE_POS.value], dtype=np.float32)
        ee_ori = None
        if LiberoResponseKeys.EE_ORI.value in response:
            ee_ori = np.array(response[LiberoResponseKeys.EE_ORI.value], dtype=np.float32)
        gripper_states = None
        if LiberoResponseKeys.GRIPPER_STATES.value in response:
            gripper_states = np.array(response[LiberoResponseKeys.GRIPPER_STATES.value], dtype=np.float32)
        language_instruction = response.get(LiberoResponseKeys.LANGUAGE_INSTRUCTION.value, None) if self.use_language else None
        done = response.get(LiberoResponseKeys.DONE.value, False)
        success = response.get(LiberoResponseKeys.SUCCESS.value, False)
        obs = LiberoObservation(
            agentview_rgb=agentview_rgb,
            eye_in_hand_rgb=eye_in_hand_rgb,
            ee_pos=ee_pos,
            ee_ori=ee_ori,
            gripper_states=gripper_states,
            language_instruction=language_instruction,
        )
        return obs, done, success


    def send_action(self, robot_action: np.ndarray) -> tuple[bool, bool]:
        """Send an action to the LIBERO server.

        Returns:
            Tuple of (done, success)
        """
        response = self.send_request(route_name=LiberoRoutes.SEND_ACTION.value, dict_data={
            LiberoRequestKeys.ROBOT_ACTION.value: robot_action.tolist(),
        })
        done = response.get(LiberoResponseKeys.DONE.value, False)
        success = response.get(LiberoResponseKeys.SUCCESS.value, False)
        return done, success


    def get_actions_from_model(self) -> list[LiberoAction]:
        """Compute next actions using the trained policy model."""
        agentview_list = self.agentview_buffer[-self.observation_buffer_size:]
        eye_in_hand_list = self.eye_in_hand_buffer[-self.observation_buffer_size:]

        if self.enable_logging and len(agentview_list) > 0:
            import cv2
            cv2.imwrite("/mnt/cluster/workspaces/mazzalore/debug_agentview.png", cv2.cvtColor(agentview_list[-1], cv2.COLOR_RGB2BGR))
            if len(eye_in_hand_list) > 0 and eye_in_hand_list[-1] is not None:
                cv2.imwrite("/mnt/cluster/workspaces/mazzalore/debug_eye_in_hand.png", cv2.cvtColor(eye_in_hand_list[-1], cv2.COLOR_RGB2BGR))

        agentview_tensors = []
        eye_in_hand_tensors = []
        for agentview, eye_in_hand in zip(agentview_list, eye_in_hand_list):
            if self.use_agentview and self.use_eye_in_hand:
                transformed = self.image_transform(
                    image=agentview,
                    **{Cameras.EYE_IN_HAND.value: eye_in_hand}
                )
                agentview_tensors.append(transformed['image'] / 255.0)
                eye_in_hand_tensors.append(transformed[Cameras.EYE_IN_HAND.value] / 255.0)
            elif self.use_agentview:
                transformed = self.image_transform(image=agentview)
                agentview_tensors.append(transformed['image'] / 255.0)
            elif self.use_eye_in_hand:
                transformed = self.image_transform(image=eye_in_hand)
                eye_in_hand_tensors.append(transformed['image'] / 255.0)
        obs_dict = {}
        if self.use_agentview:
            obs_dict[Cameras.AGENTVIEW.value] = torch.stack(agentview_tensors).unsqueeze(0)
        if self.use_eye_in_hand:
            obs_dict[Cameras.EYE_IN_HAND.value] = torch.stack(eye_in_hand_tensors).unsqueeze(0)
        if self.use_language:
            lang_list = self.language_instruction_buffer[-self.observation_buffer_size:]
            obs_dict[ObsKey.LANGUAGE.value] = [[s] for s in lang_list]

        if self.enable_logging and self.timestep == 0:
            # Debug: Check image tensor stats before normalization
            if self.use_agentview:
                img = obs_dict[Cameras.AGENTVIEW.value]
                logging.info(f"[DEBUG] agentview BEFORE norm: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")

        with torch.autocast(device_type=str(self.device), dtype=MAP_PRECISION_TO_DTYPE[self.precision]):
            with torch.no_grad():
                action_dict = self.policy.predict_action(obs_dict=obs_dict)

        self.current_all_position_actions = action_dict[self.position_key]
        self.current_all_orientations = action_dict[self.orientation_key]
        self.current_all_grippers = action_dict[self.gripper_key]

        if self.enable_logging:
            pos_val = self.current_all_position_actions[0, 0].float().cpu().numpy()
            ori_val = self.current_all_orientations[0, 0].float().cpu().numpy()
            grip_val = self.current_all_grippers[0, 0].float().cpu().numpy()
            logging.info(f"[UNNORMALIZED] pos: {pos_val}, ori: {ori_val}, grip: {grip_val}")
            # Log normalizer stats for debugging
            if self.timestep == 0:
                pos_norm = self.policy.normalizer[self.position_key]
                ori_norm = self.policy.normalizer[self.orientation_key]
                logging.info(f"[NORMALIZER] pos scale: {pos_norm.params_dict['scale'].cpu().numpy()}, "
                            f"offset: {pos_norm.params_dict['offset'].cpu().numpy()}")
                logging.info(f"[NORMALIZER] ori scale: {ori_norm.params_dict['scale'].cpu().numpy()}, "
                            f"offset: {ori_norm.params_dict['offset'].cpu().numpy()}")

        if self.temporal_agg:
            averaged_actions = self._get_exponential_averaged_actions()
            robot_action, gripper_action = self._construct_libero_action(averaged_actions)
            actions = [LiberoAction(robot_action=robot_action, gripper_action=gripper_action)]
        else:
            actions = []
            for i in range(self.prediction_horizon):
                robot_action, gripper_action = self._construct_libero_action_from_tensors(
                    self.current_all_position_actions[0, i],
                    self.current_all_orientations[0, i],
                    self.current_all_grippers[0, i],
                )
                actions.append(LiberoAction(robot_action=robot_action, gripper_action=gripper_action))

        self.timestep += 1
        return actions


    def _get_exponential_averaged_actions(self) -> dict[str, torch.Tensor]:
        """Average actions exponentially for temporal aggregation."""
        averaged = {}
        self.all_time_position_actions[
            [self.timestep], self.timestep: self.timestep + self.prediction_horizon
        ] = self.current_all_position_actions.float()
        self.all_time_populated_mask[
            [self.timestep], self.timestep: self.timestep + self.prediction_horizon
        ] = True
        actions_populated = self.all_time_populated_mask[:, self.timestep]
        actions_for_curr_step_pos = self.all_time_position_actions[:, self.timestep][actions_populated]
        exp_weights = self._compute_exp_weights(len(actions_for_curr_step_pos))
        averaged[self.position_key] = (actions_for_curr_step_pos * exp_weights).sum(dim=0)
        self.all_time_orientations[
            [self.timestep], self.timestep: self.timestep + self.prediction_horizon
        ] = self.current_all_orientations.float()
        actions_for_curr_step_ori = self.all_time_orientations[:, self.timestep][actions_populated]
        exp_weights = self._compute_exp_weights(len(actions_for_curr_step_ori))
        averaged[self.orientation_key] = (actions_for_curr_step_ori * exp_weights).sum(dim=0)
        self.all_time_grippers[
            [self.timestep], self.timestep: self.timestep + self.prediction_horizon
        ] = self.current_all_grippers.float()
        actions_for_curr_step_grip = self.all_time_grippers[:, self.timestep][actions_populated]
        exp_weights = self._compute_exp_weights(len(actions_for_curr_step_grip))
        averaged[self.gripper_key] = (actions_for_curr_step_grip * exp_weights).sum(dim=0)
        return averaged


    def _compute_exp_weights(self, n: int) -> torch.Tensor:
        """Compute exponential weights for temporal aggregation."""
        indices = np.arange(n)
        if self.favor_more_recent:
            indices = indices[::-1]
        exp_weights = np.exp(-self.exponential_decay * indices)
        exp_weights = exp_weights / exp_weights.sum()
        return torch.from_numpy(exp_weights).to(self.device).float().unsqueeze(dim=1)


    def _construct_libero_action(self, averaged_actions: dict[str, torch.Tensor]) -> tuple[np.ndarray, bool]:
        """Construct LIBERO action from averaged action tensors."""
        return self._construct_libero_action_from_tensors(
            averaged_actions[self.position_key],
            averaged_actions[self.orientation_key],
            averaged_actions[self.gripper_key],
        )


    def _construct_libero_action_from_tensors(
        self,
        position: torch.Tensor,
        orientation: torch.Tensor,
        gripper: torch.Tensor,
    ) -> tuple[np.ndarray, bool]:
        """Construct LIBERO 7D action from tensors."""
        position_action = position.cpu().detach().float().numpy().flatten()[:3]
        orientation_action = orientation.cpu().detach().float().numpy().flatten()[:3]
        gripper_raw_output = gripper.cpu().detach().float().numpy().flatten()[0]

        if self.gripper_type == GripperType.BINARY.value:
            # Apply sigmoid to convert logits to probability in range [0, 1]
            gripper_probability = 1.0 / (1.0 + np.exp(-gripper_raw_output))
            gripper_is_closed = gripper_probability > 0.5
            # Convert probability [0, 1] to LIBERO range [-1, 1] where -1 = open, +1 = closed
            gripper_action_for_libero = 2.0 * gripper_probability - 1.0
        else:
            gripper_action_for_libero = gripper_raw_output
            gripper_is_closed = gripper_raw_output > 0.0

        robot_action = np.concatenate([position_action, orientation_action, [gripper_action_for_libero]])
        return robot_action, gripper_is_closed


    def update_loop(self) -> None:
        """Main loop to collect observations, manage buffers, and send actions."""
        while True:
            obs, done, success = self.get_observation()
            if done:
                self.reset()
                continue

            if obs.agentview_rgb is not None:
                self.agentview_buffer.append(obs.agentview_rgb)
            if obs.eye_in_hand_rgb is not None:
                self.eye_in_hand_buffer.append(obs.eye_in_hand_rgb)
            if obs.ee_pos is not None:
                self.ee_pos_buffer.append(obs.ee_pos)
            if obs.ee_ori is not None:
                self.ee_ori_buffer.append(obs.ee_ori)
            if obs.gripper_states is not None:
                self.gripper_states_buffer.append(obs.gripper_states)
            if obs.language_instruction is not None:
                self.language_instruction_buffer.append(obs.language_instruction)

            self.buffer_data_counter += 1

            if self.buffer_data_counter > self.observation_buffer_size:
                if len(self.agentview_buffer) > self.observation_buffer_size:
                    self.agentview_buffer.pop(0)
                if len(self.eye_in_hand_buffer) > self.observation_buffer_size:
                    self.eye_in_hand_buffer.pop(0)
                if len(self.ee_pos_buffer) > self.observation_buffer_size:
                    self.ee_pos_buffer.pop(0)
                if len(self.ee_ori_buffer) > self.observation_buffer_size:
                    self.ee_ori_buffer.pop(0)
                if len(self.gripper_states_buffer) > self.observation_buffer_size:
                    self.gripper_states_buffer.pop(0)
                if len(self.language_instruction_buffer) > self.observation_buffer_size:
                    self.language_instruction_buffer.pop(0)
                self.buffer_data_counter -= 1

            if self.buffer_data_counter == self.observation_buffer_size:
                actions = self.get_actions_from_model()
                for i, action in enumerate(actions):
                    if self.enable_logging:
                        logging.info(f"Sending Action {i}: {action}")
                    done, success = self.send_action(robot_action=action.robot_action)
                    if done:
                        self.reset()
                        break
                    time.sleep(1 / self.update_rate_hz)


    def reset(self) -> None:
        """Reset the client state for a new episode."""
        self.timestep = 0
        self.all_time_position_actions.zero_()
        self.all_time_populated_mask.zero_()
        self.all_time_orientations.zero_()
        self.all_time_grippers.zero_()
        self.current_all_position_actions = None
        self.current_all_orientations = None
        self.current_all_grippers = None
        self.agentview_buffer.clear()
        self.eye_in_hand_buffer.clear()
        self.ee_pos_buffer.clear()
        self.ee_ori_buffer.clear()
        self.gripper_states_buffer.clear()
        self.language_instruction_buffer.clear()
        self.buffer_data_counter = 0


    def shutdown(self) -> None:
        """Shut down the client and close the ZMQ socket."""
        self.close()