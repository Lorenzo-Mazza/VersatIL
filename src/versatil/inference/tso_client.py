"""Inference client for real-time deployment on the TSO robot testbed.

Note:
    The TSO Policy Server uses delta actions for position and orientation.
    Regardless of training configuration, absolute predictions are
    converted to deltas before being sent to the server.
"""
import logging
import time

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from imitation_learning_toolkit.sockets.model_client import (
    AbstractModelClient,
    Action,
)

from versatil.data.constants import (
    ActionComputationMethod,
    BinaryGripperRange,
    Cameras,
    CoordinateSystem,
    GripperType,
    ObsKey,
    ProprioKey,
)
from versatil.data.metadata import OnTheFlyActionMetadata
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.inference.base_client import BaseInferenceClient
from versatil.inference.temporal_aggregation import TemporalAggregator
from versatil.training.constants import PrecisionType

logging.basicConfig(level=logging.INFO)


class TSOPolicyClient(AbstractModelClient, BaseInferenceClient):
    """Client for real-time inference on the TSO robot testbed."""

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
        """Initialize TSO inference client.

        Args:
            device: Device to run inference on.
            checkpoint_path: Path to checkpoint directory.
            checkpoint_name: Name of the checkpoint file.
            model_server_address: Address of the model server.
            model_server_port: Port of the model server.
            temporal_agg: Whether to use temporal aggregation.
            favor_more_recent: Weight newer predictions more heavily.
            exponential_decay: Decay factor for temporal aggregation.
            update_rate_hz: Update frequency in Hz.
            timing_log: Whether to log timing information.
            precision: Precision type for model inference.
        """
        self.temporal_agg = temporal_agg
        self.timing_log = timing_log
        BaseInferenceClient.__init__(
            self,
            device=device,
            checkpoint_path=checkpoint_path,
            checkpoint_name=checkpoint_name,
            precision=precision,
        )
        action_space = self.action_space
        self.action_dim = action_space.get_total_action_dim()
        self._setup_position_action(action_space=action_space)
        self._setup_orientation_action(action_space=action_space)
        self._setup_gripper_action(action_space=action_space)
        self._setup_observations(observation_space=self.observation_space)
        self._setup_denoising_thresholds()
        if update_rate_hz is None:
            update_rate_hz = 10.0
        AbstractModelClient.__init__(
            self,
            model_server_address=model_server_address,
            model_server_port=model_server_port,
            observation_buffer_size=self.observation_horizon,
            request_depth=self.use_depth,
            request_rectified_images=True,
            request_gripper_state=self.has_gripper,
            request_language_instruction=self.use_language,
            predicts_in_camera_frame=(
                self.position_frame == CoordinateSystem.CAMERA.value
            ),
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
                A.Resize(
                    height=self.config.task.dataloader.image_height,
                    width=self.config.task.dataloader.image_width,
                ),
                ToTensorV2(),
            ],
            additional_targets=additional_targets,
        )
        if self.has_orientation and self.orientation_dim != 1:
            raise NotImplementedError(
                "Only 1D orientation (roll) is supported for TSO."
            )
        if self.temporal_agg:
            action_keys_to_dimensions = {
                self.position_key: self.position_dim,
            }
            if self.has_orientation:
                action_keys_to_dimensions[self.orientation_key] = (
                    self.orientation_dim
                )
            if self.has_gripper:
                action_keys_to_dimensions[self.gripper_key] = (
                    self.gripper_dim
                )
            self.temporal_aggregator = TemporalAggregator(
                device=self.device,
                action_keys_to_dimensions=action_keys_to_dimensions,
                prediction_horizon=self.prediction_horizon,
                exponential_decay=exponential_decay,
                favor_more_recent=favor_more_recent,
            )
        self.timestep = 0

    def _post_load_model(self) -> None:
        """Extract depth normalization statistics from the model."""
        if Cameras.DEPTH.value in self.observation_space.cameras:
            depth_stats = self.policy.normalizer[
                Cameras.DEPTH.value
            ].params_dict["input_stats"]
            self.depth_min = float(depth_stats["min"].item())
            self.depth_max = float(depth_stats["max"].item())
            logging.info(
                f"Depth clipping range: "
                f"[{self.depth_min:.4f}, {self.depth_max:.4f}]"
            )
        else:
            self.depth_min = None
            self.depth_max = None

    def _setup_position_action(self, action_space: ActionSpace) -> None:
        """Setup position action key and metadata from ActionSpace."""
        position_camera_key = (
            ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value
        )
        position_robot_key = (
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        )
        if position_camera_key in action_space.actions_metadata:
            self.position_key = position_camera_key
        elif position_robot_key in action_space.actions_metadata:
            self.position_key = position_robot_key
        else:
            raise ValueError(
                "TSO InferenceClient requires position actions. "
                f"Expected key "
                f"'{ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value}' or "
                f"'{ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value}' in "
                f"action_space.actions_metadata. "
                f"Got keys: {list(action_space.actions_metadata.keys())}"
            )
        self.has_position = True
        position_metadata = action_space.actions_metadata[self.position_key]
        if isinstance(position_metadata, OnTheFlyActionMetadata):
            self.predicts_delta = (
                position_metadata.computation_method
                == ActionComputationMethod.DELTA.value
            )
            self.position_frame = (
                position_metadata.source_metadata.frame
            )
            self.position_dim = position_metadata.prediction_dimension
        else:
            raise ValueError(
                "TSO InferenceClient only supports "
                "OnTheFlyActionMetadata for position actions."
            )

    def _setup_orientation_action(
        self, action_space: ActionSpace
    ) -> None:
        """Setup orientation action key and metadata from ActionSpace."""
        orientation_camera_key = (
            ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_ORI.value
        )
        orientation_robot_key = (
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_ORI.value
        )
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
        orientation_metadata = action_space.actions_metadata[
            self.orientation_key
        ]
        if isinstance(orientation_metadata, OnTheFlyActionMetadata):
            self.orientation_representation = (
                orientation_metadata.source_metadata.orientation_representation
            )
            self.orientation_frame = (
                orientation_metadata.source_metadata.frame
            )
        else:
            self.orientation_representation = (
                orientation_metadata.orientation_representation
            )
            self.orientation_frame = orientation_metadata.frame
        self.orientation_dim = orientation_metadata.prediction_dimension

    def _setup_gripper_action(self, action_space: ActionSpace) -> None:
        """Setup gripper action key and metadata from ActionSpace."""
        gripper_key = ProprioKey.GRIPPER_STATE.value
        if gripper_key in action_space.actions_metadata:
            self.gripper_key = gripper_key
            self.has_gripper = True
            gripper_metadata = action_space.actions_metadata[gripper_key]
            if isinstance(gripper_metadata, OnTheFlyActionMetadata):
                self.gripper_type = (
                    gripper_metadata.source_metadata.gripper_type
                )
                self.binary_gripper_range = (
                    gripper_metadata.source_metadata.binary_gripper_range
                )
            else:
                self.gripper_type = gripper_metadata.gripper_type
                self.binary_gripper_range = (
                    gripper_metadata.binary_gripper_range
                )
            self.gripper_dim = gripper_metadata.prediction_dimension
        else:
            self.gripper_key = None
            self.has_gripper = False
            self.gripper_type = None
            self.binary_gripper_range = None
            self.gripper_dim = 0
        if (
            self.gripper_type == GripperType.BINARY.value
            and self.binary_gripper_range is None
        ):
            logging.warning(
                "Gripper binary range is not set. Assuming {0,1}."
            )
            self.binary_gripper_range = BinaryGripperRange.ZERO_ONE.value

    def _setup_observations(
        self, observation_space: ObservationSpace
    ) -> None:
        """Setup observation keys from ObservationSpace metadata."""
        position_camera_key = (
            ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value
        )
        position_robot_key = (
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        )
        self.use_depth = (
            Cameras.DEPTH.value in observation_space.cameras
        )
        self.use_language = (
            ObsKey.LANGUAGE.value
            in observation_space.observations_metadata
        )
        self.use_proprio_camera_frame = (
            position_camera_key in observation_space.observations_metadata
        )
        self.use_proprio_robot_frame = (
            position_robot_key in observation_space.observations_metadata
        )

    def _setup_denoising_thresholds(self) -> None:
        """Setup denoising thresholds from policy."""
        denoising_thresholds = (
            self.policy.denoising_thresholds.params_dict
        )
        if self.position_key in denoising_thresholds:
            self.position_delta_threshold = float(
                denoising_thresholds[self.position_key].item()
            )
            logging.info(
                f"Position delta denoising threshold "
                f"[{self.position_key}]: "
                f"{self.position_delta_threshold:.6f}"
            )
        else:
            self.position_delta_threshold = 0.0
            logging.info(
                "No position denoising threshold found, "
                "denoising disabled for position"
            )
        if (
            self.orientation_key
            and self.orientation_key in denoising_thresholds
        ):
            self.orientation_delta_threshold = float(
                denoising_thresholds[self.orientation_key].item()
            )
            logging.info(
                f"Orientation delta denoising threshold "
                f"[{self.orientation_key}]: "
                f"{self.orientation_delta_threshold:.6f}"
            )
        else:
            self.orientation_delta_threshold = 0.0
            if self.has_orientation:
                logging.info(
                    "No orientation denoising threshold found, "
                    "denoising disabled for orientation"
                )

    def get_actions_from_model(self) -> list[Action]:
        """Compute next actions using the trained policy.

        Returns:
            List of Action objects for the robot.
        """
        total_start_time = None
        preprocessing_start_time = None
        inference_start_time = None
        postprocessing_start_time = None
        preprocessing_duration = None
        inference_duration = None
        postprocessing_duration = None

        if self.timing_log:
            total_start_time = time.time()
            logging.info(
                f"\n=== TIMESTEP {self.timestep} - "
                f"Starting get_actions_from_model ==="
            )
            preprocessing_start_time = time.time()

        if self.obs_camera_frame and self.obs_robot_frame:
            state_dim = 6
        elif self.obs_camera_frame or self.obs_robot_frame:
            state_dim = 3
        else:
            state_dim = 0

        if state_dim > 0:
            last_states = self.robot_state_buffer[
                -self.observation_buffer_size :
            ]
            qpos = np.array(
                [state[:state_dim] for state in last_states]
            )
            qpos_tensor = torch.tensor(
                qpos, dtype=torch.float32
            ).unsqueeze(0)
        else:
            qpos_tensor = None

        left_image_list = self.left_image_buffer[
            -self.observation_buffer_size :
        ]
        right_image_list = self.right_image_buffer[
            -self.observation_buffer_size :
        ]

        if self.timing_log:
            depth_processing_start = time.time()

        depth_images = None
        if self.request_depth:
            depth_image_list = self.depth_buffer[
                -self.observation_buffer_size :
            ]
            transformed = [
                self.image_transform(
                    image=left_numpy,
                    right_image=right_numpy,
                    depth=depth_numpy,
                )
                for left_numpy, right_numpy, depth_numpy in zip(
                    left_image_list,
                    right_image_list,
                    depth_image_list,
                )
            ]
            depth_tensors = [t["depth"] for t in transformed]
            depth_images = (
                torch.stack(depth_tensors).unsqueeze(0).unsqueeze(-3)
            )
            if (
                self.depth_min is not None
                and self.depth_max is not None
            ):
                depth_images = torch.clamp(
                    depth_images,
                    min=self.depth_min,
                    max=self.depth_max,
                )
        else:
            transformed = [
                self.image_transform(
                    image=left_numpy, right_image=right_numpy
                )
                for left_numpy, right_numpy in zip(
                    left_image_list, right_image_list
                )
            ]

        if self.timing_log:
            logging.info(
                f"[TIMING] Depth plus RGB transform took: "
                f"{time.time() - depth_processing_start:.6f} seconds"
            )
            rgb_processing_start = time.time()

        left_tensors = [t["image"] / 255.0 for t in transformed]
        right_tensors = [
            t["right_image"] / 255.0 for t in transformed
        ]
        left_images = torch.stack(left_tensors).unsqueeze(0)
        right_images = torch.stack(right_tensors).unsqueeze(0)

        if self.timing_log:
            logging.info(
                f"[TIMING] RGB processing took: "
                f"{time.time() - rgb_processing_start:.6f} seconds"
            )

        obs_dict = {
            Cameras.LEFT.value: left_images,
            Cameras.RIGHT.value: right_images,
        }

        if state_dim > 0:
            position_robot_key = (
                ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
            )
            position_camera_key = (
                ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value
            )
            if self.obs_robot_frame and self.obs_camera_frame:
                obs_dict[position_robot_key] = qpos_tensor[
                    :, :, :3
                ]
                obs_dict[position_camera_key] = qpos_tensor[
                    :, :, 3:
                ]
            elif self.obs_robot_frame:
                obs_dict[position_robot_key] = qpos_tensor
            elif self.obs_camera_frame:
                obs_dict[position_camera_key] = qpos_tensor

        if self.request_depth:
            obs_dict[Cameras.DEPTH.value] = depth_images

        if self.request_language_instruction:
            language_instruction = self.language_instruction_buffer[
                -self.observation_buffer_size :
            ]
            obs_dict[ObsKey.LANGUAGE.value] = language_instruction

        if self.timing_log:
            preprocessing_end_time = time.time()
            preprocessing_duration = (
                preprocessing_end_time - preprocessing_start_time
            )
            logging.info(
                f"[TIMING] Input preprocessing completed in: "
                f"{preprocessing_duration:.6f} seconds"
            )

        current_roll = 0.0
        if (
            self.predicts_in_camera_frame
            and self.obs_camera_frame
            and self.obs_robot_frame
        ):
            current_robot_position = self.robot_state_buffer[-1][3:6]
            if self.has_orientation:
                current_roll = (
                    self.robot_state_buffer[-1][6]
                    if len(self.robot_state_buffer[-1]) > 6
                    else 0.0
                )
        else:
            current_robot_position = self.robot_state_buffer[-1][:3]
            if self.has_orientation:
                current_roll = (
                    self.robot_state_buffer[-1][4]
                    if len(self.robot_state_buffer[-1]) > 4
                    else 0.0
                )

        if self.timing_log:
            inference_start_time = time.time()

        action_dict = self._run_inference(obs_dict=obs_dict)

        position_predictions = action_dict[self.position_key]
        orientation_predictions = (
            action_dict[self.orientation_key]
            if self.has_orientation
            else None
        )
        gripper_predictions = (
            action_dict[self.gripper_key]
            if self.has_gripper
            else None
        )

        if self.timing_log:
            inference_end_time = time.time()
            inference_duration = (
                inference_end_time - inference_start_time
            )
            logging.info(
                f"[TIMING] Model inference completed in: "
                f"{inference_duration:.6f} seconds"
            )
            postprocessing_start_time = time.time()

        if self.temporal_agg:
            current_predictions = {
                self.position_key: position_predictions[0],
            }
            if self.has_orientation:
                current_predictions[self.orientation_key] = (
                    orientation_predictions[0]
                )
            if self.has_gripper:
                current_predictions[self.gripper_key] = (
                    gripper_predictions[0]
                )
            averaged = self.temporal_aggregator.store_and_average(
                current_predictions=current_predictions
            )
            robot_action, gripper_action = (
                self._postprocess_action_tensors(
                    raw_position_tensor=averaged[self.position_key],
                    raw_orientation_tensor=(
                        averaged.get(self.orientation_key)
                        if self.has_orientation
                        else None
                    ),
                    raw_gripper_tensor=(
                        averaged.get(self.gripper_key)
                        if self.has_gripper
                        else None
                    ),
                    current_robot_position=current_robot_position,
                    current_roll=current_roll,
                )
            )
            actions = [
                Action(
                    robot_action=robot_action,
                    gripper_action=gripper_action,
                )
            ]
        else:
            actions = []
            for step_index in range(self.prediction_horizon):
                robot_action, gripper_action = (
                    self._postprocess_action_tensors(
                        raw_position_tensor=(
                            position_predictions[0, step_index]
                        ),
                        raw_orientation_tensor=(
                            orientation_predictions[0, step_index]
                            if self.has_orientation
                            else None
                        ),
                        raw_gripper_tensor=(
                            gripper_predictions[0, step_index]
                            if self.has_gripper
                            else None
                        ),
                        current_robot_position=current_robot_position,
                        current_roll=current_roll,
                    )
                )
                actions.append(
                    Action(
                        robot_action=robot_action,
                        gripper_action=gripper_action,
                    )
                )

        if self.timing_log:
            postprocessing_end_time = time.time()
            postprocessing_duration = (
                postprocessing_end_time - postprocessing_start_time
            )
            logging.info(
                f"[TIMING] Post-processing completed in: "
                f"{postprocessing_duration:.6f} seconds"
            )

        self.timestep += 1

        if self.timing_log:
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time
            logging.info(
                f"\n[TIMING SUMMARY] Timestep {self.timestep - 1}:"
            )
            logging.info(
                f"  - Preprocessing: {preprocessing_duration:.6f}s "
                f"({preprocessing_duration / total_duration * 100:.1f}%)"
            )
            logging.info(
                f"  - Model inference: {inference_duration:.6f}s "
                f"({inference_duration / total_duration * 100:.1f}%)"
            )
            logging.info(
                f"  - Post-processing: {postprocessing_duration:.6f}s "
                f"({postprocessing_duration / total_duration * 100:.1f}%)"
            )
            logging.info(f"  - TOTAL: {total_duration:.6f}s")
            logging.info(
                f"  - Effective FPS: {1.0 / total_duration:.2f}"
            )
            logging.info(
                f"=== TIMESTEP {self.timestep - 1} COMPLETE ===\n"
            )

        if self.enable_logging:
            logging.log(level=logging.INFO, msg=f"{actions=}")
        logging.info(msg=actions)
        return actions

    def _postprocess_action_tensors(
        self,
        raw_position_tensor: torch.Tensor,
        raw_orientation_tensor: torch.Tensor | None,
        raw_gripper_tensor: torch.Tensor | None,
        current_robot_position: np.ndarray,
        current_roll: float,
    ) -> tuple[np.ndarray, np.ndarray | None | bool]:
        """Post-process raw action tensors for one step.

        Converts absolute predictions to deltas, then applies
        norm-based denoising to filter out small-magnitude deltas.
        """
        position_action = (
            raw_position_tensor.cpu().detach().float().numpy()[:3]
        )
        if not self.predicts_delta:
            position_action = position_action - current_robot_position

        if self.position_delta_threshold > 0:
            position_norm = np.linalg.norm(position_action)
            if position_norm < self.position_delta_threshold:
                position_action = np.zeros_like(position_action)

        if raw_gripper_tensor is not None:
            raw_gripper_action = (
                raw_gripper_tensor.cpu().detach().float().numpy()
            )
            if self.gripper_type == GripperType.BINARY.value:
                if (
                    self.binary_gripper_range
                    == BinaryGripperRange.ZERO_ONE.value
                ):
                    gripper_action = raw_gripper_action > 0.5
                else:
                    gripper_action = raw_gripper_action > 0.0
            else:
                gripper_action = raw_gripper_action
        else:
            gripper_action = None

        if self.has_orientation:
            if raw_orientation_tensor is None:
                raise ValueError(
                    "Orientation tensor is None but "
                    "has_orientation is True."
                )
            orientation_action = (
                raw_orientation_tensor.cpu().detach().float().numpy()
            )
            if not self.predicts_delta:
                orientation_action = (
                    orientation_action[0] - current_roll
                )

            if self.orientation_delta_threshold > 0:
                orientation_magnitude = np.abs(orientation_action)
                if (
                    orientation_magnitude
                    < self.orientation_delta_threshold
                ):
                    orientation_action = 0.0

            robot_action = np.concatenate(
                (
                    position_action,
                    [
                        orientation_action[0]
                        if isinstance(orientation_action, np.ndarray)
                        else orientation_action
                    ],
                )
            )
        else:
            robot_action = np.concatenate(
                (position_action, [0.0])
            )  # Roll = 0.0 if no orientation predicted

        return robot_action, gripper_action