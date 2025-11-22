"""Configuration validation module.

Validates consistency across all configuration components after Hydra composition.
"""
import logging

from omegaconf import DictConfig

from refactoring.data.constants import (
    GRIPPER_STATE_OBS_KEY,
    LANGUAGE_KEY,
    PHASE_LABEL_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    ImageNormalizationType,
    TOKENIZED_OBSERVATIONS_KEY, IS_PAD_OBSERVATION_KEY,
)
from refactoring.data.schemas.base import DatasetSchema
from refactoring.models.encoding.encoders.base import EncodingMixin


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidator:
    """Validates consistency across configuration components."""


    def __init__(self, cfg: DictConfig):
        """Initialize validator.

        Args:
            cfg: Complete Hydra configuration
        """
        self.cfg = cfg
        self.errors: list[str] = []
        self.warnings: list[str] = []


    def validate_all(self) -> None:
        """Run all validation checks and raise if any fail.

        Note:
            Decoder input validation (ensuring decoder requirements match encoder outputs)
            is performed at Policy instantiation time in Policy.validate_decoder(),
            since it requires instantiating the encoding pipeline to get output features.
        """
        self.validate_task_schema_consistency()
        self.validate_encoder_task_consistency()

        if self.errors:
            error_msg = "\n".join([f"  - {err}" for err in self.errors])
            raise ConfigValidationError(f"Configuration validation failed with {len(self.errors)} error(s):\n{error_msg}")
        if self.warnings:
            warning_msg = "\n".join([f"  - {warn}" for warn in self.warnings])
            logging.warning(msg=f"Configuration warnings ({len(self.warnings)}):\n{warning_msg}")


    def validate_encoder_task_consistency(self) -> None:
        """Validate that encoders match task requirements."""
        task = self.cfg.task
        obs_space = task.observation_space
        pipeline = self.cfg.policy.encoding_pipeline
        is_obs_tokenized = task.dataloader.tokenization.tokenize_observations

        if obs_space.use_language and not is_obs_tokenized:
            self.errors.append(
                "Language observations are enabled but tokenization is disabled. "
                "Language observations require tokenization to be enabled."
            )
            return  
        # Build set of available observation keys based on observation space and tokenization
        available_keys = set()
        available_keys.update(obs_space.camera_keys)

        if is_obs_tokenized and task.dataloader.tokenization.observation_tokenizer:
            token_obs_keys = task.dataloader.tokenization.observation_tokenizer.observation_keys

            # Check each observation type - if tokenized, add TOKENIZED_OBSERVATIONS_KEY
            tokenized_any = False
            if obs_space.use_proprio_base_frame and PROPRIO_OBS_ROBOT_FRAME_KEY in token_obs_keys:
                tokenized_any = True
            elif obs_space.use_proprio_base_frame:
                available_keys.add(PROPRIO_OBS_ROBOT_FRAME_KEY)

            if obs_space.use_proprio_camera_frame and PROPRIO_OBS_CAMERA_FRAME_KEY in token_obs_keys:
                tokenized_any = True
            elif obs_space.use_proprio_camera_frame:
                available_keys.add(PROPRIO_OBS_CAMERA_FRAME_KEY)

            if obs_space.use_gripper_state and GRIPPER_STATE_OBS_KEY in token_obs_keys:
                tokenized_any = True
            elif obs_space.use_gripper_state:
                available_keys.add(GRIPPER_STATE_OBS_KEY)

            if obs_space.use_language and LANGUAGE_KEY in token_obs_keys:
                tokenized_any = True
            elif obs_space.use_language:
                self.errors.append(
                    f"Language observations are enabled but '{LANGUAGE_KEY}' is not in "
                    f"observation_tokenizer.observation_keys: {token_obs_keys}"
                )

            for custom_key in obs_space.custom_obs_keys:
                if custom_key in token_obs_keys:
                    tokenized_any = True
                else:
                    available_keys.add(custom_key)

            # Add tokenized observations key if any observations are tokenized
            if tokenized_any:
                available_keys.add(TOKENIZED_OBSERVATIONS_KEY)
                available_keys.add(IS_PAD_OBSERVATION_KEY)
        else:
            # No tokenization - add raw observation keys
            if obs_space.use_proprio_base_frame:
                available_keys.add(PROPRIO_OBS_ROBOT_FRAME_KEY)
            if obs_space.use_proprio_camera_frame:
                available_keys.add(PROPRIO_OBS_CAMERA_FRAME_KEY)
            if obs_space.use_gripper_state:
                available_keys.add(GRIPPER_STATE_OBS_KEY)
            available_keys.update(obs_space.custom_obs_keys)

        # Validate encoders
        configured_encoder_inputs = set()
        for encoder_name, encoder in pipeline.encoders.items():
            encoder: EncodingMixin
            if hasattr(encoder, 'backbone_name') and 'dinov3' in encoder.backbone_name.lower():
                if task.dataloader.image_norm_type != ImageNormalizationType.IMAGENET.value:
                    self.errors.append(
                        f"Encoder '{encoder_name}' uses DINOv3 backbone which requires "
                        f"ImageNet normalization, but dataloader.image_norm_type is set to "
                        f"'{task.dataloader.image_norm_type}'. Set it to 'imagenet'."
                    )

            input_keys = encoder.input_specification.keys
            if isinstance(input_keys, str):
                input_keys = [input_keys]

            configured_encoder_inputs.update(input_keys)

            missing = set(input_keys) - available_keys
            if missing:
                self.errors.append(
                    f"Encoder '{encoder_name}' requires keys {missing} "
                    f"which are not in observation space. Please either add them to the observation space or modify encoder configuration."
                )

        uncovered_keys = available_keys - configured_encoder_inputs
        if uncovered_keys:
            self.warnings.append(
                f"Observation space contains keys {uncovered_keys} "
                f"but no encoder is configured to process them. "
            )


    def validate_task_schema_consistency(self) -> None:
        """Validate that task config requests are supported by dataset schema.

        Instantiates the dataset schema to access its attributes.
        """
        obs_space = self.cfg.task.observation_space
        action_space = self.cfg.task.action_space
        schema: DatasetSchema = self.cfg.task.dataset_schema
        raw_obs = schema.raw_observations
        missing_keys = []
        available_cameras = set(raw_obs.camera_keys) if raw_obs.camera_keys else set()
        missing_keys.extend([cam for cam in obs_space.camera_keys if cam not in available_cameras])
        if obs_space.use_proprio_base_frame and not raw_obs.robot_frame_proprio_keys:
            missing_keys.append(PROPRIO_OBS_ROBOT_FRAME_KEY)
        if obs_space.use_proprio_camera_frame and not raw_obs.camera_frame_proprio_keys:
            missing_keys.append(PROPRIO_OBS_CAMERA_FRAME_KEY)
        if action_space.has_position or action_space.has_orientation:
            if action_space.predict_in_camera_frame and not raw_obs.camera_frame_proprio_keys:
                missing_keys.append(PROPRIO_OBS_CAMERA_FRAME_KEY)
            elif not action_space.predict_in_camera_frame and not raw_obs.robot_frame_proprio_keys:
                missing_keys.append(PROPRIO_OBS_ROBOT_FRAME_KEY)
        if (action_space.has_gripper or obs_space.use_gripper_state) and not raw_obs.gripper_state_keys:
            missing_keys.append(GRIPPER_STATE_OBS_KEY)
        if obs_space.use_language and not raw_obs.language_key:
            missing_keys.append(LANGUAGE_KEY)
        if action_space.task_has_phases and not schema.has_phase_labels:
            missing_keys.append(PHASE_LABEL_KEY)
        available_custom = set(raw_obs.custom_obs_keys.keys()) if raw_obs.custom_obs_keys else set()
        missing_keys.extend([key for key in obs_space.custom_obs_keys if key not in available_custom])
        if missing_keys:
            self.errors.append(
                f"Task requires keys {set(missing_keys)} but schema doesn't provide them. "
                f"Please update your dataset schema or task configuration."
            )




def validate_config(cfg: DictConfig) -> None:
    """Convenience function to validate configuration.

    Args:
        cfg: Complete Hydra configuration

    Raises:
        ConfigValidationError: If validation fails
    """
    validator = ConfigValidator(cfg)
    validator.validate_all()
