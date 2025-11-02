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
)


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
        """Run all validation checks and raise if any fail."""
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

        available_keys = set()
        available_keys.update(obs_space.camera_keys)
        if obs_space.use_proprio_base_frame:
            available_keys.add(PROPRIO_OBS_ROBOT_FRAME_KEY)
        if obs_space.use_proprio_camera_frame:
            available_keys.add(PROPRIO_OBS_CAMERA_FRAME_KEY)
        if obs_space.use_language:
            available_keys.add(LANGUAGE_KEY)
        if obs_space.use_gripper_state:
            available_keys.add(GRIPPER_STATE_OBS_KEY)
        available_keys.update(obs_space.custom_obs_keys)

        configured_encoder_inputs = set()
        for encoder_name, encoder_config in pipeline.encoders.items():
            if hasattr(encoder_config, 'backbone') and encoder_config.backbone.startswith('dinov3') and task.dataloader_config.image_norm_type != ImageNormalizationType.IMAGENET.value:
                self.errors.append(
                    f"Encoder '{encoder_name}' uses DINOv3 backbone which requires "
                    f"ImageNet normalization, but dataloader.image_norm_type is set to "
                        f"'{task.dataloader_config.image_norm_type}'. Set it to 'imagenet'."
                    )

            input_keys = encoder_config.input_keys
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
            self.errors.append(
                f"Observation space contains keys {uncovered_keys} "
                f"but no encoder is configured to process them. "
                f"Either remove from observation_space or add corresponding encoders."
            )


    def validate_task_schema_consistency(self) -> None:
        """Validate that task config requests are supported by dataset schema."""
        obs_space = self.cfg.task.observation_space
        action_space = self.cfg.task.action_space
        schema = self.cfg.task.dataset_schema
        raw_obs = schema.raw_observations

        missing_keys = []
        available_cameras = set(raw_obs.camera_keys)
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
                f"Task requires keys {set(missing_keys)} but schema doesn't provide them."
            )


    def validate_conditional_encoder_dependencies(self) -> None:
        """Validate that conditional encoders have their dependencies satisfied."""
        pipeline = self.cfg.policy.encoding_pipeline
        available_outputs = set()
        conditional_requirements = {}
        for encoder_name, encoder_config in pipeline.encoders.items():
            output_names = encoder_config.output_keys
            for output_name in output_names:
                available_outputs.add(output_name)
            if hasattr(encoder_config, 'condition_key'):
                conditional_requirements[encoder_name] = encoder_config.condition_key

        for encoder_name, required_condition in conditional_requirements.items():
            if required_condition not in available_outputs:
                self.errors.append(
                    f"Conditional encoder '{encoder_name}' requires condition '{required_condition}', "
                    f"but no encoder produces this output. "
                    f"Available outputs: {available_outputs}"
                )
            # Additional check: conditioning must come from non-conditional encoder
            for other_name, other_config in pipeline.encoders.items():
                if required_condition in other_config.output_keys:
                    producing_encoder = other_name
                    if hasattr(other_config, 'condition_key'):
                        self.errors.append(
                            f"Conditional encoder '{encoder_name}' depends on '{required_condition}' "
                            f"from '{producing_encoder}', but '{producing_encoder}' is also conditional. "
                            f"The two-pass pipeline requires conditions to come from non-conditional encoders."
                        )
                    break


def validate_config(cfg: DictConfig) -> None:
    """Convenience function to validate configuration.

    Args:
        cfg: Complete Hydra configuration

    Raises:
        ConfigValidationError: If validation fails
    """
    validator = ConfigValidator(cfg)
    validator.validate_all()
