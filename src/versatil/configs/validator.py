"""Configuration validation module.

Validates consistency across all configuration components after Hydra composition.
"""
import logging

from omegaconf import DictConfig

from versatil.data.constants import (
    ImageNormalizationType,
    ObsKey,
    TOKENIZED_OBSERVATIONS_KEY,
    IS_PAD_OBSERVATION_KEY,
)
from versatil.data.metadata import CameraMetadata
from versatil.data.task import ObservationSpace
from versatil.models.encoding.encoders.base import EncodingMixin


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
            Task-schema consistency validation is performed by TaskSpace._validate()
            when the TaskSpace is instantiated.

            Decoder input validation (ensuring decoder requirements match encoder outputs)
            is performed at Policy instantiation time in Policy.validate_decoder(),
            since it requires instantiating the encoding pipeline to get output features.
        """
        self.validate_encoder_task_consistency()

        if self.errors:
            error_msg = "\n".join([f"  - {err}" for err in self.errors])
            raise ConfigValidationError(
                f"Configuration validation failed with {len(self.errors)} error(s):\n{error_msg}"
            )
        if self.warnings:
            warning_msg = "\n".join([f"  - {warn}" for warn in self.warnings])
            logging.warning(
                msg=f"Configuration warnings ({len(self.warnings)}):\n{warning_msg}"
            )

    def validate_encoder_task_consistency(self) -> None:
        """Validate that encoders match task requirements."""
        task = self.cfg.task
        obs_space: ObservationSpace = task.observation_space
        pipeline = self.cfg.policy.encoding_pipeline
        is_obs_tokenized = task.dataloader.tokenization.tokenize_observations

        has_language = ObsKey.LANGUAGE.value in obs_space.observations_metadata
        if has_language and not is_obs_tokenized:
            self.errors.append(
                "Language observations are enabled but tokenization is disabled. "
                "Language observations require tokenization to be enabled."
            )
            return

        # Build set of available observation keys from observations_metadata
        available_keys = set()

        if is_obs_tokenized and task.dataloader.tokenization.observation_tokenizer:
            token_obs_keys = set(
                task.dataloader.tokenization.observation_tokenizer.observation_keys
            )
            tokenized_any = False

            for obs_key, obs_meta in obs_space.observations_metadata.items():
                if isinstance(obs_meta, CameraMetadata):
                    available_keys.add(obs_key)
                elif obs_key in token_obs_keys:
                    tokenized_any = True
                else:
                    available_keys.add(obs_key)

            if has_language and ObsKey.LANGUAGE.value not in token_obs_keys:
                self.errors.append(
                    f"Language observations are enabled but '{ObsKey.LANGUAGE.value}' is not in "
                    f"observation_tokenizer.observation_keys: {token_obs_keys}"
                )

            # Add tokenized observations key if any observations are tokenized
            if tokenized_any:
                available_keys.add(TOKENIZED_OBSERVATIONS_KEY)
                available_keys.add(IS_PAD_OBSERVATION_KEY)
        else:
            # No tokenization - all observation keys are available as raw
            for obs_key, obs_meta in obs_space.observations_metadata.items():
                available_keys.add(obs_key)

        # Validate encoders
        configured_encoder_inputs = set()
        for encoder_name, encoder in pipeline.encoders.items():
            encoder: EncodingMixin
            if (
                hasattr(encoder, "backbone_name")
                and "dinov3" in encoder.backbone_name.lower()
            ):
                if (
                    task.dataloader.image_norm_type
                    != ImageNormalizationType.IMAGENET.value
                ):
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
                    f"which are not in observation space. "
                    f"Available keys: {available_keys}. "
                    f"Please either add them to the observation space or modify encoder configuration."
                )

        uncovered_keys = available_keys - configured_encoder_inputs
        uncovered_keys -= {TOKENIZED_OBSERVATIONS_KEY, IS_PAD_OBSERVATION_KEY}
        if uncovered_keys:
            self.warnings.append(
                f"Observation space contains keys {uncovered_keys} "
                f"but no encoder is configured to process them."
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
