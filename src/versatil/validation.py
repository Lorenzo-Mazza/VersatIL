"""Experiment validation module for validating global experiment configuration."""

from __future__ import annotations

import logging

from versatil.configs.main import MainConfig
from versatil.data.constants import ImageNormalizationType, ObsKey, SampleKey
from versatil.data.metadata import CameraMetadata
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.metrics.base import BaseLoss
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.variational import VariationalAlgorithm
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.decoders import MoEDecoder
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.encoding.encoders.base import EncodingMixin
from versatil.models.encoding.pipeline import EncodingPipeline
from versatil.quantization.strategies import PT2EStrategy, QuantizeApiStrategy


class ExperimentValidationError(Exception):
    """Raised when experiment validation fails."""

    pass


class ExperimentValidator:
    """Validates experiment configuration consistency.

    Validates encoder-observation consistency, decoder-encoder compatibility,
    and loss key validation.
    """

    def __init__(
        self,
        encoding_pipeline: EncodingPipeline,
        algorithm: DecodingAlgorithm,
        decoder: ActionDecoder,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        loss: BaseLoss,
        is_tokenized: bool = False,
        tokenized_obs_keys: set[str] | None = None,
        image_norm_type: str | None = None,
        quantization_config: object | None = None,
    ):
        """Initialize validator with policy components.

        Args:
            encoding_pipeline: The encoding pipeline with configured encoders.
            algorithm: The decoding algorithm (BC, diffusion, etc.).
            decoder: The action decoder architecture.
            observation_space: Task observation space configuration.
            action_space: Task action space configuration.
            loss: The loss module for training.
            is_tokenized: Whether observations are tokenized.
            tokenized_obs_keys: Keys of observations that are tokenized.
            image_norm_type: Image normalization type for DINOv3 validation.
            quantization_config: Optional quantization configuration to validate.
        """
        self.encoding_pipeline = encoding_pipeline
        self.algorithm = algorithm
        self.decoder = decoder
        self.observation_space = observation_space
        self.action_space = action_space
        self.loss = loss
        self.is_tokenized = is_tokenized
        self.tokenized_obs_keys = tokenized_obs_keys or set()
        self.image_norm_type = image_norm_type
        self.quantization_config = quantization_config
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate_all(self, validate_loss_keys: bool = True) -> None:
        """Run all validation checks and raise if any fail.

        Args:
            validate_loss_keys: Whether to validate loss keys against action heads.
        """
        self.validate_encoder_observation_consistency()
        self.validate_decoder_encoder_compatibility()
        self.validate_loss_algorithm_compatibility()
        if validate_loss_keys:
            self.validate_loss_keys()
        if self.quantization_config is not None:
            self.validate_quantization()
        if self.errors:
            error_msg = "\n".join([f"  - {err}" for err in self.errors])
            raise ExperimentValidationError(
                f"Policy validation failed with {len(self.errors)} error(s):\n{error_msg}"
            )
        if self.warnings:
            warning_msg = "\n".join([f"  - {warn}" for warn in self.warnings])
            logging.warning(
                msg=f"Policy validation warnings ({len(self.warnings)}):\n{warning_msg}"
            )

    def validate_encoder_observation_consistency(self) -> None:
        """Validate that encoder inputs match available observations."""
        has_language = (
            ObsKey.LANGUAGE.value in self.observation_space.observations_metadata
        )
        if has_language and not self.is_tokenized:
            self.errors.append(
                "Language observations are enabled but tokenization is disabled. "
                "Language observations require tokenization to be enabled."
            )
            return
        available_keys = set()
        if self.is_tokenized and self.tokenized_obs_keys:
            tokenized_any = False
            for (
                obs_key,
                obs_meta,
            ) in self.observation_space.observations_metadata.items():
                if isinstance(obs_meta, CameraMetadata):
                    available_keys.add(obs_key)
                elif obs_key in self.tokenized_obs_keys:
                    tokenized_any = True
                else:
                    available_keys.add(obs_key)

            if has_language and ObsKey.LANGUAGE.value not in self.tokenized_obs_keys:
                self.errors.append(
                    f"Language observations are enabled but '{ObsKey.LANGUAGE.value}' is not in "
                    f"observation_tokenizer.observation_keys: {self.tokenized_obs_keys}"
                )
            if tokenized_any:
                available_keys.add(SampleKey.TOKENIZED_OBSERVATIONS.value)
                available_keys.add(SampleKey.IS_PAD_OBSERVATION.value)
        else:
            for obs_key in self.observation_space.observations_metadata:
                available_keys.add(obs_key)

        configured_encoder_inputs = set()
        for encoder_name, encoder in self.encoding_pipeline.encoders.items():
            encoder: EncodingMixin
            if (
                hasattr(encoder, "backbone_name")
                and "dinov3" in encoder.backbone_name.lower()
                and self.image_norm_type != ImageNormalizationType.IMAGENET.value
            ):
                self.errors.append(
                    f"Encoder '{encoder_name}' uses DINOv3 backbone which requires "
                    f"ImageNet normalization, but image_norm_type is set to "
                    f"'{self.image_norm_type}'. Set it to 'imagenet'."
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

            for key in input_keys:
                metadata = self.observation_space.observations_metadata.get(key)
                if metadata is None:
                    continue
                error = encoder.validate_input_metadata(key=key, metadata=metadata)
                if error:
                    self.errors.append(f"Encoder '{encoder_name}': {error}")

        uncovered_keys = available_keys - configured_encoder_inputs
        uncovered_keys -= {
            SampleKey.TOKENIZED_OBSERVATIONS.value,
            SampleKey.IS_PAD_OBSERVATION.value,
        }
        if uncovered_keys:
            self.warnings.append(
                f"Observation space contains keys {uncovered_keys} "
                f"but no encoder is configured to process them."
            )

    def validate_decoder_encoder_compatibility(self) -> None:
        """Validate that decoder inputs match encoder outputs."""
        available_features = self.encoding_pipeline.get_features()
        available_feature_names = list(available_features.keys())
        decoder_input_keys = self.decoder.decoder_input.keys

        for expected_feature in decoder_input_keys:
            if expected_feature not in available_feature_names:
                self.errors.append(
                    f"Action decoding network expects input feature '{expected_feature}' "
                    f"but it's not produced by any encoder or fusion layer. "
                    f"Available features: {available_feature_names}"
                )

        self.decoder.decoder_input.validate_feature_types(
            available_features=available_features
        )

        if isinstance(self.decoder, MoEDecoder):
            self._validate_moe_gating_feature(available_feature_names)

    def _validate_moe_gating_feature(self, available_features: list[str]) -> None:
        """Validate MoE gating feature key exists."""
        has_gating = bool(self.decoder.has_gating_network)
        if not has_gating:
            return

        gating_key = self.decoder.gating_feature_key
        if gating_key in available_features:
            return

        if (
            isinstance(self.algorithm, VariationalAlgorithm)
            and gating_key == LatentKey.POSTERIOR_LATENT.value
        ):
            return

        self.errors.append(
            f"MoE decoder gating feature key '{gating_key}' not found. "
            f"Available features from encoding pipeline: {available_features}. "
            f"Algorithm provides latent: {'Yes' if isinstance(self.algorithm, VariationalAlgorithm) else 'No'}."
        )

    def validate_loss_algorithm_compatibility(self) -> None:
        """Validate that no loss module requires action-space targets when the algorithm predicts outside it."""
        if self.algorithm.predicts_in_action_space:
            return
        algorithm_name = type(self.algorithm).__name__
        for name, loss_module in self.loss.loss_modules.items():
            if loss_module.requires_action_space_targets:
                self.errors.append(
                    f"Loss module '{name}' requires action-space targets "
                    f"but algorithm '{algorithm_name}' predicts outside the "
                    f"action space (e.g. velocity or noise). Use a regression "
                    f"loss (MSE/L1) instead."
                )

    def validate_loss_keys(self) -> None:
        """Validate that loss keys reference valid action heads or auxiliary keys."""
        valid_loss_keys: set[str] = set()
        valid_loss_keys.update(self.decoder.action_heads.keys())

        for key, meta in self.action_space.actions_metadata.items():
            if not meta.requires_prediction_head:
                valid_loss_keys.add(key)

        valid_loss_keys.update(self.algorithm.get_auxiliary_output_keys())
        valid_loss_keys.update(self.decoder.get_auxiliary_output_keys())
        required_keys = self.loss.get_required_keys()
        invalid_keys = required_keys - valid_loss_keys
        if invalid_keys:
            self.errors.append(
                f"Loss module references keys {invalid_keys} that are not "
                f"defined in the action space or auxiliary keys. "
                f"Valid loss keys: {valid_loss_keys}. "
                f"Please update your loss configuration or decoder."
            )

    def validate_quantization(self) -> None:
        """Validate that quantization config is present and is a valid strategy."""
        config = self.quantization_config
        if not isinstance(config, (PT2EStrategy, QuantizeApiStrategy)):
            self.warnings.append(
                f"Quantization config is type {type(config).__name__}, "
                f"expected PT2EStrategy or QuantizeApiStrategy. "
                f"This may indicate incorrect Hydra instantiation."
            )


def validate_experiment(config: MainConfig) -> None:
    """Validate experiment configuration from instantiated MainConfig.

    Args:
        config: Instantiated MainConfig with experiment, policy, and task components.

    Raises:
        ExperimentValidationError: If validation fails.
    """
    task = config.task
    policy = config.policy
    dataloader = task.dataloader
    is_tokenized = dataloader.tokenization.tokenize_observations
    tokenized_obs_keys = set()
    if is_tokenized and dataloader.tokenization.observation_tokenizer:
        tokenized_obs_keys = set(
            dataloader.tokenization.observation_tokenizer.observation_keys
        )
    validator = ExperimentValidator(
        encoding_pipeline=policy.encoding_pipeline,
        algorithm=policy.algorithm,
        decoder=policy.decoder,
        observation_space=task.observation_space,
        action_space=task.action_space,
        loss=policy.loss_module,
        is_tokenized=is_tokenized,
        tokenized_obs_keys=tokenized_obs_keys,
        image_norm_type=dataloader.image_norm_type,
        quantization_config=config.quantization,
    )
    validator.validate_all(validate_loss_keys=config.experiment.validate_loss_keys)
