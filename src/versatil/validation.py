"""Experiment validation module for validating global training experiment configuration."""

from __future__ import annotations

import logging

from versatil.configs.main import MainConfig
from versatil.configs.training import TrainingConfig
from versatil.data.constants import (
    CameraModality,
    ImageNormalizationType,
    ObsKey,
    SampleKey,
)
from versatil.data.metadata import CameraMetadata, ObservationMetadata
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.metrics.base import BaseLoss, _merge_weights
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.variational import VariationalAlgorithm
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.decoders import MoEDecoder
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.vlm import VLMBackboneDecoderMixin
from versatil.models.encoding.encoders.base import EncodingMixin
from versatil.models.encoding.pipeline import EncodingPipeline
from versatil.models.feature_meta import FeatureMetadata, FeatureType
from versatil.models.input_specification import InputSpecification
from versatil.quantization.workflows.base import BaseQuantizationWorkflow
from versatil.training.constants import OPTIMIZER_UNMATCHED_GROUPS_NAME


class ExperimentValidationError(Exception):
    """Raised when experiment validation fails."""

    pass


class ExperimentValidator:
    """Validates experiment configuration consistency.

    Validates encoder-observation consistency, decoder-encoder compatibility,
    and loss key validation.
    When ``training_config.stages`` is provided, the validator also owns checks required by multi-stage training:
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
        quantization_config: BaseQuantizationWorkflow | None = None,
        training_config: TrainingConfig | None = None,
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
            image_norm_type: RGB image normalization type for pretrained vision
                encoder validation.
            quantization_config: Optional quantization configuration to validate.
            training_config: Training configuration. When provided with
                ``stages``, the validator also checks stage ordering, optimizer
                group references, and loss-weight paths against the loss tree.
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
        self.training_config = training_config
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
        if self.training_config is not None and self.training_config.stages:
            self.validate_stage_ordering()
            self.validate_stage_group_references()
            self.validate_stage_loss_paths()
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

    @staticmethod
    def _encoder_image_model_identifier(encoder: EncodingMixin) -> str:
        """Return normalized model identifiers exposed by image encoders."""
        identifiers = [type(encoder).__name__.lower()]
        for attribute_name in ("backbone_name", "model_name", "vision_backbone_id"):
            value = getattr(encoder, attribute_name, None)
            if isinstance(value, str):
                identifiers.append(value.lower())
        return " ".join(identifiers)

    def _validate_encoder_image_normalization(
        self, encoder_name: str, encoder: EncodingMixin
    ) -> None:
        """Validate image normalization for pretrained vision-family identifiers."""
        identifier = self._encoder_image_model_identifier(encoder)
        if not identifier:
            return
        if (
            "dinosiglip" in identifier
            or "dinov2_siglip" in identifier
            or "dinov2siglip" in identifier
            or "prismatic" in identifier
        ):
            self._require_image_norm_type(
                encoder_name=encoder_name,
                model_description="DINOv2+SigLIP vision backbone",
                required_norm_types={ImageNormalizationType.ZERO_TO_ONE.value},
                requirement_description="zero-to-one camera tensors before its internal DINOv2/SigLIP standardization",
                suggestion="'zero_to_one'",
            )
            return
        if "dinov3" in identifier:
            self._require_image_norm_type(
                encoder_name=encoder_name,
                model_description="DINOv3 backbone",
                required_norm_types={ImageNormalizationType.IMAGENET.value},
                requirement_description="ImageNet normalization",
                suggestion="'imagenet'",
            )
        if "clip" in identifier and "siglip" not in identifier:
            self._require_image_norm_type(
                encoder_name=encoder_name,
                model_description="CLIP image backbone",
                required_norm_types={ImageNormalizationType.CLIP.value},
                requirement_description="CLIP normalization",
                suggestion="'clip'",
            )
        if any(
            model_family in identifier
            for model_family in ("siglip", "paligemma", "smolvlm", "idefics")
        ):
            self._require_image_norm_type(
                encoder_name=encoder_name,
                model_description="SigLIP image backbone",
                required_norm_types={ImageNormalizationType.MINUS_ONE_TO_ONE.value},
                requirement_description="minus-one-to-one normalization",
                suggestion="'minus_one_to_one'",
            )

    def _require_image_norm_type(
        self,
        encoder_name: str,
        model_description: str,
        required_norm_types: set[str],
        requirement_description: str,
        suggestion: str,
    ) -> None:
        """Append an error when the configured image normalization is incompatible."""
        if self.image_norm_type in required_norm_types:
            return
        self.errors.append(
            f"Encoder '{encoder_name}' uses {model_description} which requires "
            f"{requirement_description}, but image_norm_type is set to "
            f"'{self.image_norm_type}'. Set it to {suggestion}."
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
        available_keys = self._available_observation_keys()
        if (
            has_language
            and self.is_tokenized
            and self.tokenized_obs_keys
            and ObsKey.LANGUAGE.value not in self.tokenized_obs_keys
        ):
            self.errors.append(
                f"Language observations are enabled but '{ObsKey.LANGUAGE.value}' is not in "
                f"observation_tokenizer.observation_keys: {self.tokenized_obs_keys}"
            )

        configured_encoder_inputs = set()
        for encoder_name, encoder in self.encoding_pipeline.encoders.items():
            encoder: EncodingMixin
            self._validate_encoder_image_normalization(
                encoder_name=encoder_name,
                encoder=encoder,
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

            self._validate_camera_modality_constraints(
                owner_name=f"Encoder '{encoder_name}'",
                input_specification=encoder.input_specification,
            )
            for key in input_keys:
                metadata = self.observation_space.observations_metadata.get(key)
                if metadata is None:
                    continue
                error = encoder.validate_input_metadata(key=key, metadata=metadata)
                if error:
                    self.errors.append(f"Encoder '{encoder_name}': {error}")

        configured_encoder_inputs.update(
            self._validate_decoder_observation_inputs(available_keys=available_keys)
        )

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

    def _available_observation_keys(self) -> set[str]:
        """Return runtime observation keys after optional observation tokenization."""
        if not self.is_tokenized or not self.tokenized_obs_keys:
            return set(self.observation_space.observations_metadata)

        available_keys = set()
        tokenized_any = False
        for obs_key, obs_meta in self.observation_space.observations_metadata.items():
            if isinstance(obs_meta, CameraMetadata):
                available_keys.add(obs_key)
            elif obs_key in self.tokenized_obs_keys:
                tokenized_any = True
            else:
                available_keys.add(obs_key)

        if tokenized_any:
            available_keys.add(SampleKey.TOKENIZED_OBSERVATIONS.value)
            available_keys.add(SampleKey.IS_PAD_OBSERVATION.value)
        return available_keys

    def _observation_feature_metadata(self) -> dict[str, FeatureMetadata]:
        """Build typed metadata for raw observations passed directly to decoders."""
        raw_features = {}
        available_keys = self._available_observation_keys()
        for key, metadata in self.observation_space.observations_metadata.items():
            if key not in available_keys:
                continue
            if isinstance(metadata, CameraMetadata):
                raw_features[key] = FeatureMetadata(
                    key=key,
                    feature_type=FeatureType.SPATIAL.value,
                    dimension=(
                        metadata.channels,
                        metadata.image_height,
                        metadata.image_width,
                    ),
                )
            elif isinstance(metadata, ObservationMetadata):
                raw_features[key] = FeatureMetadata(
                    key=key,
                    feature_type=FeatureType.FLAT.value,
                    dimension=(metadata.dimension,),
                )
        return raw_features

    def _validate_decoder_observation_inputs(
        self, available_keys: set[str]
    ) -> set[str]:
        """Validate observation tensors consumed directly by decoders."""
        configured_inputs = set(self.decoder.decoder_input.keys).intersection(
            available_keys
        )
        if not self.decoder.decoder_input.needs_raw_observations:
            return configured_inputs

        if not isinstance(self.decoder, VLMBackboneDecoderMixin):
            return configured_inputs

        vlm_backbone = self.decoder.vlm_backbone
        input_specification = vlm_backbone.input_specification
        input_keys = input_specification.keys
        configured_inputs.update(input_keys)
        self._validate_encoder_image_normalization(
            encoder_name=f"{type(self.decoder).__name__}.vlm_backbone",
            encoder=vlm_backbone,
        )
        missing = set(input_keys) - available_keys
        missing -= {SampleKey.IS_PAD_OBSERVATION.value}
        if missing:
            self.errors.append(
                f"Decoder '{type(self.decoder).__name__}' requires observation "
                f"keys {missing} which are not in observation space. "
                f"Available keys: {available_keys}."
            )
        for key in input_keys:
            metadata = self.observation_space.observations_metadata.get(key)
            if metadata is None:
                continue
            error = vlm_backbone.validate_input_metadata(key=key, metadata=metadata)
            if error:
                self.errors.append(
                    f"Decoder '{type(self.decoder).__name__}' VLM backbone: {error}"
                )
        self._validate_camera_modality_constraints(
            owner_name=f"Decoder '{type(self.decoder).__name__}' VLM backbone",
            input_specification=input_specification,
        )
        return configured_inputs

    def _validate_camera_modality_constraints(
        self,
        owner_name: str,
        input_specification: InputSpecification,
    ) -> None:
        """Validate semantic camera modality constraints against observation metadata."""
        if (
            not input_specification.exactly_one_camera_modality
            and not input_specification.required_camera_modalities
        ):
            return

        input_keys = list(input_specification.keys)
        keys_by_modality = {modality: [] for modality in CameraModality}
        for key in input_keys:
            metadata = self.observation_space.observations_metadata.get(key)
            if metadata is None or not isinstance(metadata, CameraMetadata):
                continue
            keys_by_modality[metadata.modality].append(key)

        for modality in input_specification.exactly_one_camera_modality:
            matching_keys = keys_by_modality[modality]
            if len(matching_keys) != 1:
                self.errors.append(
                    f"{owner_name} requires exactly one {modality.value} camera "
                    f"input, got {matching_keys} from input keys {input_keys}."
                )
        required_modalities = input_specification.required_camera_modalities
        required_modality_values = [modality.value for modality in required_modalities]
        if required_modalities:
            for modality, matching_keys in keys_by_modality.items():
                if modality not in required_modalities and matching_keys:
                    self.errors.append(
                        f"{owner_name} accepts only camera modalities "
                        f"{required_modality_values}, but input key(s) "
                        f"{matching_keys} have {modality.value} camera metadata."
                    )

        for modality in required_modalities:
            matching_keys = keys_by_modality[modality]
            if not matching_keys:
                self.errors.append(
                    f"{owner_name} requires a {modality.value} camera input, "
                    f"got none from input keys {input_keys}."
                )

    def validate_decoder_encoder_compatibility(self) -> None:
        """Validate that decoder inputs match encoder outputs or raw observations."""
        available_features = self.encoding_pipeline.get_features()
        available_observation_keys = self._available_observation_keys()
        available_feature_names = set(available_features.keys())
        available_input_names = (
            available_feature_names
            | available_observation_keys
            | self.algorithm.injected_feature_keys()
        )
        decoder_input_keys = self.decoder.decoder_input.keys

        for expected_feature in decoder_input_keys:
            if expected_feature not in available_input_names:
                self.errors.append(
                    f"Action decoder expects input key '{expected_feature}' but it "
                    "is neither a raw observation nor produced by any encoder or "
                    "fusion layer. Available raw observations: "
                    f"{sorted(available_observation_keys)}. Available encoded "
                    f"features: {sorted(available_feature_names)}"
                )

        self.decoder.decoder_input.validate_feature_types(
            available_features={
                **available_features,
                **self._observation_feature_metadata(),
            }
        )

        if isinstance(self.decoder, MoEDecoder):
            self._validate_moe_gating_feature(sorted(available_feature_names))

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
        valid_loss_keys.update(self.decoder.get_loss_output_keys())

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
        if config is None:
            return
        if not isinstance(config, BaseQuantizationWorkflow):
            self.errors.append(
                f"Quantization config is type {type(config).__name__}, "
                "expected a quantization workflow with quantization_mode. "
                f"This may indicate incorrect Hydra instantiation."
            )

    def validate_stage_ordering(self) -> None:
        """Validate stage names are unique and start epochs are strictly ordered.

        Note:
            Gaps between stages are allowed. Those epochs fall back to the cached
            base regime at runtime.
        """
        stages = self.training_config.stages
        names = [stage.name for stage in stages]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            self.errors.append(f"Training stage names must be unique: {duplicates}.")
        for previous, current in zip(stages, stages[1:]):
            if current.start_epoch <= previous.start_epoch:
                self.errors.append(
                    "training.stages must be listed in strictly increasing "
                    "start_epoch order."
                )
                break
        for previous, current in zip(stages, stages[1:]):
            if (
                previous.end_epoch is not None
                and previous.end_epoch > current.start_epoch
            ):
                self.errors.append("training.stages intervals must not overlap.")
                break

    def validate_stage_group_references(self) -> None:
        """Validate that every staged group name exists in the optimizer layout.

        The reserved unmatched group is always considered available because
        ``LightningPolicy`` injects it when building optimizer parameter groups.
        """
        available_groups = {OPTIMIZER_UNMATCHED_GROUPS_NAME}
        available_groups.update(
            group.name for group in self.training_config.optimizer.param_groups
        )
        for stage in self.training_config.stages:
            referenced = (
                set(stage.trainable_groups)
                | set(stage.frozen_groups)
                | set(stage.group_lrs)
                | set(stage.group_weight_decays)
            )
            missing = sorted(referenced - available_groups)
            if missing:
                self.errors.append(
                    f"Training stage '{stage.name}' references unknown optimizer "
                    f"groups {missing}. Available groups: "
                    f"{sorted(available_groups)}."
                )

    def validate_stage_loss_paths(self) -> None:
        """Validate every staged ``loss_weights`` patch against the loss tree.

        ``stage.loss_weights`` must be a nested partial tree compatible with
        ``policy.loss_module.weights``. Unknown keys and dict/scalar shape
        mismatches are rejected here so training fails before the callback ever
        mutates runtime state.
        """
        loss_tree = self.loss.weights
        uses_loss_weights = any(
            stage.loss_weights for stage in self.training_config.stages
        )
        if uses_loss_weights and not loss_tree:
            self.errors.append(
                "training.stages declare loss_weights overrides but the loss "
                "module exposes no tunable weights."
            )
            return
        for stage in self.training_config.stages:
            if not stage.loss_weights:
                continue
            try:
                _merge_weights(
                    existing_weights=loss_tree,
                    override_weights=stage.loss_weights,
                )
            except (KeyError, TypeError) as exc:
                self.errors.append(f"Training stage '{stage.name}' loss_weights: {exc}")


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
        training_config=config.training,
    )
    validator.validate_all(validate_loss_keys=config.experiment.validate_loss_keys)
