"""Policy module that handles the sequence of input encoding, output decoding, and loss computation."""

import torch
import torch.nn as nn

from versatil.common.dict_of_tensor_mixin import DictOfTensorMixin
from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.common.tensor_ops import to_device
from versatil.data.constants import MetadataPassthroughSource, SampleKey
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.processing.transform import (
    detokenize_actions,
    normalize_observation,
    tokenize_observation,
    unnormalize_actions,
)
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import Tokenizer
from versatil.metrics.base import BaseLoss, LossOutput
from versatil.metrics.losses.gripper import GripperLoss
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.encoding.pipeline import EncodingPipeline


def build_algorithm_features(
    observation: dict[str, torch.Tensor],
    encoding_pipeline: EncodingPipeline,
    decoder: ActionDecoder,
) -> dict[str, torch.Tensor]:
    """Encode observations and select the features the decoder declared.

    Merges raw observations with encoding-pipeline outputs, then filters to
    the decoder's ``decoder_input.keys`` allowlist plus their padding masks.
    Both ``Policy`` and ``ExportablePolicy`` must build features through this
    function so that exported models see exactly the training-time inputs.

    Raises:
        ValueError: If the decoder requests keys that are neither raw
            observations nor encoding-pipeline outputs.
    """
    encoded_features = encoding_pipeline(observation)
    available_features = {**observation, **encoded_features}
    selected_features: dict[str, torch.Tensor] = {}
    missing_keys: list[str] = []
    for key in decoder.decoder_input.keys:
        if key not in available_features:
            missing_keys.append(key)
            continue
        selected_features[key] = available_features[key]
        padding_key = f"{key}_{EncoderOutputKeys.PADDING_MASK.value}"
        if padding_key in available_features:
            selected_features[padding_key] = available_features[padding_key]

    if missing_keys:
        raise ValueError(
            f"Decoder requested input keys {missing_keys}, but they were not "
            f"available from raw observations or the encoding pipeline. "
            f"Available keys: {sorted(available_features.keys())}."
        )
    return selected_features


class Policy(nn.Module):
    """General policy class that orchestrates observation encoding, action decoding, and loss computation."""

    def __init__(
        self,
        encoding_pipeline: EncodingPipeline,
        algorithm: DecodingAlgorithm,
        decoder: ActionDecoder,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        prediction_horizon: int,
        observation_horizon: int,
        loss: BaseLoss,
        device: str,
        metadata_passthrough: dict[str, dict[str, str]] | None = None,
    ) -> None:
        """Initialize policy.

        Args:
            encoding_pipeline: Observation encoding pipeline.
            algorithm: Decoding algorithm (diffusion, flow matching, etc.).
            decoder: Action decoder architecture.
            observation_space: Observation space configuration.
            action_space: Action space configuration.
            prediction_horizon: Number of future actions to predict.
            observation_horizon: Number of past observations to condition on.
            loss: Loss module for training.
            device: Device to run on.
            metadata_passthrough: Mapping from source dictionaries to metadata
                keys for logging/visualization.
        """
        super().__init__()
        self.encoding_pipeline = encoding_pipeline
        self.algorithm = algorithm
        self.decoder = decoder
        self.observation_space = observation_space
        self.action_space = action_space
        self.prediction_horizon = prediction_horizon
        self.observation_horizon = observation_horizon
        self.loss_module = loss
        self.device = torch.device(device)
        self.metadata_passthrough = self._resolve_metadata_passthrough(
            metadata_passthrough=metadata_passthrough
        )
        self.normalizer: LinearNormalizer = LinearNormalizer()
        self.tokenizer = None  # Set later via set_tokenizer()
        self.denoising_thresholds = DictOfTensorMixin()

    @property
    def input_keys(self) -> list[str]:
        """Sorted observation keys the policy expects as input."""
        keys = self._encoder_observation_keys()
        if (
            self.tokenizer is not None
            and self.tokenizer.observation_tokenizer is not None
        ):
            keys.add(SampleKey.TOKENIZED_OBSERVATIONS.value)
            keys.add(SampleKey.IS_PAD_OBSERVATION.value)
        return sorted(keys)

    @property
    def output_keys(self) -> list[str]:
        """Action keys the policy produces, in action-space metadata order."""
        return list(self.decoder.get_prediction_output_keys())

    def _decoder_observation_keys(self) -> set[str]:
        """Return raw observation keys directly consumed by the decoder."""
        observation_keys = set(self.observation_space.observations_metadata.keys())
        observation_keys.update(
            {
                SampleKey.TOKENIZED_OBSERVATIONS.value,
                SampleKey.IS_PAD_OBSERVATION.value,
            }
        )
        return set(self.decoder.decoder_input.keys).intersection(observation_keys)

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        """Set normalizer for observations and actions."""
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.normalizer.to(self.device)
        self.decoder.set_normalizer(self.normalizer)

    def set_tokenizer(self, tokenizer: Tokenizer | None) -> None:
        """Set tokenizer and pass it to the decoder."""
        self.tokenizer = tokenizer
        self.encoding_pipeline.set_tokenizer(tokenizer)
        self.decoder.set_tokenizer(tokenizer)

    def set_denoising_thresholds(self, thresholds: dict[str, float]) -> None:
        """Set the denoising thresholds from training data.

        Args:
            thresholds: Dictionary mapping observation keys to their denoising thresholds.
                May be empty for precomputed actions.

        Note:
            These thresholds are computed from the dataset's action processor and stored
            via DictOfTensorMixin to persist through checkpointing.
        """
        for key, value in thresholds.items():
            self.denoising_thresholds.params_dict[key] = nn.Parameter(
                torch.tensor(value), requires_grad=False
            )

    def get_denoising_thresholds(self) -> dict[str, float]:
        """Return the stored denoising thresholds as plain floats."""
        return {
            key: float(parameter.item())
            for key, parameter in self.denoising_thresholds.params_dict.items()
        }

    def set_gripper_class_weights(self, pos_weight: torch.Tensor | None) -> None:
        """Set positive class weight for GripperLoss components in the loss module.

        Args:
            pos_weight: Tensor with positive class weight for BCE loss, or None to disable.
        """
        for module in self.loss_module.modules():
            if isinstance(module, GripperLoss):
                module.pos_weight = pos_weight

    def forward(
        self, batch: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Forward pass through observation encoding → action decoding.

        Args:
            batch: A batch dictionary containing normalized observations and actions dictionaries. Each is a dict of tensors.

        Returns:
            Decoder output dictionary containing action predictions and any architecture-specific outputs.
        """
        observation = self._strip_metadata_passthrough_observations(
            observation=batch[SampleKey.OBSERVATION.value]
        )
        actions = batch.get(SampleKey.ACTION.value)
        features = self._build_algorithm_features(observation=observation)
        return self.algorithm.forward(
            features=features, actions=actions, network=self.decoder
        )

    def compute_loss(
        self,
        batch: dict[str, dict[str, torch.Tensor]],
    ) -> LossOutput:
        """Compute loss using the configured loss module.

        The algorithm determines what the regression targets are via
        ``get_targets``. For BC this is the ground-truth actions; for
        flow matching it is the target velocity field; for diffusion it
        depends on the prediction type (noise, sample, or velocity).

        Args:
            batch: Batch dictionary containing observations and actions

        Returns:
            LossOutput with total loss and component losses
        """
        output = self.forward(batch)
        ground_truth_actions = batch[SampleKey.ACTION.value]
        targets = self.algorithm.get_targets(
            algorithm_output=output,
            ground_truth_actions=ground_truth_actions,
        )
        loss_output = self.loss_module(
            predictions=output,
            targets=targets,
            is_pad=ground_truth_actions.get(SampleKey.IS_PAD_ACTION.value),
        )
        metadata = self._collect_metadata_passthrough(
            batch=batch,
            predictions=output,
        )
        if not metadata:
            return loss_output
        return LossOutput(
            total_loss=loss_output.total_loss,
            component_losses=loss_output.component_losses,
            metadata={**loss_output.metadata, **metadata},
        )

    def _resolve_metadata_passthrough(
        self,
        metadata_passthrough: dict[str, dict[str, str]] | None,
    ) -> dict[str, dict[str, str]]:
        """Resolve and validate configured metadata passthrough mappings."""
        if metadata_passthrough is None:
            return {}
        resolved_mapping = resolve_dict_keys(dict(metadata_passthrough))
        normalized_mapping: dict[str, dict[str, str]] = {}
        for source_name, keys_mapping in resolved_mapping.items():
            valid_sources = {source.value for source in MetadataPassthroughSource}
            if source_name not in valid_sources:
                raise ValueError(
                    f"Unknown metadata passthrough source '{source_name}'. "
                    f"Valid sources: {sorted(valid_sources)}."
                )
            normalized_mapping.setdefault(source_name, {}).update(keys_mapping)
        return normalized_mapping

    def _collect_metadata_passthrough(
        self,
        batch: dict[str, dict[str, torch.Tensor]],
        predictions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Collect configured tensors from training dictionaries into metadata."""
        source_dictionaries = {
            MetadataPassthroughSource.OBSERVATION.value: batch.get(
                SampleKey.OBSERVATION.value, {}
            ),
            MetadataPassthroughSource.ACTION.value: batch.get(
                SampleKey.ACTION.value, {}
            ),
            MetadataPassthroughSource.PREDICTION.value: predictions,
        }
        metadata = {}
        for source_name, keys_mapping in self.metadata_passthrough.items():
            source_dictionary = source_dictionaries[source_name]
            for source_key, metadata_key in keys_mapping.items():
                if source_key in source_dictionary:
                    metadata[metadata_key] = source_dictionary[source_key]
        return metadata

    def _strip_metadata_passthrough_observations(
        self,
        observation: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Remove metadata-only observation keys before model encoding."""
        observation_metadata = self.metadata_passthrough.get(
            MetadataPassthroughSource.OBSERVATION.value, {}
        )
        if not observation_metadata:
            return observation
        metadata_keys = set(observation_metadata) - self._encoder_observation_keys()
        if not metadata_keys:
            return observation
        return {
            key: value for key, value in observation.items() if key not in metadata_keys
        }

    def _encoder_observation_keys(self) -> set[str]:
        """Return observation keys explicitly consumed by encoders."""
        keys: set[str] = set()
        for encoder in self.encoding_pipeline.encoders.values():
            keys.update(encoder.input_specification.keys)
        for encoder in self.encoding_pipeline.conditional_encoders.values():
            keys.update(encoder.input_specification.keys)
        keys.update(self._decoder_observation_keys())
        return keys

    def _build_algorithm_features(
        self,
        observation: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Build the feature dictionary passed into the decoding algorithm.

        The policy boundary is responsible only for collecting model inputs from
        the encoding pipeline and explicitly requested raw observation tensors.
        Algorithms add their own control tensors later, such as diffusion/flow
        timesteps or variational latents.
        """
        return build_algorithm_features(
            observation=observation,
            encoding_pipeline=self.encoding_pipeline,
            decoder=self.decoder,
        )

    def predict_action(
        self,
        obs_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Predict actions from observations.

        Args:
            obs_dict: Dictionary of observation tensors

        Returns:
            Predicted actions (on same device as policy)
        """
        obs_dict = to_device(obs_dict, device=self.device)
        normalized_observation = normalize_observation(
            observation=obs_dict,
            normalizer=self.normalizer,
            observation_space=self.observation_space,
        )
        normalized_observation = self._strip_metadata_passthrough_observations(
            observation=normalized_observation
        )
        if (
            self.tokenizer is not None
            and self.tokenizer.observation_tokenizer is not None
        ):
            normalized_observation = tokenize_observation(
                observation=normalized_observation,
                obs_tokenizer=self.tokenizer.observation_tokenizer,
            )
        features = self._build_algorithm_features(observation=normalized_observation)
        predictions = self.algorithm.predict(features=features, network=self.decoder)
        if DecoderOutputKey.PREDICTED_ACTION_TOKENS.value in predictions:
            action_tokens = predictions[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
            if self.tokenizer is None or self.tokenizer.action_tokenizer is None:
                raise RuntimeError(
                    "Action tokenizer not set. Cannot detokenize actions."
                )
            normalized_actions = detokenize_actions(
                action_tokens=action_tokens,
                action_tokenizer=self.tokenizer.action_tokenizer,
                action_space=self.action_space,
            )
            normalized_actions = to_device(normalized_actions, device=self.device)
        else:
            normalized_actions = predictions
        actions = unnormalize_actions(
            normalized_actions=normalized_actions,
            normalizer=self.normalizer,
            action_space=self.action_space,
        )
        return actions
