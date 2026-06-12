"""Policy module that handles the sequence of input encoding, output decoding, and loss computation."""

import functools
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn

from versatil.common.dict_of_tensor_mixin import DictOfTensorMixin
from versatil.common.omegaconf_ops import resolve_dict_keys
from versatil.common.tensor_ops import (
    TensorTree,
    clone_tensor_dictionary_with_replacements,
    to_device,
)
from versatil.data.constants import Cameras, MetadataPassthroughSource, SampleKey
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
from versatil.metrics.components import GripperLoss
from versatil.metrics.regularization_context import (
    PolicyForwardContext,
    PolicyGraphInputDomain,
    PolicyRegularizationGraph,
)
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.encoding.pipeline import EncodingPipeline


class Policy(nn.Module):
    """General policy class that orchestrates encoding, decoding, and loss computation."""

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
        regularizers: dict[str, nn.Module] | None = None,
        validate_loss_keys: bool = True,
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
            regularizers: Optional training regularizers. Each module receives a
                ``PolicyRegularizationGraph`` built from the current batch rather
                than the policy itself.
            validate_loss_keys: Deprecated, kept for backwards compatibility.
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
        self.regularizers = nn.ModuleDict(regularizers or {})
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
        """Sorted action keys the policy produces as output."""
        return sorted(self.decoder.get_prediction_output_keys())

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

    def set_gripper_class_weights(self, pos_weight: torch.Tensor | None) -> None:
        """Set positive class weight for GripperLoss components in the loss module.

        Args:
            pos_weight: Tensor with positive class weight for BCE loss, or None to disable.
        """
        for module in self.loss_module.modules():
            if isinstance(module, GripperLoss):
                module.pos_weight = pos_weight

    def forward(
        self, batch: dict[str, dict[str, TensorTree]]
    ) -> dict[str, torch.Tensor]:
        """Forward pass through observation encoding → action decoding.

        Args:
            batch: A batch dictionary containing normalized observations and actions dictionaries. Each is a dict of tensors.

        Returns:
            Decoder output dictionary containing action predictions and any architecture-specific outputs.
        """
        return self._build_forward_context(batch=batch).predictions

    def compute_loss(
        self,
        batch: dict[str, dict[str, TensorTree]],
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
        context = self._build_forward_context(batch=batch)
        output = context.predictions
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
        if self.regularizers:
            loss_output = self._add_regularizer_losses(
                loss_output=loss_output,
                context=context,
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

    def _build_forward_context(
        self,
        batch: dict[str, dict[str, TensorTree]],
    ) -> PolicyForwardContext:
        """Run the policy once and retain graph-boundary tensors.

        Args:
            batch: Normalized training batch with observation and action
                dictionaries. Tensor values are expected to share leading batch
                dimension ``B``.

        Returns:
            Forward context containing raw observations, encoded features,
            decoder-ready features, predictions, and actions from the same graph.
        """
        observation = self._strip_metadata_passthrough_observations(
            observation=batch[SampleKey.OBSERVATION.value]
        )
        actions = batch.get(SampleKey.ACTION.value)
        encoded_features = self._encode_observation(observation=observation)
        decoder_features = self._select_decoder_features(
            observation=observation,
            encoded_features=encoded_features,
        )
        predictions = self.forward_from_decoder_features(
            decoder_features=decoder_features,
            actions=actions,
        )
        return PolicyForwardContext(
            observation=observation,
            encoded_features=encoded_features,
            decoder_features=decoder_features,
            predictions=predictions,
            actions=actions,
        )

    def _add_regularizer_losses(
        self,
        loss_output: LossOutput,
        context: PolicyForwardContext,
    ) -> LossOutput:
        """Add configured regularizer losses to the main training loss.

        Args:
            loss_output: Loss output produced by the main configured loss module.
            context: Forward context from the same training batch.

        Returns:
            Loss output whose total loss includes every regularizer loss. Component
            losses are namespaced as ``"{regularizer_name}/{component_name}"``.
        """
        total_loss = loss_output.total_loss
        component_losses = dict(loss_output.component_losses)
        metadata = dict(loss_output.metadata)
        regularization_graph = self._build_regularization_graph(context=context)
        for regularizer_name, regularizer in self.regularizers.items():
            regularizer_output = regularizer(graph=regularization_graph)
            total_loss = total_loss + regularizer_output.total_loss
            for (
                component_name,
                component_value,
            ) in regularizer_output.component_losses.items():
                component_losses[f"{regularizer_name}/{component_name}"] = (
                    component_value
                )
            metadata.update(regularizer_output.metadata)
        return LossOutput(
            total_loss=total_loss,
            component_losses=component_losses,
            metadata=metadata,
        )

    def _build_regularization_graph(
        self,
        context: PolicyForwardContext,
    ) -> PolicyRegularizationGraph:
        """Create a policy-owned graph re-entry interface for regularizers.

        Args:
            context: Forward context from the current training batch.

        Returns:
            Batch-local graph object. Regularizers use it to re-run the same
            policy operation order with selected tensor replacements, without
            receiving the policy or its submodules directly. Every re-entry
            replays the RNG snapshot captured here, so stochastic sampling
            (algorithm timesteps, noise, dropout masks) is identical across
            perturbed forwards of the same graph.
        """
        cpu_rng_state, device_rng_state = self._capture_rng_snapshot()
        return PolicyRegularizationGraph(
            context=context,
            training=self.training,
            default_output_keys=sorted(self.decoder.get_loss_output_keys()),
            evaluate_with_replacements=functools.partial(
                self._evaluate_regularization_graph,
                cpu_rng_state=cpu_rng_state,
                device_rng_state=device_rng_state,
            ),
            deterministic_scope=self._decoder_deterministic_scope,
            action_metadata=self.action_space.actions_metadata,
        )

    def _capture_rng_snapshot(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Capture CPU and policy-device RNG states for graph replay."""
        cpu_rng_state = torch.get_rng_state()
        device_rng_state = (
            torch.cuda.get_rng_state(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        return cpu_rng_state, device_rng_state

    @contextmanager
    def _replayed_rng_scope(
        self,
        cpu_rng_state: torch.Tensor,
        device_rng_state: torch.Tensor | None,
    ) -> Iterator[None]:
        """Restore an RNG snapshot for one graph re-entry.

        Inside the scope, random draws (algorithm timestep and noise sampling,
        dropout masks) replay the snapshot, so perturbed evaluations of one
        regularization graph differ only through the replaced tensors. The
        outer RNG stream is restored on exit.
        """
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(cpu_rng_state)
            if device_rng_state is not None:
                torch.cuda.set_rng_state(device_rng_state, device=self.device)
            yield

    def _evaluate_regularization_graph(
        self,
        input_domain: str,
        context: PolicyForwardContext,
        replacements: dict[str, torch.Tensor],
        cpu_rng_state: torch.Tensor,
        device_rng_state: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the policy graph from a named boundary with replacements.

        Args:
            input_domain: Boundary to replace. Valid values are
                ``"observation"``, ``"encoded_features"``, and
                ``"decoder_features"``.
            context: Forward context used as the base graph state.
            replacements: Tensor replacements for keys at ``input_domain``. Each
                replacement must match the shape of its context tensor.
            cpu_rng_state: CPU RNG snapshot replayed for this evaluation.
            device_rng_state: Policy-device RNG snapshot, or ``None`` on CPU.

        Returns:
            Prediction dictionary from the re-entered policy graph. The operation
            order is still owned by ``Policy``: observation replacements re-run
            encoding, encoded-feature replacements re-run feature selection and
            decoding, and decoder-feature replacements re-run only the algorithm
            and decoder boundary.
        """
        domain = PolicyGraphInputDomain(input_domain)
        with self._replayed_rng_scope(
            cpu_rng_state=cpu_rng_state,
            device_rng_state=device_rng_state,
        ):
            match domain:
                case PolicyGraphInputDomain.OBSERVATION:
                    observation = clone_tensor_dictionary_with_replacements(
                        values=context.observation,
                        replacements=replacements,
                    )
                    return self.forward_from_observation(
                        observation=observation,
                        actions=context.actions,
                    )
                case PolicyGraphInputDomain.ENCODED_FEATURES:
                    encoded_features = clone_tensor_dictionary_with_replacements(
                        values=context.encoded_features,
                        replacements=replacements,
                    )
                    return self.forward_from_encoded_features(
                        observation=context.observation,
                        encoded_features=encoded_features,
                        actions=context.actions,
                    )
                case PolicyGraphInputDomain.DECODER_FEATURES:
                    decoder_features = clone_tensor_dictionary_with_replacements(
                        values=context.decoder_features,
                        replacements=replacements,
                    )
                    return self.forward_from_decoder_features(
                        decoder_features=decoder_features,
                        actions=context.actions,
                    )

    @contextmanager
    def _decoder_deterministic_scope(
        self,
        enabled: bool,
    ) -> Iterator[None]:
        """Temporarily run decoder modules in eval mode.

        Args:
            enabled: If false, leaves decoder training states unchanged.

        Yields:
            ``None`` while decoder stochastic layers are disabled. Original
            training states are restored after the scope exits.
        """
        if not enabled:
            yield
            return
        modules = list(self.decoder.modules())
        training_states = [module.training for module in modules]
        for module in modules:
            module.eval()
        try:
            yield
        finally:
            for module, training_state in zip(modules, training_states):
                module.train(mode=training_state)

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
        batch: dict[str, dict[str, TensorTree]],
        predictions: dict[str, torch.Tensor],
    ) -> dict[str, TensorTree]:
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
        observation: dict[str, TensorTree],
    ) -> dict[str, TensorTree]:
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
        observation: dict[str, TensorTree],
    ) -> dict[str, TensorTree]:
        """Build the feature dictionary passed into the decoding algorithm.

        The policy boundary is responsible only for collecting model inputs from
        the encoding pipeline and explicitly requested raw observation tensors.
        Algorithms add their own control tensors later, such as diffusion/flow
        timesteps or variational latents.
        """
        encoded_features = self._encode_observation(observation=observation)
        return self._select_decoder_features(
            observation=observation,
            encoded_features=encoded_features,
        )

    def _encode_observation(
        self,
        observation: dict[str, TensorTree],
    ) -> dict[str, torch.Tensor]:
        """Encode normalized observations through the encoding pipeline.

        Args:
            observation: Normalized observation dictionary. Image tensors are
                typically ``(B, T, C, H, W)`` or ``(B, C, H, W)`` and vector
                observations are typically ``(B, T, D)`` or ``(B, D)``.

        Returns:
            Encoded feature dictionary produced by the configured pipeline.
        """
        return self.encoding_pipeline(observation)

    def _select_decoder_features(
        self,
        observation: dict[str, TensorTree],
        encoded_features: dict[str, torch.Tensor],
    ) -> dict[str, TensorTree]:
        """Select the feature dictionary consumed by the decoder.

        Args:
            observation: Normalized raw observation tensors keyed by observation
                name.
            encoded_features: Encoding-pipeline outputs keyed by feature name.

        Returns:
            Decoder input dictionary containing exactly the keys requested by
            ``decoder.decoder_input.keys`` plus matching padding-mask tensors
            named ``"{feature_key}_padding_mask"`` when available.

        Raises:
            ValueError: If the decoder requests a key that is not available from
                raw observations or encoded features.
        """
        available_features = {**observation, **encoded_features}
        selected_features: dict[str, TensorTree] = {}
        missing_keys: list[str] = []
        for key in self.decoder.decoder_input.keys:
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

    def forward_from_decoder_features(
        self,
        decoder_features: dict[str, TensorTree],
        actions: dict[str, TensorTree] | None,
    ) -> dict[str, torch.Tensor]:
        """Run algorithm and decoder from decoder-ready features.

        Args:
            decoder_features: Feature dictionary satisfying
                ``decoder.decoder_input``. Values are batched as ``(B, ...)``.
            actions: Normalized action dictionary used by training algorithms, or
                ``None`` for action-free evaluations.

        Returns:
            Prediction dictionary from ``algorithm.forward``.
        """
        return self.algorithm.forward(
            features=decoder_features,
            actions=actions,
            network=self.decoder,
        )

    def forward_from_encoded_features(
        self,
        observation: dict[str, TensorTree],
        encoded_features: dict[str, torch.Tensor],
        actions: dict[str, TensorTree] | None,
    ) -> dict[str, torch.Tensor]:
        """Run the downstream policy graph from encoded features.

        Args:
            observation: Normalized raw observations. Raw observation tensors may
                still be selected if the decoder directly requests them.
            encoded_features: Replacement-capable encoded feature dictionary.
                Values are batched as ``(B, ...)``.
            actions: Normalized action dictionary used by training algorithms, or
                ``None`` for action-free evaluations.

        Returns:
            Prediction dictionary after decoder feature selection and
            ``algorithm.forward``.
        """
        decoder_features = self._select_decoder_features(
            observation=observation,
            encoded_features=encoded_features,
        )
        return self.forward_from_decoder_features(
            decoder_features=decoder_features,
            actions=actions,
        )

    def forward_from_observation(
        self,
        observation: dict[str, TensorTree],
        actions: dict[str, TensorTree] | None,
    ) -> dict[str, torch.Tensor]:
        """Run the full policy graph from normalized observations.

        Args:
            observation: Normalized observation dictionary with tensors sharing
                leading batch dimension ``B``.
            actions: Normalized action dictionary used by training algorithms, or
                ``None`` for action-free evaluations.

        Returns:
            Prediction dictionary after encoding, feature selection, and
            ``algorithm.forward``.
        """
        encoded_features = self._encode_observation(observation=observation)
        return self.forward_from_encoded_features(
            observation=observation,
            encoded_features=encoded_features,
            actions=actions,
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

    def get_vision_encoder_modules(self) -> dict[str, nn.Module]:
        """Get vision encoder modules that can produce spatial feature maps for explainability.

        Supports the following encoder types:
        - SpatialRGBEncoder (RGB): Has 'backbone' attribute
        - SpatialDepthEncoder: Has 'backbone' attribute
        - ConditionalCNNEncoder (FiLM): Has 'layer4' attribute
        - DFormerEncoder: Has 'stages' attribute
        - GeometricRGBDEncoder: Has 'attention_block' attribute

        Returns:
            Dictionary mapping encoder names to their encoder instances.
            Only includes encoders that produce spatial feature maps.

        Raises:
            RuntimeError: If no compatible vision encoders are found
        """
        vision_encoders = {}

        def is_vision_encoder(encoder: nn.Module) -> bool:
            """Return whether an encoder exposes a vision target layer."""
            # TIMM-based encoders (SpatialRGBEncoder, SpatialDepthEncoder)
            if hasattr(encoder, "backbone"):
                return True
            # DFormer-based encoders
            if hasattr(encoder, "stages"):
                return True
            # FiLM-conditioned ResNet (ConditionalCNNEncoder)
            if hasattr(encoder, "layer4"):
                return True
            # LightGeometric encoder
            return bool(hasattr(encoder, "attention_block"))

        for encoder_name, encoder in self.encoding_pipeline.encoders.items():
            if is_vision_encoder(encoder):
                vision_encoders[encoder_name] = encoder

        for (
            encoder_name,
            encoder,
        ) in self.encoding_pipeline.conditional_encoders.items():
            if is_vision_encoder(encoder):
                vision_encoders[encoder_name] = encoder

        if not vision_encoders:
            raise RuntimeError(
                "No compatible vision encoders found in the encoding pipeline. "
                "Explainer requires encoders that produce spatial feature maps "
                "(SpatialRGBEncoder, SpatialDepthEncoder, ConditionalCNNEncoder, DFormerEncoder, GeometricRGBDEncoder). "
                "Available encoders: "
                + str(
                    list(self.encoding_pipeline.encoders.keys())
                    + list(self.encoding_pipeline.conditional_encoders.keys())
                )
            )

        return vision_encoders

    def get_gradcam_target_layers(self, encoder_name: str) -> list[nn.Module]:
        """Get target layers for GradCAM from a specific vision encoder.

        Supports different encoder architectures:
        - TIMM backbones (SpatialRGBEncoder, SpatialDepthEncoder): Returns last stage
        - ConditionalCNNEncoder (FiLM): Returns last block of layer4
        - DFormerEncoder: Returns last stage
        - GeometricRGBDEncoder: Returns attention block

        Args:
            encoder_name: Name of the encoder in the encoding pipeline

        Returns:
            List of target layers suitable for GradCAM

        Raises:
            ValueError: If encoder doesn't exist or is not a vision encoder
            RuntimeError: If encoder architecture is not supported
        """
        vision_encoders = self.get_vision_encoder_modules()
        if encoder_name not in vision_encoders:
            raise ValueError(
                f"Encoder '{encoder_name}' not found or not a vision encoder. "
                f"Available vision encoders: {list(vision_encoders.keys())}"
            )

        encoder = vision_encoders[encoder_name]

        # TIMM-based encoders (SpatialRGBEncoder, SpatialDepthEncoder, FlatRGBEncoder)
        if hasattr(encoder, "backbone"):
            backbone = encoder.backbone
            if hasattr(backbone, "layer4"):
                return [backbone.layer4]
            elif hasattr(backbone, "stages") and len(backbone.stages) > 0:
                return [backbone.stages[-1]]
            else:
                raise RuntimeError(
                    f"Encoder '{encoder_name}' has backbone but structure not recognized. "
                    f"Backbone type: {type(backbone).__name__}"
                )

        # DFormer-based encoders
        if hasattr(encoder, "stages") and len(encoder.stages) > 0:
            return [encoder.stages[-1]]

        # FiLM-conditioned ResNet (ConditionalCNNEncoder)
        if hasattr(encoder, "layer4") and len(encoder.layer4) > 0:
            return [encoder.layer4[-1]]

        # LightGeometric encoder
        if hasattr(encoder, "attention_block"):
            return [encoder.attention_block]

        raise RuntimeError(
            f"Encoder '{encoder_name}' architecture not supported for GradCAM. "
            f"Encoder type: {type(encoder).__name__}. "
            f"Supported types: SpatialRGBEncoder, SpatialDepthEncoder, ConditionalCNNEncoder, "
            f"DFormerEncoder, GeometricRGBDEncoder."
        )

    def get_camera_to_encoder_mapping(self) -> dict[str, str]:
        """Get mapping from camera keys to their corresponding vision encoder names.

        This method only returns mappings for valid camera keys (as defined by the Cameras enum)
        that are processed by vision encoders capable of producing spatial feature maps.
        Non-camera observations like proprioceptive state or language are excluded.

        Returns:
            Dictionary mapping camera keys to vision encoder names

        Raises:
            RuntimeError: If no camera-to-encoder mappings are found
        """
        valid_camera_keys = {cam.value for cam in Cameras}
        vision_encoders = self.get_vision_encoder_modules()

        mapping = {}
        for encoder_name, encoder in vision_encoders.items():
            if not hasattr(encoder, "input_specification"):
                continue

            input_keys = encoder.input_specification.keys
            for key in input_keys:
                # Only include valid camera keys, not proprioceptive/language keys
                if key in valid_camera_keys:
                    mapping[key] = encoder_name
        # TODO: Here we ignore the case of multiple encoders with the same camera key.
        #  Although unlikely, we should safeguard against it in the future.

        if not mapping:
            raise RuntimeError(
                f"No camera-to-encoder mappings found. "
                f"Valid camera keys: {valid_camera_keys}. "
                f"Vision encoders: {list(vision_encoders.keys())}"
            )

        return mapping
