"""Encoding pipeline for multi-modal observation encoding and fusion."""

import logging

import torch
import torch.nn as nn

from versatil.common.tensor_ops import dict_apply
from versatil.data.constants import RGB_CAMERAS
from versatil.data.metadata import CameraMetadata
from versatil.data.task import ObservationSpace
from versatil.data.tokenization import Tokenizer
from versatil.models.encoding.encoders.base import EncodingMixin
from versatil.models.encoding.encoders.conditional import ConditionalEncoder
from versatil.models.encoding.fusion.base import FusionModule
from versatil.models.feature_meta import FeatureMetadata


class EncodingPipeline(nn.Module):
    """Pipeline that encodes inputs and fuses them hierarchically.

    All encoder outputs and fusion outputs are available in the final
    feature dictionary — nothing is consumed or removed.
    """

    def __init__(
        self,
        encoders: dict[str, EncodingMixin],
        observation_space: ObservationSpace,
        fusion_stages: list[FusionModule] | None = None,
    ):
        """Initialize the encoding pipeline.

        Args:
            encoders: Dictionary of instantiated encoders keyed by name.
            observation_space: Observation space with task metadata.
            fusion_stages: List of instantiated fusion modules.
        """
        super().__init__()
        self.observation_space = observation_space
        self.encoders = nn.ModuleDict()
        self.conditional_encoders = nn.ModuleDict()
        #: all feature_name (both encoders and fusion layers) -> FeatureMetadata
        self._feature_registry: dict[str, FeatureMetadata] = {}
        self._encoder_feature_keys: dict[str, list[str]] = {}
        self._setup_encoders(encoders=encoders)
        self._setup_fusion_modules(fusion_stages=fusion_stages)
        self._validate_pipeline()

    def _setup_encoders(self, encoders: dict[str, EncodingMixin]):
        """Register encoders and set per-camera image sizes from observation space.

        Args:
            encoders: Dictionary of instantiated encoders keyed by name.
        """
        camera_metadata = self.observation_space.cameras
        for encoder_name, encoder in encoders.items():
            encoder.name = encoder_name
            self._set_encoder_image_size(
                encoder_name=encoder_name,
                encoder=encoder,
                camera_metadata=camera_metadata,
            )
            feature_list = encoder.get_output_specification()
            keys = []
            for meta in feature_list:
                prefixed_key = f"{encoder_name}_{meta.key}"
                if prefixed_key in self._feature_registry:
                    raise ValueError(
                        f"Duplicate feature key '{prefixed_key}' from encoder "
                        f"'{encoder_name}'. Already registered."
                    )
                self._feature_registry[prefixed_key] = FeatureMetadata(
                    key=prefixed_key,
                    feature_type=meta.feature_type,
                    dimension=meta.dimension,
                )
                keys.append(meta.key)
            self._encoder_feature_keys[encoder_name] = keys
            if isinstance(encoder, ConditionalEncoder):
                self.conditional_encoders[encoder_name] = encoder
            else:
                self.encoders[encoder_name] = encoder

    def _set_encoder_image_size(
        self,
        encoder_name: str,
        encoder: EncodingMixin,
        camera_metadata: dict[str, CameraMetadata],
    ) -> None:
        """Pass per-camera image size to an encoder from observation space cameras.

        Args:
            encoder_name: Name of the encoder in the pipeline.
            encoder: The encoder instance.
            camera_metadata: Camera metadata from the observation space.

        Raises:
            ValueError: If cameras sharing an encoder have different resolutions.
            ValueError: If encoder has camera keys but none are in the observation space.
        """
        camera_keys = [
            key for key in encoder.input_specification.keys if key in RGB_CAMERAS
        ]
        if not camera_keys:
            return
        image_sizes = set()
        for camera_key in camera_keys:
            if camera_key not in camera_metadata:
                raise ValueError(
                    f"Encoder '{encoder_name}' expects camera key '{camera_key}' "
                    f"but it is not in the observation space cameras: "
                    f"{list(camera_metadata.keys())}"
                )
            metadata = camera_metadata[camera_key]
            image_sizes.add((metadata.image_height, metadata.image_width))
        if len(image_sizes) > 1:
            raise ValueError(
                f"Encoder '{encoder_name}' has cameras with different resolutions: "
                f"{image_sizes}. All cameras sharing a backbone must have the "
                f"same resolution."
            )
        image_height, image_width = image_sizes.pop()
        encoder.set_image_size(image_height=image_height, image_width=image_width)

    def _setup_fusion_modules(self, fusion_stages: list[FusionModule] | None):
        """Resolve input feature names, initialize layers, and register outputs.

        Fusion modules are instantiated by Hydra without knowledge of encoder output
        dimensions. This method resolves their input feature names against the registry,
        calls ``setup()`` to build projection layers, and registers each fusion output
        as a new entry in the feature registry.

        Args:
            fusion_stages: List of instantiated fusion modules, or None.
        """
        self.fusion_stages = nn.ModuleList()
        if fusion_stages:
            for fusion in fusion_stages:
                fusion.input_features = [
                    self._resolve_feature_name(f) for f in fusion.input_features
                ]
                fusion.setup(self._feature_registry)
                output_meta = fusion.get_output_specification()
                self._feature_registry[output_meta.key] = output_meta
                self.fusion_stages.append(fusion)

    def _resolve_feature_name(self, name: str) -> str:
        """Validate that a feature name exists in the registry.

        Args:
            name: Prefixed feature key from fusion config (e.g. ``"left_rgb"``).

        Returns:
            The same name if it exists in the registry.

        Raises:
            ValueError: If the name is not in the registry.
        """
        if name in self._feature_registry:
            return name
        raise ValueError(
            f"Feature '{name}' not found. "
            f"Available: {list(self._feature_registry.keys())}"
        )

    def _validate_pipeline(self) -> None:
        """Validates the entire pipeline configuration."""
        self._validate_conditional_encoders()
        self._validate_fusion_stages()

    def _validate_conditional_encoders(self) -> None:
        """Validate that each conditional encoder's conditioning key exists in non-conditional outputs.

        Raises:
            ValueError: If a conditioning key is not produced by any non-conditional encoder.
        """
        non_conditional_features = set()
        for encoder_name in self.encoders:
            for key in self._encoder_feature_keys[encoder_name]:
                non_conditional_features.add(f"{encoder_name}_{key}")
        for encoder_name, encoder in self.conditional_encoders.items():
            if encoder.condition_key not in non_conditional_features:
                raise ValueError(
                    f"Condition key '{encoder.condition_key}' for encoder "
                    f"'{encoder_name}' not available. "
                    f"Available: {non_conditional_features}"
                )

    def _validate_fusion_stages(self) -> None:
        """Validate that all fusion input features exist in the feature registry.

        Raises:
            ValueError: If a fusion stage references a feature not in the registry.
        """
        for i, fusion in enumerate(self.fusion_stages):
            for input_feat in fusion.input_features:
                if input_feat not in self._feature_registry:
                    raise ValueError(
                        f"Fusion stage {i} expects input feature '{input_feat}' "
                        f"but it's not in the registry. "
                        f"Available: {list(self._feature_registry.keys())}"
                    )

    def _flatten_observation_dict(
        self, observation: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Flatten nested observation dict (identity for flat dicts).

        Args:
            observation: Dictionary of observation tensors, possibly with nested dicts.

        Returns:
            Flattened observation dictionary.
        """
        flattened = {}
        for key, value in observation.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    flattened[nested_key] = nested_value
            else:
                flattened[key] = value
        return flattened

    def forward(
        self,
        observation: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Encode observations through all encoders and apply fusion stages.

        Runs non-conditional encoders first, then conditional encoders (which may
        depend on earlier outputs as conditioning), then fusion modules. All outputs
        are kept in the returned dictionary. Encoders whose input keys are missing
        from the observation are skipped with a warning.

        Args:
            observation: Dictionary mapping observation keys to tensors.

        Returns:
            Dictionary mapping prefixed feature names to tensors. Includes all
            encoder outputs and all fusion outputs.
        """
        flat_obs = self._flatten_observation_dict(observation)
        features = {}
        for encoder_name, encoder in self.encoders.items():
            input_keys = encoder.input_specification.keys
            missing_keys = [key for key in input_keys if key not in flat_obs]
            if missing_keys:
                logging.warning(
                    f"Encoder '{encoder_name}' skipped: missing {missing_keys}"
                )
                continue
            encoder_input = {key: flat_obs[key] for key in input_keys}
            encoded = encoder(encoder_input)
            for feature_key in self._encoder_feature_keys[encoder_name]:
                if feature_key in encoded:
                    features[f"{encoder_name}_{feature_key}"] = encoded[feature_key]

        for encoder_name, encoder in self.conditional_encoders.items():
            input_keys = encoder.input_specification.keys
            missing_keys = [key for key in input_keys if key not in flat_obs]
            if missing_keys:
                logging.warning(
                    f"Conditional encoder '{encoder_name}' skipped: missing {missing_keys}"
                )
                continue
            condition_key = encoder.condition_key
            encoder_input = {key: flat_obs[key] for key in input_keys}
            encoded = encoder(encoder_input, features[condition_key])
            for feature_key in self._encoder_feature_keys[encoder_name]:
                if feature_key in encoded:
                    features[f"{encoder_name}_{feature_key}"] = encoded[feature_key]

        for fusion_module in self.fusion_stages:
            input_features = [
                features[feat_name] for feat_name in fusion_module.input_features
            ]
            features[fusion_module.output_name] = fusion_module(input_features)

        features = dict_apply(
            features, lambda x: x.squeeze(1) if x.ndim > 1 and x.shape[1] == 1 else x
        )
        return features

    def get_feature_names(self) -> list[str]:
        """Get all feature names produced by the encoding pipeline."""
        return list(self._feature_registry.keys())

    def get_features(self) -> dict[str, FeatureMetadata]:
        """Get all feature metadata produced by the encoding pipeline."""
        return dict(self._feature_registry)

    def get_features_to_dimensions(self) -> dict[str, tuple[int, ...]]:
        """Get a dictionary of feature names to dimensions."""
        return {key: meta.dimension for key, meta in self._feature_registry.items()}

    def set_tokenizer(self, tokenizer: Tokenizer | None = None):
        """Set tokenizer and validate vocab sizes.

        Args:
            tokenizer: Tokenizer instance from data pipeline (can be None).

        Raises:
            ValueError: If observation tokenizer vocab size doesn't match encoder vocab sizes.
        """
        all_encoders = {**self.encoders, **self.conditional_encoders}
        for encoder_name, encoder in all_encoders.items():
            if not encoder.input_specification.requires_tokenized:
                continue
            if tokenizer is None or tokenizer.observation_tokenizer is None:
                raise ValueError(
                    f"Encoder '{encoder_name}' requires tokenized input, "
                    f"but no observation tokenizer is available."
                )
            data_vocab_size = tokenizer.observation_tokenizer.vocab_size
            encoder_vocab_size = encoder.get_vocab_size()
            if encoder_vocab_size < data_vocab_size:
                raise ValueError(
                    f"Vocab size mismatch: Observation tokenizer has vocab_size={data_vocab_size}, "
                    f"but encoder '{encoder_name}' only supports vocab_size={encoder_vocab_size}. "
                    f"The encoder's embedding matrix must be at least as large as the tokenizer's vocabulary. "
                    f"Observation tokenizer model: {tokenizer.observation_tokenizer.tokenizer_model}"
                )

    def __repr__(self) -> str:
        """Pretty print the pipeline structure."""
        lines = ["EncodingPipeline(", "  Encoders:"]
        for enc_name, encoder in {**self.encoders, **self.conditional_encoders}.items():
            input_keys = encoder.input_specification.keys
            output_keys = [
                f"{enc_name}_{k}" for k in self._encoder_feature_keys[enc_name]
            ]
            lines.append(
                f"    {enc_name}: {input_keys} -> {output_keys} "
                f"({encoder.__class__.__name__})"
            )
        if self.fusion_stages:
            lines.append("  Fusion stages:")
            for i, fusion in enumerate(self.fusion_stages):
                inputs = ", ".join(fusion.input_features)
                lines.append(
                    f"    {i}: [{inputs}] -> {fusion.output_name} "
                    f"({fusion.__class__.__name__})"
                )
        lines.append(")")
        return "\n".join(lines)
