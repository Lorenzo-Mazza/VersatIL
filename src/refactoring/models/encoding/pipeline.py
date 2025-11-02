import logging
from typing import Any

import torch
import torch.nn as nn
from hydra.utils import instantiate

from refactoring.models.encoding.encoders.base import EncoderOutput
from refactoring.models.encoding.encoders.conditional import ConditionalEncoder


class EncodingPipeline(nn.Module):
    """
    Pipeline that encodes inputs and fuses them hierarchically.

    Feature Consumption:
        When fusion modules combine features, the input features are "consumed" and removed
        from the output. Only the fusion output and non-consumed encoder features are kept.

        Example:
            - Encoders produce: A, B, C, D
            - Fusion 1: B + C → E (consumes B and C)
            - Fusion 2: E + D → F (consumes E and D)
            - Final output: {A, F} (not {A, B, C, D, E, F})

        This prevents feature duplication and ensures the decoder receives only
        semantically meaningful final features.
    """


    def __init__(
            self,
            encoders: dict[str, Any],
            fusion_stages: list[Any] | None = None,
    ):
        """Initializes the encoding pipeline.

                Args:
                    encoders: Dictionary of encoder configurations keyed by name.
                    fusion_stages: List of fusion stage configurations.
                """
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.conditional_encoders = nn.ModuleDict()
        #: encoder name -> `EncoderOutput`
        self.encoder_to_outputs: dict[str, EncoderOutput] = {}
        #: all feature_name (both encoders and fusion layers) -> dimension (for validation)
        self._feature_keys_to_dims: dict[str, int | tuple[int, ...]] = {}
        #: feature names consumed by fusion modules (removed from forward output)
        self._consumed_features: set[str] = set()
        self._setup_encoders(encoders=encoders)
        self._setup_fusion_modules(fusion_stages=fusion_stages)
        self._validate_pipeline()

    @property
    def flat_encoder_feature_names(self) -> set[str]:
        """Get a flat list of all encoder output feature names.

        Note:
            This is needed because some encoders could have several output features (e.g. a VLM).

        Returns:
            Set of prefixed output keys from all encoders.
        """
        result = set()
        for encoder_name, output in self.encoder_to_outputs.items():
            for feature in output.features:
                result.add(f"{encoder_name}_{feature}")
        return result


    def _setup_encoders(self, encoders: dict[str, Any]):
        """Setup and instantiate the encoders."""
        for encoder_name, encoder_config in encoders.items():
            encoder = instantiate(encoder_config)
            encoder.name = encoder_name
            output_info = encoder.get_output_specification()
            self.encoder_to_outputs[encoder_name] = output_info
            for feat_name, dim in output_info.dimensions.items():
                self._feature_keys_to_dims[f"{encoder_name}_{feat_name}"] = dim
            if isinstance(encoder, ConditionalEncoder):
                self.conditional_encoders[encoder_name] = encoder
            else:
                self.encoders[encoder_name] = encoder


    def _setup_fusion_modules(self, fusion_stages: list[dict] | None):
        """Setup and instantiate fusion modules with the correct encoded feature dimensions.

        Note:
            Fusion modules consume their input features. After fusion, only the fusion
            output is available - input features are removed from the feature dictionary.
        """
        self.fusion_stages = nn.ModuleList()
        if fusion_stages:
            for fusion_config in fusion_stages:
                fusion = instantiate(fusion_config)
                fusion.input_features = [self._resolve_feature_name(f) for f in fusion.input_features]
                fusion.setup(self._feature_dims)
                self._feature_keys_to_dims[fusion.output_name] = fusion.get_output_dim()
                # Track consumed features
                for input_feat in fusion.input_features:
                    self._consumed_features.add(input_feat)
                self.fusion_stages.append(fusion)


    def _resolve_feature_name(self, input_specification: str) -> str:
        """Resolve 'encoder_name' or 'encoder_name.output_selector' to actual feature name."""
        if '.' in input_specification:
            # Explicit output_selector: "vlm.language"
            encoder_name, selector = input_specification.split('.', 1)
            output = self.encoder_to_outputs[encoder_name]
            if selector not in output.features:
                raise ValueError(f"Invalid output_selector '{selector}' for '{encoder_name}. Available: {output.features}'")
            else:
                return f"{encoder_name}_{selector}"
        elif input_specification in self.encoder_to_outputs:
            # Otherwise check if it's a single-output encoder, does not need an output selector
            output = self.encoder_to_outputs[input_specification]
            if output.is_multi_output:
                raise ValueError(
                    f"Multi-output encoder '{input_specification}' requires selector. "
                    f"Available: {list(output.features)}"
                )
            return f"{input_specification}_{output.features[0]}"

        else:
            # Direct feature name (from fusion outputs) or invalid name (will be caught by the validation)
            return input_specification


    def _validate_pipeline(self) -> None:
        """Validates the entire pipeline configuration."""
        self._validate_encoder_outputs()
        available_features = self._validate_conditional_encoders()
        available_features = self._validate_fusion_stages(available_features)


    def _validate_encoder_outputs(self) -> None:
        """Validates encoder output keys for duplicates."""
        all_outputs = list(self.flat_encoder_feature_names)
        if len(all_outputs) != len(set(all_outputs)):
            raise ValueError("Duplicate output keys detected from encoders")


    def _validate_conditional_encoders(self) -> set[str]:
        """Validates conditional encoders and builds available features set.

        Returns:
            Set of available features after all non-conditional and conditional encoders.
        """
        available_features = set()
        for encoder_name in self.encoders:
            output = self.encoder_to_outputs[encoder_name]
            for feature in output.features:
                available_features.add(f"{encoder_name}_{feature}")

        for encoder_name, encoder in self.conditional_encoders.items():
            if encoder.condition_key not in available_features:
                raise ValueError(
                    f"Condition key '{encoder.condition_key}' for encoder '{encoder_name}' not available. "
                    f"Available: {available_features}"
                )
            output = self.encoder_to_outputs[encoder_name]
            for feature in output.features:
                available_features.add(f"{encoder_name}_{feature}")

        return available_features


    def _validate_fusion_stages(self, available_features: set[str]) -> set[str]:
        """Validates fusion stages' input features.

        Args:
            available_features: Current set of available features.

        Returns:
            Updated set of available features including fusion outputs.
        """
        for i, fusion in enumerate(self.fusion_stages):
            for input_feat in fusion.input_features:
                if input_feat not in available_features:
                    raise ValueError(
                        f"Fusion stage {i} expects input feature '{input_feat}' "
                        f"but it's not produced by any encoder or previous fusion. "
                        f"Available features: {available_features}"
                    )
            available_features.add(fusion.output_name)
        return available_features

    def _flatten_observation_dict(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Flatten nested observation dict (identity for flat dicts).

        Args:
            observation: Dictionary of observation tensors

        Returns:
            Flattened observation dictionary
        """
        return observation


    def forward(
            self,
            observation: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Encode observations and fuse features hierarchically.

        Args:
            observation: Dictionary indexed by observation keys (defined in the task observation space), containing torch tensors.

        Returns:
            Dictionary of final features. Features consumed by fusion modules are NOT included.
            Only encoder features not consumed by fusion, and fusion outputs are returned.

        Note:
            Fusion modules consume their input features. For example, if fusion combines
            features A and B into C, the returned dictionary will contain C but not A or B.
        """
        flat_obs = self._flatten_observation_dict(observation)
        features = {}

        for encoder_name, encoder in self.encoders.items():
            input_keys = encoder.input_specification.keys
            missing_keys = [key for key in input_keys if key not in flat_obs]
            if missing_keys:
                logging.warning(f"Encoder '{encoder_name}' skipped: missing {missing_keys}")
                continue
            encoder_input = {key: flat_obs[key] for key in input_keys}
            encoded = encoder(encoder_input)
            output_specification = self.encoder_to_outputs[encoder_name]
            for feature_name in output_specification.features:
                if feature_name in encoded:
                    features[f"{encoder_name}_{feature_name}"] = encoded[feature_name]

        # Conditional encoders are always applied after non-conditional ones, because they may require their outputs as conditioning features.
        for encoder_name, encoder in self.conditional_encoders.items():
            input_keys = encoder.input_specification.keys
            missing_keys = [key for key in input_keys if key not in flat_obs]
            if missing_keys:
                logging.warning(f"Conditional encoder '{encoder_name}' skipped: missing {missing_keys}")
                continue
            condition_key = encoder.condition_key
            encoder_input = {key: flat_obs[key] for key in input_keys}
            encoded = encoder(encoder_input, features[condition_key])
            output_specification = self.encoder_to_outputs[encoder_name]
            for feature_name in output_specification.features:
                if feature_name in encoded:
                    features[f"{encoder_name}_{feature_name}"] = encoded[feature_name]

        for fusion_module in self.fusion_stages:
            input_features = [features[feat_name] for feat_name in fusion_module.input_features]
            features[fusion_module.output_name] = fusion_module(input_features)

        # Remove consumed features - fusion inputs are no longer needed
        for consumed_feat in self._consumed_features:
            features.pop(consumed_feat, None)

        return features

    def get_feature_names(self) -> list[str]:
        """Get all feature names produced by the encoding pipeline.

        Note:
            This includes all features (encoders + fusion outputs) for validation purposes.
            Use get_final_feature_names() to get only features available after fusion consumption.
        """
        return list(self._feature_keys_to_dims.keys())

    def get_final_feature_names(self) -> list[str]:
        """Get final feature names after fusion consumption.

        Returns:
            List of feature names that are actually available in forward() output.
            Excludes features consumed by fusion modules.
        """
        return [feat for feat in self._feature_keys_to_dims if feat not in self._consumed_features]

    def get_features_to_dimensions(self) -> dict[str, int | tuple[int, ...]]:
        """Get a dictionary of the feature names and dimensions produced by the encoding pipeline.

        Note:
            This includes all features (encoders + fusion outputs) for validation purposes.
            Use get_final_features_to_dimensions() to get only features available after fusion consumption.
        """
        return self._feature_keys_to_dims

    def get_final_features_to_dimensions(self) -> dict[str, int | tuple[int, ...]]:
        """Get final features and dimensions after fusion consumption.

        Returns:
            Dictionary mapping feature names to dimensions for features actually available
            in forward() output. Excludes features consumed by fusion modules.
        """
        return {
            feat: dim for feat, dim in self._feature_keys_to_dims.items()
            if feat not in self._consumed_features
        }

    def __repr__(self) -> str:
        """Pretty print the pipeline structure."""
        lines = ["EncodingPipeline(", "  Encoders:"]

        for enc_name, encoder in {**self.encoders, **self.conditional_encoders}.items():
            input_keys = encoder.input_specification.keys
            output = self.encoder_to_outputs[enc_name]
            output_keys = [f"{enc_name}_{feat}" for feat in output.features]
            lines.append(f"    {enc_name}: {input_keys} -> {output_keys} ({encoder.__class__.__name__})")

        if self.fusion_stages:
            lines.append("  Fusion stages:")
            for i, fusion in enumerate(self.fusion_stages):
                inputs = ", ".join(fusion.input_features)
                lines.append(f"    {i}: [{inputs}] -> {fusion.output_name} ({fusion.__class__.__name__})")

        lines.append(")")
        return "\n".join(lines)
