"""Policy module that handles the sequence of input encoding, output decoding, and loss computation."""


import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate

from refactoring.common.tensor_ops import to_device
from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.data.constants import (
    ACTION_KEY,
    GRIPPER_ACTION_KEY,
    GRIPPER_STATE_OBS_KEY,
    IS_PAD_KEY,
    LANGUAGE_KEY,
    OBSERVATION_KEY,
    ORIENTATION_ACTION_KEY,
    PHASE_LABEL_KEY,
    POSITION_ACTION_KEY,
    PROPRIO_STATE,
    Cameras,
    GripperType,
)
from refactoring.models.decoding.constants import BINARY_LOGITS_KEY, LATENT_KEY, LOGVAR_KEY, MU_KEY, PRIOR_PREDICTION_KEY, PRIOR_TARGET_KEY
from refactoring.data.normalize.normalizer import LinearNormalizer
from refactoring.metrics.base import BaseLoss, LossOutput
from refactoring.models.decoding.algorithm.base import DecodingAlgorithm
from refactoring.models.decoding.algorithm.variational import VariationalAlgorithm
from refactoring.models.decoding.decoders import MoEDecoder
from refactoring.models.decoding.latent.gaussian_prior import GaussianPrior
from refactoring.models.decoding.decoders.base import ActionDecoder
from refactoring.models.encoding.pipeline import EncodingPipeline


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
            loss: BaseLoss,
            device: str,
            validate_loss_keys: bool = True,
    ):
        """Initialize policy.

        Args:
            encoding_pipeline: Observation encoding pipeline
            algorithm: Decoding algorithm (diffusion, flow matching, etc.)
            decoder: Action decoder architecture
            observation_space: Observation space configuration
            action_space: Action space configuration
            prediction_horizon: Number of future actions to predict
            loss: Loss module for training
            device: Device to run on
            validate_loss_keys: Whether to validate loss keys against action space
        """
        super().__init__()
        self.encoding_pipeline = encoding_pipeline
        self.algorithm = algorithm
        self.decoder = decoder
        self.observation_space = observation_space
        self.action_space = action_space
        self.prediction_horizon = prediction_horizon
        self.loss_module = loss
        self.device = torch.device(device)
        self.normalizer: LinearNormalizer = LinearNormalizer()
        self.tokenizer = None  # Set via set_tokenizer() if using tokenization

        self.validate_decoder()
        if validate_loss_keys:
            self.validate_loss_keys()


    def validate_decoder(self):
        """Validate that the decoder's input specification matches the encoding pipeline's output.

        Note:
            Uses final features (after fusion consumption) since that's what the decoder actually receives.
        """
        # Get final features - excludes features consumed by fusion modules
        available_features_to_dims = self.encoding_pipeline.get_final_features_to_dimensions()
        available_features = list(available_features_to_dims.keys())
        decoder_input_keys = self.decoder.decoder_input.keys
        for expected_feature in decoder_input_keys:
            if expected_feature not in available_features:
                raise ValueError(
                    f"Action decoding network expects input feature '{expected_feature}' "
                    f"but it's not produced by any encoder or fusion layer (or it was consumed by fusion). "
                    f"Available final features: {available_features}"
                )
        # Validate feature types expected by the decoder architecture (spatial, flat, etc.)
        self.decoder.decoder_input.validate_feature_types(available_features_to_dims=available_features_to_dims)
        # Note: Action head validation is done in ActionDecoder.__init__

        # Validate MoE gating feature key if using MoE decoder with gating network
        if isinstance(self.decoder, MoEDecoder):
            self._validate_moe_gating_feature(available_features)

    def _validate_moe_gating_feature(self, available_features: list[str]):
        """Validate that MoE gating feature key exists in available features or latent encoder.

        Args:
            available_features: List of features available from encoding pipeline

        Raises:
            ValueError: If gating feature key is not available
        """
        has_gating = bool(self.decoder.has_gating_network)
        if not has_gating:
            return  # No gating network, using external routing
        gating_key = self.decoder.gating_feature_key
        # Check if gating key is in encoding pipeline features
        if gating_key in available_features:
            return
        # Check if gating key is provided by variational algorithm
        if isinstance(self.algorithm, VariationalAlgorithm) and gating_key == LATENT_KEY:
            return
        raise ValueError(
            f"MoE decoder gating feature key '{gating_key}' not found. "
            f"Available features from encoding pipeline: {available_features}. "
            f"Algorithm provides LATENT_KEY: {'Yes' if isinstance(self.algorithm, VariationalAlgorithm) else 'No'}."
        )

    def validate_loss_keys(self):
        """Validate that all loss keys are defined in the action space or are valid auxiliary keys.

        This ensures that the loss module doesn't try to compute losses for
        actions or auxiliary keys that aren't part of the configuration.

        Raises:
            ValueError: If loss keys are invalid
        """
        valid_action_keys: set[str] = set()
        if self.action_space.has_position:
            valid_action_keys.add(POSITION_ACTION_KEY)
        if self.action_space.has_orientation:
            valid_action_keys.add(ORIENTATION_ACTION_KEY)
        if self.action_space.has_gripper:
            valid_action_keys.add(GRIPPER_ACTION_KEY)
        if self.action_space.task_has_phases:
            valid_action_keys.add(PHASE_LABEL_KEY)
        if self.action_space.custom_action_dims:
            valid_action_keys.update(self.action_space.custom_action_dims.keys())

        # Add auxiliary keys based on algorithm and decoder types
        valid_auxiliary_keys: set[str] = set()
        if isinstance(self.algorithm, VariationalAlgorithm):
            # Add keys that variational algorithms produce
            valid_auxiliary_keys.add(LATENT_KEY)  # Latent embedding
            valid_auxiliary_keys.add(MU_KEY)  # Posterior mean
            valid_auxiliary_keys.add(LOGVAR_KEY)  # Posterior log variance
            # Only add prior keys for learned priors (not GaussianPrior)
            # GaussianPrior returns {} from forward(), so it doesn't produce these keys
            if not isinstance(self.algorithm.prior, GaussianPrior):
                valid_auxiliary_keys.add(PRIOR_PREDICTION_KEY)  # Prior predictions (for learned priors)
                valid_auxiliary_keys.add(PRIOR_TARGET_KEY)  # Prior targets (for learned priors)

        # Free Transformer binary logits (for discrete latent codes)
        decoder_class_name = self.decoder.__class__.__name__
        if decoder_class_name == "FreeTransformer":
            valid_auxiliary_keys.add(BINARY_LOGITS_KEY)  # Binary mapper logits

        # Get all required keys from loss module
        required_keys = self.loss_module.get_required_keys()
        valid_keys = valid_action_keys | valid_auxiliary_keys
        invalid_keys = required_keys - valid_keys

        if invalid_keys:
            raise ValueError(
                f"Loss module references keys {invalid_keys} that are not "
                f"defined in the action space or auxiliary keys. "
                f"Valid action keys: {valid_action_keys}. "
                f"Valid auxiliary keys: {valid_auxiliary_keys}. "
                f"Please update your loss configuration or action space."
            )

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set normalizer for observations and actions."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_tokenizer(self, tokenizer):
        """Set tokenizer and pass it to the decoder."""
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.decoder.set_tokenizer(tokenizer)


    def normalize_observations(self, observation: dict[str, torch.Tensor]) ->  dict[str, torch.Tensor]:
        """Normalize observations using the policy's normalizer.

        Args:
            observation: Dictionary of observation tensors (may contain nested dicts).

        Returns:
            Dictionary of normalized observation tensors.
        """
        normalized_observation = observation.copy()

        # Flatten nested proprioceptive dict before normalization
        if PROPRIO_STATE in normalized_observation and isinstance(normalized_observation[PROPRIO_STATE], dict):
            proprio_dict = normalized_observation.pop(PROPRIO_STATE)
            normalized_observation.update(proprio_dict)

        if self.observation_space.use_language:
            # Language observations are not normalized
            del normalized_observation[LANGUAGE_KEY]
        if self.observation_space.use_gripper_state and self.observation_space.gripper_type == GripperType.BINARY.value:
            # Binary gripper actions are not normalized
            del normalized_observation[GRIPPER_STATE_OBS_KEY]
        normalized_observation = self.normalizer.normalize(normalized_observation)  # type: ignore[assignment]
        # Restore non-normalized keys
        if self.observation_space.use_language:
            normalized_observation[LANGUAGE_KEY] = observation[LANGUAGE_KEY]
        if self.observation_space.use_gripper_state and self.observation_space.gripper_type == GripperType.BINARY.value:
            normalized_observation[GRIPPER_STATE_OBS_KEY] = observation[GRIPPER_STATE_OBS_KEY]
        return normalized_observation

    def normalize_actions(self, action: dict[str, torch.Tensor]) ->  dict[str, torch.Tensor]:
        """Normalize actions using the policy's normalizer.

        Args:
            action: Dictionary of action tensors.

        Returns:
            Dictionary of normalized action tensors.
        """
        normalized_action = action.copy()
        if self.action_space.has_position:
            normalized_action[POSITION_ACTION_KEY] = self.normalizer[POSITION_ACTION_KEY].normalize(action[POSITION_ACTION_KEY])
        if self.action_space.has_orientation:
            normalized_action[ORIENTATION_ACTION_KEY] = self.normalizer[ORIENTATION_ACTION_KEY].normalize(action[ORIENTATION_ACTION_KEY])
        if self.action_space.gripper_type == GripperType.CONTINUOUS.value:
            normalized_action[GRIPPER_ACTION_KEY] = self.normalizer[GRIPPER_ACTION_KEY].normalize(action[GRIPPER_ACTION_KEY])
        if self.action_space.custom_action_dims:
            for custom_key in self.action_space.custom_action_dims:
                normalized_action[custom_key] = self.normalizer[custom_key].normalize(action[custom_key])
        # If gripper is binary, no normalization is applied. Also, no normalization is applied to padding mask and phase labels if present.
        return normalized_action

    def unnormalize_actions(self, normalized_action: dict[str, torch.Tensor]) ->  dict[str, torch.Tensor]:
        """Unnormalize actions using the policy's normalizer.

        Args:
            normalized_action: Dictionary of normalized action tensors.

        Returns:
            Dictionary of unnormalized action tensors.
        """
        action = normalized_action.copy()
        if self.action_space.has_position:
            action[POSITION_ACTION_KEY] = self.normalizer[POSITION_ACTION_KEY].unnormalize(normalized_action[POSITION_ACTION_KEY])
        if self.action_space.has_orientation:
            action[ORIENTATION_ACTION_KEY] = self.normalizer[ORIENTATION_ACTION_KEY].unnormalize(normalized_action[ORIENTATION_ACTION_KEY])
        if self.action_space.gripper_type == GripperType.CONTINUOUS.value:
            action[GRIPPER_ACTION_KEY] = self.normalizer[GRIPPER_ACTION_KEY].unnormalize(normalized_action[GRIPPER_ACTION_KEY])
        if self.action_space.custom_action_dims:
            for custom_key in self.action_space.custom_action_dims:
                action[custom_key] = self.normalizer[custom_key].unnormalize(normalized_action[custom_key])
        # If gripper is binary, no unnormalization is applied. Also, no unnormalization is applied to padding mask and phase labels if present.
        return action


    def forward(self, batch: dict[str, dict[str, torch.Tensor]]):
        """Forward pass through observation encoding → action decoding.

        Args:
            batch: A batch dictionary containing observations and actions dictionaries. Each is a dict of tensors.
        """
        normalized_obs = self.normalize_observations(batch[OBSERVATION_KEY])
        if self.decoder.decoder_input.requires_actions:
            normalized_actions = self.normalize_actions(batch[ACTION_KEY])
        else:
            normalized_actions = None
        features = self.encoding_pipeline(normalized_obs)
        return self.algorithm.forward(features=features, actions=normalized_actions, network=self.decoder)


    def compute_loss(
            self,
            batch: dict[str, dict[str, torch.Tensor]],
    ) -> LossOutput:
        """Compute loss using the configured loss module.

        Args:
            batch: Batch dictionary containing observations and actions

        Returns:
            LossOutput with total loss and component losses
        """
        output = self.forward(batch)
        loss_output = self.loss_module(
            predictions=output,
            targets=batch[ACTION_KEY],
            is_pad=batch[ACTION_KEY].get(IS_PAD_KEY, None),
        )
        return loss_output  # type: ignore[no-any-return]


    def predict_action(
            self,
            obs_dict: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Predict actions from observations.

        Args:
            obs_dict: Dictionary of observation tensors

        Returns:
            Predicted actions (on same device as policy)
        """
        # Move observations to policy's device (handles nested structures)
        obs_dict = to_device(obs_dict, self.device)
        normalized_obs = self.normalize_observations(obs_dict)
        features = self.encoding_pipeline(normalized_obs)
        normalized_actions = self.algorithm.predict(features=features, network=self.decoder)
        actions = self.unnormalize_actions(normalized_actions)
        return actions  # type: ignore[no-any-return]

    def get_vision_encoder_modules(self) -> dict[str, nn.Module]:
        """Get vision encoder modules that can produce spatial feature maps for explainability.

        Supports the following encoder types:
        - CNNEncoder (RGB): Has 'backbone' attribute
        - DepthCNNEncoder: Has 'backbone' attribute
        - ConditionalCNNEncoder (FiLM): Has 'layer4' attribute
        - DFormerEncoder: Has 'stages' attribute
        - LightGeometricEncoder: Has 'attention_block' attribute

        Returns:
            Dictionary mapping encoder names to their encoder instances.
            Only includes encoders that produce spatial feature maps.

        Raises:
            RuntimeError: If no compatible vision encoders are found
        """
        vision_encoders = {}

        # Check if encoder can produce spatial feature maps for explainability
        def is_vision_encoder(encoder: nn.Module) -> bool:
            # TIMM-based encoders (CNNEncoder, DepthCNNEncoder)
            if hasattr(encoder, 'backbone'):
                return True
            # DFormer-based encoders
            if hasattr(encoder, 'stages'):
                return True
            # FiLM-conditioned ResNet (ConditionalCNNEncoder)
            if hasattr(encoder, 'layer4'):
                return True
            # LightGeometric encoder
            return bool(hasattr(encoder, 'attention_block'))

        # Check unconditional encoders
        for encoder_name, encoder in self.encoding_pipeline.encoders.items():
            if is_vision_encoder(encoder):
                vision_encoders[encoder_name] = encoder

        # Check conditional encoders
        for encoder_name, encoder in self.encoding_pipeline.conditional_encoders.items():
            if is_vision_encoder(encoder):
                vision_encoders[encoder_name] = encoder

        if not vision_encoders:
            raise RuntimeError(
                "No compatible vision encoders found in the encoding pipeline. "
                "Explainer requires encoders that produce spatial feature maps "
                "(CNNEncoder, DepthCNNEncoder, ConditionalCNNEncoder, DFormerEncoder, LightGeometricEncoder). "
                "Available encoders: " +
                str(list(self.encoding_pipeline.encoders.keys()) +
                    list(self.encoding_pipeline.conditional_encoders.keys()))
            )

        return vision_encoders

    def get_gradcam_target_layers(self, encoder_name: str) -> list[nn.Module]:
        """Get target layers for GradCAM from a specific vision encoder.

        Supports different encoder architectures:
        - TIMM backbones (CNNEncoder, DepthCNNEncoder): Returns last stage
        - ConditionalCNNEncoder (FiLM): Returns last block of layer4
        - DFormerEncoder: Returns last stage
        - LightGeometricEncoder: Returns attention block

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

        # TIMM-based encoders (CNNEncoder, DepthCNNEncoder)
        if hasattr(encoder, 'backbone'):
            backbone = encoder.backbone

            # TimmBackbone wraps the actual model in _backbone
            if hasattr(backbone, '_backbone'):
                actual_backbone = backbone._backbone
                # ResNet-style architectures have layer1, layer2, layer3, layer4
                if hasattr(actual_backbone, 'layer4'):
                    return [actual_backbone.layer4]
                # Other architectures might have stages
                elif hasattr(actual_backbone, 'stages') and len(actual_backbone.stages) > 0:
                    return [actual_backbone.stages[-1]]
                else:
                    raise RuntimeError(
                        f"Encoder '{encoder_name}' backbone structure not recognized. "
                        f"Backbone type: {type(actual_backbone).__name__}"
                    )
            # Direct TIMM models with stages attribute
            elif hasattr(backbone, 'stages') and len(backbone.stages) > 0:
                return [backbone.stages[-1]]
            else:
                raise RuntimeError(
                    f"Encoder '{encoder_name}' has backbone but structure not recognized. "
                    f"Backbone type: {type(backbone).__name__}"
                )

        # DFormer-based encoders
        if hasattr(encoder, 'stages') and len(encoder.stages) > 0:
            return [encoder.stages[-1]]

        # FiLM-conditioned ResNet (ConditionalCNNEncoder)
        if hasattr(encoder, 'layer4') and len(encoder.layer4) > 0:
            return [encoder.layer4[-1]]

        # LightGeometric encoder
        if hasattr(encoder, 'attention_block'):
            return [encoder.attention_block]

        raise RuntimeError(
            f"Encoder '{encoder_name}' architecture not supported for GradCAM. "
            f"Encoder type: {type(encoder).__name__}. "
            f"Supported types: CNNEncoder, DepthCNNEncoder, ConditionalCNNEncoder, "
            f"DFormerEncoder, LightGeometricEncoder."
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
            if not hasattr(encoder, 'input_specification'):
                continue

            # Get the input keys this encoder expects
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


@hydra.main(version_base=None, config_path="configs", config_name="main")
def create_policy_from_config(cfg):
    """Factory function to instantiate policy from Hydra config."""
    policy = instantiate(cfg.policy)
    return policy
