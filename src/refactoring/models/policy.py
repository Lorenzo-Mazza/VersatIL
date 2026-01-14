"""Policy module that handles the sequence of input encoding, output decoding, and loss computation."""


import torch
import torch.nn as nn

from refactoring.common.tensor_ops import to_device
from refactoring.data.task import ActionSpace, ObservationSpace
from refactoring.data.constants import (
    ACTION_KEY,
    IS_PAD_ACTION_KEY,
    OBSERVATION_KEY,
    Cameras,
)
from refactoring.data.tokenization import Tokenizer
from refactoring.data.transform import (
    unnormalize_actions,
    detokenize_actions,
    normalize_observation,
    tokenize_observation,
)
from refactoring.models.decoding.action_heads import MoEHead
from refactoring.models.decoding.constants import (
    BINARY_LOGITS_KEY,
    LATENT_KEY,
    LOGVAR_KEY,
    MU_KEY,
    PRIOR_LATENT_KEY,
    PRIOR_MU_KEY,
    PRIOR_LOGVAR_KEY,
    PRIOR_PREDICTION_KEY,
    PRIOR_TARGET_KEY,
    ROUTING_WEIGHT,
)

from refactoring.common.dict_of_tensor_mixin import DictOfTensorMixin
from refactoring.data.normalization.normalizer import LinearNormalizer
from refactoring.metrics.base import BaseLoss, LossOutput
from refactoring.metrics.components import GripperLoss
from refactoring.models.decoding.algorithm.base import DecodingAlgorithm
from refactoring.models.decoding.algorithm.variational import VariationalAlgorithm
from refactoring.models.decoding.decoders import MoEDecoder
from refactoring.models.decoding.decoders.base import ActionDecoder
from refactoring.models.encoding.pipeline import EncodingPipeline
from refactoring.data.constants import TOKENIZED_ACTIONS_KEY


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
            observation_horizon: Number of past observations to condition on
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
        self.observation_horizon = observation_horizon
        self.loss_module = loss
        self.device = torch.device(device)
        self.normalizer: LinearNormalizer = LinearNormalizer()
        self.tokenizer = None  # Set later via set_tokenizer()
        self.denoising_thresholds = (
            DictOfTensorMixin()
        )  # Set later via set_denoising_thresholds()
        self.validate_decoder()
        if validate_loss_keys:
            self.validate_loss_keys()

    def validate_decoder(self):
        """Validate that the decoder's input specification matches the encoding pipeline's output.

        Note:
            Uses final features (after fusion consumption) since that's what the decoder actually receives.
        """
        # Get final features - excludes features consumed by fusion modules
        available_features_to_dims = (
            self.encoding_pipeline.get_final_features_to_dimensions()
        )
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
        self.decoder.decoder_input.validate_feature_types(
            available_features_to_dims=available_features_to_dims
        )
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
        if (
            isinstance(self.algorithm, VariationalAlgorithm)
            and gating_key == LATENT_KEY
        ):
            return
        raise ValueError(
            f"MoE decoder gating feature key '{gating_key}' not found. "
            f"Available features from encoding pipeline: {available_features}. "
            f"Algorithm provides LATENT_KEY: {'Yes' if isinstance(self.algorithm, VariationalAlgorithm) else 'No'}."
        )

    def validate_loss_keys(self):
        """Validate that all loss keys are defined in the decoder heads or in the algorithm.
        This ensures that the loss module doesn't try to compute losses for
        actions or auxiliary keys that aren't part of the configuration.

        Raises:
            ValueError: If loss keys are invalid
        """
        valid_loss_keys: set[str] = set()
        valid_loss_keys.update(self.decoder.action_heads.keys())
        # Add auxiliary action keys (requires_prediction_head=False) for metadata passthrough
        for key, meta in self.action_space.actions_metadata.items():
            if not meta.requires_prediction_head:
                valid_loss_keys.add(key)
        if isinstance(self.algorithm, VariationalAlgorithm):
            valid_loss_keys.update(
                {
                    LATENT_KEY,
                    MU_KEY,
                    LOGVAR_KEY,
                    PRIOR_LATENT_KEY,
                    PRIOR_MU_KEY,
                    PRIOR_LOGVAR_KEY,
                    PRIOR_PREDICTION_KEY,
                    PRIOR_TARGET_KEY,
                }
            )
        if isinstance(self.decoder, MoEDecoder) or any(
            isinstance(h, MoEHead) for h in self.decoder.action_heads.values()
        ):
            valid_loss_keys.add(ROUTING_WEIGHT)
        if self.decoder.__class__.__name__ == "FreeTransformerDecoder":
            valid_loss_keys.add(BINARY_LOGITS_KEY)
        required_keys = self.loss_module.get_required_keys()
        if self.decoder.supports_tokenized_actions:
            valid_loss_keys.add(TOKENIZED_ACTIONS_KEY)
        invalid_keys = required_keys - valid_loss_keys
        if invalid_keys:
            raise ValueError(
                f"Loss module references keys {invalid_keys} that are not "
                f"defined in the action space or auxiliary keys. "
                f"Valid loss keys: {valid_loss_keys}. "
                f"Please update your loss configuration or decoder."
            )

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set normalizer for observations and actions."""
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.normalizer.to(self.device)

    def set_tokenizer(self, tokenizer: Tokenizer | None):
        """Set tokenizer and pass it to the decoder."""
        self.tokenizer = tokenizer
        self.encoding_pipeline.set_tokenizer(tokenizer)
        self.decoder.set_tokenizer(tokenizer)

    def set_denoising_thresholds(self, thresholds: dict[str, float]):
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
        self, batch: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Forward pass through observation encoding → action decoding.

        Args:
            batch: A batch dictionary containing normalized observations and actions dictionaries. Each is a dict of tensors.

        Returns:
            Decoder output dictionary containing action predictions and any architecture-specific outputs.
        """
        obs = batch[OBSERVATION_KEY]
        actions = batch.get(ACTION_KEY, None)
        features = self.encoding_pipeline(obs)
        return self.algorithm.forward(
            features=features, actions=actions, network=self.decoder
        )

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
        return self.loss_module(
            predictions=output,
            targets=batch[ACTION_KEY],
            is_pad=batch[ACTION_KEY].get(IS_PAD_ACTION_KEY, None),
        )  # type: ignore[no-any-return]

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
        obs_dict = to_device(
            obs_dict, self.device
        )  # Move nested observations to policy's device
        normalized_obs = normalize_observation(
            observation=obs_dict,
            normalizer=self.normalizer,
            observation_space=self.observation_space,
        )
        if (
            self.tokenizer is not None
            and self.tokenizer.observation_tokenizer is not None
        ):
            normalized_obs = tokenize_observation(
                observation=normalized_obs,
                obs_tokenizer=self.tokenizer.observation_tokenizer,
            )
        features = self.encoding_pipeline(normalized_obs)
        predictions = self.algorithm.predict(features=features, network=self.decoder)
        if TOKENIZED_ACTIONS_KEY in predictions:
            action_tokens = predictions[TOKENIZED_ACTIONS_KEY]
            if self.tokenizer is None or self.tokenizer.action_tokenizer is None:
                raise RuntimeError(
                    "Action tokenizer not set. Cannot detokenize actions."
                )
            normalized_actions = detokenize_actions(
                action_tokens,
                action_tokenizer=self.tokenizer.action_tokenizer,
                action_space=self.action_space,
            )
            normalized_actions = to_device(normalized_actions, self.device)
        else:
            normalized_actions = predictions
        actions = unnormalize_actions(
            normalized_actions=normalized_actions,
            normalizer=self.normalizer,
            action_space=self.action_space,
        )
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

        # Check unconditional encoders
        for encoder_name, encoder in self.encoding_pipeline.encoders.items():
            if is_vision_encoder(encoder):
                vision_encoders[encoder_name] = encoder

        # Check conditional encoders
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
                "(CNNEncoder, DepthCNNEncoder, ConditionalCNNEncoder, DFormerEncoder, LightGeometricEncoder). "
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
        if hasattr(encoder, "backbone"):
            backbone = encoder.backbone

            # TimmBackbone wraps the actual model in _backbone
            if hasattr(backbone, "_backbone"):
                actual_backbone = backbone._backbone
                # ResNet-style architectures have layer1, layer2, layer3, layer4
                if hasattr(actual_backbone, "layer4"):
                    return [actual_backbone.layer4]
                # Other architectures might have stages
                elif (
                    hasattr(actual_backbone, "stages")
                    and len(actual_backbone.stages) > 0
                ):
                    return [actual_backbone.stages[-1]]
                else:
                    raise RuntimeError(
                        f"Encoder '{encoder_name}' backbone structure not recognized. "
                        f"Backbone type: {type(actual_backbone).__name__}"
                    )
            # Direct TIMM models with stages attribute
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
            if not hasattr(encoder, "input_specification"):
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
