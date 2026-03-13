"""Policy module that handles the sequence of input encoding, output decoding, and loss computation."""

import torch
import torch.nn as nn

from versatil.common.dict_of_tensor_mixin import DictOfTensorMixin
from versatil.common.tensor_ops import to_device
from versatil.data.constants import Cameras, SampleKey
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import Tokenizer
from versatil.data.transform import (
    detokenize_actions,
    normalize_observation,
    tokenize_observation,
    unnormalize_actions,
)
from versatil.metrics.base import BaseLoss, LossOutput
from versatil.metrics.components import GripperLoss
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.decoders.base import ActionDecoder
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
        validate_loss_keys: bool = True,
    ):
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
        self.normalizer: LinearNormalizer = LinearNormalizer()
        self.tokenizer = None  # Set later via set_tokenizer()
        self.denoising_thresholds = (
            DictOfTensorMixin()
        )  # Set later via set_denoising_thresholds()

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set normalizer for observations and actions."""
        self.normalizer.load_state_dict(normalizer.state_dict())
        self.normalizer.to(self.device)
        self.decoder.set_normalizer(self.normalizer)

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
        obs = batch[SampleKey.OBSERVATION.value]
        actions = batch.get(SampleKey.ACTION.value, None)
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
            targets=batch[SampleKey.ACTION.value],
            is_pad=batch[SampleKey.ACTION.value].get(
                SampleKey.IS_PAD_ACTION.value, None
            ),
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
        obs_dict = to_device(obs_dict, device=self.device)
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
        if SampleKey.TOKENIZED_ACTIONS.value in predictions:
            action_tokens = predictions[SampleKey.TOKENIZED_ACTIONS.value]
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
