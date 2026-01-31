"""Conditional U-Net Decoder for action generation.
Reference implementation: Diffusion Policy (https://arxiv.org/abs/2303.04137)
"""

import logging
from typing import Optional

import torch
from torch import nn

from versatil.data.task import ActionSpace, ObservationSpace
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.constants import FeatureType, TIMESTEP_KEY
from versatil.models.decoding.decoders.base import DecoderInput
from versatil.models.layers.conditional_unet import ConditionalUnet1D
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.unet_input_builder import UNetInputBuilder


class ConditionalUNetDecoder(ActionDecoder):
    """Conditional U-Net decoder for generative action generation.

    This architecture:
    - Uses FiLM (Feature-wise Linear Modulation) for conditioning
    - Accepts global conditioning from concatenated observation features
    - Optionally supports local (sequence-aligned) conditioning
    - Designed for use with Diffusion or Flow Matching algorithms

    The decoder expects:
    - Noisy actions as input (via actions parameter during forward)
    - Timesteps injected by the algorithm (via features[TIMESTEP_KEY])
    - Observation features for global conditioning (via features dict)

    Note: This decoder is specifically designed for diffusion/flow matching algorithms
    and expects the algorithm to handle noise scheduling and timestep injection.
    """

    def __init__(
        self,
        input_keys: list[str],
        action_space: ActionSpace,
        action_heads: dict[str, ActionHead],
        observation_space: ObservationSpace,
        observation_horizon: int,
        prediction_horizon: int,
        device: str,
        embedding_dimension: int = 256,
        down_dimensions: list[int] = None,
        kernel_size: int = 5,
        num_groups: int = 8,
        use_local_conditioning: bool = False,
        condition_predict_scale: bool = False,
    ):
        """Initialize Conditional U-Net decoder.

        Args:
            input_keys: List of feature keys expected from encoder pipeline
            action_space: Action space configuration
            action_heads: Dictionary of action head modules
            observation_space: Observation space configuration
            observation_horizon: Number of observation timesteps (for history)
            prediction_horizon: Number of actions to predict (horizon)
            device: Device to run the model on
            embedding_dimension: Diffusion timestep embedding dimension
            down_dimensions: List of channel dimensions for downsampling layers
            kernel_size: Kernel size for convolutions in residual blocks
            num_groups: Number of groups for group normalization
            use_local_conditioning: Whether to use local (sequence-aligned) conditioning
            condition_predict_scale: If True, conditions predict scaling factors in FiLM

        Raises:
            ValueError: If local conditioning is requested but not yet implemented
        """
        decoder_input = DecoderInput(
            keys=input_keys,
            raises_for_types=[FeatureType.SPATIAL.value],
            requires_actions=True,
        )
        for k, head in action_heads.items():
            if len(head.blocks) > 0:
                logging.warning(
                    f"Action heads are ignored by ConditionalUNetDecoder, but one was provided for action '{k}'. Skipping."
                )
                action_heads[
                    k
                ].blocks = (
                    nn.ModuleList()
                )  # Replace with identity; U-Net handles all processing

        super().__init__(
            decoder_input=decoder_input,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            prediction_horizon=prediction_horizon,
            observation_horizon=observation_horizon,
            device=device,
        )

        if down_dimensions is None:
            down_dimensions = [256, 512, 1024]

        if use_local_conditioning:
            raise NotImplementedError(
                "Local conditioning is not yet implemented. "
                "Use global conditioning (obs_as_global_cond=True) for now."
            )

        self.embedding_dimension = embedding_dimension
        self.down_dimensions = down_dimensions
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.use_local_conditioning = use_local_conditioning
        self.condition_predict_scale = condition_predict_scale

        self._global_conditioning_dimension: Optional[int] = None
        self._feature_projections: Optional[nn.ModuleDict] = None
        self.unet_conditioning_builder = UNetInputBuilder(
            embedding_dim=embedding_dimension, has_time_dim=self.observation_horizon > 1
        )

        # U-Net will be lazily initialized on first forward pass
        # (once we know the global conditioning dimension)
        self._unet: ConditionalUnet1D | None = None

    def _initialize_unet(self, global_conditioning_dimension: int | None = None):
        """Lazily initialize the U-Net once we know the global conditioning dimension.

        Args:
            global_conditioning_dimension: Dimensionality of optional global conditioning vector
        """
        self._global_conditioning_dimension = global_conditioning_dimension
        self._unet = ConditionalUnet1D(
            input_dimension=self.action_dim,
            local_conditioning_dimension=None,  # Not using local conditioning
            global_conditioning_dimension=global_conditioning_dimension,
            diffusion_step_embedding_dimension=self.embedding_dimension,
            down_dimensions=self.down_dimensions,
            kernel_size=self.kernel_size,
            num_groups=self.num_groups,
            condition_predict_scale=self.condition_predict_scale,
        ).to(self.device)
        logging.info(
            f"Initialized ConditionalUnet1D with global_conditioning_dimension={global_conditioning_dimension}"
        )

    def _prepare_global_conditioning(
        self, features: dict[str, torch.Tensor]
    ) -> torch.Tensor | None:
        """Extract and flatten all observation features for global FiLM conditioning.

        Args:
            features: Dictionary of encoded features from the encoding pipeline

        Returns:
            Global conditioning tensor of shape (batch_size, global_conditioning_dimension) or None if no features.

        Note:
            This method creates the U-Net lazily when firstly called.

        Raises:
            ValueError: If spatial features provided, or if feature shapes are invalid.
        """
        global_conditioning = self.unet_conditioning_builder(features)
        # Lazy initialization of U-Net on first forward pass
        if self._unet is None:
            conditioning_dim = (
                global_conditioning.shape[-1]
                if global_conditioning is not None
                else None
            )
            self._initialize_unet(conditioning_dim)

        return global_conditioning

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through the conditional U-Net.

        This method is called by the decoding algorithm (Diffusion, FlowMatching)
        which provides:
        - Noisy actions
        - Observation features dictionary containing the timestep key

        Args:
            features: Dictionary of encoded features from the encoding pipeline.
            actions: Dictionary of noise-injected actions (provided by algorithm during training)

        Returns:
            Dictionary containing denoised predictions for each action head

        Raises:
            ValueError: If timesteps or actions are missing.
        """
        if actions is None:
            raise ValueError(
                "ConditionalUNetDecoder requires 'actions' parameter. "
                "The algorithm should provide noisy actions during forward pass."
            )

        if TIMESTEP_KEY not in features:
            raise ValueError(
                f"Missing '{TIMESTEP_KEY}' in features dict. "
                "The algorithm should inject timesteps into features."
            )

        timesteps = features.pop(TIMESTEP_KEY)  # (B,) or (B, 1)
        if len(timesteps.shape) == 2:
            timesteps = timesteps.squeeze(-1)

        # Concatenate all action modalities into single tensor
        # Shape: (B, T, action_dimension) where T = prediction_horizon
        action_tensors = []
        for action_key in sorted(actions.keys()):
            action_tensors.append(actions[action_key])
        noisy_actions = torch.cat(
            action_tensors, dim=-1
        )  # (B, T, total_action_dimension)

        # Prepare global conditioning
        global_conditioning = self._prepare_global_conditioning(
            features
        )  # (B, global_conditioning_dimension)

        # Run U-Net denoising
        assert self._unet is not None, "U-Net should be initialized by now."
        denoised = self._unet(
            noisy_input=noisy_actions,
            timesteps=timesteps,
            local_conditioning=None,  # Not using local conditioning
            global_conditioning=global_conditioning,
        )  # (B, T, action_dimension)

        # Split denoised output through action heads
        outputs = {}
        start_index = 0
        for action_key in sorted(actions.keys()):
            head = self.action_heads[action_key]
            end_index = start_index + head.output_dim
            action_slice = denoised[
                ..., start_index:end_index
            ]  # (B, T, action_dimension_i)
            outputs[action_key] = action_slice
            start_index = end_index

        return outputs
