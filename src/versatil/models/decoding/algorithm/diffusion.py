"""Diffusion algorithm for action generation via iterative denoising.

This module implements diffusion-based action generation using shared diffusion
process components from diffusion_process.py. The algorithm trains a network
to denoise actions at various noise levels and uses iterative denoising for
action prediction during inference.

Shared Components Used:
    - DiffusionSchedulerConfig: Unified configuration for DDPM/DDIM schedulers
    - create_noise_scheduler(): Factory for creating noise schedulers
    - add_noise_to_tensor(): Forward diffusion process (adding noise to clean actions)
    - sample_random_timesteps(): Uniform timestep sampling for training
    - setup_inference_timesteps(): Configure scheduler for reverse diffusion

See diffusion_process.py for detailed documentation of these components.
"""

import torch

from versatil.data.constants import SampleKey
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.constants import (
    BetaSchedule,
    DecoderOutputKey,
    PredictionType,
    VarianceType,
)
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.dit_block_action_transformer import (
    DiTBlockActionTransformer,
)
from versatil.models.layers.denoising.diffusion_process import (
    DiffusionSchedulerConfig,
    SchedulerType,
    add_noise_to_tensor,
    create_noise_scheduler,
    sample_random_timesteps,
    setup_inference_timesteps,
)


class Diffusion(DecodingAlgorithm):
    """Diffusion algorithm for action prediction.

    Trains a model to denoise actions by predicting noise (or clean actions) at various
    noise levels. During inference, starts from random noise and iteratively denoises
    to generate actions.

    The diffusion process follows:
    - Training: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
    - Inference: Iteratively denoise from x_T to x_0 using learned denoising model

    Args:
        scheduler_type: Type of diffusion scheduler ("ddpm" or "ddim")
        num_train_timesteps: Number of diffusion steps during training
        num_inference_steps: Number of denoising steps during inference
        beta_start: Starting value of noise schedule
        beta_end: Ending value of noise schedule
        beta_schedule: Noise schedule type ("linear", "squaredcos_cap_v2", etc.)
        prediction_type: What the network predicts ("epsilon" for noise, "sample" for clean actions)
        scheduler_variance_type: Variance type for DDPM scheduler
        clip_sample: Whether to clip samples to [-1, 1] during inference
        set_alpha_to_one: Whether to set final alpha to 1
        steps_offset: Offset for timestep calculation
    """

    def __init__(
        self,
        scheduler_type: str = SchedulerType.DDIM.value,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = BetaSchedule.SQUAREDCOS_CAP_V2.value,
        prediction_type: str = PredictionType.EPSILON.value,
        scheduler_variance_type: str = VarianceType.FIXED_SMALL.value,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
    ):
        """Initialize Diffusion algorithm."""
        super().__init__()

        scheduler_config = DiffusionSchedulerConfig(
            scheduler_type=scheduler_type,
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=clip_sample,
            variance_type=scheduler_variance_type,
            set_alpha_to_one=set_alpha_to_one,
            steps_offset=steps_offset,
        )
        self.noise_scheduler = create_noise_scheduler(scheduler_config)

        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.prediction_type = prediction_type

    def forward(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass during training.

        Adds noise to ground-truth actions and trains the network to denoise them.

        Args:
            network: The action decoder network module (should support timestep conditioning)
            features: Dictionary of encoded features from the encoding pipeline.
            actions: Dictionary of ground truth actions. Required for diffusion training.
                Expected keys depend on action space (e.g., 'position_action', 'gripper_action')

        Returns:
            Decoder output dictionary containing:
                - Predicted noise or actions (depending on prediction_type)
                - 'target': The training target (noise or clean actions)
                - 'noise': The noise added to the clean actions
                - 'timestep': The random timesteps sampled for each action in the batch
                - 'is_pad_action': Padding mask if present

        Raises:
            ValueError: If actions are not provided (required for diffusion training)
        """
        if actions is None:
            raise ValueError("Diffusion algorithm requires actions during training")

        # Get batch size and device from actions
        first_action = next(iter(actions.values()))
        batch_size = first_action.shape[0]
        device = first_action.device

        # Sample random timesteps using shared diffusion process
        timesteps = sample_random_timesteps(
            batch_size=batch_size,
            num_train_timesteps=self.num_train_timesteps,
            device=device,
        )

        # Add noise to all action components using shared diffusion process
        noisy_actions: dict[str, torch.Tensor] = {}
        noise: dict[str, torch.Tensor] = {}
        is_pad = None
        for key, action in actions.items():
            if key == SampleKey.IS_PAD_ACTION.value:
                is_pad = action
                continue  # Skip padding mask
            noisy_actions[key], noise[key] = add_noise_to_tensor(
                clean=action,
                noise_scheduler=self.noise_scheduler,
                timesteps=timesteps,
            )

        # Add timesteps to features for eventual conditioning
        features_with_time = {**features, DecoderOutputKey.TIMESTEP.value: timesteps}

        predictions = network(features=features_with_time, actions=noisy_actions)
        if self.prediction_type == PredictionType.EPSILON.value:
            target = noise
        elif self.prediction_type == PredictionType.SAMPLE.value:
            target = actions
        elif self.prediction_type == PredictionType.VELOCITY.value:
            velocity = {}
            for key, action in actions.items():
                if key == SampleKey.IS_PAD_ACTION.value:
                    continue
                velocity[key] = self.noise_scheduler.get_velocity(
                    sample=action, noise=noise[key], timesteps=timesteps
                )
            target = velocity
        else:
            raise ValueError(
                f"Unknown prediction_type: {self.prediction_type}. "
                f"Expected one of {[e.value for e in PredictionType]}"
            )
        return {
            **predictions,
            DecoderOutputKey.TARGET_DIFFUSION.value: target,
            DecoderOutputKey.NOISE.value: noise,
            SampleKey.IS_PAD_ACTION.value: is_pad,
            DecoderOutputKey.TIMESTEP.value: timesteps,
        }

    def predict(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Inference/prediction pass.

        Generates actions by starting from random noise and iteratively denoising.

        Args:
            network: The action decoder network module
            features: Dict of encoded features from encoding pipeline

        Returns:
            Decoder output dictionary containing denoised action predictions.
        """
        first_feature = next(iter(features.values()))
        batch_size = first_feature.shape[0]
        device = first_feature.device
        dtype = first_feature.dtype

        # Initialize actions with random noise
        noisy_actions = {}
        for key, meta in network.action_space.actions_metadata.items():
            if not meta.requires_prediction_head:
                continue
            noisy_actions[key] = torch.randn(
                batch_size,
                network.prediction_horizon,
                meta.prediction_dimension,
                device=device,
                dtype=dtype,
            )
        setup_inference_timesteps(self.noise_scheduler, self.num_inference_steps)
        if isinstance(network, DiTBlockActionTransformer):
            network.enable_encoder_cache()

        # Iteratively denoise
        for t in self.noise_scheduler.timesteps:
            # Expand timestep to batch dimension
            timestep = t.unsqueeze(0).expand(batch_size).to(device)
            features_with_time = {**features, DecoderOutputKey.TIMESTEP.value: timestep}
            model_output = network(features=features_with_time, actions=noisy_actions)
            for key in noisy_actions:
                if key in model_output:
                    noisy_actions[key] = self.noise_scheduler.step(
                        model_output[key], t, noisy_actions[key]
                    ).prev_sample

        if isinstance(network, DiTBlockActionTransformer):
            network.disable_encoder_cache()

        return noisy_actions
