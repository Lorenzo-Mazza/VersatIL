"""Train diffusion action decoders and sample actions with DDIM or DDPM."""

import torch

from versatil.data.constants import SampleKey
from versatil.models.decoding.algorithm.base import (
    DecodingAlgorithm,
    resolve_feature_reference,
)
from versatil.models.decoding.constants import (
    AlgorithmContextKey,
    BetaSchedule,
    DecoderOutputKey,
    PredictionType,
    VarianceType,
)
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.layers.denoising.diffusion_process import (
    DiffusionSchedulerConfig,
    SchedulerType,
    add_noise_to_tensor,
    create_noise_scheduler,
    sample_random_timesteps,
)
from versatil.models.layers.denoising.diffusion_schedule import DiffusionSchedule


class Diffusion(DecodingAlgorithm):
    """Diffusion algorithm for action prediction.

    Trains a model to denoise actions by predicting noise (or clean actions) at various
    noise levels. During inference, starts from random noise and iteratively denoises
    to generate actions.

    Note:
        The training scheduler adds noise to target actions. ``inference_schedule``
        stores the timesteps and coefficients for denoising, computed when the
        algorithm is constructed or ``num_inference_steps`` is changed.

    Args:
        scheduler_type: Type of diffusion scheduler ("ddpm" or "ddim")
        num_train_timesteps: Number of diffusion steps during training
        num_inference_steps: Number of denoising steps during inference
        beta_start: Starting value of noise schedule
        beta_end: Ending value of noise schedule
        beta_schedule: Noise schedule type ("linear", "squaredcos_cap_v2", etc.)
        prediction_type: ``epsilon`` for noise, ``sample`` for clean actions or
            ``velocity`` for the diffusion velocity target.
        scheduler_variance_type: DDPM variance rule: ``fixed_small``,
            ``fixed_small_log`` or ``fixed_large``.
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
    ) -> None:
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
        self.noise_scheduler = create_noise_scheduler(config=scheduler_config)
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.prediction_type = prediction_type

    @property
    def num_inference_steps(self) -> int:
        """Return the number of denoising updates in the inference schedule."""
        return len(self.inference_schedule.timesteps)

    @num_inference_steps.setter
    def num_inference_steps(self, value: int) -> None:
        """Recompute inference timesteps and coefficients for the given step count.

        Args:
            value: Number of updates, between one and ``num_train_timesteps``.

        Raises:
            ValueError: If the step count or scheduler settings are unsupported.
        """
        self.inference_schedule = DiffusionSchedule(
            scheduler=self.noise_scheduler, num_inference_steps=value
        )

    def injected_feature_keys(self) -> set[str]:
        """The conditioning timestep is provided by the algorithm."""
        return {AlgorithmContextKey.TIMESTEP.value}

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
        clean_actions: dict[str, torch.Tensor] = {}
        is_pad = None
        for key, action in actions.items():
            if key == SampleKey.IS_PAD_ACTION.value:
                is_pad = action
                continue  # Skip padding mask
            clean_actions[key] = action
            noisy_actions[key], noise[key] = add_noise_to_tensor(
                clean=action,
                noise_scheduler=self.noise_scheduler,
                timesteps=timesteps,
            )

        # Add timesteps to features for eventual conditioning
        features_with_time = {**features, AlgorithmContextKey.TIMESTEP.value: timesteps}

        predictions = network(features=features_with_time, actions=noisy_actions)
        if self.prediction_type == PredictionType.EPSILON.value:
            target = noise
        elif self.prediction_type == PredictionType.SAMPLE.value:
            target = clean_actions
        elif self.prediction_type == PredictionType.VELOCITY.value:
            velocity = {}
            for key, action in clean_actions.items():
                velocity[key] = self.noise_scheduler.get_velocity(
                    sample=action, noise=noise[key], timesteps=timesteps
                )
            target = velocity
        else:
            raise ValueError(
                f"Unknown prediction_type: {self.prediction_type}. "
                f"Expected one of {[e.value for e in PredictionType]}"
            )
        outputs = {
            **predictions,
            DecoderOutputKey.TARGET_DIFFUSION.value: target,
            DecoderOutputKey.NOISE.value: noise,
            AlgorithmContextKey.TIMESTEP.value: timesteps,
        }
        if is_pad is not None:
            outputs[SampleKey.IS_PAD_ACTION.value] = is_pad
        return outputs

    @property
    def predicts_in_action_space(self) -> bool:
        """Only 'sample' prediction type outputs actions directly; 'epsilon' and 'velocity' do not."""
        return self.prediction_type == PredictionType.SAMPLE.value

    def get_targets(
        self,
        algorithm_output: dict[str, torch.Tensor],
        ground_truth_actions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return the diffusion target (noise, sample, or velocity)."""
        return algorithm_output[DecoderOutputKey.TARGET_DIFFUSION.value]

    def predict(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Sample Gaussian noise and denoise it into an action chunk.

        Note:
            ``B`` is batch size, ``S`` is the number of inference steps, ``H`` is
            the prediction horizon and ``D_k`` is the dimension of action component k.

        Args:
            network: Decoder with timestep conditioning and action-space metadata.
            features: Encoded observation features. The first floating feature
                determines the noise device, dtype and batch size.

        Returns:
            Normalized action chunks, keyed by action name, shaped ``(B, H, D_k)``.
        """
        batch_size, device, dtype = resolve_feature_reference(features=features)

        initial_noise = {}
        for key, meta in network.action_space.actions_metadata.items():
            if not meta.requires_prediction_head:
                continue
            initial_noise[key] = torch.randn(
                batch_size,
                network.prediction_horizon,
                meta.prediction_dimension,
                device=device,
                dtype=dtype,
            )  # (B, H, D_k)
        step_noise = None
        if self.inference_schedule.stochastic:
            step_noise = {
                key: torch.randn(
                    batch_size,
                    len(self.inference_schedule.timesteps),
                    *noise.shape[1:],
                    device=device,
                    dtype=dtype,
                )  # (B, S, H, D_k)
                for key, noise in initial_noise.items()
            }
        return self.predict_from_noise(
            network=network,
            features=features,
            initial_noise=initial_noise,
            step_noise=step_noise,
        )

    def predict_from_noise(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
        initial_noise: dict[str, torch.Tensor],
        step_noise: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate an action chunk from the given initial and per-step noise.

        Note:
            Tensor shapes use ``B`` for batch size, ``S`` for inference steps, ``H``
            for prediction horizon and ``D_k`` for the dimension of action component k.
            The decoder's encoded observations are cached for this call and the
            cache is cleared when sampling returns or raises an exception.
            DDPM noise at timestep zero is unused; that update is deterministic.

        Args:
            network: Decoder that predicts noise, clean actions or velocity according
                to the schedule's prediction type.
            features: Encoded observation features used at every denoising step.
            initial_noise: Standard-normal starting values, keyed by action name.
                Each tensor has shape ``(batch, prediction_horizon, action_dimension)``.
            step_noise: Standard-normal noise added during DDPM updates, keyed by
                action name. Each tensor has shape
                ``(batch, inference_steps, prediction_horizon, action_dimension)``.
                The second dimension follows ``inference_schedule.timesteps``.
                Omit for DDIM.

        Returns:
            Normalized action chunks after the last denoising step. Keys and tensor
            shapes match ``initial_noise``.

        Raises:
            ValueError: If initial noise is empty, or DDPM step noise is missing
                or has different keys or dimensions from the initial noise.
        """
        if not initial_noise:
            raise ValueError(
                "Diffusion inference requires at least one action component."
            )
        if self.inference_schedule.stochastic:
            if step_noise is None or step_noise.keys() != initial_noise.keys():
                raise ValueError(
                    "DDPM inference requires step noise for every initial-noise "
                    "action component."
                )
            for key, noise in initial_noise.items():
                expected_shape = (
                    noise.shape[0],
                    len(self.inference_schedule.timesteps),
                    *noise.shape[1:],
                )
                if step_noise[key].shape != expected_shape:
                    raise ValueError(
                        f"DDPM step noise for {key!r} must have shape {expected_shape}, "
                        f"got {tuple(step_noise[key].shape)}."
                    )
        noisy_actions = dict(initial_noise)
        reference = next(iter(noisy_actions.values()))  # (B, H, D_k)
        network.enable_encoder_cache()
        try:
            for step_index, timestep in enumerate(self.inference_schedule.timesteps):
                features_with_time = {
                    **features,
                    AlgorithmContextKey.TIMESTEP.value: torch.full(
                        (reference.shape[0],),
                        timestep,
                        device=reference.device,
                        dtype=torch.long,
                    ),  # (B,)
                }
                model_output = network(
                    features=features_with_time, actions=noisy_actions
                )  # each action: (B, H, D_k)
                noisy_actions = {
                    key: self.inference_schedule(
                        model_output=model_output[key],
                        sample=sample,
                        step_index=step_index,
                        noise=step_noise[key][:, step_index]
                        if step_noise is not None
                        else None,  # (B, S, H, D_k) -> (B, H, D_k), or None
                    )  # (B, H, D_k)
                    for key, sample in noisy_actions.items()
                }
        finally:
            network.disable_encoder_cache()
        return noisy_actions
