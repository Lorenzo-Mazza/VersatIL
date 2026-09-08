"""DDIM and DDPM sampling with timestep coefficients computed at initialization."""

import torch
from diffusers import DDIMScheduler, DDPMScheduler
from torch import nn

from versatil.models.decoding.constants import PredictionType, VarianceType
from versatil.models.layers.denoising.diffusion_process import (
    DIFFUSERS_VELOCITY_PREDICTION,
)


class DiffusionSchedule(nn.Module):
    """Compute a diffusion sampling step from the decoder output and current actions.

    Note:
        Initialization selects the inference timesteps and computes the coefficients
        for each update from a Diffusers scheduler. ``forward()`` applies the update
        using tensor operations that can be recorded by ``torch.export``.

        DDIM uses deterministic updates, equivalent to ``eta=0`` in Diffusers. DDPM
        adds Gaussian noise passed through the ``noise`` argument. Both methods use
        the scheduler's prediction type and sample-clipping settings.

        Coefficients are Python numbers computed in the scheduler's precision.
        Moving the policy to BF16 or FP16 therefore preserves their precision.
        They are derived from configuration and add no checkpoint state.
        Updates reconstruct FP32 CPU scalar tensors to preserve Diffusers'
        scalar-promotion behavior for both CPU and CUDA action tensors.

    Attributes:
        timesteps: Diffusion timesteps in the order used for inference.
        stochastic: Whether the schedule uses DDPM updates.
        coefficients: One tuple of five coefficients per inference step. It stores
            the square roots of the cumulative alpha and beta, the multipliers
            for the estimated clean sample and the update direction, and the
            multiplier for the added noise.
    """

    def __init__(
        self,
        scheduler: DDIMScheduler | DDPMScheduler,
        num_inference_steps: int,
    ) -> None:
        """Select inference timesteps and compute the coefficients for each update.

        Note:
            Calling ``set_timesteps()`` updates the supplied scheduler's inference
            timesteps in place.

        Args:
            scheduler: Scheduler containing the policy's beta schedule, prediction
                type, clipping settings and DDPM variance type.
            num_inference_steps: Number of sampling updates used to generate one
                action chunk.

        Raises:
            ValueError: If the number of steps is outside the scheduler's training
                timesteps, dynamic thresholding is enabled, or the DDPM variance
                type is not ``fixed_small``, ``fixed_small_log`` or ``fixed_large``.
        """
        super().__init__()
        if not 1 <= num_inference_steps <= scheduler.config.num_train_timesteps:
            raise ValueError(
                "num_inference_steps must be between 1 and "
                f"{scheduler.config.num_train_timesteps}, got {num_inference_steps}."
            )
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        self.timesteps = tuple(int(timestep) for timestep in scheduler.timesteps)
        self.stochastic = isinstance(scheduler, DDPMScheduler)
        self.prediction_type = scheduler.config.prediction_type
        self.clip_sample = scheduler.config.clip_sample
        self.clip_sample_range = scheduler.config.clip_sample_range
        if scheduler.config.thresholding:
            raise ValueError(
                "Diffusion graph export does not support dynamic thresholding."
            )
        if self.stochastic and scheduler.config.variance_type not in (
            VarianceType.FIXED_SMALL.value,
            VarianceType.FIXED_LARGE.value,
            VarianceType.FIXED_SMALL_LOG.value,
        ):
            raise ValueError(
                "Diffusion action sampling requires fixed_small, fixed_small_log "
                f"or fixed_large variance, got {scheduler.config.variance_type!r}. "
                "Decoder outputs contain denoising predictions without a learned "
                "variance component."
            )
        coefficients = []
        for index, timestep in enumerate(self.timesteps):
            if self.stochastic:
                previous = (
                    self.timesteps[index + 1] if index + 1 < len(self.timesteps) else -1
                )
                final_alpha = scheduler.one
            else:
                previous = (
                    timestep
                    - scheduler.config.num_train_timesteps // num_inference_steps
                )
                final_alpha = scheduler.final_alpha_cumprod
            alpha = scheduler.alphas_cumprod[timestep]
            previous_alpha = (
                scheduler.alphas_cumprod[previous] if previous >= 0 else final_alpha
            )
            beta = 1 - alpha
            current_alpha = alpha / previous_alpha
            current_beta = 1 - current_alpha
            if self.stochastic:
                original_scale = previous_alpha.sqrt() * current_beta / beta
                sample_scale = current_alpha.sqrt() * (1 - previous_alpha) / beta
                variance = ((1 - previous_alpha) / beta * current_beta).clamp(min=1e-20)
                if scheduler.config.variance_type == VarianceType.FIXED_LARGE.value:
                    variance = current_beta
                noise_scale = (
                    variance.sqrt() if timestep > 0 else torch.zeros_like(variance)
                )
                if (
                    timestep > 0
                    and scheduler.config.variance_type
                    == VarianceType.FIXED_SMALL_LOG.value
                ):
                    noise_scale = (0.5 * variance.log()).exp()
            else:
                original_scale = previous_alpha.sqrt()
                sample_scale = (1 - previous_alpha).sqrt()
                noise_scale = torch.zeros_like(alpha)
            coefficients.append(
                (
                    alpha.sqrt().item(),
                    beta.sqrt().item(),
                    original_scale.item(),
                    sample_scale.item(),
                    noise_scale.item(),
                )
            )
        self.coefficients = tuple(coefficients)

    def forward(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        step_index: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the action sample after one diffusion update.

        Args:
            model_output: Decoder prediction shaped ``(batch, horizon, action_dim)``.
                Contains noise, clean actions or velocity, as selected by the
                scheduler's prediction type.
            sample: Current noisy actions, with the same shape as ``model_output``.
            step_index: Index into ``timesteps``; zero selects the first inference
                update. This is an index, not a diffusion timestep value.
            noise: Standard-normal values with the same shape as ``sample``.
                Required for DDPM; unused for deterministic DDIM.

        Returns:
            Updated action sample with the same shape as ``sample``.

        Raises:
            ValueError: If the prediction type is unsupported or DDPM noise is missing.
        """
        alpha, beta, original_scale, sample_scale, noise_scale = (
            torch.scalar_tensor(coefficient, dtype=torch.float32, device="cpu")
            for coefficient in self.coefficients[step_index]
        )
        if self.prediction_type == PredictionType.EPSILON.value:
            original = (
                sample - beta * model_output
            ) / alpha  # (batch, horizon, action_dim)
            epsilon = model_output  # (batch, horizon, action_dim)
        elif self.prediction_type == PredictionType.SAMPLE.value:
            original = model_output  # (batch, horizon, action_dim)
            epsilon = (sample - alpha * original) / beta  # (batch, horizon, action_dim)
        elif self.prediction_type == DIFFUSERS_VELOCITY_PREDICTION:
            original = (
                alpha * sample - beta * model_output
            )  # (batch, horizon, action_dim)
            epsilon = (
                alpha * model_output + beta * sample
            )  # (batch, horizon, action_dim)
        else:
            raise ValueError(
                f"Unsupported diffusion prediction type {self.prediction_type!r}."
            )
        if self.clip_sample:
            original = original.clamp(
                -self.clip_sample_range, self.clip_sample_range
            )  # (batch, horizon, action_dim)
        direction = (
            sample if self.stochastic else epsilon
        )  # (batch, horizon, action_dim)
        result = (
            original_scale * original + sample_scale * direction
        )  # (batch, horizon, action_dim)
        if self.stochastic:
            if noise is None:
                raise ValueError(
                    "DDPM inference requires an explicit step-noise tensor."
                )
            result = result + noise_scale * noise  # (batch, horizon, action_dim)
        return result
