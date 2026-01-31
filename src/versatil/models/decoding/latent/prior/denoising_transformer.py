"""Denoising Transformer Prior for variational models.

Implements a DiT-style transformer prior that learns p(z|s) using either diffusion
or flow matching, where z is the latent variable and s is the conditioning (observations).
"""

import torch
import torch.nn as nn

from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.detr_transformer.attention import FlashAttention
from versatil.models.layers.denoising.diffusion_process import (
    DiffusionSchedulerConfig,
    add_noise_to_tensor,
    create_noise_scheduler,
    sample_random_timesteps,
    setup_inference_timesteps,
    SchedulerType,
)
from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.denoising.ode_solvers import integrate_ode
from versatil.models.layers.positional_encoding.learned import (
    LearnedPositionalEncoding1D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
)
from versatil.models.layers.swiglu import SwiGLU
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
from versatil.models.decoding.constants import (
    BetaSchedule,
    DenoisingAlgorithm,
    ODESolver,
    PredictionType,
    PRIOR_PREDICTION_KEY,
    PRIOR_TARGET_KEY,
    CLASS_TOKEN_KEY,
)
from versatil.models.decoding.latent.prior.base_prior import PriorLatentEncoder


class DiTBlock(nn.Module):
    """Diffusion Transformer block.

    Uses pre-normalization architecture with adaptive layer normalization for
    timestep conditioning.

    Args:
        embedding_dimension: Hidden dimension of the transformer.
        number_of_heads: Number of attention heads.
        feedforward_dimension: Dimension of the feedforward network.
        timestep_dimension: Dimension of the timestep embedding.
        dropout: Dropout rate.
        activation: Activation function name.
        use_gating: Whether to use gating in AdaNorm (often referred to as AdaLNZeroNorm).
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        feedforward_dimension: int,
        timestep_dimension: int,
        dropout: float = 0.1,
        activation: str = ActivationFunction.SWIGLU.value,
        use_gating: bool = True,
    ):
        super().__init__()
        self.use_gating = use_gating
        self.self_attention = FlashAttention(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            dropout=dropout,
        )
        self.norm1 = AdaNorm(
            base_norm=nn.LayerNorm(embedding_dimension),
            condition_dim=timestep_dimension,
            feature_dim=embedding_dimension,
            use_gate=use_gating
        )
        self.norm2 = AdaNorm(
            base_norm=nn.LayerNorm(embedding_dimension),
            condition_dim=timestep_dimension,
            feature_dim=embedding_dimension,
            use_gate=use_gating
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if activation == ActivationFunction.SWIGLU.value:
            self.ffn = nn.Sequential(
                SwiGLU(embedding_dimension, feedforward_dimension),
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension),
            )
        else:
            activation_fn = ActivationFunction(activation).to_torch_activation()()
            self.ffn = nn.Sequential(
                nn.Linear(embedding_dimension, feedforward_dimension),
                activation_fn,
                nn.Dropout(dropout),
                nn.Linear(feedforward_dimension, embedding_dimension),
            )

    def forward(
        self,
        x: torch.Tensor,
        timestep_embed: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        positional_encoding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through the DiT block.

        Args:
            x: Input tokens (B, T, D).
            timestep_embed: Timestep embedding (B, D).
            key_padding_mask: Padding mask (B, T).
            positional_encoding: Positional encoding (B, T, D).

        Returns:
            Output tokens (B, T, D).
        """
        residual = x
        if self.use_gating:
            x, gate1 = self.norm1(x, timestep_embed)
        else:
            x = self.norm1(x, timestep_embed)
            gate1 = 1.0
        x = self.self_attention(
            query=x,
            key=x,
            value=x,
            query_positional_encoding=positional_encoding,
            key_positional_encoding=positional_encoding,
            key_padding_mask=key_padding_mask,
        )
        x = residual + gate1 * self.dropout1(x)
        residual = x
        if self.use_gating:
            x, gate2 = self.norm2(x, timestep_embed)
        else:
            x = self.norm2(x, timestep_embed)
            gate2 = 1.0
        x = self.ffn(x)
        x = residual + gate2 * self.dropout2(x)
        return x


class DenoisingTransformerPrior(PriorLatentEncoder):
    """DiT-style transformer prior supporting diffusion and flow matching.

    Uses a non-autoregressive transformer where the noisy latent z is treated
    as a CLS token appended to observation tokens. The transformer attends
    bidirectionally across all tokens, and the final CLS representation is
    projected to predict noise (diffusion) or velocity (flow matching).

    Args:
        latent_dimension: Dimension of latent variable z.
        embedding_dimension: Hidden dimension of the transformer.
        number_of_heads: Number of attention heads.
        number_of_layers: Number of DiT blocks.
        feedforward_dimension: Dimension of the feedforward network.
        device: Device to place prior on.
        observation_horizon: Observation history size.
        algorithm_type: Algorithm type ("diffusion" or "flow_matching").
        sigma: Noise level for flow matching (0 = deterministic OT).
        ode_solver: ODE solver for flow matching ("euler", "heun", or "rk4").
        num_train_timesteps: Number of diffusion timesteps during training.
        num_inference_steps: Number of denoising/integration steps.
        beta_start: Starting beta for noise schedule (diffusion).
        beta_end: Ending beta for noise schedule (diffusion).
        beta_schedule: Type of noise schedule (diffusion).
        scheduler_type: Diffusion scheduler type.
        prediction_type: What diffusion model predicts (epsilon, sample, velocity).
        clip_sample: Whether to clip samples during diffusion.
        variance_type: Variance type for DDPM scheduler.
        dropout: Dropout rate.
        activation: Activation function name.
        exclude_keys: Keys to exclude from observations.
    """

    def __init__(
        self,
        latent_dimension: int,
        embedding_dimension: int,
        number_of_heads: int,
        number_of_layers: int,
        feedforward_dimension: int,
        device: str,
        observation_horizon: int = 1,
        algorithm_type: str = DenoisingAlgorithm.FLOW_MATCHING.value,
        sigma: float = 0.0,
        ode_solver: str = ODESolver.EULER.value,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = BetaSchedule.SQUAREDCOS_CAP_V2.value,
        scheduler_type: str = SchedulerType.DDIM.value,
        prediction_type: str = PredictionType.EPSILON.value,
        clip_sample: bool = False,
        variance_type: str | None = None,
        dropout: float = 0.1,
        activation: str = ActivationFunction.SILU.value,
        use_gating: bool = True,
        exclude_keys: list[str] | None = None,
    ):
        super().__init__(latent_dimension=latent_dimension, device=device)
        self.embedding_dimension = embedding_dimension
        self.observation_horizon = observation_horizon
        self.algorithm_type = algorithm_type
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.exclude_keys = exclude_keys or []

        if algorithm_type == DenoisingAlgorithm.FLOW_MATCHING.value:
            from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
            self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
            self.ode_solver = ode_solver
            self.noise_scheduler = None
        elif algorithm_type == DenoisingAlgorithm.DIFFUSION.value:
            self.flow_matcher = None
            self.ode_solver = None
            self.prediction_type = prediction_type
            scheduler_config = DiffusionSchedulerConfig(
                scheduler_type=scheduler_type,
                num_train_timesteps=num_train_timesteps,
                num_inference_steps=num_inference_steps,
                beta_start=beta_start,
                beta_end=beta_end,
                beta_schedule=beta_schedule,
                prediction_type=prediction_type,
                clip_sample=clip_sample,
                variance_type=variance_type,
            )
            self.noise_scheduler = create_noise_scheduler(scheduler_config)
        else:
            raise ValueError(
                f"Unknown algorithm_type: {algorithm_type}. "
                f"Expected one of {[e.value for e in DenoisingAlgorithm]}"
            )

        self.timestep_embed = SinusoidalPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
            maximum_length=num_train_timesteps,
        )
        self.timestep_mlp = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.SiLU(),
            nn.Linear(embedding_dimension, embedding_dimension),
        )
        self.latent_input_proj = nn.Linear(latent_dimension, embedding_dimension)
        self.latent_output_proj = nn.Linear(embedding_dimension, latent_dimension)
        temporal_positional_encoding = None
        if self.observation_horizon > 1:
            temporal_positional_encoding = LearnedPositionalEncoding1D(
                embedding_dimension=embedding_dimension
            )
        self.input_builder = TransformerInputBuilder(
            embedding_dim=embedding_dimension,
            has_time_dim=self.observation_horizon > 1,
            spatial_positional_encoding_layer=SinusoidalPositionalEncoding2D(
                embedding_dimension=embedding_dimension, normalize=True
            ),
            flat_positional_encoding_layer=SinusoidalPositionalEncoding1D(
                embedding_dimension=embedding_dimension,
                maximum_length=1000,
            ),
            temporal_positional_encoding_layer=temporal_positional_encoding,
        )
        self.layers = nn.ModuleList([
            DiTBlock(
                embedding_dimension=embedding_dimension,
                number_of_heads=number_of_heads,
                feedforward_dimension=feedforward_dimension,
                timestep_dimension=embedding_dimension,
                dropout=dropout,
                activation=activation,
                use_gating=use_gating
            )
            for _ in range(number_of_layers)
        ])
        self.final_norm = nn.LayerNorm(embedding_dimension)
        self.to(torch.device(device))

    def _filter_observations(
        self, observations: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Filter out excluded keys from observations."""
        return {k: v for k, v in observations.items() if k not in self.exclude_keys}

    def _get_timestep_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute timestep embedding.

        Args:
            timesteps: Timestep indices (B,).

        Returns:
            Timestep embeddings (B, D).
        """
        timestep_embedding = self.timestep_embed(timesteps.unsqueeze(-1))  # (B, 1, D)
        timestep_embedding = timestep_embedding.squeeze(1)  # (B, D)
        return self.timestep_mlp(timestep_embedding)  # (B, D)

    def _get_timestep_embedding_continuous(self, continuous_time: torch.Tensor) -> torch.Tensor:
        """Compute timestep embedding for continuous time.

        Args:
            continuous_time: Continuous time values in [0, 1] (B,).

        Returns:
            Timestep embeddings (B, D).
        """
        scaled_timesteps = (continuous_time * (self.num_train_timesteps - 1)).long()  # (B,)
        return self._get_timestep_embedding(scaled_timesteps)  # (B, D)

    def _predict_from_tokens(
        self,
        noisy_latent: torch.Tensor,
        observations: dict[str, torch.Tensor],
        timestep_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Shared transformer forward logic.

        Args:
            noisy_latent: Noisy/interpolated latent (B, latent_dim).
            observations: Filtered observation features.
            timestep_embedding: Timestep embeddings (B, D).

        Returns:
            Model output (B, latent_dim).
        """
        latent_token = self.latent_input_proj(noisy_latent)  # (B, D)
        input_obs = observations.copy()
        input_obs[CLASS_TOKEN_KEY] = latent_token
        tokens, positional_encoding, padding_mask = self.input_builder(input_obs)  # (B, T+1, D)
        for layer in self.layers:
            tokens = layer(
                tokens,
                timestep_embed=timestep_embedding,
                key_padding_mask=padding_mask,
                positional_encoding=positional_encoding,
            )
        tokens = self.final_norm(tokens)  # (B, T+1, D)
        output_token = tokens[:, -1, :]  # (B, D)
        return self.latent_output_proj(output_token)  # (B, latent_dim)

    def forward(
        self,
        target_latents: torch.Tensor,
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute denoising predictions for training.

        Args:
            target_latents: Clean latent samples from posterior (B, latent_dim).
            observations: Dictionary of conditioning features.

        Returns:
            Dictionary with PRIOR_PREDICTION_KEY and PRIOR_TARGET_KEY.
        """
        batch_size = target_latents.shape[0]
        device = target_latents.device
        filtered_obs = self._filter_observations(observations)

        if self.algorithm_type == DenoisingAlgorithm.FLOW_MATCHING.value:
            noise = torch.randn_like(target_latents)  # (B, latent_dim)
            time, interpolated_latent, target_velocity = self.flow_matcher.sample_location_and_conditional_flow(
                x0=noise, x1=target_latents
            )  # time: (B,), interpolated_latent: (B, latent_dim), target_velocity: (B, latent_dim)
            timestep_embedding = self._get_timestep_embedding_continuous(time)  # (B, D)
            predicted_velocity = self._predict_from_tokens(
                noisy_latent=interpolated_latent,
                observations=filtered_obs,
                timestep_embedding=timestep_embedding,
            )  # (B, latent_dim)
            return {PRIOR_PREDICTION_KEY: predicted_velocity, PRIOR_TARGET_KEY: target_velocity}

        timesteps = sample_random_timesteps(
            batch_size=batch_size,
            num_train_timesteps=self.num_train_timesteps,
            device=device,
        )  # (B,)
        noisy_latents, noise = add_noise_to_tensor(
            clean=target_latents,
            noise_scheduler=self.noise_scheduler,
            timesteps=timesteps,
        )  # noisy_latents: (B, latent_dim), noise: (B, latent_dim)
        timestep_embedding = self._get_timestep_embedding(timesteps)  # (B, D)
        model_output = self._predict_from_tokens(
            noisy_latent=noisy_latents,
            observations=filtered_obs,
            timestep_embedding=timestep_embedding,
        )  # (B, latent_dim)
        return {PRIOR_PREDICTION_KEY: model_output, PRIOR_TARGET_KEY: noise}

    def sample_prior(
        self,
        batch_size: int,
        observations: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sample latent variable from learned prior.

        Args:
            batch_size: Number of samples to generate.
            observations: Dictionary of conditioning features.

        Returns:
            Sampled latent embeddings (batch_size, latent_dim).
        """
        device = next(self.parameters()).device
        if observations is None:
            observations = {}
        filtered_obs = self._filter_observations(observations)
        z = torch.randn(batch_size, self.latent_dimension, device=device)  # (B, latent_dim)

        if self.algorithm_type == DenoisingAlgorithm.FLOW_MATCHING.value:
            def velocity_fn(
                current_latent: torch.Tensor, current_time: torch.Tensor
            ) -> torch.Tensor:
                timestep_embedding = self._get_timestep_embedding_continuous(current_time)  # (B, D)
                return self._predict_from_tokens(
                    noisy_latent=current_latent,
                    observations=filtered_obs,
                    timestep_embedding=timestep_embedding,
                )  # (B, latent_dim)

            return integrate_ode(
                z_init=z,
                velocity_fn=velocity_fn,
                num_steps=self.num_inference_steps,
                solver=self.ode_solver,
            )  # (B, latent_dim)

        setup_inference_timesteps(self.noise_scheduler, self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)  # (B,)
            timestep_embedding = self._get_timestep_embedding(timesteps)  # (B, D)
            model_output = self._predict_from_tokens(
                noisy_latent=z,
                observations=filtered_obs,
                timestep_embedding=timestep_embedding,
            )  # (B, latent_dim)
            z = self.noise_scheduler.step(model_output, t, z).prev_sample  # (B, latent_dim)
        return z