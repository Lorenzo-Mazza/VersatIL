"""Flow Matching algorithm for action generation via continuous normalizing flows."""

import torch

from versatil.data.constants import SampleKey
from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.constants import DecoderOutputKey, ODESolver
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.decoding.decoders.factory.dit_block_action_transformer import (
    DiTBlockActionTransformer,
)
from versatil.models.layers.denoising.ode_solvers import integrate_ode
from versatil.models.layers.denoising.timestep_sampling import (
    TimestepSampler,
    sample_timesteps,
)


class FlowMatching(DecodingAlgorithm):
    """Flow Matching algorithm for action prediction.

    Trains a model to predict velocity fields that transport samples from a noise
    distribution to the target action distribution. Uses Conditional Flow Matching (CFM)
    with optimal transport paths.

    During training, samples a time t ∈ [0,1] and trains the model to predict the
    velocity field u_t = dx/dt that moves from noise (t=0) to actions (t=1).

    During inference, integrates the learned velocity field using an ODE solver
    (Euler, Heun, or RK4) to generate actions.

    Args:
        sigma: Noise level for conditional flow matching (0 = deterministic OT).
        num_inference_steps: Number of integration steps during inference.
        ode_solver: ODE solver to use ("euler", "heun", or "rk4").
        timestep_sampler: Timestep sampling strategy.
        logit_mean: Mean for logit-normal (shifts mode; 0 centers at t=0.5).
        logit_std: Std for logit-normal (smaller = more concentrated).
        beta_alpha: First shape parameter for Beta distribution (pi0 uses 1.5).
        beta_beta: Second shape parameter for Beta distribution (pi0 uses 1.0).
        max_timestep: Upper bound s for Beta sampling (pi0 uses 0.999).
    """

    def __init__(
        self,
        sigma: float = 0.0,
        num_inference_steps: int = 10,
        ode_solver: str = ODESolver.EULER.value,
        timestep_sampler: str = TimestepSampler.BETA.value,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        max_timestep: float = 0.999,
    ):
        """Initialize Flow Matching algorithm."""
        super().__init__()

        # Lazy import: torchcfm → ot → geomloss → pykeops triggers CUDA JIT at import time.
        from torchcfm.conditional_flow_matching import (  # noqa: PLC0415
            ConditionalFlowMatcher,
        )

        self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
        self.num_inference_steps = num_inference_steps
        self.ode_solver = ode_solver
        self.timestep_sampler = timestep_sampler
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.max_timestep = max_timestep

        valid_solvers = [e.value for e in ODESolver]
        if self.ode_solver not in valid_solvers:
            raise ValueError(
                f"Unknown ODE solver: {ode_solver}. Expected one of {valid_solvers}"
            )
        valid_samplers = [e.value for e in TimestepSampler]
        if self.timestep_sampler not in valid_samplers:
            raise ValueError(
                f"Unknown timestep sampler: {timestep_sampler}. Expected one of {valid_samplers}"
            )

    def forward(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass during training.

        Samples a time t and trains the network to predict the velocity field
        that transports noise to the target actions.

        Args:
            network: The action decoder network module (should support time conditioning)
            features: Dictionary of encoded features from the encoding pipeline.
            actions: Dictionary of ground truth actions. Required for flow matching training.
                Expected keys depend on action space (e.g., 'position_action', 'gripper_action')

        Returns:
            Decoder output dictionary containing:
                - Velocity predictions for each action component
                - 'target_velocity': The true velocity field
                - 'time': The sampled time values
                - All action keys with '_pred' suffix for predictions

        Raises:
            ValueError: If actions are not provided (required for flow matching training)
        """
        if actions is None:
            raise ValueError("Flow Matching algorithm requires actions during training")

        interpolated_actions = {}
        target_velocities = {}
        noise = {}
        times, is_pad = None, None

        for key, action in actions.items():
            if key == SampleKey.IS_PAD_ACTION.value:
                is_pad = action
                continue
            noise[key] = torch.randn_like(action.float(), device=action.device)
            if times is None:
                times = sample_timesteps(
                    batch_size=action.shape[0],
                    device=action.device,
                    sampler=self.timestep_sampler,
                    logit_mean=self.logit_mean,
                    logit_std=self.logit_std,
                    beta_alpha=self.beta_alpha,
                    beta_beta=self.beta_beta,
                    max_timestep=self.max_timestep,
                )
            epsilon = torch.randn_like(action.float())
            x_t = self.flow_matcher.sample_xt(
                x0=noise[key], x1=action, t=times, epsilon=epsilon
            )
            u_t = self.flow_matcher.compute_conditional_flow(
                x0=noise[key], x1=action, t=times, xt=x_t
            )
            interpolated_actions[key] = x_t
            target_velocities[key] = u_t

        features_with_time = {**features, DecoderOutputKey.TIMESTEP.value: times}
        predictions = network(features=features_with_time, actions=interpolated_actions)
        return {
            **predictions,
            DecoderOutputKey.TARGET_VELOCITY.value: target_velocities,
            DecoderOutputKey.NOISE.value: noise,
            DecoderOutputKey.TIMESTEP.value: times,
            SampleKey.IS_PAD_ACTION.value: is_pad,
        }

    def predict(
        self,
        network: ActionDecoder,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Inference pass using ODE integration.

        Generates actions by integrating the learned velocity field from noise (t=0)
        to actions (t=1) using an ODE solver.

        Args:
            network: The action decoder network module.
            features: Dict of encoded features from encoding pipeline.

        Returns:
            Decoder output dictionary containing action predictions.
        """
        first_feature = next(iter(features.values()))
        batch_size = first_feature.shape[0]
        device = first_feature.device
        dtype = first_feature.dtype

        if isinstance(network, DiTBlockActionTransformer):
            network.enable_encoder_cache()

        trajectory: dict[str, torch.Tensor] = {}
        for key, meta in network.action_space.actions_metadata.items():
            if not meta.requires_prediction_head:
                continue
            trajectory[key] = torch.randn(
                batch_size,
                network.prediction_horizon,
                meta.prediction_dimension,
                device=device,
                dtype=dtype,
            )  # (B, H, D_k)
        action_keys = sorted(trajectory.keys())
        shapes = {k: trajectory[k].shape for k in action_keys}
        flat_action_dimensions = {
            key: shapes[key][1] * shapes[key][2] for key in action_keys
        }
        stacked = torch.cat(
            [trajectory[k].flatten(1) for k in action_keys], dim=-1
        )  # (B, H*sum(D_k))

        def velocity_wrapper(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Wrapper to compute velocities from stacked action representation, used as Callable in `integrate_ode`."""
            current_trajectory: dict[str, torch.Tensor] = {}
            offset = 0
            for current_key in action_keys:
                flat_action_size = flat_action_dimensions[current_key]
                current_trajectory[current_key] = z[
                    :, offset : offset + flat_action_size
                ].view(shapes[current_key])  # (B, H, D_k)
                offset += flat_action_size
            features_with_time = {**features, DecoderOutputKey.TIMESTEP.value: t}
            velocities = network(
                features=features_with_time, actions=current_trajectory
            )
            return torch.cat(
                [velocities[current_key].flatten(1) for current_key in action_keys],
                dim=-1,
            )  # (B, H*sum(D_k))

        stacked_final = integrate_ode(
            z_init=stacked,
            velocity_fn=velocity_wrapper,
            num_steps=self.num_inference_steps,
            solver=self.ode_solver,
        )  # (B, H*sum(D_k))
        result: dict[str, torch.Tensor] = {}
        current_offset = 0
        for key in action_keys:
            flat_action_dimension = shapes[key][1] * shapes[key][2]
            result[key] = stacked_final[
                :, current_offset : current_offset + flat_action_dimension
            ].view(shapes[key])  # (B, H, D_k)
            current_offset += flat_action_dimension

        if isinstance(network, DiTBlockActionTransformer):
            network.disable_encoder_cache()

        return result
