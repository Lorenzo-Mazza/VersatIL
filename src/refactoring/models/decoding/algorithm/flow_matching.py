"""Flow Matching algorithm for action generation via continuous normalizing flows."""

import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from refactoring.data.constants import IS_PAD_ACTION_KEY
from refactoring.models.decoding.algorithm.base import DecodingAlgorithm
from refactoring.models.decoding.constants import (
    ODESolver,
    TARGET_VELOCITY_KEY,
    TIMESTEP_KEY,
    NOISE_KEY,
)
from refactoring.models.decoding.decoders.base import ActionDecoder
from refactoring.models.layers.ode_solvers import integrate_ode


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
        sigma: Noise level for conditional flow matching (0 = deterministic OT)
        num_inference_steps: Number of integration steps during inference
        ode_solver: ODE solver to use ("euler", "heun", or "rk4")
    """

    def __init__(
        self,
        sigma: float = 0.0,
        num_inference_steps: int = 10,
        ode_solver: str = ODESolver.EULER.value,
    ):
        """Initialize Flow Matching algorithm."""
        super().__init__()

        self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)
        self.num_inference_steps = num_inference_steps
        self.ode_solver = ode_solver
        valid_solvers = [e.value for e in ODESolver]
        if self.ode_solver not in valid_solvers:
            raise ValueError(
                f"Unknown ODE solver: {ode_solver}. Expected one of {valid_solvers}"
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
        # Sample flow for each action component
        interpolated_actions = {}
        target_velocities = {}
        times, noise, is_pad = None, None, None
        for key, action in actions.items():
            if key == IS_PAD_ACTION_KEY:
                is_pad = action
                continue  # Skip padding mask
            # Sample noise from standard normal
            noise = torch.randn_like(action.float(), device=action.device)
            # Sample time, interpolated state, and target velocity
            t, x_t, u_t = self.flow_matcher.sample_location_and_conditional_flow(
                x0=noise, x1=action
            )
            interpolated_actions[key] = x_t
            target_velocities[key] = u_t
            # Times are the same for all action components
            if times is None:
                times = t
        # Add time to features for conditioning
        # Time is in [0, 1] and has shape (batch_size,)
        features_with_time = {**features, TIMESTEP_KEY: times}
        # Predict velocity field
        predictions = network(features_with_time, interpolated_actions)
        # Return predictions and targets
        return {
            **predictions,
            TARGET_VELOCITY_KEY: target_velocities,
            NOISE_KEY: noise,
            TIMESTEP_KEY: times,
            IS_PAD_ACTION_KEY: is_pad,
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
        trajectory: dict[str, torch.Tensor] = {}
        for key, meta in network.action_space.actions_metadata.items():
            trajectory[key] = torch.randn(
                batch_size,
                network.prediction_horizon,
                meta.prediction_dimension,
                device=device,
                dtype=dtype,
            )  # (B, H, D_k)
        keys = sorted(trajectory.keys())
        shapes = {k: trajectory[k].shape for k in keys}
        stacked = torch.cat(
            [trajectory[k].flatten(1) for k in keys], dim=-1
        )  # (B, H*sum(D_k))

        def velocity_fn(z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            traj: dict[str, torch.Tensor] = {}
            offset = 0
            for k in keys:
                flat_dim = shapes[k][1] * shapes[k][2]
                traj[k] = z[:, offset : offset + flat_dim].view(shapes[k])  # (B, H, D_k)
                offset += flat_dim

            features_with_time = {**features, TIMESTEP_KEY: t}
            velocities = network(features_with_time, traj)

            return torch.cat(
                [velocities[k].flatten(1) for k in keys], dim=-1
            )  # (B, H*sum(D_k))

        stacked_final = integrate_ode(
            z_init=stacked,
            velocity_fn=velocity_fn,
            num_steps=self.num_inference_steps,
            solver=self.ode_solver,
        )  # (B, H*sum(D_k))

        result: dict[str, torch.Tensor] = {}
        offset = 0
        for k in keys:
            flat_dim = shapes[k][1] * shapes[k][2]
            result[k] = stacked_final[:, offset : offset + flat_dim].view(
                shapes[k]
            )  # (B, H, D_k)
            offset += flat_dim

        return result
