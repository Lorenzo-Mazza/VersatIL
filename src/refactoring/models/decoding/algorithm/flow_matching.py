"""Flow Matching algorithm for action generation via continuous normalizing flows."""


import torch
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from refactoring.data.constants import IS_PAD_ACTION_KEY
from refactoring.models.decoding.algorithm.base import DecodingAlgorithm
from refactoring.models.decoding.constants import ODESolver, TARGET_VELOCITY_KEY, TIMESTEP_KEY, NOISE_KEY
from refactoring.models.decoding.decoders.base import ActionDecoder


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
        """Inference/prediction pass.

        Generates actions by integrating the learned velocity field from noise (t=0)
        to actions (t=1) using an ODE solver.

        Args:
            network: The action decoder network module
            features: Dict of encoded features from encoding pipeline

        Returns:
            Decoder output dictionary containing action predictions.
        """
        first_feature = next(iter(features.values()))
        batch_size = first_feature.shape[0]
        device = first_feature.device
        dtype = first_feature.dtype
        # Initialize trajectory with random noise
        trajectory = {}
        for key, meta in network.action_space.actions_metadata.items():
            trajectory[key] = torch.randn(
                batch_size, network.prediction_horizon, meta.prediction_dimension,  # type: ignore[arg-type]
                device=device, dtype=dtype
            )
        # Integration step size
        dt = 1.0 / self.num_inference_steps
        # Integrate from t=0 to t=1
        for step in range(self.num_inference_steps):
            t = step / self.num_inference_steps
            t_tensor = torch.full((batch_size,), t, device=device, dtype=dtype)
            features_with_time = {**features, TIMESTEP_KEY: t_tensor}
            # Compute velocity at current state
            if self.ode_solver == ODESolver.EULER.value:
                # Simple Euler integration: x_{t+dt} = x_t + dt * v_t
                velocities = network(features_with_time, trajectory)
                for key in trajectory:
                    if key in velocities:
                        trajectory[key] = trajectory[key] + dt * velocities[key]

            elif self.ode_solver == ODESolver.HEUN.value:
                # Heun's method (2nd order): x_{t+dt} = x_t + dt * (v_t + v_{t+dt}) / 2
                # First, compute v_t
                v_t = network(features_with_time, trajectory)

                # Compute tentative x_{t+dt} using Euler
                trajectory_tentative = {}
                for key in trajectory:
                    if key in v_t:
                        trajectory_tentative[key] = trajectory[key] + dt * v_t[key]
                    else:
                        trajectory_tentative[key] = trajectory[key]

                # Compute v_{t+dt}
                t_next = (step + 1) / self.num_inference_steps
                t_next_tensor = torch.full((batch_size,), t_next, device=device, dtype=dtype)
                features_with_time_next = {**features, TIMESTEP_KEY: t_next_tensor}
                v_t_next = network(features_with_time_next, trajectory_tentative)

                # Update using average of velocities
                for key in trajectory:
                    if key in v_t and key in v_t_next:
                        trajectory[key] = trajectory[key] + dt * (v_t[key] + v_t_next[key]) / 2

            elif self.ode_solver == ODESolver.RK4.value:
                # 4th order Runge-Kutta
                # k1 = v(t, action_embedding)
                k1 = network(features_with_time, trajectory)

                # k2 = v(t + dt/2, action_embedding + dt*k1/2)
                trajectory_k2 = {}
                for key in trajectory:
                    if key in k1:
                        trajectory_k2[key] = trajectory[key] + dt * k1[key] / 2
                    else:
                        trajectory_k2[key] = trajectory[key]

                t_mid = t + dt / 2
                t_mid_tensor = torch.full((batch_size,), t_mid, device=device, dtype=dtype)
                features_with_time_mid = {**features, TIMESTEP_KEY: t_mid_tensor}
                k2 = network(features_with_time_mid, trajectory_k2)

                # k3 = v(t + dt/2, action_embedding + dt*k2/2)
                trajectory_k3 = {}
                for key in trajectory:
                    if key in k2:
                        trajectory_k3[key] = trajectory[key] + dt * k2[key] / 2
                    else:
                        trajectory_k3[key] = trajectory[key]
                k3 = network(features_with_time_mid, trajectory_k3)

                # k4 = v(t + dt, action_embedding + dt*k3)
                trajectory_k4 = {}
                for key in trajectory:
                    if key in k3:
                        trajectory_k4[key] = trajectory[key] + dt * k3[key]
                    else:
                        trajectory_k4[key] = trajectory[key]

                t_next = t + dt
                t_next_tensor = torch.full((batch_size,), t_next, device=device, dtype=dtype)
                features_with_time_next = {**features, TIMESTEP_KEY: t_next_tensor}
                k4 = network(features_with_time_next, trajectory_k4)

                # Update: x_{t+dt} = x_t + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
                for key in trajectory:
                    if key in k1 and key in k2 and key in k3 and key in k4:
                        trajectory[key] = (
                            trajectory[key]
                            + dt * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key]) / 6
                        )

        # Return final trajectory
        return trajectory
