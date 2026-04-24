"""ODE solvers for flow matching integration.

Provides reusable ODE solver functions for integrating velocity fields
in flow matching models. Used for both latent space priors and action decoders.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from versatil.models.decoding.constants import ODESolver


def euler_step(z: Tensor, velocity: Tensor, dt: float) -> Tensor:
    """Single Euler integration step.

    Args:
        z: Current state tensor (B, D)
        velocity: Velocity at current state (B, D)
        dt: Integration step size

    Returns:
        Next state: z_{t+dt} = z_t + dt * v_t
    """
    return z + dt * velocity


def heun_step(
    z: Tensor,
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    t: float,
    dt: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Heun's method (2nd order) integration step.

    Args:
        z: Current state tensor (B, D)
        velocity_fn: Function that computes velocity given (z, t_tensor)
        t: Current time in [0, 1]
        dt: Integration step size
        batch_size: Batch size
        device: Device for tensors
        dtype: Data type for tensors

    Returns:
        Next state: z_{t+dt} = z_t + dt * (v_t + v_{t+dt}) / 2
    """
    t_tensor = torch.full((batch_size,), t, device=device, dtype=dtype)
    v_t = velocity_fn(z, t_tensor)
    z_tentative = z + dt * v_t
    t_next = t + dt
    t_next_tensor = torch.full((batch_size,), t_next, device=device, dtype=dtype)
    v_t_next = velocity_fn(z_tentative, t_next_tensor)
    return z + dt * (v_t + v_t_next) / 2


def rk4_step(
    z: Tensor,
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    t: float,
    dt: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """4th order Runge-Kutta integration step.

    Args:
        z: Current state tensor (B, D)
        velocity_fn: Function that computes velocity given (z, t_tensor)
        t: Current time in [0, 1]
        dt: Integration step size
        batch_size: Batch size
        device: Device for tensors
        dtype: Data type for tensors

    Returns:
        Next state using RK4: z_{t+dt} = z_t + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    """
    t_tensor = torch.full((batch_size,), t, device=device, dtype=dtype)
    k1 = velocity_fn(z, t_tensor)
    t_mid = t + dt / 2
    t_mid_tensor = torch.full((batch_size,), t_mid, device=device, dtype=dtype)
    k2 = velocity_fn(z + dt * k1 / 2, t_mid_tensor)
    k3 = velocity_fn(z + dt * k2 / 2, t_mid_tensor)
    t_next = t + dt
    t_next_tensor = torch.full((batch_size,), t_next, device=device, dtype=dtype)
    k4 = velocity_fn(z + dt * k3, t_next_tensor)
    return z + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def integrate_ode(
    z_init: Tensor,
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    num_steps: int,
    solver: str = ODESolver.EULER.value,
) -> Tensor:
    """Integrate ODE from t=0 to t=1 using specified solver.

    Args:
        z_init: Initial state tensor (B, D)
        velocity_fn: Function that computes velocity given (z, t_tensor)
            where t_tensor has shape (B,) with values in [0, 1]
        num_steps: Number of integration steps
        solver: ODE solver type ("euler", "heun", or "rk4")

    Returns:
        Final state tensor (B, D) after integration from t=0 to t=1

    Raises:
        ValueError: If num_steps is not positive or solver is not recognized.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    dt = 1.0 / num_steps
    z = z_init
    batch_size = z.shape[0]
    device = z.device
    dtype = z.dtype

    for step in range(num_steps):
        t = step / num_steps

        if solver == ODESolver.EULER.value:
            t_tensor = torch.full((batch_size,), t, device=device, dtype=dtype)
            v = velocity_fn(z, t_tensor)
            z = euler_step(z, v, dt)
        elif solver == ODESolver.HEUN.value:
            z = heun_step(z, velocity_fn, t, dt, batch_size, device, dtype)
        elif solver == ODESolver.RK4.value:
            z = rk4_step(z, velocity_fn, t, dt, batch_size, device, dtype)
        else:
            raise ValueError(
                f"Unknown ODE solver: {solver}. "
                f"Expected one of {[e.value for e in ODESolver]}"
            )

    return z
