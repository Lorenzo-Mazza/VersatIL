"""Tests for versatil.models.layers.denoising.ode_solvers module."""
import math
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest
import torch

from versatil.models.decoding.constants import ODESolver
from versatil.models.layers.denoising.ode_solvers import (
    euler_step,
    heun_step,
    integrate_ode,
    rk4_step,
)


class TestEulerStep:

    def test_output_shape_matches_input(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        state = flat_tensor_factory(batch_size=3, feature_dimension=8)
        velocity = torch.ones_like(state)
        result = euler_step(z=state, velocity=velocity, dt=0.1)
        assert result.shape == (3, 8)

    def test_exact_formula_z_plus_dt_times_velocity(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        state = flat_tensor_factory(batch_size=2, feature_dimension=4)
        velocity = torch.ones_like(state) * 3.0
        step_size = 0.25
        result = euler_step(z=state, velocity=velocity, dt=step_size)
        expected = state + step_size * velocity
        assert torch.allclose(result, expected, atol=1e-7)

    def test_zero_velocity_returns_same_state(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        state = flat_tensor_factory(batch_size=2, feature_dimension=4)
        velocity = torch.zeros_like(state)
        result = euler_step(z=state, velocity=velocity, dt=0.5)
        assert torch.allclose(result, state, atol=1e-7)

    def test_zero_dt_returns_same_state(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        state = flat_tensor_factory(batch_size=2, feature_dimension=4)
        velocity = torch.ones_like(state) * 5.0
        result = euler_step(z=state, velocity=velocity, dt=0.0)
        assert torch.allclose(result, state, atol=1e-7)

    def test_negative_velocity_decreases_state(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
    ):
        state = torch.ones(2, 4)
        velocity = torch.full((2, 4), -2.0)
        step_size = 0.5
        result = euler_step(z=state, velocity=velocity, dt=step_size)
        expected = state + step_size * velocity  # 1.0 + 0.5 * (-2.0) = 0.0
        assert torch.allclose(result, expected, atol=1e-7)
        assert torch.allclose(result, torch.zeros(2, 4), atol=1e-7)


class TestHeunStep:

    def test_output_shape_matches_input(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
        velocity_field_factory: Callable,
    ):
        state = flat_tensor_factory(batch_size=3, feature_dimension=8)
        velocity_fn = velocity_field_factory(
            field_type="constant", constant_velocity=1.0,
        )
        result = heun_step(
            z=state,
            velocity_fn=velocity_fn,
            t=0.0,
            dt=0.1,
            batch_size=3,
            device=state.device,
            dtype=state.dtype,
        )
        assert result.shape == (3, 8)

    def test_constant_velocity_exact_integration(
        self,
        velocity_field_factory: Callable,
    ):
        # For constant v=2.0: both evaluations give 2.0
        # Heun: z + dt * (2.0 + 2.0) / 2 = z + dt * 2.0
        batch_size = 2
        feature_dim = 4
        state = torch.zeros(batch_size, feature_dim)
        constant_velocity = 2.0
        velocity_fn = velocity_field_factory(
            field_type="constant", constant_velocity=constant_velocity,
        )
        step_size = 0.5
        result = heun_step(
            z=state,
            velocity_fn=velocity_fn,
            t=0.0,
            dt=step_size,
            batch_size=batch_size,
            device=state.device,
            dtype=state.dtype,
        )
        expected = state + step_size * constant_velocity
        assert torch.allclose(result, expected, atol=1e-6)

    def test_time_dependent_field_exact_trapezoid(
        self,
        velocity_field_factory: Callable,
    ):
        # v(z, t) = t, integrating from t=0.0 with dt=0.5
        # Heun: v_t = v(z, 0.0) = 0.0
        #        z_tent = z + 0.5 * 0.0 = z = 0
        #        v_next = v(z_tent, 0.5) = 0.5
        #        z_new = z + 0.5 * (0.0 + 0.5) / 2 = 0.125
        batch_size = 1
        feature_dim = 2
        state = torch.zeros(batch_size, feature_dim)
        velocity_fn = velocity_field_factory(field_type="time_dependent")
        result = heun_step(
            z=state,
            velocity_fn=velocity_fn,
            t=0.0,
            dt=0.5,
            batch_size=batch_size,
            device=state.device,
            dtype=state.dtype,
        )
        expected = torch.full((batch_size, feature_dim), 0.125)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_linear_field_single_step_matches_formula(
        self,
        velocity_field_factory: Callable,
    ):
        # v(z, t) = z, starting at z=1.0, dt=0.1
        # Heun: v_t = z = 1.0
        #        z_tent = 1.0 + 0.1 * 1.0 = 1.1
        #        v_next = z_tent = 1.1
        #        z_new = 1.0 + 0.1 * (1.0 + 1.1) / 2 = 1.0 + 0.1 * 1.05 = 1.105
        batch_size = 1
        feature_dim = 1
        state = torch.ones(batch_size, feature_dim)
        velocity_fn = velocity_field_factory(field_type="linear")
        step_size = 0.1
        result = heun_step(
            z=state,
            velocity_fn=velocity_fn,
            t=0.0,
            dt=step_size,
            batch_size=batch_size,
            device=state.device,
            dtype=state.dtype,
        )
        expected = torch.full((batch_size, feature_dim), 1.105)
        assert torch.allclose(result, expected, atol=1e-6)


class TestRK4Step:

    def test_output_shape_matches_input(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
        velocity_field_factory: Callable,
    ):
        state = flat_tensor_factory(batch_size=3, feature_dimension=8)
        velocity_fn = velocity_field_factory(
            field_type="constant", constant_velocity=1.0,
        )
        result = rk4_step(
            z=state,
            velocity_fn=velocity_fn,
            t=0.0,
            dt=0.1,
            batch_size=3,
            device=state.device,
            dtype=state.dtype,
        )
        assert result.shape == (3, 8)

    def test_constant_velocity_exact_integration(
        self,
        velocity_field_factory: Callable,
    ):
        batch_size = 2
        feature_dim = 4
        state = torch.zeros(batch_size, feature_dim)
        constant_velocity = 3.0
        velocity_fn = velocity_field_factory(
            field_type="constant", constant_velocity=constant_velocity,
        )
        step_size = 0.25
        result = rk4_step(
            z=state,
            velocity_fn=velocity_fn,
            t=0.0,
            dt=step_size,
            batch_size=batch_size,
            device=state.device,
            dtype=state.dtype,
        )
        expected = state + step_size * constant_velocity
        assert torch.allclose(result, expected, atol=1e-6)

    def test_time_dependent_field_quadratic_integration(
        self,
        velocity_field_factory: Callable,
    ):
        # v(z, t) = t, exact integral from t=0 to t=dt is dt^2/2
        # RK4 is exact for polynomials up to degree 3, so this should be exact
        batch_size = 1
        feature_dim = 2
        state = torch.zeros(batch_size, feature_dim)
        velocity_fn = velocity_field_factory(field_type="time_dependent")
        step_size = 0.5
        result = rk4_step(
            z=state,
            velocity_fn=velocity_fn,
            t=0.0,
            dt=step_size,
            batch_size=batch_size,
            device=state.device,
            dtype=state.dtype,
        )
        # Exact: integral of t from 0 to 0.5 = 0.5^2 / 2 = 0.125
        expected = torch.full((batch_size, feature_dim), 0.125)
        assert torch.allclose(result, expected, atol=1e-7)

    def test_linear_field_single_step_matches_formula(
        self,
        velocity_field_factory: Callable,
    ):
        # v(z, t) = z, starting at z=1.0, dt=0.1
        # k1 = 1.0
        # k2 = v(1.0 + 0.1*1.0/2) = v(1.05) = 1.05
        # k3 = v(1.0 + 0.1*1.05/2) = v(1.0525) = 1.0525
        # k4 = v(1.0 + 0.1*1.0525) = v(1.10525) = 1.10525
        # z_new = 1.0 + 0.1 * (1.0 + 2*1.05 + 2*1.0525 + 1.10525) / 6
        #       = 1.0 + 0.1 * 6.31025 / 6
        #       = 1.0 + 0.10517083...
        #       = 1.10517083...
        # Compare with exact e^0.1 = 1.10517091808...
        batch_size = 1
        feature_dim = 1
        state = torch.ones(batch_size, feature_dim)
        velocity_fn = velocity_field_factory(field_type="linear")
        step_size = 0.1
        result = rk4_step(
            z=state,
            velocity_fn=velocity_fn,
            t=0.0,
            dt=step_size,
            batch_size=batch_size,
            device=state.device,
            dtype=state.dtype,
        )
        exact = math.exp(0.1)
        assert abs(result.item() - exact) < 1e-6


class TestIntegrateODE:

    @pytest.mark.parametrize("solver", [ODESolver.EULER.value, ODESolver.HEUN.value, ODESolver.RK4.value])
    def test_output_shape_matches_input(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
        velocity_field_factory: Callable,
        solver: str,
    ):
        state = flat_tensor_factory(batch_size=3, feature_dimension=8)
        velocity_fn = velocity_field_factory(
            field_type="constant", constant_velocity=1.0,
        )
        result = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=10,
            solver=solver,
        )
        assert result.shape == (3, 8)

    @pytest.mark.parametrize("solver", [ODESolver.EULER.value, ODESolver.HEUN.value, ODESolver.RK4.value])
    def test_constant_velocity_integration_exact_for_all_solvers(
        self,
        velocity_field_factory: Callable,
        solver: str,
    ):
        # v = 1.0, integrating from t=0 to t=1 starting at z=0 gives z=1
        # All solvers should be exact for constant velocity
        batch_size = 2
        feature_dim = 4
        state = torch.zeros(batch_size, feature_dim)
        constant_velocity = 1.0
        velocity_fn = velocity_field_factory(
            field_type="constant", constant_velocity=constant_velocity,
        )
        result = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=10,
            solver=solver,
        )
        expected = torch.ones(batch_size, feature_dim) * constant_velocity
        # Constant velocity is exact regardless of step count for all solvers
        assert torch.allclose(result, expected, atol=1e-6)

    @pytest.mark.parametrize(
        "solver, expectation",
        [
            (ODESolver.EULER.value, does_not_raise()),
            (ODESolver.HEUN.value, does_not_raise()),
            (ODESolver.RK4.value, does_not_raise()),
            (
                "midpoint",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Unknown ODE solver: midpoint. "
                        f"Expected one of {[e.value for e in ODESolver]}"
                    ),
                ),
            ),
        ],
    )
    def test_solver_validation(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
        velocity_field_factory: Callable,
        solver: str,
        expectation,
    ):
        state = flat_tensor_factory(batch_size=2, feature_dimension=4)
        velocity_fn = velocity_field_factory(field_type="constant")
        with expectation:
            result = integrate_ode(
                z_init=state,
                velocity_fn=velocity_fn,
                num_steps=5,
                solver=solver,
            )
            assert result.shape == state.shape

    def test_dopri5_solver_raises_as_unimplemented(
        self,
        flat_tensor_factory: Callable[..., torch.Tensor],
        velocity_field_factory: Callable,
    ):
        # DOPRI5 is in the ODESolver enum but not implemented in integrate_ode
        state = flat_tensor_factory(batch_size=2, feature_dimension=4)
        velocity_fn = velocity_field_factory(field_type="constant")
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unknown ODE solver: {ODESolver.DOPRI5.value}. "
                f"Expected one of {[e.value for e in ODESolver]}"
            ),
        ):
            integrate_ode(
                z_init=state,
                velocity_fn=velocity_fn,
                num_steps=5,
                solver=ODESolver.DOPRI5.value,
            )

    def test_time_dependent_velocity_rk4_exact(
        self,
        velocity_field_factory: Callable,
    ):
        # v(z, t) = t, starting at z=0 => z(1) = integral of t from 0 to 1 = 0.5
        # RK4 is exact for polynomials up to degree 3
        batch_size = 2
        feature_dim = 3
        state = torch.zeros(batch_size, feature_dim)
        velocity_fn = velocity_field_factory(field_type="time_dependent")
        result = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=10,
            solver=ODESolver.RK4.value,
        )
        expected = torch.full((batch_size, feature_dim), 0.5)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_nonzero_initial_state_propagated(
        self,
        velocity_field_factory: Callable,
    ):
        # v = 2.0, z(0) = 5.0 => z(1) = 5.0 + 2.0 = 7.0
        batch_size = 2
        feature_dim = 3
        state = torch.full((batch_size, feature_dim), 5.0)
        velocity_fn = velocity_field_factory(
            field_type="constant", constant_velocity=2.0,
        )
        result = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=10,
            solver=ODESolver.EULER.value,
        )
        expected = torch.full((batch_size, feature_dim), 7.0)
        assert torch.allclose(result, expected, atol=1e-6)


class TestSolverAccuracyComparison:

    def test_heun_more_accurate_than_euler_for_exponential_ode(
        self,
        velocity_field_factory: Callable,
    ):
        # dz/dt = z with z(0) = 1 => z(1) = e
        batch_size = 1
        feature_dim = 1
        state = torch.ones(batch_size, feature_dim)
        velocity_fn = velocity_field_factory(field_type="linear")
        num_steps = 20
        exact_solution = math.e

        result_euler = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=num_steps,
            solver=ODESolver.EULER.value,
        )
        result_heun = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=num_steps,
            solver=ODESolver.HEUN.value,
        )
        error_euler = abs(result_euler.item() - exact_solution)
        error_heun = abs(result_heun.item() - exact_solution)
        assert error_heun < error_euler

    def test_rk4_more_accurate_than_heun_for_exponential_ode(
        self,
        velocity_field_factory: Callable,
    ):
        # dz/dt = z with z(0) = 1 => z(1) = e
        batch_size = 1
        feature_dim = 1
        state = torch.ones(batch_size, feature_dim)
        velocity_fn = velocity_field_factory(field_type="linear")
        num_steps = 10
        exact_solution = math.e

        result_heun = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=num_steps,
            solver=ODESolver.HEUN.value,
        )
        result_rk4 = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=num_steps,
            solver=ODESolver.RK4.value,
        )
        error_heun = abs(result_heun.item() - exact_solution)
        error_rk4 = abs(result_rk4.item() - exact_solution)
        assert error_rk4 < error_heun

    def test_rk4_converges_to_exponential_solution(
        self,
        velocity_field_factory: Callable,
    ):
        # dz/dt = z with z(0) = 1 => z(1) = e ~= 2.71828
        batch_size = 1
        feature_dim = 1
        state = torch.ones(batch_size, feature_dim)
        velocity_fn = velocity_field_factory(field_type="linear")
        result = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=50,
            solver=ODESolver.RK4.value,
        )
        assert abs(result.item() - math.e) < 1e-6

    def test_euler_converges_with_many_steps(
        self,
        velocity_field_factory: Callable,
    ):
        batch_size = 1
        feature_dim = 1
        state = torch.ones(batch_size, feature_dim)
        velocity_fn = velocity_field_factory(field_type="linear")
        result = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=10000,
            solver=ODESolver.EULER.value,
        )
        assert abs(result.item() - math.e) < 0.01

    def test_euler_error_scales_with_step_size(
        self,
        velocity_field_factory: Callable,
    ):
        # Euler is O(h) globally, so halving step count should roughly halve error
        batch_size = 1
        feature_dim = 1
        state = torch.ones(batch_size, feature_dim)
        velocity_fn = velocity_field_factory(field_type="linear")
        exact_solution = math.e

        result_coarse = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=100,
            solver=ODESolver.EULER.value,
        )
        result_fine = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=200,
            solver=ODESolver.EULER.value,
        )
        error_coarse = abs(result_coarse.item() - exact_solution)
        error_fine = abs(result_fine.item() - exact_solution)
        # Doubling steps should roughly halve error for first-order method
        convergence_ratio = error_coarse / error_fine
        assert 1.5 < convergence_ratio < 2.5

    def test_heun_second_order_convergence(
        self,
        velocity_field_factory: Callable,
    ):
        # Heun is O(h^2) globally, so halving step size -> ~4x error reduction
        batch_size = 1
        feature_dim = 1
        state = torch.ones(batch_size, feature_dim)
        velocity_fn = velocity_field_factory(field_type="linear")
        exact_solution = math.e

        result_coarse = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=50,
            solver=ODESolver.HEUN.value,
        )
        result_fine = integrate_ode(
            z_init=state,
            velocity_fn=velocity_fn,
            num_steps=100,
            solver=ODESolver.HEUN.value,
        )
        error_coarse = abs(result_coarse.item() - exact_solution)
        error_fine = abs(result_fine.item() - exact_solution)
        convergence_ratio = error_coarse / error_fine
        # Should be close to 4.0 for second-order method
        assert 3.0 < convergence_ratio < 5.0
