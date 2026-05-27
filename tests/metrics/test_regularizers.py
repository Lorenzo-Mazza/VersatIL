"""Tests for versatil.metrics.regularizers module."""

import re
from collections.abc import Callable

import pytest
import torch

from versatil.metrics.constants import MetricKey
from versatil.metrics.regularization_context import PolicyRegularizationGraph
from versatil.metrics.regularizers import (
    FiniteDifferenceLipschitzRegularizer,
    JacobianFrobeniusLipschitzRegularizer,
    RegularizerInputDomain,
    SpectralJacobianLipschitzRegularizer,
)
from versatil.models.policy import Policy


class TestFiniteDifferenceLipschitzRegularizer:
    def test_computes_linear_feature_slope(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=RegularizerInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=2.0,
            target=1.0,
            noise_scale=0.01,
        )

        output = regularizer(graph=graph)

        assert torch.allclose(
            output.component_losses[MetricKey.LIPSCHITZ_SLOPE_MEAN.value],
            torch.tensor(3.0),
        )
        assert torch.allclose(
            output.component_losses[MetricKey.LIPSCHITZ_FINITE_DIFFERENCE_LOSS.value],
            torch.tensor(4.0),
        )
        assert torch.allclose(output.total_loss, torch.tensor(8.0))

    def test_rejects_missing_key_in_explicit_domain(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=RegularizerInputDomain.ENCODED_FEATURES.value,
            input_keys=["missing"],
            output_keys=["action"],
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "FiniteDifferenceLipschitzRegularizer input keys ['missing'] "
                "were not found in domain 'encoded_features'. "
                "Available keys: ['feature']."
            ),
        ):
            regularizer(graph=graph)

    def test_detached_feature_inputs_block_encoder_gradients(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=RegularizerInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=1.0,
            target=0.0,
            noise_scale=0.01,
            detach_inputs=True,
        )

        output = regularizer(graph=graph)
        output.total_loss.backward()

        assert policy.encoding_pipeline.scale.grad is None
        assert policy.decoder.scale.grad is not None
        assert policy.decoder.scale.grad.abs() > 0.0

    def test_eval_mode_without_apply_during_eval_returns_zero_diagnostics(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        policy.eval()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=RegularizerInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=1.0,
            target=0.0,
            noise_scale=0.01,
            apply_during_eval=False,
        )

        output = regularizer(graph=graph)

        assert torch.equal(output.total_loss, torch.tensor(0.0))
        assert set(output.component_losses) == {
            MetricKey.LIPSCHITZ_FINITE_DIFFERENCE_LOSS.value,
            MetricKey.LIPSCHITZ_SLOPE_MEAN.value,
            MetricKey.LIPSCHITZ_SLOPE_MAX.value,
        }
        for component_value in output.component_losses.values():
            assert torch.equal(component_value, torch.tensor(0.0))


class TestJacobianFrobeniusLipschitzRegularizer:
    def test_computes_linear_feature_frobenius_estimate(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = JacobianFrobeniusLipschitzRegularizer(
            input_domain=RegularizerInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=2.0,
            number_of_probes=1,
        )

        output = regularizer(graph=graph)

        assert torch.allclose(
            output.component_losses[MetricKey.LIPSCHITZ_JACOBIAN_FROBENIUS_LOSS.value],
            torch.tensor(27.0),
        )
        assert torch.allclose(
            output.component_losses[MetricKey.LIPSCHITZ_JACOBIAN_FROBENIUS_NORM.value],
            torch.sqrt(torch.tensor(27.0)),
        )
        assert torch.allclose(output.total_loss, torch.tensor(54.0))

    def test_detached_feature_inputs_block_encoder_gradients(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = JacobianFrobeniusLipschitzRegularizer(
            input_domain=RegularizerInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=1.0,
            number_of_probes=1,
            detach_inputs=True,
        )

        output = regularizer(graph=graph)
        output.total_loss.backward()

        assert policy.encoding_pipeline.scale.grad is None
        assert policy.decoder.scale.grad is not None
        assert policy.decoder.scale.grad.abs() > 0.0

    def test_eval_mode_without_apply_during_eval_returns_zero_diagnostics(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        policy.eval()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = JacobianFrobeniusLipschitzRegularizer(
            input_domain=RegularizerInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=1.0,
            number_of_probes=1,
            apply_during_eval=False,
        )

        output = regularizer(graph=graph)

        assert torch.equal(output.total_loss, torch.tensor(0.0))
        assert set(output.component_losses) == {
            MetricKey.LIPSCHITZ_JACOBIAN_FROBENIUS_LOSS.value,
            MetricKey.LIPSCHITZ_JACOBIAN_FROBENIUS_NORM.value,
        }
        for component_value in output.component_losses.values():
            assert torch.equal(component_value, torch.tensor(0.0))


def test_spectral_jacobian_computes_linear_feature_spectral_norm(
    regularizer_policy_factory: Callable[..., Policy],
    regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
) -> None:
    policy = regularizer_policy_factory()
    graph = regularizer_graph_factory(policy=policy)
    regularizer = SpectralJacobianLipschitzRegularizer(
        input_domain=RegularizerInputDomain.ENCODED_FEATURES.value,
        input_keys=["feature"],
        output_keys=["action"],
        weight=2.0,
        target=1.0,
        number_of_power_iterations=1,
    )

    output = regularizer(graph=graph)

    assert torch.allclose(
        output.component_losses[MetricKey.LIPSCHITZ_SIGMA.value],
        torch.tensor(3.0),
    )
    assert torch.allclose(
        output.component_losses[MetricKey.LIPSCHITZ_SPECTRAL_JACOBIAN_LOSS.value],
        torch.tensor(4.0),
    )
    assert torch.allclose(output.total_loss, torch.tensor(8.0))


def test_spectral_jacobian_detach_inputs_false_accepts_encoded_non_leaf_inputs(
    regularizer_policy_factory: Callable[..., Policy],
    regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
) -> None:
    policy = regularizer_policy_factory()
    graph = regularizer_graph_factory(policy=policy)
    regularizer = SpectralJacobianLipschitzRegularizer(
        input_domain=RegularizerInputDomain.ENCODED_FEATURES.value,
        input_keys=["feature"],
        output_keys=["action"],
        weight=2.0,
        target=1.0,
        number_of_power_iterations=1,
        detach_inputs=False,
    )

    output = regularizer(graph=graph)

    assert graph.context.encoded_features["feature"].is_leaf is False
    assert graph.context.encoded_features["feature"].requires_grad is True
    assert torch.allclose(
        output.component_losses[MetricKey.LIPSCHITZ_SIGMA.value],
        torch.tensor(3.0),
    )
    assert torch.allclose(output.total_loss, torch.tensor(8.0))


def test_spectral_jacobian_eval_mode_without_apply_during_eval_returns_zero_diagnostics(
    regularizer_policy_factory: Callable[..., Policy],
    regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
) -> None:
    policy = regularizer_policy_factory()
    policy.eval()
    graph = regularizer_graph_factory(policy=policy)
    regularizer = SpectralJacobianLipschitzRegularizer(
        input_domain=RegularizerInputDomain.ENCODED_FEATURES.value,
        input_keys=["feature"],
        output_keys=["action"],
        weight=1.0,
        target=0.0,
        number_of_power_iterations=1,
        apply_during_eval=False,
    )

    output = regularizer(graph=graph)

    assert torch.equal(output.total_loss, torch.tensor(0.0))
    assert set(output.component_losses) == {
        MetricKey.LIPSCHITZ_SPECTRAL_JACOBIAN_LOSS.value,
        MetricKey.LIPSCHITZ_SIGMA.value,
    }
    for component_value in output.component_losses.values():
        assert torch.equal(component_value, torch.tensor(0.0))
