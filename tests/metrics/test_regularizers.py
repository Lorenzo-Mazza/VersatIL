"""Tests for versatil.metrics.regularizers module."""

import re
from collections.abc import Callable

import pytest
import torch

from versatil.common.tensor_ops import to_device
from versatil.data.constants import ObsKey, SampleKey
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

    def test_eval_mode_without_apply_during_eval_returns_no_diagnostics(
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
        assert output.component_losses == {}


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

    def test_eval_mode_without_apply_during_eval_returns_no_diagnostics(
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
        assert output.component_losses == {}

    @pytest.mark.integration
    @pytest.mark.requires_gpu
    def test_policy_loss_accepts_cuda_batch_with_raw_language_metadata(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_batch_factory: Callable[..., dict[str, dict[str, torch.Tensor]]],
    ) -> None:
        device = torch.device("cuda")
        regularizer = JacobianFrobeniusLipschitzRegularizer(
            input_domain=RegularizerInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=1e-4,
            number_of_probes=1,
            max_batch_size=2,
        )
        policy = regularizer_policy_factory().to(device=device)
        policy.regularizers = torch.nn.ModuleDict(
            {"feature_jacobian": regularizer.to(device=device)}
        )
        batch = regularizer_batch_factory(batch_size=3, feature_dimension=3)
        language_values = ["pick up object", "place object", "push object"]
        batch[SampleKey.OBSERVATION.value][ObsKey.LANGUAGE.value] = language_values
        batch = to_device(data=batch, device=device)

        output = policy.compute_loss(batch=batch)

        regularizer_key = (
            f"feature_jacobian/{MetricKey.LIPSCHITZ_JACOBIAN_FROBENIUS_LOSS.value}"
        )
        assert regularizer_key in output.component_losses
        assert output.total_loss.device.type == "cuda"
        assert torch.isfinite(output.total_loss)


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


def test_spectral_jacobian_eval_mode_without_apply_during_eval_returns_no_diagnostics(
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
    assert output.component_losses == {}
