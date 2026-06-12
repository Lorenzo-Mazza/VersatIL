"""Tests for versatil.metrics.regularizers module."""

import dataclasses
import re
from collections.abc import Callable
from contextlib import nullcontext
from contextlib import nullcontext as does_not_raise
from unittest.mock import patch

import albumentations as A
import numpy as np
import pytest
import torch

from versatil.common.tensor_ops import to_device
from versatil.data.constants import (
    ActionComputationMethod,
    CoordinateSystem,
    ObsKey,
    SampleKey,
)
from versatil.data.metadata import PositionActionMetadata
from versatil.metrics.constants import (
    FiniteDifferencePerturbationMode,
    ImageAugmentationConsistencyLossMode,
    MetricKey,
)
from versatil.metrics.regularization_context import (
    PolicyForwardContext,
    PolicyGraphInputDomain,
    PolicyRegularizationGraph,
)
from versatil.metrics.regularizers import (
    FiniteDifferenceLipschitzRegularizer,
    ImageAugmentationConsistencyRegularizer,
    JacobianFrobeniusLipschitzRegularizer,
    SpectralJacobianLipschitzRegularizer,
)
from versatil.models.policy import Policy


@pytest.fixture
def image_consistency_graph_factory() -> Callable[..., PolicyRegularizationGraph]:
    def factory(
        training: bool = True,
        channel_count: int = 3,
        image_value: float = -1.0,
    ) -> PolicyRegularizationGraph:
        image = torch.full((1, 1, channel_count, 2, 2), image_value)

        def evaluate_with_replacements(
            input_domain: str,
            context: PolicyForwardContext,
            replacements: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            observation = dict(context.observation)
            observation.update(replacements)
            image_tensor = observation["image"]
            action = image_tensor.reshape(image_tensor.shape[0], -1).mean(dim=1)
            return {"action": action.reshape(-1, 1, 1)}  # (B,) -> (B, 1, 1)

        context = PolicyForwardContext(
            observation={"image": image},
            encoded_features={},
            decoder_features={},
            predictions={"action": torch.full((1, 1, 1), image_value)},
            actions=None,
        )
        return PolicyRegularizationGraph(
            context=context,
            training=training,
            default_output_keys=["action"],
            evaluate_with_replacements=evaluate_with_replacements,
            deterministic_scope=lambda enabled: nullcontext(),
        )

    return factory


@pytest.fixture
def position_consistency_graph_factory() -> Callable[..., PolicyRegularizationGraph]:
    def factory(computation_method: str) -> PolicyRegularizationGraph:
        image = torch.full((1, 1, 3, 2, 2), -1.0)
        clean_position = torch.zeros(1, 3, 3)
        augmented_position = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                ]
            ]
        )

        def evaluate_with_replacements(
            input_domain: str,
            context: PolicyForwardContext,
            replacements: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            if replacements:
                return {"ee_pos_action": augmented_position}
            return {"ee_pos_action": clean_position}

        context = PolicyForwardContext(
            observation={"image": image},
            encoded_features={},
            decoder_features={},
            predictions={"ee_pos_action": clean_position},
            actions=None,
        )
        action_metadata = {
            "ee_pos_action": PositionActionMetadata(
                frame=CoordinateSystem.ROBOT_BASE.value,
                raw_data_column_keys=["action"],
                storage_dimension=3,
                prediction_dimension=3,
                needs_normalization=True,
                dtype="float32",
                computation_method=computation_method,
            )
        }
        return PolicyRegularizationGraph(
            context=context,
            training=True,
            default_output_keys=["ee_pos_action"],
            evaluate_with_replacements=evaluate_with_replacements,
            deterministic_scope=lambda enabled: nullcontext(),
            action_metadata=action_metadata,
        )

    return factory


class TestFiniteDifferenceLipschitzRegularizer:
    def test_computes_linear_feature_slope(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
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

    def test_uses_raw_l2_over_full_output_chunk(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory(decoder_chunk_count=2)
        graph = regularizer_graph_factory(policy=policy)
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=1.0,
            target=0.0,
            noise_scale=0.01,
        )

        output = regularizer(graph=graph)

        assert torch.allclose(
            output.component_losses[MetricKey.LIPSCHITZ_SLOPE_MEAN.value],
            3.0 * torch.sqrt(torch.tensor(2.0)),
        )

    def test_dimension_ratio_scale_uses_square_root_factor(self) -> None:
        scale = FiniteDifferenceLipschitzRegularizer._dimension_ratio_scale(
            input_dimension=16,
            output_dimension=4,
            exponent=0.5,
            reference=torch.tensor(1.0),
        )

        assert torch.allclose(scale, torch.tensor(2.0))

    def test_rng_replay_cancels_stochastic_decoder_noise_in_centered_differences(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory(decoder_output_noise_scale=10.0)
        graph = regularizer_graph_factory(policy=policy)
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=1.0,
            target=0.0,
            noise_scale=0.01,
        )

        output = regularizer(graph=graph)

        assert torch.allclose(
            output.component_losses[MetricKey.LIPSCHITZ_SLOPE_MEAN.value],
            torch.tensor(3.0),
            atol=1e-3,
        )
        assert torch.allclose(
            output.component_losses[MetricKey.LIPSCHITZ_SLOPE_MAX.value],
            torch.tensor(3.0),
            atol=1e-3,
        )

    def test_max_batch_size_caps_perturbed_forward_batch(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy, batch_size=4)
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=1.0,
            target=0.0,
            noise_scale=0.01,
            max_batch_size=2,
        )

        with patch.object(
            regularizer,
            "_forward_with_replacements",
            wraps=regularizer._forward_with_replacements,
        ) as forward_spy:
            regularizer(graph=graph)

        assert len(forward_spy.call_args_list) == 2
        for call in forward_spy.call_args_list:
            assert call.kwargs["replacements"]["feature"].shape[0] == 2

    def test_output_key_inference_falls_back_to_graph_defaults_then_predictions(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        graph_without_defaults = dataclasses.replace(graph, default_output_keys=[])
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=None,
            weight=1.0,
            target=0.0,
            noise_scale=0.01,
        )

        default_keys_output = regularizer(graph=graph)
        inferred_keys_output = regularizer(graph=graph_without_defaults)

        assert torch.allclose(
            default_keys_output.component_losses[MetricKey.LIPSCHITZ_SLOPE_MEAN.value],
            torch.tensor(3.0),
        )
        assert torch.allclose(
            inferred_keys_output.component_losses[MetricKey.LIPSCHITZ_SLOPE_MEAN.value],
            torch.tensor(3.0),
        )

    def test_rejects_missing_key_in_explicit_domain(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
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

    def test_rejects_non_floating_input_tensors(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        integer_context = dataclasses.replace(
            graph.context,
            encoded_features={"feature": torch.ones(2, 3, dtype=torch.long)},
        )
        integer_graph = dataclasses.replace(graph, context=integer_context)
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "FiniteDifferenceLipschitzRegularizer can only perturb "
                "floating-point tensors, got non-floating input keys: ['feature']."
            ),
        ):
            regularizer(graph=integer_graph)

    def test_detached_feature_inputs_block_encoder_gradients(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
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
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
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

    @pytest.mark.parametrize(
        "shape",
        [
            (2, 3, 4, 5),
            (2, 1, 3, 4, 5),
        ],
    )
    def test_channel_broadcast_perturbation_is_constant_over_spatial_axes(
        self,
        shape: tuple[int, ...],
    ) -> None:
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.OBSERVATION.value,
            input_keys=["image"],
            output_keys=["action"],
            perturbation_mode=FiniteDifferencePerturbationMode.GAUSSIAN_CHANNEL_BROADCAST.value,
        )
        domain_inputs = {"image": torch.ones(shape)}

        perturbations = regularizer._build_perturbations(domain_inputs=domain_inputs)

        perturbation = perturbations["image"]
        spatial_reference = perturbation[..., :1, :1]
        assert torch.allclose(
            perturbation,
            spatial_reference.expand_as(perturbation),
        )

    def test_channel_broadcast_perturbation_rejects_non_image_like_tensor(
        self,
    ) -> None:
        regularizer = FiniteDifferenceLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            perturbation_mode=FiniteDifferencePerturbationMode.GAUSSIAN_CHANNEL_BROADCAST.value,
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "FiniteDifferenceLipschitzRegularizer "
                "perturbation_mode='gaussian_channel_broadcast' requires input "
                "key 'feature' to have shape (B, C, H, W) or (B, T, C, H, W), "
                "got (2, 3)."
            ),
        ):
            regularizer._build_perturbations(
                domain_inputs={"feature": torch.ones(2, 3)}
            )

    @pytest.mark.parametrize(
        "weight, target, noise_scale, eps, expectation",
        [
            (0.0, 0.0, 0.01, 1e-12, does_not_raise()),
            (
                -1.0,
                1.0,
                0.01,
                1e-12,
                pytest.raises(
                    ValueError,
                    match=re.escape("weight must be non-negative, got -1.0."),
                ),
            ),
            (
                1.0,
                -0.5,
                0.01,
                1e-12,
                pytest.raises(
                    ValueError,
                    match=re.escape("target must be non-negative, got -0.5."),
                ),
            ),
            (
                1.0,
                1.0,
                0.0,
                1e-12,
                pytest.raises(
                    ValueError,
                    match=re.escape("noise_scale must be positive, got 0.0."),
                ),
            ),
            (
                1.0,
                1.0,
                0.01,
                0.0,
                pytest.raises(
                    ValueError,
                    match=re.escape("eps must be positive, got 0.0."),
                ),
            ),
        ],
    )
    def test_scalar_hyperparameter_validation(
        self,
        weight: float,
        target: float,
        noise_scale: float,
        eps: float,
        expectation,
    ) -> None:
        with expectation:
            FiniteDifferenceLipschitzRegularizer(
                input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
                input_keys=["feature"],
                output_keys=["action"],
                weight=weight,
                target=target,
                noise_scale=noise_scale,
                eps=eps,
            )

    def test_rejects_unknown_perturbation_mode(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "perturbation_mode must be one of "
                "['gaussian_channel_broadcast', 'gaussian_dense'], got missing."
            ),
        ):
            FiniteDifferenceLipschitzRegularizer(
                input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
                input_keys=["feature"],
                output_keys=["action"],
                perturbation_mode="missing",
            )


class TestImageAugmentationConsistencyRegularizer:
    def test_applies_color_then_spatial_and_penalizes_output_change(
        self,
        image_consistency_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        call_order: list[str] = []

        def shift_color(image: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            call_order.append("color")
            return np.clip(image.astype(np.int16) + 16, 0, 255).astype(np.uint8)

        def flip_spatial(image: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            call_order.append("spatial")
            return np.flip(image, axis=0).copy()

        regularizer = ImageAugmentationConsistencyRegularizer(
            input_keys=["image"],
            output_keys=["action"],
            weight=2.0,
            color_augmentation=A.Compose([A.Lambda(image=shift_color, p=1.0)]),
            spatial_augmentation=A.Compose([A.Lambda(image=flip_spatial, p=1.0)]),
            loss_mode=ImageAugmentationConsistencyLossMode.FLAT_OUTPUT_MSE.value,
        )
        graph = image_consistency_graph_factory()

        output = regularizer(graph=graph)

        expected_delta = torch.tensor(2.0 * 16.0 / 255.0)
        expected_loss = expected_delta.pow(2)
        assert call_order == ["color", "spatial"]
        assert torch.allclose(
            output.component_losses[
                MetricKey.IMAGE_AUGMENTATION_CONSISTENCY_LOSS.value
            ],
            expected_loss,
        )
        assert torch.allclose(output.total_loss, 2.0 * expected_loss)
        assert torch.allclose(
            output.component_losses[MetricKey.IMAGE_AUGMENTATION_OUTPUT_DELTA_L2.value],
            expected_delta,
        )
        assert torch.allclose(
            output.component_losses[MetricKey.IMAGE_AUGMENTATION_FLAT_OUTPUT_MSE.value],
            expected_loss,
        )

    @pytest.mark.parametrize(
        "computation_method, expected_loss",
        [
            (ActionComputationMethod.NEXT_TIMESTEP.value, 4.0 / 3.0),
            (ActionComputationMethod.DELTA.value, 1.0),
        ],
    )
    def test_position_trajectory_loss_respects_action_computation_method(
        self,
        position_consistency_graph_factory: Callable[..., PolicyRegularizationGraph],
        computation_method: str,
        expected_loss: float,
    ) -> None:
        regularizer = ImageAugmentationConsistencyRegularizer(
            input_keys=["image"],
            output_keys=["ee_pos_action"],
            weight=2.0,
            color_augmentation=A.Compose([A.InvertImg(p=1.0)]),
            loss_mode=(
                ImageAugmentationConsistencyLossMode.POSITION_TRAJECTORY_L2.value
            ),
        )
        graph = position_consistency_graph_factory(
            computation_method=computation_method
        )

        output = regularizer(graph=graph)

        expected = torch.tensor(expected_loss)
        assert torch.allclose(
            output.component_losses[
                MetricKey.IMAGE_AUGMENTATION_CONSISTENCY_LOSS.value
            ],
            expected,
        )
        assert torch.allclose(output.total_loss, 2.0 * expected)
        assert torch.allclose(
            output.component_losses[
                MetricKey.IMAGE_AUGMENTATION_POSITION_PER_STEP_L2.value
            ],
            expected,
        )

    def test_augments_4d_inputs_and_inverts_pixel_extremes_exactly(self) -> None:
        regularizer = ImageAugmentationConsistencyRegularizer(
            input_keys=["image"],
            output_keys=["action"],
            color_augmentation=A.Compose([A.InvertImg(p=1.0)]),
        )
        images = torch.full((2, 3, 4, 4), -1.0)

        augmented = regularizer._augment_tensor(value=images)

        assert augmented.shape == (2, 3, 4, 4)
        assert torch.allclose(augmented, torch.ones_like(augmented))

    def test_spatial_augmentation_parameters_are_shared_across_temporal_window(
        self,
    ) -> None:
        regularizer = ImageAugmentationConsistencyRegularizer(
            input_keys=["image"],
            output_keys=["action"],
            spatial_augmentation=A.Compose(
                [
                    A.Affine(
                        scale=(1.3, 2.0),
                        translate_percent=(0.05, 0.2),
                        p=1.0,
                    )
                ]
            ),
        )
        gradient = torch.linspace(-1.0, 1.0, steps=64).reshape(1, 8, 8)
        frame = gradient.repeat(3, 1, 1)  # (1, 8, 8) -> (3, 8, 8)
        window = frame.unsqueeze(0).repeat(4, 1, 1, 1)  # (3, 8, 8) -> (4, 3, 8, 8)
        batch = window.unsqueeze(0)  # (4, 3, 8, 8) -> (1, 4, 3, 8, 8)

        augmented = regularizer._augment_tensor(value=batch)

        for frame_index in range(1, batch.shape[1]):
            assert torch.equal(augmented[0, frame_index], augmented[0, 0])
        assert not torch.equal(augmented[0, 0], batch[0, 0])

    def test_color_augmentation_rejects_non_rgb_channel_count(
        self,
        image_consistency_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        regularizer = ImageAugmentationConsistencyRegularizer(
            input_keys=["image"],
            output_keys=["action"],
            color_augmentation=A.Compose([A.InvertImg(p=1.0)]),
        )
        graph = image_consistency_graph_factory(channel_count=1)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "color_augmentation requires RGB image tensors with 3 channels, "
                "got 1 channels."
            ),
        ):
            regularizer(graph=graph)

    def test_augment_tensor_rejects_non_image_rank(self) -> None:
        regularizer = ImageAugmentationConsistencyRegularizer(
            input_keys=["image"],
            output_keys=["action"],
            color_augmentation=A.Compose([A.InvertImg(p=1.0)]),
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "ImageAugmentationConsistencyRegularizer input tensors must have "
                "shape (B, C, H, W) or (B, T, C, H, W), got (3, 2, 2)."
            ),
        ):
            regularizer._augment_tensor(value=torch.zeros(3, 2, 2))

    @pytest.mark.parametrize(
        "color_augmentation, spatial_augmentation",
        [
            (None, None),
            (A.Compose([]), A.Compose([])),
        ],
        ids=["missing", "empty"],
    )
    def test_requires_at_least_one_augmentation(
        self,
        color_augmentation: A.Compose | None,
        spatial_augmentation: A.Compose | None,
    ) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "At least one of color_augmentation or spatial_augmentation must "
                "contain transforms."
            ),
        ):
            ImageAugmentationConsistencyRegularizer(
                input_keys=["image"],
                output_keys=["action"],
                color_augmentation=color_augmentation,
                spatial_augmentation=spatial_augmentation,
            )

    @pytest.mark.parametrize(
        "weight, input_min, input_max, max_pixel_value, loss_mode, expectation",
        [
            (
                0.0,
                -1.0,
                1.0,
                255.0,
                ImageAugmentationConsistencyLossMode.FLAT_OUTPUT_MSE.value,
                does_not_raise(),
            ),
            (
                -1.0,
                -1.0,
                1.0,
                255.0,
                ImageAugmentationConsistencyLossMode.FLAT_OUTPUT_MSE.value,
                pytest.raises(
                    ValueError,
                    match=re.escape("weight must be non-negative, got -1.0."),
                ),
            ),
            (
                1.0,
                1.0,
                1.0,
                255.0,
                ImageAugmentationConsistencyLossMode.FLAT_OUTPUT_MSE.value,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "input_max must be greater than input_min, got "
                        "input_min=1.0 and input_max=1.0."
                    ),
                ),
            ),
            (
                1.0,
                -1.0,
                1.0,
                0.0,
                ImageAugmentationConsistencyLossMode.FLAT_OUTPUT_MSE.value,
                pytest.raises(
                    ValueError,
                    match=re.escape("max_pixel_value must be positive, got 0.0."),
                ),
            ),
            (
                1.0,
                -1.0,
                1.0,
                255.0,
                "missing",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "loss_mode must be one of "
                        "['flat_output_mse', 'position_trajectory_l2'], got missing."
                    ),
                ),
            ),
        ],
    )
    def test_scalar_hyperparameter_validation(
        self,
        weight: float,
        input_min: float,
        input_max: float,
        max_pixel_value: float,
        loss_mode: str,
        expectation,
    ) -> None:
        with expectation:
            ImageAugmentationConsistencyRegularizer(
                input_keys=["image"],
                output_keys=["action"],
                weight=weight,
                color_augmentation=A.Compose([A.InvertImg(p=1.0)]),
                loss_mode=loss_mode,
                input_min=input_min,
                input_max=input_max,
                max_pixel_value=max_pixel_value,
            )

    def test_eval_mode_without_apply_during_eval_returns_no_diagnostics(
        self,
        image_consistency_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        regularizer = ImageAugmentationConsistencyRegularizer(
            input_keys=["image"],
            output_keys=["action"],
            color_augmentation=A.Compose([A.InvertImg(p=1.0)]),
            apply_during_eval=False,
        )
        graph = image_consistency_graph_factory(training=False)

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
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
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

    def test_multiple_probes_preserve_exact_estimate_for_linear_map(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = JacobianFrobeniusLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=2.0,
            number_of_probes=3,
        )

        output = regularizer(graph=graph)

        assert torch.allclose(
            output.component_losses[MetricKey.LIPSCHITZ_JACOBIAN_FROBENIUS_LOSS.value],
            torch.tensor(27.0),
        )
        assert torch.allclose(output.total_loss, torch.tensor(54.0))

    def test_dimension_ratio_scaling_uses_linear_factor(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory(decoder_output_dimension=1)
        graph = regularizer_graph_factory(policy=policy, feature_dimension=3)
        regularizer = JacobianFrobeniusLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=2.0,
            number_of_probes=1,
            scale_by_dimension_ratio=True,
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
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
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

    @pytest.mark.parametrize(
        "weight, number_of_probes, expectation",
        [
            (0.0, 1, does_not_raise()),
            (
                -1.0,
                1,
                pytest.raises(
                    ValueError,
                    match=re.escape("weight must be non-negative, got -1.0."),
                ),
            ),
            (
                1.0,
                0,
                pytest.raises(
                    ValueError,
                    match=re.escape("number_of_probes must be at least 1, got 0."),
                ),
            ),
        ],
    )
    def test_scalar_hyperparameter_validation(
        self,
        weight: float,
        number_of_probes: int,
        expectation,
    ) -> None:
        with expectation:
            JacobianFrobeniusLipschitzRegularizer(
                input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
                input_keys=["feature"],
                output_keys=["action"],
                weight=weight,
                number_of_probes=number_of_probes,
            )

    def test_eval_mode_without_apply_during_eval_returns_no_diagnostics(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        policy.eval()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = JacobianFrobeniusLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
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
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=1e-4,
            number_of_probes=1,
            max_batch_size=2,
        )
        policy = regularizer_policy_factory().to(device=device)
        policy.device = device
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


class TestSpectralJacobianLipschitzRegularizer:
    def test_computes_linear_feature_spectral_norm(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = SpectralJacobianLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
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

    def test_detach_inputs_false_uses_graph_connected_features(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = SpectralJacobianLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
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

    def test_dimension_ratio_scaling_uses_square_root_factor(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory(decoder_output_dimension=1)
        graph = regularizer_graph_factory(policy=policy, feature_dimension=3)
        regularizer = SpectralJacobianLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
            input_keys=["feature"],
            output_keys=["action"],
            weight=1.0,
            target=0.0,
            number_of_power_iterations=1,
            scale_by_dimension_ratio=True,
        )

        output = regularizer(graph=graph)

        scaled_sigma = 3.0 * torch.sqrt(torch.tensor(3.0))
        assert torch.allclose(
            output.component_losses[MetricKey.LIPSCHITZ_SIGMA.value],
            scaled_sigma,
        )
        assert torch.allclose(
            output.component_losses[MetricKey.LIPSCHITZ_SPECTRAL_JACOBIAN_LOSS.value],
            scaled_sigma.pow(2),
        )

    @pytest.mark.parametrize(
        "weight, target, number_of_power_iterations, eps, expectation",
        [
            (0.0, 0.0, 1, 1e-12, does_not_raise()),
            (
                -1.0,
                1.0,
                1,
                1e-12,
                pytest.raises(
                    ValueError,
                    match=re.escape("weight must be non-negative, got -1.0."),
                ),
            ),
            (
                1.0,
                -0.5,
                1,
                1e-12,
                pytest.raises(
                    ValueError,
                    match=re.escape("target must be non-negative, got -0.5."),
                ),
            ),
            (
                1.0,
                1.0,
                0,
                1e-12,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "number_of_power_iterations must be at least 1, got 0."
                    ),
                ),
            ),
            (
                1.0,
                1.0,
                1,
                0.0,
                pytest.raises(
                    ValueError,
                    match=re.escape("eps must be positive, got 0.0."),
                ),
            ),
        ],
    )
    def test_scalar_hyperparameter_validation(
        self,
        weight: float,
        target: float,
        number_of_power_iterations: int,
        eps: float,
        expectation,
    ) -> None:
        with expectation:
            SpectralJacobianLipschitzRegularizer(
                input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
                input_keys=["feature"],
                output_keys=["action"],
                weight=weight,
                target=target,
                number_of_power_iterations=number_of_power_iterations,
                eps=eps,
            )

    def test_eval_mode_without_apply_during_eval_returns_no_diagnostics(
        self,
        regularizer_policy_factory: Callable[..., Policy],
        regularizer_graph_factory: Callable[..., PolicyRegularizationGraph],
    ) -> None:
        policy = regularizer_policy_factory()
        policy.eval()
        graph = regularizer_graph_factory(policy=policy)
        regularizer = SpectralJacobianLipschitzRegularizer(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
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
