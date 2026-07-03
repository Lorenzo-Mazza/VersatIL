"""Tests for versatil.explainability.attribution.policy module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
import torch

from versatil.data.constants import ProprioKey
from versatil.explainability.attribution.policy import (
    EncoderCacheDisabled,
    default_output_selector,
    run_policy_for_explanation,
)


class _CachingDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cache_enabled = False
        self.suppressed = False

    @property
    def encoder_cache_enabled(self) -> bool:
        return self.cache_enabled

    def enable_encoder_cache(self) -> None:
        if not self.suppressed:
            self.cache_enabled = True

    def disable_encoder_cache(self) -> None:
        if not self.suppressed:
            self.cache_enabled = False

    def set_encoder_cache_suppressed(self, suppressed: bool) -> None:
        self.suppressed = suppressed


def test_run_policy_for_explanation_disables_decoder_cache_during_predict():
    decoder = _CachingDecoder()
    decoder.cache_enabled = True
    predictions = {ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: torch.ones(1, 1, 3)}

    def predict(
        features: dict[str, torch.Tensor],
        network: _CachingDecoder,
    ) -> dict[str, torch.Tensor]:
        # The prefix cache must stay cold during attribution even when the
        # algorithm's own predict path re-enables it.
        network.enable_encoder_cache()
        assert network.encoder_cache_enabled is False
        return predictions

    policy = MagicMock()
    policy.decoder = decoder
    policy.algorithm.predict.side_effect = predict
    policy._strip_metadata_passthrough_observations.return_value = {"camera": "value"}
    features = {"feature": torch.ones(1, 2)}
    policy._build_algorithm_features.return_value = features

    result = run_policy_for_explanation(
        policy=policy,
        observation={"camera": torch.ones(1, 1, 3, 4, 4)},
        preprocess_observation=False,
    )

    assert result is predictions
    assert decoder.encoder_cache_enabled is True
    assert decoder.suppressed is False


@pytest.mark.parametrize("cache_enabled_before", [True, False])
def test_encoder_cache_disabled_restores_prior_state(cache_enabled_before: bool):
    decoder = _CachingDecoder()
    decoder.cache_enabled = cache_enabled_before

    with EncoderCacheDisabled(decoder=decoder):
        assert decoder.encoder_cache_enabled is False
        decoder.enable_encoder_cache()
        assert decoder.encoder_cache_enabled is False

    assert decoder.encoder_cache_enabled is cache_enabled_before
    assert decoder.suppressed is False


@pytest.mark.integration
@pytest.mark.parametrize(
    "policy_case_name",
    ["spatial_resnet18", "smolvla", "paligemma_vlm", "prismatic_vlm"],
)
def test_run_policy_for_explanation_supports_real_policy_classes(
    real_explainability_policy_case_factory: Callable,
    policy_case_name: str,
):
    batch_size = (
        1 if policy_case_name.endswith("_vlm") or policy_case_name == "smolvla" else 2
    )
    case = real_explainability_policy_case_factory(
        case_name=policy_case_name,
        batch_size=batch_size,
    )

    result = run_policy_for_explanation(
        policy=case.policy,
        observation=case.observation,
        preprocess_observation=False,
    )

    assert ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value in result
    prediction = result[ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value]
    assert prediction.shape[0] == batch_size
    assert torch.isfinite(prediction).all()


class TestDefaultOutputSelector:
    def test_returns_norm_from_all_prediction_keys(self):
        predictions = {
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value: torch.tensor(
                [[[3.0, 4.0], [0.0, 0.0]]]
            ),
            "gripper": torch.tensor([[[12.0], [5.0]]]),
        }

        result = default_output_selector(predictions=predictions)

        torch.testing.assert_close(result, torch.tensor([[13.0, 5.0]]))

    def test_raises_when_predictions_are_empty(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Cannot select an explanation target from empty predictions."
            ),
        ):
            default_output_selector(predictions={})

    def test_raises_when_prediction_is_scalar(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Prediction 'position' must have at least one dimension. "
                "Got scalar tensor."
            ),
        ):
            default_output_selector(predictions={"position": torch.tensor(1.0)})

    def test_raises_when_prediction_leading_shapes_differ(self):
        predictions = {
            "position": torch.zeros(1, 2, 3),
            "gripper": torch.zeros(1, 1),
        }

        with pytest.raises(
            ValueError,
            match=re.escape(
                "All prediction tensors must share the same leading shape before "
                "concatenation. Got (1,) for 'gripper' and (1, 2) for 'position'."
            ),
        ):
            default_output_selector(predictions=predictions)
