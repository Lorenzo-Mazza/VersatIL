"""Tests for versatil.models.exportable.denoising module."""

from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.models.exportable.denoising import ExportableDenoisingPolicy
from versatil.models.exportable.metadata import (
    NoiseInput,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "diffusion,stochastic", [(False, False), (True, False), (True, True)]
)
def test_passes_explicit_noise_to_the_algorithm(
    denoising_export_policy_factory: Callable[..., MagicMock],
    token_generation_tensors_factory: Callable[[], tuple[MagicMock, MagicMock]],
    denoising_reference_factory: Callable[..., MagicMock],
    diffusion: bool,
    stochastic: bool,
) -> None:
    policy = denoising_export_policy_factory(diffusion=diffusion, stochastic=stochastic)
    resolve_reference = denoising_reference_factory(dtype=torch.bfloat16)
    exportable = ExportableDenoisingPolicy.from_policy(policy=policy)
    observation, initial_position = token_generation_tensors_factory()
    initial_orientation, step_position = token_generation_tensors_factory()
    step_orientation, prediction = token_generation_tensors_factory()
    features = {"encoded": observation}
    policy.algorithm.predict_from_noise.return_value = dict.fromkeys(
        policy.output_keys, prediction
    )
    inputs = (observation, initial_position, initial_orientation)
    expected_noise = [
        NoiseInput(name=f"initial_noise.{key}", shape=(2, 3))
        for key in policy.output_keys
    ]
    if stochastic:
        inputs += (step_position, step_orientation)
        expected_noise.extend(
            NoiseInput(name=f"step_noise.{key}", shape=(3, 2, 3))
            for key in policy.output_keys
        )
    with patch(
        "versatil.models.exportable.base.build_algorithm_features",
        return_value=features,
    ):
        outputs = exportable(*inputs)  # (batch, horizon, dimension)
    expected_arguments = {
        "network": policy.decoder,
        "features": features,
        "initial_noise": {
            "position": initial_position.to.return_value,
            "orientation": initial_orientation.to.return_value,
        },
    }
    if diffusion:
        expected_arguments["step_noise"] = (
            {
                "position": step_position.to.return_value,
                "orientation": step_orientation.to.return_value,
            }
            if stochastic
            else None
        )
    policy.algorithm.predict_from_noise.assert_called_once_with(**expected_arguments)
    resolve_reference.assert_called_once_with(features=features)
    for noise in inputs[1:]:
        noise.to.assert_called_once_with(
            device=torch.device("cpu"), dtype=torch.bfloat16
        )
    policy.algorithm.predict.assert_not_called()
    assert outputs == (prediction, prediction)
    assert exportable.export_metadata.noise_inputs == tuple(expected_noise)
    policy.decoder.enable_encoder_cache.assert_not_called()
    policy.decoder.disable_encoder_cache.assert_not_called()
