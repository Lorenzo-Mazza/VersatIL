"""Tests for versatil.models.exportable.factory module."""

import re
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.diffusion import Diffusion
from versatil.models.decoding.algorithm.flow_matching import FlowMatching
from versatil.models.exportable.factory import create_exportable_policy
from versatil.models.policy import Policy

FACTORY_MODULE = "versatil.models.exportable.factory"


@pytest.fixture
def adapter_selection_factory() -> Iterator[
    Callable[..., tuple[MagicMock, dict[str, MagicMock]]]
]:
    with (
        patch(f"{FACTORY_MODULE}.ExportablePolicy.from_policy") as base,
        patch(f"{FACTORY_MODULE}.ExportableDenoisingPolicy.from_policy") as denoising,
        patch(f"{FACTORY_MODULE}.ExportableTokenPolicy.from_policy") as autoregressive,
    ):

        def factory(
            algorithm_type: type[DecodingAlgorithm], tokenized: bool
        ) -> tuple[MagicMock, dict[str, MagicMock]]:
            policy = MagicMock(spec=Policy)
            policy.algorithm = MagicMock(spec=algorithm_type)
            policy.decoder = MagicMock()
            policy.decoder.requires_tokenized_actions = tokenized
            return policy, {
                "base": base,
                "denoising": denoising,
                "autoregressive": autoregressive,
            }

        yield factory


@pytest.mark.unit
@pytest.mark.parametrize(
    "algorithm_type,tokenized,selected",
    [
        (DecodingAlgorithm, False, "base"),
        (Diffusion, False, "denoising"),
        (FlowMatching, False, "denoising"),
        (Diffusion, True, "autoregressive"),
    ],
)
def test_selects_adapter_from_algorithm_and_token_output(
    adapter_selection_factory: Callable[..., tuple[MagicMock, dict[str, MagicMock]]],
    algorithm_type: type[DecodingAlgorithm],
    tokenized: bool,
    selected: str,
) -> None:
    policy, constructors = adapter_selection_factory(
        algorithm_type=algorithm_type, tokenized=tokenized
    )

    adapter = create_exportable_policy(policy=policy)

    constructors[selected].assert_called_once_with(policy=policy)
    assert adapter == constructors[selected].return_value
    for name, constructor in constructors.items():
        if name != selected:
            constructor.assert_not_called()


@pytest.mark.unit
class TestFromPolicy:
    @pytest.mark.parametrize(
        "encoder_keys, conditional_keys, expected",
        [
            ({"rgb": ["left", "right"]}, {}, {"left", "right"}),
            ({"rgb": ["left"]}, {"depth": ["depth"]}, {"left", "depth"}),
        ],
    )
    def test_derives_observation_keys_from_all_encoders(
        self,
        from_policy_factory,
        encoder_keys,
        conditional_keys,
        expected,
    ):
        policy = from_policy_factory(
            encoder_keys=encoder_keys,
            conditional_encoder_keys=conditional_keys,
            action_keys=["position"],
        )

        exportable = create_exportable_policy(policy=policy)

        assert set(exportable.observation_keys) == expected

    def test_tokenized_action_policy_requires_supported_adapter(
        self, from_policy_factory
    ):
        policy = from_policy_factory(action_keys=["position"])
        policy.decoder.requires_tokenized_actions = True

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Tokenized export requires BehavioralCloning with an autoregressive token decoder."
            ),
        ):
            create_exportable_policy(policy=policy)

    def test_keys_are_sorted(self, from_policy_factory):
        policy = from_policy_factory(
            encoder_keys={"rgb": ["right", "left"]},
            action_keys=["position", "gripper", "orientation"],
        )

        exportable = create_exportable_policy(policy=policy)

        assert exportable.observation_keys == sorted(exportable.observation_keys)
        assert exportable.action_keys == sorted(exportable.action_keys)

    def test_shares_policy_components(self, from_policy_factory):
        policy = from_policy_factory()

        exportable = create_exportable_policy(policy=policy)

        # Verify shared reference via mutation
        param = nn.Parameter(torch.tensor(42.0))
        policy.encoding_pipeline.test_param = param
        assert exportable.encoding_pipeline.test_param is param
