"""Tests for versatil.models.exportable_policy module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch import nn

from versatil.models.exportable_policy import ExportablePolicy
from versatil.models.policy import Policy


@pytest.fixture
def exportable_factory(
    encoding_pipeline_factory: Callable[..., MagicMock],
) -> Callable[..., ExportablePolicy]:
    """Factory for ExportablePolicy with configurable keys."""

    def factory(
        observation_keys: list[str] | None = None,
        action_keys: list[str] | None = None,
        pipeline: MagicMock | None = None,
    ) -> ExportablePolicy:
        if observation_keys is None:
            observation_keys = ["depth", "left"]
        if action_keys is None:
            action_keys = ["orientation", "position"]
        return ExportablePolicy(
            encoding_pipeline=pipeline or encoding_pipeline_factory(),
            algorithm=MagicMock(),
            decoder=MagicMock(),
            observation_keys=observation_keys,
            action_keys=action_keys,
        )

    return factory


@pytest.fixture
def from_policy_factory(
    policy_factory: Callable[..., Policy],
    vision_encoder_factory: Callable[..., MagicMock],
    encoding_pipeline_factory: Callable[..., MagicMock],
) -> Callable[..., Policy]:
    """Factory for Policy instances configured for from_policy tests."""

    def factory(
        encoder_keys: dict[str, list[str]] | None = None,
        conditional_encoder_keys: dict[str, list[str]] | None = None,
        action_keys: list[str] | None = None,
    ) -> Policy:
        if encoder_keys is None:
            encoder_keys = {"rgb": ["left", "right"]}
        if conditional_encoder_keys is None:
            conditional_encoder_keys = {}
        if action_keys is None:
            action_keys = ["position"]
        encoders = nn.ModuleDict(
            {
                name: vision_encoder_factory(input_keys=keys)
                for name, keys in encoder_keys.items()
            }
        )
        conditional_encoders = nn.ModuleDict(
            {
                name: vision_encoder_factory(input_keys=keys)
                for name, keys in conditional_encoder_keys.items()
            }
        )
        pipeline = encoding_pipeline_factory(
            encoders=encoders,
            conditional_encoders=conditional_encoders,
        )
        decoder = MagicMock()
        decoder.decoder_input.needs_raw_observations = False
        decoder.action_heads = nn.ModuleDict(
            {key: nn.Identity() for key in action_keys}
        )
        return policy_factory(
            encoding_pipeline=pipeline,
            decoder=decoder,
        )

    return factory


@pytest.fixture
def observation_tensor_factory(
    rng: np.random.Generator,
) -> Callable[..., torch.Tensor]:
    """Factory for observation tensors with configurable shape."""

    def factory(
        batch_size: int = 2,
        channels: int = 3,
        height: int = 64,
        width: int = 64,
    ) -> torch.Tensor:
        return torch.from_numpy(
            rng.standard_normal((batch_size, channels, height, width)).astype(
                np.float32
            )
        )

    return factory


@pytest.mark.unit
class TestExportablePolicyInit:
    @pytest.mark.parametrize(
        "observation_keys", [["alpha", "beta"], ["depth", "left", "right"]]
    )
    @pytest.mark.parametrize("action_keys", [["gripper", "position"], ["position"]])
    def test_stores_key_configuration(
        self,
        exportable_factory,
        observation_keys,
        action_keys,
    ):
        exportable = exportable_factory(
            observation_keys=observation_keys,
            action_keys=action_keys,
        )

        assert exportable.observation_keys == observation_keys
        assert exportable.action_keys == action_keys

    @pytest.mark.parametrize("property_name", ["observation_keys", "action_keys"])
    def test_key_properties_return_copies(self, exportable_factory, property_name):
        exportable = exportable_factory()
        original = getattr(exportable, property_name)

        returned = getattr(exportable, property_name)
        returned.append("mutated")

        assert getattr(exportable, property_name) == original


@pytest.mark.unit
class TestExportablePolicyForward:
    def test_reconstructs_observation_dict_from_positional_tensors(
        self,
        exportable_factory,
        encoding_pipeline_factory,
        observation_tensor_factory,
    ):
        pipeline = encoding_pipeline_factory()
        features = {"encoded": torch.zeros(2, 64)}
        pipeline.return_value = features
        exportable = exportable_factory(
            observation_keys=["depth", "left"],
            action_keys=["position"],
            pipeline=pipeline,
        )
        exportable.algorithm.predict.return_value = {
            "position": torch.zeros(2, 16, 3),
        }

        depth = observation_tensor_factory(channels=1)
        left = observation_tensor_factory(channels=3)
        exportable(depth, left)

        called_dict = pipeline.call_args[0][0]
        assert set(called_dict.keys()) == {"depth", "left"}
        assert torch.equal(called_dict["depth"], depth)
        assert torch.equal(called_dict["left"], left)

    def test_passes_features_and_decoder_to_algorithm_predict(
        self,
        exportable_factory,
        encoding_pipeline_factory,
    ):
        pipeline = encoding_pipeline_factory()
        requested_feature = torch.zeros(2, 64)
        pipeline.return_value = {
            "encoded": requested_feature,
            "unrequested_prefusion": torch.ones(2, 64),
        }
        exportable = exportable_factory(
            observation_keys=["left"],
            action_keys=["position"],
            pipeline=pipeline,
        )
        exportable.decoder.decoder_input.keys = ["encoded"]
        exportable.algorithm.predict.return_value = {
            "position": torch.zeros(2, 16, 3),
        }

        exportable(torch.zeros(2, 3, 64, 64))

        # Features must be filtered to the decoder's allowlist exactly like
        # Policy._build_algorithm_features; unrequested pipeline outputs
        # (e.g. pre-fusion features) must not leak into the decoder.
        exportable.algorithm.predict.assert_called_once_with(
            features={"encoded": requested_feature},
            network=exportable.decoder,
        )

    def test_raises_when_decoder_requests_unavailable_key(
        self,
        exportable_factory,
        encoding_pipeline_factory,
    ):
        pipeline = encoding_pipeline_factory()
        pipeline.return_value = {"encoded": torch.zeros(2, 64)}
        exportable = exportable_factory(
            observation_keys=["left"],
            action_keys=["position"],
            pipeline=pipeline,
        )
        exportable.decoder.decoder_input.keys = ["encoded", "proprioception"]

        with pytest.raises(ValueError, match="proprioception"):
            exportable(torch.zeros(2, 3, 64, 64))

    def test_returns_tuple_in_action_key_order(
        self,
        observation_tensor_factory,
        exportable_factory,
    ):
        exportable = exportable_factory(
            observation_keys=["left"],
            action_keys=["gripper", "orientation", "position"],
        )
        gripper = torch.zeros(2, 16, 1)
        orientation = torch.ones(2, 16, 1)
        position = torch.full((2, 16, 3), 2.0)
        exportable.algorithm.predict.return_value = {
            "position": position,
            "gripper": gripper,
            "orientation": orientation,
        }

        result = exportable(torch.zeros(2, 3, 64, 64))

        assert len(result) == 3
        assert torch.equal(result[0], gripper)
        assert torch.equal(result[1], orientation)
        assert torch.equal(result[2], position)

    @pytest.mark.parametrize("tensor_count", [1, 3])
    def test_raises_on_wrong_tensor_count(self, exportable_factory, tensor_count):
        exportable = exportable_factory(observation_keys=["depth", "left"])
        tensors = [torch.zeros(2, 3, 64, 64)] * tensor_count

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Expected 2 observation tensors matching keys "
                f"['depth', 'left'], got {tensor_count}"
            ),
        ):
            exportable(*tensors)


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

        exportable = ExportablePolicy.from_policy(policy=policy)

        assert set(exportable.observation_keys) == expected

    def test_keys_are_sorted(self, from_policy_factory):
        policy = from_policy_factory(
            encoder_keys={"rgb": ["right", "left"]},
            action_keys=["position", "gripper", "orientation"],
        )

        exportable = ExportablePolicy.from_policy(policy=policy)

        assert exportable.observation_keys == sorted(exportable.observation_keys)
        assert exportable.action_keys == sorted(exportable.action_keys)

    def test_shares_policy_components(self, from_policy_factory):
        policy = from_policy_factory()

        exportable = ExportablePolicy.from_policy(policy=policy)

        # Verify shared reference via mutation
        param = nn.Parameter(torch.tensor(42.0))
        policy.encoding_pipeline.test_param = param
        assert exportable.encoding_pipeline.test_param is param


@pytest.mark.unit
class TestGetExampleInputs:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_creates_tensors_with_provided_shapes(self, exportable_factory, batch_size):
        exportable = exportable_factory(observation_keys=["depth", "left"])

        result = exportable.get_example_inputs(
            observation_shapes={
                "depth": (1, 1, 64, 64),
                "left": (1, 3, 128, 128),
            },
            batch_size=batch_size,
        )

        assert len(result) == 2
        assert result[0].shape == (batch_size, 1, 1, 64, 64)
        assert result[1].shape == (batch_size, 1, 3, 128, 128)
        assert result[0].dtype == torch.float32
        assert torch.equal(result[0], torch.zeros_like(result[0]))

    def test_respects_custom_observation_dtypes(self, exportable_factory):
        exportable = exportable_factory(observation_keys=["image", "tokens"])

        result = exportable.get_example_inputs(
            observation_shapes={
                "image": (3, 64, 64),
                "tokens": (128,),
            },
            batch_size=2,
            observation_dtypes={"tokens": torch.long},
        )

        assert result[0].dtype == torch.float32
        assert result[1].dtype == torch.long

    def test_output_tuple_matches_observation_key_order(self, exportable_factory):
        exportable = exportable_factory(
            observation_keys=["alpha", "beta", "gamma"],
        )

        result = exportable.get_example_inputs(
            observation_shapes={
                "alpha": (3,),
                "beta": (5,),
                "gamma": (7,),
            },
            batch_size=1,
        )

        assert result[0].shape == (1, 3)
        assert result[1].shape == (1, 5)
        assert result[2].shape == (1, 7)

    def test_raises_on_missing_shape(self, exportable_factory):
        exportable = exportable_factory(observation_keys=["depth", "left"])

        with pytest.raises(
            ValueError,
            match=re.escape(
                "No shape provided for observation key 'left'. "
                "observation_shapes must cover all observation_keys. "
                "Missing keys: {'left'}"
            ),
        ):
            exportable.get_example_inputs(
                observation_shapes={"depth": (1, 1, 64, 64)},
                batch_size=1,
            )
