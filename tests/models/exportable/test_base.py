"""Tests for versatil.models.exportable.base module."""

import re

import pytest
import torch


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
                f"Expected 2 policy input tensors "
                f"(2 observations and 0 noise inputs), got {tensor_count}"
            ),
        ):
            exportable(*tensors)


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
