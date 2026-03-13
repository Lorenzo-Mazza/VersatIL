"""Tests for versatil.models.encoding.pipeline module."""
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.tokenization import Tokenizer
from versatil.models.encoding.encoders.base import EncoderOutput
from versatil.models.encoding.pipeline import EncodingPipeline


class TestSetupEncoders:

    def test_registers_unconditional_encoder_in_encoders_dict(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        assert "rgb" in pipeline.encoders
        assert "rgb" not in pipeline.conditional_encoders

    def test_registers_conditional_encoder_in_conditional_dict(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
    ):
        rgb = encoder_mock_factory(input_keys=["left"])
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
        )
        pipeline = EncodingPipeline(encoders={"rgb": rgb, "film": film})
        assert "film" in pipeline.conditional_encoders
        assert "film" not in pipeline.encoders

    def test_sets_name_attribute_on_encoder(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory()
        EncodingPipeline(encoders={"my_encoder": encoder})
        assert encoder.name == "my_encoder"

    def test_registers_prefixed_feature_dimensions(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory(
            output_features=["embedding"],
            output_dimensions={"embedding": 128},
        )
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        assert pipeline._feature_keys_to_dims["rgb_embedding"] == 128

    def test_stores_encoder_output_specifications(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory(
            output_features=["feat_a", "feat_b"],
            output_dimensions={"feat_a": 32, "feat_b": 64},
        )
        pipeline = EncodingPipeline(encoders={"vlm": encoder})
        output = pipeline.encoder_to_outputs["vlm"]
        assert output.features == ["feat_a", "feat_b"]
        assert output.dimensions == {"feat_a": 32, "feat_b": 64}


class TestSetupFusionModules:

    def test_resolves_input_feature_names(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory(
            output_features=["embedding"],
            output_dimensions={"embedding": 64},
        )
        fusion = fusion_module_mock_factory(
            input_features=["rgb"],
            output_name="fused",
            output_dimension=128,
        )
        pipeline = EncodingPipeline(
            encoders={"rgb": encoder},
            fusion_stages=[fusion],
        )
        assert "fused" in pipeline._feature_keys_to_dims
        assert pipeline._feature_keys_to_dims["fused"] == 128

    def test_tracks_consumed_features(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
    ):
        encoder_a = encoder_mock_factory(input_keys=["left"])
        encoder_b = encoder_mock_factory(input_keys=["right"])
        fusion = fusion_module_mock_factory(
            input_features=["a", "b"],
            output_name="fused",
        )
        pipeline = EncodingPipeline(
            encoders={"a": encoder_a, "b": encoder_b},
            fusion_stages=[fusion],
        )
        assert "a_embedding" in pipeline._consumed_features
        assert "b_embedding" in pipeline._consumed_features

    def test_calls_setup_on_fusion_module(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory()
        fusion = fusion_module_mock_factory(input_features=["rgb"])
        EncodingPipeline(
            encoders={"rgb": encoder},
            fusion_stages=[fusion],
        )
        fusion.setup.assert_called_once()

    def test_none_fusion_stages_creates_empty_list(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(encoders={"rgb": encoder}, fusion_stages=None)
        assert len(pipeline.fusion_stages) == 0
        assert len(pipeline._consumed_features) == 0


class TestResolveFeatureName:

    @pytest.mark.parametrize("selector, expected_result, expectation", [
        ("vlm.language", "vlm_language", does_not_raise()),
        ("vlm.nonexistent", None, pytest.raises(
            ValueError,
            match=re.escape(
                "Invalid output_selector 'nonexistent' for 'vlm. "
                "Available: ['language', 'visual']'"
            ),
        )),
    ])
    def test_dot_notation_resolution(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        selector: str,
        expected_result: str | None,
        expectation,
    ):
        encoder = encoder_mock_factory(
            output_features=["language", "visual"],
            output_dimensions={"language": 64, "visual": 128},
        )
        pipeline = EncodingPipeline(encoders={"vlm": encoder})
        with expectation:
            result = pipeline._resolve_feature_name(input_specification=selector)
            assert result == expected_result

    def test_resolves_single_output_encoder_by_name(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory(
            output_features=["embedding"],
            output_dimensions={"embedding": 64},
        )
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        result = pipeline._resolve_feature_name(input_specification="rgb")
        assert result == "rgb_embedding"

    def test_raises_for_multi_output_encoder_without_selector(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory(
            output_features=["language", "visual"],
            output_dimensions={"language": 64, "visual": 128},
        )
        pipeline = EncodingPipeline(encoders={"vlm": encoder})
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Multi-output encoder 'vlm' requires selector. "
                "Available: ['language', 'visual']"
            ),
        ):
            pipeline._resolve_feature_name(input_specification="vlm")

    def test_passes_through_unrecognized_names(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        result = pipeline._resolve_feature_name(input_specification="some_fusion_output")
        assert result == "some_fusion_output"


class TestValidatePipeline:

    def test_raises_for_duplicate_encoder_output_keys(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        # Duplicate features within a single encoder's output specification
        encoder = encoder_mock_factory(input_keys=["left"])
        encoder.get_output_specification.return_value = EncoderOutput(
            features=["embedding", "embedding"],
            dimensions={"embedding": 64},
        )
        with pytest.raises(
            ValueError,
            match="Duplicate output keys detected from encoders",
        ):
            EncodingPipeline(encoders={"rgb": encoder})

    @pytest.mark.parametrize("condition_key, expectation", [
        ("rgb_embedding", does_not_raise()),
        ("nonexistent_feature", pytest.raises(
            ValueError,
            match=re.escape(
                "Condition key 'nonexistent_feature' for encoder 'film' "
                "not available. Available: {'rgb_embedding'}"
            ),
        )),
    ])
    def test_conditional_encoder_condition_key_validation(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
        condition_key: str,
        expectation,
    ):
        rgb = encoder_mock_factory(input_keys=["left"])
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key=condition_key,
        )
        with expectation:
            EncodingPipeline(encoders={"rgb": rgb, "film": film})

    def test_raises_for_missing_fusion_input_feature(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory()
        fusion = fusion_module_mock_factory(
            input_features=["nonexistent_feature"],
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Fusion stage 0 expects input feature 'nonexistent_feature' "
                "but it's not produced by any encoder or previous fusion. "
                "Available features: {'rgb_embedding'}"
            ),
        ):
            EncodingPipeline(
                encoders={"rgb": encoder},
                fusion_stages=[fusion],
            )

    def test_chained_fusion_stages_validate_correctly(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
    ):
        encoder_a = encoder_mock_factory(input_keys=["left"])
        encoder_b = encoder_mock_factory(input_keys=["right"])
        fusion_first = fusion_module_mock_factory(
            input_features=["a", "b"],
            output_name="fused_ab",
        )
        fusion_second = fusion_module_mock_factory(
            input_features=["fused_ab"],
            output_name="final",
        )
        pipeline = EncodingPipeline(
            encoders={"a": encoder_a, "b": encoder_b},
            fusion_stages=[fusion_first, fusion_second],
        )
        assert "final" in pipeline._feature_keys_to_dims
        assert "fused_ab" in pipeline._consumed_features


class TestFlattenObservationDict:

    def test_flat_dict_returns_unchanged(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        tensor = torch.from_numpy(rng.standard_normal((2, 3)).astype(np.float32))
        observation = {"left": tensor}
        result = pipeline._flatten_observation_dict(observation=observation)
        assert "left" in result
        assert torch.equal(result["left"], tensor)

    def test_nested_dict_gets_flattened(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        image_tensor = torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32))
        proprio_tensor = torch.from_numpy(rng.standard_normal((2, 7)).astype(np.float32))
        observation = {
            "left": image_tensor,
            "robot_proprio_state": {"proprio_camera_frame": proprio_tensor},
        }
        result = pipeline._flatten_observation_dict(observation=observation)
        assert "left" in result
        assert "proprio_camera_frame" in result
        assert "robot_proprio_state" not in result
        assert torch.equal(result["proprio_camera_frame"], proprio_tensor)


class TestForward:

    def test_encodes_observations_with_correct_input_keys(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        encoder = encoder_mock_factory(input_keys=["left"])
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        image = torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32))
        observation = {"left": image}
        pipeline.forward(observation=observation)
        encoder.assert_called_once()
        call_args = encoder.call_args[0][0]
        assert "left" in call_args
        assert torch.equal(call_args["left"], image)

    def test_prefixes_encoder_output_features(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        feature_tensor = torch.from_numpy(rng.standard_normal((2, 64)).astype(np.float32))
        encoder = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": feature_tensor},
        )
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        image = torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32))
        result = pipeline.forward(observation={"left": image})
        assert "rgb_embedding" in result
        assert torch.equal(result["rgb_embedding"], feature_tensor)

    @patch("versatil.models.encoding.pipeline.logging")
    def test_skips_encoder_with_missing_keys(
        self,
        mock_logging,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        encoder = encoder_mock_factory(input_keys=["left"])
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        observation = {"right": torch.from_numpy(
            rng.standard_normal((2, 3, 84, 84)).astype(np.float32)
        )}
        result = pipeline.forward(observation=observation)
        encoder.assert_not_called()
        mock_logging.warning.assert_called_once()
        assert "rgb_embedding" not in result

    def test_applies_fusion_to_encoder_outputs(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        feature_tensor = torch.from_numpy(rng.standard_normal((2, 64)).astype(np.float32))
        fused_tensor = torch.from_numpy(rng.standard_normal((2, 128)).astype(np.float32))
        encoder = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": feature_tensor},
        )
        fusion = fusion_module_mock_factory(
            input_features=["rgb"],
            output_name="fused_visual",
            forward_return=fused_tensor,
        )
        pipeline = EncodingPipeline(
            encoders={"rgb": encoder},
            fusion_stages=[fusion],
        )
        image = torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32))
        result = pipeline.forward(observation={"left": image})
        assert "fused_visual" in result
        # Consumed feature should be removed
        assert "rgb_embedding" not in result

    def test_removes_consumed_features_from_output(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        encoder_a = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": torch.from_numpy(
                rng.standard_normal((2, 64)).astype(np.float32)
            )},
        )
        encoder_b = encoder_mock_factory(
            input_keys=["right"],
            forward_return={"embedding": torch.from_numpy(
                rng.standard_normal((2, 64)).astype(np.float32)
            )},
        )
        fusion = fusion_module_mock_factory(
            input_features=["a", "b"],
            output_name="fused",
            forward_return=torch.from_numpy(
                rng.standard_normal((2, 128)).astype(np.float32)
            ),
        )
        pipeline = EncodingPipeline(
            encoders={"a": encoder_a, "b": encoder_b},
            fusion_stages=[fusion],
        )
        observation = {
            "left": torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32)),
            "right": torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32)),
        }
        result = pipeline.forward(observation=observation)
        assert "fused" in result
        assert "a_embedding" not in result
        assert "b_embedding" not in result

    def test_skips_feature_not_returned_by_encoder(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        encoder = encoder_mock_factory(
            output_features=["embedding", "extra"],
            output_dimensions={"embedding": 64, "extra": 32},
            input_keys=["left"],
            forward_return={"embedding": torch.from_numpy(
                rng.standard_normal((2, 64)).astype(np.float32)
            )},
        )
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        image = torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32))
        result = pipeline.forward(observation={"left": image})
        assert "rgb_embedding" in result
        assert "rgb_extra" not in result

    def test_squeezes_singleton_time_dimension(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        # Feature with shape (batch, 1, dim) should be squeezed to (batch, dim)
        feature_with_time = torch.from_numpy(
            rng.standard_normal((2, 1, 64)).astype(np.float32)
        )
        encoder = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": feature_with_time},
        )
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        image = torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32))
        result = pipeline.forward(observation={"left": image})
        assert result["rgb_embedding"].shape == (2, 64)

    def test_does_not_squeeze_multi_step_time_dimension(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        feature_with_time = torch.from_numpy(
            rng.standard_normal((2, 3, 64)).astype(np.float32)
        )
        encoder = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": feature_with_time},
        )
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        image = torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32))
        result = pipeline.forward(observation={"left": image})
        assert result["rgb_embedding"].shape == (2, 3, 64)

    def test_conditional_encoder_receives_conditioning_feature(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        rgb_features = torch.from_numpy(rng.standard_normal((2, 64)).astype(np.float32))
        rgb = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": rgb_features},
        )
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
            forward_return={"embedding": torch.from_numpy(
                rng.standard_normal((2, 64)).astype(np.float32)
            )},
        )
        pipeline = EncodingPipeline(encoders={"rgb": rgb, "film": film})
        observation = {
            "left": torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32)),
            "right": torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32)),
        }
        pipeline.forward(observation=observation)
        film.assert_called_once()
        conditioning_arg = film.call_args[0][1]
        assert torch.equal(conditioning_arg, rgb_features)

    def test_conditional_encoder_skips_feature_not_returned(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        rgb_features = torch.from_numpy(rng.standard_normal((2, 64)).astype(np.float32))
        rgb = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": rgb_features},
        )
        film = conditional_encoder_mock_factory(
            output_features=["embedding", "extra"],
            output_dimensions={"embedding": 64, "extra": 32},
            input_keys=["right"],
            condition_key="rgb_embedding",
            forward_return={"embedding": torch.from_numpy(
                rng.standard_normal((2, 64)).astype(np.float32)
            )},
        )
        pipeline = EncodingPipeline(encoders={"rgb": rgb, "film": film})
        observation = {
            "left": torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32)),
            "right": torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32)),
        }
        result = pipeline.forward(observation=observation)
        assert "film_embedding" in result
        assert "film_extra" not in result

    @patch("versatil.models.encoding.pipeline.logging")
    def test_skips_conditional_encoder_with_missing_keys(
        self,
        mock_logging,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
    ):
        rgb = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": torch.from_numpy(
                rng.standard_normal((2, 64)).astype(np.float32)
            )},
        )
        film = conditional_encoder_mock_factory(
            input_keys=["depth"],
            condition_key="rgb_embedding",
        )
        pipeline = EncodingPipeline(encoders={"rgb": rgb, "film": film})
        observation = {
            "left": torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32)),
        }
        pipeline.forward(observation=observation)
        film.assert_not_called()
        assert mock_logging.warning.call_count >= 1


class TestFlattenEncoderFeatureNames:

    def test_returns_prefixed_feature_names(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory(
            output_features=["embedding"],
        )
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        result = pipeline.flatten_encoder_feature_names()
        assert result == {"rgb_embedding"}

    def test_returns_all_features_from_multi_output_encoder(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory(
            output_features=["language", "visual"],
            output_dimensions={"language": 64, "visual": 128},
        )
        pipeline = EncodingPipeline(encoders={"vlm": encoder})
        result = pipeline.flatten_encoder_feature_names()
        assert result == {"vlm_language", "vlm_visual"}


class TestGetFeatureNames:

    def test_includes_encoder_and_fusion_features(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory()
        fusion = fusion_module_mock_factory(
            input_features=["rgb"],
            output_name="fused",
        )
        pipeline = EncodingPipeline(
            encoders={"rgb": encoder},
            fusion_stages=[fusion],
        )
        names = pipeline.get_feature_names()
        assert "rgb_embedding" in names
        assert "fused" in names

    def test_returns_only_encoder_features_without_fusion(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        names = pipeline.get_feature_names()
        assert names == ["rgb_embedding"]


class TestGetFinalFeatureNames:

    def test_excludes_consumed_features(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory()
        fusion = fusion_module_mock_factory(
            input_features=["rgb"],
            output_name="fused",
        )
        pipeline = EncodingPipeline(
            encoders={"rgb": encoder},
            fusion_stages=[fusion],
        )
        final_names = pipeline.get_final_feature_names()
        assert "fused" in final_names
        assert "rgb_embedding" not in final_names

    def test_includes_unconsumed_encoder_features(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
    ):
        encoder_a = encoder_mock_factory(input_keys=["left"])
        encoder_b = encoder_mock_factory(input_keys=["right"])
        fusion = fusion_module_mock_factory(
            input_features=["a"],
            output_name="fused",
        )
        pipeline = EncodingPipeline(
            encoders={"a": encoder_a, "b": encoder_b},
            fusion_stages=[fusion],
        )
        final_names = pipeline.get_final_feature_names()
        assert "fused" in final_names
        assert "b_embedding" in final_names
        assert "a_embedding" not in final_names


class TestGetFeaturesToDimensions:

    def test_returns_all_feature_dimensions(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory(
            output_dimensions={"embedding": 64},
        )
        fusion = fusion_module_mock_factory(
            input_features=["rgb"],
            output_name="fused",
            output_dimension=128,
        )
        pipeline = EncodingPipeline(
            encoders={"rgb": encoder},
            fusion_stages=[fusion],
        )
        dims = pipeline.get_features_to_dimensions()
        assert dims["rgb_embedding"] == 64
        assert dims["fused"] == 128


class TestGetFinalFeaturesToDimensions:

    def test_excludes_consumed_feature_dimensions(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory(
            output_dimensions={"embedding": 64},
        )
        fusion = fusion_module_mock_factory(
            input_features=["rgb"],
            output_name="fused",
            output_dimension=128,
        )
        pipeline = EncodingPipeline(
            encoders={"rgb": encoder},
            fusion_stages=[fusion],
        )
        final_dims = pipeline.get_final_features_to_dimensions()
        assert "fused" in final_dims
        assert final_dims["fused"] == 128
        assert "rgb_embedding" not in final_dims


class TestSetTokenizer:

    def test_raises_when_encoder_requires_tokenized_but_no_tokenizer(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory(requires_tokenized=True)
        pipeline = EncodingPipeline(encoders={"language": encoder})
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Encoder 'language' requires tokenized input, "
                "but no observation tokenizer is available."
            ),
        ):
            pipeline.set_tokenizer(tokenizer=None)

    def test_raises_when_encoder_requires_tokenized_but_no_observation_tokenizer(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory(requires_tokenized=True)
        pipeline = EncodingPipeline(encoders={"language": encoder})
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = None
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Encoder 'language' requires tokenized input, "
                "but no observation tokenizer is available."
            ),
        ):
            pipeline.set_tokenizer(tokenizer=tokenizer)

    @pytest.mark.parametrize("encoder_vocab_size, data_vocab_size, expectation", [
        (50000, 50000, does_not_raise()),
        (50000, 30000, pytest.raises(
            ValueError,
            match=re.escape(
                "Vocab size mismatch: Observation tokenizer has vocab_size=30000, "
                "but encoder 'language' expects vocab_size=50000. "
            ),
        )),
    ])
    def test_unconditional_encoder_vocab_size_validation(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        encoder_vocab_size: int,
        data_vocab_size: int,
        expectation,
    ):
        encoder = encoder_mock_factory(
            requires_tokenized=True,
            vocab_size=encoder_vocab_size,
        )
        pipeline = EncodingPipeline(encoders={"language": encoder})
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = MagicMock()
        tokenizer.observation_tokenizer.vocab_size = data_vocab_size
        tokenizer.observation_tokenizer.tokenizer_model = "test_model"
        with expectation:
            pipeline.set_tokenizer(tokenizer=tokenizer)

    def test_skips_encoders_not_requiring_tokenized(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory(requires_tokenized=False)
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        pipeline.set_tokenizer(tokenizer=None)

    def test_validates_conditional_encoders_too(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
    ):
        rgb = encoder_mock_factory(input_keys=["left"])
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
        )
        film.input_specification.requires_tokenized = True
        film.get_vocab_size.return_value = 1000
        pipeline = EncodingPipeline(encoders={"rgb": rgb, "film": film})
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Encoder 'film' requires tokenized input, "
                "but no observation tokenizer is available."
            ),
        ):
            pipeline.set_tokenizer(tokenizer=None)

    def test_conditional_encoder_raises_when_observation_tokenizer_is_none(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
    ):
        rgb = encoder_mock_factory(input_keys=["left"])
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
        )
        film.input_specification.requires_tokenized = True
        pipeline = EncodingPipeline(encoders={"rgb": rgb, "film": film})
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = None
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Encoder 'film' requires tokenized input, "
                "but no observation tokenizer is available."
            ),
        ):
            pipeline.set_tokenizer(tokenizer=tokenizer)

    @pytest.mark.parametrize("encoder_vocab_size, data_vocab_size, expectation", [
        (50000, 50000, does_not_raise()),
        (50000, 30000, pytest.raises(
            ValueError,
            match=re.escape(
                "Vocab size mismatch: Observation tokenizer has vocab_size=30000, "
                "but encoder 'film' expects vocab_size=50000. "
            ),
        )),
    ])
    def test_conditional_encoder_vocab_size_validation(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
        encoder_vocab_size: int,
        data_vocab_size: int,
        expectation,
    ):
        rgb = encoder_mock_factory(input_keys=["left"])
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
        )
        film.input_specification.requires_tokenized = True
        film.get_vocab_size.return_value = encoder_vocab_size
        pipeline = EncodingPipeline(encoders={"rgb": rgb, "film": film})
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = MagicMock()
        tokenizer.observation_tokenizer.vocab_size = data_vocab_size
        tokenizer.observation_tokenizer.tokenizer_model = "test_model"
        with expectation:
            pipeline.set_tokenizer(tokenizer=tokenizer)


class TestRepr:

    def test_contains_encoder_names(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(encoders={"rgb_encoder": encoder})
        representation = repr(pipeline)
        assert "rgb_encoder" in representation
        assert "EncodingPipeline" in representation

    def test_contains_conditional_encoder_names(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
    ):
        rgb = encoder_mock_factory(input_keys=["left"])
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
        )
        pipeline = EncodingPipeline(encoders={"rgb": rgb, "film_encoder": film})
        representation = repr(pipeline)
        assert "rgb" in representation
        assert "film_encoder" in representation

    def test_without_fusion_stages_omits_fusion_section(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(encoders={"rgb": encoder})
        representation = repr(pipeline)
        assert "Fusion stages" not in representation

    def test_contains_fusion_stage_info(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
    ):
        encoder = encoder_mock_factory()
        fusion = fusion_module_mock_factory(
            input_features=["rgb_encoder"],
            output_name="fused_visual",
        )
        pipeline = EncodingPipeline(
            encoders={"rgb_encoder": encoder},
            fusion_stages=[fusion],
        )
        representation = repr(pipeline)
        assert "Fusion stages" in representation
        assert "fused_visual" in representation