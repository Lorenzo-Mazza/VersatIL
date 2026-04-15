"""Tests for versatil.models.encoding.pipeline module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.metadata import CameraMetadata
from versatil.data.task import ObservationSpace
from versatil.data.tokenization import Tokenizer
from versatil.models.encoding.pipeline import EncodingPipeline
from versatil.models.feature_meta import FeatureType


class TestSetupEncoders:
    def test_registers_unconditional_encoder_in_encoders_dict(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        assert "rgb" in pipeline.encoders
        assert "rgb" not in pipeline.conditional_encoders

    def test_registers_conditional_encoder_in_conditional_dict(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        rgb = encoder_mock_factory(input_keys=["left"])
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": rgb, "film": film},
        )
        assert "film" in pipeline.conditional_encoders
        assert "film" not in pipeline.encoders

    def test_sets_name_attribute_on_encoder(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory()
        EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"my_encoder": encoder},
        )
        assert encoder.name == "my_encoder"

    def test_registers_prefixed_feature_dimensions(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory(
            output_features=["embedding"],
            output_dimensions={"embedding": (128,)},
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        assert pipeline._feature_registry["rgb_embedding"].dimension == (128,)

    def test_stores_encoder_feature_keys(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory(
            output_features=["feat_a", "feat_b"],
            output_dimensions={"feat_a": (32,), "feat_b": (64,)},
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"vlm": encoder}
        )
        assert pipeline._encoder_feature_keys["vlm"] == ["feat_a", "feat_b"]

    @pytest.mark.parametrize(
        "encoder_name, output_features, expected_registry_keys",
        [
            ("left", ["rgb"], ["left_rgb"]),
            (
                "left",
                ["rgb:agentview_rgb", "rgb:eye_in_hand_rgb"],
                ["left_rgb:agentview_rgb", "left_rgb:eye_in_hand_rgb"],
            ),
            ("left", ["depth"], ["left_depth"]),
            (
                "left",
                ["depth:left", "depth:right"],
                ["left_depth:left", "left_depth:right"],
            ),
            (
                "instruction",
                ["language", "language_padding_mask"],
                ["instruction_language", "instruction_language_padding_mask"],
            ),
            (
                "vlm",
                ["rgb:left", "rgb:right", "language", "language_padding_mask"],
                [
                    "vlm_rgb:left",
                    "vlm_rgb:right",
                    "vlm_language",
                    "vlm_language_padding_mask",
                ],
            ),
        ],
    )
    def test_registry_keys_follow_encoder_name_underscore_output_key_convention(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
        encoder_name: str,
        output_features: list[str],
        expected_registry_keys: list[str],
    ):
        encoder = encoder_mock_factory(
            output_features=output_features,
            output_dimensions=dict.fromkeys(output_features, (16,)),
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={encoder_name: encoder},
        )
        for key in expected_registry_keys:
            assert key in pipeline._feature_registry, (
                f"Expected '{key}' in registry, got: "
                f"{list(pipeline._feature_registry.keys())}"
            )
        assert pipeline._encoder_feature_keys[encoder_name] == output_features


class TestSetupFusionModules:
    def test_registers_fusion_output_in_feature_registry(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory(
            output_features=["embedding"],
            output_dimensions={"embedding": (64,)},
        )
        fusion = fusion_module_mock_factory(
            input_features=["rgb_embedding"],
            output_name="fused",
            output_dimension=128,
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": encoder},
            fusion_stages=[fusion],
        )
        assert "fused" in pipeline._feature_registry
        assert pipeline._feature_registry["fused"].dimension == (128,)

    def test_calls_setup_on_fusion_module(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory()
        fusion = fusion_module_mock_factory(input_features=["rgb_embedding"])
        EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": encoder},
            fusion_stages=[fusion],
        )
        fusion.setup.assert_called_once()

    def test_none_fusion_stages_creates_empty_list(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": encoder},
            fusion_stages=None,
        )
        assert len(pipeline.fusion_stages) == 0


class TestResolveFeatureName:
    def test_resolves_existing_feature_name(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory(
            output_features=["embedding"],
            output_dimensions={"embedding": (64,)},
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        result = pipeline._resolve_feature_name(name="rgb_embedding")
        assert result == "rgb_embedding"

    def test_raises_for_unknown_feature_name(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory(
            output_features=["embedding"],
            output_dimensions={"embedding": (64,)},
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Feature 'nonexistent' not found. Available: ['rgb_embedding']"
            ),
        ):
            pipeline._resolve_feature_name(name="nonexistent")


class TestValidatePipeline:
    @pytest.mark.parametrize(
        "condition_key, expectation",
        [
            ("rgb_embedding", does_not_raise()),
            (
                "nonexistent_feature",
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Condition key 'nonexistent_feature' for encoder 'film' "
                        "not available. Available: {'rgb_embedding'}"
                    ),
                ),
            ),
        ],
    )
    def test_conditional_encoder_condition_key_validation(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
        condition_key: str,
        expectation,
        default_observation_space,
    ):
        rgb = encoder_mock_factory(input_keys=["left"])
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key=condition_key,
        )
        with expectation:
            EncodingPipeline(
                observation_space=default_observation_space,
                encoders={"rgb": rgb, "film": film},
            )

    def test_raises_for_missing_fusion_input_feature(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory()
        fusion = fusion_module_mock_factory(
            input_features=["nonexistent_feature"],
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Feature 'nonexistent_feature' not found. Available: ['rgb_embedding']"
            ),
        ):
            EncodingPipeline(
                observation_space=default_observation_space,
                encoders={"rgb": encoder},
                fusion_stages=[fusion],
            )

    def test_chained_fusion_stages_validate_correctly(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder_a = encoder_mock_factory(input_keys=["left"])
        encoder_b = encoder_mock_factory(input_keys=["right"])
        fusion_first = fusion_module_mock_factory(
            input_features=["a_embedding", "b_embedding"],
            output_name="fused_ab",
        )
        fusion_second = fusion_module_mock_factory(
            input_features=["fused_ab"],
            output_name="final",
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"a": encoder_a, "b": encoder_b},
            fusion_stages=[fusion_first, fusion_second],
        )
        assert "final" in pipeline._feature_registry
        assert "fused_ab" in pipeline._feature_registry


class TestFlattenObservationDict:
    def test_flat_dict_returns_unchanged(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
        default_observation_space,
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        tensor = torch.from_numpy(rng.standard_normal((2, 3)).astype(np.float32))
        observation = {"left": tensor}
        result = pipeline._flatten_observation_dict(observation=observation)
        assert "left" in result
        assert torch.equal(result["left"], tensor)

    def test_nested_dict_gets_flattened(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
        default_observation_space,
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        image_tensor = torch.from_numpy(
            rng.standard_normal((2, 3, 84, 84)).astype(np.float32)
        )
        proprio_tensor = torch.from_numpy(
            rng.standard_normal((2, 7)).astype(np.float32)
        )
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
        default_observation_space,
    ):
        encoder = encoder_mock_factory(input_keys=["left"])
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
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
        default_observation_space,
    ):
        feature_tensor = torch.from_numpy(
            rng.standard_normal((2, 64)).astype(np.float32)
        )
        encoder = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": feature_tensor},
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
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
        default_observation_space,
    ):
        encoder = encoder_mock_factory(input_keys=["left"])
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        observation = {
            "right": torch.from_numpy(
                rng.standard_normal((2, 3, 84, 84)).astype(np.float32)
            )
        }
        result = pipeline.forward(observation=observation)
        encoder.assert_not_called()
        mock_logging.warning.assert_called_once()
        assert "rgb_embedding" not in result

    def test_applies_fusion_to_encoder_outputs(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
        default_observation_space,
    ):
        feature_tensor = torch.from_numpy(
            rng.standard_normal((2, 64)).astype(np.float32)
        )
        fused_tensor = torch.from_numpy(
            rng.standard_normal((2, 128)).astype(np.float32)
        )
        encoder = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": feature_tensor},
        )
        fusion = fusion_module_mock_factory(
            input_features=["rgb_embedding"],
            output_name="fused_visual",
            forward_return=fused_tensor,
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": encoder},
            fusion_stages=[fusion],
        )
        image = torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32))
        result = pipeline.forward(observation={"left": image})
        assert "fused_visual" in result
        assert "rgb_embedding" in result

    def test_preserves_all_features_including_fused_inputs(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
        default_observation_space,
    ):
        encoder_a = encoder_mock_factory(
            input_keys=["left"],
            forward_return={
                "embedding": torch.from_numpy(
                    rng.standard_normal((2, 64)).astype(np.float32)
                )
            },
        )
        encoder_b = encoder_mock_factory(
            input_keys=["right"],
            forward_return={
                "embedding": torch.from_numpy(
                    rng.standard_normal((2, 64)).astype(np.float32)
                )
            },
        )
        fusion = fusion_module_mock_factory(
            input_features=["a_embedding", "b_embedding"],
            output_name="fused",
            forward_return=torch.from_numpy(
                rng.standard_normal((2, 128)).astype(np.float32)
            ),
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"a": encoder_a, "b": encoder_b},
            fusion_stages=[fusion],
        )
        observation = {
            "left": torch.from_numpy(
                rng.standard_normal((2, 3, 84, 84)).astype(np.float32)
            ),
            "right": torch.from_numpy(
                rng.standard_normal((2, 3, 84, 84)).astype(np.float32)
            ),
        }
        result = pipeline.forward(observation=observation)
        assert "fused" in result
        assert "a_embedding" in result
        assert "b_embedding" in result

    def test_skips_feature_not_returned_by_encoder(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
        default_observation_space,
    ):
        encoder = encoder_mock_factory(
            output_features=["embedding", "extra"],
            output_dimensions={"embedding": (64,), "extra": (32,)},
            input_keys=["left"],
            forward_return={
                "embedding": torch.from_numpy(
                    rng.standard_normal((2, 64)).astype(np.float32)
                )
            },
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        image = torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32))
        result = pipeline.forward(observation={"left": image})
        assert "rgb_embedding" in result
        assert "rgb_extra" not in result

    def test_squeezes_singleton_time_dimension(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
        default_observation_space,
    ):
        # Feature with shape (batch, 1, dim) should be squeezed to (batch, dim)
        feature_with_time = torch.from_numpy(
            rng.standard_normal((2, 1, 64)).astype(np.float32)
        )
        encoder = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": feature_with_time},
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        image = torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32))
        result = pipeline.forward(observation={"left": image})
        assert result["rgb_embedding"].shape == (2, 64)

    def test_does_not_squeeze_multi_step_time_dimension(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
        default_observation_space,
    ):
        feature_with_time = torch.from_numpy(
            rng.standard_normal((2, 3, 64)).astype(np.float32)
        )
        encoder = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": feature_with_time},
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        image = torch.from_numpy(rng.standard_normal((2, 3, 84, 84)).astype(np.float32))
        result = pipeline.forward(observation={"left": image})
        assert result["rgb_embedding"].shape == (2, 3, 64)

    def test_conditional_encoder_receives_conditioning_feature(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
        rng: np.random.Generator,
        default_observation_space,
    ):
        rgb_features = torch.from_numpy(rng.standard_normal((2, 64)).astype(np.float32))
        rgb = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": rgb_features},
        )
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
            forward_return={
                "embedding": torch.from_numpy(
                    rng.standard_normal((2, 64)).astype(np.float32)
                )
            },
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": rgb, "film": film},
        )
        observation = {
            "left": torch.from_numpy(
                rng.standard_normal((2, 3, 84, 84)).astype(np.float32)
            ),
            "right": torch.from_numpy(
                rng.standard_normal((2, 3, 84, 84)).astype(np.float32)
            ),
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
        default_observation_space,
    ):
        rgb_features = torch.from_numpy(rng.standard_normal((2, 64)).astype(np.float32))
        rgb = encoder_mock_factory(
            input_keys=["left"],
            forward_return={"embedding": rgb_features},
        )
        film = conditional_encoder_mock_factory(
            output_features=["embedding", "extra"],
            output_dimensions={"embedding": (64,), "extra": (32,)},
            input_keys=["right"],
            condition_key="rgb_embedding",
            forward_return={
                "embedding": torch.from_numpy(
                    rng.standard_normal((2, 64)).astype(np.float32)
                )
            },
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": rgb, "film": film},
        )
        observation = {
            "left": torch.from_numpy(
                rng.standard_normal((2, 3, 84, 84)).astype(np.float32)
            ),
            "right": torch.from_numpy(
                rng.standard_normal((2, 3, 84, 84)).astype(np.float32)
            ),
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
        default_observation_space,
    ):
        rgb = encoder_mock_factory(
            input_keys=["left"],
            forward_return={
                "embedding": torch.from_numpy(
                    rng.standard_normal((2, 64)).astype(np.float32)
                )
            },
        )
        film = conditional_encoder_mock_factory(
            input_keys=["depth"],
            condition_key="rgb_embedding",
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": rgb, "film": film},
        )
        observation = {
            "left": torch.from_numpy(
                rng.standard_normal((2, 3, 84, 84)).astype(np.float32)
            ),
        }
        pipeline.forward(observation=observation)
        film.assert_not_called()
        assert mock_logging.warning.call_count >= 1


class TestGetFeatureNames:
    def test_includes_encoder_and_fusion_features(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory()
        fusion = fusion_module_mock_factory(
            input_features=["rgb_embedding"],
            output_name="fused",
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": encoder},
            fusion_stages=[fusion],
        )
        names = pipeline.get_feature_names()
        assert "rgb_embedding" in names
        assert "fused" in names

    def test_returns_only_encoder_features_without_fusion(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        names = pipeline.get_feature_names()
        assert names == ["rgb_embedding"]


class TestGetFeaturesToDimensions:
    def test_returns_all_feature_dimensions(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory(
            output_dimensions={"embedding": (64,)},
        )
        fusion = fusion_module_mock_factory(
            input_features=["rgb_embedding"],
            output_name="fused",
            output_dimension=128,
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": encoder},
            fusion_stages=[fusion],
        )
        dims = pipeline.get_features_to_dimensions()
        assert dims["rgb_embedding"] == (64,)
        assert dims["fused"] == (128,)

    def test_get_features_returns_feature_metadata_dict(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory(
            output_features=["rgb"],
            output_dimensions={"rgb": (256,)},
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"left": encoder},
        )
        features = pipeline.get_features()
        assert "left_rgb" in features
        meta = features["left_rgb"]
        assert meta.key == "left_rgb"
        assert meta.dimension == (256,)
        assert meta.feature_type == FeatureType.FLAT.value


class TestSetTokenizer:
    def test_raises_when_encoder_requires_tokenized_but_no_tokenizer(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory(requires_tokenized=True)
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"language": encoder}
        )
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
        default_observation_space,
    ):
        encoder = encoder_mock_factory(requires_tokenized=True)
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"language": encoder}
        )
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

    @pytest.mark.parametrize(
        "encoder_vocab_size, data_vocab_size, base_vocab_size, expectation",
        [
            (50000, 50000, 50000, does_not_raise()),
            (50000, 30000, 30000, does_not_raise()),
            (30000, 30001, 30000, does_not_raise()),
            (30000, 50000, 30000, does_not_raise()),
            (
                30000,
                50000,
                50000,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Vocab size mismatch: Observation tokenizer has vocab_size=50000 "
                        "(base=50000), but encoder 'language' only supports vocab_size=30000. "
                    ),
                ),
            ),
        ],
    )
    def test_unconditional_encoder_vocab_size_validation(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        encoder_vocab_size: int,
        data_vocab_size: int,
        base_vocab_size: int,
        expectation,
        default_observation_space,
    ):
        encoder = encoder_mock_factory(
            requires_tokenized=True,
            vocab_size=encoder_vocab_size,
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"language": encoder}
        )
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = MagicMock()
        tokenizer.observation_tokenizer.vocab_size = data_vocab_size
        tokenizer.observation_tokenizer.language_tokenizer.vocab_size = base_vocab_size
        tokenizer.observation_tokenizer.tokenizer_model = "test_model"
        with expectation:
            pipeline.set_tokenizer(tokenizer=tokenizer)

    def test_skips_encoders_not_requiring_tokenized(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory(requires_tokenized=False)
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        pipeline.set_tokenizer(tokenizer=None)

    def test_validates_conditional_encoders_too(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        rgb = encoder_mock_factory(input_keys=["left"])
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
        )
        film.input_specification.requires_tokenized = True
        film.get_vocab_size.return_value = 1000
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": rgb, "film": film},
        )
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
        default_observation_space,
    ):
        rgb = encoder_mock_factory(input_keys=["left"])
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
        )
        film.input_specification.requires_tokenized = True
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": rgb, "film": film},
        )
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

    @pytest.mark.parametrize(
        "encoder_vocab_size, data_vocab_size, base_vocab_size, expectation",
        [
            (50000, 50000, 50000, does_not_raise()),
            (50000, 30000, 30000, does_not_raise()),
            (30000, 30001, 30000, does_not_raise()),
            (30000, 50000, 30000, does_not_raise()),
            (
                30000,
                50000,
                50000,
                pytest.raises(
                    ValueError,
                    match=re.escape(
                        "Vocab size mismatch: Observation tokenizer has vocab_size=50000 "
                        "(base=50000), but encoder 'film' only supports vocab_size=30000. "
                    ),
                ),
            ),
        ],
    )
    def test_conditional_encoder_vocab_size_validation(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
        encoder_vocab_size: int,
        data_vocab_size: int,
        base_vocab_size: int,
        expectation,
        default_observation_space,
    ):
        rgb = encoder_mock_factory(input_keys=["left"])
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
        )
        film.input_specification.requires_tokenized = True
        film.get_vocab_size.return_value = encoder_vocab_size
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": rgb, "film": film},
        )
        tokenizer = MagicMock(spec=Tokenizer)
        tokenizer.observation_tokenizer = MagicMock()
        tokenizer.observation_tokenizer.vocab_size = data_vocab_size
        tokenizer.observation_tokenizer.language_tokenizer.vocab_size = base_vocab_size
        tokenizer.observation_tokenizer.tokenizer_model = "test_model"
        with expectation:
            pipeline.set_tokenizer(tokenizer=tokenizer)


class TestImageSizeSetDuringSetup:
    @pytest.fixture
    def camera_observation_space_factory(
        self, observation_space_factory
    ) -> Callable[..., ObservationSpace]:
        """Factory for ObservationSpace with specific camera dimensions."""

        def factory(
            cameras: dict[str, tuple[int, int]],
        ) -> ObservationSpace:
            metadata = {}
            for camera_key, (height, width) in cameras.items():
                metadata[camera_key] = CameraMetadata(
                    camera_key=camera_key,
                    dtype="uint8",
                    channels=3,
                    image_height=height,
                    image_width=width,
                )
            return observation_space_factory(observations_metadata=metadata)

        return factory

    @pytest.mark.parametrize(
        "image_height, image_width",
        [(224, 224), (256, 256), (480, 640)],
    )
    def test_passes_camera_dimensions_to_encoder_at_construction(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        camera_observation_space_factory,
        image_height: int,
        image_width: int,
    ):
        encoder = encoder_mock_factory(input_keys=["left"], is_image_encoder=True)
        observation_space = camera_observation_space_factory(
            cameras={"left": (image_height, image_width)}
        )
        EncodingPipeline(observation_space=observation_space, encoders={"rgb": encoder})
        encoder.set_image_size.assert_called_once_with(
            image_height=image_height, image_width=image_width
        )

    def test_skips_encoder_without_camera_keys(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory(input_keys=["proprio_robot_frame"])
        encoder.set_image_size = MagicMock()
        EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"proprio": encoder},
        )
        encoder.set_image_size.assert_not_called()

    @pytest.mark.parametrize(
        "left_size, right_size, expectation",
        [
            ((256, 256), (256, 256), does_not_raise()),
            (
                (256, 256),
                (128, 128),
                pytest.raises(
                    ValueError,
                    match=r"has cameras with different resolutions",
                ),
            ),
        ],
    )
    def test_multi_camera_resolution_validation(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        camera_observation_space_factory,
        left_size: tuple[int, int],
        right_size: tuple[int, int],
        expectation,
    ):
        encoder = encoder_mock_factory(
            input_keys=["left", "right"], is_image_encoder=True
        )
        observation_space = camera_observation_space_factory(
            cameras={"left": left_size, "right": right_size}
        )
        with expectation:
            EncodingPipeline(
                observation_space=observation_space,
                encoders={"vision": encoder},
            )

    def test_raises_when_camera_not_in_observation_space(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        camera_observation_space_factory,
    ):
        encoder = encoder_mock_factory(input_keys=["left"], is_image_encoder=True)
        observation_space = camera_observation_space_factory(cameras={})
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Encoder 'rgb' expects camera key 'left' but it is not in "
                "the observation space cameras: []"
            ),
        ):
            EncodingPipeline(
                observation_space=observation_space,
                encoders={"rgb": encoder},
            )

    def test_processes_conditional_encoders_with_camera_keys(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
        camera_observation_space_factory,
    ):
        unconditional = encoder_mock_factory(input_keys=["left"], is_image_encoder=True)
        conditional = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
            is_image_encoder=True,
        )
        observation_space = camera_observation_space_factory(
            cameras={"left": (256, 256), "right": (256, 256)}
        )
        EncodingPipeline(
            observation_space=observation_space,
            encoders={"rgb": unconditional, "film": conditional},
        )
        unconditional.set_image_size.assert_called_once_with(
            image_height=256, image_width=256
        )
        conditional.set_image_size.assert_called_once_with(
            image_height=256, image_width=256
        )


class TestRepr:
    def test_contains_encoder_names(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb_encoder": encoder},
        )
        representation = repr(pipeline)
        assert "rgb_encoder" in representation
        assert "EncodingPipeline" in representation

    def test_contains_conditional_encoder_names(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        conditional_encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        rgb = encoder_mock_factory(input_keys=["left"])
        film = conditional_encoder_mock_factory(
            input_keys=["right"],
            condition_key="rgb_embedding",
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb": rgb, "film_encoder": film},
        )
        representation = repr(pipeline)
        assert "rgb" in representation
        assert "film_encoder" in representation

    def test_without_fusion_stages_omits_fusion_section(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory()
        pipeline = EncodingPipeline(
            observation_space=default_observation_space, encoders={"rgb": encoder}
        )
        representation = repr(pipeline)
        assert "Fusion stages" not in representation

    def test_contains_fusion_stage_info(
        self,
        encoder_mock_factory: Callable[..., MagicMock],
        fusion_module_mock_factory: Callable[..., MagicMock],
        default_observation_space,
    ):
        encoder = encoder_mock_factory()
        fusion = fusion_module_mock_factory(
            input_features=["rgb_encoder_embedding"],
            output_name="fused_visual",
        )
        pipeline = EncodingPipeline(
            observation_space=default_observation_space,
            encoders={"rgb_encoder": encoder},
            fusion_stages=[fusion],
        )
        representation = repr(pipeline)
        assert "Fusion stages" in representation
        assert "fused_visual" in representation
