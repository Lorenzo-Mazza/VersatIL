"""Tests for versatil.models.decoding.unet_input_builder module."""
import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.unet_input_builder import UNetInputBuilder
from versatil.models.encoding.encoders.constants import EncoderOutputKeys


@pytest.fixture
def unet_input_builder_factory() -> Callable[..., UNetInputBuilder]:
    """Factory for UNetInputBuilder instances."""

    def factory(
        embedding_dim: int = 64,
        has_time_dim: bool = False,
    ) -> UNetInputBuilder:
        return UNetInputBuilder(
            embedding_dim=embedding_dim,
            has_time_dim=has_time_dim,
        )

    return factory


class TestUNetInputBuilderInitialization:

    @pytest.mark.parametrize("embedding_dim", [32, 64])
    @pytest.mark.parametrize("has_time_dim", [True, False])
    def test_stores_configuration(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
        embedding_dim: int,
        has_time_dim: bool,
    ):
        builder = unet_input_builder_factory(
            embedding_dim=embedding_dim,
            has_time_dim=has_time_dim,
        )
        assert builder.embedding_dim == embedding_dim
        assert builder.has_time_dim is has_time_dim
        assert builder.projection is not None


class TestUNetInputBuilderFeatureFiltering:

    def test_excludes_padding_mask_keys(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        rng: np.random.Generator,
    ):
        embedding_dim = 64
        builder = unet_input_builder_factory(embedding_dim=embedding_dim)
        captured_keys: list[str] = []

        def capturing_projection(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            captured_keys.extend(features.keys())
            return features

        builder.projection.forward = capturing_projection
        features = flat_feature_factory(
            feature_dim=embedding_dim,
            feature_keys=["rgb_features"],
        )
        padding_mask_key = f"rgb_features_{EncoderOutputKeys.PADDING_MASK.value}"
        features[padding_mask_key] = torch.from_numpy(
            rng.standard_normal((2,)).astype(np.float32)
        )
        result = builder(features)
        assert padding_mask_key not in captured_keys
        assert "rgb_features" in captured_keys
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, embedding_dim)

    def test_excludes_is_pad_action_key(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dim = 64
        builder = unet_input_builder_factory(embedding_dim=embedding_dim)
        captured_keys: list[str] = []

        def capturing_projection(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            captured_keys.extend(features.keys())
            return features

        builder.projection.forward = capturing_projection
        features = flat_feature_factory(
            feature_dim=embedding_dim,
            feature_keys=["flat_feature"],
        )
        features[SampleKey.IS_PAD_ACTION.value] = torch.zeros(2, 4, dtype=torch.bool)
        result = builder(features)
        assert SampleKey.IS_PAD_ACTION.value not in captured_keys
        assert "flat_feature" in captured_keys
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, embedding_dim)


class TestUNetInputBuilderFeatureShapes:

    def test_2d_flat_feature_kept_as_is(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dim = 64
        builder = unet_input_builder_factory(embedding_dim=embedding_dim)
        builder.projection.forward = lambda features: features
        features = flat_feature_factory(
            feature_dim=embedding_dim,
            feature_keys=["pooled"],
        )
        input_value = features["pooled"]
        result = builder(features)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, embedding_dim)
        assert torch.equal(result, input_value)

    @pytest.mark.parametrize("sequence_length", [4, 8])
    def test_3d_sequential_feature_flattened(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
        sequential_feature_factory: Callable[..., dict[str, torch.Tensor]],
        sequence_length: int,
    ):
        embedding_dim = 64
        builder = unet_input_builder_factory(embedding_dim=embedding_dim)
        builder.projection.forward = lambda features: features
        features = sequential_feature_factory(
            sequence_length=sequence_length,
            feature_dimension=embedding_dim,
        )
        input_value = features["seq_feature"]
        result = builder(features)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, sequence_length * embedding_dim)
        assert torch.equal(result, input_value.reshape(2, -1))

    def test_4d_temporal_sequential_feature_flattened(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
        temporal_flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dim = 64
        observation_horizon = 2
        sequence_length = 4
        builder = unet_input_builder_factory(
            embedding_dim=embedding_dim,
            has_time_dim=True,
        )
        builder.projection.forward = lambda features: features
        features = temporal_flat_feature_factory(
            observation_horizon=observation_horizon,
            sequence_length=sequence_length,
            feature_dimension=embedding_dim,
        )
        input_value = features["temporal_seq_feature"]
        result = builder(features)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, observation_horizon * sequence_length * embedding_dim)
        assert torch.equal(result, input_value.reshape(2, -1))

    def test_4d_spatial_without_time_dim_raises(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dim = 64
        builder = unet_input_builder_factory(
            embedding_dim=embedding_dim,
            has_time_dim=False,
        )
        builder.projection.forward = lambda features: features
        feature_name = "spatial_feature"
        features = spatial_feature_factory(
            channels=embedding_dim,
            height=4,
            width=4,
            feature_keys=[feature_name],
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"4D feature '{feature_name}' with no time dimension is not supported "
                f"as input to U-Net Decoder."
            ),
        ):
            builder(features)

    def test_5d_temporal_spatial_raises(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
        temporal_spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dim = 64
        builder = unet_input_builder_factory(
            embedding_dim=embedding_dim,
            has_time_dim=True,
        )
        builder.projection.forward = lambda features: features
        feature_name = "video_feature"
        features = temporal_spatial_feature_factory(
            channels=embedding_dim,
            height=4,
            width=4,
            feature_keys=[feature_name],
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"5D feature '{feature_name}' is not supported as input to U-Net Decoder."
            ),
        ):
            builder(features)

    def test_unsupported_ndim_raises(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
        rng: np.random.Generator,
    ):
        embedding_dim = 64
        builder = unet_input_builder_factory(embedding_dim=embedding_dim)
        builder.projection.forward = lambda features: features
        feature_name = "bad_feature"
        bad_tensor = torch.from_numpy(
            rng.standard_normal((2, 3, 4, 5, 6, 7)).astype(np.float32)
        )
        features = {feature_name: bad_tensor}
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Feature '{feature_name}' has unsupported shape {bad_tensor.shape}"
            ),
        ):
            builder(features)


class TestUNetInputBuilderCLSToken:

    def test_cls_token_appended_at_end(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dim = 64
        builder = unet_input_builder_factory(embedding_dim=embedding_dim)
        builder.projection.forward = lambda features: features
        features = flat_feature_factory(
            feature_dim=embedding_dim,
            feature_keys=["aaa_feature"],
        )
        cls_features = flat_feature_factory(
            feature_dim=embedding_dim,
            feature_keys=[DecoderOutputKey.CLASS_TOKEN.value],
        )
        features.update(cls_features)
        aaa_value = features["aaa_feature"]
        cls_value = cls_features[DecoderOutputKey.CLASS_TOKEN.value]
        result = builder(features)
        assert isinstance(result, torch.Tensor)
        # Concatenated along dim=-1: aaa_feature (64) + cls_token (64) = 128
        assert result.shape == (2, 2 * embedding_dim)
        # CLS at end: first embedding_dim columns are aaa_feature, last are cls
        assert torch.equal(result[:, :embedding_dim], aaa_value)
        assert torch.equal(result[:, embedding_dim:], cls_value)


class TestUNetInputBuilderMultipleFeatures:

    def test_features_concatenated_in_sorted_order(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dim = 64
        builder = unet_input_builder_factory(embedding_dim=embedding_dim)
        builder.projection.forward = lambda features: features
        z_features = flat_feature_factory(
            feature_dim=embedding_dim,
            feature_keys=["z_feature"],
        )
        a_features = flat_feature_factory(
            feature_dim=embedding_dim,
            feature_keys=["a_feature"],
        )
        features = {**z_features, **a_features}
        a_value = a_features["a_feature"]
        z_value = z_features["z_feature"]
        result = builder(features)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2 * embedding_dim)
        # Sorted: a_feature first, z_feature second
        assert torch.equal(result[:, :embedding_dim], a_value)
        assert torch.equal(result[:, embedding_dim:], z_value)

    def test_mixed_2d_and_3d_features_concatenated(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        sequential_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dim = 64
        sequence_length = 4
        builder = unet_input_builder_factory(embedding_dim=embedding_dim)
        builder.projection.forward = lambda features: features
        flat_features = flat_feature_factory(
            feature_dim=embedding_dim,
            feature_keys=["pooled"],
        )
        seq_features = sequential_feature_factory(
            sequence_length=sequence_length,
            feature_dimension=embedding_dim,
        )
        features = {**flat_features, **seq_features}
        flat_value = flat_features["pooled"]
        seq_value = seq_features["seq_feature"]
        result = builder(features)
        assert isinstance(result, torch.Tensor)
        expected_dim = embedding_dim + sequence_length * embedding_dim
        assert result.shape == (2, expected_dim)
        # Sorted: "pooled" before "seq_feature"
        assert torch.equal(result[:, :embedding_dim], flat_value)
        assert torch.equal(result[:, embedding_dim:], seq_value.reshape(2, -1))


class TestUNetInputBuilderReturnValue:

    def test_empty_features_returns_none(
        self,
        unet_input_builder_factory: Callable[..., UNetInputBuilder],
    ):
        builder = unet_input_builder_factory(embedding_dim=64)
        builder.projection.forward = lambda features: features
        result = builder({})
        assert result is None