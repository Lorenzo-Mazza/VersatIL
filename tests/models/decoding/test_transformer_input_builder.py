"""Tests for versatil.models.decoding.transformer_input_builder module."""

import re
from collections.abc import Callable

import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.transformer_input_builder import TransformerInputBuilder
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.layers.positional_encoding.base import (
    PositionalEncoding1D,
    PositionalEncoding2D,
)
from versatil.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
)


@pytest.fixture
def transformer_input_builder_factory() -> Callable[..., TransformerInputBuilder]:
    """Factory for TransformerInputBuilder instances."""

    def factory(
        embedding_dimension: int = 64,
        has_time_dim: bool = False,
        spatial_positional_encoding_layer: PositionalEncoding2D | None = None,
        flat_positional_encoding_layer: PositionalEncoding1D | None = None,
        temporal_positional_encoding_layer: PositionalEncoding1D | None = None,
        use_camera_embeddings: bool = True,
        exclude_keys: list[str] | None = None,
    ) -> TransformerInputBuilder:
        return TransformerInputBuilder(
            embedding_dimension=embedding_dimension,
            has_time_dim=has_time_dim,
            spatial_positional_encoding_layer=spatial_positional_encoding_layer,
            flat_positional_encoding_layer=flat_positional_encoding_layer,
            temporal_positional_encoding_layer=temporal_positional_encoding_layer,
            use_camera_embeddings=use_camera_embeddings,
            exclude_keys=exclude_keys,
        )

    return factory


@pytest.fixture
def sinusoidal_pe_1d_factory() -> Callable[..., SinusoidalPositionalEncoding1D]:
    """Factory for SinusoidalPositionalEncoding1D instances."""

    def factory(
        embedding_dimension: int = 64,
    ) -> SinusoidalPositionalEncoding1D:
        return SinusoidalPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
        )

    return factory


@pytest.fixture
def sinusoidal_pe_2d_factory() -> Callable[..., SinusoidalPositionalEncoding2D]:
    """Factory for SinusoidalPositionalEncoding2D instances."""

    def factory(
        embedding_dimension: int = 64,
    ) -> SinusoidalPositionalEncoding2D:
        return SinusoidalPositionalEncoding2D(
            embedding_dimension=embedding_dimension,
        )

    return factory


class TestTransformerInputBuilderInitialization:
    @pytest.mark.parametrize("embedding_dimension", [32, 64])
    @pytest.mark.parametrize("has_time_dim", [True, False])
    @pytest.mark.parametrize("use_camera_embeddings", [True, False])
    def test_stores_configuration(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        embedding_dimension: int,
        has_time_dim: bool,
        use_camera_embeddings: bool,
    ):
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            has_time_dim=has_time_dim,
            use_camera_embeddings=use_camera_embeddings,
        )
        assert builder.embedding_dimension == embedding_dimension
        assert builder.has_time_dim is has_time_dim
        if use_camera_embeddings:
            assert builder.camera_embeddings is not None
        else:
            assert builder.camera_embeddings is None

    @pytest.mark.parametrize(
        "exclude_keys, expected",
        [
            (None, set()),
            (["key_a", "key_b"], {"key_a", "key_b"}),
        ],
    )
    def test_exclude_keys_converted_to_set(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        exclude_keys: list[str] | None,
        expected: set[str],
    ):
        builder = transformer_input_builder_factory(exclude_keys=exclude_keys)
        assert builder.exclude_keys == expected

    def test_spatial_pe_type_validation(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
    ):
        wrong_type_pe = sinusoidal_pe_1d_factory(embedding_dimension=64)
        with pytest.raises(
            ValueError,
            match="spatial_positional_encoding_layer must be PositionalEncoding2D",
        ):
            transformer_input_builder_factory(
                spatial_positional_encoding_layer=wrong_type_pe,
            )

    def test_spatial_pe_dimension_validation(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
    ):
        mismatched_pe = sinusoidal_pe_2d_factory(embedding_dimension=128)
        with pytest.raises(
            ValueError,
            match="spatial_positional_encoding_layer embedding dimension does not match",
        ):
            transformer_input_builder_factory(
                embedding_dimension=64,
                spatial_positional_encoding_layer=mismatched_pe,
            )

    def test_temporal_pe_type_validation(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
    ):
        wrong_type_pe = sinusoidal_pe_2d_factory(embedding_dimension=64)
        with pytest.raises(
            ValueError,
            match="temporal_positional_encoding_layer must be PositionalEncoding1D",
        ):
            transformer_input_builder_factory(
                temporal_positional_encoding_layer=wrong_type_pe,
            )

    def test_temporal_pe_dimension_validation(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
    ):
        mismatched_pe = sinusoidal_pe_1d_factory(embedding_dimension=128)
        with pytest.raises(
            ValueError,
            match="temporal_positional_encoding_layer embedding dimension does not match",
        ):
            transformer_input_builder_factory(
                embedding_dimension=64,
                temporal_positional_encoding_layer=mismatched_pe,
            )

    def test_flat_pe_type_validation(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
    ):
        wrong_type_pe = sinusoidal_pe_2d_factory(embedding_dimension=64)
        with pytest.raises(
            ValueError,
            match="flat_positional_encoding_layer must be PositionalEncoding1D",
        ):
            transformer_input_builder_factory(
                flat_positional_encoding_layer=wrong_type_pe,
            )

    def test_flat_pe_dimension_validation(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
    ):
        mismatched_pe = sinusoidal_pe_1d_factory(embedding_dimension=128)
        with pytest.raises(
            ValueError,
            match="flat_positional_encoding_layer embedding dimension does not match",
        ):
            transformer_input_builder_factory(
                embedding_dimension=64,
                flat_positional_encoding_layer=mismatched_pe,
            )

    def test_valid_pe_layers_stored(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sinusoidal_pe_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
    ):
        embedding_dimension = 64
        spatial_pe = sinusoidal_pe_2d_factory(embedding_dimension=embedding_dimension)
        temporal_pe = sinusoidal_pe_1d_factory(embedding_dimension=embedding_dimension)
        flat_pe = sinusoidal_pe_1d_factory(embedding_dimension=embedding_dimension)
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            spatial_positional_encoding_layer=spatial_pe,
            temporal_positional_encoding_layer=temporal_pe,
            flat_positional_encoding_layer=flat_pe,
        )
        assert builder.spatial_positional_encoding_layer is spatial_pe
        assert builder.temporal_positional_encoding_layer is temporal_pe
        assert builder.flat_positional_encoding_layer is flat_pe


class TestTransformerInputBuilderFeatureFiltering:
    def test_excludes_padding_mask_keys(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        input_tensor_factory: Callable[..., torch.Tensor],
    ):
        embedding_dimension = 64
        height, width = 4, 4
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        captured_keys: list[str] = []

        def capturing_projection(
            features: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            captured_keys.extend(features.keys())
            return features

        builder.projection.forward = capturing_projection
        features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        padding_mask_key = f"rgb_features_{EncoderOutputKeys.PADDING_MASK.value}"
        features[padding_mask_key] = input_tensor_factory(
            batch_size=2,
            input_dimension=height * width,
        )
        tokens, _, _ = builder(features)
        assert padding_mask_key not in captured_keys
        assert "rgb_features" in captured_keys
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (2, height * width, embedding_dimension)

    def test_excludes_is_pad_action_key(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        captured_keys: list[str] = []

        def capturing_projection(
            features: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            captured_keys.extend(features.keys())
            return features

        builder.projection.forward = capturing_projection
        features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=["flat_feature"],
        )
        features[SampleKey.IS_PAD_ACTION.value] = torch.zeros(2, 4, dtype=torch.bool)
        tokens, _, _ = builder(features)
        assert SampleKey.IS_PAD_ACTION.value not in captured_keys
        assert "flat_feature" in captured_keys
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (2, 1, embedding_dimension)

    def test_excludes_custom_keys(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        height, width = 4, 4
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            exclude_keys=["heavy_feature"],
            use_camera_embeddings=False,
        )
        captured_keys: list[str] = []

        def capturing_projection(
            features: dict[str, torch.Tensor],
        ) -> dict[str, torch.Tensor]:
            captured_keys.extend(features.keys())
            return features

        builder.projection.forward = capturing_projection
        features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height,
            width=width,
            feature_keys=["rgb_features"],
        )
        excluded_features = spatial_feature_factory(
            channels=embedding_dimension,
            height=8,
            width=8,
            feature_keys=["heavy_feature"],
        )
        features.update(excluded_features)
        tokens, _, _ = builder(features)
        assert "heavy_feature" not in captured_keys
        assert "rgb_features" in captured_keys
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (2, height * width, embedding_dimension)

    def test_no_features_after_filtering_raises(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
    ):
        builder = transformer_input_builder_factory(
            embedding_dimension=64,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = {
            SampleKey.IS_PAD_ACTION.value: torch.zeros(2, 4, dtype=torch.bool),
        }
        with pytest.raises(
            ValueError,
            match=re.escape(
                "TransformerInputBuilder received no features to build tokens from."
            ),
        ):
            builder(features)


class TestTransformerInputBuilderFeatureShapes:
    def test_2d_flat_feature_produces_single_token(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=["flat_feature"],
        )
        tokens, _, _ = builder(features)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (2, 1, embedding_dimension)

    @pytest.mark.parametrize("sequence_length", [4, 8])
    def test_3d_sequential_feature_without_time_dim(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sequential_feature_factory: Callable[..., dict[str, torch.Tensor]],
        sequence_length: int,
    ):
        embedding_dimension = 64
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            has_time_dim=False,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = sequential_feature_factory(
            sequence_length=sequence_length,
            feature_dimension=embedding_dimension,
        )
        tokens, _, _ = builder(features)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (2, sequence_length, embedding_dimension)

    @pytest.mark.parametrize("observation_horizon", [2, 4])
    def test_3d_temporal_feature_with_time_dim(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sequential_feature_factory: Callable[..., dict[str, torch.Tensor]],
        observation_horizon: int,
    ):
        embedding_dimension = 64
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            has_time_dim=True,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        # (B, T, Emb) — 3D with has_time_dim=True
        features = sequential_feature_factory(
            sequence_length=observation_horizon,
            feature_dimension=embedding_dimension,
            feature_keys=["temporal_feature"],
        )
        tokens, _, _ = builder(features)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (2, observation_horizon, embedding_dimension)

    @pytest.mark.parametrize("height, width", [(4, 4), (7, 7)])
    def test_4d_spatial_feature_without_time_dim(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        height: int,
        width: int,
    ):
        embedding_dimension = 64
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            has_time_dim=False,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        tokens, _, _ = builder(features)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (2, height * width, embedding_dimension)

    def test_4d_temporal_sequential_feature_with_time_dim(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        temporal_flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        observation_horizon = 2
        sequence_length = 4
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            has_time_dim=True,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = temporal_flat_feature_factory(
            observation_horizon=observation_horizon,
            sequence_length=sequence_length,
            feature_dimension=embedding_dimension,
        )
        tokens, _, _ = builder(features)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (
            2,
            observation_horizon * sequence_length,
            embedding_dimension,
        )

    @pytest.mark.parametrize(
        "observation_horizon, height, width", [(2, 4, 4), (3, 7, 7)]
    )
    def test_5d_temporal_spatial_feature(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        temporal_spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        observation_horizon: int,
        height: int,
        width: int,
    ):
        embedding_dimension = 64
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            has_time_dim=True,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = temporal_spatial_feature_factory(
            observation_horizon=observation_horizon,
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        tokens, _, _ = builder(features)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (
            2,
            observation_horizon * height * width,
            embedding_dimension,
        )

    def test_unsupported_feature_ndim_raises(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
    ):
        embedding_dimension = 64
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        feature_name = "bad_feature"
        bad_tensor = torch.zeros(2, 3, 4, 5, 6, 7)
        features = {feature_name: bad_tensor}
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Feature '{feature_name}' has unsupported shape {bad_tensor.shape}"
            ),
        ):
            builder(features)


class TestTransformerInputBuilderPaddingMask:
    def test_no_padding_returns_none_mask(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=["flat_feature"],
        )
        _, _, padding_mask = builder(features)
        # All-zeros mask is optimized to None
        assert padding_mask is None

    def test_1d_padding_mask_unsqueezed(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        batch_size = 2
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=embedding_dimension,
            feature_keys=["flat_feature"],
        )
        input_mask = torch.tensor([True, False])
        features["flat_feature_padding_mask"] = input_mask
        _, _, padding_mask = builder(features)
        assert isinstance(padding_mask, torch.Tensor)
        assert padding_mask.shape == (batch_size, 1)
        assert padding_mask.dtype == torch.bool
        assert torch.equal(padding_mask, input_mask.unsqueeze(1))

    def test_2d_padding_mask_kept(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sequential_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        batch_size = 2
        sequence_length = 4
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            has_time_dim=False,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = sequential_feature_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dimension=embedding_dimension,
        )
        input_mask = torch.tensor(
            [
                [True, False, False, True],
                [False, True, False, False],
            ]
        )
        features["seq_feature_padding_mask"] = input_mask
        _, _, padding_mask = builder(features)
        assert isinstance(padding_mask, torch.Tensor)
        assert padding_mask.shape == (batch_size, sequence_length)
        assert torch.equal(padding_mask, input_mask)

    def test_3d_padding_mask_reshaped(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        temporal_flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        batch_size = 2
        observation_horizon = 2
        sequence_length = 4
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            has_time_dim=True,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = temporal_flat_feature_factory(
            batch_size=batch_size,
            observation_horizon=observation_horizon,
            sequence_length=sequence_length,
            feature_dimension=embedding_dimension,
        )
        # 3D mask (B, T, Seq) should be reshaped to (B, T*Seq)
        input_mask = torch.ones(
            batch_size,
            observation_horizon,
            sequence_length,
            dtype=torch.bool,
        )
        features["temporal_seq_feature_padding_mask"] = input_mask
        _, _, padding_mask = builder(features)
        assert isinstance(padding_mask, torch.Tensor)
        assert padding_mask.shape == (batch_size, observation_horizon * sequence_length)
        assert torch.equal(padding_mask, input_mask.reshape(batch_size, -1))

    def test_unsupported_padding_mask_ndim_raises(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=["flat_feature"],
        )
        # 4D mask is unsupported
        feature_name = "flat_feature"
        mask_ndim = 4
        features[f"{feature_name}_padding_mask"] = torch.zeros(
            2, 3, 4, 5, dtype=torch.bool
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Padding masks not supported for spatial features, "
                f"got {mask_ndim} for {feature_name}"
            ),
        ):
            builder(features)

    def test_padding_mask_sequence_length_mismatch_raises(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        batch_size = 2
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        feature_name = "flat_feature"
        features = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=embedding_dimension,
            feature_keys=[feature_name],
        )
        input_mask = torch.zeros(batch_size, 2, dtype=torch.bool)
        features[f"{feature_name}_padding_mask"] = input_mask
        expected_shape = (batch_size, 1)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Padding mask for feature '{feature_name}' must have shape "
                f"{expected_shape} after flattening, got {input_mask.shape}."
            ),
        ):
            builder(features)

    def test_padding_mask_batch_size_mismatch_raises(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        batch_size = 2
        wrong_batch_size = 3
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        feature_name = "flat_feature"
        features = flat_feature_factory(
            batch_size=batch_size,
            feature_dim=embedding_dimension,
            feature_keys=[feature_name],
        )
        input_mask = torch.zeros(wrong_batch_size, dtype=torch.bool)
        features[f"{feature_name}_padding_mask"] = input_mask
        reshaped_shape = (wrong_batch_size, 1)
        expected_shape = (batch_size, 1)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Padding mask for feature '{feature_name}' must have shape "
                f"{expected_shape} after flattening, got torch.Size({list(reshaped_shape)})."
            ),
        ):
            builder(features)

    def test_padding_mask_converted_to_bool(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sequential_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        batch_size = 2
        sequence_length = 4
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            has_time_dim=False,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        feature_name = "seq_feature"
        features = sequential_feature_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dimension=embedding_dimension,
            feature_keys=[feature_name],
        )
        input_mask = torch.zeros(batch_size, sequence_length, dtype=torch.float32)
        input_mask[:, 1] = 1.0
        features[f"{feature_name}_padding_mask"] = input_mask
        _, _, padding_mask = builder(features)
        assert isinstance(padding_mask, torch.Tensor)
        assert padding_mask.dtype == torch.bool
        assert padding_mask[:, 1].all()
        assert not padding_mask[:, 0].any()

    def test_action_padding_mask_fallback(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sequential_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        batch_size = 2
        sequence_length = 4
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            has_time_dim=False,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        # Feature name contains "action" — triggers fallback to IS_PAD_ACTION mask
        features = sequential_feature_factory(
            batch_size=batch_size,
            sequence_length=sequence_length,
            feature_dimension=embedding_dimension,
            feature_keys=["action_embedding"],
        )
        action_pad_mask = torch.tensor(
            [
                [True, False, False, True],
                [False, True, False, False],
            ]
        )
        features[SampleKey.IS_PAD_ACTION.value] = action_pad_mask
        _, _, padding_mask = builder(features)
        assert isinstance(padding_mask, torch.Tensor)
        assert padding_mask.shape == (batch_size, sequence_length)
        assert torch.equal(padding_mask, action_pad_mask)


class TestTransformerInputBuilderPositionalEncodings:
    def test_no_pe_layers_returns_none(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=["flat_feature"],
        )
        _, positional_encodings, _ = builder(features)
        assert positional_encodings is None

    def test_flat_pe_only_covers_all_tokens(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        height, width = 4, 4
        flat_pe = sinusoidal_pe_1d_factory(embedding_dimension=embedding_dimension)
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            flat_positional_encoding_layer=flat_pe,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        flat_features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=["proprio"],
        )
        features.update(flat_features)
        total_sequence_length = height * width + 1
        _, positional_encodings, _ = builder(features)
        assert isinstance(positional_encodings, torch.Tensor)
        assert positional_encodings.shape == (
            2,
            total_sequence_length,
            embedding_dimension,
        )
        # Flat PE covers all tokens — should be non-zero
        assert not torch.all(positional_encodings == 0.0)

    def test_spatial_pe_with_spatial_features(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        height, width = 4, 4
        spatial_pe = sinusoidal_pe_2d_factory(embedding_dimension=embedding_dimension)
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            spatial_positional_encoding_layer=spatial_pe,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        _, positional_encodings, _ = builder(features)
        assert isinstance(positional_encodings, torch.Tensor)
        assert positional_encodings.shape == (2, height * width, embedding_dimension)
        # Sinusoidal PE should produce non-zero values
        assert not torch.all(positional_encodings == 0.0)

    def test_spatial_and_temporal_pe_combined(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sinusoidal_pe_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        temporal_spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        observation_horizon = 2
        height, width = 4, 4
        spatial_pe = sinusoidal_pe_2d_factory(embedding_dimension=embedding_dimension)
        temporal_pe = sinusoidal_pe_1d_factory(embedding_dimension=embedding_dimension)
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            has_time_dim=True,
            spatial_positional_encoding_layer=spatial_pe,
            temporal_positional_encoding_layer=temporal_pe,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = temporal_spatial_feature_factory(
            observation_horizon=observation_horizon,
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        _, positional_encodings, _ = builder(features)
        assert isinstance(positional_encodings, torch.Tensor)
        assert positional_encodings.shape == (
            2,
            observation_horizon * height * width,
            embedding_dimension,
        )
        # Temporal PE should make tokens at the same spatial position but different
        # time steps have different PE (frame 0 vs frame 1 at spatial position 0)
        tokens_per_frame = height * width
        frame_0_pe = positional_encodings[:, :tokens_per_frame, :]
        frame_1_pe = positional_encodings[:, tokens_per_frame : 2 * tokens_per_frame, :]
        assert not torch.equal(frame_0_pe, frame_1_pe)

    def test_spatial_pe_with_flat_tokens_zero_padded(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        height, width = 4, 4
        spatial_pe = sinusoidal_pe_2d_factory(embedding_dimension=embedding_dimension)
        # No flat PE — flat tokens get zero-padded PE
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            spatial_positional_encoding_layer=spatial_pe,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        flat_features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=["proprio"],
        )
        features.update(flat_features)
        total_sequence_length = height * width + 1
        _, positional_encodings, _ = builder(features)
        assert isinstance(positional_encodings, torch.Tensor)
        assert positional_encodings.shape == (
            2,
            total_sequence_length,
            embedding_dimension,
        )
        # Flat portion (last token) should be all zeros
        flat_pe_portion = positional_encodings[:, height * width :, :]
        assert torch.all(flat_pe_portion == 0.0)

    def test_spatial_pe_with_flat_pe_for_flat_tokens(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_1d_factory: Callable[..., SinusoidalPositionalEncoding1D],
        sinusoidal_pe_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        height, width = 4, 4
        spatial_pe = sinusoidal_pe_2d_factory(embedding_dimension=embedding_dimension)
        flat_pe = sinusoidal_pe_1d_factory(embedding_dimension=embedding_dimension)
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            spatial_positional_encoding_layer=spatial_pe,
            flat_positional_encoding_layer=flat_pe,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        flat_features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=["proprio"],
        )
        features.update(flat_features)
        total_sequence_length = height * width + 1
        _, positional_encodings, _ = builder(features)
        assert isinstance(positional_encodings, torch.Tensor)
        assert positional_encodings.shape == (
            2,
            total_sequence_length,
            embedding_dimension,
        )
        # Flat portion should NOT be all zeros (flat PE was applied)
        flat_pe_portion = positional_encodings[:, height * width :, :]
        assert not torch.all(flat_pe_portion == 0.0)


class TestTransformerInputBuilderCLSToken:
    def test_cls_token_appended_at_end(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=["aaa_feature"],
        )
        cls_features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=[DecoderOutputKey.CLASS_TOKEN.value],
        )
        features.update(cls_features)
        cls_value = cls_features[DecoderOutputKey.CLASS_TOKEN.value]
        tokens, _, _ = builder(features)
        assert isinstance(tokens, torch.Tensor)
        # 2 flat features: aaa_feature (1 token) + cls_token (1 token appended at end)
        assert tokens.shape == (2, 2, embedding_dimension)
        # CLS token is always last
        assert torch.equal(tokens[:, -1, :], cls_value)

    def test_cls_token_with_spatial_features(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        height, width = 4, 4
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        cls_features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=[DecoderOutputKey.CLASS_TOKEN.value],
        )
        features.update(cls_features)
        cls_value = cls_features[DecoderOutputKey.CLASS_TOKEN.value]
        tokens, _, _ = builder(features)
        assert isinstance(tokens, torch.Tensor)
        # Spatial tokens (H*W) + CLS token (1) appended at end
        assert tokens.shape == (2, height * width + 1, embedding_dimension)
        assert torch.equal(tokens[:, -1, :], cls_value)


class TestTransformerInputBuilderCameraEmbeddings:
    def test_camera_embeddings_disabled_pe_identical(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        height, width = 4, 4
        spatial_pe = sinusoidal_pe_2d_factory(embedding_dimension=embedding_dimension)
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            spatial_positional_encoding_layer=spatial_pe,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        assert builder.camera_embeddings is None
        features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height,
            width=width,
            feature_keys=["left_camera", "right_camera"],
        )
        _, positional_encodings, _ = builder(features)
        assert isinstance(positional_encodings, torch.Tensor)
        # Without camera embeddings, both cameras get identical spatial PE
        left_pe = positional_encodings[:, : height * width, :]
        right_pe = positional_encodings[:, height * width :, :]
        assert torch.equal(left_pe, right_pe)

    def test_camera_embeddings_added_to_spatial_pe(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        sinusoidal_pe_2d_factory: Callable[..., SinusoidalPositionalEncoding2D],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        height, width = 4, 4
        spatial_pe = sinusoidal_pe_2d_factory(embedding_dimension=embedding_dimension)
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            spatial_positional_encoding_layer=spatial_pe,
            use_camera_embeddings=True,
        )
        builder.projection.forward = lambda features: features
        features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height,
            width=width,
            feature_keys=["left_camera", "right_camera"],
        )
        _, positional_encodings, _ = builder(features)
        assert isinstance(positional_encodings, torch.Tensor)
        total_tokens = 2 * height * width
        assert positional_encodings.shape == (2, total_tokens, embedding_dimension)
        # Camera embeddings make PE different for each camera's tokens
        left_pe = positional_encodings[:, : height * width, :]
        right_pe = positional_encodings[:, height * width :, :]
        assert not torch.equal(left_pe, right_pe)


class TestTransformerInputBuilderMultipleFeatures:
    def test_multiple_spatial_features_concatenated(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        height_1, width_1 = 4, 4
        height_2, width_2 = 3, 3
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        depth_features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height_1,
            width=width_1,
            feature_keys=["depth"],
        )
        rgb_features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height_2,
            width=width_2,
            feature_keys=["rgb"],
        )
        features = {**depth_features, **rgb_features}
        tokens, _, _ = builder(features)
        assert isinstance(tokens, torch.Tensor)
        expected_sequence_length = height_1 * width_1 + height_2 * width_2
        assert tokens.shape == (2, expected_sequence_length, embedding_dimension)
        # Sorted order: "depth" before "rgb" — verify by value
        expected_depth = depth_features["depth"].flatten(2).transpose(1, 2)
        expected_rgb = rgb_features["rgb"].flatten(2).transpose(1, 2)
        assert torch.equal(tokens[:, : height_1 * width_1, :], expected_depth)
        assert torch.equal(tokens[:, height_1 * width_1 :, :], expected_rgb)

    def test_mixed_spatial_and_flat_features(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        height, width = 4, 4
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        features = spatial_feature_factory(
            channels=embedding_dimension,
            height=height,
            width=width,
        )
        flat_features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=["proprio"],
        )
        features.update(flat_features)
        spatial_value = features["rgb_features"]
        flat_value = flat_features["proprio"]
        tokens, _, _ = builder(features)
        assert isinstance(tokens, torch.Tensor)
        # Spatial first (H*W), then flat (1)
        assert tokens.shape == (2, height * width + 1, embedding_dimension)
        # Verify spatial tokens come first by checking the flat token at the end
        expected_spatial = spatial_value.flatten(2).transpose(1, 2)
        assert torch.equal(tokens[:, : height * width, :], expected_spatial)
        assert torch.equal(tokens[:, -1, :], flat_value)

    def test_features_processed_in_sorted_order(
        self,
        transformer_input_builder_factory: Callable[..., TransformerInputBuilder],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dimension = 64
        builder = transformer_input_builder_factory(
            embedding_dimension=embedding_dimension,
            use_camera_embeddings=False,
        )
        builder.projection.forward = lambda features: features
        a_features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=["a_feature"],
        )
        z_features = flat_feature_factory(
            feature_dim=embedding_dimension,
            feature_keys=["z_feature"],
        )
        features = {}
        features.update(z_features)
        features.update(a_features)
        a_value = a_features["a_feature"]
        z_value = z_features["z_feature"]
        tokens, _, _ = builder(features)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.shape == (2, 2, embedding_dimension)
        # Sorted order: a_feature first, z_feature second
        assert torch.equal(tokens[:, 0, :], a_value)
        assert torch.equal(tokens[:, 1, :], z_value)
