"""Tests for versatil.models.layers.feature_projection module."""
import re
from collections.abc import Callable

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.models.layers.feature_projection import FeatureProjection


@pytest.fixture
def feature_projection_factory() -> Callable[..., FeatureProjection]:
    """Factory for FeatureProjection instances with configurable fields."""
    def factory(
        embedding_dim: int = 64,
        has_time_dim: bool = False,
    ) -> FeatureProjection:
        return FeatureProjection(
            embedding_dim=embedding_dim,
            has_time_dim=has_time_dim,
        )
    return factory


@pytest.fixture
def flat_feature_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for flat feature dictionaries with shape (B, C)."""
    def factory(
        keys: list[str] | None = None,
        batch_size: int = 2,
        channel_dim: int = 32,
    ) -> dict[str, torch.Tensor]:
        if keys is None:
            keys = ["flat_feature"]
        return {
            key: torch.from_numpy(
                rng.standard_normal((batch_size, channel_dim)).astype(np.float32)
            )
            for key in keys
        }
    return factory


@pytest.fixture
def sequential_feature_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for sequential feature dictionaries with shape (B, T, C)."""
    def factory(
        keys: list[str] | None = None,
        batch_size: int = 2,
        temporal_length: int = 4,
        channel_dim: int = 32,
    ) -> dict[str, torch.Tensor]:
        if keys is None:
            keys = ["sequential_feature"]
        return {
            key: torch.from_numpy(
                rng.standard_normal((batch_size, temporal_length, channel_dim)).astype(np.float32)
            )
            for key in keys
        }
    return factory


@pytest.fixture
def spatial_feature_factory(
    rng: np.random.Generator,
) -> Callable[..., dict[str, torch.Tensor]]:
    """Factory for spatial feature dictionaries with shape (B, C, H, W)."""
    def factory(
        keys: list[str] | None = None,
        batch_size: int = 2,
        channel_dim: int = 32,
        height: int = 7,
        width: int = 7,
    ) -> dict[str, torch.Tensor]:
        if keys is None:
            keys = ["spatial_feature"]
        return {
            key: torch.from_numpy(
                rng.standard_normal((batch_size, channel_dim, height, width)).astype(np.float32)
            )
            for key in keys
        }
    return factory


class TestFeatureProjectionInitialization:

    @pytest.mark.parametrize("embedding_dim", [64, 128])
    @pytest.mark.parametrize("has_time_dim", [True, False])
    def test_stores_configuration(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        embedding_dim: int,
        has_time_dim: bool,
    ):
        projection = feature_projection_factory(
            embedding_dim=embedding_dim,
            has_time_dim=has_time_dim,
        )
        assert projection.embedding_dim == embedding_dim
        assert projection.has_time_dim is has_time_dim

    def test_starts_with_empty_linear_projections(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
    ):
        projection = feature_projection_factory(embedding_dim=64)
        assert len(projection.linear_projections) == 0

    def test_starts_with_empty_spatial_projections(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
    ):
        projection = feature_projection_factory(embedding_dim=64)
        assert len(projection.spatial_projections) == 0


class TestFeatureProjectionCreateProjectionLayer:

    @pytest.mark.parametrize("channel_dim, expected_type", [
        (64, nn.Identity),
        (32, nn.Linear),
        (128, nn.Linear),
    ])
    def test_flat_feature_creates_correct_layer_type(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
        channel_dim: int,
        expected_type: type,
    ):
        embedding_dim = 64
        projection = feature_projection_factory(embedding_dim=embedding_dim)
        features = flat_feature_factory(channel_dim=channel_dim)
        feature_tensor = features["flat_feature"]
        layer = projection._create_projection_layer(feature=feature_tensor)
        assert isinstance(layer, expected_type)

    @pytest.mark.parametrize("channel_dim, expected_type", [
        (64, nn.Identity),
        (32, nn.Conv2d),
        (128, nn.Conv2d),
    ])
    def test_spatial_feature_creates_correct_layer_type(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
        channel_dim: int,
        expected_type: type,
    ):
        embedding_dim = 64
        projection = feature_projection_factory(embedding_dim=embedding_dim)
        features = spatial_feature_factory(channel_dim=channel_dim)
        feature_tensor = features["spatial_feature"]
        layer = projection._create_projection_layer(feature=feature_tensor)
        assert isinstance(layer, expected_type)

    def test_sequential_feature_creates_linear_layer(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        sequential_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        projection = feature_projection_factory(embedding_dim=64)
        features = sequential_feature_factory(channel_dim=32)
        feature_tensor = features["sequential_feature"]
        layer = projection._create_projection_layer(feature=feature_tensor)
        assert isinstance(layer, nn.Linear)

    def test_sequential_feature_identity_when_dim_matches(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        sequential_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dim = 32
        projection = feature_projection_factory(embedding_dim=embedding_dim)
        features = sequential_feature_factory(channel_dim=embedding_dim)
        feature_tensor = features["sequential_feature"]
        layer = projection._create_projection_layer(feature=feature_tensor)
        assert isinstance(layer, nn.Identity)


class TestFeatureProjectionForward:

    def test_flat_features_projected_to_embedding_dim(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        embedding_dim = 64
        projection = feature_projection_factory(embedding_dim=embedding_dim)
        features = flat_feature_factory(batch_size=batch_size, channel_dim=32)
        output = projection(features)
        assert output["flat_feature"].shape == (batch_size, embedding_dim)

    def test_sequential_features_projected_to_embedding_dim(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        sequential_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        temporal_length = 4
        embedding_dim = 64
        projection = feature_projection_factory(embedding_dim=embedding_dim)
        features = sequential_feature_factory(
            batch_size=batch_size,
            temporal_length=temporal_length,
            channel_dim=32,
        )
        output = projection(features)
        assert output["sequential_feature"].shape == (batch_size, temporal_length, embedding_dim)

    def test_spatial_features_projected_to_embedding_dim(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        height = 7
        width = 7
        embedding_dim = 64
        projection = feature_projection_factory(embedding_dim=embedding_dim)
        features = spatial_feature_factory(
            batch_size=batch_size,
            channel_dim=32,
            height=height,
            width=width,
        )
        output = projection(features)
        assert output["spatial_feature"].shape == (batch_size, embedding_dim, height, width)

    def test_multiple_features_projected_independently(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        embedding_dim = 64
        projection = feature_projection_factory(embedding_dim=embedding_dim)
        features = flat_feature_factory(
            keys=["feature_a", "feature_b"],
            batch_size=batch_size,
            channel_dim=32,
        )
        output = projection(features)
        assert "feature_a" in output
        assert "feature_b" in output
        assert output["feature_a"].shape == (batch_size, embedding_dim)
        assert output["feature_b"].shape == (batch_size, embedding_dim)

    def test_identity_when_dim_matches_embedding_dim(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        embedding_dim = 64
        projection = feature_projection_factory(embedding_dim=embedding_dim)
        features = flat_feature_factory(channel_dim=embedding_dim)
        output = projection(features)
        # Identity projection should preserve the values exactly
        torch.testing.assert_close(output["flat_feature"], features["flat_feature"])

    def test_lazy_creates_projections_on_first_call(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        projection = feature_projection_factory(embedding_dim=64)
        assert len(projection.linear_projections) == 0
        features = flat_feature_factory(channel_dim=32)
        projection(features)
        assert len(projection.linear_projections) == 1
        assert "flat_feature" in projection.linear_projections

    def test_lazy_creates_spatial_projections_on_first_call(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        spatial_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        projection = feature_projection_factory(embedding_dim=64)
        assert len(projection.spatial_projections) == 0
        features = spatial_feature_factory(channel_dim=32)
        projection(features)
        assert len(projection.spatial_projections) == 1
        assert "spatial_feature" in projection.spatial_projections

    def test_reuses_projections_on_subsequent_calls(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        projection = feature_projection_factory(embedding_dim=64)
        features = flat_feature_factory(channel_dim=32)
        projection(features)
        layer_after_first_call = projection.linear_projections["flat_feature"]
        projection(features)
        layer_after_second_call = projection.linear_projections["flat_feature"]
        assert layer_after_first_call is layer_after_second_call


class TestFeatureProjectionProjectAndConcatenate:

    def test_concatenates_along_specified_dimension(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        embedding_dim = 64
        projection = feature_projection_factory(embedding_dim=embedding_dim)
        features = flat_feature_factory(
            keys=["feature_a", "feature_b"],
            batch_size=batch_size,
            channel_dim=32,
        )
        result = projection.project_and_concatenate(
            features=features,
            concatenation_dimension=-1,
        )
        assert result.shape == (batch_size, embedding_dim * 2)

    def test_raises_value_error_on_empty_dict(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
    ):
        projection = feature_projection_factory(embedding_dim=64)
        with pytest.raises(
            ValueError,
            match=re.escape("No features to concatenate"),
        ):
            projection.project_and_concatenate(features={})

    def test_single_feature_returns_projected_tensor(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        embedding_dim = 64
        projection = feature_projection_factory(embedding_dim=embedding_dim)
        features = flat_feature_factory(batch_size=batch_size, channel_dim=32)
        result = projection.project_and_concatenate(features=features)
        assert result.shape == (batch_size, embedding_dim)

    def test_concatenation_along_batch_dimension(
        self,
        feature_projection_factory: Callable[..., FeatureProjection],
        flat_feature_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        batch_size = 2
        embedding_dim = 64
        projection = feature_projection_factory(embedding_dim=embedding_dim)
        features = flat_feature_factory(
            keys=["feature_a", "feature_b"],
            batch_size=batch_size,
            channel_dim=32,
        )
        result = projection.project_and_concatenate(
            features=features,
            concatenation_dimension=0,
        )
        assert result.shape == (batch_size * 2, embedding_dim)