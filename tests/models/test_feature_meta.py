"""Tests for versatil.models.feature_meta module."""

from contextlib import nullcontext as does_not_raise

import pytest

from versatil.models.feature_meta import (
    FeatureMetadata,
    FeatureType,
    infer_feature_type,
)


class TestInferFeatureType:
    @pytest.mark.parametrize(
        "dimension, expected_type, expectation",
        [
            ((256,), FeatureType.FLAT.value, does_not_raise()),
            ((128, 256), FeatureType.SEQUENTIAL.value, does_not_raise()),
            ((512, 7, 7), FeatureType.SPATIAL.value, does_not_raise()),
            (
                (1, 2, 3, 4),
                None,
                pytest.raises(
                    ValueError,
                    match="Cannot infer feature type from dimension: \\(1, 2, 3, 4\\)",
                ),
            ),
            (
                (),
                None,
                pytest.raises(
                    ValueError,
                    match="Cannot infer feature type from dimension: \\(\\)",
                ),
            ),
        ],
    )
    def test_infers_type_from_tuple_length(
        self,
        dimension: tuple[int, ...],
        expected_type: str | None,
        expectation,
    ):
        with expectation:
            result = infer_feature_type(dimension)
            assert result == expected_type


class TestFeatureMetadata:
    def test_is_frozen(self):
        meta = FeatureMetadata(
            key="rgb",
            feature_type=FeatureType.FLAT.value,
            dimension=(256,),
        )
        with pytest.raises(AttributeError):
            meta.key = "changed"

    @pytest.mark.parametrize("key", ["rgb", "language"])
    @pytest.mark.parametrize(
        "feature_type", [FeatureType.FLAT.value, FeatureType.SPATIAL.value]
    )
    @pytest.mark.parametrize("dimension", [(64,), (3, 14, 14)])
    def test_stores_fields(
        self,
        key: str,
        feature_type: str,
        dimension: tuple[int, ...],
    ):
        meta = FeatureMetadata(
            key=key,
            feature_type=feature_type,
            dimension=dimension,
        )
        assert meta.key == key
        assert meta.feature_type == feature_type
        assert meta.dimension == dimension
