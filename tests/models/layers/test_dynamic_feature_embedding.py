"""Tests for versatil.models.layers.dynamic_feature_embedding module."""
from collections.abc import Callable

import pytest
import torch

from versatil.models.layers.dynamic_feature_embedding import DynamicFeatureEmbedding


@pytest.fixture
def dynamic_feature_embedding_factory() -> Callable[..., DynamicFeatureEmbedding]:
    """Factory for DynamicFeatureEmbedding instances."""
    def factory(
        embedding_dim: int = 64,
    ) -> DynamicFeatureEmbedding:
        return DynamicFeatureEmbedding(embedding_dim=embedding_dim)
    return factory


class TestDynamicFeatureEmbeddingInitialization:

    @pytest.mark.parametrize("embedding_dim", [32, 128])
    def test_stores_embedding_dim(
        self,
        dynamic_feature_embedding_factory: Callable[..., DynamicFeatureEmbedding],
        embedding_dim: int,
    ):
        module = dynamic_feature_embedding_factory(embedding_dim=embedding_dim)
        assert module.embedding_dim == embedding_dim

    def test_starts_with_empty_embeddings(
        self,
        dynamic_feature_embedding_factory: Callable[..., DynamicFeatureEmbedding],
    ):
        module = dynamic_feature_embedding_factory()
        assert len(module.embeddings) == 0


class TestDynamicFeatureEmbeddingForward:

    def test_creates_embedding_on_first_access(
        self,
        dynamic_feature_embedding_factory: Callable[..., DynamicFeatureEmbedding],
    ):
        module = dynamic_feature_embedding_factory()
        assert "rgb_features" not in module.embeddings
        module(name="rgb_features", device=torch.device("cpu"))
        assert "rgb_features" in module.embeddings

    def test_returns_same_embedding_on_second_access(
        self,
        dynamic_feature_embedding_factory: Callable[..., DynamicFeatureEmbedding],
    ):
        module = dynamic_feature_embedding_factory()
        first = module(name="rgb_features", device=torch.device("cpu"))
        second = module(name="rgb_features", device=torch.device("cpu"))
        assert torch.equal(first, second)
        assert len(module.embeddings) == 1

    @pytest.mark.parametrize("embedding_dim", [32, 128])
    def test_output_shape(
        self,
        dynamic_feature_embedding_factory: Callable[..., DynamicFeatureEmbedding],
        embedding_dim: int,
    ):
        module = dynamic_feature_embedding_factory(embedding_dim=embedding_dim)
        output = module(name="test_feature", device=torch.device("cpu"))
        assert output.shape == (1, 1, embedding_dim)

    def test_replaces_dots_in_keys(
        self,
        dynamic_feature_embedding_factory: Callable[..., DynamicFeatureEmbedding],
    ):
        module = dynamic_feature_embedding_factory()
        module(name="encoder.layer.output", device=torch.device("cpu"))
        assert "encoder_layer_output" in module.embeddings
        assert "encoder.layer.output" not in module.embeddings

    def test_creates_separate_embeddings_for_different_names(
        self,
        dynamic_feature_embedding_factory: Callable[..., DynamicFeatureEmbedding],
    ):
        module = dynamic_feature_embedding_factory()
        first = module(name="feature_a", device=torch.device("cpu"))
        second = module(name="feature_b", device=torch.device("cpu"))
        assert len(module.embeddings) == 2
        assert not torch.equal(first, second)

    def test_embedding_is_trainable(
        self,
        dynamic_feature_embedding_factory: Callable[..., DynamicFeatureEmbedding],
    ):
        module = dynamic_feature_embedding_factory()
        module(name="test", device=torch.device("cpu"))
        # Functional consequence: embedding appears in model.parameters() and is trainable
        named_params = dict(module.named_parameters())
        assert "embeddings.test" in named_params
        assert named_params["embeddings.test"].requires_grad is True


class TestDynamicFeatureEmbeddingLoadFromStateDict:

    def test_load_state_dict_creates_embeddings_from_checkpoint(
        self,
        dynamic_feature_embedding_factory: Callable[..., DynamicFeatureEmbedding],
    ):
        embedding_dim = 64
        # Create source module with embeddings
        source = dynamic_feature_embedding_factory(embedding_dim=embedding_dim)
        source(name="rgb_features", device=torch.device("cpu"))
        source(name="depth_features", device=torch.device("cpu"))
        state_dict = source.state_dict()

        # Load into fresh module that has no embeddings
        target = dynamic_feature_embedding_factory(embedding_dim=embedding_dim)
        assert len(target.embeddings) == 0
        target.load_state_dict(state_dict)
        assert "rgb_features" in target.embeddings
        assert "depth_features" in target.embeddings

    def test_loaded_embeddings_match_source_values(
        self,
        dynamic_feature_embedding_factory: Callable[..., DynamicFeatureEmbedding],
    ):
        embedding_dim = 32
        source = dynamic_feature_embedding_factory(embedding_dim=embedding_dim)
        source(name="test_feat", device=torch.device("cpu"))
        state_dict = source.state_dict()

        target = dynamic_feature_embedding_factory(embedding_dim=embedding_dim)
        target.load_state_dict(state_dict)
        torch.testing.assert_close(
            target.embeddings["test_feat"],
            source.embeddings["test_feat"],
        )