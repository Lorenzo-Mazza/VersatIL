"""Tests for versatil.common.set_cache_dir module."""

import os
from unittest.mock import patch

import pytest

from versatil.common.set_cache_dir import setup_cache_directories


@pytest.fixture
def cache_directory(tmp_path):
    """Temporary cache directory for testing."""
    return tmp_path / "test_cache"


@pytest.mark.unit
class TestSetupCacheDirectories:
    def test_sets_hf_home_environment_variable(self, cache_directory):
        with patch.dict(os.environ, {}, clear=False):
            setup_cache_directories(cache_dir=cache_directory)
            expected = str(cache_directory / "huggingface")
            assert os.environ["HF_HOME"] == expected

    def test_sets_hf_hub_cache_environment_variable(self, cache_directory):
        with patch.dict(os.environ, {}, clear=False):
            setup_cache_directories(cache_dir=cache_directory)
            expected = str(cache_directory / "huggingface" / "hub")
            assert os.environ["HF_HUB_CACHE"] == expected

    def test_sets_torch_home_environment_variable(self, cache_directory):
        with patch.dict(os.environ, {}, clear=False):
            setup_cache_directories(cache_dir=cache_directory)
            expected = str(cache_directory / "torch")
            assert os.environ["TORCH_HOME"] == expected

    def test_creates_huggingface_transformers_directory(self, cache_directory):
        setup_cache_directories(cache_dir=cache_directory)
        assert (cache_directory / "huggingface" / "transformers").is_dir()

    def test_creates_huggingface_hub_directory(self, cache_directory):
        setup_cache_directories(cache_dir=cache_directory)
        assert (cache_directory / "huggingface" / "hub").is_dir()

    def test_creates_torch_hub_directory(self, cache_directory):
        setup_cache_directories(cache_dir=cache_directory)
        assert (cache_directory / "torch" / "hub").is_dir()

    def test_accepts_string_path(self, cache_directory):
        setup_cache_directories(cache_dir=str(cache_directory))
        assert (cache_directory / "huggingface" / "hub").is_dir()

    def test_idempotent_on_existing_directories(self, cache_directory):
        setup_cache_directories(cache_dir=cache_directory)
        setup_cache_directories(cache_dir=cache_directory)
        assert (cache_directory / "torch" / "hub").is_dir()
