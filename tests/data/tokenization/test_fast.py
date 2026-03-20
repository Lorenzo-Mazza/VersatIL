"""Tests for versatil.data.tokenization.fast module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from versatil.data.tokenization.fast import (
    _load_bpe_tokenizer,
    _load_processor_config,
    _resolve_processor_class,
    load_fast_processor,
)


@pytest.fixture
def processor_config_factory(tmp_path):
    """Factory for processor_config.json files with configurable values."""

    def factory(
        scale: int = 10,
        vocab_size: int = 2048,
        min_token: int = -354,
        action_dim: int | None = None,
        time_horizon: int | None = None,
        include_auto_map: bool = True,
    ) -> Path:
        config = {
            "scale": scale,
            "vocab_size": vocab_size,
            "min_token": min_token,
            "action_dim": action_dim,
            "time_horizon": time_horizon,
            "processor_class": "UniversalActionProcessor",
        }
        if include_auto_map:
            config["auto_map"] = {
                "AutoProcessor": "processing_action_tokenizer.UniversalActionProcessor"
            }
        config_path = tmp_path / "processor_config.json"
        config_path.write_text(json.dumps(config))
        return tmp_path

    return factory


class TestLoadProcessorConfig:
    def test_loads_from_local_path(self, processor_config_factory):
        local_path = processor_config_factory(scale=42, vocab_size=1024)
        config = _load_processor_config(
            model_path=str(local_path),
            local_path=local_path,
            is_local=True,
        )
        assert config["scale"] == 42
        assert config["vocab_size"] == 1024

    def test_loads_from_hub_via_hf_hub_download(self, tmp_path):
        config_content = json.dumps({"scale": 7, "vocab_size": 512})
        config_file = tmp_path / "processor_config.json"
        config_file.write_text(config_content)

        with patch(
            "versatil.data.tokenization.fast.hf_hub_download",
            return_value=str(config_file),
        ) as mock_download:
            config = _load_processor_config(
                model_path="some-org/some-model",
                local_path=Path("nonexistent"),
                is_local=False,
            )

        assert config["scale"] == 7
        assert config["vocab_size"] == 512
        mock_download.assert_called_once_with(
            repo_id="some-org/some-model",
            filename="processor_config.json",
        )


class TestLoadBpeTokenizer:
    def test_local_loads_from_bpe_tokenizer_subfolder(self, tmp_path):
        with patch(
            "versatil.data.tokenization.fast.PreTrainedTokenizerFast"
        ) as mock_tokenizer_cls:
            mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

            _load_bpe_tokenizer(
                model_path=str(tmp_path),
                local_path=tmp_path,
                is_local=True,
            )

        mock_tokenizer_cls.from_pretrained.assert_called_once_with(
            str(tmp_path / "bpe_tokenizer")
        )

    def test_hub_loads_from_model_root(self):
        with patch(
            "versatil.data.tokenization.fast.PreTrainedTokenizerFast"
        ) as mock_tokenizer_cls:
            mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

            _load_bpe_tokenizer(
                model_path="some-org/some-model",
                local_path=Path("nonexistent"),
                is_local=False,
            )

        mock_tokenizer_cls.from_pretrained.assert_called_once_with(
            "some-org/some-model"
        )


class TestResolveProcessorClass:
    def test_resolves_class_from_auto_map(self):
        config = {
            "auto_map": {
                "AutoProcessor": "processing_action_tokenizer.UniversalActionProcessor"
            }
        }
        mock_class = MagicMock()
        with patch(
            "versatil.data.tokenization.fast.get_class_from_dynamic_module",
            return_value=mock_class,
        ) as mock_resolve:
            result = _resolve_processor_class(
                config=config,
                model_path="some-org/some-model",
            )

        assert result is mock_class
        mock_resolve.assert_called_once_with(
            "processing_action_tokenizer.UniversalActionProcessor",
            "some-org/some-model",
            trust_remote_code=True,
        )

    def test_falls_back_to_default_hub_model_without_auto_map(self):
        config = {"scale": 10}
        mock_class = MagicMock()
        with patch(
            "versatil.data.tokenization.fast.get_class_from_dynamic_module",
            return_value=mock_class,
        ) as mock_resolve:
            result = _resolve_processor_class(
                config=config,
                model_path="/some/local/path",
            )

        assert result is mock_class
        mock_resolve.assert_called_once_with(
            "processing_action_tokenizer.UniversalActionProcessor",
            "physical-intelligence/fast",
            trust_remote_code=True,
        )


class TestLoadFastProcessor:
    @pytest.mark.parametrize(
        "scale, vocab_size, min_token",
        [
            (10, 2048, -354),
            (5, 1024, 0),
        ],
    )
    def test_forwards_config_values_to_processor_constructor(
        self, tmp_path, scale, vocab_size, min_token
    ):
        config = {
            "scale": scale,
            "vocab_size": vocab_size,
            "min_token": min_token,
            "action_dim": 7,
            "time_horizon": 10,
            "auto_map": {
                "AutoProcessor": "processing_action_tokenizer.UniversalActionProcessor"
            },
        }
        (tmp_path / "processor_config.json").write_text(json.dumps(config))
        (tmp_path / "bpe_tokenizer").mkdir()

        mock_processor_class = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch(
                "versatil.data.tokenization.fast.PreTrainedTokenizerFast"
            ) as mock_tok_cls,
            patch(
                "versatil.data.tokenization.fast.get_class_from_dynamic_module",
                return_value=mock_processor_class,
            ),
        ):
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer
            load_fast_processor(str(tmp_path))

        mock_processor_class.assert_called_once_with(
            bpe_tokenizer=mock_tokenizer,
            scale=scale,
            vocab_size=vocab_size,
            min_token=min_token,
            action_dim=7,
            time_horizon=10,
        )


@pytest.mark.integration
class TestLoadFastProcessorIntegration:
    def test_loads_from_hub_and_encodes(self, rng):
        processor = load_fast_processor("physical-intelligence/fast")
        actions = rng.standard_normal((1, 10, 4)).astype(np.float32) * 0.5
        tokens = processor(actions)
        assert len(tokens) == 1
        assert len(tokens[0]) > 0

    def test_hub_processor_decode_roundtrip(self, rng):
        processor = load_fast_processor("physical-intelligence/fast")
        time_horizon = 8
        action_dimension = 5
        actions = (
            rng.standard_normal((1, time_horizon, action_dimension)).astype(np.float32)
            * 0.3
        )
        tokens = processor(actions)
        decoded = processor.decode(
            tokens, time_horizon=time_horizon, action_dim=action_dimension
        )
        assert decoded.shape == (1, time_horizon, action_dimension)

    def test_save_and_load_local_roundtrip(self, rng, tmp_path):
        processor = load_fast_processor("physical-intelligence/fast")
        save_path = tmp_path / "saved_processor"
        processor.save_pretrained(str(save_path))

        loaded = load_fast_processor(str(save_path))
        actions = rng.standard_normal((1, 6, 3)).astype(np.float32) * 0.4
        tokens = loaded(actions)
        decoded = loaded.decode(tokens, time_horizon=6, action_dim=3)
        assert decoded.shape == (1, 6, 3)
