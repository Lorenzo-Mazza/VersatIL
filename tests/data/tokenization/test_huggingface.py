"""Tests for versatil.data.tokenization.huggingface module."""

from unittest.mock import MagicMock, patch

import pytest
from tokenizers import processors

from versatil.data.tokenization import huggingface as huggingface_tokenization


def test_load_huggingface_tokenizer_patches_roberta_processing_cls_alias(
    monkeypatch: pytest.MonkeyPatch,
):
    original_roberta_processing = huggingface_tokenization._ORIGINAL_ROBERTA_PROCESSING
    monkeypatch.setattr(
        processors,
        "RobertaProcessing",
        original_roberta_processing,
    )
    tokenizer = MagicMock()

    with patch(
        "versatil.data.tokenization.huggingface.AutoTokenizer.from_pretrained",
        return_value=tokenizer,
    ) as mock_from_pretrained:
        result = huggingface_tokenization.load_huggingface_tokenizer(
            tokenizer_model="roberta-base"
        )

    post_processor = processors.RobertaProcessing(
        sep=("</s>", 2),
        cls=("<s>", 0),
    )

    assert result is tokenizer
    assert isinstance(post_processor, processors.PostProcessor)
    mock_from_pretrained.assert_called_once_with("roberta-base")
