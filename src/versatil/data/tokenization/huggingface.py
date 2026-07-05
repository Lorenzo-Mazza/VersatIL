"""Compatibility helpers for HuggingFace tokenizer loading."""

import inspect
from pathlib import Path

from tokenizers import processors
from transformers import AutoTokenizer, PreTrainedTokenizerBase

_ORIGINAL_ROBERTA_PROCESSING = processors.RobertaProcessing


def _roberta_processing_with_cls_alias(
    sep: tuple[str, int],
    cls: tuple[str, int] | None = None,
    cls_token: tuple[str, int] | None = None,
    trim_offsets: bool = True,
    add_prefix_space: bool = True,
) -> processors.PostProcessor:
    """Build RobertaProcessing across tokenizers keyword-name variants."""
    resolved_cls_token = cls_token if cls_token is not None else cls
    if resolved_cls_token is None:
        raise TypeError("RobertaProcessing requires cls or cls_token.")
    return _ORIGINAL_ROBERTA_PROCESSING(
        sep=sep,
        cls_token=resolved_cls_token,
        trim_offsets=trim_offsets,
        add_prefix_space=add_prefix_space,
    )


def _patch_roberta_processing_cls_alias() -> None:
    signature = inspect.signature(processors.RobertaProcessing)
    if "cls" in signature.parameters:
        return
    processors.RobertaProcessing = _roberta_processing_with_cls_alias


def load_huggingface_tokenizer(
    tokenizer_model: str | Path,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizerBase:
    """Load a HuggingFace tokenizer with local compatibility patches applied.

    Args:
        tokenizer_model: HuggingFace model identifier or local path.
        trust_remote_code: Whether to allow tokenizers that ship custom code.
    """
    _patch_roberta_processing_cls_alias()
    return AutoTokenizer.from_pretrained(
        tokenizer_model, trust_remote_code=trust_remote_code
    )
