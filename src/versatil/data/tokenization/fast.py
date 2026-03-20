"""FAST action processor loading compatible with transformers>=5.0.

Transformers 5.0 changed ProcessorMixin._get_arguments_from_pretrained to load
any attribute containing "tokenizer" from a subfolder. This breaks custom processors
like physical-intelligence/fast's UniversalActionProcessor, whose bpe_tokenizer
attribute has no corresponding subfolder on the hub. This module bypasses
AutoProcessor.from_pretrained by loading the config and tokenizer separately, then
constructing the processor directly.
"""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.processing_utils import ProcessorMixin

_PROCESSOR_CONFIG_FILENAME = "processor_config.json"
_BPE_TOKENIZER_SUBFOLDER = "bpe_tokenizer"
_DEFAULT_HUB_MODEL = "physical-intelligence/fast"
_DEFAULT_CLASS_REF = "processing_action_tokenizer.UniversalActionProcessor"


def load_fast_processor(model_path: str) -> ProcessorMixin:  # UniversalActionProcessor
    """Load a FAST action processor from a HuggingFace hub model or local directory.

    Args:
        model_path: HuggingFace model ID or local directory path.

    Returns:
        Instantiated FAST action processor.
    """
    local_path = Path(model_path)
    is_local = local_path.is_dir()
    config = _load_processor_config(
        model_path=model_path, local_path=local_path, is_local=is_local
    )
    bpe_tokenizer = _load_bpe_tokenizer(
        model_path=model_path, local_path=local_path, is_local=is_local
    )
    processor_class = _resolve_processor_class(config=config, model_path=model_path)
    return processor_class(
        bpe_tokenizer=bpe_tokenizer,
        scale=config.get("scale", 10),
        vocab_size=config.get("vocab_size", 2048),
        min_token=config.get("min_token", 0),
        action_dim=config.get("action_dim"),
        time_horizon=config.get("time_horizon"),
    )


def _load_processor_config(model_path: str, local_path: Path, is_local: bool) -> dict:
    """Load processor_config.json from local path or hub."""
    if is_local:
        config_file = local_path / _PROCESSOR_CONFIG_FILENAME
    else:
        config_file = hf_hub_download(
            repo_id=model_path, filename=_PROCESSOR_CONFIG_FILENAME
        )
    with open(config_file) as f:
        return json.load(f)


def _load_bpe_tokenizer(
    model_path: str, local_path: Path, is_local: bool
) -> PreTrainedTokenizerFast:
    """Load the BPE tokenizer from root (hub) or bpe_tokenizer/ subfolder (local)."""
    if is_local:
        tokenizer_path = str(local_path / _BPE_TOKENIZER_SUBFOLDER)
    else:
        tokenizer_path = model_path
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


def _resolve_processor_class(config: dict, model_path: str) -> type:
    """Resolve the processor class from config auto_map.

    When auto_map is missing (local checkpoints saved by save_pretrained),
    falls back to the default hub model for class resolution.
    """
    if "auto_map" in config:
        class_ref = config["auto_map"]["AutoProcessor"]
        return get_class_from_dynamic_module(
            class_ref, model_path, trust_remote_code=True
        )
    return get_class_from_dynamic_module(
        _DEFAULT_CLASS_REF, _DEFAULT_HUB_MODEL, trust_remote_code=True
    )
