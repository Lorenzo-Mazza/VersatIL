"""Mappings between action-local token IDs and model token IDs."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch

from versatil.data.constants import ActionTokenIdMappingType
from versatil.data.tokenization.huggingface import load_huggingface_tokenizer


class ActionTokenIdMapping(ABC):
    """Maps discretizer-local action IDs to the model token-id space."""

    @abstractmethod
    def model_token_count(self, action_token_count: int) -> int:
        """Return the model token count before the action tokenizer adds EOS."""

    def tokenizer_vocab_size(self, action_token_count: int) -> int:
        """Return the action tokenizer vocabulary size including EOS."""
        return self.model_token_count(action_token_count) + 1

    def eos_token_id(self, action_token_count: int) -> int:
        """Return the EOS token ID used after encoded action tokens."""
        return self.model_token_count(action_token_count)

    @abstractmethod
    def encode(self, local_token_ids: list[int] | np.ndarray) -> np.ndarray:
        """Map local action IDs to model token IDs."""

    @abstractmethod
    def decode(self, model_token_ids: np.ndarray | torch.Tensor) -> np.ndarray:
        """Map model token IDs back to local action IDs."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable state."""

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load serializable state."""

    def save_pretrained(self, path: Path) -> None:
        """Save optional external assets."""
        del path

    def load_pretrained_assets(self, path: Path) -> None:
        """Load optional external assets."""
        del path


class IdentityActionTokenIdMapping(ActionTokenIdMapping):
    """Use action IDs directly as model token IDs."""

    def model_token_count(self, action_token_count: int) -> int:
        """Return the unchanged action-token count."""
        return action_token_count

    def encode(self, local_token_ids: list[int] | np.ndarray) -> np.ndarray:
        """Return local action IDs unchanged as model token IDs."""
        return np.asarray(local_token_ids)

    def decode(self, model_token_ids: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return model token IDs unchanged as local action IDs."""
        if isinstance(model_token_ids, torch.Tensor):
            return model_token_ids.detach().cpu().numpy()
        return np.asarray(model_token_ids)

    def state_dict(self) -> dict[str, Any]:
        """Return serializable identity mapping state."""
        return {"type": ActionTokenIdMappingType.IDENTITY.value}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load identity mapping state."""
        del state_dict


class LanguageVocabularyActionTokenIdMapping(ActionTokenIdMapping):
    """Place action IDs in the tail of a language tokenizer's token-id space."""

    def __init__(
        self,
        language_tokenizer_model: str,
        num_special_tokens_to_skip: int = 128,
    ):
        """Initialize mapping into the tail of a language tokenizer vocabulary."""
        self.language_tokenizer_model = language_tokenizer_model
        self.num_special_tokens_to_skip = num_special_tokens_to_skip
        self.language_tokenizer = load_huggingface_tokenizer(
            tokenizer_model=language_tokenizer_model
        )
        if self.language_tokenizer.pad_token is None:
            self.language_tokenizer.pad_token = self.language_tokenizer.eos_token

    def model_token_count(self, action_token_count: int) -> int:
        """Return the language tokenizer vocabulary size if action IDs fit."""
        required_token_count = action_token_count + self.num_special_tokens_to_skip
        if self.language_tokenizer.vocab_size < required_token_count:
            raise ValueError(
                "Language tokenizer token count "
                f"({self.language_tokenizer.vocab_size}) is too small to hold "
                f"action tokens ({action_token_count}) plus skipped special tokens "
                f"({self.num_special_tokens_to_skip}). Required: {required_token_count}"
            )
        self._validate_eos_does_not_overlap_action_tokens(
            action_token_count=action_token_count
        )
        return self.language_tokenizer.vocab_size

    def tokenizer_vocab_size(self, action_token_count: int) -> int:
        """Return the language tokenizer vocabulary size without adding EOS."""
        return self.model_token_count(action_token_count=action_token_count)

    def eos_token_id(self, action_token_count: int) -> int:
        """Return the native language-tokenizer EOS ID."""
        self.model_token_count(action_token_count=action_token_count)
        eos_token_id = self.language_tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError(
                "Language tokenizer must define eos_token_id when used for "
                "action-token EOS."
            )
        return int(eos_token_id)

    def _validate_eos_does_not_overlap_action_tokens(
        self,
        action_token_count: int,
    ) -> None:
        """Ensure language EOS is not reused as an action token ID."""
        eos_token_id = self.language_tokenizer.eos_token_id
        if eos_token_id is None:
            raise ValueError(
                "Language tokenizer must define eos_token_id when used for "
                "action-token EOS."
            )
        max_action_token_id = (
            self.language_tokenizer.vocab_size - 1 - self.num_special_tokens_to_skip
        )
        min_action_token_id = max_action_token_id - action_token_count + 1
        if min_action_token_id <= int(eos_token_id) <= max_action_token_id:
            raise ValueError(
                "Language tokenizer EOS token overlaps with mapped action-token "
                "IDs. Increase num_special_tokens_to_skip or use another "
                f"tokenizer. eos_token_id={eos_token_id}, action_token_id_range="
                f"[{min_action_token_id}, {max_action_token_id}]."
            )

    def encode(self, local_token_ids: list[int] | np.ndarray) -> np.ndarray:
        """Map local action IDs to language-token tail IDs."""
        local_tokens = np.asarray(local_token_ids)
        return (
            self.language_tokenizer.vocab_size
            - 1
            - self.num_special_tokens_to_skip
            - local_tokens
        )

    def decode(self, model_token_ids: np.ndarray | torch.Tensor) -> np.ndarray:
        """Map language-token tail IDs back to local action IDs."""
        if isinstance(model_token_ids, torch.Tensor):
            model_token_ids = model_token_ids.detach().cpu().numpy()
        return (
            self.language_tokenizer.vocab_size
            - 1
            - self.num_special_tokens_to_skip
            - np.asarray(model_token_ids)
        )

    def state_dict(self) -> dict[str, Any]:
        """Return serializable language-vocabulary mapping state."""
        return {
            "type": ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value,
            "language_tokenizer_model": self.language_tokenizer_model,
            "num_special_tokens_to_skip": self.num_special_tokens_to_skip,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load language-vocabulary mapping state."""
        tokenizer_model = state_dict["language_tokenizer_model"]
        needs_reload = tokenizer_model != self.language_tokenizer_model
        self.language_tokenizer_model = tokenizer_model
        self.num_special_tokens_to_skip = state_dict["num_special_tokens_to_skip"]
        if needs_reload:
            self.language_tokenizer = load_huggingface_tokenizer(
                tokenizer_model=tokenizer_model
            )
            if self.language_tokenizer.pad_token is None:
                self.language_tokenizer.pad_token = self.language_tokenizer.eos_token

    def save_pretrained(self, path: Path) -> None:
        """Save the language tokenizer assets."""
        self.language_tokenizer.save_pretrained(path / "language_tokenizer")

    def load_pretrained_assets(self, path: Path) -> None:
        """Load saved language tokenizer assets when present."""
        language_path = path / "language_tokenizer"
        if language_path.exists():
            self.language_tokenizer = load_huggingface_tokenizer(
                tokenizer_model=language_path
            )
