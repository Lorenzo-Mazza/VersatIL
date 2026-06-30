"""Action tokenizer for continuous-action discretizers and model token IDs."""

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from versatil.data.constants import (
    ActionDiscretizerType,
    ActionTokenIdMappingType,
    SampleKey,
)
from versatil.data.tokenization.action_discretizer import (
    ActionDiscretizer,
    BinnedActionDiscretizer,
    FastActionDiscretizer,
    UniformBinnedActionDiscretizer,
)
from versatil.data.tokenization.action_token_id_mapping import (
    ActionTokenIdMapping,
    IdentityActionTokenIdMapping,
    LanguageVocabularyActionTokenIdMapping,
)

# Disable tokenizers parallelism to avoid fork warnings with DataLoader workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class ActionTokenizer:
    """Tokenizes continuous action chunks for discrete action-token decoders.

    A single action chunk has shape (time_horizon, action_dim). A batch of
    chunks has shape (batch_size, time_horizon, action_dim). Encoded model
    token tensors have shape (max_token_len,) for one chunk or
    (batch_size, max_token_len) for a batch.

    The tokenizer has two explicit parts:
    - an action discretizer, e.g. FAST or per-value binning;
    - a token-id mapping, e.g. identity IDs or language-tokenizer tail IDs.

    The class owns sequence behavior: EOS, padding, masks, and
    batch dispatch.
    """

    def __init__(
        self,
        action_discretizer: ActionDiscretizer | None = None,
        token_id_mapping: ActionTokenIdMapping | None = None,
        max_token_len: int = 256,
        pad_token_id: int = 0,
        device: torch.device | None = None,
    ):
        """Initialize action tokenizer.

        Args:
            action_discretizer: Continuous-action discretizer. Defaults to FAST.
            token_id_mapping: Mapping from action-local IDs to model token IDs.
                Defaults to identity.
            max_token_len: Maximum token sequence length after EOS and padding.
            pad_token_id: Token ID to use for padding.
            device: Target device for returned token tensors.
        """
        self.device = device if device is not None else torch.device("cpu")
        self.max_token_len = max_token_len
        self.pad_token_id = pad_token_id
        self.eos_token_id: int | None = None
        self.vocab_size: int | None = None

        if action_discretizer is None:
            action_discretizer = FastActionDiscretizer()
        if token_id_mapping is None:
            token_id_mapping = IdentityActionTokenIdMapping()

        self.action_discretizer = action_discretizer
        self.token_id_mapping = token_id_mapping
        self._is_fitted = self.action_discretizer.is_fitted
        self._refresh_vocabulary_if_available()

    def fit(self, action_chunks: np.ndarray) -> None:
        """Fit the action discretizer on normalized action chunks.

        Args:
            action_chunks: Array with shape
                (num_chunks, time_horizon, action_dim).
        """
        logging.info(
            f"Fitting action discretizer on {action_chunks.shape[0]} chunks "
            f"(time_horizon={action_chunks.shape[1]}, action_dim={action_chunks.shape[2]})"
        )
        self.action_discretizer.fit(action_chunks)
        self._is_fitted = self.action_discretizer.is_fitted
        self._refresh_vocabulary_if_available(force=True)
        logging.info(
            "Fitted action tokenizer "
            f"(discretizer={type(self.action_discretizer).__name__}, "
            f"token_id_mapping={type(self.token_id_mapping).__name__}, "
            f"vocab_size={self.vocab_size})"
        )

    def _refresh_vocabulary_if_available(self, force: bool = False) -> None:
        """Refresh model vocabulary size and EOS ID when discretizer metadata exists."""
        if (
            not force
            and not self.action_discretizer.is_fitted
            and not isinstance(
                self.token_id_mapping, LanguageVocabularyActionTokenIdMapping
            )
        ):
            return
        self.eos_token_id = self.token_id_mapping.eos_token_id(
            self.action_discretizer.token_count
        )
        self.vocab_size = self.token_id_mapping.tokenizer_vocab_size(
            self.action_discretizer.token_count
        )
        if self.eos_token_id < 0 or self.eos_token_id >= self.vocab_size:
            raise ValueError(
                "Action tokenizer EOS token ID must be inside vocabulary: "
                f"eos_token_id={self.eos_token_id}, vocab_size={self.vocab_size}."
            )

    def encode_chunk(
        self,
        action_chunk: np.ndarray | torch.Tensor,
        is_pad_mask: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode one normalized action chunk.

        Args:
            action_chunk: Array or tensor with shape (time_horizon, action_dim).
            is_pad_mask: Optional boolean mask with shape (time_horizon,), where
                True marks padded action rows to drop before tokenization.

        Returns:
            Dictionary containing token IDs and token padding mask, each with
            shape (max_token_len,).
        """
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before encoding")

        action_chunk_to_tokenize = self._select_valid_actions(
            action_chunk=action_chunk,
            is_pad_mask=is_pad_mask,
        )
        local_tokens = self.action_discretizer.encode(action_chunk_to_tokenize)
        tokens = self.token_id_mapping.encode(local_tokens).tolist()
        return self._pad_and_append_eos(tokens)

    def encode_batch(
        self,
        action_chunks: np.ndarray | torch.Tensor,
        is_pad_mask: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of normalized action chunks.

        Args:
            action_chunks: Array or tensor with shape
                (batch_size, time_horizon, action_dim).
            is_pad_mask: Optional boolean mask with shape
                (batch_size, time_horizon), where True marks padded action rows.

        Returns:
            Dictionary containing token IDs and token padding mask, each with
            shape (batch_size, max_token_len).
        """
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before encoding")
        all_tokens = []
        all_is_pad = []
        for i in range(action_chunks.shape[0]):
            chunk_pad_mask = is_pad_mask[i] if is_pad_mask is not None else None
            result = self.encode_chunk(action_chunks[i], is_pad_mask=chunk_pad_mask)
            all_tokens.append(result[SampleKey.TOKENIZED_ACTIONS.value])
            all_is_pad.append(result[SampleKey.IS_PAD_ACTION.value])

        return {
            SampleKey.TOKENIZED_ACTIONS.value: torch.stack(all_tokens),
            SampleKey.IS_PAD_ACTION.value: torch.stack(all_is_pad),
        }

    def encode(
        self,
        action_chunks: np.ndarray | torch.Tensor,
        is_pad_mask: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode one action chunk or a batch of action chunks.

        2D input must have shape (time_horizon, action_dim). 3D input must have
        shape (batch_size, time_horizon, action_dim).
        """
        action_chunks_data = (
            action_chunks
            if isinstance(action_chunks, torch.Tensor)
            else np.asarray(action_chunks)
        )
        if action_chunks_data.ndim == 2:
            return self.encode_chunk(
                action_chunk=action_chunks, is_pad_mask=is_pad_mask
            )
        if action_chunks_data.ndim == 3:
            return self.encode_batch(
                action_chunks=action_chunks, is_pad_mask=is_pad_mask
            )
        raise ValueError(
            f"Expected 2D or 3D input, got shape {action_chunks_data.shape}"
        )

    def decode_chunk(self, tokens: torch.Tensor | list[int] | np.ndarray) -> np.ndarray:
        """Decode one model token sequence into one action chunk.

        Args:
            tokens: 1D model token IDs with shape (token_sequence_len,). The
                sequence may include EOS and trailing padding IDs.

        Returns:
            Normalized action chunk with shape (time_horizon, action_dim).
        """
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before decoding")

        token_ids_array = self._to_numpy_tokens(tokens)
        local_tokens = self._strip_and_unmap_tokens(token_ids_array)
        return self.action_discretizer.decode([local_tokens.tolist()])[0]

    def decode_batch(self, tokens: torch.Tensor | np.ndarray) -> np.ndarray:
        """Decode a batch of model token sequences into action chunks.

        Args:
            tokens: 2D model token IDs with shape
                (batch_size, token_sequence_len). Sequences may include EOS and
                trailing padding IDs.

        Returns:
            Normalized action chunks with shape
            (batch_size, time_horizon, action_dim).
        """
        if not self._is_fitted:
            raise RuntimeError("Tokenizer must be fitted or loaded before decoding")

        token_ids_array = (
            tokens.detach().cpu().numpy()
            if isinstance(tokens, torch.Tensor)
            else tokens
        )
        local_token_sequences = [
            self._strip_and_unmap_tokens(sample_tokens).tolist()
            for sample_tokens in token_ids_array
        ]
        return self.action_discretizer.decode(local_token_sequences)

    def decode(self, tokens: torch.Tensor | list[int] | np.ndarray) -> np.ndarray:
        """Decode one model token sequence or a batch of model token sequences.

        1D token input has shape (token_sequence_len,) and returns shape
        (time_horizon, action_dim). 2D token input has shape
        (batch_size, token_sequence_len) and returns shape
        (batch_size, time_horizon, action_dim).
        """
        token_ids_array = self._to_numpy_tokens(tokens)
        if token_ids_array.ndim == 1:
            return self.decode_chunk(tokens)
        if token_ids_array.ndim == 2:
            return self.decode_batch(tokens)
        raise ValueError(f"Expected 1D or 2D input, got shape {token_ids_array.shape}")

    def to(self, device: torch.device) -> "ActionTokenizer":
        """Move tokenizer tensors to a device."""
        self.device = device
        self.action_discretizer.to(device)
        return self

    def save_pretrained(self, path: str | Path) -> None:
        """Save tokenizer state and optional external assets."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted tokenizer")

        torch.save(self.state_dict(), path / "action_tokenizer_state.pt")
        self.action_discretizer.save_pretrained(path)
        self.token_id_mapping.save_pretrained(path)
        logging.info(f"Saved action tokenizer to {path}")

    def state_dict(self) -> dict[str, Any]:
        """Get serializable tokenizer state."""
        return {
            "action_discretizer": self.action_discretizer.state_dict(),
            "token_id_mapping": self.token_id_mapping.state_dict(),
            "max_token_len": self.max_token_len,
            "pad_token_id": self.pad_token_id,
            "vocab_size": self.vocab_size,
            "eos_token_id": self.eos_token_id,
            "is_fitted": self._is_fitted,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load tokenizer state."""
        self.max_token_len = state_dict.get("max_token_len", self.max_token_len)
        self.pad_token_id = state_dict.get("pad_token_id", self.pad_token_id)
        self.vocab_size = state_dict["vocab_size"]
        self.eos_token_id = state_dict.get("eos_token_id")
        self._is_fitted = state_dict["is_fitted"]
        self.action_discretizer.load_state_dict(state_dict["action_discretizer"])
        self.token_id_mapping.load_state_dict(state_dict["token_id_mapping"])

    @classmethod
    def from_pretrained(
        cls, path: str | Path, device: torch.device | None = None
    ) -> "ActionTokenizer":
        """Load tokenizer from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {path}")

        state_dict = torch.load(
            path / "action_tokenizer_state.pt",
            map_location=device or torch.device("cpu"),
            weights_only=False,
        )

        tokenizer = cls._from_state_dict(state_dict=state_dict, device=device)
        tokenizer.load_state_dict(state_dict)
        tokenizer.action_discretizer.load_pretrained_assets(path)
        tokenizer.token_id_mapping.load_pretrained_assets(path)
        logging.info(f"Loaded action tokenizer from {path}")
        return tokenizer

    def _pad_and_append_eos(self, tokens: list[int]) -> dict[str, torch.Tensor]:
        """Append EOS and pad one token sequence to ``max_token_len``."""
        tokens_len = len(tokens)
        if tokens_len >= self.max_token_len:
            raise ValueError(
                "Encoded action token sequence does not fit in max_token_len "
                f"after EOS: action_token_count={tokens_len}, "
                f"max_token_len={self.max_token_len}. Increase max_token_len "
                "or use a tokenizer that emits fewer action tokens."
            )

        tokens.append(self.eos_token_id)
        sequence_len = tokens_len + 1
        padding_len = self.max_token_len - sequence_len
        if padding_len > 0:
            tokens = tokens + [self.pad_token_id] * padding_len
        is_pad = [False] * sequence_len + [True] * padding_len

        return {
            SampleKey.TOKENIZED_ACTIONS.value: torch.tensor(
                tokens, dtype=torch.long, device=self.device
            ),
            SampleKey.IS_PAD_ACTION.value: torch.tensor(
                is_pad, dtype=torch.bool, device=self.device
            ),
        }

    def _strip_and_unmap_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """Remove special tokens and map model IDs back to local action IDs."""
        valid_tokens = self._strip_decode_special_tokens(tokens)
        return self.token_id_mapping.decode(valid_tokens).astype(np.int64)

    def _strip_decode_special_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """Remove EOS and trailing padding without dropping valid zero token IDs."""
        token_ids_array = np.asarray(tokens)
        if self.eos_token_id is not None:
            eos_indices = np.flatnonzero(token_ids_array == self.eos_token_id)
            if eos_indices.size > 0:
                return token_ids_array[: eos_indices[0]]

        end_index = token_ids_array.shape[0]
        while end_index > 0 and token_ids_array[end_index - 1] == self.pad_token_id:
            end_index -= 1
        return token_ids_array[:end_index]

    def _select_valid_actions(
        self,
        action_chunk: np.ndarray | torch.Tensor,
        is_pad_mask: torch.Tensor | np.ndarray | None,
    ) -> np.ndarray:
        """Drop padded action rows before fitting or encoding one chunk."""
        if is_pad_mask is None:
            if isinstance(action_chunk, torch.Tensor):
                return action_chunk.detach().cpu().numpy()
            return action_chunk

        if isinstance(is_pad_mask, torch.Tensor):
            padding_mask_array = is_pad_mask.detach().cpu().numpy()
        else:
            padding_mask_array = is_pad_mask
        valid_mask = ~padding_mask_array
        if isinstance(action_chunk, torch.Tensor):
            return action_chunk[valid_mask].detach().cpu().numpy()
        return action_chunk[valid_mask]

    @staticmethod
    def _to_numpy_tokens(tokens: torch.Tensor | list[int] | np.ndarray) -> np.ndarray:
        """Convert token containers to a NumPy array."""
        if isinstance(tokens, torch.Tensor):
            return tokens.detach().cpu().numpy()
        if isinstance(tokens, list):
            return np.array(tokens)
        return tokens

    @classmethod
    def _from_state_dict(
        cls,
        state_dict: dict[str, Any],
        device: torch.device | None,
    ) -> "ActionTokenizer":
        """Create an action tokenizer with components described by serialized state."""
        action_discretizer = _build_action_discretizer_from_state(
            state_dict["action_discretizer"],
            device=device,
        )
        token_id_mapping = _build_token_id_mapping_from_state(
            state_dict["token_id_mapping"]
        )
        return cls(
            action_discretizer=action_discretizer,
            token_id_mapping=token_id_mapping,
            max_token_len=state_dict.get("max_token_len", 256),
            pad_token_id=state_dict.get("pad_token_id", 0),
            device=device,
        )


def _build_action_discretizer_from_state(
    state_dict: dict[str, Any],
    device: torch.device | None,
) -> ActionDiscretizer:
    """Instantiate an action discretizer from serialized state."""
    match state_dict["type"]:
        case ActionDiscretizerType.FAST.value:
            return FastActionDiscretizer(
                use_pretrained=state_dict["use_pretrained"],
                tokenizer_model=state_dict.get(
                    "tokenizer_model", "physical-intelligence/fast"
                ),
            )
        case ActionDiscretizerType.BINNED.value:
            return BinnedActionDiscretizer(
                num_bins=state_dict["num_bins"],
                device=device,
            )
        case ActionDiscretizerType.UNIFORM_BINNED.value:
            return UniformBinnedActionDiscretizer(
                num_bins=state_dict["num_bins"],
                min_value=state_dict.get("min_value", -1.0),
                max_value=state_dict.get("max_value", 1.0),
            )
        case unsupported_type:
            raise ValueError(f"Unsupported action discretizer type: {unsupported_type}")


def _build_token_id_mapping_from_state(
    state_dict: dict[str, Any],
) -> ActionTokenIdMapping:
    """Instantiate a token-id mapping from serialized state."""
    match state_dict["type"]:
        case ActionTokenIdMappingType.IDENTITY.value:
            return IdentityActionTokenIdMapping()
        case ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value:
            return LanguageVocabularyActionTokenIdMapping(
                language_tokenizer_model=state_dict["language_tokenizer_model"],
                num_special_tokens_to_skip=state_dict["num_special_tokens_to_skip"],
            )
        case unsupported_type:
            raise ValueError(
                f"Unsupported action token-id mapping type: {unsupported_type}"
            )
