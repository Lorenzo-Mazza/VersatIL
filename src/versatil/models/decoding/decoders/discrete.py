"""Base decoder for tokenized action prediction."""

import torch
import torch.nn as nn

from versatil.data.constants import SampleKey
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import ActionTokenizer, Tokenizer
from versatil.models.decoding.action_heads.base import BaseActionHead
from versatil.models.decoding.constants import ActionHeadLayout, DecoderOutputKey
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput


class DiscreteDecoder(ActionDecoder):
    """Base class for decoders trained on tokenized action targets.

    Shape notation:
        B: batch size, A: target action-token length, D: token embedding
        dimension, V: action-token vocabulary size.
    """

    requires_tokenized_actions: bool = True

    def __init__(
        self,
        decoder_input: DecoderInput,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        action_heads: dict[str, BaseActionHead],
        device: str,
        observation_horizon: int,
        prediction_horizon: int,
        temperature: float,
        learnable_temperature: bool,
        deterministic: bool,
    ) -> None:
        """Initialize common discrete-action decoder state."""
        super().__init__(
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
        )
        self.deterministic: bool = deterministic
        self.temperature: nn.Parameter = nn.Parameter(
            torch.tensor(temperature, dtype=torch.float32),
            requires_grad=learnable_temperature,
        )
        self.token_embedding: nn.Module | None = None
        self.vocab_size: int | None = None

    def _init_action_bos_embedding(
        self,
        embedding_dimension: int,
        initializer_range: float,
    ) -> None:
        """Create the learned BOS embedding used before action tokens."""
        self.action_bos_embedding = nn.Parameter(torch.empty(1, 1, embedding_dimension))
        nn.init.normal_(
            self.action_bos_embedding,
            mean=0.0,
            std=initializer_range,
        )

    def _action_token_initializer_range(self) -> float:
        """Return the normal initializer std used for token embeddings."""
        raise NotImplementedError

    def _action_token_embedding_dimension(self) -> int:
        """Return the embedding dimension consumed by token embeddings."""
        raise NotImplementedError(
            f"{type(self).__name__} must define the action-token embedding dimension."
        )

    def set_tokenizer(self, tokenizer: Tokenizer | None = None) -> None:
        """Set tokenizer and bind a vocabulary action head when configured."""
        action_tokenizer = self._require_action_tokenizer(tokenizer=tokenizer)
        self.tokenizer = action_tokenizer
        if self.action_head_layout == ActionHeadLayout.VOCABULARY:
            self._bind_vocabulary_action_tokenizer(action_tokenizer=action_tokenizer)

    def _require_action_tokenizer(
        self,
        tokenizer: Tokenizer | None,
    ) -> ActionTokenizer:
        """Return the action tokenizer required by discrete decoders."""
        if tokenizer is None or tokenizer.action_tokenizer is None:
            raise ValueError(
                f"{type(self).__name__} requires a tokenizer for tokenized action prediction."
            )
        return tokenizer.action_tokenizer

    def _bind_vocabulary_action_tokenizer(
        self,
        action_tokenizer: ActionTokenizer,
    ) -> None:
        """Tie the local vocabulary head to newly created token embeddings."""
        device = self.temperature.device
        self.vocab_size = action_tokenizer.vocab_size
        embedding_dimension = self._action_token_embedding_dimension()
        output_block_in_features = self.action_heads[
            DecoderOutputKey.ACTION_LOGITS.value
        ].output_proj.in_features
        initializer_range = self._action_token_initializer_range()

        if output_block_in_features != embedding_dimension:
            token_input_embedding = nn.Embedding(
                self.vocab_size,
                output_block_in_features,
            ).to(device)
            token_projection = nn.Linear(
                output_block_in_features,
                embedding_dimension,
            ).to(device)
            self.token_embedding = nn.Sequential(
                token_input_embedding,
                token_projection,
            ).to(device)
            nn.init.normal_(
                token_projection.weight,
                mean=0.0,
                std=initializer_range,
            )
        else:
            token_input_embedding = nn.Embedding(
                self.vocab_size,
                embedding_dimension,
            ).to(device)
            self.token_embedding = token_input_embedding

        nn.init.normal_(
            token_input_embedding.weight,
            mean=0.0,
            std=initializer_range,
        )
        lm_head = nn.Linear(
            output_block_in_features,
            self.vocab_size,
            bias=False,
            device=device,
        )
        lm_head.weight = token_input_embedding.weight
        self.action_heads[
            DecoderOutputKey.ACTION_LOGITS.value
        ].output_dim = self.vocab_size
        self.action_heads[DecoderOutputKey.ACTION_LOGITS.value].output_proj = lm_head

    def _validate_action_tokenizer_is_set(self) -> None:
        """Ensure tokenizer-dependent action-token modules are initialized."""
        if (
            self.token_embedding is None
            or self.tokenizer is None
            or self.vocab_size is None
        ):
            raise ValueError(
                f"{type(self).__name__} requires set_tokenizer() to be called before forward."
            )

    def _get_max_generation_steps(self, available_context_steps: int) -> int:
        """Return the action-token generation cap within context capacity."""
        if self.tokenizer is None:
            raise ValueError(
                f"{type(self).__name__} requires set_tokenizer() to be called before inference."
            )
        if available_context_steps < 1:
            raise ValueError(
                f"{type(self).__name__} has no context capacity left for action-token generation."
            )
        return min(int(self.tokenizer.max_token_len), available_context_steps)

    def _get_target_token_ids(
        self,
        actions: dict[str, torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        """Read teacher-forcing target token IDs from the action dictionary."""
        if SampleKey.TOKENIZED_ACTIONS.value not in actions:
            raise ValueError(
                f"{type(self).__name__} training requires "
                f"'{SampleKey.TOKENIZED_ACTIONS.value}' in actions."
            )
        target_token_ids = actions[SampleKey.TOKENIZED_ACTIONS.value]
        if target_token_ids.ndim != 2:
            raise ValueError(
                f"'{SampleKey.TOKENIZED_ACTIONS.value}' must have shape "
                f"(B, token_length), got {target_token_ids.shape}."
            )
        if target_token_ids.shape[0] != batch_size:
            raise ValueError(
                f"'{SampleKey.TOKENIZED_ACTIONS.value}' batch size must match "
                f"feature batch size {batch_size}, got {target_token_ids.shape[0]}."
            )
        return target_token_ids

    def _expand_action_bos_embedding(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Expand the learned BOS embedding to the current batch."""
        return self.action_bos_embedding.to(device=device, dtype=dtype).expand(
            batch_size,
            -1,
            -1,
        )

    def _sample_next_action_token(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample or greedily choose the next action token from logits."""
        logits_scaled = logits / self.temperature.clamp(min=0.01)
        if self.deterministic:
            return torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits_scaled, dim=-1)
        return torch.multinomial(probs.squeeze(1), num_samples=1)
