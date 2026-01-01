"""Simple embedding layer encoder for encoding prompt tokens into continuous embeddings.
This encoder contains a single nn.Embedding layer to convert tokenized sequences
into embeddings.
"""
import torch
import torch.nn as nn

from refactoring.data.constants import TOKENIZED_OBSERVATIONS_KEY, IS_PAD_OBSERVATION_KEY
from refactoring.models.encoding.encoders.base import EncoderInput, EncoderOutput
from refactoring.models.encoding.encoders.constants import EncoderOutputKeys
from refactoring.models.encoding.encoders.unconditional import Encoder


class Embedder(Encoder):
    """Simple embedding layer encoder for tokenized inputs.

    This encoder consists of a single nn.Embedding layer that converts token IDs
    into dense embeddings.
    """
    def __init__(
            self,
            embedding_dim: int,
            max_token_len: int,
            initializer_range: float = 0.02,
    ):
        """Initialize embedder.

        Args:
            embedding_dim: Dimension of the embedding vectors
            max_token_len: Maximum sequence length (for validation and output spec)
            initializer_range: std of the embedding module initial weights.
        """
        specification = EncoderInput(keys=[TOKENIZED_OBSERVATIONS_KEY, IS_PAD_OBSERVATION_KEY],
                                     required=[TOKENIZED_OBSERVATIONS_KEY, IS_PAD_OBSERVATION_KEY], requires_tokenized=True)
        super().__init__(
            input_specification=specification,
            pretrained=False,
            frozen=False,
        )
        self.initializer_range = initializer_range
        self.vocab_size = None # To be set when loading tokenizer
        self.embedding_dim = embedding_dim
        self.max_token_len = max_token_len
        self.embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=embedding_dim,
        )
        self.padding_mask_name = f"{EncoderOutputKeys.TOKEN_EMBEDDING.value}_{EncoderOutputKeys.PADDING_MASK.value}"


    def set_tokenizer_vocab_size(self, vocab_size: int):
        """Set the vocabulary size for the embedding layer.

        Args:
            vocab_size: Size of the tokenizer vocabulary
        """
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
        )
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=self.initializer_range,
        )

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Load state dict with dynamic embedding resize for lazy-initialized vocab size."""
        embedding_key = prefix + "embedding.weight"
        if embedding_key in state_dict:
            weight = state_dict[embedding_key]
            vocab_size, emb_dim = weight.shape
            if self.embedding.weight.shape[0] != vocab_size:
                self.vocab_size = vocab_size
                self.embedding = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=emb_dim,
                    device=weight.device,
                )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Embed tokenized input.

        Args:
            inputs: Dictionary with single key specified in input_specification.
                The tensor should have shape (batch, seq_len) with token IDs.

        Returns:
            Dictionary with "TOKEN_EMBEDDING" key and tensor of shape (batch, seq_len, embedding_dim), and padding mask.

        Raises:
            ValueError: If sequence length exceeds max_token_len
        """
        tokens = inputs[TOKENIZED_OBSERVATIONS_KEY]
        token_padding_mask = inputs[IS_PAD_OBSERVATION_KEY]
        obs_horizon = 1
        match tokens.ndim:
            case 2:
                batch_size, seq_len = tokens.shape
            case 3:
                # (batch, obs_horizon, seq_len) - flatten obs_horizon into batch
                batch_size, obs_horizon, seq_len = tokens.shape
                tokens = tokens.view(batch_size * obs_horizon, seq_len)
            case _:
                raise ValueError(
                    f"Expected tokenized input to have shape (batch, seq_len) or "
                    f"(batch, obs_horizon, seq_len), got shape {tokens.shape}"
                )
        if seq_len > self.max_token_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_token_len {self.max_token_len}"
            )
        embeddings = self.embedding(tokens)
        if obs_horizon > 1:
            # Reshape back to (batch, obs_horizon, seq_len, embedding_dim)
            embeddings = embeddings.view(batch_size, obs_horizon, seq_len, self.embedding_dim)
            token_padding_mask = token_padding_mask.view(batch_size, obs_horizon, seq_len)
        return {EncoderOutputKeys.TOKEN_EMBEDDING.value: embeddings,
                self.padding_mask_name: token_padding_mask
                }


    def get_output_specification(self) -> EncoderOutput:
        """Get encoder output specification.

        Returns:
            EncoderOutput with feature "embedding" and dimension (max_token_len, embedding_dim)
        """
        return EncoderOutput(
            features=[EncoderOutputKeys.TOKEN_EMBEDDING.value, self.padding_mask_name],
            dimensions={EncoderOutputKeys.TOKEN_EMBEDDING.value: (self.max_token_len, self.embedding_dim),
                        self.padding_mask_name: (self.max_token_len,)},
        )


    def get_vocab_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Vocabulary size of the embedding layer
        """
        return self.vocab_size



