import logging

import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import BaseModelOutput

from versatil.data.constants import SampleKey
from versatil.models.encoding.encoders.base import EncoderInput, EncoderOutput
from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    EncoderOutputKeys,
    PoolingMethod,
    LanguageEncoderType,
)
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.layers import LearnedAggregation


class LanguageEncoder(Encoder):
    """Language encoder using Transformers library."""

    def __init__(
        self,
        pretrained: bool,
        frozen: bool,
        pooling_method: str = PoolingMethod.DEFAULT.value,
        model_name: str = LanguageEncoderType.BERT_BASE.value,
        attention_type: str = AttentionImplementation.SDPA.value,
        max_token_len: int = 128,
        use_embeddings_only: bool = False,
    ):
        """
        Args:
            pretrained: Whether to use pretrained weights
            frozen: Whether to freeze backbone weights
            pooling_method: How to extract features from transformer output
            model_name: Model identifier from LanguageEncoderType
            attention_type: Attention implementation to use
            max_token_len: Maximum token sequence length for the encoder
            use_embeddings_only: If True, use only the pretrained token embedding layer
        """
        specification = EncoderInput(
            keys=[
                SampleKey.TOKENIZED_OBSERVATIONS.value,
                SampleKey.IS_PAD_OBSERVATION.value,
            ],
            required=[SampleKey.TOKENIZED_OBSERVATIONS.value],
            requires_tokenized=True,
        )
        super().__init__(
            input_specification=specification, pretrained=pretrained, frozen=frozen
        )
        self.language_key = SampleKey.TOKENIZED_OBSERVATIONS.value
        self.pooling_method = pooling_method
        self.attention_type = attention_type
        self.model_name = model_name
        self.max_token_len = max_token_len
        self.use_embeddings_only = use_embeddings_only
        if self.use_embeddings_only and self.pooling_method != PoolingMethod.NONE.value:
            raise ValueError(
                "use_embeddings_only=True is only compatible with pooling_method=PoolingMethod.NONE"
            )
        self._build_encoder()
        self.feature_dim = self.encoder.embedding_dim if self.use_embeddings_only else self.config.hidden_size
        self.pooling_head: LearnedAggregation | None = None
        self._setup_pooling()
        self.padding_mask_name = (
            f"{EncoderOutputKeys.LANGUAGE.value}_{EncoderOutputKeys.PADDING_MASK.value}"
        )

        if frozen:
            super()._freeze_weights()

    def _build_encoder(self):
        """Build language encoder and tokenizer."""
        self.config = AutoConfig.from_pretrained(self.model_name)
        if self.use_embeddings_only:
            # Models like ALBERT use factorized embeddings where
            # embedding_size != hidden_size
            if hasattr(self.config, "embedding_size"):
                embedding_dim = self.config.embedding_size
            elif hasattr(self.config, "hidden_size"):
                embedding_dim = self.config.hidden_size
            else:
                raise ValueError(
                    f"Config for {self.model_name} has neither "
                    f"'embedding_size' nor 'hidden_size'"
                )
            self.encoder = nn.Embedding(
                num_embeddings=self.config.vocab_size,
                embedding_dim=embedding_dim,
            )
            if self.pretrained:
                temp_model = AutoModel.from_pretrained(self.model_name)
                source_emb = temp_model.get_input_embeddings()
                self.encoder.load_state_dict(source_emb.state_dict())
                del temp_model
            else:
                nn.init.normal_(self.encoder.weight, mean=0.0, std=0.02)
        else:
            if self.pretrained:
                self.encoder = AutoModel.from_pretrained(
                    self.model_name,
                    attn_implementation=self.attention_type,
                )
            else:
                self.encoder = AutoModel.from_config(
                    self.config, attn_implementation=self.attention_type
                )

    def _setup_pooling(self):
        """Set-up pooling head and output dimensionality accordingly."""
        if self.pooling_method == PoolingMethod.LEARNED_AGGREGATION.value:
            self.pooling_head = LearnedAggregation(self.feature_dim).to(
                self.encoder.device
            )

        if self.pooling_method == PoolingMethod.NONE.value:
            self.output_dim = (self.max_token_len, self.feature_dim)
            self.padding_dim = self.max_token_len
        else:
            self.output_dim = self.feature_dim
            self.padding_dim = 1

    def _pool_features(self, outputs: BaseModelOutput) -> torch.Tensor:
        """Pool features using the encoder pooling head."""
        if outputs.last_hidden_state is None:
            raise RuntimeError("last_hidden_state must be present in model output")
        if self.pooling_method == PoolingMethod.DEFAULT.value:
            return outputs.last_hidden_state[:, 0]  # CLS token
        elif self.pooling_method == PoolingMethod.AVERAGE.value:
            return outputs.last_hidden_state[:, 1:].mean(
                dim=1
            )  # GAP on tokens (exclude CLS)
        elif self.pooling_method == PoolingMethod.LEARNED_AGGREGATION.value:
            if self.pooling_head is None:
                raise RuntimeError(
                    "pooling_head must be initialized for LEARNED_AGGREGATION"
                )
            result: torch.Tensor = self.pooling_head(
                outputs.last_hidden_state[:, 1:]
            )  # Learned agg on tokens (exclude CLS)
            return result
        elif self.pooling_method == PoolingMethod.NONE.value:
            return outputs.last_hidden_state
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")

    def _pad_text_inputs(
        self, text_input_ids: torch.Tensor, language_mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pad or truncate text inputs to max_token_len."""
        if text_input_ids.shape[1] > self.max_token_len:
            text_input_ids = text_input_ids[:, : self.max_token_len]
            if language_mask is not None:
                language_mask = language_mask[:, : self.max_token_len]
            logging.warning(
                f"Input text length {text_input_ids.shape[1]} exceeds max_token_len "
                f"{self.max_token_len}. Truncating input."
            )
        elif text_input_ids.shape[1] < self.max_token_len:
            pad_length = self.max_token_len - text_input_ids.shape[1]
            pad_tensor = torch.zeros(
                (text_input_ids.shape[0], pad_length),
                dtype=text_input_ids.dtype,
                device=text_input_ids.device,
            )
            text_input_ids = torch.cat([text_input_ids, pad_tensor], dim=1)
            if language_mask is not None:
                pad_mask = torch.ones(
                    (language_mask.shape[0], pad_length),
                    dtype=language_mask.dtype,
                    device=language_mask.device,
                )
                language_mask = torch.cat([language_mask, pad_mask], dim=1)
            logging.warning(
                f"Input text length {text_input_ids.shape[1]} less than max_token_len "
                f"{self.max_token_len}. Padding input."
            )
        return text_input_ids, language_mask

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        """
        Forward pass through language encoder.

        Args:
            inputs: Dict with pre-tokenized text input (token IDs as tensors).
                Expected key: SampleKey.TOKENIZED_OBSERVATIONS.value

        Returns:
            Encoded text features, wrapped in a dictionary indexed by the `LANGUAGE_FEATURES` constant.
        """
        # Text is pre-tokenized in SampleBuilder
        if self.language_key not in inputs:
            raise ValueError(
                f"Language encoder expects pre-tokenized input. "
                f"Expected key '{self.language_key}' not found in inputs. "
                f"Ensure tokenization is enabled in DataloaderConfig."
            )
        text_input_ids = inputs[self.language_key]
        if not isinstance(text_input_ids, torch.Tensor):
            raise ValueError("tokenized_observations must be a tensor")

        language_mask = inputs.get(SampleKey.IS_PAD_OBSERVATION.value, None)
        T = None
        has_time = False
        device = next(self.encoder.parameters()).device

        if text_input_ids.dim() == 3:
            B, T, seq_len = text_input_ids.shape
            text_input_ids = text_input_ids.reshape(B * T, -1)
            language_mask = (
                language_mask.reshape(B * T, -1) if language_mask is not None else None
            )
            has_time = True
        else:
            B = text_input_ids.shape[0]

        text_input_ids, language_mask = self._pad_text_inputs(
            text_input_ids, language_mask
        )
        # Create attention mask from padding mask
        if language_mask is not None:
            attention_mask = ~language_mask
        else:
            attention_mask = torch.ones_like(text_input_ids, dtype=torch.bool)

        if self.use_embeddings_only:
            features = self.encoder(text_input_ids.to(device))
        else:
            encoder_inputs = {
                "input_ids": text_input_ids.to(device),
                "attention_mask": attention_mask.to(device),
            }
            outputs = self.encoder(**encoder_inputs, return_dict=True)
            features = self._pool_features(outputs)

        padding_mask = ~attention_mask  # B, max_token_len*T, True for padding positions
        if has_time:
            features = features.reshape(B, T, *features.shape[1:])
            if padding_mask.ndim >= 2:
                if self.pooling_method == PoolingMethod.NONE.value:
                    padding_mask = padding_mask.reshape(B, T, self.max_token_len)
                else:
                    padding_mask = torch.zeros(
                        B, T, dtype=torch.bool, device=features.device
                    )
        else:
            if self.pooling_method != PoolingMethod.NONE.value:
                padding_mask = torch.zeros(B, dtype=torch.bool, device=features.device)
        return {
            EncoderOutputKeys.LANGUAGE.value: features,
            self.padding_mask_name: padding_mask,
        }

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.LANGUAGE.value, self.padding_mask_name],
            dimensions={
                EncoderOutputKeys.LANGUAGE.value: self.output_dim,
                self.padding_mask_name: self.padding_dim,
            },
        )

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the language encoder.

        Returns:
            Vocabulary size of the language model
        """
        return self.config.vocab_size
