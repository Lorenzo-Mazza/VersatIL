import logging

import torch
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import BaseModelOutput

from refactoring.data.constants import TOKENIZED_OBSERVATIONS_KEY, IS_PAD_OBSERVATION_KEY
from refactoring.models.encoding.encoders.base import EncoderInput, EncoderOutput
from refactoring.models.encoding.encoders.constants import (
    AttentionImplementation,
    EncoderOutputKeys,
    PoolingMethod,
    LanguageEncoderType,
)
from refactoring.models.encoding.encoders.unconditional import Encoder
from refactoring.models.layers import LearnedAggregation


class LanguageEncoder(Encoder):
    """Language encoder using Transformers library."""
    def __init__(
            self,
            pretrained: bool,
            frozen: bool,
            pooling_method: str = PoolingMethod.DEFAULT.value,
            model_name: str = LanguageEncoderType.BERT_BASE.value,
            attention_type: str = AttentionImplementation.SDPA.value,
    ):
        """
        Args:
            pretrained: Whether to use pretrained weights
            frozen: Whether to freeze backbone weights
            pooling_method: How to extract features from transformer output
            model_name: Model identifier from LanguageEncoderType
            attention_type: Attention implementation to use
        """
        specification = EncoderInput(keys=[TOKENIZED_OBSERVATIONS_KEY, IS_PAD_OBSERVATION_KEY],
                                     required=[TOKENIZED_OBSERVATIONS_KEY], requires_tokenized=True)
        super().__init__(input_specification=specification, pretrained=pretrained, frozen=frozen)
        self.language_key = TOKENIZED_OBSERVATIONS_KEY
        self.pooling_method = pooling_method
        self.attention_type = attention_type
        self.model_name = model_name
        self._build_encoder()
        self.feature_dim = self.encoder.config.hidden_size
        self.pooling_head: LearnedAggregation | None = None
        self._setup_pooling()
        self.padding_mask_name = f"{EncoderOutputKeys.LANGUAGE.value}_{EncoderOutputKeys.PADDING_MASK.value}"

        if frozen:
            super()._freeze_weights()



    def _build_encoder(self):
        """Build language encoder and tokenizer."""
        config = AutoConfig.from_pretrained(self.model_name)
        if self.pretrained:
            self.encoder = AutoModel.from_pretrained(self.model_name,attn_implementation=self.attention_type, use_safetensors=True)
        else:
            config = AutoConfig.from_pretrained(self.model_name)
            self.encoder = AutoModel.from_config(config, attn_implementation=self.attention_type)

        self.max_text_length = config.max_position_embeddings



    def _setup_pooling(self):
        """Set-up pooling head and output dimensionality accordingly."""
        if self.pooling_method == PoolingMethod.LEARNED_AGGREGATION.value:
            self.pooling_head = LearnedAggregation(self.feature_dim).to(self.encoder.device)

        if self.pooling_method == PoolingMethod.NONE.value:
            self.output_dim = (self.max_text_length, self.feature_dim)
            self.padding_dim = self.max_text_length
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
            return outputs.last_hidden_state[:, 1:].mean(dim=1)  # GAP on tokens (exclude CLS)
        elif self.pooling_method == PoolingMethod.LEARNED_AGGREGATION.value:
            if self.pooling_head is None:
                raise RuntimeError("pooling_head must be initialized for LEARNED_AGGREGATION")
            result: torch.Tensor = self.pooling_head(outputs.last_hidden_state[:, 1:])  # Learned agg on tokens (exclude CLS)
            return result
        elif self.pooling_method == PoolingMethod.NONE.value:
            return outputs.last_hidden_state
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling_method}")


    def _pad_text_inputs(self, text_input_ids: torch.Tensor, language_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Pad or truncate text inputs to max_text_length."""
        if text_input_ids.shape[1] > self.max_text_length:
            text_input_ids = text_input_ids[:, :self.max_text_length]
            if language_mask is not None:
                language_mask = language_mask[:, :self.max_text_length]
            logging.warning(f"Input text length {text_input_ids.shape[1]} exceeds max_text_length "
                            f"{self.max_text_length}. Truncating input.")
        elif text_input_ids.shape[1] < self.max_text_length:
            pad_length = self.max_text_length - text_input_ids.shape[1]
            pad_tensor = torch.zeros((text_input_ids.shape[0], pad_length), dtype=text_input_ids.dtype,
                                     device=text_input_ids.device)
            text_input_ids = torch.cat([text_input_ids, pad_tensor], dim=1)
            if language_mask is not None:
                pad_mask = torch.ones((language_mask.shape[0], pad_length), dtype=language_mask.dtype,
                                      device=language_mask.device)
                language_mask = torch.cat([language_mask, pad_mask], dim=1)
            logging.warning(f"Input text length {text_input_ids.shape[1]} less than max_text_length "
                            f"{self.max_text_length}. Padding input.")
        return text_input_ids, language_mask


    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        """
        Forward pass through language encoder.

        Args:
            inputs: Dict with pre-tokenized text input (token IDs as tensors).
                Expected key: TOKENIZED_OBSERVATIONS_KEY

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

        language_mask = inputs.get(IS_PAD_OBSERVATION_KEY, None)
        T = None
        has_time = False
        if text_input_ids.dim() == 3:
            B, T, seq_len = text_input_ids.shape
            text_input_ids = text_input_ids.reshape(B * T, -1)
            language_mask = language_mask.reshape(B * T, -1) if language_mask is not None else None
            has_time = True
        else:
            B = text_input_ids.shape[0]

        text_input_ids, language_mask = self._pad_text_inputs(text_input_ids, language_mask)
        # Create attention mask from padding mask
        if language_mask is not None:
            attention_mask = ~language_mask
        else:
            attention_mask = torch.ones_like(text_input_ids, dtype=torch.bool)
        encoder_inputs = {
            "input_ids": text_input_ids.to(self.encoder.device),
            "attention_mask": attention_mask.to(self.encoder.device),
        }
        outputs = self.encoder(**encoder_inputs, return_dict=True)
        features = self._pool_features(outputs)
        padding_mask = ~attention_mask # B, max_text_length*T, True for padding positions
        if has_time:
            features = features.reshape(B, T, *features.shape[1:])
            if padding_mask.ndim >= 2:
                if self.pooling_method == PoolingMethod.NONE.value:
                    padding_mask = padding_mask.reshape(B, T, self.max_text_length)
                else:
                    padding_mask = torch.zeros(B, T, dtype=torch.bool, device=features.device)
        else:
            if self.pooling_method != PoolingMethod.NONE.value:
                padding_mask = torch.zeros(B, dtype=torch.bool, device=features.device)
        return {
            EncoderOutputKeys.LANGUAGE.value: features,
            self.padding_mask_name: padding_mask
        }


    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.LANGUAGE.value, self.padding_mask_name],
            dimensions={EncoderOutputKeys.LANGUAGE.value: self.output_dim,
                        self.padding_mask_name: self.padding_dim},
        )

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the language encoder.

        Returns:
            Vocabulary size of the language model
        """
        return self.encoder.config.vocab_size
