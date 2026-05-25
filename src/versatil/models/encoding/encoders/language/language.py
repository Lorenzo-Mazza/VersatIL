"""Language encoder using HuggingFace Transformers models."""

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from versatil.data.constants import SampleKey
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.adaptation.lora import (
    LoRAAdaptation,
    apply_lora_config,
)
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    EncoderOutputKeys,
    LanguageEncoderType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.language_mixin import LanguageEncoderMixin
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import (
    FeatureMetadata,
    FeatureType,
    infer_feature_type,
)
from versatil.models.layers.pooling.pooling_head import create_token_pooling_head


class LanguageEncoder(LanguageEncoderMixin, Encoder):
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
        model_dtype: str | None = None,
        lora_config: LoRAAdaptation | None = None,
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
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
            lora_config: Optional LoRA adapter configuration.
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
            input_specification=specification,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
        )
        self._setup_language_keys(output_modality=EncoderOutputKeys.LANGUAGE.value)
        self.pooling_method = pooling_method
        self.attention_type = attention_type
        self.model_name = model_name
        self.max_token_len = max_token_len
        self.use_embeddings_only = use_embeddings_only
        self.lora_config = lora_config
        if self.use_embeddings_only and lora_config is not None and lora_config.enabled:
            raise ValueError("LoRA is not supported when use_embeddings_only=True.")
        if self.use_embeddings_only and self.pooling_method != PoolingMethod.NONE.value:
            raise ValueError(
                "use_embeddings_only=True is only compatible with pooling_method=PoolingMethod.NONE"
            )
        self._build_encoder()
        self.feature_dim = (
            self.encoder.embedding_dim
            if self.use_embeddings_only
            else self.config.hidden_size
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._has_cls_token = tokenizer.cls_token_id is not None
        self._num_prefix_tokens = 1 if self._has_cls_token else 0
        self.token_pooling_head = create_token_pooling_head(
            pooling_method=pooling_method,
            input_dimension=self.feature_dim,
            sequence_length=self.max_token_len,
            num_prefix_tokens=self._num_prefix_tokens,
        )
        self.output_dim = self.token_pooling_head.output_dim
        self.padding_dim = (
            self.max_token_len - self._num_prefix_tokens
            if pooling_method == PoolingMethod.NONE.value
            else 1
        )
        if frozen:
            super()._freeze_weights()
        self._apply_model_dtype()

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
                    self.model_name, attn_implementation=self.attention_type
                )
            else:
                self.encoder = AutoModel.from_config(
                    self.config, attn_implementation=self.attention_type
                )
            self.encoder = apply_lora_config(
                model=self.encoder,
                lora_config=self.lora_config,
                frozen=self.frozen,
            )

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode pre-tokenized text into language features.

        Args:
            inputs: Dict with text token IDs as (B, S) and optional padding mask.

        Returns:
            Dict with language features and padding mask.
        """
        text_input_ids, language_mask = self._extract_text_inputs(inputs=inputs)
        device = next(self.encoder.parameters()).device
        batch_size = text_input_ids.shape[0]
        text_input_ids, language_mask = self._pad_text_inputs(
            text_input_ids=text_input_ids,
            language_mask=language_mask,
            max_length=self.max_token_len,
        )
        attention_mask = self._build_attention_mask(
            language_mask=language_mask, text_input_ids=text_input_ids
        )
        if self.use_embeddings_only:
            features = self.token_pooling_head(
                self.encoder(text_input_ids.to(device)),
                padding_mask=~attention_mask.bool(),
            )
        else:
            encoder_inputs = {
                "input_ids": text_input_ids.to(device),
                "attention_mask": attention_mask.to(device),
            }
            outputs = self.encoder(**encoder_inputs, return_dict=True)
            if outputs.last_hidden_state is None:
                raise RuntimeError("last_hidden_state must be present in model output")
            features = self.token_pooling_head(
                outputs.last_hidden_state,
                padding_mask=~attention_mask.bool(),
            )
        padding_mask = self._build_output_padding_mask(
            attention_mask=attention_mask,
            pooling_method=self.pooling_method,
            batch_size=batch_size,
            device=features.device,
            num_prefix_tokens=self._num_prefix_tokens,
        )
        return {
            EncoderOutputKeys.LANGUAGE.value: features,
            self.padding_mask_name: padding_mask,
        }

    def validate_input_metadata(self, key: str, metadata: BaseMetadata) -> str | None:
        """Validate that input metadata is not camera metadata.

        Args:
            key: Observation key being validated.
            metadata: Metadata from the observation space.

        Returns:
            Error message if incompatible, None if valid.
        """
        if isinstance(metadata, CameraMetadata):
            return (
                f"LanguageEncoder cannot process image data for '{key}'. "
                f"Got CameraMetadata, expected tokenized text input."
            )
        return None

    def get_output_specification(self) -> list[FeatureMetadata]:
        """Get structured output specification with feature names and dimensions.

        Returns:
            List of FeatureMetadata with language features and padding mask.
        """
        language_dim = (
            (self.output_dim,) if isinstance(self.output_dim, int) else self.output_dim
        )
        padding_dim = (
            (self.padding_dim,)
            if isinstance(self.padding_dim, int)
            else self.padding_dim
        )
        return [
            FeatureMetadata(
                key=EncoderOutputKeys.LANGUAGE.value,
                feature_type=infer_feature_type(language_dim),
                dimension=language_dim,
            ),
            FeatureMetadata(
                key=self.padding_mask_name,
                feature_type=FeatureType.FLAT.value,
                dimension=padding_dim,
            ),
        ]

    def get_vocab_size(self) -> int:
        """Get the vocabulary size of the language encoder.

        Returns:
            Vocabulary size of the language model
        """
        return self.config.vocab_size
