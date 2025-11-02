
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from refactoring.data.constants import LANGUAGE_KEY
from refactoring.models.encoding.encoders.base import EncoderInput, EncoderOutput
from refactoring.models.encoding.encoders.constants import (
    AttentionImplementation,
    EncoderOutputKeys,
    FeatureExtractionMethod,
    LanguageEncoderType,
)
from refactoring.models.encoding.encoders.unconditional import Encoder
from refactoring.models.layers import LearnedAggregation


class LanguageEncoder(Encoder):
    """Language encoder using Transformers library."""
    def __init__(
            self,
            input_keys: str | list[str],
            pretrained: bool,
            frozen: bool,
            feature_extraction_method: str,
            model_name: str = LanguageEncoderType.BERT_BASE.value,
            max_length: int = 77,  # CLIP-style default
            add_special_tokens: bool = True,
            attention_type: str = AttentionImplementation.SDPA.value,
    ):
        """
        Args:
            input_keys: Keys for text input in input dict
            pretrained: Whether to use pretrained weights
            frozen: Whether to freeze backbone weights
            feature_extraction_method: How to extract features from transformer output
            model_name: Model identifier from LanguageEncoderType
            max_length: Maximum sequence length for tokenization
            add_special_tokens: Whether to add [CLS], [SEP] tokens
            attention_type: Attention implementation to use
        """
        specification = EncoderInput(keys=input_keys, required=[LANGUAGE_KEY])
        super().__init__(input_specification=specification, pretrained=pretrained, frozen=frozen)
        self.language_key = LANGUAGE_KEY
        self.feature_extraction_method = feature_extraction_method
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.attention_type = attention_type
        self.model_name = model_name
        self._build_encoder()
        self.feature_dim = self.encoder.config.hidden_size
        self.pooling_head: LearnedAggregation | None = None
        self._setup_feature_extractor()
        if frozen:
            super()._freeze_weights()



    def _build_encoder(self):
        """Build language encoder and tokenizer."""
        self.encoder = AutoModel.from_pretrained(self.model_name, device_map="auto",
                                                 attn_implementation=self.attention_type, use_safetensors=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def _setup_feature_extractor(self):
        """Setup feature extraction layer based on configuration."""
        if self.feature_extraction_method == FeatureExtractionMethod.LEARNED_AGGREGATION.value:
            self.pooling_head = LearnedAggregation(self.feature_dim).to(self.encoder.device)


    def _extract_features(self, outputs: BaseModelOutput) -> torch.Tensor:
        """Extract features from transformer outputs based on specified method."""
        if outputs.last_hidden_state is None:
            raise RuntimeError("last_hidden_state must be present in model output")
        if self.feature_extraction_method == FeatureExtractionMethod.CLS_TOKEN.value:
            return outputs.last_hidden_state[:, 0]  # CLS token
        elif self.feature_extraction_method == FeatureExtractionMethod.AVERAGE_PATCH_TOKENS.value:
            return outputs.last_hidden_state[:, 1:].mean(dim=1)  # GAP on tokens (exclude CLS)
        elif self.feature_extraction_method == FeatureExtractionMethod.LEARNED_AGGREGATION.value:
            if self.pooling_head is None:
                raise RuntimeError("pooling_head must be initialized for LEARNED_AGGREGATION")
            result: torch.Tensor = self.pooling_head(outputs.last_hidden_state[:, 1:])  # Learned agg on tokens (exclude CLS)
            return result
        else:
            raise ValueError(f"Unsupported feature extraction method: {self.feature_extraction_method}")



    def forward(self, inputs: dict[str, list[list[str]] | list[str]]) -> dict[str, torch.Tensor]:  # type: ignore[override]
        """
        Forward pass through language encoder.

        Args:
            inputs: Dict with text input (either a list (batch size long) of strings or a list of list of strings
             (batch size long, each containing time steps). Strings are not tokenized yet.

        Returns:
            Encoded text features, wrapped in a dictionary indexed by the `LANGUAGE_FEATURES` constant.
        """
        language_instruction = inputs[self.language_key]
        T = None
        has_time = False
        B = len(language_instruction)
        if isinstance(language_instruction, list) and language_instruction and isinstance(language_instruction[0], list):
            T = len(language_instruction[0])
            language_instruction = [time for batch in language_instruction for time in batch]
            has_time = True

        encoded = self.tokenizer(
            language_instruction,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt"
        ).to(self.encoder.device)
        outputs = self.encoder(**encoded, return_dict=True)
        features = self._extract_features(outputs)
        if has_time:
            if T is None:
                raise RuntimeError("T must be set when has_time is True")
            features = features.reshape(B, T, -1)  # Batch, Time, Features
        return {EncoderOutputKeys.LANGUAGE.value: features}


    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.LANGUAGE.value],
            dimensions={EncoderOutputKeys.LANGUAGE.value: self.feature_dim},
        )
