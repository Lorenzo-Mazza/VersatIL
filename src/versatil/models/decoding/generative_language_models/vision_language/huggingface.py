"""HuggingFace-backed VLM base class."""

import abc

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, PretrainedConfig
from transformers.cache_utils import Cache

from versatil.models.adaptation.lora import (
    LoRAAdaptation,
    apply_lora_config,
)
from versatil.models.decoding.generative_language_models.base import (
    CausalLanguageModelOutput,
)
from versatil.models.decoding.generative_language_models.vision_language.base import (
    GenerativeVLM,
)
from versatil.models.encoding.encoders.constants import AttentionImplementation


class HuggingFaceGenerativeVLM(GenerativeVLM, abc.ABC):
    """Base class for VLMs loaded through HuggingFace image-text generation APIs."""

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        model_name: str,
        attention_type: str = AttentionImplementation.SDPA.value,
        model_dtype: str | None = None,
        max_text_length: int | None = None,
        lora_config: LoRAAdaptation | None = None,
    ):
        """Load or initialize a HuggingFace VLM component.

        Args:
            input_keys: RGB camera keys consumed by the VLM.
            pretrained: Whether to load pretrained HuggingFace weights.
            frozen: Whether to freeze all model weights.
            model_name: HuggingFace model identifier.
            attention_type: HuggingFace attention implementation.
            model_dtype: Optional precision string for model parameter dtype.
            max_text_length: Optional text sequence length. Defaults to the
                text config maximum position count.
            lora_config: Optional LoRA adapter configuration.
        """
        super().__init__(
            input_keys=input_keys,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
            max_text_length=max_text_length,
        )
        self.model_name = model_name
        self.lora_config = lora_config
        config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_name,
                attn_implementation=attention_type,
            )
        else:
            self.vlm = AutoModelForImageTextToText.from_config(
                config,
                attn_implementation=attention_type,
            )
        self.vlm = apply_lora_config(
            model=self.vlm,
            lora_config=lora_config,
            frozen=frozen,
        )
        self.image_size: int = config.vision_config.image_size
        self.hidden_dim: int = config.text_config.hidden_size
        self.num_image_tokens_per_camera: int = self._compute_num_image_tokens(
            config=config
        )
        self.max_text_length = (
            max_text_length
            if max_text_length is not None
            else config.text_config.max_position_embeddings
        )
        if frozen:
            super()._freeze_weights()
        self._apply_model_dtype()

    @abc.abstractmethod
    def _compute_num_image_tokens(self, config: PretrainedConfig) -> int:
        """Return the number of image tokens per camera for this VLM."""
        raise NotImplementedError

    def resize_token_embeddings(self, vocabulary_size: int) -> None:
        """Resize the HuggingFace VLM token embeddings."""
        self.vlm.resize_token_embeddings(vocabulary_size)
        self._get_language_model().config.vocab_size = vocabulary_size

    def forward_language_model(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | tuple[tuple[torch.Tensor, ...], ...] | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        output_hidden_states: bool = True,
    ) -> CausalLanguageModelOutput:
        """Run the nested language model and project hidden states to logits."""
        language_model = self._get_language_model()
        if cache_position is not None:
            language_output = language_model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        else:
            language_output = language_model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )
        last_hidden_state = language_output.last_hidden_state
        hidden_states = language_output.hidden_states
        if hidden_states is None:
            hidden_states = (last_hidden_state,)
        return CausalLanguageModelOutput(
            hidden_states=hidden_states,
            logits=self.vlm.get_output_embeddings()(last_hidden_state),
            past_key_values=language_output.past_key_values,
        )
