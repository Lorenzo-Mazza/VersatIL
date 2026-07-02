"""Base class for vision-language models used by VLA decoders."""

import abc
import math

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.cache_utils import Cache

from versatil.data.constants import CameraModality, SampleKey
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.decoding.generative_language_models.base import (
    CausalLanguageModelOutput,
    GenerativeLanguageModel,
)
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.encoding.encoders.language_mixin import LanguageEncoderMixin
from versatil.models.input_specification import InputSpecification
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding


class GenerativeVLM(LanguageEncoderMixin, GenerativeLanguageModel, abc.ABC):
    """Base for VLM components that fuse vision and language in one language-model pass."""

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        model_dtype: str | None = None,
        max_text_length: int | None = None,
    ):
        """Initialize common VLM input wiring and runtime options.

        Args:
            input_keys: RGB camera keys consumed by the VLM. Tokenized text and
                padding-mask keys are added by the base class.
            pretrained: Whether the concrete subclass loads pretrained weights.
            frozen: Whether all model weights are frozen.
            model_dtype: Optional precision string for model parameter dtype.
            max_text_length: Optional text sequence length used when running the
                language model. ``None`` lets the concrete subclass set a
                model-specific default.
        """
        if isinstance(input_keys, str):
            input_keys = [input_keys]
        token_keys = {
            SampleKey.TOKENIZED_OBSERVATIONS.value,
            SampleKey.IS_PAD_OBSERVATION.value,
        }
        camera_keys = [key for key in input_keys if key not in token_keys]
        all_keys = camera_keys + [
            SampleKey.TOKENIZED_OBSERVATIONS.value,
            SampleKey.IS_PAD_OBSERVATION.value,
        ]
        specification = InputSpecification(
            keys=all_keys,
            required_camera_modalities=[CameraModality.RGB],
            required=[SampleKey.TOKENIZED_OBSERVATIONS.value],
            requires_tokenized=True,
        )
        super().__init__(
            input_specification=specification,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
        )
        self.camera_keys = camera_keys
        self.is_multi_camera = len(camera_keys) > 1
        self._setup_language_keys(
            output_modality=EncoderOutputKeys.FUSED_RGB_LANGUAGE.value
        )
        if max_text_length is not None:
            self.max_text_length = max_text_length

    @property
    def total_image_tokens(self) -> int:
        """Total image tokens across all cameras."""
        return int(self.num_image_tokens_per_camera * len(self.camera_keys))

    def _get_image_token_grid(self) -> tuple[int, int] | None:
        """Return a square image-token grid when the token count allows it.

        Returns:
            ``(height, width)`` when image tokens form a square grid, otherwise
            ``None``. Attribution map conversion can still infer square grids
            from the captured token count and raises when the shape is
            ambiguous.
        """
        grid_size = math.isqrt(self.num_image_tokens_per_camera)
        if grid_size * grid_size != self.num_image_tokens_per_camera:
            return None
        return grid_size, grid_size

    @abc.abstractmethod
    def _compute_num_image_tokens(self, config: PretrainedConfig) -> int:
        """Return the number of image tokens per camera for this VLM."""
        raise NotImplementedError

    @abc.abstractmethod
    def _embed_images(
        self, inputs: dict[str, torch.Tensor], batch_size: int
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Embed camera images into token sequences.

        Args:
            inputs: Dict with camera images as (B, C, H, W) per camera key.
            batch_size: Batch size.

        Returns:
            (image_embeddings, image_pad_masks) — lists of tensors, one per camera
            or one combined, ready for concatenation with language embeddings.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_language_model(self) -> nn.Module:
        """Return the language model submodule (e.g. ``vlm.language_model`` or ``vlm.text_model``)."""
        raise NotImplementedError

    def _embed_language(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to dense embeddings with the language-model embedding table."""
        return self._get_language_model().get_input_embeddings()(token_ids)

    def embed_input_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed language-vocabulary token IDs with the VLM language tower."""
        return self._embed_language(token_ids=token_ids)

    def forward_language_model(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | tuple[tuple[torch.Tensor, ...], ...] | None = None,
        use_cache: bool = False,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool = True,
    ) -> CausalLanguageModelOutput:
        """Run the VLM language tower over caller-provided token inputs.

        Args:
            input_ids: Optional token IDs with shape ``(B, S)``.
            inputs_embeds: Optional token embeddings with shape ``(B, S, D)``.
            attention_mask: Optional language-model attention mask.
            past_key_values: Optional cached key/value tensors.
            use_cache: Whether to return/update cached key/value tensors.
            cache_position: Optional HuggingFace KV-cache slots for the tokens
                in this call. During cached decoding, if the prefix has length
                ``P``, the next token uses cache slot ``P`` so its key/value is
                appended after the prefix.
            position_ids: Optional positions for the language model positional
                encoding, with shape ``(B, S)``. These should count real tokens,
                not padding: ``[PAD, PAD, t0, t1]`` should pass
                ``[0, 0, 0, 1]`` so ``t0`` and ``t1`` get positions ``0`` and
                ``1``.
            output_hidden_states: Whether to return hidden states.

        Returns:
            Causal language-model output with logits shape ``(B, S, V)``.
        """
        language_model_inputs = {
            "input_ids": input_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        if cache_position is not None:
            language_model_inputs["cache_position"] = cache_position
        if position_ids is not None:
            language_model_inputs["position_ids"] = position_ids
        return self._get_language_model()(**language_model_inputs)

    def _scale_language_embeddings(
        self, language_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Apply model-specific scaling to language embeddings.

        Override for custom scaling. Default is identity.
        """
        return language_embeddings

    def _merge_image_language_embeddings(
        self,
        image_embeddings: list[torch.Tensor],
        image_pad_masks: list[torch.Tensor],
        language_embeddings: torch.Tensor,
        language_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Order image and language token embeddings into one sequence.

        Default places image tokens before the language tokens (PaliGemma and
        SmolVLM convention). Override for models with a different multimodal
        token order.

        Args:
            image_embeddings: Per-camera image token embeddings, each
                ``(B, image_token_count, D)``.
            image_pad_masks: Matching boolean padding masks, ``True`` = padded.
            language_embeddings: Language token embeddings ``(B, S, D)``.
            language_pad_mask: Language padding mask ``(B, S)``.

        Returns:
            Merged embeddings ``(B, total_token_count, D)`` and padding mask
            ``(B, total_token_count)``.
        """
        merged_embeddings = torch.cat(
            image_embeddings + [language_embeddings],
            dim=1,
        )
        merged_padding_mask = torch.cat(
            image_pad_masks + [language_pad_mask],
            dim=1,
        )
        return merged_embeddings, merged_padding_mask

    def _assemble_multimodal_embeddings(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed images and text and merge them into one input sequence.

        Args:
            inputs: Camera images, tokenized text, and optional padding mask.

        Returns:
            Input embeddings ``(B, S, D)`` and padding mask ``(B, S)`` where
            ``True`` marks padding. The embeddings have not been run through
            the language-model transformer.
        """
        text_input_ids, language_mask = self._extract_text_inputs(inputs=inputs)
        batch_size = text_input_ids.shape[0]
        text_input_ids, language_mask = self._pad_text_inputs(
            text_input_ids=text_input_ids,
            language_mask=language_mask,
            max_length=self.max_text_length,
        )
        image_embeddings, image_pad_masks = self._embed_images(
            inputs=inputs, batch_size=batch_size
        )
        language_embeddings = self._embed_language(text_input_ids)
        language_embeddings = self._scale_language_embeddings(language_embeddings)
        text_attention_mask = self._build_attention_mask(
            language_mask=language_mask, text_input_ids=text_input_ids
        )
        text_pad_mask = ~text_attention_mask.bool()
        return self._merge_image_language_embeddings(
            image_embeddings=image_embeddings,
            image_pad_masks=image_pad_masks,
            language_embeddings=language_embeddings,
            language_pad_mask=text_pad_mask,
        )

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Embed images and text, concatenate, and run the language model.

        Args:
            inputs: Camera images, tokenized text, and optional padding mask.

        Returns:
            Dict with fused sequential features and padding mask.
        """
        inputs_embeds, full_padding_mask = self._assemble_multimodal_embeddings(
            inputs=inputs
        )
        full_attention_mask = (~full_padding_mask).to(torch.long)
        lm_outputs = self._get_language_model()(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
        )
        fused_features = lm_outputs.last_hidden_state
        return {
            EncoderOutputKeys.FUSED_RGB_LANGUAGE.value: fused_features,
            self.padding_mask_name: full_padding_mask,
        }

    def _validate_inputs(self, inputs: dict[str, torch.Tensor]) -> None:
        """Validate VLM inputs before optional temporal flattening."""
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"VLM input '{key}' must be a torch.Tensor, "
                    f"got {type(value).__name__}."
                )

    def _flatten_temporal(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], int, int] | None:
        """Merge a leading observation-horizon dimension when present."""
        self._validate_inputs(inputs=inputs)
        text_input = inputs.get(self.language_key)
        if text_input is None or text_input.dim() < 3:
            return None
        batch_size = text_input.shape[0]
        temporal_length = text_input.shape[1]
        flattened = {}
        for key, tensor in inputs.items():
            if tensor.dim() < 3:
                flattened[key] = tensor
                continue
            if tensor.shape[0] != batch_size or tensor.shape[1] != temporal_length:
                raise ValueError(
                    f"VLM input '{key}' has leading shape "
                    f"{tuple(tensor.shape[:2])}, expected "
                    f"({batch_size}, {temporal_length})."
                )
            flattened[key] = tensor.reshape(
                batch_size * temporal_length, *tensor.shape[2:]
            )
        return flattened, batch_size, temporal_length

    @staticmethod
    def _unflatten_temporal(
        outputs: dict[str, torch.Tensor],
        batch_size: int,
        temporal_length: int,
    ) -> dict[str, torch.Tensor]:
        """Restore a leading observation-horizon dimension in VLM outputs."""
        return {
            key: tensor.reshape(batch_size, temporal_length, *tensor.shape[1:])
            for key, tensor in outputs.items()
        }

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run the VLM on raw inputs with or without an observation horizon."""
        temporal = self._flatten_temporal(inputs=inputs)
        if temporal is None:
            return self.encode(inputs=inputs)
        flattened, batch_size, temporal_length = temporal
        outputs = self.encode(inputs=flattened)
        return self._unflatten_temporal(
            outputs=outputs,
            batch_size=batch_size,
            temporal_length=temporal_length,
        )

    def build_prefix(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build VLM prefix input embeddings for VLA decoder execution.

        The prefix is the pre-transformer multimodal input sequence. The
        consuming decoder runs the language-model layers over it exactly once
        itself (interleaved expert layers or causal-LM prefill), so the prefix
        must not already be language-model output states.
        """
        temporal = self._flatten_temporal(inputs=inputs)
        if temporal is None:
            return self._assemble_multimodal_embeddings(inputs=inputs)
        flattened, batch_size, temporal_length = temporal
        prefix, padding_mask = self._assemble_multimodal_embeddings(inputs=flattened)
        sequence_length, hidden_dim = prefix.shape[1], prefix.shape[2]
        # (B*T, S, D) -> (B, T*S, D)
        prefix = prefix.reshape(
            batch_size, temporal_length * sequence_length, hidden_dim
        )
        padding_mask = padding_mask.reshape(
            batch_size, temporal_length * sequence_length
        )
        return prefix, padding_mask

    def validate_input_metadata(self, key: str, metadata: BaseMetadata) -> str | None:
        """Validate metadata for VLM camera and tokenized-text inputs."""
        if key in self.camera_keys:
            if not isinstance(metadata, CameraMetadata):
                return f"Expected CameraMetadata for '{key}', got {type(metadata).__name__}"
        else:
            if isinstance(metadata, CameraMetadata):
                return (
                    f"{type(self).__name__} cannot process image data for '{key}'. "
                    f"Got CameraMetadata, expected tokenized text input."
                )
        return None

    def get_vocab_size(self) -> int:
        """Return the VLM language model vocabulary size."""
        return self._get_language_model().config.vocab_size

    def get_backbone_layers(self) -> nn.ModuleList:
        """Return the language-model transformer layers for interleaved decoding."""
        return self._get_language_model().layers

    @property
    def layers(self) -> nn.ModuleList:
        """Language-model transformer layers used by VLA backbones."""
        return self.get_backbone_layers()

    def get_rotary_embedding(self) -> nn.Module:
        """Return the language-model rotary positional encoding module."""
        return self._get_language_model().rotary_emb

    @property
    def rotary_embedding(self) -> nn.Module:
        """Rotary embedding module used by the language-model transformer layers."""
        return self.get_rotary_embedding()

    def get_backbone_hidden_dim(self) -> int:
        """Return the VLM language model hidden dimension."""
        return self.hidden_dim

    def get_text_config(self) -> PretrainedConfig:
        """Return the VLM language model configuration."""
        return self._get_language_model().config

    @property
    def text_config(self) -> PretrainedConfig:
        """Text-model config for constructing VLA expert layers."""
        return self.get_text_config()

    @staticmethod
    def extract_key_value(
        vlm_layer: nn.Module,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project hidden states to key and value through a VLM layer's projections.

        Does not apply RoPE — use ``extract_query_key_value`` when positional
        information is needed.

        Args:
            vlm_layer: Pretrained VLM transformer layer.
            hidden_states: (B, S, D).

        Returns:
            (keys, values) each (B, S, key_value_dimension).
        """
        normalized = vlm_layer.input_layernorm(hidden_states)
        return (
            vlm_layer.self_attn.k_proj(normalized),
            vlm_layer.self_attn.v_proj(normalized),
        )

    @staticmethod
    def compute_rope(
        rotary_embedding: nn.Module,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (cos, sin) RoPE components for given positions.

        Args:
            rotary_embedding: The language-model rotary embedding module.
            hidden_states: Tensor whose dtype/device to match.
            position_ids: Position indices (B, S).

        Returns:
            (cos, sin) each (B, 1, S, head_dim) for head broadcast.
        """
        cos, sin = rotary_embedding(hidden_states, position_ids)
        # (B, S, head_dim) → (B, 1, S, head_dim) for head broadcast
        return cos.unsqueeze(1), sin.unsqueeze(1)

    @staticmethod
    def build_additive_attention_mask(
        attention_mask: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Convert a bool mask to the additive mask expected by HF decoder layers.

        Args:
            attention_mask: Boolean attention mask where ``True`` means masked.
                Accepts any broadcast-compatible shape used by HF decoder layers,
                typically ``(B, 1, Q, K)``.
            dtype: Floating dtype for the returned additive mask.

        Returns:
            Additive attention mask with ``0`` for valid locations and
            ``torch.finfo(dtype).min`` for masked locations. Returns ``None``
            only when no mask is provided. An all-visible mask is returned as
            explicit zeros: HF decoder layers fall back to causal attention
            when given ``None``, which would silently override the intended
            bidirectional structure.

        Raises:
            ValueError: If ``dtype`` is not a floating-point dtype.
        """
        if attention_mask is None:
            return None
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError(f"dtype must be floating point, got {dtype}.")
        boolean_mask = attention_mask.to(dtype=torch.bool)
        additive_mask = torch.zeros(
            boolean_mask.shape,
            dtype=dtype,
            device=boolean_mask.device,
        )
        return additive_mask.masked_fill(boolean_mask, torch.finfo(dtype).min)

    @staticmethod
    def extract_key_value_with_rope(
        vlm_layer: nn.Module,
        hidden_states: torch.Tensor,
        rotary_embedding: nn.Module,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project hidden states to K/V with RoPE applied to keys.

        Args:
            vlm_layer: Pretrained VLM transformer layer.
            hidden_states: (B, S, D).
            rotary_embedding: The language-model rotary embedding module.
            position_ids: (B, S) position indices.

        Returns:
            (keys, values) — keys have RoPE applied and are flattened to
            (B, S, key_value_dimension), values are flat (B, S, key_value_dimension).
        """
        normalized = vlm_layer.input_layernorm(hidden_states)
        attention = vlm_layer.self_attn
        batch_size, sequence_length, _ = normalized.shape
        head_dimension = attention.head_dim
        number_of_key_value_heads = attention.config.num_key_value_heads
        keys_flat = attention.k_proj(normalized)
        values_flat = attention.v_proj(normalized)
        keys_headed = keys_flat.view(
            batch_size, sequence_length, number_of_key_value_heads, head_dimension
        ).transpose(1, 2)
        cos, sin = GenerativeVLM.compute_rope(
            rotary_embedding=rotary_embedding,
            hidden_states=keys_headed,
            position_ids=position_ids[:, :sequence_length],
        )
        keys_headed = RotaryPositionalEncoding.apply_rotation_half(
            tensor=keys_headed, sine=sin, cosine=cos
        )
        keys_with_rope = (
            keys_headed.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, -1)
        )
        return keys_with_rope, values_flat

    @staticmethod
    def extract_query_key_value(
        vlm_layer: nn.Module,
        hidden_states: torch.Tensor,
        rotary_embedding: nn.Module,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project hidden states to Q/K/V and apply rotary positional encoding.

        Works for any HF decoder layer that exposes ``input_layernorm``,
        ``self_attn.{q,k,v}_proj``, ``head_dim``, and
        ``config.num_{attention,key_value}_heads``.

        Args:
            vlm_layer: Pretrained VLM transformer layer.
            hidden_states: (B, S, D).
            rotary_embedding: The language-model rotary embedding module.
            position_ids: (B, S) position indices.

        Returns:
            (query, key, value) with RoPE on Q and K.
            Shapes: Q (B, H, S, D_head), K (B, KV_H, S, D_head), V same as K.
        """
        normalized = vlm_layer.input_layernorm(hidden_states)
        attention = vlm_layer.self_attn
        batch_size, sequence_length, _ = normalized.shape
        head_dimension = attention.head_dim
        number_of_heads = attention.config.num_attention_heads
        number_of_key_value_heads = attention.config.num_key_value_heads
        query = (
            attention.q_proj(normalized)
            .view(batch_size, sequence_length, number_of_heads, head_dimension)
            .transpose(1, 2)
        )
        key = (
            attention.k_proj(normalized)
            .view(
                batch_size, sequence_length, number_of_key_value_heads, head_dimension
            )
            .transpose(1, 2)
        )
        value = (
            attention.v_proj(normalized)
            .view(
                batch_size, sequence_length, number_of_key_value_heads, head_dimension
            )
            .transpose(1, 2)
        )
        cos, sin = GenerativeVLM.compute_rope(
            rotary_embedding=rotary_embedding,
            hidden_states=value,
            position_ids=position_ids[:, :sequence_length],
        )
        query = RotaryPositionalEncoding.apply_rotation_half(
            tensor=query, sine=sin, cosine=cos
        )
        key = RotaryPositionalEncoding.apply_rotation_half(
            tensor=key, sine=sin, cosine=cos
        )
        return query, key, value

    @staticmethod
    def apply_residual_feedforward(
        vlm_layer: nn.Module,
        vlm_residual: torch.Tensor,
        vlm_attention_output: torch.Tensor,
    ) -> torch.Tensor:
        """Complete a transformer layer after externally computed attention.

        Called by interleaved decoders that run joint attention outside the VLM
        layer and need it to finish: O-projection → normalization → residual
        add → feedforward → residual add.

        Handles both Llama-style layers (single post-attention norm) and
        Gemma2-style layers (sandwich norms around sublayer outputs).

        Args:
            vlm_layer: Pretrained VLM transformer layer.
            vlm_residual: Hidden states before attention (B, S, D).
            vlm_attention_output: Raw attention output before O-projection (B, S, H*D_head).

        Returns:
            Updated hidden states (B, S, D).
        """
        hidden_states = vlm_layer.self_attn.o_proj(vlm_attention_output)
        if hasattr(vlm_layer, "pre_feedforward_layernorm"):
            hidden_states = vlm_layer.post_attention_layernorm(hidden_states)
            hidden_states = vlm_residual + hidden_states
            residual = hidden_states
            hidden_states = vlm_layer.pre_feedforward_layernorm(hidden_states)
            hidden_states = vlm_layer.mlp(hidden_states)
            hidden_states = vlm_layer.post_feedforward_layernorm(hidden_states)
        else:
            hidden_states = vlm_residual + hidden_states
            residual = hidden_states
            hidden_states = vlm_layer.post_attention_layernorm(hidden_states)
            hidden_states = vlm_layer.mlp(hidden_states)
        return residual + hidden_states
