"""Base class for single-stream generative VLM encoders.

Provides the common encode flow shared by all HuggingFace VLMs that follow
the pattern: embed images → embed text → concatenate → run language model.
Subclasses only implement model-specific image embedding and the path to
the language model submodule.
"""

import abc

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig

from versatil.data.constants import RGB_CAMERAS, SampleKey
from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    EncoderOutputKeys,
)
from versatil.models.encoding.encoders.language_mixin import LanguageEncoderMixin
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, FeatureType
from versatil.models.layers.positional_encoding.rotary import RotaryPositionalEncoding


class GenerativeVLMEncoder(LanguageEncoderMixin, Encoder, abc.ABC):
    """Base for VLM encoders that fuse vision and language in a single LM pass.

    Handles VLM loading, dtype casting, text embedding, attention mask
    construction, output specification, input validation, and backbone
    accessors for interleaved decoders. Subclasses provide the
    model-specific image embedding and language model access path.
    """

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        model_name: str,
        attention_type: str = AttentionImplementation.SDPA.value,
        use_embeddings_only: bool = False,
        model_dtype: str | None = None,
        max_text_length: int | None = None,
    ):
        if isinstance(input_keys, str):
            input_keys = [input_keys]
        all_keys = list(input_keys) + [SampleKey.TOKENIZED_OBSERVATIONS.value]
        specification = EncoderInput(
            keys=all_keys,
            at_least_one_of_groups=[RGB_CAMERAS],
            required=[SampleKey.TOKENIZED_OBSERVATIONS.value],
            requires_tokenized=True,
        )
        super().__init__(
            input_specification=specification,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
        )
        self.camera_keys = [
            key for key in self.input_specification.keys if key in RGB_CAMERAS
        ]
        self._setup_language_keys(
            output_modality=EncoderOutputKeys.FUSED_RGB_LANGUAGE.value
        )
        self.model_name = model_name
        config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            self.vlm = AutoModel.from_pretrained(
                model_name, attn_implementation=attention_type
            )
        else:
            self.vlm = AutoModel.from_config(config, attn_implementation=attention_type)
        self.image_size: int = config.vision_config.image_size
        self.hidden_dim: int = config.text_config.hidden_size
        self.num_image_tokens_per_camera: int = self._compute_num_image_tokens(config)
        self.max_text_length: int = (
            max_text_length
            if max_text_length is not None
            else config.text_config.max_position_embeddings
        )
        self.use_embeddings_only = use_embeddings_only
        if frozen:
            super()._freeze_weights()
        self._apply_model_dtype()

    @property
    def total_image_tokens(self) -> int:
        """Total image tokens across all cameras."""
        return int(self.num_image_tokens_per_camera * len(self.camera_keys))

    @abc.abstractmethod
    def _compute_num_image_tokens(self, config: AutoConfig) -> int:
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
        """Map token IDs to dense embeddings via the LM's embedding table."""
        return self._get_language_model().get_input_embeddings()(token_ids)

    def _scale_language_embeddings(
        self, language_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Apply model-specific scaling to language embeddings.

        Override for custom scaling. Default is identity.
        """
        return language_embeddings

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Embed images and text, concatenate, and run the language model.

        Args:
            inputs: Camera images, tokenized text, and optional padding mask.

        Returns:
            Dict with fused sequential features and padding mask.
        """
        text_input_ids, language_mask = self._extract_text_inputs(inputs=inputs)
        batch_size = text_input_ids.shape[0]
        pad_length = (
            text_input_ids.shape[1]
            if self.use_embeddings_only
            else self.max_text_length
        )
        text_input_ids, language_mask = self._pad_text_inputs(
            text_input_ids=text_input_ids,
            language_mask=language_mask,
            max_length=pad_length,
        )
        image_embeddings, image_pad_masks = self._embed_images(
            inputs=inputs, batch_size=batch_size
        )
        language_embeddings = self._embed_language(text_input_ids)
        language_embeddings = self._scale_language_embeddings(language_embeddings)
        all_embeddings = image_embeddings + [language_embeddings]
        inputs_embeds = torch.cat(all_embeddings, dim=1)
        text_attention_mask = self._build_attention_mask(
            language_mask=language_mask, text_input_ids=text_input_ids
        )
        text_pad_mask = ~text_attention_mask.bool()
        full_padding_mask = torch.cat(image_pad_masks + [text_pad_mask], dim=1)
        full_attention_mask = (~full_padding_mask).to(torch.long)
        if self.use_embeddings_only:
            fused_features = inputs_embeds
        else:
            lm_outputs = self._get_language_model()(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
            )
            fused_features = lm_outputs.last_hidden_state
        return {
            EncoderOutputKeys.FUSED_RGB_LANGUAGE.value: fused_features,
            self.padding_mask_name: full_padding_mask,
        }

    def validate_input_metadata(self, key: str, metadata: BaseMetadata) -> str | None:
        if key in self.camera_keys:
            if not isinstance(metadata, CameraMetadata):
                return f"Expected CameraMetadata for '{key}', got {type(metadata).__name__}"
            if not metadata.is_rgb:
                return (
                    f"Expected 3-channel RGB for '{key}', "
                    f"got {metadata.channels} channels"
                )
        else:
            if isinstance(metadata, CameraMetadata):
                return (
                    f"{type(self).__name__} cannot process image data for '{key}'. "
                    f"Got CameraMetadata, expected tokenized text input."
                )
        return None

    def get_output_specification(self) -> list[FeatureMetadata]:
        total_sequence_length = self.total_image_tokens + self.max_text_length
        return [
            FeatureMetadata(
                key=EncoderOutputKeys.FUSED_RGB_LANGUAGE.value,
                feature_type=FeatureType.SEQUENTIAL.value,
                dimension=(total_sequence_length, self.hidden_dim),
            ),
            FeatureMetadata(
                key=self.padding_mask_name,
                feature_type=FeatureType.FLAT.value,
                dimension=(total_sequence_length,),
            ),
        ]

    def get_vocab_size(self) -> int:
        return self._get_language_model().config.vocab_size

    def get_backbone_layers(self) -> nn.ModuleList:
        """Return the LM transformer layers for interleaved decoding."""
        return self._get_language_model().layers

    def get_rotary_embedding(self) -> nn.Module:
        """Return the LM rotary positional encoding module."""
        return self._get_language_model().rotary_emb

    def get_backbone_hidden_dim(self) -> int:
        return self.hidden_dim

    def get_text_config(self) -> PretrainedConfig:
        return self._get_language_model().config

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
            rotary_embedding: The LM's rotary embedding module.
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
            when no mask is provided or when the mask has no masked entries.

        Raises:
            ValueError: If ``dtype`` is not a floating-point dtype.
        """
        if attention_mask is None:
            return None
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError(f"dtype must be floating point, got {dtype}.")
        boolean_mask = attention_mask.to(dtype=torch.bool)
        if not boolean_mask.any():
            return None
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
            rotary_embedding: The LM's rotary embedding module.
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
        cos, sin = GenerativeVLMEncoder.compute_rope(
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
            rotary_embedding: The LM's rotary embedding module.
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
        cos, sin = GenerativeVLMEncoder.compute_rope(
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
