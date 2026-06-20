"""Prismatic VLM component for VLA decoders."""

import json
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers.cache_utils import Cache

from versatil.data.constants import (
    CLIP_RGB_MEAN,
    CLIP_RGB_STD,
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    SIGLIP_RGB_MEAN,
    SIGLIP_RGB_STD,
)
from versatil.models.adaptation.lora import LoRAAdaptation, apply_lora_config
from versatil.models.decoding.generative_language_models.base import (
    CausalLanguageModelOutput,
)
from versatil.models.decoding.generative_language_models.constants import (
    PRISMATIC_CHECKPOINT_FILENAME,
    PRISMATIC_CONFIG_FILENAME,
    PRISMATIC_LLM_BACKBONES,
    PRISMATIC_PAD_TO_MULTIPLE_OF,
    PRISMATIC_REPOSITORY_ID,
    PRISMATIC_VISION_BACKBONES,
    PRISMATIC_VISION_CHECKPOINT_KEY_RENAMES,
    PRISMATIC_VISION_IMAGE_SIZES,
    PrismaticLLMBackboneType,
    PrismaticModelType,
    PrismaticVisionBackboneType,
)
from versatil.models.decoding.generative_language_models.vision_language.base import (
    GenerativeVLM,
)
from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    FlatBackboneType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.image_mixin import resize_to_target_size
from versatil.models.encoding.encoders.rgb.flat import FlatRGBEncoder

type PrismaticModelConfigValue = str | int | float | bool | None


class PrismaticVLM(GenerativeVLM):
    """Raw Prismatic VLM checkpoint loader for interleaved VLA decoders."""

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        model_name: str = PrismaticModelType.PRISM_DINOSIGLIP_224PX_7B.value,
        repository_id: str = PRISMATIC_REPOSITORY_ID,
        attention_type: str = AttentionImplementation.SDPA.value,
        model_dtype: str | None = None,
        max_text_length: int | None = None,
        lora_config: LoRAAdaptation | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        """Load or initialize a raw Prismatic VLM.

        Args:
            input_keys: RGB camera keys consumed by the VLM.
            pretrained: Whether to load the raw Prismatic checkpoint.
            frozen: Whether to freeze all model weights.
            model_name: Prismatic checkpoint folder name, or a local checkpoint
                directory containing ``config.json`` and ``checkpoints``.
            repository_id: HuggingFace repository containing raw Prismatic
                checkpoint folders.
            attention_type: HuggingFace attention implementation for the
                language model.
            model_dtype: Optional precision string for model parameter dtype.
            max_text_length: Optional text sequence length. Defaults to the
                raw Prismatic ``llm_max_length`` field.
            lora_config: Optional LoRA adapter configuration for the language
                model.
            gradient_checkpointing: Whether to enable activation checkpointing
                in the language model during training.
        """
        super().__init__(
            input_keys=input_keys,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
            max_text_length=max_text_length,
        )
        self.model_name = model_name
        self.repository_id = repository_id
        self.lora_config = lora_config
        self.gradient_checkpointing = gradient_checkpointing
        model_config = self._load_model_config(
            model_name=model_name,
            repository_id=repository_id,
        )
        self.prismatic_config = model_config
        self.vision_backbone_id = str(model_config["vision_backbone_id"])
        self.llm_backbone_id = str(model_config["llm_backbone_id"])
        self.arch_specifier = str(model_config["arch_specifier"])
        self.vision_backbone_type = self._resolve_vision_backbone_type(
            vision_backbone_id=self.vision_backbone_id
        )
        self.vision_backbone_types = PRISMATIC_VISION_BACKBONES[
            self.vision_backbone_type
        ]
        self.image_size = PRISMATIC_VISION_IMAGE_SIZES[self.vision_backbone_type]
        # Raw DinoSigLIP checkpoints do not store the frozen vision towers, so
        # the timm towers must load their own pretrained weights.
        self.vision_encoders = self._build_vision_encoders(pretrained=pretrained)
        self.num_image_tokens_per_camera = self._resolve_num_image_tokens()
        self.vision_embedding_dimension = sum(
            int(encoder.feature_dim) for encoder in self.vision_encoders
        )
        self.language_model = self._build_language_model(
            llm_backbone_id=self.llm_backbone_id,
            attention_type=attention_type,
        )
        self.hidden_dim = int(self.language_model.config.hidden_size)
        self.projector = self._build_projector(
            arch_specifier=self.arch_specifier,
            vision_dimension=self.vision_embedding_dimension,
            language_dimension=self.hidden_dim,
        )
        self.max_text_length = (
            max_text_length
            if max_text_length is not None
            else int(model_config["llm_max_length"])
        )
        if pretrained:
            checkpoint_path = self._resolve_checkpoint_path(
                model_name=model_name,
                repository_id=repository_id,
            )
            self._load_prismatic_checkpoint(checkpoint_path=checkpoint_path)
        if lora_config is not None and lora_config.enabled:
            # PEFT mutates custom modules in place; assigning the returned wrapper
            # here would recursively register this module as its own child.
            apply_lora_config(model=self, lora_config=lora_config, frozen=frozen)
        if gradient_checkpointing:
            self.language_model.gradient_checkpointing_enable()
            self.language_model.config.use_cache = False
        if frozen:
            super()._freeze_weights()
        self._apply_model_dtype()

    @staticmethod
    def _resolve_config_path(
        model_name: str,
        repository_id: str,
    ) -> Path:
        """Resolve a raw Prismatic config file.

        Args:
            model_name: Prismatic checkpoint folder or local directory.
            repository_id: HuggingFace repository id.

        Returns:
            Local path to ``config.json``.
        """
        local_path = Path(model_name)
        if local_path.is_dir():
            return local_path / PRISMATIC_CONFIG_FILENAME
        downloaded = hf_hub_download(
            repo_id=repository_id,
            filename=f"{model_name}/{PRISMATIC_CONFIG_FILENAME}",
        )
        return Path(downloaded)

    @classmethod
    def _load_model_config(
        cls,
        model_name: str,
        repository_id: str,
    ) -> dict[str, PrismaticModelConfigValue]:
        """Read the Prismatic model section from ``config.json``.

        Args:
            model_name: Prismatic checkpoint folder or local directory.
            repository_id: HuggingFace repository id.

        Returns:
            Model configuration dictionary.
        """
        config_path = cls._resolve_config_path(
            model_name=model_name,
            repository_id=repository_id,
        )
        with config_path.open() as config_file:
            raw_config = json.load(config_file)
        return raw_config["model"]

    @staticmethod
    def _resolve_checkpoint_path(
        model_name: str,
        repository_id: str,
    ) -> Path:
        """Resolve a raw Prismatic checkpoint file.

        Args:
            model_name: Prismatic checkpoint folder or local directory.
            repository_id: HuggingFace repository id.

        Returns:
            Local path to ``latest-checkpoint.pt``.
        """
        local_path = Path(model_name)
        if local_path.is_dir():
            return local_path / PRISMATIC_CHECKPOINT_FILENAME
        downloaded = hf_hub_download(
            repo_id=repository_id,
            filename=f"{model_name}/{PRISMATIC_CHECKPOINT_FILENAME}",
        )
        return Path(downloaded)

    @staticmethod
    def _resolve_vision_backbone_type(
        vision_backbone_id: str,
    ) -> PrismaticVisionBackboneType:
        """Validate and return a Prismatic vision backbone id."""
        supported_values = [
            model_type.value for model_type in PrismaticVisionBackboneType
        ]
        if vision_backbone_id not in supported_values:
            raise ValueError(
                f"Unsupported Prismatic vision_backbone_id '{vision_backbone_id}'. "
                f"Supported values: {supported_values}."
            )
        return PrismaticVisionBackboneType(vision_backbone_id)

    def _build_vision_encoders(self, pretrained: bool) -> nn.ModuleList:
        """Build Prismatic timm vision towers as flat RGB encoders.

        Args:
            pretrained: Whether timm should initialize towers from pretrained
                weights.

        Returns:
            Vision encoder towers with Prismatic patch-token settings.
        """
        encoders = []
        for backbone_type in self.vision_backbone_types:
            encoders.append(
                FlatRGBEncoder(
                    input_keys=self.camera_keys,
                    pretrained=pretrained,
                    frozen=False,
                    pooling_method=PoolingMethod.NONE.value,
                    backbone=backbone_type.value,
                    image_size=self.image_size,
                    intermediate_layer_index=-2,
                    model_dtype=None,
                    lora_config=None,
                )
            )
        return nn.ModuleList(encoders)

    def _resolve_num_image_tokens(self) -> int:
        """Return the shared patch-token count across Prismatic vision towers."""
        patch_counts = [
            int(encoder.backbone.patch_embed.num_patches)
            for encoder in self.vision_encoders
        ]
        if len(set(patch_counts)) != 1:
            raise ValueError(
                "Prismatic vision towers must produce the same number of patch "
                f"tokens, got {patch_counts}."
            )
        return patch_counts[0]

    @staticmethod
    def _resolve_llm_model_name(llm_backbone_id: str) -> str:
        """Map a Prismatic LLM id to a HuggingFace model id."""
        supported_values = [model_type.value for model_type in PrismaticLLMBackboneType]
        if llm_backbone_id not in supported_values:
            raise ValueError(
                f"Unsupported Prismatic llm_backbone_id '{llm_backbone_id}'. "
                f"Supported values: {supported_values}."
            )
        llm_backbone_type = PrismaticLLMBackboneType(llm_backbone_id)
        return PRISMATIC_LLM_BACKBONES[llm_backbone_type]

    @staticmethod
    def _pad_to_multiple(value: int, multiple: int) -> int:
        """Round ``value`` up to a multiple."""
        return multiple * ((value + multiple - 1) // multiple)

    def _build_language_model(
        self,
        llm_backbone_id: str,
        attention_type: str,
    ) -> nn.Module:
        """Build the HuggingFace causal language model used inside Prismatic.

        Args:
            llm_backbone_id: Prismatic LLM backbone id.
            attention_type: HuggingFace attention implementation.

        Returns:
            Causal language model with Prismatic pad-token resizing applied.
        """
        language_model_name = self._resolve_llm_model_name(
            llm_backbone_id=llm_backbone_id
        )
        config = AutoConfig.from_pretrained(language_model_name)
        config.output_hidden_states = True
        language_model = AutoModelForCausalLM.from_config(
            config,
            attn_implementation=attention_type,
        )
        pad_token_id = int(config.vocab_size)
        language_model.config.pad_token_id = pad_token_id
        language_model.resize_token_embeddings(
            self._pad_to_multiple(
                value=pad_token_id + 1,
                multiple=PRISMATIC_PAD_TO_MULTIPLE_OF,
            ),
            pad_to_multiple_of=PRISMATIC_PAD_TO_MULTIPLE_OF,
        )
        language_model.config.use_cache = False
        return language_model

    @staticmethod
    def _build_projector(
        arch_specifier: str,
        vision_dimension: int,
        language_dimension: int,
    ) -> nn.Module:
        """Build the visual projector configured by Prismatic.

        Args:
            arch_specifier: Raw Prismatic architecture specifier.
            vision_dimension: Visual patch feature dimension.
            language_dimension: Language-model hidden dimension.

        Returns:
            Visual projector module.
        """
        if arch_specifier == "linear":
            return nn.Sequential(
                OrderedDict(
                    [
                        (
                            "projector",
                            nn.Linear(
                                vision_dimension,
                                language_dimension,
                                bias=True,
                            ),
                        )
                    ]
                )
            )
        if arch_specifier.endswith("fused-gelu-mlp"):
            initial_projection_dimension = vision_dimension * 4
            return nn.Sequential(
                OrderedDict(
                    [
                        (
                            "projector",
                            nn.Sequential(
                                nn.Linear(
                                    vision_dimension,
                                    initial_projection_dimension,
                                    bias=True,
                                ),
                                nn.GELU(),
                                nn.Linear(
                                    initial_projection_dimension,
                                    language_dimension,
                                    bias=True,
                                ),
                                nn.GELU(),
                                nn.Linear(
                                    language_dimension,
                                    language_dimension,
                                    bias=True,
                                ),
                            ),
                        )
                    ]
                )
            )
        if arch_specifier.endswith("gelu-mlp"):
            return nn.Sequential(
                OrderedDict(
                    [
                        (
                            "projector",
                            nn.Sequential(
                                nn.Linear(
                                    vision_dimension,
                                    language_dimension,
                                    bias=True,
                                ),
                                nn.GELU(),
                                nn.Linear(
                                    language_dimension,
                                    language_dimension,
                                    bias=True,
                                ),
                            ),
                        )
                    ]
                )
            )
        raise ValueError(f"Unsupported Prismatic arch_specifier '{arch_specifier}'.")

    def _load_prismatic_checkpoint(self, checkpoint_path: Path) -> None:
        """Load raw Prismatic checkpoint weights.

        Args:
            checkpoint_path: Path to ``latest-checkpoint.pt``.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint["model"]
        self.projector.load_state_dict(model_state["projector"])
        language_state = {
            key.removeprefix("llm."): value
            for key, value in model_state["llm_backbone"].items()
        }
        self.language_model.load_state_dict(language_state)
        if "vision_backbone" in model_state:
            vision_state = {
                self._rename_vision_checkpoint_key(key): value
                for key, value in model_state["vision_backbone"].items()
            }
            self.vision_encoders.load_state_dict(vision_state)

    @staticmethod
    def _rename_vision_checkpoint_key(checkpoint_key: str) -> str:
        """Map raw Prismatic vision checkpoint names to VersatIL module names."""
        renamed_key = checkpoint_key
        for source_name, target_name in PRISMATIC_VISION_CHECKPOINT_KEY_RENAMES.items():
            renamed_key = renamed_key.replace(source_name, target_name)
        return renamed_key

    def _get_language_model(self) -> nn.Module:
        """Return the decoder-only language model submodule."""
        return self.language_model.model

    def _compute_num_image_tokens(self, config: PretrainedConfig) -> int:
        """Return Prismatic image-token count per camera."""
        del config
        return self.num_image_tokens_per_camera

    @staticmethod
    def _standardization_stats(
        backbone_type: FlatBackboneType,
    ) -> tuple[list[float], list[float]]:
        """Return RGB standardization statistics for a Prismatic timm tower."""
        match backbone_type:
            case (
                FlatBackboneType.CLIP_VITL14_224_OPENAI
                | FlatBackboneType.CLIP_VITL14_336_OPENAI
            ):
                return CLIP_RGB_MEAN, CLIP_RGB_STD
            case (
                FlatBackboneType.SIGLIP_SO400M_224 | FlatBackboneType.SIGLIP_SO400M_384
            ):
                return SIGLIP_RGB_MEAN, SIGLIP_RGB_STD
            case _:
                return IMAGENET_RGB_MEAN, IMAGENET_RGB_STD

    @classmethod
    def _standardize_images(
        cls,
        images: torch.Tensor,
        backbone_type: FlatBackboneType,
    ) -> torch.Tensor:
        """Standardize zero-to-one RGB images for one Prismatic timm tower."""
        mean, standard_deviation = cls._standardization_stats(
            backbone_type=backbone_type
        )
        mean_tensor = images.new_tensor(mean).view(1, 3, 1, 1)
        standard_deviation_tensor = images.new_tensor(standard_deviation).view(
            1, 3, 1, 1
        )
        return (images - mean_tensor) / standard_deviation_tensor

    def _encode_image_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Encode one camera image batch through all configured vision towers."""
        images = resize_to_target_size(
            images=images,
            target_height=self.image_size,
            target_width=self.image_size,
        )
        patch_features = []
        for backbone_type, encoder in zip(
            self.vision_backbone_types,
            self.vision_encoders,
            strict=True,
        ):
            standardized_images = self._standardize_images(
                images=images,
                backbone_type=backbone_type,
            )
            patch_features.append(encoder._encode_single_image(standardized_images))
        return torch.cat(patch_features, dim=2)

    def _embed_images(
        self,
        inputs: dict[str, torch.Tensor],
        batch_size: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Encode and project each configured camera image.

        Args:
            inputs: Dict with camera images as ``(B, C, H, W)`` per camera key.
            batch_size: Batch size.

        Returns:
            Projected image embeddings and matching padding masks.
        """
        image_embeddings = []
        image_pad_masks = []
        for camera_key in self.camera_keys:
            patch_features = self._encode_image_patch_tokens(images=inputs[camera_key])
            projected_embeddings = self.projector(patch_features)
            image_embeddings.append(projected_embeddings)
            image_pad_masks.append(
                torch.zeros(
                    batch_size,
                    self.num_image_tokens_per_camera,
                    dtype=torch.bool,
                    device=projected_embeddings.device,
                )
            )
        return image_embeddings, image_pad_masks

    def _merge_image_language_embeddings(
        self,
        image_embeddings: list[torch.Tensor],
        image_pad_masks: list[torch.Tensor],
        language_embeddings: torch.Tensor,
        language_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Insert image patch tokens after the BOS token (Prismatic convention).

        The pretrained Prismatic LLM expects ``[BOS, patches, text]``; the text
        is right-padded so BOS is always at position zero.
        """
        merged_embeddings = torch.cat(
            [
                language_embeddings[:, :1, :],
                *image_embeddings,
                language_embeddings[:, 1:, :],
            ],
            dim=1,
        )
        merged_padding_mask = torch.cat(
            [
                language_pad_mask[:, :1],
                *image_pad_masks,
                language_pad_mask[:, 1:],
            ],
            dim=1,
        )
        return merged_embeddings, merged_padding_mask

    def get_vocab_size(self) -> int:
        """Return the Prismatic language vocabulary size."""
        return int(self.language_model.config.vocab_size)

    def resize_token_embeddings(self, vocabulary_size: int) -> None:
        """Resize the Prismatic causal language-model token embeddings and output head."""
        self.language_model.resize_token_embeddings(vocabulary_size)

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
        """Run the Prismatic language model over token IDs or embeddings.

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
        return self.language_model(**language_model_inputs)

    def get_text_config(self) -> PretrainedConfig:
        """Return the Prismatic text model configuration."""
        return self._get_language_model().config
