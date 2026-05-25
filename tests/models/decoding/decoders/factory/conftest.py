"""Shared fixtures for decoder factory tests."""

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import LlamaConfig

from versatil.data.constants import Cameras
from versatil.models.decoding.generative_language_models.constants import (
    PRISMATIC_CONFIG_FILENAME,
    PRISMATIC_VISION_BACKBONES,
    PRISMATIC_VISION_IMAGE_SIZES,
    PrismaticLLMBackboneType,
    PrismaticModelType,
    PrismaticVisionBackboneType,
)
from versatil.models.decoding.generative_language_models.vision_language.prismatic import (
    PrismaticVLM,
)
from versatil.models.encoding.encoders.constants import (
    AttentionImplementation,
    FlatBackboneType,
)

DEFAULT_PRISMATIC_HIDDEN_DIMENSION = 16
DEFAULT_PRISMATIC_VOCABULARY_SIZE = 32
DEFAULT_PRISMATIC_TEXT_LENGTH = 5
DEFAULT_PRISMATIC_IMAGE_SIZE = 32


@pytest.fixture
def tiny_prismatic_vlm_factory(tmp_path: Path) -> Callable[..., PrismaticVLM]:
    """Factory for tiny Prismatic VLMs backed by real modules."""

    def factory(
        hidden_dimension: int = DEFAULT_PRISMATIC_HIDDEN_DIMENSION,
        vocabulary_size: int = DEFAULT_PRISMATIC_VOCABULARY_SIZE,
        max_text_length: int = DEFAULT_PRISMATIC_TEXT_LENGTH,
    ) -> PrismaticVLM:
        model_dir = tmp_path / PrismaticModelType.PRISM_DINOSIGLIP_224PX_7B.value
        model_dir.mkdir(exist_ok=True)
        (model_dir / PRISMATIC_CONFIG_FILENAME).write_text(
            json.dumps(
                {
                    "model": {
                        "model_id": (
                            PrismaticModelType.PRISM_DINOSIGLIP_224PX_7B.value
                        ),
                        "vision_backbone_id": (
                            PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_224PX.value
                        ),
                        "llm_backbone_id": PrismaticLLMBackboneType.LLAMA2_7B_PURE.value,
                        "arch_specifier": "linear",
                        "image_resize_strategy": "resize-naive",
                        "llm_max_length": max_text_length,
                    }
                }
            )
        )
        checkpoint_dir = model_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        (checkpoint_dir / "latest-checkpoint.pt").touch()
        tiny_text_config = LlamaConfig(
            vocab_size=vocabulary_size,
            hidden_size=hidden_dimension,
            intermediate_size=hidden_dimension * 2,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
        )
        with (
            patch.dict(
                PRISMATIC_VISION_BACKBONES,
                {
                    PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_224PX: (
                        FlatBackboneType.DEIT_TINY,
                        FlatBackboneType.DEIT_TINY,
                    )
                },
            ),
            patch.dict(
                PRISMATIC_VISION_IMAGE_SIZES,
                {PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_224PX: 32},
            ),
            patch(
                "versatil.models.decoding.generative_language_models.vision_language.prismatic.AutoConfig.from_pretrained",
                autospec=True,
                return_value=tiny_text_config,
            ),
            patch.object(
                PrismaticVLM,
                "_load_prismatic_checkpoint",
                autospec=True,
                return_value=None,
            ),
        ):
            backbone = PrismaticVLM(
                input_keys=[Cameras.LEFT.value],
                pretrained=True,
                frozen=False,
                model_name=str(model_dir),
                repository_id="test/prismatic",
                attention_type=AttentionImplementation.SDPA.value,
                model_dtype=None,
                max_text_length=None,
                lora_config=None,
            )
        backbone.eval()
        return backbone

    return factory
