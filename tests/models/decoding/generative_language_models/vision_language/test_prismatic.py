"""Tests for versatil.models.decoding.generative_language_models.vision_language.prismatic module."""

import gc
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys
from transformers import (
    AutoModelForCausalLM,
    LlamaConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import BaseModelOutput

from versatil.data.constants import (
    CLIP_RGB_MEAN,
    CLIP_RGB_STD,
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    SIGLIP_RGB_MEAN,
    SIGLIP_RGB_STD,
    Cameras,
    SampleKey,
)
from versatil.models.adaptation.constants import LoRATargetModulePreset
from versatil.models.adaptation.lora import LoRAAdaptation
from versatil.models.decoding.generative_language_models.constants import (
    PRISMATIC_CHECKPOINT_FILENAME,
    PRISMATIC_CONFIG_FILENAME,
    PRISMATIC_LLM_BACKBONES,
    PRISMATIC_PAD_TO_MULTIPLE_OF,
    PRISMATIC_REPOSITORY_ID,
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
    EncoderOutputKeys,
    FlatBackboneType,
    PoolingMethod,
)
from versatil.models.encoding.encoders.rgb.flat import FlatRGBEncoder
from versatil.models.encoding.explainability import (
    ActivationLayout,
    ExplanationTargetKind,
)

HIDDEN_DIMENSION = 8
VISION_DIMENSION = 5
VOCABULARY_SIZE = 16
IMAGE_SIZE = 8
NUM_PATCHES = 3
MAX_TEXT_LENGTH = 6
TINY_PRISMATIC_HIDDEN_DIMENSION = 32


def _load_state_result() -> MagicMock:
    result = MagicMock(spec=_IncompatibleKeys)
    result.missing_keys = []
    result.unexpected_keys = []
    return result


@dataclass
class PrismaticMockDependencies:
    vision_encoders: list[MagicMock]
    vision_encoder_collection: MagicMock
    vision_loaded_state_dict: dict[str, torch.Tensor]
    projector: MagicMock
    projector_loaded_state_dict: dict[str, torch.Tensor]
    language_model: MagicMock
    language_loaded_state_dict: dict[str, torch.Tensor]
    language_backbone: MagicMock
    language_embedding: MagicMock
    vision_encoder_factory: MagicMock
    auto_config: MagicMock
    language_model_constructor: MagicMock
    projector_factory: MagicMock
    torch_load: MagicMock


@pytest.fixture
def prismatic_config_dir_factory(tmp_path: Path) -> Callable[..., Path]:
    def factory(
        vision_backbone_id: str = (
            PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_224PX.value
        ),
        llm_backbone_id: str = PrismaticLLMBackboneType.LLAMA2_7B_PURE.value,
        arch_specifier: str = "no-align+fused-gelu-mlp",
        llm_max_length: int = MAX_TEXT_LENGTH,
    ) -> Path:
        config_dir = tmp_path / PrismaticModelType.PRISM_DINOSIGLIP_224PX_7B.value
        config_dir.mkdir()
        config_path = config_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "model": {
                        "model_id": (
                            PrismaticModelType.PRISM_DINOSIGLIP_224PX_7B.value
                        ),
                        "vision_backbone_id": vision_backbone_id,
                        "llm_backbone_id": llm_backbone_id,
                        "arch_specifier": arch_specifier,
                        "image_resize_strategy": "resize-naive",
                        "llm_max_length": llm_max_length,
                    }
                }
            )
        )
        checkpoint_dir = config_dir / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "latest-checkpoint.pt").touch()
        return config_dir

    return factory


@pytest.fixture
def tiny_prismatic_backbone_factory(
    prismatic_config_dir_factory: Callable[..., Path],
) -> Callable[..., PrismaticVLM]:
    def factory(
        lora_config: LoRAAdaptation | None = None,
        hidden_dimension: int = TINY_PRISMATIC_HIDDEN_DIMENSION,
        gradient_checkpointing: bool = False,
    ) -> PrismaticVLM:
        prismatic_config_dir = prismatic_config_dir_factory()
        tiny_text_config = LlamaConfig(
            vocab_size=128,
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
                model_name=str(prismatic_config_dir),
                repository_id="test/prismatic",
                attention_type=AttentionImplementation.SDPA.value,
                model_dtype=None,
                max_text_length=None,
                lora_config=lora_config,
                gradient_checkpointing=gradient_checkpointing,
            )
        backbone.eval()
        return backbone

    return factory


@pytest.fixture
def prismatic_mock_dependencies() -> PrismaticMockDependencies:
    vision_loaded_state_dict = {}
    dino_encoder = MagicMock(spec=FlatRGBEncoder)
    dino_encoder.feature_dim = 2
    dino_encoder.backbone = MagicMock(spec=nn.Module)
    dino_encoder.backbone.patch_embed = MagicMock(spec=nn.Module)
    dino_encoder.backbone.patch_embed.num_patches = NUM_PATCHES
    siglip_encoder = MagicMock(spec=FlatRGBEncoder)
    siglip_encoder.feature_dim = 3
    siglip_encoder.backbone = MagicMock(spec=nn.Module)
    siglip_encoder.backbone.patch_embed = MagicMock(spec=nn.Module)
    siglip_encoder.backbone.patch_embed.num_patches = NUM_PATCHES
    vision_encoders = [dino_encoder, siglip_encoder]
    vision_encoder_collection = MagicMock(spec=nn.ModuleList)
    vision_encoder_collection.__iter__.side_effect = lambda: iter(vision_encoders)

    def encode_dino(images: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            images.shape[0],
            NUM_PATCHES,
            2,
            device=images.device,
            dtype=images.dtype,
        )

    def encode_siglip(images: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (images.shape[0], NUM_PATCHES, 3),
            2.0,
            device=images.device,
            dtype=images.dtype,
        )

    def load_vision_state_dict(
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> MagicMock:
        del strict
        vision_loaded_state_dict.clear()
        vision_loaded_state_dict.update(state_dict)
        return _load_state_result()

    dino_encoder._encode_single_image.side_effect = encode_dino
    siglip_encoder._encode_single_image.side_effect = encode_siglip
    vision_encoder_collection.load_state_dict.side_effect = load_vision_state_dict
    projector_loaded_state_dict = {}
    projector = MagicMock(spec=nn.Module)

    def projector_forward(image_patches: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            image_patches.shape[0],
            image_patches.shape[1],
            HIDDEN_DIMENSION,
            device=image_patches.device,
            dtype=image_patches.dtype,
        )

    def load_projector_state_dict(
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> MagicMock:
        del strict
        projector_loaded_state_dict.clear()
        projector_loaded_state_dict.update(state_dict)
        return _load_state_result()

    projector.side_effect = projector_forward
    projector.load_state_dict.side_effect = load_projector_state_dict
    language_config = MagicMock(spec=PretrainedConfig)
    language_config.hidden_size = HIDDEN_DIMENSION
    language_config.vocab_size = VOCABULARY_SIZE
    language_config.output_hidden_states = False
    language_config.pad_token_id = None
    language_embedding = MagicMock(spec=nn.Embedding)

    def embed_tokens(token_ids: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            token_ids.shape[0],
            token_ids.shape[1],
            HIDDEN_DIMENSION,
            dtype=torch.float32,
            device=token_ids.device,
        )

    language_embedding.side_effect = embed_tokens
    language_backbone = MagicMock(spec=nn.Module)
    language_backbone.config = language_config
    language_backbone.layers = [MagicMock(spec=nn.Identity)]
    language_backbone.rotary_emb = MagicMock(spec=nn.Identity)
    language_backbone.get_input_embeddings = MagicMock(
        spec=PreTrainedModel.get_input_embeddings,
        return_value=language_embedding,
    )

    def language_forward(
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> MagicMock:
        del attention_mask
        output = MagicMock(spec=BaseModelOutput)
        output.last_hidden_state = inputs_embeds + 1.0
        return output

    language_backbone.side_effect = language_forward
    language_loaded_state_dict = {}
    language_model = MagicMock(spec=nn.Module)
    language_model.config = language_config
    language_model.model = language_backbone

    def resize_token_embeddings(
        new_num_tokens: int,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        del pad_to_multiple_of
        language_config.vocab_size = new_num_tokens

    def load_language_state_dict(
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> MagicMock:
        del strict
        language_loaded_state_dict.clear()
        language_loaded_state_dict.update(state_dict)
        return _load_state_result()

    language_model.resize_token_embeddings = MagicMock(
        spec=PreTrainedModel.resize_token_embeddings,
        side_effect=resize_token_embeddings,
    )
    language_model.load_state_dict = MagicMock(
        spec=nn.Module.load_state_dict,
        side_effect=load_language_state_dict,
    )
    checkpoint = {
        "model": {
            "projector": {"projector.0.weight": torch.ones(1)},
            "llm_backbone": {"llm.model.embed_tokens.weight": torch.ones(1)},
            "vision_backbone": {"dino_featurizer.weight": torch.ones(1)},
        }
    }
    vision_encoder_patcher = patch.object(
        PrismaticVLM,
        "_build_vision_encoders",
        autospec=True,
        return_value=vision_encoder_collection,
    )
    auto_config_patcher = patch(
        "versatil.models.decoding.generative_language_models.vision_language.prismatic.AutoConfig.from_pretrained",
        autospec=True,
        return_value=language_config,
    )
    language_model_patcher = patch(
        "versatil.models.decoding.generative_language_models.vision_language.prismatic.AutoModelForCausalLM.from_config",
        new=MagicMock(
            spec=AutoModelForCausalLM.from_config,
            return_value=language_model,
        ),
    )
    projector_patcher = patch.object(
        PrismaticVLM,
        "_build_projector",
        autospec=True,
        return_value=projector,
    )
    torch_load_patcher = patch(
        "versatil.models.decoding.generative_language_models.vision_language.prismatic.torch.load",
        autospec=True,
        return_value=checkpoint,
    )
    patchers = [
        vision_encoder_patcher,
        auto_config_patcher,
        language_model_patcher,
        projector_patcher,
        torch_load_patcher,
    ]
    vision_encoder_factory = vision_encoder_patcher.start()
    auto_config = auto_config_patcher.start()
    language_model_constructor = language_model_patcher.start()
    projector_factory = projector_patcher.start()
    torch_load = torch_load_patcher.start()
    yield PrismaticMockDependencies(
        vision_encoders=vision_encoders,
        vision_encoder_collection=vision_encoder_collection,
        vision_loaded_state_dict=vision_loaded_state_dict,
        projector=projector,
        projector_loaded_state_dict=projector_loaded_state_dict,
        language_model=language_model,
        language_loaded_state_dict=language_loaded_state_dict,
        language_backbone=language_backbone,
        language_embedding=language_embedding,
        vision_encoder_factory=vision_encoder_factory,
        auto_config=auto_config,
        language_model_constructor=language_model_constructor,
        projector_factory=projector_factory,
        torch_load=torch_load,
    )
    for patcher in reversed(patchers):
        patcher.stop()


@pytest.mark.unit
class TestPrismaticVLMInitialization:
    def test_stores_configuration_and_builds_language_model(
        self,
        prismatic_config_dir_factory: Callable[..., Path],
        prismatic_mock_dependencies: PrismaticMockDependencies,
    ) -> None:
        prismatic_config_dir = prismatic_config_dir_factory()
        backbone = PrismaticVLM(
            input_keys=[Cameras.LEFT.value],
            pretrained=False,
            frozen=False,
            model_name=str(prismatic_config_dir),
            repository_id="test/prismatic",
            attention_type=AttentionImplementation.SDPA.value,
            model_dtype=None,
            max_text_length=None,
            lora_config=None,
        )

        assert backbone.model_name == str(prismatic_config_dir)
        assert backbone.repository_id == "test/prismatic"
        assert (
            backbone.vision_backbone_id
            == PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_224PX.value
        )
        assert backbone.llm_backbone_id == PrismaticLLMBackboneType.LLAMA2_7B_PURE.value
        assert backbone.image_size == 224
        assert backbone.hidden_dim == HIDDEN_DIMENSION
        assert backbone.max_text_length == MAX_TEXT_LENGTH
        assert backbone.num_image_tokens_per_camera == NUM_PATCHES
        assert backbone.total_image_tokens == NUM_PATCHES
        assert backbone.get_vocab_size() == PRISMATIC_PAD_TO_MULTIPLE_OF
        assert backbone.get_text_config() == backbone._get_language_model().config
        assert (
            prismatic_mock_dependencies.language_model.config.pad_token_id
            == VOCABULARY_SIZE
        )
        assert (
            prismatic_mock_dependencies.language_model.config.vocab_size
            == PRISMATIC_PAD_TO_MULTIPLE_OF
        )
        resize_call = (
            prismatic_mock_dependencies.language_model.resize_token_embeddings.call_args
        )
        assert resize_call.args == (PRISMATIC_PAD_TO_MULTIPLE_OF,)
        assert resize_call.kwargs == {
            "pad_to_multiple_of": PRISMATIC_PAD_TO_MULTIPLE_OF
        }
        prismatic_mock_dependencies.vision_encoder_factory.assert_called_once()
        vision_factory_call = (
            prismatic_mock_dependencies.vision_encoder_factory.call_args
        )
        assert vision_factory_call.args[0] is backbone
        # The towers follow the VLM pretrained flag: raw DinoSigLIP checkpoints
        # do not store vision weights, so pretrained=True must load timm weights.
        assert vision_factory_call.kwargs == {"pretrained": False}
        prismatic_mock_dependencies.auto_config.assert_called_once_with(
            PRISMATIC_LLM_BACKBONES[PrismaticLLMBackboneType.LLAMA2_7B_PURE]
        )
        language_config = prismatic_mock_dependencies.auto_config.return_value
        assert prismatic_mock_dependencies.language_model_constructor.call_count == 1
        language_model_call = (
            prismatic_mock_dependencies.language_model_constructor.call_args
        )
        assert language_model_call.args == (language_config,)
        assert language_model_call.kwargs == {
            "attn_implementation": AttentionImplementation.SDPA.value
        }
        prismatic_mock_dependencies.projector_factory.assert_called_once_with(
            arch_specifier="no-align+fused-gelu-mlp",
            vision_dimension=VISION_DIMENSION,
            language_dimension=HIDDEN_DIMENSION,
        )

    def test_pretrained_loads_raw_checkpoint_keys(
        self,
        prismatic_config_dir_factory: Callable[..., Path],
        prismatic_mock_dependencies: PrismaticMockDependencies,
    ) -> None:
        prismatic_config_dir = prismatic_config_dir_factory()
        PrismaticVLM(
            input_keys=[Cameras.LEFT.value],
            pretrained=True,
            frozen=False,
            model_name=str(prismatic_config_dir),
            repository_id="test/prismatic",
            attention_type=AttentionImplementation.SDPA.value,
            model_dtype=None,
            max_text_length=None,
            lora_config=None,
        )

        torch.testing.assert_close(
            prismatic_mock_dependencies.projector_loaded_state_dict[
                "projector.0.weight"
            ],
            torch.ones(1),
        )
        torch.testing.assert_close(
            prismatic_mock_dependencies.language_loaded_state_dict[
                "model.embed_tokens.weight"
            ],
            torch.ones(1),
        )
        torch.testing.assert_close(
            prismatic_mock_dependencies.vision_loaded_state_dict["0.backbone.weight"],
            torch.ones(1),
        )
        prismatic_mock_dependencies.torch_load.assert_called_once_with(
            prismatic_config_dir / "checkpoints" / "latest-checkpoint.pt",
            map_location="cpu",
        )

    def test_unsupported_llm_backbone_raises(self) -> None:
        expected_message = (
            "Unsupported Prismatic llm_backbone_id 'unsupported'. "
            "Supported values: "
            f"{[model_type.value for model_type in PrismaticLLMBackboneType]}."
        )
        with pytest.raises(
            ValueError,
            match=re.escape(expected_message),
        ):
            PrismaticVLM._resolve_llm_model_name(llm_backbone_id="unsupported")

    def test_unsupported_vision_backbone_raises(self) -> None:
        expected_message = (
            "Unsupported Prismatic vision_backbone_id 'unsupported'. "
            "Supported values: "
            f"{[model_type.value for model_type in PrismaticVisionBackboneType]}."
        )
        with pytest.raises(
            ValueError,
            match=re.escape(expected_message),
        ):
            PrismaticVLM._resolve_vision_backbone_type(vision_backbone_id="unsupported")

    @pytest.mark.parametrize(
        "value, multiple, expected",
        [
            (16, 8, 16),
            (17, 8, 24),
            (0, 8, 0),
        ],
    )
    def test_pad_to_multiple_rounds_up_to_requested_boundary(
        self,
        value: int,
        multiple: int,
        expected: int,
    ) -> None:
        assert PrismaticVLM._pad_to_multiple(value=value, multiple=multiple) == expected

    @pytest.mark.parametrize(
        "backbone_type, expected_mean, expected_standard_deviation",
        [
            (
                FlatBackboneType.CLIP_VITL14_224_OPENAI,
                CLIP_RGB_MEAN,
                CLIP_RGB_STD,
            ),
            (
                FlatBackboneType.SIGLIP_SO400M_224,
                SIGLIP_RGB_MEAN,
                SIGLIP_RGB_STD,
            ),
            (
                FlatBackboneType.DINOV2_VITL14_REG4,
                IMAGENET_RGB_MEAN,
                IMAGENET_RGB_STD,
            ),
        ],
    )
    def test_standardization_stats_follow_backbone_family(
        self,
        backbone_type: FlatBackboneType,
        expected_mean: list[float],
        expected_standard_deviation: list[float],
    ) -> None:
        mean, standard_deviation = PrismaticVLM._standardization_stats(
            backbone_type=backbone_type
        )

        assert mean == expected_mean
        assert standard_deviation == expected_standard_deviation

    def test_build_vision_encoders_constructs_flat_rgb_towers(self) -> None:
        backbone = MagicMock(spec=PrismaticVLM)
        backbone.camera_keys = [Cameras.LEFT.value]
        backbone.image_size = 224
        backbone.vision_backbone_types = (
            FlatBackboneType.DINOV2_VITL14_REG4,
            FlatBackboneType.SIGLIP_SO400M_224,
        )
        dino_encoder = MagicMock(spec=FlatRGBEncoder)
        siglip_encoder = MagicMock(spec=FlatRGBEncoder)

        with (
            patch(
                "versatil.models.decoding.generative_language_models.vision_language.prismatic.FlatRGBEncoder",
                autospec=True,
                side_effect=[dino_encoder, siglip_encoder],
            ) as flat_encoder_mock,
            patch(
                "versatil.models.decoding.generative_language_models.vision_language.prismatic.nn.ModuleList",
                autospec=True,
                side_effect=lambda encoders: encoders,
            ) as module_list_mock,
        ):
            encoders = PrismaticVLM._build_vision_encoders(
                backbone,
                pretrained=False,
            )

        assert encoders == [dino_encoder, siglip_encoder]
        assert flat_encoder_mock.call_args_list[0].kwargs == {
            "input_keys": [Cameras.LEFT.value],
            "pretrained": False,
            "frozen": False,
            "pooling_method": PoolingMethod.NONE.value,
            "backbone": FlatBackboneType.DINOV2_VITL14_REG4.value,
            "image_size": 224,
            "intermediate_layer_index": -2,
            "model_dtype": None,
            "lora_config": None,
        }
        assert flat_encoder_mock.call_args_list[1].kwargs == {
            "input_keys": [Cameras.LEFT.value],
            "pretrained": False,
            "frozen": False,
            "pooling_method": PoolingMethod.NONE.value,
            "backbone": FlatBackboneType.SIGLIP_SO400M_224.value,
            "image_size": 224,
            "intermediate_layer_index": -2,
            "model_dtype": None,
            "lora_config": None,
        }
        module_list_mock.assert_called_once_with([dino_encoder, siglip_encoder])

    def test_resolve_num_image_tokens_rejects_mismatched_towers(self) -> None:
        backbone = MagicMock(spec=PrismaticVLM)
        first_encoder = MagicMock(spec=FlatRGBEncoder)
        first_encoder.backbone = MagicMock(spec=nn.Module)
        first_encoder.backbone.patch_embed = MagicMock(spec=nn.Module)
        first_encoder.backbone.patch_embed.num_patches = 4
        second_encoder = MagicMock(spec=FlatRGBEncoder)
        second_encoder.backbone = MagicMock(spec=nn.Module)
        second_encoder.backbone.patch_embed = MagicMock(spec=nn.Module)
        second_encoder.backbone.patch_embed.num_patches = 5
        backbone.vision_encoders = [first_encoder, second_encoder]
        expected_message = (
            "Prismatic vision towers must produce the same number of patch "
            "tokens, got [4, 5]."
        )

        with pytest.raises(ValueError, match=re.escape(expected_message)):
            PrismaticVLM._resolve_num_image_tokens(backbone)
            PrismaticVLM._resolve_num_image_tokens(backbone)

    @pytest.mark.parametrize(
        "checkpoint_key, expected_key",
        [
            ("dino_featurizer.blocks.0.weight", "0.backbone.blocks.0.weight"),
            ("siglip_featurizer.blocks.0.weight", "1.backbone.blocks.0.weight"),
            ("clip_featurizer.blocks.0.weight", "1.backbone.blocks.0.weight"),
            ("featurizer.blocks.0.weight", "0.backbone.blocks.0.weight"),
        ],
    )
    def test_rename_vision_checkpoint_key(
        self,
        checkpoint_key: str,
        expected_key: str,
    ) -> None:
        assert (
            PrismaticVLM._rename_vision_checkpoint_key(checkpoint_key) == expected_key
        )


@pytest.mark.unit
class TestPrismaticVLMPathResolution:
    def test_resolve_config_path_uses_local_model_directory(
        self,
        prismatic_config_dir_factory: Callable[..., Path],
    ) -> None:
        prismatic_config_dir = prismatic_config_dir_factory()

        config_path = PrismaticVLM._resolve_config_path(
            model_name=str(prismatic_config_dir),
            repository_id="test/prismatic",
        )

        assert config_path == prismatic_config_dir / PRISMATIC_CONFIG_FILENAME

    def test_resolve_config_path_downloads_from_hub(self) -> None:
        downloaded_path = "/cache/prismatic/config.json"
        with patch(
            "versatil.models.decoding.generative_language_models.vision_language.prismatic.hf_hub_download",
            autospec=True,
            return_value=downloaded_path,
        ) as download_mock:
            config_path = PrismaticVLM._resolve_config_path(
                model_name=PrismaticModelType.PRISM_DINOSIGLIP_224PX_7B.value,
                repository_id=PRISMATIC_REPOSITORY_ID,
            )

        assert config_path == Path(downloaded_path)
        download_mock.assert_called_once_with(
            repo_id=PRISMATIC_REPOSITORY_ID,
            filename=(
                f"{PrismaticModelType.PRISM_DINOSIGLIP_224PX_7B.value}/"
                f"{PRISMATIC_CONFIG_FILENAME}"
            ),
        )

    def test_resolve_checkpoint_path_uses_local_model_directory(
        self,
        prismatic_config_dir_factory: Callable[..., Path],
    ) -> None:
        prismatic_config_dir = prismatic_config_dir_factory()

        checkpoint_path = PrismaticVLM._resolve_checkpoint_path(
            model_name=str(prismatic_config_dir),
            repository_id="test/prismatic",
        )

        assert checkpoint_path == prismatic_config_dir / PRISMATIC_CHECKPOINT_FILENAME

    def test_resolve_checkpoint_path_downloads_from_hub(self) -> None:
        downloaded_path = "/cache/prismatic/latest-checkpoint.pt"
        with patch(
            "versatil.models.decoding.generative_language_models.vision_language.prismatic.hf_hub_download",
            autospec=True,
            return_value=downloaded_path,
        ) as download_mock:
            checkpoint_path = PrismaticVLM._resolve_checkpoint_path(
                model_name=PrismaticModelType.PRISM_DINOSIGLIP_224PX_7B.value,
                repository_id=PRISMATIC_REPOSITORY_ID,
            )

        assert checkpoint_path == Path(downloaded_path)
        download_mock.assert_called_once_with(
            repo_id=PRISMATIC_REPOSITORY_ID,
            filename=(
                f"{PrismaticModelType.PRISM_DINOSIGLIP_224PX_7B.value}/"
                f"{PRISMATIC_CHECKPOINT_FILENAME}"
            ),
        )


@pytest.mark.unit
def test_unsupported_projector_architecture_specifier_raises() -> None:
    arch_specifier = "unsupported"
    expected_message = f"Unsupported Prismatic arch_specifier '{arch_specifier}'."

    with pytest.raises(
        ValueError,
        match=re.escape(expected_message),
    ):
        PrismaticVLM._build_projector(
            arch_specifier=arch_specifier,
            vision_dimension=VISION_DIMENSION,
            language_dimension=HIDDEN_DIMENSION,
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    "arch_specifier, expected_linear_layers, expected_gelu_layers, expected_state_keys",
    [
        ("linear", 1, 0, ["projector.weight", "projector.bias"]),
        (
            "no-align+gelu-mlp",
            2,
            1,
            [
                "projector.0.weight",
                "projector.0.bias",
                "projector.2.weight",
                "projector.2.bias",
            ],
        ),
        (
            "no-align+fused-gelu-mlp",
            3,
            2,
            [
                "projector.0.weight",
                "projector.0.bias",
                "projector.2.weight",
                "projector.2.bias",
                "projector.4.weight",
                "projector.4.bias",
            ],
        ),
    ],
)
def test_prismatic_projector_forward_uses_real_layers(
    arch_specifier: str,
    expected_linear_layers: int,
    expected_gelu_layers: int,
    expected_state_keys: list[str],
) -> None:
    projector = PrismaticVLM._build_projector(
        arch_specifier=arch_specifier,
        vision_dimension=VISION_DIMENSION,
        language_dimension=HIDDEN_DIMENSION,
    )
    linear_layers = [
        module for module in projector.modules() if isinstance(module, nn.Linear)
    ]
    gelu_layers = [
        module for module in projector.modules() if isinstance(module, nn.GELU)
    ]
    for linear_layer in linear_layers:
        nn.init.constant_(linear_layer.weight, 0.1)
        nn.init.zeros_(linear_layer.bias)

    zero_patches = torch.zeros(2, NUM_PATCHES, VISION_DIMENSION)
    one_patches = torch.ones(2, NUM_PATCHES, VISION_DIMENSION)
    zero_projected = projector(zero_patches)
    one_projected = projector(one_patches)

    assert len(linear_layers) == expected_linear_layers
    assert len(gelu_layers) == expected_gelu_layers
    assert list(projector.state_dict()) == expected_state_keys
    torch.testing.assert_close(
        zero_projected,
        torch.zeros(2, NUM_PATCHES, HIDDEN_DIMENSION),
    )
    assert not torch.allclose(zero_projected, one_projected)


@pytest.mark.unit
class TestPrismaticVLMCheckpointLoading:
    def test_load_prismatic_checkpoint_skips_missing_vision_backbone(
        self,
    ) -> None:
        backbone = MagicMock(spec=PrismaticVLM)
        backbone.projector = MagicMock(spec=nn.Module)
        backbone.language_model = MagicMock(spec=nn.Module)
        backbone.vision_encoders = MagicMock(spec=nn.ModuleList)
        checkpoint_path = Path("checkpoint.pt")
        checkpoint = {
            "model": {
                "projector": {"projector.weight": torch.ones(1)},
                "llm_backbone": {"llm.model.weight": torch.ones(1)},
            }
        }

        with patch(
            "versatil.models.decoding.generative_language_models.vision_language.prismatic.torch.load",
            autospec=True,
            return_value=checkpoint,
        ) as torch_load:
            PrismaticVLM._load_prismatic_checkpoint(
                backbone,
                checkpoint_path=checkpoint_path,
            )

        torch_load.assert_called_once_with(checkpoint_path, map_location="cpu")
        backbone.projector.load_state_dict.assert_called_once_with(
            checkpoint["model"]["projector"]
        )
        backbone.language_model.load_state_dict.assert_called_once()
        language_state = backbone.language_model.load_state_dict.call_args.args[0]
        assert list(language_state) == ["model.weight"]
        torch.testing.assert_close(language_state["model.weight"], torch.ones(1))
        backbone.vision_encoders.load_state_dict.assert_not_called()

    def test_get_language_model_returns_decoder_submodule(
        self,
    ) -> None:
        backbone = MagicMock(spec=PrismaticVLM)
        backbone.lora_config = None
        decoder_model = MagicMock(spec=nn.Module)
        language_model = MagicMock(spec=nn.Module)
        language_model.model = decoder_model
        backbone.language_model = language_model

        output = PrismaticVLM._get_language_model(backbone)

        assert output == decoder_model


@pytest.mark.integration
class TestPrismaticVLMCompositeLoRAIntegration:
    def test_resize_token_embeddings_after_composite_lora(
        self,
        tiny_prismatic_backbone_factory: Callable[..., PrismaticVLM],
    ) -> None:
        lora_config = LoRAAdaptation(
            enabled=True,
            rank=2,
            alpha=4,
            target_modules=LoRATargetModulePreset.LLAMA_QUERY_VALUE_PROJECTIONS.value,
        )
        backbone = tiny_prismatic_backbone_factory(lora_config=lora_config)
        vocabulary_size = int(backbone.language_model.config.vocab_size) + 4

        backbone.resize_token_embeddings(vocabulary_size=vocabulary_size)

        assert backbone.language_model.config.vocab_size == vocabulary_size
        assert backbone.language_model.get_input_embeddings().num_embeddings == (
            vocabulary_size
        )
        assert backbone.language_model.get_output_embeddings().out_features == (
            vocabulary_size
        )


@pytest.mark.unit
class TestPrismaticVLMForward:
    def test_forward_concatenates_projected_images_and_text(
        self,
        prismatic_config_dir_factory: Callable[..., Path],
        prismatic_mock_dependencies: PrismaticMockDependencies,
    ) -> None:
        prismatic_config_dir = prismatic_config_dir_factory()
        backbone = PrismaticVLM(
            input_keys=[Cameras.LEFT.value, Cameras.RIGHT.value],
            pretrained=False,
            frozen=False,
            model_name=str(prismatic_config_dir),
            repository_id="test/prismatic",
            attention_type=AttentionImplementation.SDPA.value,
            model_dtype=None,
            max_text_length=None,
            lora_config=None,
        )
        inputs = {
            Cameras.LEFT.value: torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE),
            Cameras.RIGHT.value: torch.zeros(2, 3, IMAGE_SIZE, IMAGE_SIZE),
            SampleKey.TOKENIZED_OBSERVATIONS.value: torch.zeros(
                2,
                MAX_TEXT_LENGTH,
                dtype=torch.long,
            ),
            SampleKey.IS_PAD_OBSERVATION.value: torch.zeros(
                2,
                MAX_TEXT_LENGTH,
                dtype=torch.bool,
            ),
        }
        output = backbone(inputs=inputs)

        fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
        padding_mask = output[backbone.padding_mask_name]
        assert fused.shape == (
            2,
            NUM_PATCHES * 2 + MAX_TEXT_LENGTH,
            HIDDEN_DIMENSION,
        )
        assert padding_mask.shape == (2, NUM_PATCHES * 2 + MAX_TEXT_LENGTH)
        assert not padding_mask.any()
        assert prismatic_mock_dependencies.vision_loaded_state_dict == {}
        assert prismatic_mock_dependencies.projector_loaded_state_dict == {}
        assert prismatic_mock_dependencies.language_loaded_state_dict == {}
        dino_encoder, siglip_encoder = prismatic_mock_dependencies.vision_encoders
        assert dino_encoder._encode_single_image.call_count == 2
        assert siglip_encoder._encode_single_image.call_count == 2
        assert prismatic_mock_dependencies.projector.call_count == 2
        for call in dino_encoder._encode_single_image.call_args_list:
            assert call.args[0].shape == (2, 3, 224, 224)
        for call in siglip_encoder._encode_single_image.call_args_list:
            assert call.args[0].shape == (2, 3, 224, 224)
        for call in prismatic_mock_dependencies.projector.call_args_list:
            assert call.args[0].shape == (2, NUM_PATCHES, VISION_DIMENSION)
        get_embeddings_mock = (
            prismatic_mock_dependencies.language_backbone.get_input_embeddings
        )
        assert get_embeddings_mock.call_count == 1
        assert get_embeddings_mock.call_args.args == ()
        assert get_embeddings_mock.call_args.kwargs == {}
        prismatic_mock_dependencies.language_embedding.assert_called_once()
        embedding_call = prismatic_mock_dependencies.language_embedding.call_args
        assert embedding_call.args[0].shape == (2, MAX_TEXT_LENGTH)
        prismatic_mock_dependencies.language_backbone.assert_called_once()
        language_call = prismatic_mock_dependencies.language_backbone.call_args
        assert language_call.kwargs["inputs_embeds"].shape == (
            2,
            NUM_PATCHES * 2 + MAX_TEXT_LENGTH,
            HIDDEN_DIMENSION,
        )
        assert language_call.kwargs["attention_mask"].shape == (
            2,
            NUM_PATCHES * 2 + MAX_TEXT_LENGTH,
        )


@pytest.mark.integration
@pytest.mark.parametrize("lora_enabled", [False, True])
def test_forward_pass_with_real_tiny_modules(
    tiny_prismatic_backbone_factory: Callable[..., PrismaticVLM],
    lora_enabled: bool,
    parameter_count: Callable[[torch.nn.Module], int],
    trainable_parameter_count: Callable[[torch.nn.Module], int],
) -> None:
    batch_size = 2
    lora_config = (
        LoRAAdaptation(
            enabled=True,
            rank=2,
            alpha=4,
            target_modules=LoRATargetModulePreset.LLAMA_QUERY_VALUE_PROJECTIONS.value,
        )
        if lora_enabled
        else None
    )
    backbone = tiny_prismatic_backbone_factory(lora_config=lora_config)
    inputs = {
        Cameras.LEFT.value: torch.zeros(
            batch_size,
            3,
            backbone.image_size,
            backbone.image_size,
        ),
        SampleKey.TOKENIZED_OBSERVATIONS.value: torch.zeros(
            batch_size,
            4,
            dtype=torch.long,
        ),
    }

    with torch.no_grad():
        output = backbone(inputs=inputs)
        image_conditioned_inputs = {
            **inputs,
            Cameras.LEFT.value: torch.ones(
                batch_size,
                3,
                backbone.image_size,
                backbone.image_size,
            ),
        }
        image_conditioned_output = backbone(inputs=image_conditioned_inputs)

    fused = output[EncoderOutputKeys.FUSED_RGB_LANGUAGE.value]
    image_conditioned_fused = image_conditioned_output[
        EncoderOutputKeys.FUSED_RGB_LANGUAGE.value
    ]
    padding_mask = output[backbone.padding_mask_name]
    assert fused.shape == (
        batch_size,
        backbone.num_image_tokens_per_camera + backbone.max_text_length,
        TINY_PRISMATIC_HIDDEN_DIMENSION,
    )
    assert padding_mask.shape == (
        batch_size,
        backbone.num_image_tokens_per_camera + backbone.max_text_length,
    )
    expected_padding_mask = torch.zeros_like(padding_mask)
    expected_padding_mask[:, -2:] = True
    torch.testing.assert_close(padding_mask, expected_padding_mask)
    assert not torch.allclose(fused, image_conditioned_fused)
    if lora_enabled:
        trainable_parameter_names = [
            name
            for name, parameter in backbone.named_parameters()
            if parameter.requires_grad
        ]
        trainable_parameters = trainable_parameter_count(backbone)
        total_parameters = parameter_count(backbone)
        assert trainable_parameter_names
        assert all("lora_" in name for name in trainable_parameter_names)
        assert 0 < trainable_parameters < total_parameters


@pytest.mark.integration
def test_real_tiny_vision_tower_explainability_target_captures_patch_tokens(
    tiny_prismatic_backbone_factory: Callable[..., PrismaticVLM],
) -> None:
    batch_size = 1
    backbone = tiny_prismatic_backbone_factory(lora_config=None)
    target = backbone.vision_encoders[0].get_explainability_targets()[0]
    captured_output = {}

    def capture_output(
        module: nn.Module,
        module_input: tuple[torch.Tensor, ...],
        module_output: torch.Tensor,
    ) -> None:
        del module, module_input
        captured_output["patch_tokens"] = module_output

    handle = target.layer.register_forward_hook(capture_output)
    try:
        with torch.no_grad():
            backbone(
                inputs={
                    Cameras.LEFT.value: torch.zeros(
                        batch_size,
                        3,
                        backbone.image_size,
                        backbone.image_size,
                    ),
                    SampleKey.TOKENIZED_OBSERVATIONS.value: torch.zeros(
                        batch_size,
                        4,
                        dtype=torch.long,
                    ),
                }
            )
    finally:
        handle.remove()

    patch_tokens = captured_output["patch_tokens"]
    assert target.target_kind == ExplanationTargetKind.TOKEN_SEQUENCE.value
    assert target.activation_layout == ActivationLayout.NLC.value
    assert target.patch_grid == (2, 2)
    assert patch_tokens.shape[0] == batch_size
    assert patch_tokens.dim() == 3


@pytest.mark.integration
def test_gradient_checkpointing_is_enabled_on_real_tiny_language_model(
    tiny_prismatic_backbone_factory: Callable[..., PrismaticVLM],
) -> None:
    backbone = tiny_prismatic_backbone_factory(gradient_checkpointing=True)

    assert backbone.gradient_checkpointing
    assert backbone.language_model.is_gradient_checkpointing
    assert not backbone.language_model.config.use_cache


@pytest.mark.integration
def test_forward_language_model_cache_matches_full_sequence_with_real_tiny_modules(
    tiny_prismatic_backbone_factory: Callable[..., PrismaticVLM],
) -> None:
    backbone = tiny_prismatic_backbone_factory(lora_config=None)
    batch_size = 2
    sequence_length = 5
    input_ids = torch.arange(
        batch_size * sequence_length,
        dtype=torch.long,
    ).reshape(batch_size, sequence_length)
    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long)

    with torch.no_grad():
        full_output = backbone.forward_language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        prefix_output = backbone.forward_language_model(
            input_ids=input_ids[:, :-1],
            attention_mask=attention_mask[:, :-1],
            use_cache=True,
        )
        cached_output = backbone.forward_language_model(
            input_ids=input_ids[:, -1:],
            attention_mask=attention_mask,
            past_key_values=prefix_output.past_key_values,
            use_cache=True,
            cache_position=torch.arange(sequence_length - 1, sequence_length),
        )

    if prefix_output.past_key_values is None:
        raise ValueError("Prismatic tiny language model did not return cache.")
    torch.testing.assert_close(
        cached_output.logits,
        full_output.logits[:, -1:, :],
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.fixture(scope="module")
def real_default_prismatic_backbone() -> PrismaticVLM:
    backbone = PrismaticVLM(
        input_keys=[Cameras.LEFT.value],
        pretrained=True,
        frozen=True,
        model_name=PrismaticModelType.PRISM_DINOSIGLIP_224PX_7B.value,
        repository_id=PRISMATIC_REPOSITORY_ID,
        attention_type=AttentionImplementation.SDPA.value,
        model_dtype=None,
        max_text_length=4,
        lora_config=None,
    )
    backbone.eval()
    yield backbone
    del backbone
    gc.collect()


@pytest.mark.slow
@pytest.mark.integration
def test_default_prismatic_model_loads_real_weights(
    real_default_prismatic_backbone: PrismaticVLM,
) -> None:
    backbone = real_default_prismatic_backbone
    assert (
        backbone.vision_backbone_id
        == PrismaticVisionBackboneType.DINOSIGLIP_VIT_SO_224PX.value
    )
    assert backbone.llm_backbone_id == PrismaticLLMBackboneType.LLAMA2_7B_PURE.value
    assert backbone.image_size == 224
    assert backbone.hidden_dim == 4096
    assert backbone.num_image_tokens_per_camera > 0
    assert backbone.get_vocab_size() % PRISMATIC_PAD_TO_MULTIPLE_OF == 0
    assert not next(backbone.parameters()).requires_grad
