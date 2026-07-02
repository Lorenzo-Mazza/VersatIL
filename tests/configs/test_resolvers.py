"""Tests for versatil.configs resolver registration."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from versatil.configs import register_resolvers
from versatil.data.constants import (
    ActionDiscretizerType,
    ActionTokenIdMappingType,
    BinaryGripperRange,
    Cameras,
    CoordinateSystem,
    DatasetType,
    GripperType,
    ImageNormalizationType,
    KinematicsNormalizationType,
    ObsKey,
    OrientationRepresentation,
    ProprioKey,
    RawCameraKey,
    TokenizerType,
)
from versatil.metrics.constants import MetadataKey
from versatil.metrics.kernels import KernelType
from versatil.models.adaptation.constants import LoRATargetModulePreset
from versatil.models.decoding.constants import (
    DenoisingAlgorithm,
    DiTType,
    LatentKey,
    MoERoutingType,
)
from versatil.models.decoding.generative_language_models.constants import (
    PRISMATIC_LLM_BACKBONES,
    PrismaticLLMBackboneType,
    PrismaticModelType,
)
from versatil.models.encoding.encoders.constants import (
    BatchNormHandling,
    LanguageEncoderType,
    PoolingMethod,
    RGBBackboneType,
)
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.constants import (
    AttentionType,
    ConditioningType,
    PositionalEncodingType,
)
from versatil.models.layers.denoising.diffusion_process import SchedulerType
from versatil.models.layers.denoising.timestep_sampling import TimestepSampler
from versatil.models.layers.normalization.constants import NormalizationType
from versatil.training.constants import Float32MatmulPrecision, PrecisionType

register_resolvers()


ENUM_RESOLVER_CASES = [
    ("cameras", "LEFT", Cameras.LEFT.value),
    ("cameras", "RIGHT", Cameras.RIGHT.value),
    ("cameras", "DEPTH", Cameras.DEPTH.value),
    ("raw_camera", "LEFT", RawCameraKey.LEFT.value),
    ("gripper", "BINARY", GripperType.BINARY.value),
    ("gripper", "CONTINUOUS", GripperType.CONTINUOUS.value),
    ("orientation", "ROLL", OrientationRepresentation.ROLL.value),
    ("orientation", "EULER", OrientationRepresentation.EULER.value),
    ("rgb_backbone", "RESNET18", RGBBackboneType.RESNET18.value),
    ("precision", "FP32", PrecisionType.FP32.value),
    ("precision", "BF16_MIXED", PrecisionType.BF16_MIXED.value),
    ("float32_matmul", "HIGHEST", Float32MatmulPrecision.HIGHEST.value),
    ("batch_norm_handling", "FROZEN", BatchNormHandling.FROZEN.value),
    ("pooling_method", "SPATIAL_SOFTMAX", PoolingMethod.SPATIAL_SOFTMAX.value),
    ("language_model", "BERT_BASE", LanguageEncoderType.BERT_BASE.value),
    (
        "lora_target_modules",
        "ALL_LINEAR",
        LoRATargetModulePreset.ALL_LINEAR.value,
    ),
    (
        "prismatic_model",
        "PRISM_DINOSIGLIP_224PX_7B",
        PrismaticModelType.PRISM_DINOSIGLIP_224PX_7B.value,
    ),
    (
        "prismatic_llm_model",
        "LLAMA2_7B_PURE",
        PRISMATIC_LLM_BACKBONES[PrismaticLLMBackboneType.LLAMA2_7B_PURE],
    ),
    ("activation_function", "RELU", ActivationFunction.RELU.value),
    ("activation_function", "GELU", ActivationFunction.GELU.value),
    ("normalization", "LAYER_NORM", NormalizationType.LAYER_NORM.value),
    ("attention", "MULTI_HEAD", AttentionType.MULTI_HEAD.value),
    ("pos_encoding", "SINUSOIDAL", PositionalEncodingType.SINUSOIDAL.value),
    ("pos_encoding", "LEARNED", PositionalEncodingType.LEARNED.value),
    ("tokenizer_type", "FAST", TokenizerType.FAST.value),
    ("action_discretizer", "FAST", ActionDiscretizerType.FAST.value),
    (
        "action_token_id_mapping",
        "IDENTITY",
        ActionTokenIdMappingType.IDENTITY.value,
    ),
    ("kinematics_norm_type", "MIN_MAX", KinematicsNormalizationType.MIN_MAX.value),
    ("image_norm_type", "ZERO_TO_ONE", ImageNormalizationType.ZERO_TO_ONE.value),
    ("image_norm_type", "CLIP", ImageNormalizationType.CLIP.value),
    ("obs_key", "LANGUAGE", ObsKey.LANGUAGE.value),
    ("moe_routing_type", "SOFT", MoERoutingType.SOFT.value),
    ("coordinate_system", "ROBOT_BASE", CoordinateSystem.ROBOT_BASE.value),
    ("gripper_range", "ZERO_ONE", BinaryGripperRange.ZERO_ONE.value),
    ("proprio_key", "GRIPPER_STATE", ProprioKey.GRIPPER_STATE.value),
    ("latent_key", "POSTERIOR_LATENT", LatentKey.POSTERIOR_LATENT.value),
    ("scheduler_type", "DDIM", SchedulerType.DDIM.value),
    ("denoising_algorithm", "DIFFUSION", DenoisingAlgorithm.DIFFUSION.value),
    ("conditioning_type", "ADALN", ConditioningType.ADALN.value),
    ("metadata_key", "POSTERIOR_Z", MetadataKey.POSTERIOR_Z.value),
    ("dit_type", "DIT_BLOCK", DiTType.DIT_BLOCK.value),
    ("timestep_sampler", "UNIFORM", TimestepSampler.UNIFORM.value),
    ("dataset_type", "LIBERO", DatasetType.LIBERO.value),
    ("kernel_type", "RBF", KernelType.RBF.value),
]


@pytest.mark.unit
class TestEnumResolvers:
    @pytest.mark.parametrize(
        "resolver_name, member_name, expected_value",
        ENUM_RESOLVER_CASES,
        ids=[f"{r}:{m}" for r, m, _ in ENUM_RESOLVER_CASES],
    )
    def test_enum_resolver_returns_correct_value(
        self, resolver_name, member_name, expected_value
    ):
        cfg = OmegaConf.create({"result": f"${{{resolver_name}:{member_name}}}"})
        assert cfg.result == expected_value

    def test_invalid_enum_name_raises_interpolation_error(self):
        cfg = OmegaConf.create({"invalid": "${cameras:NONEXISTENT}"})
        with pytest.raises(InterpolationResolutionError):
            _ = cfg.invalid

    def test_resolver_works_inside_list(self):
        cfg = OmegaConf.create(
            {
                "camera_keys": [
                    "${cameras:LEFT}",
                    "${cameras:RIGHT}",
                    "${cameras:DEPTH}",
                ]
            }
        )
        assert cfg.camera_keys == [
            Cameras.LEFT.value,
            Cameras.RIGHT.value,
            Cameras.DEPTH.value,
        ]

    def test_resolver_works_in_nested_config(self):
        cfg = OmegaConf.create(
            {
                "task": {
                    "cameras": ["${cameras:LEFT}"],
                    "gripper_type": "${gripper:BINARY}",
                }
            }
        )
        assert cfg.task.cameras == [Cameras.LEFT.value]
        assert cfg.task.gripper_type == GripperType.BINARY.value

    def test_resolver_combined_with_omegaconf_interpolation(self):
        cfg = OmegaConf.create(
            {
                "default_camera": "${cameras:LEFT}",
                "selected_camera": "${default_camera}",
            }
        )
        assert cfg.default_camera == Cameras.LEFT.value
        assert cfg.selected_camera == Cameras.LEFT.value


@pytest.mark.unit
class TestMultiplicationResolver:
    @pytest.mark.parametrize(
        "left, right, expected_value",
        [
            (768, 7, 5376),
            ("24", "3", 72),
            (0.5, 1024, 512.0),
        ],
    )
    def test_mul_resolver_returns_numeric_product(
        self,
        left: int | float | str,
        right: int | float | str,
        expected_value: int | float,
    ) -> None:
        cfg = OmegaConf.create({"result": f"${{mul:{left},{right}}}"})
        expected_type = int if isinstance(expected_value, int) else float
        assert cfg.result == expected_value
        assert isinstance(cfg.result, expected_type)


@pytest.mark.unit
class TestIntegerMultiplicationResolver:
    @pytest.mark.parametrize(
        "left, right, expected_value",
        [
            (0.5, 960, 480),
            ("0.75", "640", 480),
        ],
    )
    def test_int_mul_resolver_returns_integer_product(
        self,
        left: int | float | str,
        right: int | float | str,
        expected_value: int,
    ) -> None:
        cfg = OmegaConf.create({"result": f"${{int_mul:{left},{right}}}"})
        assert cfg.result == expected_value
        assert isinstance(cfg.result, int)


@pytest.mark.unit
class TestActionSpacePredictionDimensionResolver:
    def test_returns_total_dimension_for_predicted_actions(self) -> None:
        cfg = OmegaConf.create(
            {
                "action_space": {
                    "actions_metadata": {
                        "position": {"prediction_dimension": 3},
                        "orientation": {"prediction_dimension": 3},
                        "gripper": {"prediction_dimension": 1},
                        "phase": {
                            "prediction_dimension": 4,
                            "requires_prediction_head": False,
                        },
                    }
                },
                "result": "${action_space_prediction_dimension:${action_space}}",
            }
        )

        assert cfg.result == 7


@pytest.mark.unit
class TestPathResolvers:
    def test_env_resolver_reads_environment_variable(self):
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            cfg = OmegaConf.create({"val": "${env:TEST_VAR}"})
            assert cfg.val == "test_value"

    def test_env_resolver_returns_default_when_variable_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = OmegaConf.create({"val": "${env:NONEXISTENT_VAR,fallback}"})
            assert cfg.val == "fallback"

    def test_checkpoint_dir_resolver_uses_env_variable(self):
        with patch.dict(os.environ, {"VERSATIL_CHECKPOINT_DIR": "/data/checkpoints"}):
            cfg = OmegaConf.create({"dir": "${checkpoint_dir:}"})
            assert cfg.dir == "/data/checkpoints"

    def test_checkpoint_dir_resolver_appends_subpath(self):
        with patch.dict(os.environ, {"VERSATIL_CHECKPOINT_DIR": "/data/checkpoints"}):
            cfg = OmegaConf.create({"dir": "${checkpoint_dir:experiment_1}"})
            assert cfg.dir == str(Path("/data/checkpoints") / "experiment_1")

    def test_checkpoint_dir_resolver_defaults_to_cwd(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = OmegaConf.create({"dir": "${checkpoint_dir:}"})
            assert cfg.dir == "."

    def test_zarr_dir_resolver_uses_env_variable(self):
        with patch.dict(os.environ, {"VERSATIL_ZARR_DIR": "/data/zarr"}):
            cfg = OmegaConf.create({"dir": "${zarr_dir:}"})
            assert cfg.dir == "/data/zarr"

    def test_zarr_dir_resolver_appends_subpath(self):
        with patch.dict(os.environ, {"VERSATIL_ZARR_DIR": "/data/zarr"}):
            cfg = OmegaConf.create({"dir": "${zarr_dir:my_dataset}"})
            assert cfg.dir == str(Path("/data/zarr") / "my_dataset")

    def test_cache_dir_resolver_uses_env_variable(self):
        with patch.dict(os.environ, {"VERSATIL_CACHE_DIR": "/custom/cache"}):
            cfg = OmegaConf.create({"dir": "${cache_dir:}"})
            assert cfg.dir == "/custom/cache"

    def test_cache_dir_resolver_defaults_to_home_cache(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = OmegaConf.create({"dir": "${cache_dir:}"})
            assert cfg.dir == str(Path.home() / ".cache" / "versatil")

    def test_bowel_retraction_dir_resolver_uses_env_variable(self):
        with patch.dict(
            os.environ,
            {"VERSATIL_BOWEL_RETRACTION_DIR": "/data/bowel_retraction"},
        ):
            cfg = OmegaConf.create({"dir": "${bowel_retraction_dir:}"})
            assert cfg.dir == "/data/bowel_retraction"

    def test_bowel_retraction_dir_resolver_appends_subpath(self):
        with patch.dict(
            os.environ,
            {"VERSATIL_BOWEL_RETRACTION_DIR": "/data/bowel_retraction"},
        ):
            cfg = OmegaConf.create({"dir": "${bowel_retraction_dir:v1}"})
            assert cfg.dir == str(Path("/data/bowel_retraction") / "v1")

    def test_multimodal_peg_transfer_dir_resolver_uses_env_variable(self):
        with patch.dict(
            os.environ,
            {"VERSATIL_MULTIMODAL_PEG_TRANSFER_DIR": "/data/multimodal_peg_transfer"},
        ):
            cfg = OmegaConf.create({"dir": "${multimodal_peg_transfer_dir:}"})
            assert cfg.dir == "/data/multimodal_peg_transfer"

    def test_multimodal_peg_transfer_dir_resolver_appends_subpath(self):
        with patch.dict(
            os.environ,
            {"VERSATIL_MULTIMODAL_PEG_TRANSFER_DIR": "/data/multimodal_peg_transfer"},
        ):
            cfg = OmegaConf.create({"dir": "${multimodal_peg_transfer_dir:session_1}"})
            assert cfg.dir == str(Path("/data/multimodal_peg_transfer") / "session_1")


@pytest.mark.unit
class TestNumericResolvers:
    @pytest.mark.parametrize(
        "num_epochs, fraction, expected_epoch",
        [
            pytest.param(2000, 0.4, 800, id="synthetic-budget"),
            pytest.param(50, 0.4, 20, id="short-budget"),
            pytest.param(2, 0.8, 1, id="keeps-valid-order"),
        ],
    )
    def test_stage_split_epoch_returns_valid_integer_boundary(
        self,
        num_epochs: int,
        fraction: float,
        expected_epoch: int,
    ):
        cfg = OmegaConf.create(
            {"split": f"${{stage_split_epoch:{num_epochs},{fraction}}}"}
        )

        assert cfg.split == expected_epoch

    @pytest.mark.parametrize(
        "num_epochs, fraction",
        [
            pytest.param(0, 0.2, id="non-positive-epochs"),
            pytest.param(10, 0.0, id="zero-fraction"),
            pytest.param(10, 1.0, id="one-fraction"),
        ],
    )
    def test_stage_split_epoch_rejects_invalid_inputs(
        self,
        num_epochs: int,
        fraction: float,
    ):
        cfg = OmegaConf.create(
            {"split": f"${{stage_split_epoch:{num_epochs},{fraction}}}"}
        )

        with pytest.raises(InterpolationResolutionError):
            _ = cfg.split
