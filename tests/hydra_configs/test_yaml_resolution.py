"""Tests for Hydra YAML configuration composition and validation."""

import glob
import warnings
from contextlib import AbstractContextManager
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from hydra.utils import instantiate
from omegaconf import ListConfig, OmegaConf

import versatil.configs  # noqa: F401 — registers ConfigStore entries
from versatil.configs.training import TrainingConfig, TrainingStageConfig
from versatil.data.constants import (
    ActionDiscretizerType,
    ActionTokenIdMappingType,
    ImageNormalizationType,
    ObsKey,
    ProprioKey,
)
from versatil.metrics.constants import ImageAugmentationConsistencyLossMode
from versatil.metrics.regularization_context import PolicyGraphInputDomain
from versatil.metrics.regularizers import (
    FiniteDifferenceLipschitzRegularizer,
    ImageAugmentationConsistencyRegularizer,
    JacobianFrobeniusLipschitzRegularizer,
    SpectralJacobianLipschitzRegularizer,
)
from versatil.models.decoding.constants import LatentKey, TimeConditioning

HYDRA_CONFIGS_ROOT = str(Path(__file__).parents[2] / "hydra_configs")

ALL_YAML_FILES = sorted(
    glob.glob(
        str(Path(HYDRA_CONFIGS_ROOT) / "**" / "*.yaml"),
        recursive=True,
    )
)

# Partial configs that use `override /x: y` in defaults or place a list at the
# primary config's top level (via ``# @package path.to.list``) cannot be
# composed standalone — Hydra only accepts DictConfig as a primary. They are
# composed transitively through a consuming parent (e.g. sweeps).
_PARTIAL_CONFIG_PREFIXES = (
    "synthetic_presets/",
    "training/multistage/",
    "training/optimizer/param_groups/",
)

ALL_CONFIG_IDS = [
    config_id
    for config_id in (
        str(Path(p).relative_to(HYDRA_CONFIGS_ROOT)).removesuffix(".yaml")
        for p in ALL_YAML_FILES
    )
    if not config_id.startswith(_PARTIAL_CONFIG_PREFIXES)
]

DIT_PRIOR_ALGORITHM_CONFIGS = [
    "policy/algorithm/bc_with_dit_prior",
    "policy/algorithm/flow_matching_with_dit_prior",
    "policy/algorithm/bc_with_denoising_flow_matching_prior",
]

VLM_IMAGE_NORM_CONFIGS = [
    pytest.param(
        "end_to_end_training_runs/libero_plus/vision_sweep/clip_vitb32",
        ImageNormalizationType.CLIP.value,
        "clip",
        id="clip-vlm",
    ),
    pytest.param(
        "end_to_end_training_runs/libero_plus/vision_sweep/siglip2_base",
        ImageNormalizationType.MINUS_ONE_TO_ONE.value,
        "siglip",
        id="siglip-vlm",
    ),
    pytest.param(
        "end_to_end_training_runs/libero_lerobot/pi0",
        ImageNormalizationType.MINUS_ONE_TO_ONE.value,
        "paligemma",
        id="paligemma-pi0",
    ),
    pytest.param(
        "end_to_end_training_runs/libero_lerobot/pi0_fast",
        ImageNormalizationType.MINUS_ONE_TO_ONE.value,
        "paligemma",
        id="paligemma-pi0-fast",
    ),
    pytest.param(
        "end_to_end_training_runs/libero_lerobot/pi05",
        ImageNormalizationType.MINUS_ONE_TO_ONE.value,
        "paligemma",
        id="paligemma-pi05",
    ),
    pytest.param(
        "end_to_end_training_runs/libero_lerobot/openvla",
        ImageNormalizationType.ZERO_TO_ONE.value,
        "prism",
        id="prismatic-openvla",
    ),
    pytest.param(
        "end_to_end_training_runs/libero_lerobot/openvla_oft",
        ImageNormalizationType.ZERO_TO_ONE.value,
        "prism",
        id="prismatic-openvla-oft",
    ),
    pytest.param(
        "end_to_end_training_runs/libero_lerobot/smolvla",
        ImageNormalizationType.MINUS_ONE_TO_ONE.value,
        "smolvlm",
        id="smolvlm-smolvla",
    ),
]


@pytest.mark.unit
class TestHydraComposition:
    @pytest.mark.parametrize("config_name", ALL_CONFIG_IDS)
    def test_composes_without_error(self, config_name: str):
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(config_name=config_name)
            assert config is not None

    def test_synthetic_dit_prior_uses_joint_dropout_fix_recipe(self) -> None:
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(config_name="end_to_end_training_runs/synthetic/lat_flow")

        prior = config.policy.algorithm.prior

        assert config.training.stages == []
        assert prior.prior_target_key == LatentKey.POSTERIOR_MU.value
        assert prior.latent_standardization_enabled is False
        assert prior.require_fitted_latent_standardization is False
        assert config.policy.loss.loss_modules.denoising_prior.weight == pytest.approx(
            0.01
        )
        assert (
            config.policy.loss.loss_modules.posterior_geometry.std_weight
            == pytest.approx(0.05)
        )

    @pytest.mark.parametrize("config_name", DIT_PRIOR_ALGORITHM_CONFIGS)
    def test_dit_prior_algorithm_configs_use_standardized_mu_targets(
        self,
        config_name: str,
    ) -> None:
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(config_name=config_name)

        prior = OmegaConf.select(config, "policy.algorithm.prior")

        assert prior is not None

        assert prior.prior_target_key == LatentKey.POSTERIOR_MU.value
        assert prior.latent_standardization_enabled is True
        assert prior.latent_standardization_eps == pytest.approx(1.0e-6)
        assert prior.latent_standardization_max_batches is None
        assert prior.require_fitted_latent_standardization is False

    @pytest.mark.parametrize(
        "config_name, expected_image_norm_type, expected_model_identifier",
        VLM_IMAGE_NORM_CONFIGS,
    )
    def test_vlm_runs_use_matching_image_normalization(
        self,
        config_name: str,
        expected_image_norm_type: str,
        expected_model_identifier: str,
    ) -> None:
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(config_name=config_name)

        assert config.task.dataloader.image_norm_type == expected_image_norm_type

        model_identifiers = []
        for encoder in config.policy.encoding_pipeline.encoders.values():
            for field in ("model_name", "backbone_name"):
                value = encoder.get(field)
                if isinstance(value, str):
                    model_identifiers.append(value.lower())
        vlm_backbone = config.policy.decoder.get("vlm_backbone")
        if vlm_backbone is not None:
            value = vlm_backbone.get("model_name")
            if isinstance(value, str):
                model_identifiers.append(value.lower())
        assert any(
            expected_model_identifier in identifier for identifier in model_identifiers
        )

    def test_pi0_fast_uses_paligemma_fast_action_tokens(self) -> None:
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(
                config_name="end_to_end_training_runs/libero_lerobot/pi0_fast"
            )

        tokenization = config.task.dataloader.tokenization
        action_tokenizer = tokenization.action_tokenizer

        assert tokenization.tokenize_observations is True
        assert tokenization.tokenize_actions is True
        assert tokenization.observation_tokenizer.observation_keys == [
            ObsKey.LANGUAGE.value,
            ProprioKey.EE_POS.value,
            ProprioKey.EE_ORI.value,
            ProprioKey.GRIPPER_STATE.value,
        ]
        assert (
            action_tokenizer.action_discretizer.type == ActionDiscretizerType.FAST.value
        )
        assert action_tokenizer.action_discretizer.use_pretrained is False
        assert (
            action_tokenizer.token_id_mapping.type
            == ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value
        )
        assert (
            action_tokenizer.token_id_mapping.language_tokenizer_model
            == config.policy.decoder.vlm_backbone.model_name
        )
        assert config.policy.encoding_pipeline.encoders == {}

    def test_openvla_and_pi0_fast_use_expected_prefix_attention(self) -> None:
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            openvla_preset = compose(config_name="policy/decoder/openvla")
            pi0_fast_config = compose(config_name="policy/decoder/pi0_fast")

        assert openvla_preset.policy.decoder.causal_prefix is True
        assert pi0_fast_config.policy.decoder.causal_prefix is False

    def test_openvla_libero_uses_prismatic_binned_language_tokens(self) -> None:
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(
                config_name="end_to_end_training_runs/libero_lerobot/openvla"
            )

        tokenization = config.task.dataloader.tokenization
        action_tokenizer = tokenization.action_tokenizer

        assert config.task.prediction_horizon == 8
        assert config.policy.encoding_pipeline.encoders == {}
        assert tokenization.tokenize_observations is True
        assert tokenization.tokenize_actions is True
        assert tokenization.observation_tokenizer.raw_text is True
        assert tokenization.observation_tokenizer.observation_keys == [
            ObsKey.LANGUAGE.value
        ]
        assert (
            action_tokenizer.action_discretizer.type
            == ActionDiscretizerType.BINNED.value
        )
        assert action_tokenizer.action_discretizer.num_bins == 256
        assert (
            action_tokenizer.token_id_mapping.type
            == ActionTokenIdMappingType.LANGUAGE_VOCABULARY.value
        )
        assert (
            action_tokenizer.token_id_mapping.language_tokenizer_model
            == tokenization.observation_tokenizer.tokenizer_model
        )
        assert action_tokenizer.token_id_mapping.num_special_tokens_to_skip == 0

    def test_openvla_oft_libero_uses_prismatic_l1_regression(self) -> None:
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(
                config_name="end_to_end_training_runs/libero_lerobot/openvla_oft"
            )

        tokenization = config.task.dataloader.tokenization
        decoder = config.policy.decoder
        loss_modules = config.policy.loss.loss_modules

        assert config.task.prediction_horizon == 8
        assert config.policy.encoding_pipeline.encoders == {}
        assert tokenization.tokenize_observations is True
        assert tokenization.tokenize_actions is False
        assert tokenization.observation_tokenizer.raw_text is True
        assert decoder.slots_per_action_dimension is True
        assert decoder.causal_action_slots is True
        assert list(decoder.action_heads.keys()) == ["joint_action"]
        action_head = decoder.action_heads.joint_action
        assert action_head.input_dim == 28672
        assert len(action_head.blocks) == 4
        assert (
            action_head.blocks[0]._target_
            == "versatil.models.decoding.action_heads.blocks.MLPBlock"
        )
        assert action_head.blocks[0].input_dim == 28672
        assert list(action_head.blocks[0].hidden_dims) == [4096]
        assert (
            action_head.blocks[-1]._target_
            == "versatil.models.decoding.action_heads.blocks.LayerNormBlock"
        )
        assert action_head.blocks[-1].input_dim == 4096
        assert "BehavioralCloning" in config.policy.algorithm._target_
        assert loss_modules.regression_loss.mse_weight == 0.0
        assert loss_modules.regression_loss.l1_weight == 1.0
        assert loss_modules.gripper_loss.mse_weight == 0.0
        assert loss_modules.gripper_loss.l1_weight == 0.05

    def test_pi05_libero_uses_adanorm_with_tokenized_proprio(self) -> None:
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(config_name="end_to_end_training_runs/libero_lerobot/pi05")

        decoder = config.policy.decoder
        tokenization = config.task.dataloader.tokenization

        assert config.task.prediction_horizon == 10
        assert config.policy.encoding_pipeline.encoders == {}
        assert decoder.input_keys == []
        assert decoder.time_conditioning == TimeConditioning.ADANORM.value
        assert decoder.proprioceptive_feature_key is None
        assert tokenization.observation_tokenizer.max_token_len == 200
        assert tokenization.observation_tokenizer.raw_text is False
        assert tokenization.observation_tokenizer.observation_keys == [
            ObsKey.LANGUAGE.value,
            ProprioKey.EE_POS.value,
            ProprioKey.EE_ORI.value,
            ProprioKey.GRIPPER_STATE.value,
        ]

    @pytest.mark.parametrize(
        (
            "config_name",
            "expected_target",
            "expected_max_batch_size",
        ),
        [
            pytest.param(
                "end_to_end_training_runs/libero_lerobot/"
                "action_transformer_finite_difference_lipschitz",
                f"{FiniteDifferenceLipschitzRegularizer.__module__}."
                f"{FiniteDifferenceLipschitzRegularizer.__name__}",
                8,
                id="finite-difference",
            ),
            pytest.param(
                "end_to_end_training_runs/libero_lerobot/"
                "action_transformer_spectral_jacobian_lipschitz",
                f"{SpectralJacobianLipschitzRegularizer.__module__}."
                f"{SpectralJacobianLipschitzRegularizer.__name__}",
                4,
                id="spectral-jacobian",
            ),
            pytest.param(
                "end_to_end_training_runs/libero_lerobot/"
                "action_transformer_jacobian_frobenius_lipschitz",
                f"{JacobianFrobeniusLipschitzRegularizer.__module__}."
                f"{JacobianFrobeniusLipschitzRegularizer.__name__}",
                4,
                id="jacobian-frobenius",
            ),
        ],
    )
    def test_libero_action_transformer_lipschitz_variants_target_visual_features(
        self,
        config_name: str,
        expected_target: str,
        expected_max_batch_size: int,
    ) -> None:
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(config_name=config_name)

        regularizer = config.policy.regularizers.visual_lipschitz

        assert regularizer._target_ == expected_target
        assert regularizer.input_domain == PolicyGraphInputDomain.ENCODED_FEATURES.value
        assert list(regularizer.input_keys) == ["left_rgb", "right_rgb"]
        assert regularizer.output_keys is None
        assert regularizer.detach_inputs is True
        assert regularizer.disable_decoder_stochastic is True
        assert regularizer.scale_by_dimension_ratio is False
        assert regularizer.max_batch_size == expected_max_batch_size
        assert list(config.policy.decoder.input_keys) == ["left_rgb", "right_rgb"]

    def test_libero_action_transformer_channel_broadcast_fd_targets_visual_features(
        self,
    ) -> None:
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(
                config_name="end_to_end_training_runs/libero_lerobot/"
                "action_transformer_finite_difference_lipschitz_channel_broadcast"
            )

        regularizer = config.policy.regularizers.visual_lipschitz

        assert (
            regularizer._target_
            == f"{FiniteDifferenceLipschitzRegularizer.__module__}."
            f"{FiniteDifferenceLipschitzRegularizer.__name__}"
        )
        assert regularizer.input_domain == PolicyGraphInputDomain.ENCODED_FEATURES.value
        assert list(regularizer.input_keys) == ["left_rgb", "right_rgb"]
        assert regularizer.weight == 1.0
        assert regularizer.target == 0.01
        assert regularizer.noise_scale == 0.01
        assert regularizer.perturbation_mode == "gaussian_channel_broadcast"
        assert regularizer.max_batch_size == 8

    def test_libero_action_transformer_image_augmentation_consistency_targets_rgb_observations(
        self,
    ) -> None:
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(
                config_name="end_to_end_training_runs/libero_lerobot/"
                "action_transformer_image_augmentation_consistency"
            )

        regularizer = config.policy.regularizers.visual_consistency

        assert (
            regularizer._target_
            == f"{ImageAugmentationConsistencyRegularizer.__module__}."
            f"{ImageAugmentationConsistencyRegularizer.__name__}"
        )
        assert list(regularizer.input_keys) == ["agentview_rgb", "eye_in_hand_rgb"]
        assert regularizer.output_keys is None
        assert regularizer.weight == 1.0
        assert (
            regularizer.loss_mode
            == ImageAugmentationConsistencyLossMode.POSITION_TRAJECTORY_L2.value
        )
        assert regularizer.max_batch_size == 8
        assert regularizer.detach_inputs is True
        assert regularizer.detach_targets is True
        assert len(regularizer.color_augmentation.transforms) == 7
        assert len(regularizer.spatial_augmentation.transforms) == 2

        geometric = config.policy.regularizers.agentview_geometric_consistency
        assert list(geometric.input_keys) == ["agentview_rgb"]
        assert geometric.color_augmentation is None
        assert len(geometric.spatial_augmentation.transforms) == 1
        assert (
            geometric.loss_mode
            == ImageAugmentationConsistencyLossMode.POSITION_TRAJECTORY_L2.value
        )

    def test_image_augmentation_regularizers_instantiate_albumentations_cleanly(
        self,
    ) -> None:
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(
                config_name="end_to_end_training_runs/libero_lerobot/"
                "action_transformer_image_augmentation_consistency"
            )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            visual = instantiate(config.policy.regularizers.visual_consistency)
            geometric = instantiate(
                config.policy.regularizers.agentview_geometric_consistency
            )

        compression = visual.color_augmentation.transforms[-1]
        assert compression.quality_range == (50, 100)
        dropout = visual.spatial_augmentation.transforms[-1]
        assert dropout.num_holes_range == (1, 8)
        assert dropout.hole_height_range == (8, 8)
        assert dropout.hole_width_range == (8, 8)
        assert geometric.color_augmentation is None
        assert len(geometric.spatial_augmentation.transforms) == 1

    @pytest.mark.parametrize(
        "preset_name", ["vae_frozen_prior", "vae_frozen_prior_with_backbone"]
    )
    def test_multistage_preset_validates_against_training_stage_config(
        self, preset_name: str
    ) -> None:
        preset_path = (
            Path(HYDRA_CONFIGS_ROOT) / "training" / "multistage" / f"{preset_name}.yaml"
        )
        raw = OmegaConf.load(preset_path)
        assert isinstance(raw, ListConfig)

        typed_parent = OmegaConf.structured(TrainingConfig())
        merged = OmegaConf.merge(typed_parent, OmegaConf.create({"stages": list(raw)}))

        for stage in merged.stages:
            assert OmegaConf.get_type(stage) is TrainingStageConfig


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_name, overrides, expectation",
    [
        pytest.param(
            "training/default",
            ["++num_epochs=5"],
            does_not_raise(),
            id="training-valid-num_epochs",
        ),
        pytest.param(
            "training/default",
            ["++use_ema=true"],
            does_not_raise(),
            id="training-valid-use_ema",
        ),
        pytest.param(
            "training/default",
            ["++clip_gradient_norm=true"],
            does_not_raise(),
            id="training-valid-clip_gradient_norm",
        ),
        pytest.param(
            "training/default",
            ["++clip_max_norm=5.0"],
            does_not_raise(),
            id="training-valid-clip_max_norm",
        ),
        pytest.param(
            "policy/algorithm/diffusion",
            ["++num_train_timesteps=200"],
            does_not_raise(),
            id="diffusion-valid-num_train_timesteps",
        ),
        pytest.param(
            "policy/algorithm/diffusion",
            ["++num_inference_steps=50"],
            does_not_raise(),
            id="diffusion-valid-num_inference_steps",
        ),
        pytest.param(
            "policy/algorithm/diffusion",
            ["++clip_sample=false"],
            does_not_raise(),
            id="diffusion-valid-clip_sample",
        ),
        pytest.param(
            "policy/algorithm/flow_matching",
            ["++num_inference_steps=20"],
            does_not_raise(),
            id="flow_matching-valid-num_inference_steps",
        ),
        pytest.param(
            "policy/algorithm/flow_matching",
            ["++sigma=0.1"],
            does_not_raise(),
            id="flow_matching-valid-sigma",
        ),
        pytest.param(
            "training/optimizer/adamw",
            ["++lr=0.01"],
            does_not_raise(),
            id="adamw-valid-lr",
        ),
        pytest.param(
            "training/optimizer/adamw",
            ["++weight_decay=0.001"],
            does_not_raise(),
            id="adamw-valid-weight_decay",
        ),
        pytest.param(
            "training/optimizer/adam",
            ["++lr=0.001"],
            does_not_raise(),
            id="adam-valid-lr",
        ),
        pytest.param(
            "training/optimizer/sgd",
            ["++lr=0.05"],
            does_not_raise(),
            id="sgd-valid-lr",
        ),
        pytest.param(
            "training/optimizer/sgd",
            ["++momentum=0.99"],
            does_not_raise(),
            id="sgd-valid-momentum",
        ),
        pytest.param(
            "training/default",
            ["fake_key=1"],
            pytest.raises(ConfigCompositionException),
            id="training_default-unknown_key-fake_key",
        ),
        pytest.param(
            "training/ema_cosine_schedule",
            ["bogus=true"],
            pytest.raises(ConfigCompositionException),
            id="training_ema_cosine-unknown_key-bogus",
        ),
        pytest.param(
            "policy/algorithm/behavioral_cloning",
            ["nonexistent=5"],
            pytest.raises(ConfigCompositionException),
            id="behavioral_cloning-unknown_key-nonexistent",
        ),
        pytest.param(
            "policy/algorithm/diffusion",
            ["wrong_param=10"],
            pytest.raises(ConfigCompositionException),
            id="diffusion-unknown_key-wrong_param",
        ),
        pytest.param(
            "policy/algorithm/flow_matching",
            ["bad_key=0.5"],
            pytest.raises(ConfigCompositionException),
            id="flow_matching-unknown_key-bad_key",
        ),
        pytest.param(
            "training/optimizer/adamw",
            ["invalid_field=0.1"],
            pytest.raises(ConfigCompositionException),
            id="adamw-unknown_key-invalid_field",
        ),
        pytest.param(
            "training/optimizer/sgd",
            ["unknown=true"],
            pytest.raises(ConfigCompositionException),
            id="sgd-unknown_key-unknown",
        ),
        pytest.param(
            "training/default",
            ["use_ema=not_a_bool"],
            pytest.raises(ConfigCompositionException),
            id="training-wrong_type-use_ema_not_bool",
        ),
        pytest.param(
            "training/default",
            ["num_epochs=not_an_int"],
            pytest.raises(ConfigCompositionException),
            id="training-wrong_type-num_epochs_not_int",
        ),
        pytest.param(
            "training/default",
            ["clip_max_norm=not_a_float"],
            pytest.raises(ConfigCompositionException),
            id="training-wrong_type-clip_max_norm_not_float",
        ),
        pytest.param(
            "policy/algorithm/diffusion",
            ["num_train_timesteps=abc"],
            pytest.raises(ConfigCompositionException),
            id="diffusion-wrong_type-num_train_timesteps_not_int",
        ),
        pytest.param(
            "policy/algorithm/diffusion",
            ["clip_sample=maybe"],
            pytest.raises(ConfigCompositionException),
            id="diffusion-wrong_type-clip_sample_not_bool",
        ),
        pytest.param(
            "policy/algorithm/diffusion",
            ["scheduler_type=nonexistent_scheduler"],
            pytest.raises(ConfigCompositionException),
            id="diffusion-wrong_enum-scheduler_type",
        ),
        pytest.param(
            "policy/algorithm/flow_matching",
            ["ode_solver=invalid_solver"],
            pytest.raises(ConfigCompositionException),
            id="flow_matching-wrong_enum-ode_solver",
        ),
    ],
)
def test_override_validation(
    config_name: str,
    overrides: list[str],
    expectation: AbstractContextManager[None],
):
    with (
        initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None),
        expectation,
    ):
        compose(config_name=config_name, overrides=overrides)
