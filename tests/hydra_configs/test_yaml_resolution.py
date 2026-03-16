"""Tests for Hydra YAML configuration composition and validation."""
import glob
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException

import versatil.configs  # noqa: F401 — registers ConfigStore entries

HYDRA_CONFIGS_ROOT = str(Path(__file__).parents[2] / "hydra_configs")

ALL_YAML_FILES = sorted(glob.glob(
    str(Path(HYDRA_CONFIGS_ROOT) / "**" / "*.yaml"),
    recursive=True,
))

ALL_CONFIG_IDS = [
    str(Path(p).relative_to(HYDRA_CONFIGS_ROOT)).removesuffix(".yaml")
    for p in ALL_YAML_FILES
]


@pytest.mark.unit
class TestHydraComposition:

    @pytest.mark.parametrize("config_name", ALL_CONFIG_IDS)
    def test_composes_without_error(self, config_name: str):
        with initialize_config_dir(config_dir=HYDRA_CONFIGS_ROOT, version_base=None):
            config = compose(config_name=config_name)
            assert config is not None


@pytest.mark.unit
class TestHydraConfigValidation:

    @pytest.mark.parametrize(
        "config_name, overrides, expectation",
        [
            # --- VALID OVERRIDES: training ---
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
            # --- VALID OVERRIDES: algorithm/diffusion ---
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
            # --- VALID OVERRIDES: algorithm/flow_matching ---
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
            # --- VALID OVERRIDES: optimizer/adamw ---
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
            # --- VALID OVERRIDES: optimizer/adam ---
            pytest.param(
                "training/optimizer/adam",
                ["++lr=0.001"],
                does_not_raise(),
                id="adam-valid-lr",
            ),
            # --- VALID OVERRIDES: optimizer/sgd ---
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
            # --- INVALID OVERRIDES: unknown keys ---
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
            # --- INVALID OVERRIDES: wrong types ---
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
            # --- INVALID OVERRIDES: wrong enum values ---
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
        self,
        config_name: str,
        overrides: list[str],
        expectation: object,
    ):
        with initialize_config_dir(
            config_dir=HYDRA_CONFIGS_ROOT, version_base=None
        ):
            with expectation:
                compose(config_name=config_name, overrides=overrides)
