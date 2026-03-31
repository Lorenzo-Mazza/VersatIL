"""Tests for versatil.inference.policy_loader module."""

import logging
import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from versatil.data.constants import Cameras
from versatil.inference.policy_loading.float_loader import PolicyLoader
from versatil.training.constants import PrecisionType

BASE_LOADER_MODULE = "versatil.inference.policy_loading.base"
FLOAT_LOADER_MODULE = "versatil.inference.policy_loading.float_loader"


@pytest.fixture(autouse=True)
def _patch_torch_compile():
    # torch.compile on MagicMock causes infinite dynamo graph expansion.
    with patch(
        f"{FLOAT_LOADER_MODULE}.torch.compile",
        side_effect=lambda model, **kwargs: model,
    ):
        yield


@pytest.fixture
def mock_config() -> MagicMock:
    config = MagicMock()
    config.training = MagicMock()
    config.experiment.validate_loss_keys = True
    config.task.dataloader.tokenization.tokenize_observations = False
    mock_policy = MagicMock()
    mock_policy.observation_space = MagicMock()
    mock_policy.action_space = MagicMock()
    mock_policy.prediction_horizon = 16
    mock_policy.decoder.observation_horizon = 2
    mock_policy.to.return_value = mock_policy
    mock_policy.eval.return_value = mock_policy
    config.policy = mock_policy
    return config


@pytest.fixture
def mock_checkpoint() -> dict:
    return {
        "state_dict": {
            "policy.decoder.weight": torch.tensor([1.0, 2.0]),
            "policy.encoding_pipeline.weight": torch.tensor([3.0, 4.0]),
        }
    }


@pytest.fixture
def policy_loader_factory(
    tmp_path, mock_config, mock_checkpoint
) -> Callable[..., PolicyLoader]:
    def factory(
        device: torch.device = torch.device("cpu"),
        checkpoint_name: str = "last.ckpt",
        precision: str = PrecisionType.BF16_MIXED.value,
        seed: int = 42,
        config: MagicMock | None = None,
        checkpoint: dict | None = None,
        create_tokenizer_dir: bool = False,
        compile_model: bool = False,
    ) -> PolicyLoader:
        if config is None:
            config = mock_config
        if checkpoint is None:
            checkpoint = mock_checkpoint

        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: true")
        checkpoint_file = tmp_path / checkpoint_name
        checkpoint_file.write_text("dummy")

        if create_tokenizer_dir:
            tokenizer_dir = tmp_path / "tokenizer"
            tokenizer_dir.mkdir(exist_ok=True)

        mock_lightning = MagicMock()
        mock_lightning.state_dict.return_value = dict(checkpoint["state_dict"])

        with (
            patch(
                f"{BASE_LOADER_MODULE}.OmegaConf.load",
                return_value=MagicMock(),
            ),
            patch(
                f"{BASE_LOADER_MODULE}.hydra.utils.instantiate",
                return_value=config,
            ),
            patch(f"{BASE_LOADER_MODULE}.validate_experiment"),
            patch(
                f"{FLOAT_LOADER_MODULE}.torch.load",
                return_value=checkpoint,
            ),
            patch(
                f"{FLOAT_LOADER_MODULE}.LightningPolicy",
                return_value=mock_lightning,
            ),
            patch(
                f"{BASE_LOADER_MODULE}.Tokenizer.from_pretrained",
                return_value=MagicMock(),
            ),
        ):
            loader = PolicyLoader(
                device=device,
                checkpoint_path=str(tmp_path),
                checkpoint_name=checkpoint_name,
                precision=precision,
                seed=seed,
                compile_model=compile_model,
            )
        return loader

    return factory


@pytest.mark.unit
class TestPolicyLoaderInitialization:
    @pytest.mark.parametrize(
        "precision",
        [
            PrecisionType.BF16_MIXED.value,
            PrecisionType.FP32.value,
        ],
    )
    @pytest.mark.parametrize("seed", [42, 123])
    def test_stores_configuration(self, policy_loader_factory, precision, seed):
        loader = policy_loader_factory(precision=precision, seed=seed)
        assert loader._precision == precision
        assert loader._device == torch.device("cpu")

    def test_config_file_not_found_raises(self, tmp_path):
        with pytest.raises(
            FileNotFoundError,
            match=re.escape(f"Config file not found at {tmp_path / 'config.yaml'}."),
        ):
            PolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=str(tmp_path),
            )

    def test_checkpoint_file_not_found_raises(self, tmp_path, mock_config):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: true")

        with (
            patch(
                f"{BASE_LOADER_MODULE}.OmegaConf.load",
                return_value=MagicMock(),
            ),
            patch(
                f"{BASE_LOADER_MODULE}.hydra.utils.instantiate",
                return_value=mock_config,
            ),
            patch(f"{BASE_LOADER_MODULE}.validate_experiment"),
            pytest.raises(
                FileNotFoundError,
                match=re.escape(f"No checkpoint found at {tmp_path / 'last.ckpt'}."),
            ),
        ):
            PolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=str(tmp_path),
            )


@pytest.mark.unit
class TestPolicyLoaderSeedBehavior:
    def test_set_seed_uses_default_rng_not_global_seed(self, policy_loader_factory):
        loader = policy_loader_factory(seed=99)
        assert loader._rng.random() >= 0

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_torch_manual_seed_produces_deterministic_values(
        self, policy_loader_factory, seed
    ):
        _loader = policy_loader_factory(seed=seed)
        # Using torch.randn directly (not rng fixture) because this test
        # verifies torch.manual_seed global seed determinism specifically.
        value_a = torch.randn(1).item()
        torch.manual_seed(seed)
        value_b = torch.randn(1).item()
        assert value_a == value_b


@pytest.mark.unit
class TestPolicyLoaderProperties:
    def test_device_returns_configured_device(self, policy_loader_factory):
        loader = policy_loader_factory(device=torch.device("cpu"))
        assert loader.device == torch.device("cpu")

    def test_checkpoint_path_returns_configured_path(
        self, policy_loader_factory, tmp_path
    ):
        loader = policy_loader_factory()
        assert loader.checkpoint_path == str(tmp_path)

    def test_policy_property_exposes_loaded_policy(self, policy_loader_factory):
        loader = policy_loader_factory()
        loader._policy.some_attribute = "test_value"
        assert loader.policy.some_attribute == "test_value"

    def test_config_property_exposes_loaded_config(self, policy_loader_factory):
        loader = policy_loader_factory()
        loader._config.some_attribute = "test_value"
        assert loader.config.some_attribute == "test_value"

    def test_observation_space_delegates_to_policy(self, policy_loader_factory):
        loader = policy_loader_factory()
        expected = loader._policy.observation_space
        assert loader.observation_space == expected

    def test_action_space_delegates_to_policy(self, policy_loader_factory):
        loader = policy_loader_factory()
        expected = loader._policy.action_space
        assert loader.action_space == expected

    def test_prediction_horizon_delegates_to_policy(self, policy_loader_factory):
        loader = policy_loader_factory()
        assert loader.prediction_horizon == 16

    def test_observation_horizon_delegates_to_decoder(self, policy_loader_factory):
        loader = policy_loader_factory()
        assert loader.observation_horizon == 2


@pytest.mark.unit
class TestPolicyLoaderTokenizer:
    def test_tokenizer_is_none_when_no_tokenizer_directory(self, policy_loader_factory):
        loader = policy_loader_factory(create_tokenizer_dir=False)
        mock_config = loader._config
        mock_config.policy.set_tokenizer.assert_not_called()

    def test_raises_when_tokenization_required_but_no_directory(
        self, policy_loader_factory, mock_config
    ):
        mock_config.task.dataloader.tokenization.tokenize_observations = True
        with pytest.raises(
            FileNotFoundError,
            match="Config requires observation tokenization but no tokenizer found",
        ):
            policy_loader_factory(config=mock_config, create_tokenizer_dir=False)

    def test_tokenizer_loaded_when_directory_exists(
        self, tmp_path, mock_config, mock_checkpoint
    ):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: true")
        checkpoint_file = tmp_path / "last.ckpt"
        checkpoint_file.write_text("dummy")
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()

        mock_tokenizer = MagicMock()
        mock_tokenizer.to.return_value = mock_tokenizer

        mock_lightning = MagicMock()
        mock_lightning.state_dict.return_value = dict(mock_checkpoint["state_dict"])

        with (
            patch(
                f"{BASE_LOADER_MODULE}.OmegaConf.load",
                return_value=MagicMock(),
            ),
            patch(
                f"{BASE_LOADER_MODULE}.hydra.utils.instantiate",
                return_value=mock_config,
            ),
            patch(f"{BASE_LOADER_MODULE}.validate_experiment"),
            patch(
                f"{FLOAT_LOADER_MODULE}.torch.load",
                return_value=mock_checkpoint,
            ),
            patch(
                f"{FLOAT_LOADER_MODULE}.LightningPolicy",
                return_value=mock_lightning,
            ),
            patch(
                f"{BASE_LOADER_MODULE}.Tokenizer.from_pretrained",
                return_value=mock_tokenizer,
            ),
        ):
            loader = PolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=str(tmp_path),
            )

        mock_config.policy.set_tokenizer.assert_called_once_with(mock_tokenizer)
        mock_tokenizer.to.assert_called_once_with(torch.device("cpu"))
        assert loader.tokenizer == mock_tokenizer


@pytest.mark.unit
class TestPolicyLoaderPrecisionConversion:
    @pytest.mark.parametrize(
        "precision, should_convert",
        [
            (PrecisionType.BF16_MIXED.value, True),
            (PrecisionType.FP16_MIXED.value, True),
            (PrecisionType.FP32.value, False),
        ],
    )
    def test_precision_conversion_applied_when_needed(
        self,
        tmp_path,
        mock_config,
        mock_checkpoint,
        precision,
        should_convert,
    ):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: true")
        checkpoint_file = tmp_path / "last.ckpt"
        checkpoint_file.write_text("dummy")

        mock_policy = mock_config.policy
        mock_policy.to.return_value = mock_policy
        mock_policy.eval.return_value = mock_policy

        mock_lightning = MagicMock()
        mock_lightning.state_dict.return_value = dict(mock_checkpoint["state_dict"])

        with (
            patch(
                f"{BASE_LOADER_MODULE}.OmegaConf.load",
                return_value=MagicMock(),
            ),
            patch(
                f"{BASE_LOADER_MODULE}.hydra.utils.instantiate",
                return_value=mock_config,
            ),
            patch(f"{BASE_LOADER_MODULE}.validate_experiment"),
            patch(
                f"{FLOAT_LOADER_MODULE}.torch.load",
                return_value=mock_checkpoint,
            ),
            patch(
                f"{FLOAT_LOADER_MODULE}.LightningPolicy",
                return_value=mock_lightning,
            ),
        ):
            _loader = PolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=str(tmp_path),
                precision=precision,
            )

        dtype_call_args = [
            call
            for call in mock_policy.to.call_args_list
            if len(call.args) > 0 and isinstance(call.args[0], torch.dtype)
        ]
        if should_convert:
            expected_dtype = PrecisionType(precision).get_model_dtype()
            assert any(call.args[0] == expected_dtype for call in dtype_call_args), (
                f"Expected .to({expected_dtype}) to be called, "
                f"but calls were: {mock_policy.to.call_args_list}"
            )
        else:
            assert len(dtype_call_args) == 0, (
                f"Expected no dtype conversion for {precision}, "
                f"but .to() was called with dtype args: {dtype_call_args}"
            )


@pytest.mark.unit
class TestPolicyLoaderCheckpointValidation:
    def test_raises_on_critical_prefix_mismatch(self, tmp_path, mock_config):
        checkpoint_state = {
            "policy.decoder.layer.weight": torch.tensor([1.0]),
            "policy.decoder.layer.bias": torch.tensor([2.0]),
        }
        mock_checkpoint = {"state_dict": checkpoint_state}

        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: true")
        checkpoint_file = tmp_path / "last.ckpt"
        checkpoint_file.write_text("dummy")

        mock_lightning = MagicMock()
        mock_lightning.state_dict.return_value = {
            "policy.other.weight": torch.tensor([5.0]),
        }

        with (
            patch(
                f"{BASE_LOADER_MODULE}.OmegaConf.load",
                return_value=MagicMock(),
            ),
            patch(
                f"{BASE_LOADER_MODULE}.hydra.utils.instantiate",
                return_value=mock_config,
            ),
            patch(f"{BASE_LOADER_MODULE}.validate_experiment"),
            patch(
                f"{FLOAT_LOADER_MODULE}.torch.load",
                return_value=mock_checkpoint,
            ),
            patch(
                f"{FLOAT_LOADER_MODULE}.LightningPolicy",
                return_value=mock_lightning,
            ),
            pytest.raises(
                RuntimeError,
                match=re.escape(
                    "Checkpoint loading validation failed with 1 critical error(s)."
                ),
            ),
        ):
            PolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=str(tmp_path),
            )

    def test_raises_on_lazy_module_prefix_mismatch(self, tmp_path, mock_config):
        lazy_prefix = (
            "policy.decoder.architecture.feature_projection."
            "linear_projections.proj.weight"
        )
        checkpoint_state = {
            lazy_prefix: torch.tensor([1.0]),
        }
        mock_checkpoint = {"state_dict": checkpoint_state}

        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: true")
        checkpoint_file = tmp_path / "last.ckpt"
        checkpoint_file.write_text("dummy")

        mock_lightning = MagicMock()
        mock_lightning.state_dict.return_value = {
            "policy.decoder.other.weight": torch.tensor([5.0]),
        }

        with (
            patch(
                f"{BASE_LOADER_MODULE}.OmegaConf.load",
                return_value=MagicMock(),
            ),
            patch(
                f"{BASE_LOADER_MODULE}.hydra.utils.instantiate",
                return_value=mock_config,
            ),
            patch(f"{BASE_LOADER_MODULE}.validate_experiment"),
            patch(
                f"{FLOAT_LOADER_MODULE}.torch.load",
                return_value=mock_checkpoint,
            ),
            patch(
                f"{FLOAT_LOADER_MODULE}.LightningPolicy",
                return_value=mock_lightning,
            ),
            pytest.raises(
                RuntimeError,
                match=re.escape(
                    "Checkpoint loading validation failed with "
                    "1 critical error(s). "
                    "The model will NOT produce correct outputs. "
                    "First error: CRITICAL: FeatureProjection linear "
                    "failed to load. Checkpoint has 1 keys but model "
                    "has NONE. Example keys: "
                    f"['{lazy_prefix}']"
                ),
            ),
        ):
            PolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=str(tmp_path),
            )

    def test_raises_on_weight_value_mismatch(self, tmp_path, mock_config):
        shared_key = "policy.decoder.weight"
        checkpoint_state = {
            shared_key: torch.tensor([1.0, 2.0]),
        }
        mock_checkpoint = {"state_dict": checkpoint_state}

        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: true")
        checkpoint_file = tmp_path / "last.ckpt"
        checkpoint_file.write_text("dummy")

        mock_lightning = MagicMock()
        mock_lightning.state_dict.return_value = {
            shared_key: torch.tensor([99.0, 99.0]),
        }

        with (
            patch(
                f"{BASE_LOADER_MODULE}.OmegaConf.load",
                return_value=MagicMock(),
            ),
            patch(
                f"{BASE_LOADER_MODULE}.hydra.utils.instantiate",
                return_value=mock_config,
            ),
            patch(f"{BASE_LOADER_MODULE}.validate_experiment"),
            patch(
                f"{FLOAT_LOADER_MODULE}.torch.load",
                return_value=mock_checkpoint,
            ),
            patch(
                f"{FLOAT_LOADER_MODULE}.LightningPolicy",
                return_value=mock_lightning,
            ),
            pytest.raises(
                RuntimeError,
                match=re.escape(f"Weight mismatch for '{shared_key}'"),
            ),
        ):
            PolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=str(tmp_path),
            )

    def test_passes_when_checkpoint_matches_model(self, policy_loader_factory):
        loader = policy_loader_factory()
        assert loader.policy.prediction_horizon == 16


@pytest.mark.unit
class TestPolicyLoaderCompileFlag:
    @pytest.mark.parametrize("compile_model", [True, False])
    def test_compile_flag_controls_policy_compilation(
        self,
        tmp_path,
        mock_config,
        mock_checkpoint,
        compile_model,
    ):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: true")
        (tmp_path / "last.ckpt").write_text("dummy")

        mock_lightning = MagicMock()
        mock_lightning.state_dict.return_value = dict(mock_checkpoint["state_dict"])
        compiled_policy = MagicMock()

        with (
            patch(f"{BASE_LOADER_MODULE}.OmegaConf.load", return_value=MagicMock()),
            patch(
                f"{BASE_LOADER_MODULE}.hydra.utils.instantiate",
                return_value=mock_config,
            ),
            patch(f"{BASE_LOADER_MODULE}.validate_experiment"),
            patch(f"{FLOAT_LOADER_MODULE}.torch.load", return_value=mock_checkpoint),
            patch(
                f"{FLOAT_LOADER_MODULE}.LightningPolicy", return_value=mock_lightning
            ),
            patch(
                f"{BASE_LOADER_MODULE}.Tokenizer.from_pretrained",
                return_value=MagicMock(),
            ),
            patch(
                f"{FLOAT_LOADER_MODULE}.torch.compile", return_value=compiled_policy
            ) as mock_compile,
        ):
            PolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=str(tmp_path),
                compile_model=compile_model,
            )

        assert mock_compile.called == compile_model


@pytest.mark.unit
class TestPolicyLoaderRunInference:
    def test_run_inference_calls_predict_action(self, policy_loader_factory, rng):
        loader = policy_loader_factory()
        mock_obs = {
            "left": torch.from_numpy(
                rng.standard_normal((1, 1, 3, 64, 64)).astype(np.float32)
            )
        }
        expected_result = {
            "position": torch.from_numpy(
                rng.standard_normal((1, 16, 3)).astype(np.float32)
            )
        }
        loader._policy.predict_action.return_value = expected_result

        result = loader.run_inference(obs_dict=mock_obs)

        loader._policy.predict_action.assert_called_once_with(obs_dict=mock_obs)
        assert torch.equal(result["position"], expected_result["position"])

    @pytest.mark.parametrize(
        "precision, expected_dtype",
        [
            (PrecisionType.BF16_MIXED.value, torch.bfloat16),
            (PrecisionType.FP16_MIXED.value, torch.float16),
            (PrecisionType.FP32.value, torch.float32),
        ],
    )
    def test_run_inference_uses_correct_autocast_dtype(
        self, policy_loader_factory, rng, precision, expected_dtype
    ):
        loader = policy_loader_factory(precision=precision)
        captured_dtype = []

        def capturing_predict(obs_dict):
            captured_dtype.append(torch.get_autocast_dtype("cpu"))
            return {
                "position": torch.from_numpy(
                    rng.standard_normal((1, 16, 3)).astype(np.float32)
                )
            }

        loader._policy.predict_action = capturing_predict

        loader.run_inference(
            obs_dict={
                "dummy": torch.from_numpy(rng.standard_normal((1,)).astype(np.float32))
            }
        )

        assert len(captured_dtype) == 1
        assert captured_dtype[0] == expected_dtype

    def test_run_inference_produces_no_gradients(self, policy_loader_factory, rng):
        loader = policy_loader_factory()
        weight = torch.nn.Parameter(
            torch.from_numpy(rng.standard_normal((3, 3)).astype(np.float32))
        )

        def mock_predict(obs_dict):
            return {
                "action": weight
                @ torch.from_numpy(rng.standard_normal((3, 1)).astype(np.float32))
            }

        loader._policy.predict_action = mock_predict

        result = loader.run_inference(
            obs_dict={
                "dummy": torch.from_numpy(rng.standard_normal((1,)).astype(np.float32))
            }
        )
        assert not result["action"].requires_grad


@pytest.mark.unit
class TestDenoisingThresholds:
    def test_empty_params_dict_returns_empty(self, policy_loader_factory):
        loader = policy_loader_factory()
        loader._policy.denoising_thresholds.params_dict = nn.ParameterDict()
        loader._policy.action_space.actions_metadata = {
            "position": MagicMock(),
        }
        assert loader.denoising_thresholds == {}

    def test_all_keys_match_action_space(self, policy_loader_factory, rng):
        loader = policy_loader_factory()
        position_value = float(rng.standard_normal(1).astype(np.float32).item())
        orientation_value = float(rng.standard_normal(1).astype(np.float32).item())
        loader._policy.denoising_thresholds.params_dict = nn.ParameterDict(
            {
                "position": nn.Parameter(
                    torch.tensor(position_value), requires_grad=False
                ),
                "orientation": nn.Parameter(
                    torch.tensor(orientation_value), requires_grad=False
                ),
            }
        )
        loader._policy.action_space.actions_metadata = {
            "position": MagicMock(),
            "orientation": MagicMock(),
        }
        result = loader.denoising_thresholds
        assert result == {
            "position": pytest.approx(position_value),
            "orientation": pytest.approx(orientation_value),
        }

    def test_keys_not_in_action_space_are_filtered_out(
        self, policy_loader_factory, rng
    ):
        loader = policy_loader_factory()
        threshold_value = float(rng.standard_normal(1).astype(np.float32).item())
        loader._policy.denoising_thresholds.params_dict = nn.ParameterDict(
            {
                "unknown_key": nn.Parameter(
                    torch.tensor(threshold_value), requires_grad=False
                ),
            }
        )
        loader._policy.action_space.actions_metadata = {
            "position": MagicMock(),
        }
        assert loader.denoising_thresholds == {}

    def test_mixed_keys_returns_only_matching(self, policy_loader_factory, rng):
        loader = policy_loader_factory()
        position_value = float(rng.standard_normal(1).astype(np.float32).item())
        extra_value = float(rng.standard_normal(1).astype(np.float32).item())
        loader._policy.denoising_thresholds.params_dict = nn.ParameterDict(
            {
                "position": nn.Parameter(
                    torch.tensor(position_value), requires_grad=False
                ),
                "nonexistent": nn.Parameter(
                    torch.tensor(extra_value), requires_grad=False
                ),
            }
        )
        loader._policy.action_space.actions_metadata = {
            "position": MagicMock(),
            "gripper": MagicMock(),
        }
        result = loader.denoising_thresholds
        assert result == {"position": pytest.approx(position_value)}
        assert "nonexistent" not in result
        assert "gripper" not in result


@pytest.mark.unit
class TestDepthClampRange:
    def test_depth_key_absent_returns_none(self, policy_loader_factory):
        loader = policy_loader_factory()
        loader._policy.normalizer.params_dict = nn.ParameterDict()
        assert loader.depth_clamp_range is None

    def test_depth_key_present_but_input_stats_none_returns_none(
        self, policy_loader_factory
    ):
        loader = policy_loader_factory()
        depth_key = Cameras.DEPTH.value
        depth_params = nn.ParameterDict()
        loader._policy.normalizer.params_dict = nn.ParameterDict(
            {depth_key: depth_params}
        )
        mock_single_field = MagicMock()
        mock_single_field.params_dict.get.return_value = None
        with patch.object(
            type(loader._policy.normalizer),
            "__getitem__",
            return_value=mock_single_field,
        ):
            assert loader.depth_clamp_range is None

    def test_depth_key_with_valid_stats_returns_min_max_tuple(
        self, policy_loader_factory
    ):
        loader = policy_loader_factory()
        depth_key = Cameras.DEPTH.value
        depth_params = nn.ParameterDict()
        loader._policy.normalizer.params_dict = nn.ParameterDict(
            {depth_key: depth_params}
        )
        min_value = 0.1
        max_value = 5.0
        mock_stats = {
            "min": torch.tensor(min_value),
            "max": torch.tensor(max_value),
        }
        mock_single_field = MagicMock()
        mock_single_field.params_dict.get.return_value = mock_stats
        with patch.object(
            type(loader._policy.normalizer),
            "__getitem__",
            return_value=mock_single_field,
        ):
            result = loader.depth_clamp_range
        assert result == (
            pytest.approx(min_value),
            pytest.approx(max_value),
        )


@pytest.mark.unit
class TestCheckpointValidationWarning:
    def test_logs_warning_when_checkpoint_has_more_keys_than_model(
        self, tmp_path, mock_config, caplog
    ):
        checkpoint_state = {
            "policy.decoder.layer_a.weight": torch.tensor([1.0]),
            "policy.decoder.layer_a.bias": torch.tensor([2.0]),
            "policy.decoder.layer_b.weight": torch.tensor([3.0]),
        }
        mock_checkpoint = {"state_dict": checkpoint_state}

        config_file = tmp_path / "config.yaml"
        config_file.write_text("dummy: true")
        checkpoint_file = tmp_path / "last.ckpt"
        checkpoint_file.write_text("dummy")

        model_state = {
            "policy.decoder.layer_a.weight": torch.tensor([1.0]),
            "policy.decoder.layer_a.bias": torch.tensor([2.0]),
            "policy.encoding_pipeline.weight": torch.tensor([3.0, 4.0]),
        }
        mock_lightning = MagicMock()
        mock_lightning.state_dict.return_value = model_state

        with (
            patch(
                f"{BASE_LOADER_MODULE}.OmegaConf.load",
                return_value=MagicMock(),
            ),
            patch(
                f"{BASE_LOADER_MODULE}.hydra.utils.instantiate",
                return_value=mock_config,
            ),
            patch(f"{BASE_LOADER_MODULE}.validate_experiment"),
            patch(
                f"{FLOAT_LOADER_MODULE}.torch.load",
                return_value=mock_checkpoint,
            ),
            patch(
                f"{FLOAT_LOADER_MODULE}.LightningPolicy",
                return_value=mock_lightning,
            ),
            caplog.at_level(logging.WARNING),
        ):
            _loader = PolicyLoader(
                device=torch.device("cpu"),
                checkpoint_path=str(tmp_path),
            )

        assert "policy.decoder." in caplog.text
        assert "Matched: 2/3" in caplog.text
