"""Tests for versatil.data.tokenization.observation_tokenizer."""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.data.constants import ObsKey, ProprioKey, SampleKey
from versatil.data.tokenization.observation_tokenizer import ObservationTokenizer


@pytest.fixture
def mock_obs_auto_tokenizer():
    """Patches AutoTokenizer in observation_tokenizer module."""
    with patch(
        "versatil.data.tokenization.observation_tokenizer.AutoTokenizer"
    ) as mock:
        mock.from_pretrained.return_value = MagicMock(
            vocab_size=30000, pad_token="[PAD]"
        )
        yield mock


@pytest.fixture
def observation_tokenizer_factory(mock_obs_auto_tokenizer):
    """Factory for ObservationTokenizer with AutoTokenizer mocked."""

    def factory(
        tokenizer_model: str = "test-model",
        observation_keys: list[str] | None = None,
        bin_continuous_data: bool = True,
        num_bins: int = 256,
        max_token_len: int = 256,
        device: torch.device | None = None,
    ) -> ObservationTokenizer:
        if observation_keys is None:
            observation_keys = [ObsKey.LANGUAGE.value]
        return ObservationTokenizer(
            tokenizer_model=tokenizer_model,
            observation_keys=observation_keys,
            bin_continuous_data=bin_continuous_data,
            num_bins=num_bins,
            max_token_len=max_token_len,
            device=device,
        )

    return factory


@pytest.fixture
def mock_language_tokenizer_result():
    """Factory for mock tokenizer __call__ result."""

    def factory(batch_size: int = 2, max_length: int = 16) -> dict:
        return {
            "input_ids": torch.ones((batch_size, max_length), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, max_length), dtype=torch.long),
        }

    return factory


@pytest.fixture
def observation_dict_factory(rng):
    """Factory for observation dicts with optional language and proprioceptive data."""

    def factory(
        language: Any = None,
        proprio_keys: list[str] | None = None,
        batch_size: int = 1,
        observation_dim: int = 7,
        as_torch: bool = False,
    ) -> dict[str, str | list | np.ndarray | torch.Tensor]:
        observations = {}
        if language is not None:
            observations[ObsKey.LANGUAGE.value] = language
        if proprio_keys is not None:
            for key in proprio_keys:
                data = rng.standard_normal((batch_size, observation_dim)).astype(
                    np.float32
                )
                if as_torch:
                    data = torch.from_numpy(data)
                observations[key] = data
        return observations

    return factory


@pytest.fixture
def training_data_factory(rng):
    """Factory for training data dicts used to fit binning tokenizers."""

    def factory(
        proprio_keys: list[str] | None = None,
        num_samples: int = 50,
        observation_dim: int = 7,
    ) -> dict[str, np.ndarray]:
        if proprio_keys is None:
            proprio_keys = [ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value]
        return {
            key: rng.standard_normal((num_samples, observation_dim)).astype(np.float32)
            for key in proprio_keys
        }

    return factory


class TestObservationTokenizerInit:
    def test_stores_tokenizer_model(self, observation_tokenizer_factory):
        tokenizer = observation_tokenizer_factory(tokenizer_model="test-model")
        assert tokenizer.tokenizer_model == "test-model"

    def test_stores_observation_keys(self, observation_tokenizer_factory):
        keys = [ObsKey.LANGUAGE.value, ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value]
        tokenizer = observation_tokenizer_factory(observation_keys=keys)
        assert tokenizer.observation_keys == keys

    def test_stores_bin_continuous_data(self, observation_tokenizer_factory):
        tokenizer = observation_tokenizer_factory(bin_continuous_data=False)
        assert tokenizer.bin_continuous_data is False

    def test_stores_num_bins(self, observation_tokenizer_factory):
        tokenizer = observation_tokenizer_factory(num_bins=128)
        assert tokenizer.num_bins == 128

    def test_stores_max_token_len(self, observation_tokenizer_factory):
        tokenizer = observation_tokenizer_factory(max_token_len=512)
        assert tokenizer.max_token_len == 512

    def test_default_device_is_cpu(self, observation_tokenizer_factory):
        tokenizer = observation_tokenizer_factory()
        assert tokenizer.device == torch.device("cpu")

    def test_stores_explicit_device(self, observation_tokenizer_factory, device):
        tokenizer = observation_tokenizer_factory(device=device)
        assert tokenizer.device == device

    def test_vocab_size_from_language_tokenizer(self, mock_obs_auto_tokenizer):
        mock_obs_auto_tokenizer.from_pretrained.return_value = MagicMock(
            vocab_size=32000, pad_token="[PAD]"
        )
        tokenizer = ObservationTokenizer(
            tokenizer_model="test-model",
            observation_keys=[ObsKey.LANGUAGE.value],
        )
        assert tokenizer.vocab_size == 32000

    def test_is_fitted_false_on_init(self, observation_tokenizer_factory):
        tokenizer = observation_tokenizer_factory()
        assert tokenizer._is_fitted is False

    def test_sets_pad_token_from_eos_when_none(self, mock_obs_auto_tokenizer):
        mock_tok = MagicMock(vocab_size=30000, pad_token=None, eos_token="<eos>")
        mock_obs_auto_tokenizer.from_pretrained.return_value = mock_tok
        ObservationTokenizer(
            tokenizer_model="test-model",
            observation_keys=[ObsKey.LANGUAGE.value],
        )
        assert mock_tok.pad_token == "<eos>"

    def test_empty_binning_tokenizers_on_init(self, observation_tokenizer_factory):
        tokenizer = observation_tokenizer_factory()
        assert tokenizer.binning_tokenizers == {}


class TestObservationTokenizerFit:
    def test_fit_without_binning_sets_fitted(self, observation_tokenizer_factory):
        tokenizer = observation_tokenizer_factory(bin_continuous_data=False)
        tokenizer.fit({})
        assert tokenizer._is_fitted is True
        assert len(tokenizer.binning_tokenizers) == 0

    def test_fit_without_binning_logs_info(self, observation_tokenizer_factory):
        tokenizer = observation_tokenizer_factory(bin_continuous_data=False)
        with patch(
            "versatil.data.tokenization.observation_tokenizer.logging"
        ) as mock_logging:
            tokenizer.fit({})
            mock_logging.info.assert_called_once()
            assert "Binning disabled" in str(mock_logging.info.call_args)

    def test_fit_skips_language_key(self, observation_tokenizer_factory):
        tokenizer = observation_tokenizer_factory(
            observation_keys=[ObsKey.LANGUAGE.value],
            bin_continuous_data=True,
            num_bins=16,
        )
        data = {ObsKey.LANGUAGE.value: ["some text"]}
        tokenizer.fit(data)
        assert ObsKey.LANGUAGE.value not in tokenizer.binning_tokenizers

    @pytest.mark.parametrize(
        "proprio_key",
        [
            ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value,
            ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value,
            ProprioKey.GRIPPER_STATE.value,
        ],
    )
    def test_fit_creates_binning_tokenizer_per_key(
        self, observation_tokenizer_factory, training_data_factory, proprio_key
    ):
        tokenizer = observation_tokenizer_factory(
            observation_keys=[ObsKey.LANGUAGE.value, proprio_key],
            bin_continuous_data=True,
            num_bins=16,
        )
        data = training_data_factory(proprio_keys=[proprio_key])
        tokenizer.fit(data)
        assert proprio_key in tokenizer.binning_tokenizers
        assert tokenizer.binning_tokenizers[proprio_key]._is_fitted is True

    def test_fit_warns_when_key_missing_from_data(self, observation_tokenizer_factory):
        missing_key = ProprioKey.GRIPPER_STATE.value
        tokenizer = observation_tokenizer_factory(
            observation_keys=[missing_key],
            bin_continuous_data=True,
        )
        with patch(
            "versatil.data.tokenization.observation_tokenizer.logging"
        ) as mock_logging:
            tokenizer.fit({})
            mock_logging.warning.assert_called_once()
            assert missing_key in str(mock_logging.warning.call_args)

    def test_fit_with_multiple_proprio_keys(
        self, observation_tokenizer_factory, training_data_factory
    ):
        robot_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        camera_key = ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = observation_tokenizer_factory(
            observation_keys=[ObsKey.LANGUAGE.value, robot_key, camera_key],
            bin_continuous_data=True,
            num_bins=16,
        )
        data = training_data_factory(proprio_keys=[robot_key, camera_key])
        tokenizer.fit(data)
        assert robot_key in tokenizer.binning_tokenizers
        assert camera_key in tokenizer.binning_tokenizers

    def test_fit_with_binning_logs_info(
        self, observation_tokenizer_factory, training_data_factory
    ):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = observation_tokenizer_factory(
            observation_keys=[proprio_key],
            bin_continuous_data=True,
            num_bins=16,
        )
        data = training_data_factory(proprio_keys=[proprio_key])
        with patch(
            "versatil.data.tokenization.observation_tokenizer.logging"
        ) as mock_logging:
            tokenizer.fit(data)
            mock_logging.info.assert_called()
            log_message = str(mock_logging.info.call_args)
            assert "1 keys" in log_message


class TestObservationTokenizerBuildPrompts:
    @pytest.mark.parametrize(
        "language_input, expected_text",
        [
            ("Pick up the block", "pick up the block"),
            ("Pick_Up\nBlock", "pick up block"),
            ("  GRASP needle  ", "grasp needle"),
        ],
    )
    def test_language_prompt_format_and_cleaning(
        self,
        observation_tokenizer_factory,
        observation_dict_factory,
        language_input,
        expected_text,
    ):
        tokenizer = observation_tokenizer_factory(bin_continuous_data=False)
        tokenizer._is_fitted = True
        observations = observation_dict_factory(language=[language_input])
        prompts = tokenizer._build_prompts(observations)
        assert len(prompts) == 1
        assert prompts[0].startswith("Task: ")
        assert prompts[0].endswith(";\n")
        assert expected_text in prompts[0]

    def test_continuous_key_without_binning_uses_raw_floats(
        self, observation_tokenizer_factory
    ):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = observation_tokenizer_factory(
            observation_keys=[proprio_key],
            bin_continuous_data=False,
        )
        tokenizer._is_fitted = True
        data = np.array([[1.234, 5.678]], dtype=np.float32)
        observations = {proprio_key: data}
        prompts = tokenizer._build_prompts(observations)
        assert "1.234" in prompts[0]
        assert "5.678" in prompts[0]

    def test_continuous_key_with_binning_uses_bin_tokens(
        self,
        observation_tokenizer_factory,
        training_data_factory,
        observation_dict_factory,
    ):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = observation_tokenizer_factory(
            observation_keys=[proprio_key],
            bin_continuous_data=True,
            num_bins=16,
        )
        training_data = training_data_factory(
            proprio_keys=[proprio_key], num_samples=100, observation_dim=3
        )
        tokenizer.fit(training_data)
        observations = observation_dict_factory(
            proprio_keys=[proprio_key], observation_dim=3
        )
        prompts = tokenizer._build_prompts(observations)
        key_readable = proprio_key.replace("_", " ")
        assert f"{key_readable}:" in prompts[0]

    def test_key_readable_replaces_underscores(
        self, observation_tokenizer_factory, observation_dict_factory
    ):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = observation_tokenizer_factory(
            observation_keys=[proprio_key],
            bin_continuous_data=False,
        )
        tokenizer._is_fitted = True
        observations = observation_dict_factory(
            proprio_keys=[proprio_key], observation_dim=1
        )
        prompts = tokenizer._build_prompts(observations)
        assert "proprio robot frame:" in prompts[0]

    def test_multiple_keys_joined_with_comma(
        self, observation_tokenizer_factory, observation_dict_factory
    ):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = observation_tokenizer_factory(
            observation_keys=[ObsKey.LANGUAGE.value, proprio_key],
            bin_continuous_data=False,
        )
        tokenizer._is_fitted = True
        observations = observation_dict_factory(
            language=["grasp needle"],
            proprio_keys=[proprio_key],
            observation_dim=2,
        )
        prompts = tokenizer._build_prompts(observations)
        assert ", " in prompts[0]
        assert "Task: grasp needle" in prompts[0]
        assert "proprio robot frame:" in prompts[0]

    def test_batch_size_from_list_length(
        self, observation_tokenizer_factory, observation_dict_factory
    ):
        tokenizer = observation_tokenizer_factory(bin_continuous_data=False)
        tokenizer._is_fitted = True
        observations = observation_dict_factory(language=["text1", "text2", "text3"])
        prompts = tokenizer._build_prompts(observations)
        assert len(prompts) == 3

    def test_missing_key_warns_and_skips(
        self, observation_tokenizer_factory, observation_dict_factory
    ):
        tokenizer = observation_tokenizer_factory(
            observation_keys=[ObsKey.LANGUAGE.value, "nonexistent_key"],
            bin_continuous_data=False,
        )
        tokenizer._is_fitted = True
        observations = observation_dict_factory(language=["text"])
        with patch(
            "versatil.data.tokenization.observation_tokenizer.logging"
        ) as mock_logging:
            prompts = tokenizer._build_prompts(observations)
            mock_logging.warning.assert_called_once()
        assert len(prompts) == 1

    def test_torch_tensor_input(
        self, observation_tokenizer_factory, observation_dict_factory
    ):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = observation_tokenizer_factory(
            observation_keys=[proprio_key],
            bin_continuous_data=False,
        )
        tokenizer._is_fitted = True
        observations = observation_dict_factory(
            proprio_keys=[proprio_key], observation_dim=3, as_torch=True
        )
        prompts = tokenizer._build_prompts(observations)
        assert len(prompts) == 1

    def test_language_plain_string_input(
        self, observation_tokenizer_factory, observation_dict_factory
    ):
        tokenizer = observation_tokenizer_factory(bin_continuous_data=False)
        tokenizer._is_fitted = True
        observations = observation_dict_factory(language="grasp the block")
        prompts = tokenizer._build_prompts(observations)
        assert len(prompts) == 1
        assert "grasp the block" in prompts[0]

    def test_language_non_string_non_list_raises_type_error(
        self, observation_tokenizer_factory, observation_dict_factory
    ):
        tokenizer = observation_tokenizer_factory(bin_continuous_data=False)
        tokenizer._is_fitted = True
        observations = observation_dict_factory(language=12345)
        with pytest.raises(TypeError, match="Expected str for language data"):
            tokenizer._build_prompts(observations)

    def test_language_list_of_lists_joins_inner_list(
        self, observation_tokenizer_factory, observation_dict_factory
    ):
        tokenizer = observation_tokenizer_factory(bin_continuous_data=False)
        tokenizer._is_fitted = True
        observations = observation_dict_factory(
            language=[["pick", "up", "block"], ["grasp", "needle"]]
        )
        prompts = tokenizer._build_prompts(observations)
        assert len(prompts) == 2
        assert "pick up block" in prompts[0]
        assert "grasp needle" in prompts[1]

    @pytest.mark.parametrize("batch_size", [2, 3, 5])
    def test_torch_tensor_batch_size_greater_than_one(
        self, observation_tokenizer_factory, observation_dict_factory, batch_size
    ):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = observation_tokenizer_factory(
            observation_keys=[proprio_key],
            bin_continuous_data=False,
        )
        tokenizer._is_fitted = True
        observations = observation_dict_factory(
            proprio_keys=[proprio_key],
            batch_size=batch_size,
            observation_dim=2,
            as_torch=True,
        )
        prompts = tokenizer._build_prompts(observations)
        assert len(prompts) == batch_size


class TestObservationTokenizerTokenize:
    def test_tokenize_raises_when_not_fitted(self, observation_tokenizer_factory):
        tokenizer = observation_tokenizer_factory()
        with pytest.raises(RuntimeError, match="fitted before encoding"):
            tokenizer.tokenize({ObsKey.LANGUAGE.value: ["text"]})

    def test_tokenize_returns_correct_keys(
        self,
        mock_obs_auto_tokenizer,
        observation_tokenizer_factory,
        mock_language_tokenizer_result,
    ):
        mock_tok = mock_obs_auto_tokenizer.from_pretrained.return_value
        mock_tok.return_value = mock_language_tokenizer_result(
            batch_size=1, max_length=16
        )
        tokenizer = observation_tokenizer_factory(
            bin_continuous_data=False, max_token_len=16
        )
        tokenizer._is_fitted = True
        result = tokenizer.tokenize({ObsKey.LANGUAGE.value: ["text"]})
        assert SampleKey.TOKENIZED_OBSERVATIONS.value in result
        assert SampleKey.IS_PAD_OBSERVATION.value in result

    def test_tokenize_output_shape(
        self,
        mock_obs_auto_tokenizer,
        observation_tokenizer_factory,
        mock_language_tokenizer_result,
    ):
        max_token_len = 16
        batch_size = 3
        mock_tok = mock_obs_auto_tokenizer.from_pretrained.return_value
        mock_tok.return_value = mock_language_tokenizer_result(
            batch_size=batch_size, max_length=max_token_len
        )
        tokenizer = observation_tokenizer_factory(
            bin_continuous_data=False, max_token_len=max_token_len
        )
        tokenizer._is_fitted = True
        observations = {ObsKey.LANGUAGE.value: ["t1", "t2", "t3"]}
        result = tokenizer.tokenize(observations)
        assert result[SampleKey.TOKENIZED_OBSERVATIONS.value].shape == (
            batch_size,
            max_token_len,
        )

    @pytest.mark.parametrize(
        "batch_size, time_steps",
        [(2, 3), (1, 5), (4, 2)],
    )
    def test_tokenize_with_time_dimension_flattens_and_reshapes(
        self,
        mock_obs_auto_tokenizer,
        observation_tokenizer_factory,
        mock_language_tokenizer_result,
        batch_size,
        time_steps,
    ):
        max_token_len = 8
        mock_tok = mock_obs_auto_tokenizer.from_pretrained.return_value
        mock_tok.return_value = mock_language_tokenizer_result(
            batch_size=batch_size * time_steps, max_length=max_token_len
        )
        tokenizer = observation_tokenizer_factory(
            observation_keys=["test_key"],
            bin_continuous_data=False,
            max_token_len=max_token_len,
        )
        tokenizer._is_fitted = True
        observations = {
            "test_key": torch.zeros((batch_size, time_steps, 4), dtype=torch.float32)
        }
        result = tokenizer.tokenize(observations)
        tokens = result[SampleKey.TOKENIZED_OBSERVATIONS.value]
        assert tokens.shape == (batch_size, time_steps, max_token_len)


class TestObservationTokenizerTo:
    def test_to_updates_device(self, observation_tokenizer_factory, device):
        tokenizer = observation_tokenizer_factory()
        tokenizer.to(device)
        assert tokenizer.device == device

    def test_to_returns_self(self, observation_tokenizer_factory, device):
        tokenizer = observation_tokenizer_factory()
        result = tokenizer.to(device)
        assert result is tokenizer

    def test_to_moves_binning_tokenizers(
        self, observation_tokenizer_factory, training_data_factory, device
    ):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = observation_tokenizer_factory(
            observation_keys=[proprio_key],
            bin_continuous_data=True,
            num_bins=16,
        )
        data = training_data_factory(proprio_keys=[proprio_key])
        tokenizer.fit(data)
        tokenizer.to(device)
        assert tokenizer.binning_tokenizers[proprio_key].device == device


class TestObservationTokenizerStateDict:
    def test_state_dict_keys(self, observation_tokenizer_factory):
        tokenizer = observation_tokenizer_factory(
            bin_continuous_data=True, num_bins=128, max_token_len=512
        )
        tokenizer._is_fitted = True
        state = tokenizer.state_dict()
        expected_keys = {
            "tokenizer_model",
            "observation_keys",
            "bin_continuous_data",
            "num_bins",
            "max_token_len",
            "vocab_size",
            "binning_tokenizers",
            "is_fitted",
        }
        assert set(state.keys()) == expected_keys

    @pytest.mark.parametrize(
        "observation_keys, bin_continuous_data, num_bins, max_token_len",
        [
            ([ObsKey.LANGUAGE.value], True, 128, 512),
            ([ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value], False, 64, 256),
            (
                [ObsKey.LANGUAGE.value, ProprioKey.GRIPPER_STATE.value],
                True,
                32,
                128,
            ),
        ],
    )
    def test_state_dict_values(
        self,
        observation_tokenizer_factory,
        observation_keys,
        bin_continuous_data,
        num_bins,
        max_token_len,
    ):
        tokenizer = observation_tokenizer_factory(
            tokenizer_model="test-model",
            observation_keys=observation_keys,
            bin_continuous_data=bin_continuous_data,
            num_bins=num_bins,
            max_token_len=max_token_len,
        )
        tokenizer._is_fitted = True
        state = tokenizer.state_dict()
        assert state["tokenizer_model"] == "test-model"
        assert state["observation_keys"] == observation_keys
        assert state["bin_continuous_data"] is bin_continuous_data
        assert state["num_bins"] == num_bins
        assert state["max_token_len"] == max_token_len
        assert state["is_fitted"] is True

    def test_state_dict_includes_binning_tokenizer_states(
        self, observation_tokenizer_factory, training_data_factory
    ):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = observation_tokenizer_factory(
            observation_keys=[proprio_key],
            bin_continuous_data=True,
            num_bins=16,
        )
        data = training_data_factory(proprio_keys=[proprio_key])
        tokenizer.fit(data)
        state = tokenizer.state_dict()
        assert proprio_key in state["binning_tokenizers"]
        assert "num_bins" in state["binning_tokenizers"][proprio_key]


class TestObservationTokenizerLoadStateDict:
    @pytest.mark.parametrize(
        "tokenizer_model, observation_keys, bin_continuous_data, num_bins,"
        " max_token_len, vocab_size",
        [
            (
                "restored-model",
                [ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value],
                False,
                64,
                128,
                50000,
            ),
            (
                "bert-base",
                [ObsKey.LANGUAGE.value, ProprioKey.GRIPPER_STATE.value],
                True,
                256,
                512,
                30000,
            ),
            (
                "custom-tokenizer",
                [ProprioKey.CAMERA_FRAME_CARTESIAN_TIP_POS.value],
                True,
                32,
                64,
                10000,
            ),
        ],
    )
    def test_load_state_dict_restores_fields(
        self,
        observation_tokenizer_factory,
        tokenizer_model,
        observation_keys,
        bin_continuous_data,
        num_bins,
        max_token_len,
        vocab_size,
    ):
        tokenizer = observation_tokenizer_factory()
        state = {
            "tokenizer_model": tokenizer_model,
            "observation_keys": observation_keys,
            "bin_continuous_data": bin_continuous_data,
            "num_bins": num_bins,
            "max_token_len": max_token_len,
            "vocab_size": vocab_size,
            "is_fitted": True,
            "binning_tokenizers": {},
        }
        tokenizer.load_state_dict(state)
        assert tokenizer.tokenizer_model == tokenizer_model
        assert tokenizer.observation_keys == observation_keys
        assert tokenizer.bin_continuous_data is bin_continuous_data
        assert tokenizer.num_bins == num_bins
        assert tokenizer.max_token_len == max_token_len
        assert tokenizer.vocab_size == vocab_size
        assert tokenizer._is_fitted is True

    def test_load_state_dict_restores_binning_tokenizers(
        self, observation_tokenizer_factory, training_data_factory
    ):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        original = observation_tokenizer_factory(
            observation_keys=[proprio_key],
            bin_continuous_data=True,
            num_bins=16,
        )
        data = training_data_factory(proprio_keys=[proprio_key])
        original.fit(data)
        state = original.state_dict()
        restored = observation_tokenizer_factory()
        restored.load_state_dict(state)
        assert proprio_key in restored.binning_tokenizers
        assert restored.binning_tokenizers[proprio_key]._is_fitted is True


class TestObservationTokenizerSavePretrained:
    def test_save_creates_directory(self, observation_tokenizer_factory, tmp_path):
        tokenizer = observation_tokenizer_factory()
        tokenizer._is_fitted = True
        save_path = tmp_path / "obs_tokenizer"
        tokenizer.save_pretrained(save_path)
        assert save_path.exists()

    def test_save_calls_language_tokenizer_save(
        self, mock_obs_auto_tokenizer, observation_tokenizer_factory, tmp_path
    ):
        tokenizer = observation_tokenizer_factory()
        tokenizer._is_fitted = True
        save_path = tmp_path / "obs_tokenizer"
        tokenizer.save_pretrained(save_path)
        mock_tok = mock_obs_auto_tokenizer.from_pretrained.return_value
        mock_tok.save_pretrained.assert_called_once_with(
            save_path / "language_tokenizer"
        )

    def test_save_pretrained_logs_info(self, observation_tokenizer_factory, tmp_path):
        tokenizer = observation_tokenizer_factory()
        tokenizer._is_fitted = True
        save_path = tmp_path / "obs_tokenizer"
        with patch(
            "versatil.data.tokenization.observation_tokenizer.logging"
        ) as mock_logging:
            tokenizer.save_pretrained(save_path)
            mock_logging.info.assert_called_once()
            assert str(save_path) in str(mock_logging.info.call_args)


class TestObservationTokenizerFromPretrained:
    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Tokenizer path not found"):
            ObservationTokenizer.from_pretrained("/nonexistent/path")

    @patch("versatil.data.tokenization.observation_tokenizer.torch.load")
    @patch("versatil.data.tokenization.observation_tokenizer.AutoTokenizer")
    def test_loads_state_dict_and_language_tokenizer(
        self, mock_auto_tokenizer, mock_torch_load, tmp_path
    ):
        save_path = tmp_path / "obs_tokenizer"
        save_path.mkdir(parents=True)
        (save_path / "observation_tokenizer_state.pt").touch()
        (save_path / "language_tokenizer").mkdir()
        mock_torch_load.return_value = {
            "tokenizer_model": "test-model",
            "observation_keys": [ObsKey.LANGUAGE.value],
            "bin_continuous_data": False,
            "num_bins": 256,
            "max_token_len": 256,
            "vocab_size": 30000,
            "is_fitted": True,
            "binning_tokenizers": {},
        }
        mock_auto_tokenizer.from_pretrained.return_value = MagicMock(
            vocab_size=30000, pad_token="[PAD]"
        )
        loaded = ObservationTokenizer.from_pretrained(save_path)
        assert loaded.tokenizer_model == "test-model"
        assert loaded._is_fitted is True
        mock_auto_tokenizer.from_pretrained.assert_any_call(
            save_path / "language_tokenizer"
        )

    @patch("versatil.data.tokenization.observation_tokenizer.torch.load")
    @patch("versatil.data.tokenization.observation_tokenizer.AutoTokenizer")
    def test_from_pretrained_logs_info(
        self, mock_auto_tokenizer, mock_torch_load, tmp_path
    ):
        save_path = tmp_path / "obs_tokenizer"
        save_path.mkdir(parents=True)
        (save_path / "observation_tokenizer_state.pt").touch()
        (save_path / "language_tokenizer").mkdir()
        mock_torch_load.return_value = {
            "tokenizer_model": "test-model",
            "observation_keys": [ObsKey.LANGUAGE.value],
            "bin_continuous_data": False,
            "num_bins": 256,
            "max_token_len": 256,
            "vocab_size": 30000,
            "is_fitted": True,
            "binning_tokenizers": {},
        }
        mock_auto_tokenizer.from_pretrained.return_value = MagicMock(
            vocab_size=30000, pad_token="[PAD]"
        )
        with patch(
            "versatil.data.tokenization.observation_tokenizer.logging"
        ) as mock_logging:
            ObservationTokenizer.from_pretrained(save_path)
            mock_logging.info.assert_called()
            assert str(save_path) in str(mock_logging.info.call_args)


@pytest.mark.integration
class TestObservationTokenizerIntegrationLanguageOnly:
    def test_tokenize_language_observations(self, device):
        tokenizer = ObservationTokenizer(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[ObsKey.LANGUAGE.value],
            bin_continuous_data=False,
            max_token_len=32,
            device=device,
        )
        tokenizer.fit({})
        observations = {
            ObsKey.LANGUAGE.value: ["pick up the block", "grasp the needle"]
        }
        result = tokenizer.tokenize(observations)
        tokens = result[SampleKey.TOKENIZED_OBSERVATIONS.value]
        is_pad = result[SampleKey.IS_PAD_OBSERVATION.value]
        assert tokens.shape == (2, 32)
        assert is_pad.shape == (2, 32)
        assert tokens.device.type == device.type

    def test_prompt_uses_task_prefix(self, device):
        tokenizer = ObservationTokenizer(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[ObsKey.LANGUAGE.value],
            bin_continuous_data=False,
            device=device,
        )
        tokenizer.fit({})
        observations = {ObsKey.LANGUAGE.value: ["grasp needle"]}
        prompts = tokenizer._build_prompts(observations)
        assert prompts[0].startswith("Task: grasp needle")


@pytest.mark.integration
class TestObservationTokenizerIntegrationWithBinning:
    def test_fit_and_tokenize_with_proprio(
        self, training_data_factory, observation_dict_factory, device
    ):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = ObservationTokenizer(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[ObsKey.LANGUAGE.value, proprio_key],
            bin_continuous_data=True,
            num_bins=64,
            max_token_len=64,
            device=device,
        )
        training_data = training_data_factory(
            proprio_keys=[proprio_key], num_samples=100
        )
        tokenizer.fit(training_data)
        assert proprio_key in tokenizer.binning_tokenizers
        observations = observation_dict_factory(
            language=["pick up block"],
            proprio_keys=[proprio_key],
            as_torch=True,
        )
        result = tokenizer.tokenize(observations)
        assert result[SampleKey.TOKENIZED_OBSERVATIONS.value].shape == (1, 64)


@pytest.mark.integration
class TestObservationTokenizerIntegrationSaveLoad:
    def test_save_and_load_roundtrip(self, training_data_factory, device, tmp_path):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = ObservationTokenizer(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[ObsKey.LANGUAGE.value, proprio_key],
            bin_continuous_data=True,
            num_bins=32,
            max_token_len=64,
            device=device,
        )
        training_data = training_data_factory(
            proprio_keys=[proprio_key], num_samples=100
        )
        tokenizer.fit(training_data)
        save_path = tmp_path / "obs_tokenizer"
        tokenizer.save_pretrained(save_path)
        loaded = ObservationTokenizer.from_pretrained(save_path, device=device)
        assert loaded.tokenizer_model == tokenizer.tokenizer_model
        assert loaded.observation_keys == tokenizer.observation_keys
        assert loaded.num_bins == tokenizer.num_bins
        assert loaded.max_token_len == tokenizer.max_token_len
        assert loaded._is_fitted is True
        assert proprio_key in loaded.binning_tokenizers

    def test_loaded_tokenizer_produces_identical_tokens(
        self, training_data_factory, observation_dict_factory, device, tmp_path
    ):
        proprio_key = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
        tokenizer = ObservationTokenizer(
            tokenizer_model="google/bert_uncased_L-2_H-128_A-2",
            observation_keys=[ObsKey.LANGUAGE.value, proprio_key],
            bin_continuous_data=True,
            num_bins=32,
            max_token_len=64,
            device=device,
        )
        training_data = training_data_factory(
            proprio_keys=[proprio_key], num_samples=100
        )
        tokenizer.fit(training_data)

        save_path = tmp_path / "obs_tokenizer"
        tokenizer.save_pretrained(save_path)
        loaded = ObservationTokenizer.from_pretrained(save_path, device=device)

        observations = observation_dict_factory(
            language=["pick up the red block"],
            proprio_keys=[proprio_key],
            as_torch=True,
        )
        original_result = tokenizer.tokenize(observations)
        loaded_result = loaded.tokenize(observations)

        original_tokens = original_result[SampleKey.TOKENIZED_OBSERVATIONS.value]
        loaded_tokens = loaded_result[SampleKey.TOKENIZED_OBSERVATIONS.value]
        assert torch.equal(original_tokens, loaded_tokens)

        original_pad = original_result[SampleKey.IS_PAD_OBSERVATION.value]
        loaded_pad = loaded_result[SampleKey.IS_PAD_OBSERVATION.value]
        assert torch.equal(original_pad, loaded_pad)
