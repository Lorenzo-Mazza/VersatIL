"""Tests for versatil.checkpoint_loading.compressed_policy module."""

import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torchao.quantization import Int8WeightOnlyConfig, quantize_

from versatil.checkpoint_loading.compressed_policy import CompressedCheckpointLoader
from versatil.checkpoint_loading.metadata import CheckpointMetadata
from versatil.data.constants import Cameras, DatasetType, ProprioKey, SampleKey
from versatil.data.normalization.normalizer import LinearNormalizer
from versatil.data.tokenization.action_discretizer import (
    ActionDiscretizer,
    BinnedActionDiscretizer,
    FastActionDiscretizer,
)
from versatil.data.tokenization.action_tokenizer import ActionTokenizer
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.inference.inference_client import infer_rotate_images
from versatil.inference.policy_runtime.compressed_runtime import CompressedPolicyRuntime
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.exportable.metadata import (
    PolicyExportMetadata,
    PredictionOutput,
)
from versatil.post_training_compression.constants import QuantizationWorkflow
from versatil.post_training_compression.serialization import save_compressed_model
from versatil.training.constants import CheckpointFilename

COMPRESSED_CHECKPOINT_MODULE = "versatil.checkpoint_loading.compressed_policy"
OBSERVATION_KEY = ProprioKey.ROBOT_FRAME_CARTESIAN_TIP_POS.value
ACTION_KEY = ProprioKey.EE_POS_ACTION.value


class _TokenArtifact(nn.Module):
    """Repeat a stored token sequence for each observation window.

    Note:
        B denotes batch size and L denotes token sequence length.
    """

    def __init__(self, tokens: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("tokens", tokens)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.tokens.expand(observations.shape[0], -1)  # (1, L) -> (B, L)


@pytest.fixture
def saved_config_factory() -> Callable[..., DictConfig]:
    def factory(tokenize_observations: bool = False) -> DictConfig:
        return OmegaConf.create(
            {
                "task": {
                    "dataset_schema": {
                        "_target_": "versatil.data.raw.schemas.lerobot.LeRobotDatasetSchemaV30",
                        "dataset_type": DatasetType.LIBERO.value,
                        "dataset_path": "/unavailable/training_dataset",
                    },
                    "dataloader": {
                        "tokenization": {
                            "tokenize_observations": tokenize_observations,
                        }
                    },
                    "observation_space": {
                        "_target_": "versatil.data.task.ObservationSpace",
                        "observations_metadata": {
                            OBSERVATION_KEY: {
                                "_target_": "versatil.data.metadata.PositionObservationMetadata",
                                "raw_data_column_keys": ["x", "y", "z"],
                                "dimension": 3,
                                "dtype": "float32",
                                "needs_normalization": True,
                                "frame": "robot_base",
                            }
                        },
                    },
                    "action_space": {
                        "_target_": "versatil.data.task.ActionSpace",
                        "actions_metadata": {
                            ACTION_KEY: {
                                "_target_": "versatil.data.metadata.PositionActionMetadata",
                                "frame": "robot_base",
                                "raw_data_column_keys": ["x", "y", "z"],
                                "storage_dimension": 3,
                                "prediction_dimension": 3,
                                "needs_normalization": True,
                                "dtype": "float32",
                            }
                        },
                    },
                    "prediction_horizon": 2,
                    "observation_horizon": 1,
                },
                "policy": {
                    "_target_": "versatil.models.policy.Policy",
                    "observation_space": "${task.observation_space}",
                    "action_space": "${task.action_space}",
                    "prediction_horizon": "${task.prediction_horizon}",
                    "observation_horizon": "${task.observation_horizon}",
                    "decoder": {
                        "_target_": "versatil.models.decoding.decoders.factory.autoregressive_vla.AutoregressiveVLADecoder",
                        "observation_horizon": 2,
                        "vlm_backbone": {
                            "_target_": "versatil.models.decoding.generative_language_models.vision_language.paligemma.PaliGemmaVLM",
                            "model_name": "/unavailable/original_model_assets",
                        },
                    },
                },
            }
        )

    return factory


@pytest.fixture
def portable_artifact_factory(
    tmp_path: Path,
    rng: np.random.Generator,
    saved_config_factory: Callable[..., DictConfig],
) -> Callable[..., tuple[Path, dict[str, torch.Tensor], torch.Tensor]]:
    """Build an artifact with local config, normalization and optional token assets.

    Note:
        B denotes batch size, H action horizon, D action dimension and L token
        sequence length.
    """

    def factory(
        output: PredictionOutput,
        quantized: bool,
    ) -> tuple[Path, dict[str, torch.Tensor], torch.Tensor]:
        observations = torch.from_numpy(
            rng.uniform(0.0, 2.0, size=(3, 2, 3)).astype(np.float32)
        )  # (B, H, D)
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                OBSERVATION_KEY: torch.tensor(
                    [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]
                ),  # (2, D)
                ACTION_KEY: torch.tensor([[2.0, 4.0, 6.0], [4.0, 8.0, 12.0]]),  # (2, D)
            }
        )
        normalized = normalizer[OBSERVATION_KEY].normalize(observations)  # (B, H, D)
        tokenizer = None
        if output == PredictionOutput.ACTION_TOKENS:
            action_tokenizer = ActionTokenizer(
                action_discretizer=BinnedActionDiscretizer(num_bins=16),
                max_token_len=7,
                pad_token_id=16,
                device=torch.device("cpu"),
            )
            action_tokenizer.fit(
                action_chunks=rng.uniform(-1.0, 1.0, size=(8, 2, 3)).astype(np.float32)
            )
            tokens = action_tokenizer.encode(action_chunks=normalized[:1])[
                SampleKey.TOKENIZED_ACTIONS.value
            ]  # (1, H, D) -> (1, L)
            model = _TokenArtifact(tokens=tokens)
            decoded = torch.from_numpy(
                action_tokenizer.decode(tokens[0])  # (L,) -> (H, D)
            ).float()  # (H, D)
            normalized_actions = decoded.unsqueeze(0).expand(
                3, -1, -1
            )  # (H, D) -> (B, H, D)
            tokenizer = Tokenizer(action_tokenizer=action_tokenizer)
            output_keys = [DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
        else:
            model = nn.Sequential(nn.Linear(in_features=3, out_features=3, bias=False))
            with torch.no_grad():
                model[0].weight.copy_(
                    torch.from_numpy(
                        rng.uniform(-0.5, 0.5, size=(3, 3)).astype(np.float32)
                    )
                )  # (D, D)
            if quantized:
                quantize_(model=model, config=Int8WeightOnlyConfig())
                assert type(model[0].weight).__name__ == "Int8Tensor"
            with torch.no_grad():
                normalized_actions = model(normalized)  # (B, H, D)
            output_keys = [ACTION_KEY]
        model.eval()
        expected_actions = normalizer[ACTION_KEY].unnormalize(
            normalized_actions
        )  # (B, H, D)
        directory = tmp_path / "portable_artifact"
        save_compressed_model(
            converted_model=model,
            example_inputs=(normalized[:2],),  # (B, H, D) -> (2, H, D)
            save_directory=str(directory),
            input_keys=[OBSERVATION_KEY],
            output_keys=output_keys,
            normalizer=normalizer,
            training_checkpoint_path=str(tmp_path / "unavailable_original_checkpoint"),
            quantization_config={},
            quantization_workflow=(
                QuantizationWorkflow.EAGER.value
                if quantized
                else QuantizationWorkflow.NONE.value
            ),
            export_metadata=PolicyExportMetadata(output=output),
            tokenizer=tokenizer,
            denoising_thresholds={ACTION_KEY: 0.05, "unpredicted": 1.0},
        )
        OmegaConf.save(
            config=saved_config_factory(tokenize_observations=False),
            f=directory / CheckpointFilename.CONFIG.value,
        )
        return directory, {OBSERVATION_KEY: observations}, expected_actions

    return factory


@pytest.fixture
def export_metadata_loader_factory(
    checkpoint_metadata_factory: Callable[..., CheckpointMetadata],
) -> Callable[..., CompressedCheckpointLoader]:
    """Build a loader with independently controlled artifact and tokenizer metadata."""

    def factory(
        output: PredictionOutput,
        output_keys: list[str],
        has_tokenizer: bool = True,
        has_action_tokenizer: bool = True,
        binned: bool = True,
        supported_discretizer: bool = True,
        fitted: bool = True,
        tokenizer_shape: tuple[int | None, int | None] = (4, 3),
    ) -> CompressedCheckpointLoader:
        loader = CompressedCheckpointLoader.__new__(CompressedCheckpointLoader)
        loader._checkpoint_path = "/tmp/compressed"
        loader._export_metadata = MagicMock(spec=PolicyExportMetadata)
        loader._export_metadata.output = output
        loader._output_keys = output_keys
        loader._checkpoint_metadata = checkpoint_metadata_factory(
            prediction_horizon=4, observation_horizon=2
        )
        loader._tokenizer = MagicMock() if has_tokenizer else None
        if loader._tokenizer is not None:
            if has_action_tokenizer:
                discretizer = MagicMock(
                    spec=(BinnedActionDiscretizer if binned else FastActionDiscretizer)
                    if supported_discretizer
                    else ActionDiscretizer
                )
                discretizer.time_horizon, discretizer.action_dim = tokenizer_shape
                discretizer.is_fitted = fitted
                loader._tokenizer.action_tokenizer.action_discretizer = discretizer
            else:
                loader._tokenizer.action_tokenizer = None
        return loader

    return factory


@pytest.mark.unit
class TestCompressedCheckpointLoaderExportMetadata:
    @pytest.mark.parametrize("output", list(PredictionOutput))
    @pytest.mark.parametrize("has_local_tokenizer", [False, True])
    def test_resolves_tokenizer_assets_for_the_prediction_format(
        self,
        export_metadata_loader_factory: Callable[..., CompressedCheckpointLoader],
        output: PredictionOutput,
        has_local_tokenizer: bool,
    ) -> None:
        loader = export_metadata_loader_factory(output=output, output_keys=[])
        tokenizer_directory = CheckpointFilename.TOKENIZER_DIR.value
        local_path = f"/tmp/compressed/{tokenizer_directory}"
        remote_path = f"/tmp/training/{tokenizer_directory}"
        expected_path = local_path if has_local_tokenizer else remote_path
        expectation = (
            pytest.raises(
                FileNotFoundError,
                match=re.escape(
                    "Action-token artifacts require local tokenizer assets at "
                    f"{local_path}."
                ),
            )
            if output == PredictionOutput.ACTION_TOKENS and not has_local_tokenizer
            else does_not_raise()
        )

        with (
            patch(
                f"{COMPRESSED_CHECKPOINT_MODULE}.os.path.isdir",
                side_effect=lambda path: path == remote_path or has_local_tokenizer,
            ) as directory_exists,
            patch(
                f"{COMPRESSED_CHECKPOINT_MODULE}.os.path.exists",
                side_effect=lambda path: path == remote_path or has_local_tokenizer,
            ) as path_exists,
            expectation,
        ):
            actual_path = loader._resolve_tokenizer_path(
                training_checkpoint_path="/tmp/training"
            )
            assert actual_path == expected_path

        if output == PredictionOutput.ACTION_TOKENS:
            directory_exists.assert_called_once_with(local_path)
            path_exists.assert_not_called()
        else:
            path_exists.assert_called_once_with(local_path)
            directory_exists.assert_not_called()

    @pytest.mark.parametrize("output", list(PredictionOutput))
    @pytest.mark.parametrize("binned", [False, True])
    def test_accepts_matching_output_and_tokenizer_requirements(
        self,
        export_metadata_loader_factory: Callable[..., CompressedCheckpointLoader],
        output: PredictionOutput,
        binned: bool,
    ) -> None:
        output_keys = (
            [DecoderOutputKey.PREDICTED_ACTION_TOKENS.value]
            if output == PredictionOutput.ACTION_TOKENS
            else ["position"]
        )
        loader = export_metadata_loader_factory(
            output=output,
            output_keys=output_keys,
            has_tokenizer=output == PredictionOutput.ACTION_TOKENS,
            tokenizer_shape=(4, 3),
            binned=binned,
        )

        loader._validate_export_metadata()

        if output == PredictionOutput.ACTION_TOKENS:
            loader.action_space.get_total_action_dim.assert_called_once_with()
        else:
            loader.action_space.get_total_action_dim.assert_not_called()

    @pytest.mark.parametrize("output_keys", [[], ["position", "orientation"]])
    def test_rejects_incompatible_token_output_keys(
        self,
        export_metadata_loader_factory: Callable[..., CompressedCheckpointLoader],
        output_keys: list[str],
    ) -> None:
        loader = export_metadata_loader_factory(
            output=PredictionOutput.ACTION_TOKENS,
            output_keys=output_keys,
        )
        token_key = DecoderOutputKey.PREDICTED_ACTION_TOKENS.value

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Action-token artifacts require output_keys=[{token_key!r}], "
                f"got {output_keys!r}."
            ),
        ):
            loader._validate_export_metadata()

    @pytest.mark.parametrize(
        "has_tokenizer,has_action_tokenizer", [(False, False), (True, False)]
    )
    def test_requires_saved_action_tokenizer(
        self,
        export_metadata_loader_factory: Callable[..., CompressedCheckpointLoader],
        has_tokenizer: bool,
        has_action_tokenizer: bool,
    ) -> None:
        loader = export_metadata_loader_factory(
            output=PredictionOutput.ACTION_TOKENS,
            output_keys=[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
            has_tokenizer=has_tokenizer,
            has_action_tokenizer=has_action_tokenizer,
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Action-token artifacts require a saved action tokenizer in the "
                "compressed checkpoint's tokenizer directory."
            ),
        ):
            loader._validate_export_metadata()

    def test_rejects_incompatible_discretizer_assets(
        self,
        export_metadata_loader_factory: Callable[..., CompressedCheckpointLoader],
    ) -> None:
        loader = export_metadata_loader_factory(
            output=PredictionOutput.ACTION_TOKENS,
            output_keys=[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
            supported_discretizer=False,
        )
        discretizer_name = type(
            loader.tokenizer.action_tokenizer.action_discretizer
        ).__name__

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Action-token artifacts require a binned or FAST action discretizer; "
                f"loaded {discretizer_name}."
            ),
        ):
            loader._validate_export_metadata()

    @pytest.mark.parametrize("binned", [False, True])
    def test_requires_fitted_discretizer_assets(
        self,
        export_metadata_loader_factory: Callable[..., CompressedCheckpointLoader],
        binned: bool,
    ) -> None:
        loader = export_metadata_loader_factory(
            output=PredictionOutput.ACTION_TOKENS,
            output_keys=[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
            binned=binned,
            fitted=False,
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Action-token artifacts require a fitted action discretizer."
            ),
        ):
            loader._validate_export_metadata()

    @pytest.mark.parametrize("tokenizer_shape", [(4, 3), (2, 3), (None, None)])
    @pytest.mark.parametrize("binned", [False, True])
    def test_checks_tokenizer_shape_against_policy_actions(
        self,
        export_metadata_loader_factory: Callable[..., CompressedCheckpointLoader],
        tokenizer_shape: tuple[int | None, int | None],
        binned: bool,
    ) -> None:
        loader = export_metadata_loader_factory(
            output=PredictionOutput.ACTION_TOKENS,
            output_keys=[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
            tokenizer_shape=tokenizer_shape,
            binned=binned,
        )
        expectation = (
            does_not_raise()
            if tokenizer_shape == (4, 3)
            else pytest.raises(
                ValueError,
                match=re.escape(
                    f"Saved action tokenizer decodes shape {tokenizer_shape}, "
                    "but the policy requires (4, 3)."
                ),
            )
        )

        with expectation:
            loader._validate_export_metadata()


@pytest.mark.unit
class TestCompressedCheckpointLoaderTrainingConfig:
    @pytest.mark.parametrize("local_exists", [False, True])
    def test_constructs_only_spaces_from_the_selected_config(
        self,
        checkpoint_config_factory: Callable[..., MagicMock],
        checkpoint_metadata_factory: Callable[..., CheckpointMetadata],
        local_exists: bool,
    ) -> None:
        loader = CompressedCheckpointLoader.__new__(CompressedCheckpointLoader)
        loader._device = torch.device("cpu")
        loader._checkpoint_path = "/tmp/compressed"
        config = checkpoint_config_factory()
        metadata = checkpoint_metadata_factory(
            prediction_horizon=4, observation_horizon=2
        )
        config_path = (
            "/tmp/compressed/config.yaml" if local_exists else "/tmp/train/config.yaml"
        )

        with (
            patch(
                f"{COMPRESSED_CHECKPOINT_MODULE}.os.path.exists",
                side_effect=lambda path: path == config_path,
            ),
            patch(
                f"{COMPRESSED_CHECKPOINT_MODULE}.OmegaConf.load", return_value=config
            ) as load_config,
            patch(
                f"{COMPRESSED_CHECKPOINT_MODULE}.hydra.utils.instantiate",
                side_effect=[metadata.observation_space, metadata.action_space],
            ) as instantiate,
        ):
            loader._load_training_config(training_checkpoint_path="/tmp/train")

        load_config.assert_called_once_with(config_path)
        assert [call.args for call in instantiate.call_args_list] == [
            (config.policy.observation_space,),
            (config.policy.action_space,),
        ]
        assert loader.observation_space == metadata.observation_space
        assert loader.action_space == metadata.action_space
        assert loader.prediction_horizon == 4
        assert loader.observation_horizon == 2
        config.policy.to.assert_not_called()
        assert not hasattr(loader, "policy")

    def test_requires_a_saved_config(self) -> None:
        loader = CompressedCheckpointLoader.__new__(CompressedCheckpointLoader)
        loader._checkpoint_path = "/tmp/compressed"
        with (
            patch(f"{COMPRESSED_CHECKPOINT_MODULE}.os.path.exists", return_value=False),
            patch(f"{COMPRESSED_CHECKPOINT_MODULE}.OmegaConf.load") as load_config,
            pytest.raises(
                FileNotFoundError,
                match=re.escape("Config file not found at /tmp/train/config.yaml."),
            ),
        ):
            loader._load_training_config(training_checkpoint_path="/tmp/train")
        load_config.assert_not_called()


@pytest.mark.unit
class TestCompressedCheckpointLoaderMetadataProperties:
    @pytest.mark.parametrize(
        "property_name, attribute_name",
        [
            ("input_keys", "_input_keys"),
            ("output_keys", "_output_keys"),
        ],
    )
    def test_key_properties_return_copies(
        self,
        property_name: str,
        attribute_name: str,
    ) -> None:
        loader = CompressedCheckpointLoader.__new__(CompressedCheckpointLoader)
        setattr(loader, attribute_name, ["left", "depth"])

        returned = getattr(loader, property_name)
        returned.append("mutated")

        assert getattr(loader, attribute_name) == ["left", "depth"]

    def test_depth_clamp_range_uses_compressed_normalizer(
        self, checkpoint_metadata_factory: Callable[..., CheckpointMetadata]
    ) -> None:
        normalizer = MagicMock(spec=LinearNormalizer)
        normalizer.params_dict = {Cameras.DEPTH.value: MagicMock()}
        stats = {"min": MagicMock(), "max": MagicMock()}
        stats["min"].item.return_value = 0.1
        stats["max"].item.return_value = 0.9
        normalizer.__getitem__.return_value.params_dict.get.return_value = stats
        loader = CompressedCheckpointLoader.__new__(CompressedCheckpointLoader)
        loader._normalizer = normalizer
        loader._checkpoint_metadata = checkpoint_metadata_factory()
        loader.observation_space.depth_cameras = {Cameras.DEPTH.value: MagicMock()}

        result = loader.depth_clamp_ranges

        assert len(result) == 1
        minimum, maximum = result[Cameras.DEPTH.value]
        assert minimum == pytest.approx(0.1, abs=1e-5)
        assert maximum == pytest.approx(0.9, abs=1e-5)
        normalizer.__getitem__.assert_called_once_with(Cameras.DEPTH.value)


@pytest.mark.integration
@pytest.mark.parametrize(
    "output,quantized",
    [
        (PredictionOutput.ACTIONS, False),
        (PredictionOutput.ACTIONS, True),
        (PredictionOutput.ACTION_TOKENS, False),
    ],
    ids=["continuous_float", "continuous_int8", "binned_tokens"],
)
def test_saved_artifact_reconstructs_actions_with_original_assets_unavailable(
    portable_artifact_factory: Callable[
        ..., tuple[Path, dict[str, torch.Tensor], torch.Tensor]
    ],
    output: PredictionOutput,
    quantized: bool,
) -> None:
    directory, observations, expected = portable_artifact_factory(
        output=output, quantized=quantized
    )
    with (
        patch(
            "versatil.models.policy.Policy.__init__",
            side_effect=AssertionError("Original policy construction was requested."),
        ) as construct_policy,
        patch(
            "versatil.data.raw.schemas.lerobot.LeRobotDatasetSchemaV30.__init__",
            side_effect=AssertionError("Training dataset construction was requested."),
        ) as construct_dataset,
        patch(
            "versatil.models.decoding.generative_language_models.vision_language.paligemma.PaliGemmaVLM.__init__",
            side_effect=AssertionError("Original model assets were requested."),
        ) as construct_backbone,
    ):
        runtime = CompressedPolicyRuntime(
            device=torch.device("cpu"),
            checkpoint_path=str(directory),
            compile_model=False,
        )
        actual = runtime.run_inference(obs_dict=observations)[ACTION_KEY]  # (B, H, D)

    construct_policy.assert_not_called()
    construct_dataset.assert_not_called()
    construct_backbone.assert_not_called()
    torch.testing.assert_close(actual, expected)
    assert runtime.observation_horizon == 2
    assert runtime.config.policy.observation_horizon == 1
    assert runtime.prediction_horizon == 2
    assert runtime.action_space.get_total_action_dim() == 3
    assert list(runtime.observation_space.observations_metadata) == [OBSERVATION_KEY]
    assert runtime.denoising_thresholds == {ACTION_KEY: 0.05}
    assert infer_rotate_images(config=runtime.config) is True
    assert not (directory.parent / "unavailable_original_checkpoint").exists()
    assert not hasattr(runtime.checkpoint_loader, "policy")


@pytest.mark.integration
def test_compressed_observation_tokenization_requires_saved_assets(
    portable_artifact_factory: Callable[
        ..., tuple[Path, dict[str, torch.Tensor], torch.Tensor]
    ],
    saved_config_factory: Callable[..., DictConfig],
) -> None:
    directory, _, _ = portable_artifact_factory(
        output=PredictionOutput.ACTIONS, quantized=False
    )
    OmegaConf.save(
        config=saved_config_factory(tokenize_observations=True),
        f=directory / CheckpointFilename.CONFIG.value,
    )
    tokenizer_path = (
        directory.parent
        / "unavailable_original_checkpoint"
        / CheckpointFilename.TOKENIZER_DIR.value
    )

    with pytest.raises(
        FileNotFoundError,
        match=re.escape(
            "Config requires observation tokenization but no tokenizer "
            f"found at {tokenizer_path}. Save the tokenizer alongside "
            "the checkpoint."
        ),
    ):
        CompressedCheckpointLoader(
            device=torch.device("cpu"), checkpoint_path=str(directory)
        )
