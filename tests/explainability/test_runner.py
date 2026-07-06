"""Tests for versatil.explainability.runner module."""

import inspect
import re
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from tso_robotics_sockets import CompressionType

import versatil.endpoints.explain as explain_endpoint
from versatil.configs.explainability import ExplanationWriterConfig
from versatil.configs.inference_client import InferenceClientConfig
from versatil.data.constants import Cameras
from versatil.explainability.constants import (
    VALID_EXPLANATION_SOURCE_TYPES,
    VALID_EXPLANATION_TYPES,
    ExplanationSourceType,
    ExplanationType,
)
from versatil.explainability.runner import ExplainabilityRunner
from versatil.explainability.sources.typedefs import ExplanationBatch
from versatil.training.constants import PrecisionType


@pytest.fixture
def runner_factory(tmp_path: Path) -> Callable[..., ExplainabilityRunner]:
    def factory(
        source: str = ExplanationSourceType.DATASET.value,
        explanation_types: list[str] | None = None,
        target_camera_keys: list[str] | None = None,
        target_vision_module_names: list[str] | None = None,
        save_overlays: bool = False,
        save_raw_heatmaps: bool = False,
        overlay_image_format: str = "png",
        data_path_override: str | list[str] | None = None,
        max_samples: int | None = 3,
    ) -> ExplainabilityRunner:
        policy = MagicMock()
        policy.eval = MagicMock()
        checkpoint_loader = MagicMock()
        checkpoint_loader.config = MagicMock()
        checkpoint_loader.config.experiment.precision = PrecisionType.FP32.value
        checkpoint_loader.policy = policy

        with patch(
            "versatil.explainability.runner.FloatCheckpointLoader",
            return_value=checkpoint_loader,
        ):
            return ExplainabilityRunner(
                checkpoint_path="/tmp/checkpoint",
                checkpoint_name="last.ckpt",
                output_directory=str(tmp_path),
                device="cpu",
                source=source,
                split="all",
                sample_stride=2,
                max_samples=max_samples,
                batch_size=2,
                explanation_types=explanation_types,
                target_camera_keys=target_camera_keys,
                target_vision_module_names=target_vision_module_names,
                writer=ExplanationWriterConfig(
                    save_overlays=save_overlays,
                    save_raw_heatmaps=save_raw_heatmaps,
                    overlay_image_format=overlay_image_format,
                ),
                data_path_override=data_path_override,
            )

    return factory


class TestExplainabilityRunner:
    def test_run_uses_dataset_source(
        self,
        runner_factory: Callable[..., ExplainabilityRunner],
    ):
        runner = runner_factory(explanation_types=[ExplanationType.GRADCAM.value])
        batch = ExplanationBatch(
            observation={Cameras.AGENTVIEW.value: torch.zeros(1, 1, 3, 4, 4)},
            actions=None,
            display_observation={Cameras.AGENTVIEW.value: torch.zeros(1, 1, 3, 4, 4)},
            metadata={"source": ExplanationSourceType.DATASET.value},
            preprocess_observation=False,
        )

        with (
            patch(
                "versatil.explainability.runner.DatasetExplanationSource",
                return_value=[batch],
            ) as mock_source_class,
            patch.object(runner, "explain_batch") as mock_explain_batch,
        ):
            runner.run()

        mock_source_class.assert_called_once_with(
            config=runner.config,
            policy=runner.policy,
            split="all",
            batch_size=2,
            sample_stride=2,
            max_samples=3,
            data_path_override=None,
        )
        mock_explain_batch.assert_called_once_with(batch=batch)

    def test_run_passes_data_path_override_to_dataset_source(
        self,
        runner_factory: Callable[..., ExplainabilityRunner],
    ):
        data_path_override = "/tmp/inference.zarr"
        runner = runner_factory(
            explanation_types=[ExplanationType.GRADCAM.value],
            data_path_override=data_path_override,
        )
        batch = ExplanationBatch(
            observation={Cameras.AGENTVIEW.value: torch.zeros(1, 1, 3, 4, 4)},
            actions=None,
            display_observation={Cameras.AGENTVIEW.value: torch.zeros(1, 1, 3, 4, 4)},
            metadata={"source": ExplanationSourceType.DATASET.value},
            preprocess_observation=False,
        )

        with (
            patch(
                "versatil.explainability.runner.DatasetExplanationSource",
                return_value=[batch],
            ) as mock_source_class,
            patch.object(runner, "explain_batch"),
        ):
            runner.run()

        assert mock_source_class.call_args.kwargs["data_path_override"] == (
            data_path_override
        )
        assert "zarr_cache_directory" not in mock_source_class.call_args.kwargs

    def test_rejects_non_positive_max_samples(
        self,
        runner_factory: Callable[..., ExplainabilityRunner],
    ) -> None:
        error_message = "max_samples must be positive when set. Got: 0"

        with pytest.raises(ValueError, match=error_message):
            runner_factory(max_samples=0)

    def test_run_online_source_drives_inference_client(
        self,
        runner_factory: Callable[..., ExplainabilityRunner],
    ):
        runner = runner_factory(source=ExplanationSourceType.ONLINE_INFERENCE.value)
        policy_runtime = MagicMock()
        observation_transport = MagicMock()
        action_transport = MagicMock()
        online_source = MagicMock()
        client = MagicMock()

        with (
            patch(
                "versatil.explainability.runner.ExplainabilityPolicyRuntime",
                return_value=policy_runtime,
            ) as mock_runtime_class,
            patch(
                "versatil.explainability.runner.SocketObservationTransport",
                return_value=observation_transport,
            ) as mock_observation_transport_class,
            patch(
                "versatil.explainability.runner.SocketActionTransport",
                return_value=action_transport,
            ) as mock_action_transport_class,
            patch(
                "versatil.explainability.runner.InferenceClient",
                return_value=client,
            ) as mock_client_class,
            patch.object(
                runner,
                "build_online_source",
                return_value=online_source,
            ) as mock_build_online_source,
        ):
            runner.run()

        mock_runtime_class.assert_called_once_with(
            checkpoint_loader=runner.checkpoint_loader,
            checkpoint_name=runner.checkpoint_name,
        )
        mock_observation_transport_class.assert_called_once_with(
            server_address="127.0.0.1",
            server_port=5555,
            request_timeout_seconds=None,
        )
        mock_action_transport_class.assert_called_once_with(
            server_address="127.0.0.1",
            server_port=5555,
            request_timeout_seconds=None,
        )
        mock_build_online_source.assert_called_once_with()
        mock_client_class.assert_called_once_with(
            policy_runtime=policy_runtime,
            observation_transport=observation_transport,
            action_transport=action_transport,
            temporal_aggregation=False,
            action_execution_horizon=None,
            compression_type="raw",
            max_timesteps=800,
            timing_log=False,
            update_rate_hz=None,
            online_explanation_source=online_source,
        )
        client.run_episode.assert_called_once_with(max_steps=5)
        client.shutdown.assert_called_once_with()

    def test_run_online_source_uses_default_step_guard_without_sample_cap(
        self,
        runner_factory: Callable[..., ExplainabilityRunner],
    ) -> None:
        runner = runner_factory(
            source=ExplanationSourceType.ONLINE_INFERENCE.value,
            max_samples=None,
        )
        client = MagicMock()

        with (
            patch("versatil.explainability.runner.ExplainabilityPolicyRuntime"),
            patch("versatil.explainability.runner.SocketObservationTransport"),
            patch("versatil.explainability.runner.SocketActionTransport"),
            patch(
                "versatil.explainability.runner.InferenceClient",
                return_value=client,
            ),
            patch.object(runner, "build_online_source", return_value=MagicMock()),
        ):
            runner.run()

        client.run_episode.assert_called_once_with(max_steps=1000000)
        client.shutdown.assert_called_once_with()

    def test_build_online_source_uses_sample_stride(
        self,
        runner_factory: Callable[..., ExplainabilityRunner],
    ):
        runner = runner_factory(explanation_types=[ExplanationType.GRADCAM.value])

        with patch(
            "versatil.explainability.runner.OnlineInferenceExplanationSource",
            return_value=MagicMock(),
        ) as mock_online_source_class:
            runner.build_online_source()

        mock_online_source_class.assert_called_once_with(
            consumer=runner,
            sample_stride=2,
            max_samples=3,
        )

    def test_compute_heatmaps_filters_target_camera_and_visual_module_names(
        self,
        runner_factory: Callable[..., ExplainabilityRunner],
    ):
        runner = runner_factory(
            explanation_types=[ExplanationType.GRADCAM.value],
            target_camera_keys=[Cameras.EYE_IN_HAND.value],
            target_vision_module_names=["decoder.vlm_backbone.vision_encoders.0"],
        )
        expected_heatmap = torch.zeros(1, 1, 4, 4)
        observation = {Cameras.EYE_IN_HAND.value: torch.zeros(1, 1, 3, 4, 4)}
        actions = {"tokens": torch.ones(1, 2, dtype=torch.long)}
        heatmap_function = MagicMock(
            return_value={Cameras.EYE_IN_HAND.value: expected_heatmap}
        )
        runner.explanation_heatmaps = {
            ExplanationType.GRADCAM.value: heatmap_function,
        }

        result = runner._compute_heatmaps(
            observation=observation,
            actions=actions,
            explanation_type=ExplanationType.GRADCAM.value,
            preprocess_observation=False,
        )

        assert result == {Cameras.EYE_IN_HAND.value: expected_heatmap}
        heatmap_function.assert_called_once_with(
            policy=runner.policy,
            observation=observation,
            actions=actions,
            target_camera=Cameras.EYE_IN_HAND.value,
            target_vision_module_names=["decoder.vlm_backbone.vision_encoders.0"],
            preprocess_observation=False,
        )

    def test_get_target_cameras_rejects_empty_camera_allowlist(
        self,
        runner_factory: Callable[..., ExplainabilityRunner],
    ):
        runner = runner_factory(target_camera_keys=[])
        expected_message = "target_camera_keys must not be empty when set."

        with pytest.raises(ValueError, match=expected_message):
            runner._get_target_cameras()

    def test_save_overlays_uses_heatmap_camera_key(
        self,
        runner_factory: Callable[..., ExplainabilityRunner],
    ):
        runner = runner_factory(save_overlays=True, overlay_image_format=".jpg")
        batch = ExplanationBatch(
            observation={},
            actions=None,
            display_observation={
                Cameras.EYE_IN_HAND.value: torch.full((1, 1, 3, 4, 4), 0.5)
            },
            metadata={
                "source": ExplanationSourceType.DATASET.value,
                "split": "all",
                "sample_indices": [7],
            },
            preprocess_observation=False,
        )

        with patch("versatil.explainability.writer.cv2.imwrite") as mock_write:
            runner.writer.save_overlays(
                heatmaps={Cameras.EYE_IN_HAND.value: torch.ones(1, 1, 4, 4)},
                explanation_type=ExplanationType.GRADCAM.value,
                batch=batch,
                batch_counter=runner._batch_counter,
            )

        output_path = mock_write.call_args.args[0]
        assert "eye_in_hand" in output_path
        assert "sample_7" in output_path
        assert output_path.endswith(".jpg")

    def test_overlay_image_format_rejects_path_like_values(
        self,
        runner_factory: Callable[..., ExplainabilityRunner],
    ):
        with pytest.raises(ValueError, match="must be a file extension"):
            runner_factory(overlay_image_format="nested/png")


def test_endpoint_is_hydra_facing_and_not_schema_specific():
    endpoint_source = inspect.getsource(explain_endpoint)

    assert "@hydra.main" in endpoint_source
    assert "end_to_end_explain/default.yaml" in endpoint_source
    assert "end_to_end_explainability" not in endpoint_source
    assert "pd.read_csv" not in endpoint_source
    assert "get_image_path_column" not in endpoint_source
    assert "Cameras.LEFT" not in endpoint_source


class TestRunnerValidation:
    def _make_runner(self, tmp_path: Path, **overrides):
        checkpoint_loader = MagicMock()
        checkpoint_loader.config = MagicMock()
        checkpoint_loader.policy = MagicMock()
        arguments = {
            "checkpoint_path": str(tmp_path / "checkpoint"),
            "checkpoint_name": "last.ckpt",
            "output_directory": str(tmp_path / "out"),
            "device": "cpu",
        }
        arguments.update(overrides)
        with patch(
            "versatil.explainability.runner.FloatCheckpointLoader",
            return_value=checkpoint_loader,
        ):
            return ExplainabilityRunner(**arguments)

    def test_auto_device_and_default_output_directory(self, tmp_path: Path):
        runner = self._make_runner(tmp_path, device="auto", output_directory=None)
        assert runner.device.type in ("cpu", "cuda")
        assert "explainability" in str(runner.output_directory)

    def test_invalid_source_raises(self, tmp_path: Path):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"source must be one of {list(VALID_EXPLANATION_SOURCE_TYPES)}. "
                "Got: not_a_source"
            ),
        ):
            self._make_runner(tmp_path, source="not_a_source")

    @pytest.mark.parametrize(
        "overrides, error_message",
        [
            ({"sample_stride": 0}, "sample_stride must be positive. Got: 0"),
            ({"max_samples": 0}, "max_samples must be positive when set. Got: 0"),
        ],
    )
    def test_invalid_sampling_raises(self, tmp_path: Path, overrides, error_message):
        with pytest.raises(ValueError, match=re.escape(error_message)):
            self._make_runner(tmp_path, **overrides)

    @pytest.mark.parametrize(
        "online_overrides, error_message",
        [
            ({"model_server_port": 0}, "model_server_port must be positive. Got: 0"),
            (
                {"action_execution_horizon": 0},
                "action_execution_horizon must be positive when set. Got: 0",
            ),
            (
                {"update_rate_hz": 0.0},
                "update_rate_hz must be positive when set. Got: 0.0",
            ),
            (
                {"temporal_max_timesteps": 0},
                "temporal_max_timesteps must be positive. Got: 0",
            ),
            (
                {"compression_type": "bogus"},
                f"compression_type must be one of "
                f"{[member.value for member in CompressionType]}. Got: bogus",
            ),
        ],
    )
    def test_invalid_online_configuration_raises(
        self, tmp_path: Path, online_overrides, error_message
    ):
        with pytest.raises(ValueError, match=re.escape(error_message)):
            self._make_runner(
                tmp_path,
                source=ExplanationSourceType.ONLINE_INFERENCE.value,
                online=InferenceClientConfig(**online_overrides),
            )

    def test_invalid_explanation_type_raises(self, tmp_path: Path):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Unsupported explanation_types ['not_a_method']. "
                f"Use one or more of: {list(VALID_EXPLANATION_TYPES)}"
            ),
        ):
            self._make_runner(tmp_path, explanation_types=["not_a_method"])
