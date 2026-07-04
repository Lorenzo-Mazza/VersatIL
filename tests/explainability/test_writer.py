"""Tests for versatil.explainability.writer module."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from versatil.data.constants import Cameras
from versatil.explainability.constants import ExplanationSourceType, ExplanationType
from versatil.explainability.sources.typedefs import ExplanationBatch
from versatil.explainability.writer import ExplanationWriter


class TestExplanationWriterFilenames:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (ExplanationType.GRADCAM_PLUS_PLUS.value, "gradcam_plus_plus"),
            ("agentview rgb", "agentview_rgb"),
            ("///", ""),
        ],
    )
    def test_sanitize_uses_stable_readable_filename_tokens(
        self,
        value: str,
        expected: str,
    ) -> None:
        result = ExplanationWriter.sanitize(value=value)

        assert result == expected

    def test_save_overlays_writes_gradcam_plus_plus_png_filename(
        self,
        tmp_path: Path,
    ) -> None:
        writer = ExplanationWriter(
            output_directory=tmp_path,
            image_weight=0.5,
            overlay_image_format="png",
        )
        batch = ExplanationBatch(
            observation={},
            actions=None,
            display_observation={
                Cameras.AGENTVIEW.value: torch.full((1, 1, 3, 4, 4), 0.5),
            },
            metadata={
                "source": ExplanationSourceType.DATASET.value,
                "split": "all",
                "sample_indices": [7],
            },
            preprocess_observation=False,
        )

        with (
            patch(
                "versatil.explainability.writer.show_cam_on_image",
                return_value=np.zeros((4, 4, 3), dtype=np.uint8),
            ) as mock_show_cam,
            patch("versatil.explainability.writer.cv2.imwrite") as mock_write,
        ):
            writer.save_overlays(
                heatmaps={Cameras.AGENTVIEW.value: torch.ones(1, 1, 4, 4)},
                explanation_type=ExplanationType.GRADCAM_PLUS_PLUS.value,
                batch=batch,
                batch_counter=0,
            )

        output_path = Path(mock_write.call_args.args[0])
        assert output_path.name == ("sample_7_t0_gradcam_plus_plus_agentview_rgb.png")
        assert "gradcam__" not in output_path.name
        mock_show_cam.assert_called_once()

    def test_save_raw_heatmaps_writes_gradcam_plus_plus_tensor_filename(
        self,
        tmp_path: Path,
    ) -> None:
        writer = ExplanationWriter(
            output_directory=tmp_path,
            image_weight=0.5,
            overlay_image_format="png",
        )

        with patch("versatil.explainability.writer.torch.save") as mock_save:
            writer.save_raw_heatmaps(
                heatmaps={Cameras.AGENTVIEW.value: torch.ones(1, 1, 4, 4)},
                explanation_type=ExplanationType.GRADCAM_PLUS_PLUS.value,
                metadata={
                    "source": ExplanationSourceType.DATASET.value,
                    "split": "all",
                },
                batch_counter=4,
            )

        output_path = mock_save.call_args.args[1]
        assert output_path == (
            tmp_path
            / ExplanationSourceType.DATASET.value
            / "all"
            / "batch_4_gradcam_plus_plus.pt"
        )


@pytest.mark.unit
class TestWriterHelpers:
    @pytest.mark.parametrize(
        "image_format", ["", "nested/png", "\\bad", ".", "notaformat"]
    )
    def test_invalid_overlay_extension_raises(self, image_format: str):
        with pytest.raises(ValueError):
            ExplanationWriter.normalize_image_extension(image_format)

    def test_extension_gains_leading_dot(self):
        assert ExplanationWriter.normalize_image_extension("PNG ") == ".png"

    def test_single_channel_image_is_replicated_to_rgb(self):
        image = torch.rand(1, 4, 4)
        array = ExplanationWriter.image_tensor_to_numpy(image)
        assert array.shape == (4, 4, 3)
        assert np.allclose(array[..., 0], array[..., 1])

    def test_invalid_channel_count_raises(self):
        with pytest.raises(ValueError, match="1 or 3 channels"):
            ExplanationWriter.image_tensor_to_numpy(torch.rand(2, 4, 4))

    def test_out_of_range_image_is_rescaled(self):
        image = torch.tensor([[[-1.0, 3.0]]]).repeat(3, 1, 1)
        array = ExplanationWriter.image_tensor_to_numpy(image)
        assert float(array.min()) >= 0.0
        assert float(array.max()) <= 1.0

    @pytest.fixture
    def writer(self, tmp_path: Path) -> ExplanationWriter:
        return ExplanationWriter(
            output_directory=tmp_path,
            image_weight=0.5,
            overlay_image_format="png",
        )

    def test_sample_label_prefers_sample_indices(self, writer: ExplanationWriter):
        label = writer.sample_label(
            metadata={"sample_indices": [7, 9]}, batch_index=1, batch_counter=0
        )
        assert label == "sample_9"

    def test_sample_label_uses_environment_and_timestep(
        self, writer: ExplanationWriter
    ):
        label = writer.sample_label(
            metadata={"environment_indices": [4], "timestep": 12},
            batch_index=0,
            batch_counter=3,
        )
        assert label == "env_4_step_12"

    def test_sample_label_falls_back_to_batch_counter(self, writer: ExplanationWriter):
        label = writer.sample_label(metadata={}, batch_index=2, batch_counter=5)
        assert label == "batch_5_row_2"
