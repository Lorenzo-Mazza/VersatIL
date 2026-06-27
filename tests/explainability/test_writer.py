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
