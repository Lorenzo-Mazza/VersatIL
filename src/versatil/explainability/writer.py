"""Disk writer for explainability outputs."""

import re
from pathlib import Path

import cv2
import numpy as np
import torch

from versatil.explainability.sources.typedefs import ExplanationBatch
from versatil.explainability.visualization import show_cam_on_image

FILENAME_PLUS_TOKEN = "_plus"
UNSAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9_.-]+")
REPEATED_UNDERSCORE_PATTERN = re.compile(r"_+")


class ExplanationWriter:
    """Write raw heatmaps and image overlays to disk."""

    def __init__(
        self,
        output_directory: Path,
        image_weight: float,
        overlay_image_format: str,
    ) -> None:
        """Initialize the writer.

        Args:
            output_directory: Root directory where explanation files are written.
            image_weight: Original-image blend weight used when saving overlay
                images.
            overlay_image_format: Image file format for overlays. Values may
                include a leading dot, such as ``png`` or ``.jpg``.

        Raises:
            ValueError: If ``overlay_image_format`` is empty, path-like, or not
                supported by OpenCV on this system.
        """
        self.output_directory = output_directory
        self.image_weight = image_weight
        self.overlay_image_extension = self.normalize_image_extension(
            image_format=overlay_image_format
        )

    def save_raw_heatmaps(
        self,
        heatmaps: dict[str, torch.Tensor],
        explanation_type: str,
        metadata: dict,
        batch_counter: int,
    ) -> None:
        """Write heatmap tensors for one batch and method to disk.

        Args:
            heatmaps: Heatmaps keyed by camera name.
            explanation_type: Explanation method that produced the maps.
            metadata: Source metadata for this batch.
            batch_counter: Runner-local batch ordinal used in filenames.
        """
        output_path = (
            self.output_directory
            / self.source_directory(metadata=metadata)
            / f"batch_{batch_counter}_{self.sanitize(explanation_type)}.pt"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": metadata,
                "heatmaps": {
                    camera: heatmap.detach().float().cpu()
                    for camera, heatmap in heatmaps.items()
                },
            },
            output_path,
        )

    def save_overlays(
        self,
        heatmaps: dict[str, torch.Tensor],
        explanation_type: str,
        batch: ExplanationBatch,
        batch_counter: int,
    ) -> None:
        """Write image overlays for one batch and method to disk.

        Args:
            heatmaps: Heatmaps keyed by camera name.
            explanation_type: Explanation method that produced the maps.
            batch: Source batch containing display images and metadata.
            batch_counter: Runner-local batch ordinal used in fallback names.

        Raises:
            RuntimeError: If a heatmap has no matching display camera tensor.
        """
        output_directory = self.output_directory / self.source_directory(
            metadata=batch.metadata
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        for camera, heatmap in heatmaps.items():
            image_tensor = batch.display_observation.get(camera)
            if image_tensor is None:
                raise RuntimeError(
                    f"No display observation found for heatmap camera '{camera}'."
                )
            cpu_heatmap = heatmap.detach().float().cpu()
            for batch_index in range(cpu_heatmap.shape[0]):
                sample_label = self.sample_label(
                    metadata=batch.metadata,
                    batch_index=batch_index,
                    batch_counter=batch_counter,
                )
                for temporal_index in range(cpu_heatmap.shape[1]):
                    image = self.select_image_tensor(
                        image_tensor=image_tensor,
                        batch_index=batch_index,
                        temporal_index=temporal_index,
                    )
                    overlay = show_cam_on_image(
                        image=self.image_tensor_to_numpy(image=image),
                        mask=cpu_heatmap[batch_index, temporal_index].numpy(),
                        use_rgb=True,
                        image_weight=self.image_weight,
                    )
                    filename = (
                        f"{sample_label}_t{temporal_index}_"
                        f"{self.sanitize(explanation_type)}_"
                        f"{self.sanitize(camera)}{self.overlay_image_extension}"
                    )
                    cv2.imwrite(
                        str(output_directory / filename),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                    )

    @staticmethod
    def normalize_image_extension(image_format: str) -> str:
        """Normalize and validate an overlay image extension.

        Args:
            image_format: Image file format, with or without a leading dot.

        Returns:
            Lowercase extension including the leading dot.

        Raises:
            ValueError: If ``image_format`` is empty, path-like, or not
                supported by OpenCV on this system.
        """
        extension = image_format.strip().lower()
        if not extension:
            raise ValueError("overlay_image_format cannot be empty.")
        if "/" in extension or "\\" in extension:
            raise ValueError(
                f"overlay_image_format must be a file extension, got: {image_format}"
            )
        if not extension.startswith("."):
            extension = f".{extension}"
        if extension == ".":
            raise ValueError("overlay_image_format cannot be only '.'.")
        if not cv2.haveImageWriter(extension):
            raise ValueError(
                f"OpenCV cannot write overlay images with extension '{extension}'."
            )
        return extension

    @staticmethod
    def select_image_tensor(
        image_tensor: torch.Tensor,
        batch_index: int,
        temporal_index: int,
    ) -> torch.Tensor:
        """Select one ``(C, H, W)`` image from a camera batch.

        Args:
            image_tensor: Camera tensor with shape ``(B, T, C, H, W)`` or
                ``(B, C, H, W)``.
            batch_index: Batch row index.
            temporal_index: Observation-window index.

        Returns:
            Single image tensor with shape ``(C, H, W)``.

        Raises:
            ValueError: If ``image_tensor`` rank is unsupported.
        """
        if image_tensor.dim() == 5:
            return image_tensor[batch_index, temporal_index]
        if image_tensor.dim() == 4:
            return image_tensor[batch_index]
        raise ValueError(
            f"Display image tensor must be 4D or 5D. Got: {tuple(image_tensor.shape)}"
        )

    @staticmethod
    def image_tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
        """Convert one image tensor to normalized RGB numpy format.

        Args:
            image: Image tensor with shape ``(C, H, W)``.

        Returns:
            RGB image array in ``[0, 1]``.

        Raises:
            ValueError: If the channel count is not 1 or 3.
        """
        if image.shape[0] == 3:
            image_array = image.float().permute(1, 2, 0).numpy()
        elif image.shape[0] == 1:
            single_channel = image[0].float().numpy()
            image_array = np.repeat(single_channel[:, :, np.newaxis], 3, axis=2)
        else:
            raise ValueError(
                f"Overlay images must have 1 or 3 channels. Got: {image.shape[0]}"
            )

        image_array = np.nan_to_num(image_array)
        minimum = float(np.min(image_array))
        maximum = float(np.max(image_array))
        if minimum < 0.0 or maximum > 1.0:
            image_array = (image_array - minimum) / (maximum - minimum + 1e-8)
        return np.clip(image_array, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def sanitize(value: str) -> str:
        """Convert metadata strings into filesystem-safe fragments.

        Args:
            value: Raw value used in a path or filename.

        Returns:
            String containing only path-safe alphanumeric, underscore, dot, and
            dash characters.
        """
        sanitized = value.strip().replace("+", FILENAME_PLUS_TOKEN)
        sanitized = UNSAFE_FILENAME_PATTERN.sub("_", sanitized)
        sanitized = REPEATED_UNDERSCORE_PATTERN.sub("_", sanitized)
        return sanitized.strip("_")

    @staticmethod
    def source_directory(metadata: dict) -> Path:
        """Return the source-specific output subdirectory.

        Args:
            metadata: Source metadata attached to the explanation batch.

        Returns:
            ``source`` or ``source/split`` path fragment.
        """
        source = str(metadata.get("source", "unknown"))
        split = metadata.get("split")
        if split is None:
            return Path(source)
        return Path(source) / str(split)

    def sample_label(
        self,
        metadata: dict,
        batch_index: int,
        batch_counter: int,
    ) -> str:
        """Build a stable filename prefix for one batch row.

        Args:
            metadata: Source metadata for the current batch.
            batch_index: Row within the current batch.
            batch_counter: Runner-local batch ordinal used as a fallback.

        Returns:
            Filesystem-safe label.
        """
        sample_indices = metadata.get("sample_indices")
        if isinstance(sample_indices, list) and batch_index < len(sample_indices):
            return f"sample_{sample_indices[batch_index]}"

        environment_indices = metadata.get("environment_indices")
        if isinstance(environment_indices, list) and batch_index < len(
            environment_indices
        ):
            timestep = metadata.get("timestep", batch_counter)
            return f"env_{environment_indices[batch_index]}_step_{timestep}"

        return f"batch_{batch_counter}_row_{batch_index}"
