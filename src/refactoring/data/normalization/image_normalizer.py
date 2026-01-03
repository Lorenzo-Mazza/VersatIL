# mypy: ignore-errors
import numpy as np
import torch

from refactoring.data.constants import (
    IMAGENET_DEPTH_MEAN,
    IMAGENET_DEPTH_STD,
    IMAGENET_RGB_MEAN,
    IMAGENET_RGB_STD,
    ImageNormalizationType,
)
from refactoring.data.normalization.normalizer import (
    LinearNormalizer,
    SequentialNormalizer,
    SingleFieldLinearNormalizer,
)


def create_image_normalizer(
    input_min: float | np.ndarray,
    input_max: float | np.ndarray,
    input_mean: float | np.ndarray,
    input_std: float | np.ndarray,
    norm_type: str,
    device: torch.device | None = None,
    standardization_mean: float | np.ndarray | None = None,
    standardization_std: float | np.ndarray | None = None,
) -> LinearNormalizer | SequentialNormalizer:
    """Create image normalizer with linear scaling and optional standardization.

    Note:
        The function handles RGB (multi-channel) and depth (single-channel) normalization:
        1. Linear scaling from [input_min, input_max] to [output_min, output_max]
        2. Optional standardization using provided mean/std (e.g., ImageNet stats)

    Args:
        input_min: Minimum value(s) in input range (scalar or per-channel array)
        input_max: Maximum value(s) in input range (scalar or per-channel array)
        input_mean: Mean of input data (scalar or per-channel array)
        input_std: Standard deviation of input data (scalar or per-channel array)
        norm_type: Normalization type (from ImageNormalizationType or DepthNormalizationType)
        device: Target device for tensors
        standardization_mean: Mean for optional second-stage standardization
        standardization_std: Std for optional second-stage standardization

    Returns:
        SingleFieldLinearNormalizer for scaling, or SequentialNormalizer for scaling + standardization.
    """
    output_min, output_max = _get_output_range(norm_type)

    stage1 = _create_linear_scaling_normalizer(
        input_min=input_min,
        input_max=input_max,
        input_mean=input_mean,
        input_std=input_std,
        output_min=output_min,
        output_max=output_max,
        device=device,
    )

    if standardization_mean is not None and standardization_std is not None:
        scaled_mean = _compute_scaled_values(
            input_mean, input_min, input_max, output_min, output_max
        )
        scaled_std = _compute_scaled_values(
            input_std, input_min, input_max, output_min, output_max, is_std=True
        )

        stage2 = _create_standardization_normalizer(
            input_min=output_min,
            input_max=output_max,
            input_mean=scaled_mean,
            input_std=scaled_std,
            standardization_mean=standardization_mean,
            standardization_std=standardization_std,
            device=device,
        )
        return SequentialNormalizer(normalizers=[stage1, stage2])

    return stage1  # type: ignore[return-value]


def get_rgb_image_normalizer(
    norm_type: str = ImageNormalizationType.ZERO_TO_ONE.value,
    device: torch.device | None = None,
) -> SingleFieldLinearNormalizer | SequentialNormalizer:
    """Create normalizer for RGB images in [0, 1] range.

    Assumes images have already been converted from uint8 to float32 via /255.
    For IMAGENET type, applies per-channel normalization directly (no scaling needed).

    Args:
        norm_type: Type of normalization (ZERO_TO_ONE, MINUS_ONE_TO_ONE, IMAGENET)
        device: Target device for tensors

    Returns:
        Configured normalizer
    """
    if norm_type == ImageNormalizationType.IMAGENET.value:
        return _create_standardization_normalizer(
            input_min=np.zeros(3, dtype=np.float32),
            input_max=np.ones(3, dtype=np.float32),
            input_mean=np.full(3, 0.5, dtype=np.float32),
            input_std=np.full(3, np.sqrt(1.0 / 12.0), dtype=np.float32),
            standardization_mean=np.array(IMAGENET_RGB_MEAN, dtype=np.float32),
            standardization_std=np.array(IMAGENET_RGB_STD, dtype=np.float32),
            device=device,
        )

    input_min = 0.0
    input_max = 1.0
    input_mean = 0.5
    input_std = np.sqrt(1.0 / 12.0)

    return create_image_normalizer(
        input_min=input_min,
        input_max=input_max,
        input_mean=input_mean,
        input_std=input_std,
        norm_type=norm_type,
        device=device,
    )


def get_depth_image_normalizer(
    input_min: float,
    input_max: float,
    input_mean: float,
    input_std: float,
    norm_type: str = ImageNormalizationType.ZERO_TO_ONE.value,
    device: torch.device | None = None,
) -> SingleFieldLinearNormalizer | SequentialNormalizer:
    """Create normalizer for depth images.

    Convenience wrapper around create_image_normalizer for depth images.
    Handles IMAGENET normalization by adding appropriate standardization.

    Args:
        input_min: Minimum depth value in dataset
        input_max: Maximum depth value in dataset
        input_mean: Mean depth value in dataset
        input_std: Standard deviation of depth values
        norm_type: Type of normalization
        device: Target device for tensors

    Returns:
        Configured normalizer
    """
    if norm_type == ImageNormalizationType.IMAGENET.value:
        standardization_mean = IMAGENET_DEPTH_MEAN
        standardization_std = IMAGENET_DEPTH_STD
    else:
        standardization_mean = None
        standardization_std = None

    return create_image_normalizer(
        input_min=input_min,
        input_max=input_max,
        input_mean=input_mean,
        input_std=input_std,
        norm_type=norm_type,
        device=device,
        standardization_mean=standardization_mean,
        standardization_std=standardization_std,
    )


def get_range_normalizer_from_stat(
    stat: dict,
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-7,
) -> SingleFieldLinearNormalizer:
    """Create normalizer from pre-computed statistics.

    Args:
        stat: Dictionary with 'min', 'max', 'mean', 'std' keys
        output_max: Maximum value of output range
        output_min: Minimum value of output range
        range_eps: Epsilon for handling zero-range dimensions

    Returns:
        Configured normalizer
    """
    input_max = stat["max"]
    input_min = stat["min"]
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def array_to_stats(arr: np.ndarray) -> dict:
    """Convert array to statistics dictionary.

    Args:
        arr: Input array

    Returns:
        Dictionary with min, max, mean, std
    """
    stat = {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
    }
    return stat


def _get_output_range(norm_type: str) -> tuple[float, float]:
    """Determine output range based on normalization type.

    Args:
        norm_type: Normalization type constant

    Returns:
        Tuple of (output_min, output_max)
    """
    if norm_type in [
        ImageNormalizationType.ZERO_TO_ONE.value,
        ImageNormalizationType.IMAGENET.value,
    ]:
        return 0.0, 1.0
    elif norm_type == ImageNormalizationType.MINUS_ONE_TO_ONE.value:
        return -1.0, 1.0
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")


def _to_tensor(
    value: float | np.ndarray,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert scalar or array to tensor.

    Args:
        value: Input value (scalar or array)
        dtype: Target dtype
        device: Target device

    Returns:
        Tensor with appropriate shape
    """
    if isinstance(value, np.ndarray):
        # Handle numpy arrays (including 0-d arrays from .mean(), .std(), etc.)
        if value.ndim == 0:
            # 0-d array (numpy scalar) - convert to 1-d tensor
            tensor = torch.tensor([value.item()], dtype=dtype)
        else:
            tensor = torch.from_numpy(value).to(dtype=dtype)
    elif isinstance(value, (int, float, np.number)):
        # Handle Python scalars and numpy scalar types
        tensor = torch.tensor([float(value)], dtype=dtype)
    else:
        tensor = torch.as_tensor(value, dtype=dtype)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def _create_linear_scaling_normalizer(
    input_min: float | np.ndarray,
    input_max: float | np.ndarray,
    input_mean: float | np.ndarray,
    input_std: float | np.ndarray,
    output_min: float,
    output_max: float,
    device: torch.device | None = None,
) -> SingleFieldLinearNormalizer:
    """Create normalizer for linear scaling.

    Handles both scalar (single-channel) and array (multi-channel) inputs.

    Args:
        input_min: Minimum input value(s)
        input_max: Maximum input value(s)
        input_mean: Mean input value(s)
        input_std: Standard deviation of input(s)
        output_min: Minimum output value
        output_max: Maximum output value
        device: Target device

    Returns:
        Configured linear normalizer
    """
    if isinstance(input_min, (int, float)):
        scale = (output_max - output_min) / (input_max - input_min)
        offset = output_min - scale * input_min
    else:
        scale = (output_max - output_min) / (input_max - input_min)
        offset = output_min - scale * input_min

    scale = _to_tensor(scale, device=device)
    offset = _to_tensor(offset, device=device)

    stat = {
        "min": _to_tensor(input_min, device=device),
        "max": _to_tensor(input_max, device=device),
        "mean": _to_tensor(input_mean, device=device),
        "std": _to_tensor(input_std, device=device),
    }

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def _create_standardization_normalizer(
    input_min: float | np.ndarray,
    input_max: float | np.ndarray,
    input_mean: float | np.ndarray,
    input_std: float | np.ndarray,
    standardization_mean: float | np.ndarray,
    standardization_std: float | np.ndarray,
    device: torch.device | None = None,
) -> SingleFieldLinearNormalizer:
    """Create normalizer for standardization (z-score normalization).

    Handles both scalar (single-channel) and array (multi-channel) inputs.
    For multi-channel, performs per-channel standardization.

    Args:
        input_min: Minimum input value(s) (after first-stage scaling if applicable)
        input_max: Maximum input value(s) (after first-stage scaling if applicable)
        input_mean: Mean input value(s) (after first-stage scaling if applicable)
        input_std: Standard deviation (after first-stage scaling if applicable)
        standardization_mean: Target mean for standardization (scalar or per-channel)
        standardization_std: Target std for standardization (scalar or per-channel)
        device: Target device

    Returns:
        Configured standardization normalizer
    """
    if isinstance(standardization_std, (int, float)):
        scale = 1.0 / standardization_std
        offset = -standardization_mean / standardization_std
    else:
        scale = 1.0 / standardization_std
        offset = -standardization_mean / standardization_std

    scale = _to_tensor(scale, device=device)
    offset = _to_tensor(offset, device=device)

    stat = {
        "min": _to_tensor(input_min, device=device),
        "max": _to_tensor(input_max, device=device),
        "mean": _to_tensor(input_mean, device=device),
        "std": _to_tensor(input_std, device=device),
    }

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )


def _compute_scaled_values(
    values: float | np.ndarray,
    input_min: float | np.ndarray,
    input_max: float | np.ndarray,
    output_min: float,
    output_max: float,
    is_std: bool = False,
) -> float | np.ndarray:
    """Compute values after linear scaling.

    Args:
        values: Original values (mean or std), scalar or array (when multi-channel).
        input_min: Original minimum
        input_max: Original maximum
        output_min: Target minimum
        output_max: Target maximum
        is_std: If True, compute std scaling; if False, compute mean scaling

    Returns:
        Scaled values
    """
    input_range = input_max - input_min
    output_range = output_max - output_min
    if is_std:
        return values / input_range * output_range
    else:
        return output_min + (values - input_min) / input_range * output_range
