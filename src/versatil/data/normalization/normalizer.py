"""Linear normalization module (min-max scaling, z-score, demean).
Inspired by https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/common/normalizer.py
"""

import numpy as np
import torch
import torch.nn as nn

from versatil.common.dict_of_tensor_mixin import DictOfTensorMixin
from versatil.common.tensor_ops import dict_apply
from versatil.data.constants import KinematicsNormalizationType


class LinearNormalizer(DictOfTensorMixin):
    """Multi-key normalizer that stores per-key normalization parameters.

    Supports dict input (per-key normalization) and tensor input (single-key '_default').
    """

    available_modes = [mode.value for mode in KinematicsNormalizationType]

    @torch.no_grad()
    def fit(
        self,
        data: dict | torch.Tensor | np.ndarray,
        last_n_dims: int = 1,
        dtype: torch.dtype = torch.float32,
        mode: str = KinematicsNormalizationType.MIN_MAX.value,
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
        fit_offset: bool = True,
        device: torch.device | str | None = None,
        clamp_range: bool = True,
        min_std: float = 2e-2,
        min_range: float = 4e-2,
        sample_size: int | dict[str, int] = 0,
    ):
        """Fit normalization parameters to data.

        Args:
            data: Dict of tensors (per-key fit) or a single tensor (stored under '_default').
            last_n_dims: Number of trailing dimensions to treat as features.
            dtype: Data type for the normalizer parameters.
            mode: Normalization mode ('min_max', 'gaussian', or 'demean').
            output_max: Maximum output value for min_max mode.
            output_min: Minimum output value for min_max mode.
            range_eps: Epsilon for detecting constant dimensions.
            fit_offset: Whether to fit an offset (centering).
            device: Device to place parameters on.
            clamp_range: Whether to clamp std/range to minimum values.
            min_std: Minimum std value when clamp_range=True and mode='gaussian'.
            min_range: Minimum range value when clamp_range=True and mode='min_max'.
            sample_size: Number of data rows to store as an empirical sample under
                each key. Pass an int to use the same budget for every key, or a
                ``{key: size}`` dict to store samples only for selected keys
                (missing keys default to 0 = no sample stored).
        """
        if isinstance(data, dict):
            for key, value in data.items():
                per_key_size = (
                    sample_size.get(key, 0)
                    if isinstance(sample_size, dict)
                    else sample_size
                )
                self.params_dict[key] = _fit(
                    value,
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset,
                    device=device,
                    clamp_range=clamp_range,
                    min_std=min_std,
                    min_range=min_range,
                    sample_size=per_key_size,
                )
        else:
            single_size = (
                sample_size.get("_default", 0)
                if isinstance(sample_size, dict)
                else sample_size
            )
            self.params_dict["_default"] = _fit(
                data,
                last_n_dims=last_n_dims,
                dtype=dtype,
                mode=mode,
                output_max=output_max,
                output_min=output_min,
                range_eps=range_eps,
                fit_offset=fit_offset,
                device=device,
                clamp_range=clamp_range,
                min_std=min_std,
                min_range=min_range,
                sample_size=single_size,
            )

    def __call__(self, x: dict | torch.Tensor | np.ndarray) -> dict | torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        params = self.params_dict[key]
        if "_is_sequential" in params and params["_is_sequential"].item() == 1.0:
            num_normalizers = int(params["_num_normalizers"].item())
            normalizers = []

            for i in range(num_normalizers):
                scale = params[f"_norm_{i}_scale"]
                offset = params[f"_norm_{i}_offset"]
                stats = {}
                for stat_key in ["min", "max", "mean", "std"]:
                    if f"_norm_{i}_stat_{stat_key}" in params:
                        stats[stat_key] = params[f"_norm_{i}_stat_{stat_key}"]

                norm = SingleFieldLinearNormalizer.create_manual(
                    scale=scale, offset=offset, input_stats_dict=stats
                )
                normalizers.append(norm)

            return SequentialNormalizer(normalizers=normalizers)
        else:
            return SingleFieldLinearNormalizer(params)

    def __setitem__(self, key: str, value):
        if isinstance(value, SequentialNormalizer):
            normalizer_list = (
                list(value.normalizers)
                if hasattr(value.normalizers, "__iter__")
                else []
            )

            params_dict = nn.ParameterDict()

            params_dict["scale"] = value.params_dict["scale"].clone().detach()
            params_dict["offset"] = value.params_dict["offset"].clone().detach()

            input_stats = normalizer_list[0].params_dict["input_stats"]
            params_dict["input_stats"] = nn.ParameterDict(
                {k: v.clone().detach() for k, v in input_stats.items()}
            )

            params_dict["_is_sequential"] = nn.Parameter(
                torch.tensor([1.0]), requires_grad=False
            )
            params_dict["_num_normalizers"] = nn.Parameter(
                torch.tensor([float(len(normalizer_list))]), requires_grad=False
            )

            for i, normalizer in enumerate(normalizer_list):
                params_dict[f"_norm_{i}_scale"] = (
                    normalizer.params_dict["scale"].clone().detach()
                )
                params_dict[f"_norm_{i}_offset"] = (
                    normalizer.params_dict["offset"].clone().detach()
                )
                for stat_key, stat_value in normalizer.params_dict[
                    "input_stats"
                ].items():
                    params_dict[f"_norm_{i}_stat_{stat_key}"] = (
                        stat_value.clone().detach()
                    )

            for parameter in params_dict.parameters():
                parameter.requires_grad_(False)

            self.params_dict[key] = params_dict
        else:
            self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        with torch.no_grad():
            if isinstance(x, dict):
                result = {}
                for key, value in x.items():
                    if key not in self.params_dict:
                        result[key] = value
                        continue
                    params = self.params_dict[key]
                    result[key] = _normalize(value, params, forward=forward)
                return result
            else:
                if "_default" not in self.params_dict:
                    raise RuntimeError("Not initialized")
                params = self.params_dict["_default"]
                return _normalize(x, params, forward=forward)

    def normalize(self, x: dict | torch.Tensor | np.ndarray) -> dict | torch.Tensor:
        """Normalize input data. Dict keys without parameters pass through unchanged."""
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: dict | torch.Tensor | np.ndarray) -> dict | torch.Tensor:
        """Unnormalize input data back to original scale."""
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> dict:
        """Get input statistics (min, max, mean, std) for all keys."""
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]

        result = {}
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self, key: str = "_default") -> dict:
        """Get normalized output statistics for a specific key.

        Args:
            key: Parameter key. Use '_default' for tensor-fitted normalizers,
                or a specific key for dict-fitted normalizers.

        Returns:
            Dict with normalized min, max, mean, and scaled std.
        """
        params = self.params_dict[key]
        input_stats = params["input_stats"]
        scale = params["scale"]
        single_field = SingleFieldLinearNormalizer(params)
        result = {}
        if "min" in input_stats:
            result["min"] = single_field.normalize(input_stats["min"])
        if "max" in input_stats:
            result["max"] = single_field.normalize(input_stats["max"])
        if "mean" in input_stats:
            result["mean"] = single_field.normalize(input_stats["mean"])
        if "std" in input_stats:
            result["std"] = torch.abs(scale) * input_stats["std"]
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    """Single-field normalizer with scale and offset parameters."""

    available_modes = [mode.value for mode in KinematicsNormalizationType]

    @torch.no_grad()
    def fit(
        self,
        data: torch.Tensor | np.ndarray,
        last_n_dims: int = 1,
        dtype: torch.dtype = torch.float32,
        mode: str = KinematicsNormalizationType.MIN_MAX.value,
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-10,
        fit_offset: bool = True,
        device: torch.device | str | None = "cpu",
        clamp_range: bool = True,
        min_std: float = 2e-2,
        min_range: float = 4e-2,
        sample_size: int = 0,
    ):
        """Fit scale and offset for one field from data statistics."""
        self.params_dict = _fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
            device=device,
            clamp_range=clamp_range,
            min_std=min_std,
            min_range=min_range,
            sample_size=sample_size,
        )

    @classmethod
    def create_fit(
        cls,
        data: torch.Tensor | np.ndarray,
        last_n_dims: int = 1,
        dtype: torch.dtype = torch.float32,
        mode: str = KinematicsNormalizationType.MIN_MAX.value,
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-10,
        fit_offset: bool = True,
        device: torch.device | str | None = "cpu",
        clamp_range: bool = True,
        min_std: float = 2e-2,
        min_range: float = 4e-2,
    ):
        """Create a fitted normalizer in one step."""
        obj = cls()
        obj.fit(
            data,
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset,
            device=device,
            clamp_range=clamp_range,
            min_std=min_std,
            min_range=min_range,
        )
        return obj

    @classmethod
    def create_manual(
        cls,
        scale: torch.Tensor | np.ndarray,
        offset: torch.Tensor | np.ndarray,
        input_stats_dict: dict[str, torch.Tensor | np.ndarray],
    ):
        """Create a normalizer from explicit scale, offset, and input stats."""

        def to_tensor(x):
            """Convert input to a flat float tensor."""
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x

        for tensor in [offset] + list(input_stats_dict.values()):
            if tensor.shape != scale.shape:
                raise ValueError(
                    f"All inputs must have same shape as scale {scale.shape}, got {tensor.shape}"
                )
            if tensor.dtype != scale.dtype:
                raise ValueError(
                    f"All inputs must have same dtype as scale {scale.dtype}, got {tensor.dtype}"
                )

        params_dict = nn.ParameterDict(
            {
                "scale": to_tensor(scale),
                "offset": to_tensor(offset),
                "input_stats": nn.ParameterDict(
                    dict_apply(input_stats_dict, to_tensor)
                ),
            }
        )
        for parameter in params_dict.parameters():
            parameter.requires_grad_(False)
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype: torch.dtype = torch.float32):
        """Create an identity normalizer (scale=1, offset=0)."""
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Apply normalization: x * scale + offset."""
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Apply inverse normalization: (x - offset) / scale."""
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self) -> nn.ParameterDict:
        """Get input statistics (min, max, mean, std)."""
        return self.params_dict["input_stats"]

    def get_output_stats(self) -> dict:
        """Get normalized output statistics."""
        input_stats = self.params_dict["input_stats"]
        scale = self.params_dict["scale"]
        result = {}
        if "min" in input_stats:
            result["min"] = self.normalize(input_stats["min"])
        if "max" in input_stats:
            result["max"] = self.normalize(input_stats["max"])
        if "mean" in input_stats:
            result["mean"] = self.normalize(input_stats["mean"])
        if "std" in input_stats:
            result["std"] = torch.abs(scale) * input_stats["std"]
        return result

    def get_input_sample(self) -> torch.Tensor | None:
        """Return the raw subsample stored at fit time, or ``None`` if absent.

        Present only when ``fit(..., sample_size>0)`` was used.
        """
        if "sample" in self.params_dict:
            return self.params_dict["sample"]
        return None

    def get_output_sample(self) -> torch.Tensor | None:
        """Return the stored subsample in normalized output space, or ``None``."""
        raw_sample = self.get_input_sample()
        if raw_sample is None:
            return None
        return self.normalize(raw_sample)

    def __call__(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        return self.normalize(x)


def _fit(
    data: torch.Tensor | np.ndarray,
    last_n_dims: int = 1,
    dtype: torch.dtype = torch.float32,
    mode: str = KinematicsNormalizationType.MIN_MAX.value,
    output_max: float = 1.0,
    output_min: float = -1.0,
    range_eps: float = 1e-10,
    fit_offset: bool = True,
    device: torch.device | str | None = None,
    clamp_range: bool = True,
    min_std: float = 2e-2,
    min_range: float = 4e-2,
    sample_size: int = 0,
) -> nn.ParameterDict:
    """Fit normalization parameters to data.

    Args:
        data: Input data to fit normalization on.
        last_n_dims: Number of trailing dimensions to treat as features.
        dtype: Data type for the normalizer parameters.
        mode: Normalization mode ('min_max', 'gaussian', or 'demean').
        output_max: Maximum output value for min_max mode.
        output_min: Minimum output value for min_max mode.
        range_eps: Epsilon for detecting constant dimensions.
        fit_offset: Whether to fit an offset (centering).
        device: Device to place parameters on.
        clamp_range: Whether to clamp std/range to minimum values.
        min_std: Minimum std value when clamp_range=True and mode='gaussian'.
        min_range: Minimum range value when clamp_range=True and mode='min_max'.
        sample_size: If > 0, store a random subsample of the flattened input data
            under key ``sample`` in the returned ParameterDict. The sample
            preserves the empirical distribution of the data and can be used
            by downstream consumers that need density information, not just
            marginal statistics (e.g. data-aware GMM init). The subsample is
            drawn without replacement and capped at ``min(sample_size, N)``.

    Returns:
        ParameterDict with scale, offset, input_stats, and optionally sample.
    """
    valid_modes = [mode_type.value for mode_type in KinematicsNormalizationType]
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")
    if last_n_dims < 0:
        raise ValueError(f"last_n_dims must be >= 0, got {last_n_dims}")
    if output_max <= output_min:
        raise ValueError(
            f"output_max ({output_max}) must be greater than output_min ({output_min})"
        )

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    tensor_data: torch.Tensor = data
    if dtype is not None:
        tensor_data = tensor_data.type(dtype)
    if device is not None:
        tensor_data = tensor_data.to(device)

    dim = 1
    if last_n_dims > 0:
        dim = int(np.prod(tensor_data.shape[-last_n_dims:]))
    tensor_data = tensor_data.reshape(-1, dim)

    input_min, _ = tensor_data.min(dim=0)
    input_max, _ = tensor_data.max(dim=0)
    input_mean = tensor_data.mean(dim=0)
    input_std = tensor_data.std(dim=0)

    if mode == KinematicsNormalizationType.MIN_MAX.value:
        if fit_offset:
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            if clamp_range:
                input_range = torch.clamp(input_range, min=min_range)
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        else:
            if output_max <= 0:
                raise ValueError(
                    f"output_max must be > 0 when fit_offset=False, got {output_max}"
                )
            if output_min >= 0:
                raise ValueError(
                    f"output_min must be < 0 when fit_offset=False, got {output_min}"
                )
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            if clamp_range:
                input_abs = torch.clamp(input_abs, min=min_range)
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == KinematicsNormalizationType.GAUSSIAN.value:
        std_for_scale = input_std.clone()
        ignore_dim = input_std < range_eps
        if clamp_range:
            std_for_scale = torch.clamp(std_for_scale, min=min_std)
        std_for_scale[ignore_dim] = 1
        scale = 1 / std_for_scale
        offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)
    elif mode == KinematicsNormalizationType.DEMEAN.value:
        scale = torch.ones_like(input_mean)
        offset = -input_mean if fit_offset else torch.zeros_like(input_mean)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    parameters = nn.ParameterDict(
        {
            "scale": scale,
            "offset": offset,
            "input_stats": nn.ParameterDict(
                {
                    "min": input_min,
                    "max": input_max,
                    "mean": input_mean,
                    "std": input_std,
                }
            ),
        }
    )
    if sample_size > 0:
        n_rows = tensor_data.shape[0]
        take = min(sample_size, n_rows)
        indices = torch.randperm(n_rows, device=tensor_data.device)[:take]
        parameters["sample"] = tensor_data[indices].clone().detach()
    for parameter in parameters.parameters():
        parameter.requires_grad_(False)
    return parameters


def _trailing_dims_flatten_to(shape: torch.Size, stat_dim: int) -> bool:
    """Whether a suffix of the shape multiplies to exactly stat_dim elements."""
    trailing_size = 1
    for dim_size in reversed(shape):
        trailing_size *= dim_size
        if trailing_size == stat_dim:
            return True
        if trailing_size > stat_dim:
            return False
    return False


@torch.no_grad()
def _normalize(
    x: torch.Tensor | np.ndarray,
    params: nn.ParameterDict,
    forward: bool = True,
) -> torch.Tensor:
    """Apply linear normalization (forward) or its inverse (backward).

    Args:
        x: Input tensor or numpy array.
        params: ParameterDict with 'scale' and 'offset' keys.
        forward: If True, compute x * scale + offset. If False, compute (x - offset) / scale.

    Returns:
        Normalized or unnormalized tensor with same shape as input.
    """
    if "scale" not in params:
        raise ValueError("params must contain 'scale' key")
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    stat_dim = scale.shape[0]
    source_shape = x.shape
    if stat_dim == 1:
        pass  # scalar stats broadcast over any shape
    elif _trailing_dims_flatten_to(shape=source_shape, stat_dim=stat_dim):
        x = x.reshape(-1, stat_dim)
    elif x.ndim >= 3 and x.shape[-3] == stat_dim:
        # Channels-first image data: broadcast over the channel dimension.
        scale = scale.view(stat_dim, 1, 1)
        offset = offset.view(stat_dim, 1, 1)
    else:
        raise ValueError(
            f"Cannot align normalizer stats of size {stat_dim} with input of "
            f"shape {tuple(source_shape)}: expected trailing dimensions to "
            "flatten to the stat size or, for channels-first images, the "
            "third-to-last dimension to match."
        )
    x = x * scale + offset if forward else (x - offset) / scale
    return x.reshape(source_shape)


class SequentialNormalizer(SingleFieldLinearNormalizer):
    """Applies a sequence of already-fitted normalizers."""

    def __init__(self, normalizers: list[SingleFieldLinearNormalizer]):
        super().__init__()
        self.normalizers = nn.ModuleList(normalizers)
        if len(self.normalizers) == 0:
            raise ValueError("At least one normalizer required")
        effective_scale = self.normalizers[0].params_dict["scale"].clone().detach()
        effective_offset = self.normalizers[0].params_dict["offset"].clone().detach()
        for normalizer in self.normalizers[1:]:
            normalizer_scale = normalizer.params_dict["scale"].clone().detach()
            normalizer_offset = normalizer.params_dict["offset"].clone().detach()
            effective_offset = normalizer_offset + normalizer_scale * effective_offset
            effective_scale = normalizer_scale * effective_scale

        input_stats = self.normalizers[0].params_dict["input_stats"]
        self.params_dict = nn.ParameterDict(
            {
                "scale": nn.Parameter(effective_scale, requires_grad=False),
                "offset": nn.Parameter(effective_offset, requires_grad=False),
                "input_stats": nn.ParameterDict(
                    {
                        k: nn.Parameter(v.clone().detach(), requires_grad=False)
                        for k, v in input_stats.items()
                    }
                ),
            }
        )

    def fit(
        self,
        data: torch.Tensor | np.ndarray = None,
        last_n_dims: int = 1,
        dtype: torch.dtype = torch.float32,
        mode: str = KinematicsNormalizationType.MIN_MAX.value,
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-10,
        fit_offset: bool = True,
        device: torch.device | str | None = "cpu",
        clamp_range: bool = True,
        min_std: float = 2e-2,
        min_range: float = 4e-2,
    ) -> None:
        """Fit per-key normalizers over a dict of arrays."""
        raise NotImplementedError("SequentialNormalizer does not support fitting")

    def normalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Apply all normalizers in sequence."""
        for normalizer in self.normalizers:
            x = normalizer.normalize(x)
        return x

    def unnormalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Apply all normalizers in reverse sequence."""
        for normalizer in reversed(self.normalizers):
            x = normalizer.unnormalize(x)
        return x

    def get_input_stats(self) -> nn.ParameterDict:
        """Get input stats of the first normalizer."""
        return self.normalizers[0].get_input_stats()

    def get_output_stats(self) -> dict:
        """Get output stats after propagating through all normalizers."""
        stats = self.get_input_stats()
        for normalizer in self.normalizers:
            scale = normalizer.params_dict["scale"]
            offset = normalizer.params_dict["offset"]
            stats = {
                "min": scale * stats["min"] + offset,
                "max": scale * stats["max"] + offset,
                "mean": scale * stats["mean"] + offset,
                "std": torch.abs(scale) * stats["std"],
            }
        return stats
