"""A normalizer nn.Module that performs linear normalization (min-max scaling or z-score normalization).

Taken from the diffusion policy repository.
"""
import numpy as np
import torch
import torch.nn as nn

from versatil.common.tensor_ops import dict_apply
from versatil.common.dict_of_tensor_mixin import DictOfTensorMixin
from versatil.data.constants import KinematicsNormalizationType


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = [
        KinematicsNormalizationType.MIN_MAX.value,
        KinematicsNormalizationType.GAUSSIAN.value,
        KinematicsNormalizationType.DEMEAN.value,
    ]

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
    ):
        if isinstance(data, dict):
            for key, value in data.items():
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
                )
        else:
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
            )

    def __call__(self, x: dict | torch.Tensor | np.ndarray) -> dict | torch.Tensor:
        return self.normalize(x)

    def __getitem__(self, key: str):
        params = self.params_dict[key]
        # Check if this is a sequential normalizer by looking for the metadata
        if "_is_sequential" in params and params["_is_sequential"].item() == 1.0:
            # Reconstruct SequentialNormalizer from stored params
            num_normalizers = int(params["_num_normalizers"].item())
            normalizers = []

            for i in range(num_normalizers):
                # Reconstruct each normalizer's params
                scale = params[f"_norm_{i}_scale"]
                offset = params[f"_norm_{i}_offset"]
                stats = {}
                for k in ["min", "max", "mean", "std"]:
                    if f"_norm_{i}_stat_{k}" in params:
                        stats[k] = params[f"_norm_{i}_stat_{k}"]

                norm = SingleFieldLinearNormalizer.create_manual(
                    scale=scale, offset=offset, input_stats_dict=stats
                )
                normalizers.append(norm)

            return SequentialNormalizer(normalizers=normalizers)
        else:
            return SingleFieldLinearNormalizer(params)

    def __setitem__(self, key: str, value):
        if isinstance(value, SequentialNormalizer):
            # Extract normalizers list
            normalizer_list = (
                list(value.normalizers)
                if hasattr(value.normalizers, "__iter__")
                else []
            )

            # Create enhanced params_dict with metadata for reconstruction
            params_dict = nn.ParameterDict()

            # Store the effective scale/offset from SequentialNormalizer
            params_dict["scale"] = value.params_dict["scale"].clone().detach()
            params_dict["offset"] = value.params_dict["offset"].clone().detach()

            # Store input stats from first normalizer
            input_stats = normalizer_list[0].params_dict["input_stats"]
            params_dict["input_stats"] = nn.ParameterDict(
                {k: v.clone().detach() for k, v in input_stats.items()}
            )

            # Store metadata
            params_dict["_is_sequential"] = nn.Parameter(
                torch.tensor([1.0]), requires_grad=False
            )
            params_dict["_num_normalizers"] = nn.Parameter(
                torch.tensor([float(len(normalizer_list))]), requires_grad=False
            )

            # Store individual normalizer params for reconstruction
            for i, n in enumerate(normalizer_list):
                params_dict[f"_norm_{i}_scale"] = (
                    n.params_dict["scale"].clone().detach()
                )
                params_dict[f"_norm_{i}_offset"] = (
                    n.params_dict["offset"].clone().detach()
                )
                for k, v in n.params_dict["input_stats"].items():
                    params_dict[f"_norm_{i}_stat_{k}"] = v.clone().detach()

            # Ensure no gradients
            for p in params_dict.parameters():
                p.requires_grad_(False)

            self.params_dict[key] = params_dict
        else:
            # Regular SingleFieldLinearNormalizer
            self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        with torch.no_grad():
            if isinstance(x, dict):
                result = {}
                for key, value in x.items():
                    # Skip keys that don't have normalization parameters (e.g., language)
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
        return self._normalize_impl(x, forward=True)  # type: ignore[no-any-return]

    def unnormalize(self, x: dict | torch.Tensor | np.ndarray) -> dict | torch.Tensor:
        return self._normalize_impl(x, forward=False)  # type: ignore[no-any-return]

    def get_input_stats(self) -> dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and "_default" in self.params_dict:
            return self.params_dict["_default"]["input_stats"]  # type: ignore[no-any-return]

        result = {}
        for key, value in self.params_dict.items():
            if key != "_default":
                result[key] = value["input_stats"]
        return result

    def get_output_stats(self, key="_default"):
        input_stats = self.params_dict["input_stats"]
        scale = self.params_dict["scale"]
        dict_to_return = {}
        if "min" in input_stats:
            dict_to_return["min"] = self.normalize(input_stats["min"])
        if "max" in input_stats:
            dict_to_return["max"] = self.normalize(input_stats["max"])
        if "mean" in input_stats:
            dict_to_return["mean"] = self.normalize(input_stats["mean"])
        if "std" in input_stats:
            dict_to_return["std"] = torch.abs(scale) * input_stats["std"]
        return dict_to_return


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = [
        KinematicsNormalizationType.MIN_MAX.value,
        KinematicsNormalizationType.GAUSSIAN.value,
        KinematicsNormalizationType.DEMEAN.value,
    ]

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
    ):
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
        )

    @classmethod
    def create_fit(cls, data: torch.Tensor | np.ndarray, **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj

    @classmethod
    def create_manual(
        cls,
        scale: torch.Tensor | np.ndarray,
        offset: torch.Tensor | np.ndarray,
        input_stats_dict: dict[str, torch.Tensor | np.ndarray],
    ):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x

        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype

        params_dict = nn.ParameterDict(
            {
                "scale": to_tensor(scale),
                "offset": to_tensor(offset),
                "input_stats": nn.ParameterDict(
                    dict_apply(input_stats_dict, to_tensor)
                ),  # type: ignore[arg-type]
            }
        )
        for p in params_dict.parameters():
            p.requires_grad_(False)
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            "min": torch.tensor([-1], dtype=dtype),
            "max": torch.tensor([1], dtype=dtype),
            "mean": torch.tensor([0], dtype=dtype),
            "std": torch.tensor([1], dtype=dtype),
        }
        return cls.create_manual(scale, offset, input_stats_dict)  # type: ignore[arg-type]

    def normalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)  # type: ignore[no-any-return]

    def unnormalize(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)  # type: ignore[no-any-return]

    def get_input_stats(self):
        return self.params_dict["input_stats"]

    def get_output_stats(self):
        input_stats = self.params_dict["input_stats"]
        scale = self.params_dict["scale"]
        dict_to_return = {}
        if "min" in input_stats:
            dict_to_return["min"] = self.normalize(input_stats["min"])
        if "max" in input_stats:
            dict_to_return["max"] = self.normalize(input_stats["max"])
        if "mean" in input_stats:
            dict_to_return["mean"] = self.normalize(input_stats["mean"])
        if "std" in input_stats:
            dict_to_return["std"] = torch.abs(scale) * input_stats["std"]
        return dict_to_return

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
):
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
        clamp_range: Whether to clamp std/range to minimum values. Useful for
            datasets with very small action deltas.
        min_std: Minimum std value when clamp_range=True and mode='gaussian'.
        min_range: Minimum range value when clamp_range=True and mode='min_max'.

    Returns:
        ParameterDict with scale, offset, and input_stats.
    """
    assert mode in [
        KinematicsNormalizationType.MIN_MAX.value,
        KinematicsNormalizationType.GAUSSIAN.value,
        KinematicsNormalizationType.DEMEAN.value,
    ]

    assert last_n_dims >= 0
    assert output_max > output_min

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    # After conversion, data is guaranteed to be a Tensor
    tensor_data: torch.Tensor = data  # Narrow type for mypy
    if dtype is not None:
        tensor_data = tensor_data.type(dtype)
    if device is not None:
        tensor_data = tensor_data.to(
            device
        )  # Move data to device before computing stats

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = int(np.prod(tensor_data.shape[-last_n_dims:]))
    tensor_data = tensor_data.reshape(-1, dim)

    # compute input stats min max mean std
    input_min, _ = tensor_data.min(dim=0)
    input_max, _ = tensor_data.max(dim=0)
    input_mean = tensor_data.mean(dim=0)
    input_std = tensor_data.std(dim=0)

    # compute scale and offset
    if mode == KinematicsNormalizationType.MIN_MAX.value:
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            if clamp_range:
                input_range = torch.clamp(input_range, min=min_range)
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            if clamp_range:
                input_abs = torch.clamp(input_abs, min=min_range)
            # don't scale constant channels
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == KinematicsNormalizationType.GAUSSIAN.value:
        # Optionally clamp std to avoid too small values
        std_for_scale = input_std.clone()
        if clamp_range:
            std_for_scale = torch.clamp(std_for_scale, min=min_std)
        ignore_dim = std_for_scale < range_eps
        std_for_scale[ignore_dim] = 1
        scale = 1 / std_for_scale

        offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)
    elif mode == KinematicsNormalizationType.DEMEAN.value:
        # Demean only: subtract mean, no scaling
        scale = torch.ones_like(input_mean)
        offset = -input_mean if fit_offset else torch.zeros_like(input_mean)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # save
    this_params = nn.ParameterDict(
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
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


@torch.no_grad()
def _normalize(x, params, forward=True):
    assert "scale" in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params["scale"]
    offset = params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    x = x * scale + offset if forward else (x - offset) / scale
    x = x.reshape(src_shape)
    return x


class SequentialNormalizer(SingleFieldLinearNormalizer):
    """
    A normalizer that applies a sequence of normalizers. The normalizers must be instances of SingleFieldLinearNormalizer
     and be already fitted.
    """

    def __init__(self, normalizers):
        super().__init__()
        self.normalizers = nn.ModuleList(normalizers)
        if len(self.normalizers) == 0:
            raise ValueError("At least one normalizer required")
        # Compute effective scale and offset
        eff_scale = self.normalizers[0].params_dict["scale"].clone().detach()
        eff_offset = self.normalizers[0].params_dict["offset"].clone().detach()
        for n in self.normalizers[1:]:
            s = n.params_dict["scale"].clone().detach()
            o = n.params_dict["offset"].clone().detach()
            eff_offset = o + s * eff_offset
            eff_scale = s * eff_scale

        input_stats = self.normalizers[0].params_dict["input_stats"]
        self.params_dict = nn.ParameterDict(
            {
                "scale": nn.Parameter(eff_scale, requires_grad=False),
                "offset": nn.Parameter(eff_offset, requires_grad=False),
                "input_stats": nn.ParameterDict(
                    {
                        k: nn.Parameter(v.clone().detach(), requires_grad=False)
                        for k, v in input_stats.items()
                    }
                ),
            }
        )

    def fit(self, *args, **kwargs):
        raise NotImplementedError("SequentialNormalizer does not support fitting")

    def normalize(self, x):
        """Apply all normalizers in sequence."""
        for n in self.normalizers:
            x = n.normalize(x)
        return x

    def unnormalize(self, x):
        """Apply all normalizers in reverse sequence."""
        for n in reversed(self.normalizers):
            x = n.unnormalize(x)
        return x

    def get_input_stats(self):
        """Get input stats of the first normalizer."""
        return self.normalizers[0].get_input_stats()

    def get_output_stats(self):
        """Get output stats of the last normalizer."""
        stats = self.get_input_stats()
        for n in self.normalizers:
            scale = n.params_dict["scale"]
            offset = n.params_dict["offset"]
            stats = {
                "min": scale * stats["min"] + offset,
                "max": scale * stats["max"] + offset,
                "mean": scale * stats["mean"] + offset,
                "std": torch.abs(scale) * stats["std"],
            }
        return stats
