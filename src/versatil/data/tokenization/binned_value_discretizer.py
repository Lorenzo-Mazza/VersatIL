"""Binned discretizer for continuous values.

This module provides the shared binning used by observation and action
tokenization. Uniform binning places equal-width bins over a fixed range and
decodes through bin centers. Quantile binning adapts bin edges to the data
distribution and decodes through per-bin data means, which stays exact for
heavily repeated values (binary grippers, zero deltas) where quantile edges
collapse into duplicates.
"""

import logging
from typing import Any

import numpy as np
import torch

from versatil.data.constants import BinningStrategy


class BinnedValueDiscretizer:
    """Discretizer mapping continuous values to per-dimension bin IDs.

    Attributes:
        num_bins: Number of discrete bins per dimension.
        binning_strategy: Bin-edge placement strategy from ``BinningStrategy``.
        device: Target device for tensors.
        bin_edges: Bin boundaries for each dimension, shape (D, num_bins-1).
        bin_values: Decode value for each bin, shape (D, num_bins).
    """

    def __init__(
        self,
        num_bins: int = 256,
        device: torch.device | None = None,
        binning_strategy: str = BinningStrategy.UNIFORM.value,
        min_value: float = -1.0,
        max_value: float = 1.0,
    ):
        """Initialize discretizer.

        Args:
            num_bins: Number of discrete bins per dimension.
            device: Target device for tensors. If None, uses CPU.
            binning_strategy: ``uniform`` places equal-width bins over
                ``[min_value, max_value]`` and expects inputs normalized to
                that range. ``quantile`` fits edges to the data distribution
                and ignores the range bounds.
            min_value: Lower bound of the uniform bin range.
            max_value: Upper bound of the uniform bin range.

        Raises:
            ValueError: If ``binning_strategy`` is not a known strategy or the
                range bounds are inverted.
        """
        valid_strategies = [member.value for member in BinningStrategy]
        if binning_strategy not in valid_strategies:
            raise ValueError(
                f"Unknown binning_strategy: {binning_strategy}. "
                f"Expected one of {valid_strategies}."
            )
        if min_value >= max_value:
            raise ValueError(
                f"min_value must be smaller than max_value, got "
                f"{min_value} >= {max_value}."
            )
        self.num_bins = num_bins
        self.binning_strategy = binning_strategy
        self.min_value = min_value
        self.max_value = max_value
        self.device = device if device is not None else torch.device("cpu")
        self.bin_edges: torch.Tensor | None = None
        self.bin_values: torch.Tensor | None = None
        self._is_fitted = False

    def fit(self, normalized_data: np.ndarray) -> None:
        """Fit bin edges and per-bin decode values.

        Args:
            normalized_data: Normalized data of shape (N, D) where:
                N = number of samples
                D = feature dimension
                Values should be normalized (typically [-1, 1]).
        """
        if normalized_data.ndim == 3:
            normalized_data = normalized_data.reshape(-1, normalized_data.shape[-1])

        n_samples, n_dims = normalized_data.shape
        if self.binning_strategy == BinningStrategy.UNIFORM.value:
            uniform_edges = np.linspace(
                self.min_value, self.max_value, self.num_bins + 1
            )[1:-1]
            bin_edges = np.tile(uniform_edges, (n_dims, 1))
        else:
            quantiles = np.linspace(0, 1, self.num_bins + 1)[1:-1]
            bin_edges = np.zeros((n_dims, len(quantiles)))
            for dim in range(n_dims):
                bin_edges[dim] = np.quantile(normalized_data[:, dim], quantiles)

        self.bin_edges = torch.tensor(
            bin_edges, dtype=torch.float32, device=self.device
        )
        self.bin_values = self._compute_bin_values(normalized_data=normalized_data)
        self._is_fitted = True

        logging.info(
            f"Fitted {self.binning_strategy} binned value discretizer with "
            f"{self.num_bins} bins on {n_samples} samples with {n_dims} dimensions"
        )

    def _compute_bin_values(self, normalized_data: np.ndarray) -> torch.Tensor:
        """Build the per-bin decode table.

        Uniform bins decode to geometric bin centers. Quantile bins decode to
        the mean of the training values inside each bin, because duplicate
        quantile edges make geometric centers wrong for repeated values (a
        binary gripper's +1 would otherwise decode to the midpoint of a bin
        spanning both modes). Empty bins fall back to geometric centers.
        """
        n_dims = self.bin_edges.shape[0]
        bin_values = torch.stack(
            [self._geometric_bin_centers(dim) for dim in range(n_dims)]
        )
        if self.binning_strategy == BinningStrategy.UNIFORM.value:
            return bin_values

        edges = self.bin_edges.cpu().numpy()
        for dim in range(n_dims):
            values = normalized_data[:, dim]
            bin_ids = np.searchsorted(edges[dim], values, side="right")
            sums = np.zeros(self.num_bins)
            counts = np.zeros(self.num_bins)
            np.add.at(sums, bin_ids, values)
            np.add.at(counts, bin_ids, 1)
            occupied = counts > 0
            means = np.zeros(self.num_bins)
            means[occupied] = sums[occupied] / counts[occupied]
            bin_values[dim, torch.from_numpy(occupied)] = torch.tensor(
                means[occupied], dtype=torch.float32, device=self.device
            )
        return bin_values

    def encode(self, normalized_data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode normalized data to discrete IDs.

        Args:
            normalized_data: Normalized data of shape (..., D).

        Returns:
            ID tensor of shape (..., D) with integer values in [0, num_bins-1].

        Raises:
            RuntimeError: If discretizer has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Discretizer must be fitted before encoding")

        if isinstance(normalized_data, np.ndarray):
            normalized_data = torch.tensor(
                normalized_data, dtype=torch.float32, device=self.device
            )
        else:
            normalized_data = normalized_data.to(self.device)

        original_shape = normalized_data.shape
        data_flat = normalized_data.reshape(-1, original_shape[-1])
        tokens = torch.zeros_like(data_flat, dtype=torch.long)
        for dim in range(original_shape[-1]):
            # right=True gives floor semantics: values exactly on a bin edge
            # fall into the higher bin, matching np.digitize.
            tokens[:, dim] = torch.searchsorted(
                self.bin_edges[dim], data_flat[:, dim].contiguous(), right=True
            )
        return tokens.reshape(original_shape)

    def decode(self, tokens: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Decode discrete IDs back to normalized data.

        Args:
            tokens: ID tensor of shape (..., D) with values in [0, num_bins-1].

        Returns:
            Reconstructed normalized data of shape (..., D).

        Raises:
            RuntimeError: If discretizer has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Discretizer must be fitted before decoding")

        if isinstance(tokens, np.ndarray):
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
        else:
            tokens = tokens.to(self.device)

        original_shape = tokens.shape
        tokens_flat = tokens.reshape(-1, original_shape[-1])
        decoded = torch.zeros(
            tokens_flat.shape, dtype=torch.float32, device=self.device
        )
        bin_values = self._get_bin_values()
        for dim in range(original_shape[-1]):
            decoded[:, dim] = bin_values[dim][tokens_flat[:, dim]]

        return decoded.reshape(original_shape)

    def _get_bin_values(self) -> torch.Tensor:
        """Return the decode table, rebuilding geometric centers if absent.

        Older serialized states predate ``bin_values``; they decode through
        geometric centers computed from the stored edges.
        """
        if self.bin_values is not None:
            return self.bin_values
        n_dims = self.bin_edges.shape[0]
        return torch.stack([self._geometric_bin_centers(dim) for dim in range(n_dims)])

    def _geometric_bin_centers(self, dim: int) -> torch.Tensor:
        """Compute geometric bin centers for a given dimension.

        Args:
            dim: Dimension index.

        Returns:
            Tensor of bin centers of shape (num_bins,).
        """
        edges = self.bin_edges[dim]
        bin_centers = torch.zeros(self.num_bins, device=self.device)

        bin_centers[0] = edges[0] - (edges[1] - edges[0]) / 2
        bin_centers[-1] = edges[-1] + (edges[-1] - edges[-2]) / 2

        for i in range(1, self.num_bins - 1):
            bin_centers[i] = (edges[i - 1] + edges[i]) / 2

        return bin_centers

    def to(self, device: torch.device) -> "BinnedValueDiscretizer":
        """Move discretizer to specified device.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self.device = device
        if self.bin_edges is not None:
            self.bin_edges = self.bin_edges.to(device)
        if self.bin_values is not None:
            self.bin_values = self.bin_values.to(device)
        return self

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization.

        Returns:
            Dictionary containing discretizer state.
        """
        return {
            "num_bins": self.num_bins,
            "binning_strategy": self.binning_strategy,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "bin_edges": self.bin_edges.cpu() if self.bin_edges is not None else None,
            "bin_values": (
                self.bin_values.cpu() if self.bin_values is not None else None
            ),
            "is_fitted": self._is_fitted,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary.

        States saved before the strategy split default to quantile binning
        with geometric-center decoding, preserving their trained behavior.

        Args:
            state_dict: State dictionary from state_dict().
        """
        self.num_bins = state_dict["num_bins"]
        self.binning_strategy = state_dict.get(
            "binning_strategy", BinningStrategy.QUANTILE.value
        )
        self.min_value = state_dict.get("min_value", self.min_value)
        self.max_value = state_dict.get("max_value", self.max_value)
        self._is_fitted = state_dict["is_fitted"]

        if state_dict["bin_edges"] is None:
            self.bin_edges = None
        else:
            self.bin_edges = state_dict["bin_edges"].to(self.device)
        bin_values = state_dict.get("bin_values")
        self.bin_values = None if bin_values is None else bin_values.to(self.device)
