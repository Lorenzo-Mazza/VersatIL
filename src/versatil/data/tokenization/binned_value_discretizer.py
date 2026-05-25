"""Quantile-based discretizer for continuous values.

This module provides the shared quantile binning used by observation and
action tokenization.
"""

import logging
from typing import Any

import numpy as np
import torch


class BinnedValueDiscretizer:
    """Quantile-based discretizer.

    Converts continuous values into discrete IDs by binning them according
    to learned quantile boundaries. Each dimension is binned independently.

    Attributes:
        num_bins: Number of discrete bins per dimension.
        device: Target device for tensors.
        bin_edges: Quantile boundaries for each dimension, shape (D, num_bins-1).
    """

    def __init__(self, num_bins: int = 256, device: torch.device | None = None):
        """Initialize discretizer.

        Args:
            num_bins: Number of discrete bins per dimension.
            device: Target device for tensors. If None, uses CPU.
        """
        self.num_bins = num_bins
        self.device = device if device is not None else torch.device("cpu")
        self.bin_edges = None
        self._is_fitted = False

    def fit(self, normalized_data: np.ndarray) -> None:
        """Fit discretizer by computing quantile bin edges.

        Args:
            normalized_data: Normalized data of shape (N, D) where:
                N = number of samples
                D = feature dimension
                Values should be normalized (typically [-1, 1]).
        """
        if normalized_data.ndim == 3:
            normalized_data = normalized_data.reshape(-1, normalized_data.shape[-1])

        n_samples, n_dims = normalized_data.shape
        quantiles = np.linspace(0, 1, self.num_bins + 1)[1:-1]

        bin_edges = np.zeros((n_dims, len(quantiles)))
        for dim in range(n_dims):
            bin_edges[dim] = np.quantile(normalized_data[:, dim], quantiles)

        self.bin_edges = torch.tensor(
            bin_edges, dtype=torch.float32, device=self.device
        )
        self._is_fitted = True

        logging.info(
            f"Fitted binned value discretizer with {self.num_bins} bins on "
            f"{n_samples} samples with {n_dims} dimensions"
        )

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
            tokens[:, dim] = torch.searchsorted(
                self.bin_edges[dim], data_flat[:, dim].contiguous(), right=False
            )
        return tokens.reshape(original_shape)

    def decode(self, tokens: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Decode discrete IDs back to normalized data.

        Decoding uses bin centers as the reconstructed values.

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
        for dim in range(original_shape[-1]):
            bin_centers = self._get_bin_centers(dim)
            decoded[:, dim] = bin_centers[tokens_flat[:, dim]]

        return decoded.reshape(original_shape)

    def _get_bin_centers(self, dim: int) -> torch.Tensor:
        """Compute bin centers for a given dimension.

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
        return self

    def state_dict(self) -> dict[str, Any]:
        """Get state dictionary for serialization.

        Returns:
            Dictionary containing discretizer state.
        """
        return {
            "num_bins": self.num_bins,
            "bin_edges": self.bin_edges.cpu() if self.bin_edges is not None else None,
            "is_fitted": self._is_fitted,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state dictionary.

        Args:
            state_dict: State dictionary from state_dict().
        """
        self.num_bins = state_dict["num_bins"]
        self._is_fitted = state_dict["is_fitted"]

        if state_dict["bin_edges"] is None:
            self.bin_edges = None
        else:
            self.bin_edges = state_dict["bin_edges"].to(self.device)
