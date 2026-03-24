"""Calibration data provider for post-training static quantization."""

from collections.abc import Iterator

import torch

from versatil.data.constants import SampleKey


class CalibrationDataProvider:
    """Yields observation tuples from a VersatIL dataloader for calibration.

    Extracts observations from dataloader batches and yields them as
    positional tensor tuples matching the ExportablePolicy's key order.

    The VersatIL dataloader already normalizes and tokenizes observations,
    so no additional preprocessing is needed here.
    """

    def __init__(
        self,
        dataloader: torch.utils.data.DataLoader,
        observation_keys: list[str],
        num_calibration_steps: int = 128,
        device: torch.device | None = None,
    ) -> None:
        """Initialize calibration data provider.

        Args:
            dataloader: VersatIL dataloader yielding normalized, tokenized samples.
            observation_keys: Keys defining positional argument order.
            num_calibration_steps: Maximum number of calibration batches.
            device: Device for calibration tensors. Defaults to CPU.
        """
        self._dataloader = dataloader
        self._observation_keys = observation_keys
        self._num_calibration_steps = num_calibration_steps
        if device is None:
            device = torch.device("cpu")
        self.device = device

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """Yield calibration batches as positional tensor tuples on self.device."""
        for step, batch in enumerate(self._dataloader):
            if step >= self._num_calibration_steps:
                break
            observation = batch[SampleKey.OBSERVATION.value]
            yield tuple(
                observation[key].to(self.device) for key in self._observation_keys
            )

    def get_single_batch(self) -> tuple[torch.Tensor, ...]:
        """Return the first calibration batch for torch.export example inputs.

        Returns:
            First calibration batch as a tuple of tensors.
        """
        return next(iter(self))
