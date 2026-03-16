"""Dict-keyed observation history buffer for inference clients."""

from typing import Any


class ObservationBuffer:
    """Sliding window of observations keyed by observation name.

    Accumulates observations until the required horizon is reached,
    then evicts the oldest on each new addition.
    """

    def __init__(self, buffer_size: int, required_keys: list[str]):
        """Initialize observation buffer.

        Args:
            buffer_size: Number of observations to buffer (observation horizon).
            required_keys: Observation keys that must be present in each add().
        """
        if buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
        self.buffer_size = buffer_size
        self.required_keys = required_keys
        self._buffers: dict[str, list[Any]] = {key: [] for key in required_keys}
        self._count = 0

    def add(self, observations: dict[str, Any]) -> None:
        """Add a set of observations to the buffer.

        Args:
            observations: Dict mapping observation key to value.
                Must contain all required keys.
        """
        for key in self.required_keys:
            if key not in observations:
                raise ValueError(
                    f"Missing required observation key '{key}'. "
                    f"Got keys: {list(observations.keys())}"
                )
            self._buffers[key].append(observations[key])
            if len(self._buffers[key]) > self.buffer_size:
                self._buffers[key].pop(0)
        self._count = min(self._count + 1, self.buffer_size)

    def is_ready(self) -> bool:
        """Check if enough observations are buffered for inference."""
        return self._count >= self.buffer_size

    def get_recent(self, count: int | None = None) -> dict[str, list[Any]]:
        """Get the most recent observations from the buffer.

        Args:
            count: Number of recent observations. Defaults to buffer_size.

        Returns:
            Dict mapping observation key to list of recent values.
        """
        if count is None:
            count = self.buffer_size
        if count == 0:
            return {key: [] for key in self._buffers}
        return {key: buffer[-count:] for key, buffer in self._buffers.items()}

    def reset(self) -> None:
        """Clear all buffered observations."""
        for key in self._buffers:
            self._buffers[key].clear()
        self._count = 0