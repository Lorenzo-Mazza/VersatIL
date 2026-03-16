"""Tests for versatil.inference.observation_buffer module."""
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise

import pytest

from versatil.inference.observation_buffer import ObservationBuffer


@pytest.fixture
def observation_buffer_factory() -> Callable[..., ObservationBuffer]:
    def factory(
        buffer_size: int = 3,
        required_keys: list[str] | None = None,
    ) -> ObservationBuffer:
        if required_keys is None:
            required_keys = ["rgb", "depth"]
        return ObservationBuffer(
            buffer_size=buffer_size,
            required_keys=required_keys,
        )

    return factory


@pytest.mark.unit
class TestObservationBufferInitialization:

    @pytest.mark.parametrize("buffer_size", [1, 5])
    @pytest.mark.parametrize(
        "required_keys", [["rgb"], ["rgb", "depth", "proprio"]]
    )
    def test_stores_configuration(
        self, observation_buffer_factory, buffer_size, required_keys
    ):
        buffer = observation_buffer_factory(
            buffer_size=buffer_size,
            required_keys=required_keys,
        )
        assert buffer.buffer_size == buffer_size
        assert buffer.required_keys == required_keys

    @pytest.mark.parametrize(
        "buffer_size, expectation",
        [
            (1, does_not_raise()),
            (0, pytest.raises(ValueError, match=re.escape(
                "buffer_size must be >= 1, got 0"
            ))),
            (-3, pytest.raises(ValueError, match=re.escape(
                "buffer_size must be >= 1, got -3"
            ))),
        ],
    )
    def test_buffer_size_validation(
        self, observation_buffer_factory, buffer_size, expectation
    ):
        with expectation:
            observation_buffer_factory(buffer_size=buffer_size)

    def test_initial_state_is_empty(self, observation_buffer_factory):
        buffer = observation_buffer_factory(
            buffer_size=3, required_keys=["rgb", "depth"]
        )
        assert not buffer.is_ready()
        recent = buffer.get_recent()
        assert recent == {"rgb": [], "depth": []}


@pytest.mark.unit
class TestObservationBufferAdd:

    def test_add_accumulates_observations(self, observation_buffer_factory):
        buffer = observation_buffer_factory(
            buffer_size=3, required_keys=["rgb"]
        )
        buffer.add(observations={"rgb": "frame_0"})
        buffer.add(observations={"rgb": "frame_1"})
        recent = buffer.get_recent()
        assert recent["rgb"] == ["frame_0", "frame_1"]

    def test_add_raises_on_missing_required_key(
        self, observation_buffer_factory
    ):
        buffer = observation_buffer_factory(
            buffer_size=2, required_keys=["rgb", "depth"]
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Missing required observation key 'depth'. "
                "Got keys: ['rgb']"
            ),
        ):
            buffer.add(observations={"rgb": "frame_0"})

    def test_add_ignores_extra_keys(self, observation_buffer_factory):
        buffer = observation_buffer_factory(
            buffer_size=2, required_keys=["rgb"]
        )
        buffer.add(observations={"rgb": "frame_0", "extra": "ignored"})
        recent = buffer.get_recent()
        assert "extra" not in recent
        assert recent["rgb"] == ["frame_0"]

    def test_add_evicts_oldest_when_full(self, observation_buffer_factory):
        buffer = observation_buffer_factory(
            buffer_size=2, required_keys=["rgb"]
        )
        buffer.add(observations={"rgb": "frame_0"})
        buffer.add(observations={"rgb": "frame_1"})
        buffer.add(observations={"rgb": "frame_2"})
        recent = buffer.get_recent()
        assert recent["rgb"] == ["frame_1", "frame_2"]

    def test_eviction_preserves_all_required_keys(
        self, observation_buffer_factory
    ):
        buffer = observation_buffer_factory(
            buffer_size=2, required_keys=["rgb", "depth"]
        )
        buffer.add(observations={"rgb": "r0", "depth": "d0"})
        buffer.add(observations={"rgb": "r1", "depth": "d1"})
        buffer.add(observations={"rgb": "r2", "depth": "d2"})
        recent = buffer.get_recent()
        assert recent["rgb"] == ["r1", "r2"]
        assert recent["depth"] == ["d1", "d2"]


@pytest.mark.unit
class TestObservationBufferIsReady:

    def test_not_ready_until_buffer_full(self, observation_buffer_factory):
        buffer = observation_buffer_factory(
            buffer_size=3, required_keys=["rgb"]
        )
        buffer.add(observations={"rgb": "frame_0"})
        assert not buffer.is_ready()
        buffer.add(observations={"rgb": "frame_1"})
        assert not buffer.is_ready()
        buffer.add(observations={"rgb": "frame_2"})
        assert buffer.is_ready()

    def test_stays_ready_after_eviction(self, observation_buffer_factory):
        buffer = observation_buffer_factory(
            buffer_size=2, required_keys=["rgb"]
        )
        buffer.add(observations={"rgb": "frame_0"})
        buffer.add(observations={"rgb": "frame_1"})
        assert buffer.is_ready()
        buffer.add(observations={"rgb": "frame_2"})
        assert buffer.is_ready()

    def test_buffer_size_one_ready_after_single_add(
        self, observation_buffer_factory
    ):
        buffer = observation_buffer_factory(
            buffer_size=1, required_keys=["rgb"]
        )
        buffer.add(observations={"rgb": "frame_0"})
        assert buffer.is_ready()


@pytest.mark.unit
class TestObservationBufferGetRecent:

    def test_get_recent_returns_last_n_observations(
        self, observation_buffer_factory
    ):
        buffer = observation_buffer_factory(
            buffer_size=4, required_keys=["rgb"]
        )
        for index in range(4):
            buffer.add(observations={"rgb": f"frame_{index}"})
        recent = buffer.get_recent(count=2)
        assert recent["rgb"] == ["frame_2", "frame_3"]

    def test_get_recent_defaults_to_buffer_size(
        self, observation_buffer_factory
    ):
        buffer = observation_buffer_factory(
            buffer_size=3, required_keys=["rgb"]
        )
        for index in range(3):
            buffer.add(observations={"rgb": f"frame_{index}"})
        recent_default = buffer.get_recent()
        recent_explicit = buffer.get_recent(count=3)
        assert recent_default["rgb"] == recent_explicit["rgb"]

    def test_get_recent_with_count_larger_than_buffered(
        self, observation_buffer_factory
    ):
        buffer = observation_buffer_factory(
            buffer_size=5, required_keys=["rgb"]
        )
        buffer.add(observations={"rgb": "frame_0"})
        buffer.add(observations={"rgb": "frame_1"})
        # Requesting more than available returns what exists
        recent = buffer.get_recent(count=10)
        assert recent["rgb"] == ["frame_0", "frame_1"]

    def test_get_recent_with_count_zero(
        self, observation_buffer_factory
    ):
        # count=0 should return empty lists, not the entire buffer.
        # Python's list[-0:] returns the full list, so the source guards
        # against this edge case explicitly.
        buffer = observation_buffer_factory(
            buffer_size=3, required_keys=["rgb", "depth"]
        )
        buffer.add(observations={"rgb": "r0", "depth": "d0"})
        buffer.add(observations={"rgb": "r1", "depth": "d1"})
        buffer.add(observations={"rgb": "r2", "depth": "d2"})

        recent = buffer.get_recent(count=0)

        assert recent["rgb"] == []
        assert recent["depth"] == []


@pytest.mark.unit
class TestObservationBufferReset:

    def test_reset_clears_all_buffers(self, observation_buffer_factory):
        buffer = observation_buffer_factory(
            buffer_size=2, required_keys=["rgb", "depth"]
        )
        buffer.add(observations={"rgb": "r0", "depth": "d0"})
        buffer.add(observations={"rgb": "r1", "depth": "d1"})
        assert buffer.is_ready()

        buffer.reset()

        assert not buffer.is_ready()
        recent = buffer.get_recent()
        assert recent == {"rgb": [], "depth": []}

    def test_reset_allows_reuse(self, observation_buffer_factory):
        buffer = observation_buffer_factory(
            buffer_size=2, required_keys=["rgb"]
        )
        buffer.add(observations={"rgb": "old_0"})
        buffer.add(observations={"rgb": "old_1"})
        buffer.reset()
        buffer.add(observations={"rgb": "new_0"})
        buffer.add(observations={"rgb": "new_1"})
        assert buffer.is_ready()
        recent = buffer.get_recent()
        assert recent["rgb"] == ["new_0", "new_1"]
