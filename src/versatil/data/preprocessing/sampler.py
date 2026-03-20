import numba
import numpy as np

from versatil.data.preprocessing.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int = 0,
    pad_after: int = 0,
    skip_initial: int = 0,
    debug: bool = True,
) -> np.ndarray:
    """Builds a list of starting points for fixed-length sequences across multiple episodes.

    Episodes are sections of data defined by their ending positions in episode_ends (cumulative, like [5, 10, 15] for episodes of length 5 each).
    This function finds all possible starting positions where you can pull a sequence of 'sequence_length' steps, allowing some overhang (padding)
     at the start or end of an episode.
    It skips episodes marked False in episode_mask.
    For each valid start, it calculates:
    - Where the sequence actually starts and ends in the full data buffer.
    - How much padding is needed at the beginning or end if the sequence hangs over the episode edges.
    The output is an array of these details for every possible sequence.

    Example: If an episode has 10 steps, sequence_length=3, pad_before=1, pad_after=1:
    - Possible starts range from -1 (pads 1 at start) to 8 (pads 1 at end).
    - For start=-1: buffer 0 to 2, pad 1 at beginning.

    Args:
        episode_ends: Array like [5, 12, 20] where each number is the end index of an episode in the full data.
        sequence_length: How long each sequence should be (e.g., 10 steps).
        episode_mask: Boolean array same size as episode_ends; True means include this episode.
        pad_before: Max steps to allow overhanging at the start (will pad with repeats).
        pad_after: Max steps to allow overhanging at the end.
        skip_initial: Ignore the first N steps of each episode (e.g., for warm-up periods).
        debug: If True, checks that calculations make sense (like no negative pads).

    Returns:
        Array with rows [buffer_start, buffer_end, pad_start, pad_end] for each sequence.
    """
    assert episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)
    indices_list: list[list[int]] = []
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = skip_initial - pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, skip_initial) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (
                    buffer_end_idx - buffer_start_idx
                )
            indices_list.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )

    indices: np.ndarray
    if len(indices_list) == 0:
        indices = np.zeros((0, 4), dtype=np.int64)
    else:
        indices = np.array(indices_list)
    return indices


def get_val_mask(n_episodes: int, val_ratio: float, seed: int = 0) -> np.ndarray:
    """Makes a boolean array to pick which episodes are for validation.

    Randomly chooses a fraction (val_ratio) of episodes to mark as True (validation set).
    Uses a random seed for repeatability. If val_ratio=0, all False.

    Args:
        n_episodes: Total number of episodes.
        val_ratio: Fraction (0.0 to 1.0) to use for validation.
        seed: Random seed for selection.

    Returns:
        Boolean array where True means validation episode.
    """
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask
    n_val = round(n_episodes * val_ratio)
    n_val = max(0, min(n_val, n_episodes))
    if n_val == 0:
        return val_mask
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask: np.ndarray, max_n: int | None, seed: int = 0) -> np.ndarray:
    """Reduces the number of True values in a mask to at most max_n.

    If there are more than max_n Trues, randomly pick max_n of them to keep True, rest False.
    Useful for limiting training data size.

    Args:
        mask: Boolean array (e.g., training mask).
        max_n: Max number of Trues to keep (if None, no change).
        seed: Random seed.

    Returns:
        New mask with limited Trues.
    """
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask


class SequenceSampler:
    """Tool to pull out fixed-length chunks (sequences) from a big dataset of episodes.

    First, it pre-calculates all possible starting points for sequences using create_indices.
    This handles episode boundaries, padding if sequences overlap edges, and skipping episodes or initial steps.
    When you ask for a sequence by index, it slices the data, adds padding (by repeating edge values), and returns a dict of arrays (one per data key like 'observations', 'actions').
    For some keys, it can load only the first K steps to save time/memory, filling the rest with defaults (0 or nan).

    Example use: In training ML models on time-series data from robot episodes, where you want overlapping windows of 10 steps each.

    Attributes:
        indices: List of precomputed sequence starts/ends/pads.
        keys: Which data fields to include (e.g., ['obs', 'action']).
        sequence_length: Length of each chunk.
        replay_buffer: The big dataset holder.
        key_first_k: For certain keys, load only first K steps (perf hack).
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        key_first_k=None,
        episode_mask: np.ndarray | None = None,
        skip_initial: int = 0,
        pad_with_zeros: bool = True,
    ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """
        super().__init__()
        if key_first_k is None:
            key_first_k = {}
        assert sequence_length >= 1
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)
        if np.any(episode_mask):
            indices = create_indices(
                episode_ends,
                sequence_length=sequence_length,
                pad_before=pad_before,
                pad_after=pad_after,
                episode_mask=episode_mask,
                skip_initial=skip_initial,
            )
        else:
            indices = np.zeros((0, 4), dtype=np.int64)
        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices
        self.keys = list(keys)
        self.sequence_length = sequence_length
        self.episode_mask = episode_mask
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
        self.pad_with_zeros = pad_with_zeros

    def __len__(self):
        """How many sequences are available."""
        return len(self.indices)

    def sample_sequence(self, idx: int) -> dict[str, np.ndarray]:
        """Gets one sequence as a dict of arrays.

        Uses the precomputed indices to slice the data.
        If padding needed, repeats the first or last value.
        For limited keys, loads partial data and fills rest.

        Args:
            idx: Which sequence (0 to len-1).

        Returns:
            Dict like {'obs': array(seq_len, obs_dim), 'action': array(seq_len, act_dim)}.
        """

        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]
        result = {}
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                # Action data
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # Observation data
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                fill_value: str | float | int
                if input_arr.dtype == np.dtype("O") or input_arr.dtype.kind == "U":
                    # String dtypes: use empty string
                    fill_value = ""
                elif np.issubdtype(input_arr.dtype, np.floating):
                    fill_value = np.nan
                else:
                    fill_value = 0

                sample = np.full(
                    (n_data,) + input_arr.shape[1:],
                    fill_value=fill_value,
                    dtype=input_arr.dtype,
                )
                try:
                    sample[:k_data] = input_arr[
                        buffer_start_idx : buffer_start_idx + k_data
                    ]
                except Exception as e:
                    raise ValueError(
                        f"Error sampling key {key} at index {idx}: {e}"
                    ) from e
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                # If the sample does not fill the whole sequence length, we need to pad
                if self.pad_with_zeros:
                    data = np.zeros(
                        shape=(self.sequence_length,) + input_arr.shape[1:],
                        dtype=input_arr.dtype,
                    )
                    data[sample_start_idx:sample_end_idx] = sample
                else:
                    # Pad with repeated values
                    data = np.empty(
                        shape=(self.sequence_length,) + input_arr.shape[1:],
                        dtype=input_arr.dtype,
                    )
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < self.sequence_length:
                        data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result
