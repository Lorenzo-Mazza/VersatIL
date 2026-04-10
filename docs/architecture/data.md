# Data Pipeline

The VersatIL data pipeline transforms raw demonstration data into normalized, batched training samples. The full flow is:

```
Raw Episodes (CSV / HDF5 / LeRobot)
  -> DatasetSchema.extract_episode()
  -> Zarr Store (.zarr)
  -> EpisodicDataset.__getitem__()
  -> SampleBuilder.build_sample()
  -> DataLoader (batching)
  -> Policy
```

## Dataset Schema (Ingestion)

A `DatasetSchema` defines the structure of a raw dataset -- what observations, actions, and cameras exist -- and how to extract episodes into a standardized Zarr store. Schemas define *storage structure*, not runtime behavior.

```python
class DatasetSchema(abc.ABC):
    def __init__(self, zarr_path: str, metadata: DatasetMetadata, dataset_type: str): ...

    @abc.abstractmethod
    def extract_episode(self, episode_source, resizer, depth_resizer) -> dict[str, np.ndarray]: ...

    def get_required_zarr_keys(self) -> list[str]: ...
```

### Supported Raw Formats

| Format | Schema Class | Source |
|--------|-------------|--------|
| CSV + image folders | `CsvDatasetSchema` | TSO Lab format |
| HDF5 | `Hdf5DatasetSchema` | LIBERO / robomimic |
| HuggingFace LeRobot | `LeRobotDatasetSchemaV30` | LeRobot datasets |

New formats are supported by subclassing `DatasetSchema` and implementing `extract_episode()`.

### Zarr Store Creation

Zarr stores are created automatically on the first training run if absent. The `get_dataloaders()` function calls `_ensure_zarr_exists()`, which dispatches to the appropriate creation function based on schema type:

```python
if isinstance(schema, Hdf5DatasetSchema):
    create_replay_buffer_from_hdf5(schema=schema)
elif isinstance(schema, CsvDatasetSchema):
    create_replay_buffer(schema=schema, datasets_paths=datasets_paths)
elif isinstance(schema, LeRobotDatasetSchemaV30):
    create_replay_buffer_from_lerobot(schema=schema)
```

!!! info "Raw Key Remapping"
    Raw data formats use their own naming conventions (e.g., LIBERO uses `observation.images.image`, original HDF5 uses `agentview_image`). During Zarr creation, raw camera keys (`RawCameraKey`) are remapped to standardized pipeline camera keys (`Cameras`) via `RAW_TO_CAMERA_MAPPING`. After Zarr creation, only pipeline keys exist -- training and inference never see raw format keys.

## Task Definition

`TaskSpace` combines an `ActionSpace`, `ObservationSpace`, and `DatasetSchema` to define what data an experiment uses at runtime. The same Zarr store can power different tasks (vision-only, state-only, vision-language) without data duplication.

### ObservationSpace

Defines which observations the policy receives as input.

```python
class ObservationSpace:
    def __init__(self, observations_metadata: dict[str, ObservationMetadata | CameraMetadata]): ...
```

Provides typed access to observation subsets:

- `cameras` -- RGB and depth camera observations
- `position_observations` -- End-effector position
- `orientation_observations` -- End-effector orientation
- `gripper_observations` -- Gripper state
- `proprioceptive_observations` -- All proprioceptive data (position, orientation, gripper)
- `custom_observations` -- Any additional observation types

The `get_required_zarr_keys()` method returns the list of Zarr keys this space requires.

### ActionSpace

Defines what actions the policy predicts and how they are computed.

```python
class ActionSpace:
    def __init__(
        self,
        actions_metadata: dict[str, ActionMetadata],
        use_gripper_class_weights: bool = False,
        denoise_actions: bool = True,
        denoising_percentile: float = 15.0,
    ): ...
```

Actions can be:

- **Precomputed** (`PrecomputedActionMetadata`) -- loaded directly from the Zarr store
- **On-the-fly** (`OnTheFlyActionMetadata`) -- computed per-sample during data loading from consecutive observations

Action types include position, orientation, and gripper, each with configurable computation methods:

| Computation Method | Description |
|-------------------|-------------|
| `NEXT_TIMESTEP` | Predict the next absolute value |
| `DELTA` | Predict the difference between consecutive timesteps |

!!! warning "Binary Gripper Constraint"
    Binary grippers only support the `NEXT_TIMESTEP` method. Using `DELTA` with a binary gripper raises a `ValueError`.

### Temporal Horizons

Configured in `TaskSpace`:

- **`observation_horizon`** -- Number of historical timesteps loaded as input (minimum 1)
- **`prediction_horizon`** -- Number of future action timesteps to predict (the chunk size)

### Validation

`TaskSpace._validate()` checks at initialization that:

- All requested observation and action keys exist in the dataset schema
- Metadata is consistent between the schema and task requirements
- Camera keys are valid members of the `Cameras` enum
- Temporal horizons are positive integers

## Data Loading Pipeline

### EpisodicDataset

`EpisodicDataset` is the core PyTorch `Dataset` that loads temporal windows from a Zarr store.

```python
class EpisodicDataset(data.Dataset):
    def __init__(
        self,
        zarr_path: str,
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        dataloader_config: DataLoaderConfig,
        pred_horizon: int,
        obs_horizon: int,
        train: bool = True,
        seed: int = 42,
    ): ...
```

Key responsibilities:

1. **Replay buffer loading** -- Loads the Zarr store via `ReplayBuffer`, optionally preloading into memory
2. **Episode splitting** -- Creates train/val splits using configurable `val_ratio` and `total_ratio`
3. **Sequence sampling** -- Uses `SequenceSampler` to extract overlapping temporal windows with padding
4. **Downsampling** -- Optionally subsamples episodes by taking every n-th step
5. **Key validation** -- Raises `KeyError` if required Zarr keys are missing

Each `__getitem__` call:

1. Samples a padded sequence from the replay buffer via `SequenceSampler`
2. Computes actions via `ActionProcessor`
3. Builds the training sample via `SampleBuilder`

### ActionProcessor

`ActionProcessor` computes actions from observations, supporting both precomputed and on-the-fly computation.

For on-the-fly actions, the processor:

- Computes position deltas or absolute values
- Computes orientation deltas using quaternion, euler, or roll representations
- Handles binary and continuous gripper actions
- Applies **denoising** to position deltas: movements below a percentile-based threshold are zeroed out

```python
class ActionProcessor:
    def compute_sample_actions(
        self,
        padded_data: dict[str, np.ndarray],
        action_slice_start: int,
        action_slice_end: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, ActionMetadata]]: ...
```

!!! note "Action Denoising"
    Denoising thresholds are computed from the training data by examining the distribution of observation delta magnitudes. Movements below the configured percentile (default 15th) are zeroed, reducing noise from sensor jitter. Thresholds are stored in the checkpoint and reapplied during inference.

### SampleBuilder

`SampleBuilder` assembles the final training sample from raw episode data.

```python
class SampleBuilder:
    def build_sample(
        self,
        padded_data: dict[str, np.ndarray],
        action_data: dict[str, np.ndarray],
        action_meta: dict[str, ActionMetadata],
        start_idx: int,
        sampler_indices: np.ndarray,
    ) -> dict[str, dict[str, torch.Tensor]]: ...
```

The output sample is a nested dict with two top-level keys:

```python
{
    "observation": {
        "left": Tensor,         # (T_obs, C, H, W)
        "right": Tensor,        # (T_obs, C, H, W)
        "depth": Tensor,        # (T_obs, 1, H, W)
        "proprio_robot_frame": Tensor,  # (T_obs, D)
        ...
    },
    "action": {
        "proprio_robot_frame": Tensor,  # (T_pred, D)
        "gripper_state_obs": Tensor,    # (T_pred, 1)
        "is_pad": Tensor,               # (T_pred,) boolean mask
        ...
    },
}
```

Processing steps within `build_sample`:

1. **Image processing** -- Slices the observation window, applies augmentation, converts to `(T, C, H, W)` tensors
2. **Proprioceptive slicing** -- Extracts and casts observation data to appropriate dtypes
3. **Action slicing** -- Extracts the prediction horizon window from computed actions
4. **Padding mask** -- Computes `is_pad` mask indicating which action timesteps are padded beyond episode boundaries
5. **Normalization and tokenization** -- Applied as the final step if a normalizer/tokenizer is set

### get_dataloaders

The `get_dataloaders()` factory function orchestrates the full pipeline:

```python
def get_dataloaders(config: DictConfig) -> tuple[
    DataLoader, DataLoader | None, LinearNormalizer, Tokenizer | None, float | None
]: ...
```

Steps:

1. Validates the `DataLoaderConfig`
2. Ensures the Zarr store exists (creates it from raw data if needed)
3. Creates `EpisodicDataset` for training
4. Fits a `LinearNormalizer` and optional `Tokenizer` on training data
5. Sets normalizer and tokenizer on the training dataset
6. Creates the training `DataLoader` with pinned memory and persistent workers
7. Optionally creates a validation dataset and `DataLoader` (skipped when `val_ratio=0`)
8. Computes gripper class imbalance weights if configured

## Normalization

`LinearNormalizer` stores per-key normalization parameters and supports three modes:

| Mode | Description |
|------|-------------|
| `min_max` | Scales data to `[output_min, output_max]` (default `[-1, 1]`) |
| `gaussian` | Z-score normalization (zero mean, unit variance) |
| `demean` | Subtracts mean only |

The `TransformBuilder` fits separate normalizers for:

- **Kinematics** (proprioceptive data and actions) -- configurable mode, optional winsorization
- **RGB images** -- configurable via `image_norm_type` (scale to `[0, 1]`, `[-1, 1]`, or ImageNet normalization)
- **Depth images** -- fitted from depth statistics, with optional winsorization to clip outlier depth values

!!! info "Normalization Exclusions"
    Binary gripper actions and language instructions are not normalized. The normalizer is stored in the checkpoint and loaded at inference time.

## Image Processing

[`ImageProcessor`][versatil.data.processing.image_processor.ImageProcessor] handles per-camera image processing: resize, augmentation, normalization, and channel reorder. 

```python
ImageProcessor(
    color_augmentation=A.Compose(...) | None,
    spatial_augmentation=A.Compose(...) | None,
    camera_metadata={"left": CameraMetadata(...), ...},
    train=True,
)
```

**Training pipeline:** resize -> color augmentation (RGB only) -> spatial augmentation -> normalize -> channel reorder.

**Inference pipeline:** resize -> normalize -> channel reorder.

Two independent [Albumentations](https://albumentations.ai/) pipelines are supported:

- **Color augmentation** -- Photometric transforms (brightness, contrast, hue, etc.)
- **Spatial augmentation** -- Geometric transforms (rotation, flips, etc.)

Image sizes come from per-camera `CameraMetadata` in the observation space. Depth images use nearest-neighbor interpolation to preserve depth values; RGB images use bilinear interpolation.

## Tokenization

The `Tokenizer` wraps two optional sub-tokenizers:

- **`ObservationTokenizer`** -- Tokenizes language instructions and proprioceptive data
- **`ActionTokenizer`** -- Tokenizes continuous actions into discrete tokens (FAST-style)

Tokenization is applied after normalization in `SampleBuilder.normalize_and_tokenize_sample()`. The tokenizer is saved alongside the checkpoint and loaded during inference via `PolicyLoader`.
