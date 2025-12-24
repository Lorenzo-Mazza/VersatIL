# Surg-IL: Imitation Learning Framework for Robotic Surgery

A modular, composable framework for training vision-based imitation learning policies on surgical manipulation tasks. Built with PyTorch Lightning and Hydra for reproducible research.

## 🎯 Overview

Surg-IL provides a flexible architecture where policies are composed from modular components:

```
Policy = Encoding Pipeline + Algorithm + Decoder + Loss
```

- **EncodingPipeline**: Multi-modal observation encoding (RGB, depth, proprioception, language) with hierarchical fusion that turns raw data into features.
- **Algorithm**: Learning paradigm (Behavioral Cloning, Diffusion, Flow Matching, Variational) that specifies how to train a policy.
- **Decoder**: Neural architecture (Diffusion Transformer, DETR, GPT, UNet) that is used to decode features into robot actions.
- **Loss**: Composable loss modules (MSE, Cross-Entropy, KL, etc.).

The Surg-IL library enables rapid experimentation with different combinations of components without code duplication.

---

## 🚀 Quick Start

### Installation

**Prerequisites:**
- Python 3.11+
- CUDA 12.4+ (required for training)
- Git credentials for private repositories

**Setup:**
###### Install Conda/Mamba from miniforge
Follow the instructions here https://github.com/conda-forge/miniforge

```bash
# 1. Clone repository
git clone https://gitlab.com/nct_tso_public/surg-il.git
cd surg-il

# 2. Configure git credentials
git config --global credential.helper store
# 3. Create environment (use Mamba for faster installation)
mamba env create -f environment.yml
mamba activate surg-il

# 4. Install dependencies with uv
UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX uv sync
```

NB: If you are installing from the cluster, make sure to do the installation from inside an interactive job with 1 GPU and 1 CPU available.
This is needed to install flash-attn properly, because it needs the paths to CUDA libraries.
To request such a job, run `srun --gres=gpu:1 --cpus-per-task=1 --pty bash` on g27vmsteffi.
```bash
srun --gres=gpu:1 --cpus-per-task=1 --pty bash
# Then run installation commands above
```

### Training Your First Model

**1. Default Training:**
```bash
# Train Action Chunking Transformer on bowel retraction
python -m src.refactoring.endpoints.train --config-name act_bowel_retraction

```

**2. Override Configuration:**
```bash
# Change batch size
python -m src.refactoring.endpoints.train \
    --config-name act_bowel_retraction \
    task.dataloader.batch_size=64

# Disable EMA
python -m src.refactoring.endpoints.train \
    --config-name act_bowel_retraction \
    training.use_ema=false

# Change learning rate
python -m src.refactoring.endpoints.train \
    --config-name act_bowel_retraction \
    training.optimizer.lr=1e-4
```

**3. Distributed Training (SLURM):**
Not supported yet

---

## 🏗️ Architecture

### Policy Composition

The `Policy` class orchestrates three stages:

```python
# 1. Encode observations
features = encoding_pipeline(observations)  # Multi-modal → unified representation

# 2. Decode actions
predictions = algorithm.forward(
    decoder(features),  # Algorithm uses decoder architecture
    actions=ground_truth  # During training
)

# 3. Compute loss
loss = loss_module(predictions, targets)
```

### Feature Naming Contract ⚠️

Feature names follow a strict naming convention using the `EncoderOutputKeys` enum.

**Rule**: `feature_name = "{encoder_name}_{EncoderOutputKeys.value}"`

**EncoderOutputKeys enum** (`src/refactoring/models/encoding/encoders/constants.py`):
```python
class EncoderOutputKeys(str, enum.Enum):
    RGB = "image"            # For RGB features
    RGBD = "rgbd"           # For RGB-D features
    LANGUAGE = "language"   # For language features
    DEPTH = "depth"         # For depth features
    PROPRIOCEPTIVE = "proprio"  # For proprioceptive features
    DEFAULT = "default"     # For generic features
```

**Concrete examples**:
```python
# Encoder config name (from YAML) + EncoderOutputKeys.value = feature name
"left_rgb"      + "image"    = "left_rgb_image"      # RGB encoder
"right_rgb"     + "image"    = "right_rgb_image"     # RGB encoder
"depth_encoder" + "depth"    = "depth_encoder_depth" # Depth encoder
"proprio"       + "proprio"  = "proprio_proprio"     # Proprioceptive encoder
"tokenizer"     + "language" = "tokenizer_language"  # Language/tokenizer encoder
```

**Fusion outputs** do NOT use this convention (they specify `output_name` directly):
```python
# Fusion example
fusion = AttentionFusion(
    input_features=["left_rgb_image", "right_rgb_image"],  # Use encoder feature names
    output_name="fused_visual"  # Direct name (no prefix)
)
```

**Complete pipeline example**:
```python
# 1. Encoders produce: {encoder_name}_{EncoderOutputKeys.value}
encoders = {
    "left_rgb": CNNEncoder(...)    → produces "left_rgb_image"
    "robot_state": ProprioEncoder(...) → produces "robot_state_proprio"
}

# 2. Fusion consumes encoder outputs, produces new feature
fusion = ConcatFusion(
    input_features=["left_rgb_image", "robot_state_proprio"],
    output_name="fused_features"  # New feature, not prefixed
)
# Available features after fusion: {"fused_features"}
# Note: left_rgb_image and proprio_proprio are consumed!

# 3. Decoder requests features by exact name
decoder_input = DecoderInput(
    keys=["fused_features"]  # Must match exactly
)
```

### Feature Types & Validation

Features are classified by tensor shape for validation:

| FeatureType | Shape | Example | Generated By |
|-------------|-------|---------|--------------|
| **FLAT** | `(D,)` or `int` | `(512,)` | Pooled CNN, MLP, fused features |
| **SPATIAL** | `(C, H, W)` | `(256, 16, 16)` | CNN feature maps (before pooling) |
| **SEQUENTIAL** | `(T, D)` | `(10, 512)` | Temporal sequences, transformer outputs |

**Validation** catches errors at instantiation:
```python
# Decoder can require specific feature types
decoder_input = DecoderInput(
    keys=["fused_visual"],
    required_types=[FeatureType.FLAT.value]  # Must have ≥1 FLAT feature
)

# Policy validation will raise:
# ValueError: Decoder requires FLAT features but only SPATIAL provided
# (Caught when Policy() is instantiated, NOT during training!)
```

### Feature Consumption Pattern

Fusion modules consume input features to prevent duplication:

```python
# Encoders produce: A, B, C, D
# Fusion 1: B + C → E  (consumes B and C)
# Fusion 2: E + D → F  (consumes E and D)
# Final output: {A, F}  ← Only these go to decoder
```

---

## 🧩 Available Components

### Algorithms

Located in `src/refactoring/models/decoding/algorithm/`:

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **BehavioralCloning** | Direct supervised learning | 
| **ActionDiffusion** | DDPM-based action generation | 
| **FlowMatching** | Continuous normalizing flows | 
| **VariationalAlgorithm** | VAE wrapper for any algorithm | Adds latent variables to any base algorithm |

**Variational Pattern** :
```python
# Wrap any algorithm with variational inference
VariationalAlgorithm(
    base_algorithm=BehavioralCloning(),
    posterior_encoder=VAETransformerEncoder(...),  # q(z|a,s)
    prior=DiffusionPrior(...)  # p(z|s) - can be Gaussian or learned
)
```

### Decoders

Located in `src/refactoring/models/decoding/decoders/factory/`:

| Decoder               |  Architecture                | Best For                                           |
|-----------------------|------------------------------|----------------------------------------------------|
| **ActionTransformer** | Vanilla transformer          | General-purpose baseline                           |
| **ACT**               | Action Chunking Transformer  | Long-horizon tasks (from ACT paper)                |
| **PhaseACT**          | ACT + phase prediction       | Multi-phase surgical procedures                    |
| **FASTGPTDecoder**    | Autoregressive GPT           | Discrete action tokenization (FAST)                |
| **FASTDETRDecoder**   | DETR with FAST               | (Experimental)
| **DPTransformer**     | Diffusion Policy transformer | From Diffusion Policy paper (not implemented yet ) |
| **ConditionalUNet**   | UNet with FiLM               | From Diffusion Policy paper (not implemented yet ) |
| **DiTTransformer**    | Diffusion Transformer        | From DiT Policy paper (not implemented yet )       |

### Encoders

Located in `src/refactoring/models/encoding/encoders/`:

**RGB Encoders** (`rgb/`):
- `CNNEncoder`: ResNet/EfficientNet backbones
- `ViTEncoder`: Vision Transformers
- `ConditionalCNN`: FiLM-conditioned CNN

**Depth Encoders** (`depth/`):
- `DepthCNN`: Depth-specific CNN

**Proprioceptive Encoders** (`proprioceptive/`):
- `ProprioceptiveEncoder`: MLP for robot state

**Language Encoders** (`language/`):
- `LanguageEncoder`: CLIP/T5 text encoders
- `TokenizerEncoder`: For discrete tokenization and embedding (no full encoding)

#### Encoder Input Keys ⚠️

Encoder `input_keys` must use appropriate constants from `src/refactoring/data/constants.py`:

| Encoder Type | Constants to Use | Example Values |
|--------------|------------------|----------------|
| **RGB/Depth** | `Cameras` enum | `Cameras.LEFT.value` → `"left"`<br>`Cameras.RIGHT.value` → `"right"`<br>`Cameras.DEPTH.value` → `"depth"` |
| **Proprioceptive** | Observation key constants | `PROPRIO_OBS_ROBOT_FRAME_KEY` → `"proprio_robot_frame"`<br>`PROPRIO_OBS_CAMERA_FRAME_KEY` → `"proprio_camera_frame"`<br>`GRIPPER_STATE_OBS_KEY` → `"gripper_state_obs"` |
| **Language** | Language key constant | `LANGUAGE_KEY` → `"language_instruction"` |
| **Custom** | Custom observation keys | Any string matching your Zarr dataset keys |

**Note on custom observation keys**:
- Custom keys are defined in your **DatasetSchema** when creating the Zarr dataset
- They must match keys in your **ObservationSpace** configuration
- Use string literals directly in encoder config: `input_keys: ["robot_ee_force"]`

**Example YAML configs:**
```yaml
# RGB encoder (using Hydra resolver)
left_rgb:
  _target_: refactoring.models.encoding.encoders.rgb.cnn.CNNEncoder
  input_keys: ${cameras:LEFT}  # Resolves to "left"

# Proprioceptive encoder
proprio:
  _target_: refactoring.models.encoding.encoders.proprioceptive.ProprioceptiveEncoder
  input_keys:
    - ${obs_key:PROPRIO_CAMERA_FRAME}  # Resolves to "proprio_camera_frame"
```

These keys match the observation dictionary keys from the dataset and ensure type safety across the pipeline.

### Fusion Modules

Located in `src/refactoring/models/encoding/fusion/`:

| Fusion | Method |
|--------|--------|
| **ConcatFusion** | Concatenate + MLP | 
| **MLPFusion** | Learned projection | 
| **AttentionFusion** | Cross-attention |
| **SequentialFusion** | Chain multiple fusions | 

---

## 📁 Project Structure

```
src/refactoring/
├── configs/                 # Hydra configuration dataclasses
│   ├── main.py             # MainConfig (composes all configs)
│   ├── experiment.py       # Experiment tracking, WandB, checkpointing
│   ├── training.py         # Optimizer, LR schedule, EMA, gradient clipping
│   ├── policy.py           # Policy composition config
│   ├── task/               # Task definitions
│   │   ├── task.py         # ActionSpace, ObservationSpace, TaskConfig
│   │   ├── dataloader.py   # DataLoader settings
│   │   └── dataset/        # Dataset schema definitions
│   ├── encoding/           # Encoder and fusion configs
│   ├── decoding/           # Decoder and algorithm configs
│   └── loss.py             # Loss module configs
│
├── models/                  # Neural network implementations
│   ├── policy.py           # Policy orchestrates encoding → decoding → loss
│   ├── encoding/
│   │   ├── pipeline.py     # EncodingPipeline: multi-encoder + fusion
│   │   ├── encoders/       # RGB, depth, proprio, language encoders
│   │   └── fusion/         # Fusion modules (attention, MLP, concat)
│   ├── decoding/
│   │   ├── algorithm/      # Training algorithms (BC, diffusion, flow matching)
│   │   ├── decoders/       # Architecture implementations
│   │   └── latent/         # VAE components (priors, posteriors)
│   └── layers/             # Reusable torch components (transformers, attention, etc.)
│
├── data/                    # Data loading and preprocessing
│   ├── episodic_dataset.py # Loads temporal windows from Zarr
│   ├── dataloader.py       # DataLoader factory
│   ├── sample_builder.py   # Constructs training samples
│   ├── action_processor.py # Computes actions from proprioceptive data
│   ├── normalize/          # Data normalization
│   ├── tokenize/           # Action tokenization (FAST)
│   └── preprocessing/      # Zarr dataset creation
│
├── training/                # Training infrastructure
│   ├── lightning_policy.py # PyTorch Lightning wrapper
│   ├── callbacks.py        # EMA, gradient norms, confusion matrices
│   └── workspace.py        # Training orchestration
│
├── metrics/                 # Loss functions and metrics
│   ├── base.py             # BaseLoss interface
│   ├── composite.py        # Composite loss (multiple losses)
│   └── accumulators.py     # Metric accumulation across batches
│
└── endpoints/               # Training/inference entry points
    ├── train.py            # Main training script (Hydra-based)
    ├── test.py             # Evaluation script
    └── explain.py          # Model interpretation
```

---

## ⚙️ Configuration System

### Hydra Composition

Configs use Hydra's composition pattern:

```yaml
# experiments/act_bowel_retraction.yaml
defaults:
  - experiment: bowel_retraction_fast_decoder
  - task/dataset_schema: bowel_retraction_v2
  - task/dataloader: act_default
  - task/action_space: br_position_gripper_deltas_cf
  - task/observation_space: br_rgb_proprio_cf
  - training: act_default
  - policy/encoding_pipeline: rgb_language_proprio
  - policy/decoder: act
  - policy/algorithm: behavioral_cloning
  - policy/loss: act_default
  - _self_

# Override specific values
task:
  observation_horizon: 1
  prediction_horizon: 10
```

### Key Config Groups

| Config Group | Location | Purpose |
|--------------|----------|---------|
| `experiment` | `experiments/experiment/` | Name, checkpointing, WandB |
| `task/dataset_schema` | `experiments/task/dataset/` | Dataset schema (observation/action keys) |
| `task/dataloader` | `experiments/task/dataloader/` | Batch size, workers, augmentation |
| `task/action_space` | `experiments/task/action_space/` | Action dimensions, gripper, orientation |
| `task/observation_space` | `experiments/task/observation_space/` | Cameras, proprio, language |
| `training` | `experiments/training/` | Optimizer, LR, EMA, gradient clipping |
| `policy/encoding_pipeline` | `experiments/policy/encoding_pipeline/` | Encoder + fusion composition |
| `policy/decoder` | `experiments/policy/decoder/` | Decoder architecture |
| `policy/algorithm` | `experiments/policy/algorithm/` | BC, diffusion, flow matching, variational |
| `policy/loss` | `experiments/policy/loss/` | Loss function configuration |

### Interpolation

Reference other config values using `${}`:

```yaml
policy:
  prediction_horizon: ${task.prediction_horizon}
  observation_space: ${task.observation_space}
  device: ${experiment.device}
```

---

## 📦 Data Pipeline

### Dataset Schema vs Task Config

**Key Distinction**: Schemas define what's **in the dataset**, Tasks define what's **used at runtime**.

| Concept | Purpose | Example                                                                     |
|---------|---------|-----------------------------------------------------------------------------|
| **DatasetSchema** | Defines raw data structure | "Dataset has RGB images, depth, 3D position, gripper state"                 |
| **TaskConfig** | Defines what to use for training | "Use left RGB only, predict as actions the position deltas in camera frame" |

**Why separate?** A single dataset can support multiple tasks:
- Task A: RGB-only, predict absolute positions
- Task B: RGB+depth, predict position deltas
- Task C: RGB+proprio, multi-phase prediction

### Data Flow

```
Raw Episodes (CSV + images)
  ↓
[DatasetSchema] ← Defines how to read kinematics textual data, locate images/depth maps
  ↓
Zarr Dataset (.zarr file) ← Compressed, chunked storage. This is supposed to be created only ONCE and contain all relevant keys
  ↓
[TaskConfig] ← Defines what observations/actions to use
  ↓
EpisodicDataset ← Loads temporal windows from Zarr
  ↓
SampleBuilder ← Constructs training samples, i.e. chunks (obs, actions, masks)
  ↓
DataLoader ← Batching, normalization, augmentation
  ↓
Policy ← Training
```

### Dataset Schema

**Location**: `src/refactoring/data/schemas/`

**Purpose**: Defines the structure of **raw data** for a specific dataset.

```python
class DatasetSchema(abc.ABC):
    """Defines how to:
    1. Extract observations from CSV (position, gripper, phases)
    2. Locate image/depth files
    3. Create Zarr arrays with correct shapes/dtypes
    """

    def __init__(
        self,
        dataset_folders: list[str],        # Where raw data lives
        zarr_path: str,                    # Where to save Zarr
        raw_observations: DatasetMetadataConfig,  # CSV column mappings
        image_path_config: ImagePathConfig,       # Image path patterns
        has_phase_labels: bool = False
    ):
        ...
```

**Example**: Bowel Retraction Schema (`src/refactoring/data/schemas/bowel_retraction.py`):

```python
class BowelRetractionSchema(DatasetSchema):
    """Schema for bowel retraction dataset.

    CSV Structure:
    - Columns: frameLeftPath, frameRightPath,
               relative_tip_position_x/y/z,
               camera_frame_tip_position_x/y/z,
               open (gripper), task_phase

    File Structure:
    - episodes/
      ├── episode_001/
      │   ├── framesLeft/      # RGB images
      │   ├── framesRight/
      │   ├── depth/           # Depth maps (.npy)
      │   └── data.csv         # Timestamped observations
      └── episode_002/...
    """

    def __init__(self, dataset_folders, zarr_path, ...):
        raw_obs_config = DatasetMetadataConfig(
            robot_frame_proprio_keys=["relative_tip_position_x", "relative_tip_position_y", "relative_tip_position_z"],
            camera_frame_proprio_keys=["camera_frame_tip_position_x", "camera_frame_tip_position_y", "camera_frame_tip_position_z"],
            gripper_state_keys=["open"],
            camera_keys=["left", "right", "depth"],
            use_rectified_images=True,
            image_width=480,
            image_height=270
        )
        super().__init__(dataset_folders, zarr_path, raw_obs_config, ...)

    def get_image_path_column(self, camera: str) -> str:
        """Map camera name to CSV column."""
        return {
            "left": "frameLeftRectifiedPath",
            "right": "frameRightRectifiedPath"
        }[camera]

    def compute_depth_path(self, base_image_path: str) -> str:
        """Convert image path to depth path."""
        # framesLeftRectified/frame_0001.png → depth/depth_0001.npy
        return re.sub(r'framesLeftRectified/frame_(\d+).png',
                      r'depth/depth_\1.npy', base_image_path)
```

### Creating a Zarr Dataset

**Step 1**: Define your schema (or use existing):
```bash
# experiments/task/dataset/my_dataset.yaml
_target_: refactoring.data.schemas.my_dataset.MyDatasetSchema
dataset_folders:
  - /data/my_dataset/train
  - /data/my_dataset/val
zarr_path: /data/my_dataset/dataset.zarr
has_phase_labels: false
```

**Step 2**: Create Zarr from raw CSV+images:
```python
from hydra.utils import instantiate
from omegaconf import OmegaConf
from refactoring.data.preprocessing.replay_buffer import ReplayBuffer

# Load schema config
cfg = OmegaConf.load("experiments/task/dataset/my_dataset.yaml")
schema = instantiate(cfg)

# Create Zarr dataset
replay_buffer = ReplayBuffer(schema)
replay_buffer.create_zarr()  # Processes all episodes → .zarr file
```

**Output Zarr structure**:
```
dataset.zarr/
├── left/                    # RGB images (T, H, W, 3) uint8
├── right/
├── depth/                   # Depth maps (T, H, W) float32
├── proprio_robot_frame/     # (T, 3) float32
├── proprio_camera_frame/    # (T, 3) float32
├── gripper_state/           # (T, 1) float32
├── episode_ends/            # (num_episodes,) - cumulative timestep counts
├── language/                # (T,) str - per-timestep instructions
└── [additional keys/           # (T, ?) float32]

```

### Task Configuration

**Location**: `experiments/task/`

**Purpose**: Defines what data to **use at runtime** from the Zarr dataset.

**Components**:

1. **ActionSpace** (`action_space/`): What actions to predict
```yaml
# experiments/task/action_space/position_gripper_deltas_cf.yaml
has_position: true
position_dim: 3
has_orientation: false
has_gripper: true
gripper_type: binary
predict_in_camera_frame: true  # Use camera frame proprio
deltas_as_actions: true        # Predict deltas, not absolute
```

2. **ObservationSpace** (`observation_space/`): What observations to use
```yaml
# experiments/task/observation_space/rgb_proprio_cf.yaml
camera_keys:
  - left   # ← Must match schema's camera_keys
  - right
use_proprioceptive_data: true
use_proprio_camera_frame: true 
use_language: false
```

**Validation**: At dataset loading, checks that `ObservationSpace` and `ActionSpace` only request keys that exist in the schema:

```python
# In EpisodicDataset.__init__()
required_zarr_keys = (
    observation_space.get_required_zarr_keys() +  # e.g., ["left", "right", "proprio_camera_frame"]
    action_space.get_required_zarr_keys()         # e.g., ["proprio_camera_frame", "gripper_state"]
)

schema_keys = schema.get_required_zarr_keys()  # What's actually in Zarr

missing = set(required_zarr_keys) - set(schema_keys)
if missing:
    raise ValueError(f"TaskSpace requests {missing} but schema doesn't provide them")
```

### Adding a New Dataset Schema

**Step 1**: Define CSV structure constants
```python
# src/refactoring/data/schemas/my_dataset.py
MY_DATASET_ROBOT_FRAME_COLS = ["ee_pos_x", "ee_pos_y", "ee_pos_z", "ee_roll"]
MY_DATASET_GRIPPER_COL = "gripper_width"
MY_DATASET_LEFT_IMAGE_KEY = "left_camera_path"
MY_DATASET_LANGUAGE_KEY = "instruction"
```

**Step 2**: Implement schema class
```python
class MyDatasetSchema(DatasetSchema):
    def __init__(self, dataset_folders, zarr_path, ...):
        raw_obs_config = DatasetMetadataConfig(
            robot_frame_proprio_keys=MY_DATASET_ROBOT_FRAME_COLS,
            camera_frame_proprio_keys=None,  # This dataset has no camera frame
            gripper_state_keys=[MY_DATASET_GRIPPER_COL],
            camera_keys=["left", "depth"],
            language_key=MY_DATASET_LANGUAGE_KEY,
            image_width=640,
            image_height=480,
            has_orientation=True,
            orientation_dim=1  # Roll only
        )

        image_path_config = ImagePathConfig(
            left_image_key=MY_DATASET_LEFT_IMAGE_KEY,
            depth_dir_pattern="depth_maps",
            depth_file_pattern=r'depth_\1.png'
        )

        super().__init__(dataset_folders, zarr_path, raw_obs_config, image_path_config, ...)

    def get_image_path_column(self, camera: str) -> str:
        if camera == "left":
            return MY_DATASET_LEFT_IMAGE_KEY
        # ...

    def compute_depth_path(self, base_image_path: str) -> str:
        # /path/to/rgb/frame_0001.png → /path/to/depth_maps/depth_0001.png
        return re.sub(r'/rgb/frame_(\d+).png',
                      r'/depth_maps/depth_\1.png', base_image_path)
```

**Step 3**: Create Hydra config
```yaml
# experiments/task/dataset/my_dataset.yaml
_target_: refactoring.data.schemas.my_dataset.MyDatasetSchema
dataset_folders:
  - /data/my_dataset/train
zarr_path: /data/my_dataset/train.zarr
has_phase_labels: false
```

**Step 4**: Use in task config
```yaml
# experiments/my_task.yaml
defaults:
  - task/dataset_schema: my_dataset  # ← References my_dataset.yaml
  - task/action_space: position_orientation_gripper_rf  # Robot frame
  - task/observation_space: rgb_depth_proprio_rf
  # ...
---

## 🎓 Common Patterns

### Adding a New Encoder

**1. Define config** (`src/refactoring/configs/encoding/encoder.py`):
```python
@dataclass
class MyEncoderConfig(EncoderConfig):
    _target_: str = "refactoring.models.encoding.encoders.my_encoder.MyEncoder"
    feature_dim: int = 256
```

**2. Implement encoder** (`src/refactoring/models/encoding/encoders/my_encoder.py`):
```python
from refactoring.models.encoding.encoders.base import Encoder, EncoderOutput

class MyEncoder(Encoder):
    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=["embedding"],
            dimensions={"embedding": self.feature_dim}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"embedding": self.encode(x)}
```

**3. Create YAML config** (`experiments/policy/encoding/my_encoder.yaml`):
```yaml
_target_: refactoring.models.encoding.encoders.my_encoder.MyEncoder
feature_dim: 512
```

### Adding a New Algorithm

**1. Inherit from `DecodingAlgorithm`** (`src/refactoring/models/decoding/algorithm/base.py`):
```python
class MyAlgorithm(DecodingAlgorithm):
    def forward(self, decoder_output, actions, **kwargs) -> dict:
        """Training forward pass with ground truth actions."""
        ...
        return {"my_loss": loss}

    def predict(self, decoder_output, **kwargs) -> dict:
        """Inference pass (no actions provided)."""
        ...
        return {"action": predicted_actions}
```

**2. Create config and YAML** following encoder pattern above.

### Composing a New Policy

Create a new experiment YAML that composes existing components:

```yaml
# experiments/my_experiment.yaml
defaults:
  - experiment: bowel_retraction_fast_decoder
  - task/dataset_schema: bowel_retraction_v2
  - task/dataloader: act_default
  - task/action_space: br_position_gripper_deltas_cf
  - task/observation_space: br_rgb_proprio_cf
  - training: act_default
  - policy/encoding_pipeline: rgb_language_proprio   # Reuse existing
  - policy/decoder: act                              # Reuse existing
  - policy/algorithm: flow_matching                  # Try different algorithm
  - policy/loss: act_default                         # Reuse existing
  - _self_

# Override specific parameters
training:
  optimizer:
    lr: 5e-5
task:
  prediction_horizon: 20
```

---

## 📊 Monitoring & Logging

### WandB Integration

Set environment variable:
```bash
export WANDB_API_KEY=your_key_here
```
Or add it to your `.bashrc` profile, for persistent settings.

Logged metrics:
- **Per epoch**: `train_loss`, `val_loss`, `train/<metric>`, `val/<metric>`
- **Per step** (every 50 steps): `grad_norm`, `grad_norm_group_<idx>`
- **EMA**: `ema_decay` (every 100 steps)
- **Phase models**: Confusion matrices (every validation epoch)

### Checkpointing

**Best models** (based on `val_loss`):
```
checkpoints/experiment_name/best-epoch=XX-val_loss=Y.YYYY.ckpt
```

**Latest checkpoints** (every N epochs):
```
checkpoints/experiment_name/latest-epoch=XX.ckpt
checkpoints/experiment_name/last.ckpt
```

**Resume training:**
```bash
python -m src.refactoring.endpoints.train \
    --config-name act_bowel_retraction \
    experiment.resume_from=/path/to/checkpoint.ckpt
```

---

## 🧪 Testing

```bash
# Run unit tests only (default)
pytest

# Run all tests including integration tests
pytest -m ""

# Run specific test file
pytest tests/models/test_policy.py

# Run tests by marker
pytest -m "unit"           # Fast unit tests with mocked dependencies
pytest -m "integration"    # Slower tests with real model downloads
pytest -m "requires_gpu"   # GPU-required tests
pytest -m "not slow"       # Skip slow tests
```

---

## 📝 Code Style

- **Docstrings**: Google-style
- **Type hints**: Required for all functions
- **Formatter**: Black (line length 88, Python 3.11)
- **No inline imports**: All imports at module top

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

---

## 🐛 Troubleshooting

### CUDA Issues
- Verify CUDA 12.4 with `nvidia-smi`
- Check `torch.cuda.is_available()` returns `True`
- For cluster: Request GPU in interactive job before installing

### SLURM/NCCL Errors
- Set `export NCCL_P2P_DISABLE=1` in SLURM script
- Check `WORLD_SIZE` and `SLURM_PROCID` environment variables

### Data Loading
- Verify Zarr dataset paths and permissions
- Check dataset schema matches your data
- Ensure sufficient disk space for Zarr cache


### Feature Mismatch Errors
```
ValueError: Action decoding network expects input feature 'fused_visual'
but it's not produced by any encoder
```
**Solution**: Ensure decoder's `input_keys` match encoding pipeline's output features.
Check `encoding_pipeline.get_final_features_to_dimensions()`.

