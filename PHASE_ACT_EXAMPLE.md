# Phase ACT Configuration Example

This example shows how to configure the refactored architecture to replicate the `phase_detr_vae.py` behavior: ACT decoder with phase-conditioned action heads (MoE) for position and gripper actions, plus a phase classifier.

## Architecture Overview

```
Multi-Camera Images (left, right, depth) + Proprioceptive State
    ↓
Encoding Pipeline (CNN encoders + fusion)
    ↓
ACT Decoder (Transformer with VAE)
    ↓
├─ Phase Classifier Head (predicts phase logits)
├─ Position MoE Head (5 experts, one per phase, routed by phase logits)
└─ Gripper MoE Head (5 experts, one per phase, routed by phase logits)
```

## Complete Configuration

```python
from dataclasses import dataclass, field
from typing import Dict, List
from omegaconf import MISSING

from refactoring.configs.main import MainConfig
from refactoring.configs.policy import PolicyConfig
from refactoring.configs.data.task import TaskSpaceConfig
from refactoring.data.task import ActionSpace, ObservationSpace
from refactoring.configs.encoding.pipeline import EncodingPipelineConfig
from refactoring.configs.encoding.encoder import RGBCNNEncoderConfig, ProprioceptiveEncoderConfig, DepthCNNEncoderConfig
from refactoring.configs.encoding.fusion import ConcatFusionConfig, MLPFusionConfig
from refactoring.configs.decoding.decoder import ACTConfig
from refactoring.configs.decoding.action_head import MixtureOfExpertsHeadConfig, MLPActionHeadConfig
from refactoring.configs.training import TrainingConfig
from refactoring.configs.data.dataloader import DataLoaderConfig
from refactoring.data.constants import (
    Cameras,
    POSITION_ACTION_KEY,
    GRIPPER_ACTION_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
    PHASE_LABEL_KEY
)
from refactoring.models.decoding.constants import MoERoutingType


# ============================================================================
# 1. TaskSpace Configuration
# ============================================================================

@dataclass
class PhaseACTTaskConfig(TaskSpaceConfig):
    """TaskSpace configuration for phase-conditioned manipulation."""

    # Action space: 3D position + gripper
    action_space: ActionSpace = field(default_factory=lambda: ActionSpace(
        has_position=True,
        position_dim=3,
        position_in_camera_frame=False,  # Robot frame

        has_orientation=False,  # No orientation prediction

        has_gripper=True,
        gripper_type='binary',  # Binary gripper (open/close)

        deltas_as_actions=False,  # Predict absolute positions
        denoise_actions=True,

        task_has_phases=True,  # Enable phase prediction
        number_of_phases=5,  # 5 surgical phases
    ))

    # Observation space: RGB cameras + depth + proprio
    observation_space: ObservationSpace = field(default_factory=lambda: ObservationSpace(
        # Multi-camera RGB
        camera_keys=[Cameras.LEFT.value, Cameras.RIGHT.value, Cameras.DEPTH.value],

        # Proprioceptive state (robot joint positions)
        use_proprioceptive_data=True,
        use_proprio_robot_frame=True,
        use_proprio_camera_frame=False,

        # Gripper state in observations
        use_gripper_state=True,
        gripper_type='binary',

        # No language for surgical task
        use_language=False,
    ))

    # Temporal horizons
    observation_horizon: int = 1  # Single timestep observation
    prediction_horizon: int = 30  # Predict 30 action steps (chunk_size)


# ============================================================================
# 2. Encoding Pipeline Configuration
# ============================================================================

@dataclass
class PhaseACTEncodingConfig(EncodingPipelineConfig):
    """Encoding pipeline: multi-camera CNNs + proprio + fusion."""

    encoders: Dict = field(default_factory=lambda: {
        # Left camera RGB encoder
        "left_rgb": RGBCNNEncoderConfig(
            camera_key=Cameras.LEFT.value,
            backbone='resnet18',
            pretrained=True,
            frozen=False,
            output_feature_dim=512,
            spatial_pooling='spatial_softmax',  # Use spatial softmax pooling
            normalize=True,
        ),

        # Right camera RGB encoder
        "right_rgb": RGBCNNEncoderConfig(
            camera_key=Cameras.RIGHT.value,
            backbone='resnet18',
            pretrained=True,
            frozen=False,
            output_feature_dim=512,
            spatial_pooling='spatial_softmax',
            normalize=True,
        ),

        # Depth encoder (geometric attention variant)
        "depth": DepthCNNEncoderConfig(
            camera_key=Cameras.DEPTH.value,
            backbone='dformer',  # Use DFormer for geometric attention
            pretrained=True,
            frozen=False,
            output_feature_dim=512,
            spatial_pooling='spatial_softmax',
            normalize=True,
        ),

        # Proprioceptive encoder
        "proprio": ProprioceptiveEncoderConfig(
            input_key=PROPRIO_OBS_ROBOT_FRAME_KEY,
            hidden_dims=[256],
            output_dim=256,
            activation='relu',
            normalization=True,
        ),
    })

    fusion_stages: List = field(default_factory=lambda: [
        # Stage 1: Concatenate all camera features
        ConcatFusionConfig(
            input_features=['left_rgb_features', 'right_rgb_features', 'depth_features'],
            output_feature_name='visual_concat',
        ),

        # Stage 2: MLP to fuse visual features
        MLPFusionConfig(
            input_feature='visual_concat',
            output_feature_name='visual_fused',
            hidden_dims=[1024, 512],
            output_dim=512,
            activation='relu',
            dropout=0.1,
        ),

        # Stage 3: Concatenate visual + proprio
        ConcatFusionConfig(
            input_features=['visual_fused', 'proprio_features'],
            output_feature_name='observation_embedding',
        ),

        # Stage 4: Final MLP fusion
        MLPFusionConfig(
            input_feature='observation_embedding',
            output_feature_name='fused_embedding',
            hidden_dims=[512],
            output_dim=256,  # Hidden dimension for ACT
            activation='relu',
            dropout=0.1,
        ),
    ])


# ============================================================================
# 3. Action Heads Configuration
# ============================================================================

# Base action head for position (used by all phase experts)
position_base_head = MLPActionHeadConfig(
    input_dim=256,  # ACT hidden_dim
    output_dim=3,  # 3D position
    hidden_dims=[128],  # Single hidden layer (256 -> 128 -> 3)
    activation='silu',
    dropout=0.1,
)

# Base action head for gripper (used by all phase experts)
gripper_base_head = MLPActionHeadConfig(
    input_dim=256,
    output_dim=1,  # Binary gripper
    hidden_dims=[64],  # Smaller network (256 -> 64 -> 1)
    activation='silu',
    dropout=0.1,
)

# Phase classifier head (single output, not phase-specific)
phase_classifier_head = MLPActionHeadConfig(
    input_dim=256,
    output_dim=5,  # 5 phases
    hidden_dims=[128],  # 256 -> 128 -> 5
    activation='relu',
    dropout=0.1,
)

# MoE head for position actions (5 phase experts with learned routing)
position_moe_head = MixtureOfExpertsHeadConfig(
    output_dim=3,
    base_expert_config=position_base_head,
    num_experts=5,  # One expert per phase

    # No gating network - use external routing from phase classifier
    gating_input_dim=None,

    routing_type=MoERoutingType.SOFT.value,
    temperature=100.0,  # High temperature for sharper routing
    learnable_temperature=True,  # Learn temperature during training
)

# MoE head for gripper actions (5 phase experts with learned routing)
gripper_moe_head = MixtureOfExpertsHeadConfig(
    output_dim=1,
    base_expert_config=gripper_base_head,
    num_experts=5,

    gating_input_dim=None,  # External routing

    routing_type=MoERoutingType.SOFT.value,
    temperature=100.0,
    learnable_temperature=True,
)


# ============================================================================
# 4. PhaseACT Decoder Configuration
# ============================================================================

@dataclass
class PhaseACTDecoderConfig:
    """Phase-conditioned ACT decoder with MoE routing."""

    # Target class - PhaseACT extends ACT with phase-based routing
    _target_: str = "refactoring.models.decoding.decoders.phase_act.PhaseACT"

    # Which head provides routing weights (phase classifier)
    phase_routing_key: str = PHASE_LABEL_KEY

    # Transformer architecture
    embedding_dimension: int = 256
    number_of_heads: int = 8
    feedforward_dimension: int = 2048
    number_of_encoder_layers: int = 4
    number_of_decoder_layers: int = 6
    activation: str = 'relu'
    dropout: float = 0.1
    normalize_before: bool = False

    # VAE configuration
    use_vae_latent: bool = True
    vae_latent_dimension: int = 32

    # Positional encoding
    positional_encoding_type: str = 'sinusoidal'

    # Action heads mapping
    action_heads: Dict = field(default_factory=lambda: {
        # Phase classifier (predicts phase logits)
        # This head's output is used as routing weights for MoE heads
        PHASE_LABEL_KEY: phase_classifier_head,

        # Position action head (MoE with 5 phase experts)
        # Receives phase logits as routing weights
        POSITION_ACTION_KEY: position_moe_head,

        # Gripper action head (MoE with 5 phase experts)
        # Receives phase logits as routing weights
        GRIPPER_ACTION_KEY: gripper_moe_head,
    })


# ============================================================================
# 5. Policy Configuration
# ============================================================================

@dataclass
class PhaseACTPolicyConfig(PolicyConfig):
    """Complete policy configuration."""

    encoding_pipeline: EncodingPipelineConfig = field(
        default_factory=PhaseACTEncodingConfig
    )

    decoder: ACTConfig = field(
        default_factory=PhaseACTDecoderConfig
    )

    device: str = "cuda"


# ============================================================================
# 6. Training Configuration
# ============================================================================

@dataclass
class PhaseACTTrainingConfig(TrainingConfig):
    """Training configuration."""

    num_epochs: int = 2000
    batch_size: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4

    # Optimizer
    optimizer: str = 'adamw'

    # Learning rate schedule
    lr_scheduler: str = 'cosine'
    warmup_steps: int = 100

    # Gradient clipping
    gradient_clip_norm: float = 10.0

    # EMA for stable training
    use_ema: bool = True
    ema_decay: float = 0.999


# ============================================================================
# 7. DataLoader Configuration
# ============================================================================

@dataclass
class PhaseACTDataLoaderConfig(DataLoaderConfig):
    """DataLoader configuration."""

    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True

    # Augmentation
    use_augmentation: bool = True
    augmentation_config: Dict = field(default_factory=lambda: {
        'random_crop': {'enabled': True, 'scale': (0.9, 1.0)},
        'color_jitter': {'enabled': True, 'brightness': 0.2, 'contrast': 0.2},
    })


# ============================================================================
# 8. Main Configuration
# ============================================================================

@dataclass
class PhaseACTMainConfig(MainConfig):
    """Complete configuration for Phase ACT experiment."""

    task: TaskSpaceConfig = field(default_factory=PhaseACTTaskConfig)
    policy: PolicyConfig = field(default_factory=PhaseACTPolicyConfig)
    training: TrainingConfig = field(default_factory=PhaseACTTrainingConfig)
    dataloader: DataLoaderConfig = field(default_factory=PhaseACTDataLoaderConfig)

    # Experiment tracking
    experiment_name: str = "phase_act_needle_threading"
    save_dir: str = "./experiments/phase_act"

    # Checkpointing
    checkpoint_frequency: int = 100
    save_latest_every_n_epochs: int = 10

    # Logging
    log_frequency: int = 10
    use_wandb: bool = True
    wandb_project: str = "surgical_robotics"
```

## Key Differences from Old Architecture

### 1. **Modular Action Heads**
**Old:**
```python
self.position_action_heads = nn.ModuleList([
    nn.Sequential(nn.LayerNorm(...), nn.Linear(...), ...)
    for _ in range(n_phases)
])
```

**New:**
```python
position_moe_head = MixtureOfExpertsHeadConfig(
    base_expert_config=position_base_head,  # Define once
    num_experts=5,  # Replicate automatically
)
```

### 2. **External Phase Routing**
**Old:** Phase logits computed in forward, routing done manually
```python
phase_logits = self.phase_head(cls_token)
phase_probs = F.softmax(phase_logits / self.temperature, dim=-1)
# Manual weighted sum of expert outputs
```

**New:** Phase classifier outputs routing weights automatically
```python
# Phase classifier outputs routing logits
phase_logits = phase_classifier(features)  # (B, 5)

# MoE heads consume routing weights automatically
position_output = position_moe_head(
    features,
    routing_weights=phase_logits  # External routing
)
```

### 3. **Encoding Pipeline**
**Old:** Manual backbone building and fusion
```python
self.backbones = nn.ModuleDict(build_backbones(...))
self.fusion_proj = nn.Conv2d(...)
```

**New:** Declarative encoding pipeline
```python
encoders: {
    "left_rgb": RGBCNNEncoderConfig(...),
    "right_rgb": RGBCNNEncoderConfig(...),
}
fusion_stages: [
    ConcatFusionConfig(...),
    MLPFusionConfig(...),
]
```

## Usage

```python
from hydra import compose, initialize
from hydra.utils import instantiate

# Load configuration
with initialize(config_path="configs"):
    cfg = compose(config_name="phase_act_config")

# Instantiate policy
policy = instantiate(cfg.policy)

# Forward pass
outputs = policy(
    observations={
        'left': left_images,      # (B, 3, H, W)
        'right': right_images,    # (B, 3, H, W)
        'depth': depth_images,    # (B, 1, H, W)
        'proprio_robot_frame': robot_state,  # (B, proprio_dim)
    },
    actions={
        'position_action': position_actions,  # (B, horizon, 3)
        'gripper_action': gripper_actions,    # (B, horizon, 1)
    }
)

# Outputs contain:
# - outputs['position_action']: (B, horizon, 3) - Routed position predictions
# - outputs['gripper_action']: (B, horizon, 1) - Routed gripper predictions
# - outputs['phase_label']: (B, 5) - Phase logits/probabilities
# - outputs['routing_weights']: (B, 5) - Expert routing weights
# - outputs['expert_outputs']: Expert-specific predictions
```

## Real Data Keys Used

From `refactoring.data.constants`:

### Observation Keys
- `Cameras.LEFT.value` = `"left"` - Left camera RGB
- `Cameras.RIGHT.value` = `"right"` - Right camera RGB
- `Cameras.DEPTH.value` = `"depth"` - Depth camera
- `PROPRIO_OBS_ROBOT_FRAME_KEY` = `"proprio_robot_frame"` - Robot joint angles
- `GRIPPER_STATE_OBS_KEY` = `"gripper_state_obs"` - Gripper state

### Action Keys
- `POSITION_ACTION_KEY` = `"position_action"` - 3D end-effector position
- `GRIPPER_ACTION_KEY` = `"gripper_action"` - Gripper open/close
- `PHASE_LABEL_KEY` = `"phase_label"` - Surgical phase label

### Other Keys
- `IS_PAD_KEY` = `"is_pad"` - Padding mask for variable-length sequences

## How Routing Actually Works

See `ROUTING_EXPLANATION.md` for detailed execution flow, but here's the quick version:

### Step 1: PhaseACT Decoder Forward Pass
```python
# In PhaseACT._apply_action_heads() (src/refactoring/models/decoding/decoders/phase_act.py)

action_embeddings = transformer_decode(observations)  # (B, 30, 256)

# Call phase classifier first
phase_logits = phase_classifier(action_embeddings)  # (B, 30, 5)

# Pass phase logits to MoE heads as routing weights
position = position_moe_head(action_embeddings, routing_weights=phase_logits)
gripper = gripper_moe_head(action_embeddings, routing_weights=phase_logits)
```

### Step 2: MoE Head Internal Routing
```python
# In MixtureOfExpertsHead.forward() (src/refactoring/models/decoding/action_heads/mixture_of_experts.py)

def forward(self, features, routing_weights):
    # Use external routing weights (from phase classifier)
    weights = self.compute_routing_weights(features, routing_weights)
    # weights = softmax(phase_logits / temperature)

    # Run all 5 experts
    expert_outputs = [expert_i(features) for expert_i in self.experts]

    # Weighted combination based on phase
    final_output = sum(weights[:,:,i] * expert_outputs[i] for i in range(5))

    return {
        'action': final_output,
        'routing_weights': weights,
        'expert_outputs': expert_outputs
    }
```

### Step 3: Output Structure
```python
outputs = {
    'phase_label': (B, 30, 5),                    # Phase predictions/routing weights
    'position_action': (B, 30, 3),                # Routed position predictions
    'position_action_routing_weights': (B, 30, 5), # Which experts were used
    'position_action_expert_outputs': (B, 30, 5, 3), # All expert predictions
    'gripper_action': (B, 30, 1),                 # Routed gripper predictions
    'gripper_action_routing_weights': (B, 30, 5),
    'gripper_action_expert_outputs': (B, 30, 5, 1),
}
```

**Key Point**: The `PhaseACT` decoder is what orchestrates the routing by:
1. Calling the phase classifier first
2. Passing its output to MoE heads as `routing_weights`
3. The MoE heads internally handle the weighted expert combination
