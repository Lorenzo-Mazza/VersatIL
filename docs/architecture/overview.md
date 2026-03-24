# Architecture Overview

## Policy Composition

A VersatIL policy is built from four decoupled components, orchestrated by the `Policy` class:

**Policy = EncodingPipeline + Algorithm + ActionDecoder + Loss**

| Component | Responsibility |
|---|---|
| **EncodingPipeline** | Multi-modal observation encoding with optional fusion |
| **Algorithm** | Learning paradigm (how to train and predict) |
| **ActionDecoder** | Neural network architecture (how to process features) |
| **Loss** | Composable objective function |

```python
# 1. Encode observations
features = encoding_pipeline(observations)  # Multi-modal → unified representation

# 2. Decode actions (algorithm orchestrates the decoder internally)
predictions = algorithm.forward(
    network=decoder,       # Algorithm receives decoder as a callable
    features=features,
    actions=ground_truth,  # During training
)

# 3. Compute loss
loss = loss_module(predictions, targets)
```

The `Policy.forward()` method executes encoding and decoding. `Policy.compute_loss()` calls `forward()` and then passes predictions through the loss module.

At inference time, `Policy.predict_action()` normalizes raw observations, encodes them, calls `algorithm.predict()` (which does not require ground-truth actions), and unnormalizes the output.

For the full API, see the [Policy reference](../reference/versatil/models/policy.md).

## Algorithm vs Architecture Separation

VersatIL decouples the learning paradigm from the neural network structure. The two axes are composable — certain pairings are naturally constrained by their mathematical formulation (e.g., timestep-conditioned decoders require a generative algorithm that provides timesteps).

**Algorithm** defines *how* to train and predict:

- `BehavioralCloning` -- direct supervised learning of expert actions
- `Diffusion` -- iterative denoising via Denoising Score Matching
- `FlowMatching` -- continuous normalizing flows
- `VariationalAlgorithm` -- wraps any base algorithm with VAE-style latent variables

**ActionDecoder** defines *what* neural network processes features:

- Transformer-based (ACT, DiT, GPT, DETR, Free Transformer, etc.)
- UNet-based (Conditional Action UNet for Diffusion Policy)
- MoE wrappers (applicable on top of any decoder)

```python
# Same decoder architecture, different algorithms
Policy(encoding_pipeline=..., algorithm=BehavioralCloning(), decoder=ACT(...), loss=...)
Policy(encoding_pipeline=..., algorithm=Diffusion(...),       decoder=ACT(...), loss=...)

# Same algorithm, different decoder architectures
Policy(encoding_pipeline=..., algorithm=FlowMatching(...), decoder=DiTBlockDecoder(...), loss=...)
Policy(encoding_pipeline=..., algorithm=FlowMatching(...), decoder=GPTActionDecoder(...), loss=...)
```

The `DecodingAlgorithm` base class defines two abstract methods:

- `forward(network, features, actions)` -- training pass (with ground-truth actions)
- `predict(network, features)` -- inference pass (without actions)

The algorithm receives the decoder as a `network` parameter and orchestrates its use. For generative algorithms, this means adding noise, calling the decoder to denoise, and computing the training objective internally.

## Composable Loss

The loss module is decoupled from the decoder architecture. Some loss terms are naturally tied to specific algorithms (e.g., probability measures like KL divergence and MMD to variational inference). Loss components are combined via weighted sums in `ComposableLoss`:

- **Regression losses** -- MSE, L1, Huber for continuous action prediction
- **Classification losses** -- BCE for binary gripper, cross-entropy for tokenized actions
- **Probability measures** -- KL divergence, Maximum Mean Discrepancy with configurable kernels for variational algorithms
- **Optimal transport** -- Sinkhorn divergence for action sequence matching

Loss configs are named by composition (e.g., `regression_KL.yaml`), not by dataset or decoder.

## Feature Naming Contract

VersatIL uses strict naming conventions to wire encoders to decoders automatically. Instead of manually passing tensors, components match by name.

**The rule:** `feature_name = "{encoder_name}_{output_key}"`

| Encoder name | Output key | Feature name |
|---|---|---|
| `left_eye` | `rgb` | `left_eye_rgb` |
| `robot_state` | `proprio` | `robot_state_proprio` |
| `vlm_model` | `rgb` | `vlm_model_rgb` |
| `vlm_model` | `language` | `vlm_model_language` |

For multi-output encoders (e.g., VLMs), use dot notation in fusion/decoder configs to select specific outputs: `vlm_model.rgb`, `vlm_model.language`.

**Fusion outputs** specify `output_name` directly:

```python
fusion = AttentionFusion(
    input_features=["left_eye_rgb", "right_eye_rgb"],
    output_name="fused_visual"  # Direct name, no prefix
)
```

**Decoder inputs** reference encoder or fusion output names via `input_keys`.

### Feature Consumption

When fusion modules combine features, the input features are consumed and removed from the output dictionary. Only fusion outputs and non-consumed encoder features reach the decoder.

```
Encoders produce: A, B, C, D
Fusion 1: B + C → E  (consumes B and C)
Fusion 2: E + D → F  (consumes E and D)
Final output: {A, F}  (not {A, B, C, D, E, F})
```

## Feature Types

Feature dimensions determine their type, which decoders use for validation:

| Type | Dimension | Example |
|---|---|---|
| **SPATIAL** | `(C, H, W)` | CNN feature maps |
| **SEQUENTIAL** | `(T, D)` | Transformer token sequences |
| **FLAT** | `int` or `(D,)` | Pooled embeddings |

Decoders declare which feature types they require via `DecoderInput.required_types` and which they reject via `raises_for_types`.

## Runtime Validation

Configuration errors are caught at initialization, not during training. Validation happens at three levels:

### 1. Encoding Pipeline Validation

On `EncodingPipeline.__init__()`:

- Encoder output keys are checked for duplicates
- Conditional encoder `condition_key` references are verified against available features
- Fusion input features are validated against encoder outputs and prior fusion outputs

### 2. Decoder Input Validation

During experiment validation (`ExperimentValidator.validate_decoder_encoder_compatibility()` in `validation.py`), the decoder's `DecoderInput` is validated against the encoding pipeline's final features:

```python
available_features = encoding_pipeline.get_final_features_to_dimensions()
decoder.decoder_input.validate_feature_types(
    available_features_to_dims=available_features
)
```

This checks that:

- All required feature keys exist
- Required feature types (SPATIAL, SEQUENTIAL, FLAT) are present
- No rejected feature types are provided

### 3. Task Space Validation

On `TaskSpace.__init__()`:

- Action space keys exist in the dataset schema
- Observation space keys exist in the dataset schema
- Camera keys are valid `Cameras` enum values
- On-the-fly action metadata matches schema observation metadata

!!! warning "Feature type mismatches"
    A decoder expecting **FLAT** features (1D) but receiving **SPATIAL** features (3D) from a CNN raises a `ValueError` at initialization. Add a fusion or pooling stage to resolve the mismatch.

## Observation and Action Spaces

**ObservationSpace** defines what data the policy receives:

- Camera observations (RGB, depth) via `CameraMetadata`
- Proprioceptive state (position, orientation, gripper) via typed metadata classes
- Language instructions via tokenized observations

**ActionSpace** defines what the policy predicts:

- Position actions (with configurable coordinate frame)
- Orientation actions (roll, euler, quaternion representations)
- Gripper actions (binary or continuous)
- Actions can be precomputed (stored in Zarr) or computed on-the-fly (e.g., deltas from consecutive states)

Both spaces expose `get_required_zarr_keys()` to declare which keys must exist in the dataset. `TaskSpace` validates these keys against the dataset schema at initialization.

## From Training to Deployment

After training, a policy checkpoint can be deployed directly via `PolicyLoader` (float inference with `torch.compile`) or compressed first for edge deployment:

```
Training checkpoint (.ckpt)
  → PostTrainingCompressor.compress()
    → Preparation → Pruning → Quantization
  → Compressed checkpoint (.pt2)
    → CompressedPolicyLoader
      → InferenceClient (ZMQ transport to robot/simulation)
```

Both `PolicyLoader` and `CompressedPolicyLoader` implement the same inference interface (`run_inference(obs_dict) → action_dict`), so the `InferenceClient` works with either. See [Post-Training Compression](post_training_compression.md) for details on the compression pipeline.
