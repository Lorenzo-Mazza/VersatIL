# Architecture Overview

## Policy Composition

A VersatIL policy is built from four decoupled components, orchestrated by the [`Policy`][versatil.models.policy.Policy] class:

**[`Policy`][versatil.models.policy.Policy] = [`EncodingPipeline`][versatil.models.encoding.pipeline.EncodingPipeline] + Algorithm + [`ActionDecoder`][versatil.models.decoding.decoders.base.ActionDecoder] + Loss**

| Component | Responsibility |
|---|---|
| **[`EncodingPipeline`][versatil.models.encoding.pipeline.EncodingPipeline]** | Multi-modal observation encoding with optional fusion |
| **Algorithm** | Learning paradigm (how to train and predict) |
| **[`ActionDecoder`][versatil.models.decoding.decoders.base.ActionDecoder]** | Neural network architecture (how to process features) |
| **Loss** | Composable objective function |

```python
# 1. Encode observations
features = encoding_pipeline(observations)  # Multi-modal -> unified representation

# 2. Decode actions (algorithm orchestrates the decoder internally)
predictions = algorithm.forward(
    network=decoder,       # Algorithm receives decoder as a callable
    features=features,
    actions=ground_truth,  # During training
)

# 3. Compute loss
targets = algorithm.get_targets(
    algorithm_output=predictions,
    ground_truth_actions=ground_truth,
)
loss = loss_module(predictions, targets)
```

The `Policy.forward()` method executes encoding and decoding. `Policy.compute_loss()` calls `forward()` and then passes predictions through the loss module.

At inference time, `Policy.predict_action()` normalizes raw observations, encodes them, calls `algorithm.predict()` (which does not require ground-truth actions), and unnormalizes the output.

For the full API, see [`Policy`][versatil.models.policy.Policy].

## Algorithm vs Architecture Separation

VersatIL decouples the learning paradigm from the neural network structure. The two axes are composable — certain pairings are naturally constrained by their mathematical formulation (e.g., timestep-conditioned decoders require a generative algorithm that provides timesteps).

**Algorithm** defines *how* to train and predict:

- [`BehavioralCloning`][versatil.models.decoding.algorithm.behavior_cloning.BehavioralCloning] -- direct supervised learning of expert actions
- [`Diffusion`][versatil.models.decoding.algorithm.diffusion.Diffusion] -- iterative denoising via Denoising Score Matching
- [`FlowMatching`][versatil.models.decoding.algorithm.flow_matching.FlowMatching] -- continuous normalizing flows
- [`VariationalAlgorithm`][versatil.models.decoding.algorithm.variational.VariationalAlgorithm] -- wraps any base algorithm with VAE-style latent variables

**[`ActionDecoder`][versatil.models.decoding.decoders.base.ActionDecoder]** defines *what* neural network processes features:

- Transformer-based (ACT, DiT, GPT, DETR, etc.)
- UNet-based (Conditional Action UNet for Diffusion Policy)
- VLA decoders that run generative VLM backbones
- MoE wrappers (applicable on top of any decoder)

```python
# Same decoder architecture, different algorithms
Policy(encoding_pipeline=..., algorithm=BehavioralCloning(), decoder=ACT(...), loss=...)
Policy(encoding_pipeline=..., algorithm=Diffusion(...),       decoder=ACT(...), loss=...)

# Same algorithm, different decoder architectures
Policy(encoding_pipeline=..., algorithm=FlowMatching(...), decoder=DiTBlockActionTransformer(...), loss=...)
Policy(encoding_pipeline=..., algorithm=BehavioralCloning(...), decoder=GPTActionTransformer(...), loss=...)
```

The [`DecodingAlgorithm`][versatil.models.decoding.algorithm.base.DecodingAlgorithm] base class defines two abstract methods:
- `forward(network, features, actions)` -- training pass (with ground-truth actions)
- `predict(network, features)` -- inference pass (without actions)

The algorithm receives the decoder as a `network` parameter and orchestrates its use. For generative algorithms, this means adding noise, calling the decoder to denoise, and computing the training objective internally.


## VLM Backbone Wiring

Pi0/SmolVLA-style decoders configure a `vlm_backbone` and run it directly on
normalized/tokenized image-text observations during decoder forward. Those
decoders declare `needs_raw_observations=True`, so [`Policy`][versatil.models.policy.Policy]
passes the normalized/tokenized observation tensors through the feature
dictionary. OpenVLA/OpenVLA-OFT-style decoders follow the same raw-observation
path and use their configured VLM backbone to build the language/image prefix.

## Composable Loss

The loss module is decoupled from the decoder architecture. Some loss terms are naturally tied to specific algorithms (e.g., probability measures like KL divergence and MMD to variational inference). Loss components are combined via weighted sums in `ComposableLoss`:

- **Regression losses** -- MSE, L1, Huber for continuous action prediction
- **Classification losses** -- BCE for binary gripper, cross-entropy for tokenized actions
- **Probability measures** -- KL divergence, Maximum Mean Discrepancy with configurable kernels for variational algorithms
- **Optimal transport** -- Sinkhorn divergence for action sequence matching

Loss configs are named by composition (e.g., `regression_KL.yaml`), not by dataset or decoder.

## Feature Naming Contract

VersatIL uses strict naming conventions to wire encoders to decoders automatically. Instead of manually passing tensors, components match by name.

**The rule:** `feature_name = "{encoder_name}_{modality}"`
For multi-output encoders, each modality produces a separate feature: `vlm_rgb`, `vlm_language`.


| Encoder name | Modality | Feature name |
|---|---|---|
| `left_eye` | `rgb` | `left_eye_rgb` |
| `robot_state` | `proprio` | `robot_state_proprio` |
| `vlm_model` | `fused_rgb_language` | `vlm_model_fused_rgb_language` |

!!! note "Multi-camera naming"
    For multi-camera encoders, the modality includes the camera key separated by a colon: `{encoder_name}_{modality}:{camera_key}` (e.g., `stereo_rgb:left`, `stereo_rgb:right`).

!!! note "Pipeline prefixing"
    The encoding pipeline always prepends each encoder's output key with the encoder name. Encoders return raw modality keys (e.g., `rgb`), the pipeline produces prefixed keys (e.g., `left_eye_rgb`).

**Fusion outputs** specify `output_name` directly:

```python
fusion = AttentionFusion(
    input_features=["left_eye_rgb", "right_eye_rgb"],
    output_name="fused_visual"  # Direct name, no prefix
)
```

**Decoder inputs** reference encoder or fusion output names via `input_keys`.

All encoder and fusion outputs persist in the feature dictionary -- fusion does not consume its inputs.

## Feature Types

Feature dimensions are declared via `FeatureMetadata(key, feature_type, dimension)`:

| Type | Dimension | Example |
|---|---|---|
| **SPATIAL** | `(C, H, W)` | CNN feature maps |
| **SEQUENTIAL** | `(S, D)` | Transformer token sequences, VLM fused embeddings |
| **FLAT** | `(D,)` | Pooled embeddings |

Decoders declare which feature types they require via `DecoderInput.required_types` and which they reject via `raises_for_types`. See [Encoding Pipeline](encoding.md#featuremetadata) for details.


## Runtime Validation

Configuration errors are caught at initialization, not during training. Validation happens at three levels:

### 1. Encoding Pipeline Validation

On `EncodingPipeline.__init__()`:

- Encoder output keys are checked for duplicates
- Conditional encoder `condition_key` references are verified against available features
- Fusion input features are validated against encoder outputs and prior fusion outputs

### 2. Decoder Input Validation

During experiment validation (`ExperimentValidator.validate_decoder_encoder_compatibility()` in `validation.py`), the decoder's [`DecoderInput`][versatil.models.decoding.decoders.base.DecoderInput] is validated against the encoding pipeline's final features:

```python
available_features = encoding_pipeline.get_features()
decoder.decoder_input.validate_feature_types(
    available_features=available_features
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
- Camera keys are valid [`Cameras`][versatil.data.constants.Cameras] enum values
- On-the-fly action metadata matches schema observation metadata

!!! warning "Feature type mismatches"
    A decoder expecting **FLAT** features (1D) but receiving **SPATIAL** features (3D) from a CNN raises a `ValueError` at initialization. Add a fusion or pooling stage to resolve the mismatch.

## Observation and Action Spaces

**[`ObservationSpace`][versatil.data.task.ObservationSpace]** defines what data the policy receives:

- Camera observations (RGB, depth) via [`CameraMetadata`][versatil.data.metadata.CameraMetadata]
- Proprioceptive state (position, orientation, gripper) via typed metadata classes
- Language instructions via tokenized observations

**[`ActionSpace`][versatil.data.task.ActionSpace]** defines what the policy predicts:

- Position actions (with configurable coordinate frame)
- Orientation actions (roll, euler, quaternion representations)
- Gripper actions (binary or continuous)
- Actions can be precomputed (stored in Zarr) or computed on-the-fly (e.g., deltas from consecutive states)

Both spaces expose `get_required_zarr_keys()` to declare which keys must exist in the dataset. [`TaskSpace`][versatil.data.task.TaskSpace] validates these keys against the dataset schema at initialization.

## From Training to Deployment

After training, a policy checkpoint can be deployed directly via [`PolicyLoader`][versatil.inference.policy_loading.float_loader.PolicyLoader] (float inference with `torch.compile`) or compressed first for edge deployment:

```
Training checkpoint (.ckpt)
  → PostTrainingCompressor.compress()
    → Preparation → Pruning → Quantization
  → Compressed checkpoint (.pt2)
    → CompressedPolicyLoader
      → InferenceClient (ZMQ transport to robot/simulation)
```

Both [`PolicyLoader`][versatil.inference.policy_loading.float_loader.PolicyLoader] and [`CompressedPolicyLoader`][versatil.inference.policy_loading.compressed_loader.CompressedPolicyLoader] implement the same inference interface (`run_inference(obs_dict) → action_dict`), so the [`InferenceClient`][versatil.inference.inference_client.InferenceClient] works with either. See [Post-Training Compression](post_training_compression.md) for details on the compression pipeline.
