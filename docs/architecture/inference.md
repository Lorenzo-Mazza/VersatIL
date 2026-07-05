# Inference Pipeline

The VersatIL inference pipeline connects a trained policy to any environment server — real robot hardware, simulation, or custom setups. Communication is transport-agnostic: the [`ObservationTransport`][versatil.inference.protocol.ObservationTransport] and [`ActionTransport`][versatil.inference.protocol.ActionTransport] Python protocols decouple the inference loop from the transport mechanism.

The built-in ZMQ transport relies on our own PyPI packages:

- [**tso-robotics-sockets**](https://pypi.org/project/tso-robotics-sockets/): Generic ZMQ socket transport with protocol keys (`ServerRoute`, `InferenceRequestKey`, `CompressionType`).
- [**versatil-constants**](https://pypi.org/project/versatil-constants/): Shared domain constants for structured action/observation message passing (`ActionComponent`, `ActionMetadataField`, `ObsKey`, `GripperType`, `OrientationRepresentation`).

Both libraries define the message format. Any server implementing the protocol can be integrated.

```
ObservationTransport.receive()
  -> ObservationPreprocessor.parse_response()        (decompress, rotate)
  -> ObservationBuffer.add()                         (accumulate temporal history)
  -> [when buffer is ready]
  -> ObservationPreprocessor.transform_camera_observations()  (resize, normalize, depth clamp)
  -> PolicyRuntime.run_inference()
  -> TemporalAggregator.store_and_average()          [optional]
  -> ActionPostprocessor.format_action()
  -> ActionPostprocessor.build_action_metadata()
  -> ActionTransport.send()
```

## Transport Protocol Pattern

Communication with the environment server is abstracted behind two `typing.Protocol` interfaces, allowing in principle any transport mechanism (ZMQ sockets, HTTP, shared memory, direct function calls) to be used.

```python
class ObservationTransport(Protocol):
    def receive(self, requested_keys: list[str], compression_type: str) -> dict: ...
    def register(self, client_name: str) -> dict: ...
    def close(self) -> None: ...

class ActionTransport(Protocol):
    def send(self, actions: dict, action_metadata: dict) -> dict: ...
```

### ZMQ Implementation

The built-in [`SocketObservationTransport`][versatil.inference.socket_transport.SocketObservationTransport] and [`SocketActionTransport`][versatil.inference.socket_transport.SocketActionTransport] use the `tso-robotics-sockets` package for ZMQ-based communication:

```python
class SocketObservationTransport:
    def __init__(
        self,
        server_address: str = "127.0.0.1",
        server_port: int = 5555,
        request_timeout_seconds: float | None = None,
    ): ...
```

The transport sends requests via `ServerRoute` enums (e.g., `GET_OBSERVATION`, `REGISTER_CLIENT`, `SEND_ACTION`) and handles serialization/deserialization automatically.

### Custom Transports

Any object satisfying the [`ObservationTransport`][versatil.inference.protocol.ObservationTransport] or [`ActionTransport`][versatil.inference.protocol.ActionTransport] protocol can be used. Custom environments can integrate directly, without going through ZMQ.

## Observation Preprocessing

[`ObservationPreprocessor`][versatil.inference.observation_preprocessor.ObservationPreprocessor] converts raw server responses into model-ready tensors.

```python
class ObservationPreprocessor:
    def __init__(
        self,
        camera_keys: list[str],
        state_keys: list[str],
        has_language: bool,
        camera_metadata: dict[str, CameraMetadata],
        compression_type: str = "raw",
        rotate_images: bool = False,
        depth_clamp_ranges: dict[str, tuple[float, float]] | None = None,
        state_dtypes: dict[str, str] | None = None,
    ): ...
```

### Response Parsing

The preprocessor handles both single-environment and multi-environment server responses. Multi-environment responses contain dict-valued observation data keyed by environment index.

```python
def parse_response(self, response: dict) -> dict[int, dict[str, np.ndarray | str]]:
    """Returns dict mapping environment index to observation dict."""
```

For single-environment responses, the result is wrapped as environment index `0`.

### Image Decompression

Raw image data from the transport is decompressed using `tso_robotics_sockets.decompress_array()` with the configured compression type (e.g., raw, JPEG, PNG). If `rotate_images` is enabled, images are flipped 180 degrees for camera mounting correction.

### Camera Transforms

The `transform_camera_observations()` method applies consistent transforms to all camera images per timestep:

1. **Resize** -- All RGB and depth images are resized to per-camera dimensions from [`RGBCameraMetadata`][versatil.data.metadata.RGBCameraMetadata] and [`DepthCameraMetadata`][versatil.data.metadata.DepthCameraMetadata]
2. **RGB normalization** -- uint8 images are converted to float32 in `[0, 1]`
3. **Depth clamping** -- Depth values are clamped to `[depth_min, depth_max]` derived from training normalizer statistics

```python
def transform_camera_observations(
    self, recent_observations: dict[str, list]
) -> dict[str, torch.Tensor]:
    """Returns dict mapping camera key to tensor (observation_horizon, C, H, W)."""
```

!!! note "Depth Clamp Ranges"
    Per-camera depth clamp ranges are exposed through `PolicyRuntime.depth_clamp_ranges`. Float runtimes read them from the policy normalizer, while compressed runtimes read them from the serialized compressed normalizer.

## Observation Buffering

[`ObservationBuffer`][versatil.inference.observation_buffer.ObservationBuffer] maintains a sliding window of observations per environment, accumulating timesteps until the required observation horizon is reached.

```python
class ObservationBuffer:
    def __init__(self, buffer_size: int, required_keys: list[str]): ...
    def add(self, observations: dict[str, Any]) -> None: ...
    def is_ready(self) -> bool: ...
    def get_recent(self, count: int | None = None) -> dict[str, list[Any]]: ...
    def reset(self) -> None: ...
```

The buffer enforces that all `required_keys` are present in every `add()` call. Once `buffer_size` observations have been accumulated, `is_ready()` returns `True` and inference can proceed. Older observations are evicted on each new addition.

## Checkpoint Loading And Runtime

Checkpoint loading and inference execution are separate responsibilities:

- [`BaseCheckpointLoader`][versatil.checkpoint_loading.base.BaseCheckpointLoader] loads config, tokenizer, policy metadata, normalizer state, and checkpoint weights.
- [`PolicyRuntime`][versatil.inference.policy_runtime.base.PolicyRuntime] is the inference-facing interface consumed by [`InferenceClient`][versatil.inference.inference_client.InferenceClient].
- Runtime implementations compose checkpoint loaders and delegate shared metadata such as tokenizer, task spaces, horizons, denoising thresholds, and depth clamp range.

[`FloatPolicyRuntime`][versatil.inference.policy_runtime.float_runtime.FloatPolicyRuntime] uses [`FloatCheckpointLoader`][versatil.checkpoint_loading.float_policy.FloatCheckpointLoader] to restore a regular training checkpoint, then runs the restored PyTorch policy.

```python
class FloatPolicyRuntime:
    def __init__(
        self,
        device: torch.device,
        checkpoint_path: str,
        checkpoint_name: str = "last.ckpt",
        seed: int = 42,
        compile_model: bool = True,
    ): ...
```

### Float Checkpoint Restoration

1. Loads `config.yaml` from the checkpoint directory
2. Instantiates the full experiment configuration via `hydra.utils.instantiate()`
3. Validates the configuration with `validate_experiment()`
4. Loads the tokenizer from the `tokenizer/` subdirectory (if present)
5. Loads model weights via `LightningPolicy.load_state_dict()`
6. Validates critical checkpoint components (encoder, decoder, normalizer weights)
7. Exposes the restored policy and metadata to the runtime

### Inference Execution

```python
def run_inference(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Run policy inference with autocast and no_grad."""
```

Inference runs under `torch.autocast` with the configured precision and `torch.no_grad()` for memory efficiency.

### Runtime Metadata

[`PolicyRuntime`][versatil.inference.policy_runtime.base.PolicyRuntime] exposes key metadata through the composed checkpoint loader:

- `denoising_thresholds` -- Per-action-key thresholds for denoising action data, if it was used during training, consumed by [`ActionPostprocessor`][versatil.inference.action_postprocessor.ActionPostprocessor]
- `depth_clamp_ranges` -- Per-camera min/max depth map values from the normalizer, used by [`ObservationPreprocessor`][versatil.inference.observation_preprocessor.ObservationPreprocessor]
- `observation_space` / `action_space` -- The policy's task spaces
- `prediction_horizon` / `observation_horizon` -- Temporal window sizes

## Compressed Policy Runtime

[`CompressedPolicyRuntime`][versatil.inference.policy_runtime.compressed_runtime.CompressedPolicyRuntime] runs compressed artifacts produced by the [post-training compression pipeline](post_training_compression.md). Supported artifact formats are:

- `torch_export_pt2`: a Torch Export `.pt2` archive loaded with `torch.export.load()`.
- `executorch_pte`: an ExecuTorch `.pte` artifact executed through the ExecuTorch adapter.

The runtime uses [`CompressedCheckpointLoader`][versatil.checkpoint_loading.compressed_policy.CompressedCheckpointLoader] to read compression metadata, normalizer state, tokenizer assets, artifact format, input/output key ordering, and the original training config. It shares the same `run_inference()` interface as [`FloatPolicyRuntime`][versatil.inference.policy_runtime.float_runtime.FloatPolicyRuntime], so [`InferenceClient`][versatil.inference.inference_client.InferenceClient] works with either runtime.

```python
runtime = CompressedPolicyRuntime(
    device=torch.device("cpu"),
    checkpoint_path="/path/to/compressed/<timestamp>",
)
actions = runtime.run_inference(obs_dict=observations)
```

For `.pt2` artifacts, the runtime reconstructs the PT2E backend when the metadata says the model came from the PT2E workflow, validates the target device, and applies `torch.compile` when appropriate. For `.pte` artifacts, the runtime requires CPU execution and calls the ExecuTorch adapter directly.

## Action Postprocessing

[`ActionPostprocessor`][versatil.inference.action_postprocessor.ActionPostprocessor] converts raw policy output tensors into structured action dictionaries for the server.

```python
class ActionPostprocessor:
    def __init__(
        self,
        action_space: ActionSpace,
        denoising_thresholds: dict[str, float],
    ): ...
```

### Structured Actions

The `format_action()` method produces a dict keyed by `ActionComponent` values:

```python
{
    "position": [dx, dy, dz],
    "orientation": [roll],
    "gripper": [1.0],
}
```

### Gripper Sigmoid

For binary grippers, the raw predicted value is passed through a sigmoid function and thresholded at 0.5:

```python
probability = 1.0 / (1.0 + np.exp(-raw_value[0]))
# BinaryGripperRange.ZERO_ONE: threshold to {0, 1}
# BinaryGripperRange.MINUS_ONE_ONE: threshold to {-1, 1}
```

### Action Denoising

Denoising thresholds computed during training are applied at inference time. If the L2 norm of a predicted action is below the threshold for its key, the action is zeroed out:

```python
if threshold is not None and np.linalg.norm(value) < threshold:
    value = np.zeros_like(value)
```

### Action Metadata

`build_action_metadata()` constructs a metadata dict sent alongside actions, containing per-component information:

| Field | Description |
|-------|-------------|
| `dimension` | Number of values for this component |
| `action_type` | Computation method (delta or next_timestep) |
| `frame` | Coordinate frame (robot or camera) |
| `orientation_representation` | Quaternion, euler, or roll |
| `gripper_type` | Binary or continuous |
| `binary_gripper_range` | Value range for binary grippers |

## Temporal Aggregation

[`TemporalAggregator`][versatil.inference.temporal_aggregation.TemporalAggregator] performs exponential-weighted averaging of overlapping action predictions across timesteps. This smooths the output when multiple inference steps produce predictions for the same future timestep.

```python
class TemporalAggregator:
    def __init__(
        self,
        device: torch.device,
        action_keys_to_dimensions: dict[str, int],
        prediction_horizon: int,
        max_timesteps: int = 10000,
        exponential_decay: float = 0.01,
        favor_more_recent: bool = True,
    ): ...
```

### How It Works

At each timestep, the aggregator:

1. Stores the full predicted action chunk (prediction_horizon steps into the future)
2. Identifies all previous predictions that overlap with the current timestep
3. Computes exponential weights: `w_i = exp(-decay * i) / sum(weights)`
4. Returns the weighted average across all overlapping predictions

When `favor_more_recent=True` (default), newer predictions receive higher weight. The decay factor controls how aggressively older predictions are down-weighted.

```python
def store_and_average(
    self, current_predictions: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Returns dict mapping action key to averaged tensor of shape (dimension,)."""
```

!!! tip "When to Use Temporal Aggregation"
    Temporal aggregation reduces jitter in action execution at the cost of increased latency. It is most beneficial for tasks with smooth, continuous motions and less suitable for tasks requiring rapid discrete state changes.

## [`InferenceClient`][versatil.inference.inference_client.InferenceClient]

[`InferenceClient`][versatil.inference.inference_client.InferenceClient] orchestrates the full inference loop, managing multiple environments simultaneously.

```python
class InferenceClient:
    def __init__(
        self,
        policy_runtime: PolicyRuntime,
        observation_transport: ObservationTransport,
        action_transport: ActionTransport,
        temporal_aggregation: bool = False,
        action_execution_horizon: int | None = None,
        favor_more_recent: bool = True,
        exponential_decay: float = 0.01,
        compression_type: str = "raw",
        max_timesteps: int = 800,
        timing_log: bool = False,
        update_rate_hz: float | None = None,
        online_explanation_source: OnlineExplanationSource | None = None,
    ): ...
```

### Episode Loop

The `run_episode()` method first calls `observation_transport.register(client_name=...)` to register with the server, then executes steps until the server signals completion or `max_steps` is reached. Each `step()` performs:

1. **Receive** -- Request observations from the server via [`ObservationTransport`][versatil.inference.protocol.ObservationTransport]
2. **Status check** -- Handle server status (finished, error, processing, creating environment)
3. **Reset handling** -- Reset per-environment state when the server signals environment resets
4. **Parse** -- Convert raw response into per-environment observation dicts (decompress, rotate)
5. **Buffer** -- Add parsed observations to per-environment [`ObservationBuffer`][versatil.inference.observation_buffer.ObservationBuffer] instances
6. **Remove inactive environments** -- Remove environments no longer present in server responses via `_remove_inactive_environments()`
7. **Transform** -- For ready buffers, apply camera transformations (resize, normalize, depth clamp) via `ObservationPreprocessor.transform_camera_observations()`
8. **Infer** -- Batch transformed observations from all ready environments and run `PolicyRuntime.run_inference()`
9. **Aggregate** -- Optionally apply [`TemporalAggregator`][versatil.inference.temporal_aggregation.TemporalAggregator] per environment
10. **Postprocess** -- Format actions via [`ActionPostprocessor`][versatil.inference.action_postprocessor.ActionPostprocessor]
11. **Send** -- Transmit structured actions and metadata via [`ActionTransport`][versatil.inference.protocol.ActionTransport]

### Multi-Environment Support

The client maintains per-environment state ([`EnvironmentState`][versatil.inference.inference_client.EnvironmentState]) containing an [`ObservationBuffer`][versatil.inference.observation_buffer.ObservationBuffer] and optional [`TemporalAggregator`][versatil.inference.temporal_aggregation.TemporalAggregator]. Environments are:

- **Created** when first seen in a server response
- **Updated** with each new observation
- **Reset** when the server sends a reset signal for that environment index
- **Removed** when no longer present in server responses

Inference batches observations from all environments with full buffers (`is_ready() == True`) into a single forward pass, then distributes the results back to individual environments.

### Rate Limiting

When `update_rate_hz` is set, the client sleeps after each action send to maintain the target environment update rate. This is useful when the downstream robot controller needs time to execute the predicted movement or the environment transition dynamics need time to settle.
The `timing_log` flag enables per-step timing breakdowns (preprocessing, inference, postprocessing) logged at each step.

## Simulation Servers

We provide ZMQ server wrappers for common robot learning simulators, so policies can be rolled out without extra glue code:

| Simulator | Original | ZMQ Server Wrapper |
|---|---|---|
| LIBERO / LIBERO-PRO | [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/master) | [simulation_libero](https://github.com/nct-tso-robotics/simulation_libero) |
| LIBERO+ | [GitHub](https://github.com/sylvestf/LIBERO-plus) | [simulation_libero_plus](https://github.com/nct-tso-robotics/simulation_libero_plus) |
| MetaWorld | [GitHub](https://github.com/Farama-Foundation/Metaworld) | [simulation_metaworld](https://github.com/nct-tso-robotics/simulation_metaworld) |
| PushT | [Diffusion Policy PushT](https://github.com/real-stanford/diffusion_policy/tree/main/diffusion_policy/env/pusht) | [simulation_pusht](https://github.com/nct-tso-robotics/simulation_pusht) |
| Franka Kitchen | [relay-policy-learning](https://github.com/google-research/relay-policy-learning) | [simulation_kitchen](https://github.com/nct-tso-robotics/simulation_kitchen) |
| BlockPush | [IBC BlockPushing](https://github.com/google-research/ibc/tree/master/environments/block_pushing) | [simulation_block_push](https://github.com/nct-tso-robotics/simulation_block_push) |
| UR3 BlockPush | [VQ-BeT UR3](https://github.com/jayLEE0301/vq_bet_official/tree/main/envs/ur3) | [simulation_ur3_block_push](https://github.com/nct-tso-robotics/simulation_ur3_block_push) |
| Multimodal Ant | [VQ-BeT AntEnv](https://github.com/jayLEE0301/vq_bet_official/tree/main/envs/antenv) | [simulation_multimodal_ant](https://github.com/nct-tso-robotics/simulation_multimodal_ant) |

The built-in ZMQ transport works for both simulation and real hardware. For custom environments, implement the [`ObservationTransport`][versatil.inference.protocol.ObservationTransport] and [`ActionTransport`][versatil.inference.protocol.ActionTransport] protocols with any transport mechanism.
