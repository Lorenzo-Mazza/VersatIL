# Deploying a Policy

This tutorial takes a trained checkpoint and runs it against an environment.
Inference uses a server-client architecture: an **environment server** owns
the environment and steps it, while the **policy client** (VersatIL) receives
observations, runs inference, and sends actions back over ZMQ. The
environment behind the server is irrelevant to the client: a simulated
benchmark, a real robot wrapping its drivers and cameras, or anything else
that implements the transport protocol. Server and client can run on
different machines, within the same network.

This tutorial uses a simulated benchmark as the running example, but only
Step 1 is simulation-specific; the client side is identical everywhere. For
how the client works internally (transports, preprocessing, temporal
aggregation) and how to implement a server or a custom transport for your own
hardware, see the [inference architecture](../architecture/inference.md).

## Prerequisites

- A training checkpoint directory produced by `versatil.endpoints.train`. It
  contains the checkpoint file (`last.ckpt` by default), the resolved
  `config.yaml`, and the fitted normalizer and tokenizer state. The client
  rebuilds the policy from this directory.
- An environment server. For the simulated benchmarks, ready-made servers
  are installed from their own repositories:

| Benchmark | Server repository |
|-----------|-------------------|
| LIBERO / LIBERO-PRO | [simulation_libero](https://github.com/nct-tso-robotics/simulation_libero) |
| LIBERO-Plus | [simulation_libero_plus](https://github.com/nct-tso-robotics/simulation_libero_plus) |
| Meta-World | [simulation_metaworld](https://github.com/nct-tso-robotics/simulation_metaworld) |
| PushT | [simulation_pusht](https://github.com/nct-tso-robotics/simulation_pusht) |
| Block Pushing | [simulation_block_push](https://github.com/nct-tso-robotics/simulation_block_push) |
| Multimodal Ant | [simulation_multimodal_ant](https://github.com/nct-tso-robotics/simulation_multimodal_ant) |
| Franka Kitchen | [simulation_kitchen](https://github.com/nct-tso-robotics/simulation_kitchen) |
| UR3 Block Push | [simulation_ur3_block_push](https://github.com/nct-tso-robotics/simulation_ur3_block_push) |

Each server repository documents its own installation and benchmark-specific
options. The servers run environments in parallel batches, track per-task
success rates, and record rollout videos and trajectory CSVs.

## Step 1: Start the Simulation Server

On the simulation machine, start the server for your benchmark. For LIBERO:

```bash
python -m versatil_inference.run_evaluation \
    --task_suite_name libero_spatial \
    --num_trials_per_task 10 \
    --max_parallel_envs 10 \
    --port 5556 \
    --output_folder ./results
```

On headless machines (clusters without a display), select a headless MuJoCo
rendering backend first:

```bash
export MUJOCO_GL=egl
```

The server binds the port and waits for a client to register. Argument names
vary slightly per benchmark (for example Meta-World uses `--benchmark_name`
and `--number_of_trials`); check the server repository's README.

## Step 2: Run the Policy Client

On the policy machine, point the deployment endpoint at the checkpoint and pass
the server IP and port:

```bash
python -m versatil.endpoints.deploy \
    checkpoint_path=/path/to/checkpoint_dir \
    client.model_server_address=127.0.0.1 \
    client.model_server_port=5556
```

The client registers with the server and drives the evaluation loop until the
server reports completion. Results (per-task success rates, videos,
trajectory CSVs) are written on the server side, in `--output_folder` or in
the checkpoint's `rollouts/` directory when unset.

The endpoint is Hydra-based, so every setting is a `key=value` override:

| Override | Default | Meaning |
|----------|---------|---------|
| `checkpoint_path` | required | Checkpoint directory to load. |
| `checkpoint_name` | `last.ckpt` | Checkpoint file inside the directory. |
| `device` | auto | `cuda` when available, else `cpu`; set explicitly to pin. |
| `compile_model` | `true` | Compile the policy with `torch.compile`. The first inference call is slow while kernels compile. |
| `client.model_server_address` | `127.0.0.1` | Environment server address. |
| `client.model_server_port` | `5555` | Environment server port. |
| `client.temporal_aggregation` | `false` | Query the policy every step and ensemble overlapping chunk predictions with exponentially weighted averaging; one action is executed per step. |
| `client.action_execution_horizon` | full chunk | Actions executed from each predicted chunk before re-predicting. Only used when `client.temporal_aggregation=false`. |
| `client.compression_type` | `raw` | Wire compression for camera observations (`raw`, `jpeg`, `png`); match the server setting. |
| `client.request_timeout_seconds` | none | Fail instead of blocking forever when the server dies. |
| `client.update_rate_hz` | none | Fixed action-send rate for real-time control; leave unset for simulation, where the server paces the loop. |

### Chunk Execution Modes

The two settings select between mutually exclusive execution modes:

- **Chunked execution** (`temporal_aggregation=false`, the default): the
  policy predicts a chunk and the client executes
  `action_execution_horizon` actions from it before re-predicting. The
  default executes the full chunk open-loop; lowering it re-predicts more
  often and reacts faster at the cost of more inference calls. It cannot
  exceed the policy's prediction horizon.
- **Temporal ensemble** (`temporal_aggregation=true`): the policy is queried
  at every environment step and the overlapping predictions for the current
  timestep are averaged with exponential weighting, smoothing chunk
  boundaries. Exactly one action is executed per step, so
  `action_execution_horizon` is ignored.

!!! warning
    With chunked execution, a policy trained with `observation_horizon > 1`
    receives history frames spaced `action_execution_horizon` steps apart,
    while its training windows were contiguous. The client logs a warning
    for this combination; prefer the temporal ensemble for such policies.

Re-predicting after every 8 executed actions:

```bash
python -m versatil.endpoints.deploy \
    checkpoint_path=/path/to/checkpoint_dir \
    client.model_server_address=10.0.0.1 \
    client.model_server_port=5556 \
    client.action_execution_horizon=8
```

## Compressed Checkpoints

The endpoint detects [post-training compressed](../architecture/post_training_compression.md)
artifacts automatically: pass the `compressed/<timestamp>/` directory as
`checkpoint_path` and the client loads the compressed runtime instead of the
floating-point policy, typically with `device=cpu`.

## Beyond Simulation

Nothing above changes on real hardware. A robot-side environment server
publishes camera and proprioception observations and executes received
actions, speaking the same protocol the simulation servers speak (the
`tso-robotics-sockets` package provides the ZMQ implementation). The client
command stays exactly the same; for real-time control, set
`client.update_rate_hz` to the robot's control rate so actions are sent at a
fixed frequency instead of as fast as inference allows.

## Troubleshooting

- **Client hangs at startup**: the server is not reachable. Check address,
  port, and firewalls, and set `client.request_timeout_seconds=30` to fail
  fast instead of blocking.
- **Server crashes on rendering**: set `MUJOCO_GL=egl` (or `osmesa`) on
  headless machines. If EGL still fails to initialize a device display
  (common on nodes without the nvidia EGL userspace driver), enumerate the
  EGL devices and point `MUJOCO_EGL_DEVICE_ID` at the Mesa software device;
  rendering then runs on the CPU.
- **Slow first prediction**: expected with `compile_model=true`; disable it
  for quick smoke tests.
- **Observation key errors**: the checkpoint's observation space must match
  what the server provides. Evaluate checkpoints against the benchmark they
  were trained on, for example a LIBERO-trained policy against the LIBERO
  server.
