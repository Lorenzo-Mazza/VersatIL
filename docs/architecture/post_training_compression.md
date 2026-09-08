# Post-Training Compression

**What is post-training compression?**
The post-training compression (PTC) pipeline turns a trained policy checkpoint into a deployment artifact for edge or resource-constrained hardware. A PTC run can export a floating-point model, apply pruning, quantize the policy, and save either a Torch Export `.pt2` artifact or an ExecuTorch `.pte` artifact.

The pipeline owns the end-to-end compression job:
checkpoint loading, optional model preparation, pruning, quantization workflow
execution, deployment backend export, serialization, and reporting.

Quantization details live in the dedicated [Quantization](quantization.md) page.
This page describes how PTC orchestrates those workflows.

## Architecture

The PTC package is centered on
[`PostTrainingCompressor`][versatil.post_training_compression.compressor.PostTrainingCompressor].
It operates on [`CompressionTarget`][versatil.post_training_compression.compression_target.CompressionTarget]
entries, then delegates quantization and deployment to separate abstractions:

- **Quantization workflows** decide how the policy is exported or quantized:
  no quantization, eager, or PyTorch 2 Export.
- **Deployment backends** validate workflow modes, PT2E pairings and eager
  representations, then produce Torch Export `.pt2` or ExecuTorch `.pte` artifacts.

Within eager quantization, targets select and filter linear layers and construct
conversion metadata. Quantization schemas supply TorchAO preparation and
conversion configurations and check numerical, dtype, device and calibration
requirements. The workflow executes preparation, calibration, conversion and export.

## Pipeline Flow

```
PostTrainingCompressor.compress()
|
+-- resolve_modules()                    Per-module targets or global fallback
+-- _resolve_quantization_workflow()      none, eager, or pt2e path
+-- deployment_backend.validate_quantization()
|   Validate workflow mode and PT2E backend names
|
+-- workflow.load_policy_context()        Load float or QAT-prepared checkpoint
+-- validate()                            Check preparation/pruning module paths
+-- workflow.validate_targets()           Check quantization target paths
|
+-- _prepare_and_prune()                  Per target:
|   +-- prepare_batchnorms()              Replace FrozenBN with standard BN
|   +-- fuse_conv_batchnorm()             Fold BN weights into Conv2d
|   +-- pruner.prune() x N                Apply pruners sequentially
|
+-- create_exportable_policy()           Adapter for the policy's prediction procedure
+-- workflow.quantize()                  Quantize/export; validate eager target formats
+-- deployment_backend.export()          Build .pt2 descriptor or .pte bytes
+-- save_compressed_model()               Artifact, metadata, normalizer, tokenizer
|
+-- optional QuantizationReport           Coverage, size, divergence, speed
```

## Key Classes

| Class | Module | Role |
|-------|--------|------|
| [`PostTrainingCompressor`][versatil.post_training_compression.compressor.PostTrainingCompressor] | `src/versatil/post_training_compression/compressor.py` | Pipeline orchestrator. Resolves targets, validates compatibility, prepares/prunes, exports, saves. |
| [`CompressionTarget`][versatil.post_training_compression.compression_target.CompressionTarget] | `src/versatil/post_training_compression/compression_target.py` | Per-module preparation and pruning config: `module_path`, preparation, pruning list. |
| [`QuantizationModuleTarget`][versatil.quantization.module_target.QuantizationModuleTarget] | `src/versatil/quantization/module_target.py` | Per-module quantization scope; eager targets select/filter linears and construct conversion metadata. |
| [`QuantizationSchema`][versatil.quantization.schemas.base.QuantizationSchema] | `src/versatil/quantization/schemas/base.py` | TorchAO preparation and conversion configurations, with numerical, dtype, device and calibration checks. |
| [`ExportablePolicy`][versatil.models.exportable.base.ExportablePolicy] | `src/versatil/models/exportable/base.py` | Ordered tensor inputs and outputs for graph capture; specialized adapters handle denoising and token generation. |
| [`DeploymentBackend`][versatil.post_training_compression.deployment_backends.base.DeploymentBackend] | `src/versatil/post_training_compression/deployment_backends/base.py` | Workflow/representation validation and deployment artifact generation. |
| [`TorchInductorBackend`][versatil.post_training_compression.deployment_backends.torch_inductor.TorchInductorBackend] | `src/versatil/post_training_compression/deployment_backends/torch_inductor.py` | Saves Torch Export `.pt2` artifacts. |
| [`ExecutorchXNNPACKBackend`][versatil.post_training_compression.deployment_backends.executorch_xnnpack.ExecutorchXNNPACKBackend] | `src/versatil/post_training_compression/deployment_backends/executorch_xnnpack.py` | Lowers exported programs to ExecuTorch XNNPACK `.pte` artifacts. |
| [`CompressedCheckpointLoader`][versatil.checkpoint_loading.compressed_policy.CompressedCheckpointLoader] | `src/versatil/checkpoint_loading/compressed_policy.py` | Restores inference metadata, normalizer and tokenizer; locates the deployment artifact. |
| [`CompressedPolicyRuntime`][versatil.inference.policy_runtime.compressed_runtime.CompressedPolicyRuntime] | `src/versatil/inference/policy_runtime/compressed_runtime.py` | Runs compressed policies through the inference runtime interface. |

## Compression Targets

`CompressionTarget` lets the config apply preparation and pruning globally or
to selected submodules. Each target contains:

- `module_path`: dotted module path, or `""` for the root policy;
- `preparation`: optional BatchNorm replacement and fusion settings;
- `pruning`: ordered list of pruners.

When `modules` is empty, PTC creates a single root target from the top-level
`preparation` and `pruning` fields.

Quantization targets are configured separately under `quantization.targets`.
See [Quantization](quantization.md) for the target schema.

## Preparation

Preparation runs before pruning and quantization:

- **`prepare_batchnorms_for_quantization()`** replaces non-standard BatchNorm
  variants with standard `nn.BatchNorm2d` in eval mode with tracking disabled.
- **`fuse_all_conv_batchnorm_pairs()`** folds consecutive Conv2d and BatchNorm2d
  pairs into a single Conv2d with adjusted weights and bias, replacing the
  BatchNorm with `nn.Identity` where appropriate.

## Pruning

Pruning is specified as a list of
[`BasePruner`][versatil.post_training_compression.pruning.base.BasePruner]
instances. The list is applied sequentially, so structured and unstructured
pruning can be composed on the same target.

- **[`UnstructuredPruner`][versatil.post_training_compression.pruning.unstructured.UnstructuredPruner]**:
  global L1 magnitude pruning. Defaults to convolution and linear layers;
  normalization scales and embedding tables are never pruned.
- **[`StructuredPruner`][versatil.post_training_compression.pruning.structured.StructuredPruner]**:
  per-layer channel pruning along a configured dimension. By default it targets
  Conv1d, Conv2d, and Linear layers.

## Quantization Hook

PTC calls exactly one workflow mode per compression run:

- `none`: float export through `NoQuantizationWorkflow`;
- `eager`: eager torchao PTQ or eager QAT conversion;
- `pt2e`: PyTorch 2 Export graph quantization.

The workflow returns a `QuantizedContext` containing the exported float graph,
the exported or quantized graph, example inputs, and serialized workflow mode.
See [Quantization](quantization.md) for the workflow contract, QAT behavior,
calibration, and PT2E backend details.

## Deployment Backends

The compressor calls `DeploymentBackend.validate_quantization()` before loading
the checkpoint. This checks the workflow mode and selected PT2E backend names.
It also passes the backend instance to `workflow.quantize()`, where
`validate_eager_target()` checks representation-specific requirements before
eager conversion. Artifact generation runs after the workflow returns a
`QuantizedContext`.

| Backend | Artifact format | Output file | Notes |
|---------|-----------------|-------------|-------|
| [`TorchInductorBackend`][versatil.post_training_compression.deployment_backends.torch_inductor.TorchInductorBackend] | `torch_export_pt2` | `compressed_policy.pt2` | Default backend. Saves the selected exported module as a Torch Export archive. |
| [`ExecutorchXNNPACKBackend`][versatil.post_training_compression.deployment_backends.executorch_xnnpack.ExecutorchXNNPACKBackend] | `executorch_pte` | `compressed_policy.pte` | Lowers the selected exported program with ExecuTorch XNNPACK. |

The deployment backend is stored in metadata so inference can load the artifact according to its file format.
For PT2E quantization, `TorchInductorBackend` pairs with `X86InductorBackend`,
and `ExecutorchXNNPACKBackend` pairs with `XNNPACKPT2EBackend`.

## Compressed Checkpoints

A compressed checkpoint directory contains:

```
compressed/<timestamp>/
+-- compressed_policy.pt2 | compressed_policy.pte
+-- normalizer.pt
+-- compression_metadata.json
+-- quantization_config.yaml
+-- config.yaml
+-- tokenizer/
```

`compression_metadata.json` records:

- model filename and artifact format;
- deployment backend name;
- input and output key ordering;
- source training checkpoint path;
- torch and torchao versions;
- workflow mode (`none`, `eager`, or `pt2e`);
- PT2E backend flags when applicable;
- eager target records, including `schema`, `schema_parameters`, selected/skipped
  layers and converted weight classes;
- the number of calibration observation batches consumed;
- policy graph output meaning and additional input shapes.

`CompressedCheckpointLoader` constructs the observation and action spaces from
the saved configuration and restores the normalizer, tokenizer and denoising
thresholds. `CheckpointMetadata` groups the spaces and observation/prediction
horizons used by checkpoint loaders. The compressed loader retains the rest of
the YAML configuration for inference settings, including camera rotation.

`CompressedPolicyRuntime` loads the deployment artifact and exposes the same
inference interface as `FloatPolicyRuntime`. Floating and QAT checkpoint loaders
also construct a `Policy` for native prediction and training.

!!! note "Runtime dependencies"

    Compressed policy loading and action reconstruction require VersatIL and
    the selected runtime and tokenizer dependencies. Torch Export
    artifacts execute through PyTorch; ExecuTorch artifacts use the ExecuTorch
    runtime. Tokenizer assets and the normalizer are saved with the artifact.

## Hydra Configuration

PTC configs live under `src/versatil/hydra_configs/end_to_end_ptq/`. Top-level fields serve
as defaults for preparation and pruning. Entries in `modules` can override
preparation and pruning for specific submodules. Quantization is configured once
at the top level through `quantization`, and module-level quantization
granularity is expressed inside `quantization.targets`.

```yaml
checkpoint_path: ???
checkpoint_name: last.ckpt
output_directory: null
calibration_steps: 16
generate_report: false

preparation:
  replace_frozen_batchnorm: true
  fuse_conv_batchnorm: true

pruning:
  - _target_: versatil.post_training_compression.pruning.UnstructuredPruner
    amount: 0.5

quantization:
  _target_: versatil.quantization.workflows.pt2e.PT2EQuantizationWorkflow
  targets:
    - _target_: versatil.quantization.module_target.PT2EQuantizationModuleTarget
      module_path: ""
      pt2e_backend:
        _target_: versatil.quantization.pt2e.backends.x86_inductor.X86InductorBackend
        is_dynamic: false
        is_qat: false
        reduce_range: false

deployment_backend:
  _target_: versatil.post_training_compression.deployment_backends.torch_inductor.TorchInductorBackend

modules: []
```

Set `quantization: null` for floating-point export. Replace the `quantization` block with
an eager or PT2E workflow as described in [Quantization](quantization.md).

## Compatibility Rules

- A compression run uses one quantization workflow: `none`, `eager`, or `pt2e`.
- `modules` configures preparation and pruning; `quantization.targets` configures
  module-level quantization.
- The current compressor builds its policy context and example inputs on CPU.
  Export and calibration use that same device, matching the device-specific
  operations recorded in the graph.
