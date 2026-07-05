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
- **Deployment backends** decide the final artifact format: e.g. Torch Export
  `.pt2` or ExecuTorch `.pte`.

## Pipeline Flow

```
PostTrainingCompressor.compress()
|
+-- resolve_modules()                    Per-module targets or global fallback
+-- _resolve_quantization_workflow()      none, eager, or pt2e path
+-- _validate_deployment_backend_compatibility()
|   Validate workflow mode vs deployment backend
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
+-- ExportablePolicy.from_policy()        Positional tensor I/O wrapper
+-- workflow.quantize()                   Export or quantize policy
+-- deployment_backend.export()           Build .pt2 descriptor or .pte bytes
+-- save_compressed_model()               Artifact, metadata, normalizer, tokenizer
|
+-- optional QuantizationReport           Coverage, size, divergence, speed
```

## Key Classes

| Class | Module | Role |
|-------|--------|------|
| [`PostTrainingCompressor`][versatil.post_training_compression.compressor.PostTrainingCompressor] | `src/versatil/post_training_compression/compressor.py` | Pipeline orchestrator. Resolves targets, validates compatibility, prepares/prunes, exports, saves. |
| [`CompressionTarget`][versatil.post_training_compression.compression_target.CompressionTarget] | `src/versatil/post_training_compression/compression_target.py` | Per-module preparation and pruning config: `module_path`, preparation, pruning list. |
| [`QuantizationModuleTarget`][versatil.quantization.module_target.QuantizationModuleTarget] | `src/versatil/quantization/module_target.py` | Per-module quantization target owned by the selected quantization workflow. |
| [`ExportablePolicy`][versatil.models.exportable_policy.ExportablePolicy] | `src/versatil/models/exportable_policy.py` | Wraps `Policy` with positional tensor I/O for `torch.export`. |
| [`DeploymentBackend`][versatil.post_training_compression.deployment_backends.base.DeploymentBackend] | `src/versatil/post_training_compression/deployment_backends/base.py` | Base class for final deployment artifact generation. |
| [`TorchInductorBackend`][versatil.post_training_compression.deployment_backends.torch_inductor.TorchInductorBackend] | `src/versatil/post_training_compression/deployment_backends/torch_inductor.py` | Saves Torch Export `.pt2` artifacts. |
| [`ExecutorchXNNPACKBackend`][versatil.post_training_compression.deployment_backends.executorch_xnnpack.ExecutorchXNNPACKBackend] | `src/versatil/post_training_compression/deployment_backends/executorch_xnnpack.py` | Lowers exported programs to ExecuTorch XNNPACK `.pte` artifacts. |
| [`CompressedCheckpointLoader`][versatil.checkpoint_loading.compressed_policy.CompressedCheckpointLoader] | `src/versatil/checkpoint_loading/compressed_policy.py` | Loads compressed checkpoint metadata, normalizer, tokenizer, and deployment artifact. |
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

Deployment backends run after the workflow returns a `QuantizedContext`.

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
- PT2E backend flags when applicable.

`CompressedCheckpointLoader` reads the metadata, normalizer, tokenizer, and
artifact. `CompressedPolicyRuntime` then exposes the same runtime interface used
by the inference client for floating-point policies.

!!! warning "Compressed artifacts are not standalone"

    Currently, compressed models are not fully standalone: they still require
    a complete VersatIL installation, including its dependencies. Since this
    is not ideal for edge deployment, self-contained edge-device inference
    runtime is currently under development.

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
- `modules` do not carry quantization configs. Use `quantization.targets` for
  module-level quantization.
- Export currently runs on CPU. PT2E calibration also runs on CPU because the
  exported graph records device metadata.
