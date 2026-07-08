# Compressing a Trained Policy

Post-training compression (PTC) shrinks a trained checkpoint and speeds up
CPU inference for deployment on edge or resource-constrained hardware,
without retraining. A compression run executes a pipeline of stages, each of
them optional: preparation, pruning, quantization, and export to a
deployable artifact. Internals live in the
[post-training compression](../architecture/post_training_compression.md) and
[quantization](../architecture/quantization.md) architecture pages.

In this tutorial you build one compression config from scratch and grow it
stage by stage. Every step is runnable, starting from a training checkpoint
directory; each run writes a self-contained artifact plus normalizer and
metadata to `<checkpoint_path>/compressed/<timestamp>/`.

## Step 1: A Float Export Baseline

Create `configs/my_compression.yaml`:

```yaml
# @package _global_
defaults:
  - /post_training_compression
  - _self_

checkpoint_path: /path/to/training/checkpoint
```

and run it:

```bash
python -m versatil.endpoints.post_training_compress \
    --config-dir configs \
    --config-name my_compression
```

This does no pruning and no quantization yet, but it is not a no-op:
preparation runs by default, replacing frozen BatchNorm layers with their
inference equivalent and folding BatchNorm into the preceding convolution
(`preparation.replace_frozen_batchnorm` and
`preparation.fuse_conv_batchnorm`, both on by default). Fused Conv+BN later
quantizes as one op instead of two, which is what the CPU backends expect.
The prepared policy is exported through `torch.export` as a float `.pt2`
artifact: your baseline for comparing everything that follows.

Useful fields at this stage:

| Field | Default | Meaning |
|-------|---------|---------|
| `checkpoint_name` | `last.ckpt` | Checkpoint file inside the directory. |
| `output_directory` | `compressed/<timestamp>` | Where the artifact is written. |
| `generate_report` | `false` | Write a compression report: op coverage, size reduction, output divergence vs the float policy. |

## Step 2: Add Pruning

Pruning runs after preparation and removes the least important weights.
VersatIL ships two kinds:

- **Unstructured pruning** zeroes individual weights, ranked globally by L1
  magnitude across all convolution and linear layers. The tensors keep their
  shapes, so accuracy degrades gracefully even at high sparsity, but the
  speedup depends on how well the export stack exploits the resulting
  sparsity.
- **Structured pruning** zeroes whole channels (rows or columns of a layer,
  ranked per layer by Lp-norm). It is coarser and costs more accuracy per
  removed weight, but the removed structure is what dense CPU kernels
  actually skip.

Add a `pruning` list to the config. Pruners apply sequentially and sparsity
accumulates, so the two kinds can be mixed:

```yaml
pruning:
  - _target_: versatil.post_training_compression.pruning.UnstructuredPruner
    amount: 0.3
  - _target_: versatil.post_training_compression.pruning.StructuredPruner
    amount: 0.2
```

Re-run the endpoint: the artifact is still float, now sparse. Pruning is
independent of everything that follows; keep it, tune the amounts, or drop
the list again.

## Step 3: Add Quantization

Quantization is configured as a workflow. Start with PT2E graph
quantization, which quantizes the exported graph: add a `quantization` block
selecting the workflow, a target (which submodule to quantize; `""` is the
whole policy), and a backend quantizer. For an x86 CPU server:

```yaml
calibration_steps: 32
quantization:
  _target_: versatil.quantization.workflows.pt2e.PT2EQuantizationWorkflow
  targets:
    - _target_: versatil.quantization.module_target.PT2EQuantizationModuleTarget
      module_path: ""
      pt2e_backend:
        _target_: versatil.quantization.pt2e.backends.x86_inductor.X86InductorBackend
        is_dynamic: false
```

`is_dynamic: false` selects static quantization, which needs calibration:
the compressor streams `calibration_steps` batches from the checkpoint's
dataloader through the prepared graph to fit the activation observers.
Dynamic schemes (`is_dynamic: true`) skip calibration.

Re-run: the artifact is now a statically quantized `.pt2`.

## Step 4: Retarget to Edge Hardware

The backend pair decides the artifact format: the X86 Inductor quantizer
pairs with `.pt2`, while XNNPACK pairs with ExecuTorch `.pte`. To produce an
ExecuTorch artifact for ARM and mobile-class targets, swap the quantizer and
add the matching deployment backend:

```yaml
quantization:
  _target_: versatil.quantization.workflows.pt2e.PT2EQuantizationWorkflow
  targets:
    - _target_: versatil.quantization.module_target.PT2EQuantizationModuleTarget
      module_path: ""
      pt2e_backend:
        _target_: versatil.quantization.pt2e.backends.xnnpack.XNNPACKPT2EBackend
        is_dynamic: false
        is_per_channel: true
deployment_backend:
  _target_: versatil.post_training_compression.deployment_backends.executorch_xnnpack.ExecutorchXNNPACKBackend
  max_batch_size: 32
```

## Step 5: Eager Quantization Instead of PT2E

The alternative workflow is eager torchao quantization, which mutates the
`nn.Module` before export instead of quantizing the exported graph. Eager
and PT2E cannot be combined in one run; replace the `quantization` block:

```yaml
quantization:
  _target_: versatil.quantization.workflows.eager.EagerQuantizationWorkflow
  targets:
    - _target_: versatil.quantization.module_target.EagerQuantizationModuleTarget
      module_path: ""
      quantize_config:
        _target_: torchao.quantization.Int8DynamicActivationIntxWeightConfig
        weight_dtype: ${torch_dtype:int4}
        weight_granularity:
          _target_: torchao.quantization.PerGroup
          group_size: 32
  is_qat: false
  auto_filter_incompatible_linears: true
```

The `quantize_config` is a plain torchao config, so any scheme torchao
supports for `nn.Linear` composes here, from 8-bit down to 2-bit.

## Converting a QAT Checkpoint

A checkpoint trained with
[quantization-aware training](quantization_aware_training.md) contains
fake-quant modules rather than quantized weights. The eager block from
Step 5 performs the conversion when `is_qat: true` is set: the compressor
reloads the checkpoint through the QAT policy context and applies
`QATConfig(step="convert")` instead of fresh PTQ.

The `quantize_config` must match the preset the policy was trained with;
converting with a different scheme silently mismatches the fake-quant
statistics the weights were trained under.

The converted model deploys like any other eager-quantized policy, and the
`deployment_backend` picks the lowering: with none set, the compressor
defaults to the Torch Inductor backend and writes a `.pt2` that runs through
PyTorch on x86 CPUs; with the ExecuTorch XNNPACK backend from Step 4, it
lowers to a `.pte` for ARM and mobile-class targets.

## Deploying the Artifact

The [deployment endpoint](inference.md) detects compressed artifacts
automatically; point it at the compressed directory:

```bash
python -m versatil.endpoints.deploy \
    checkpoint_path=/path/to/checkpoint/compressed/<timestamp> \
    device=cpu \
    client.model_server_address=10.0.0.1 \
    client.model_server_port=5556
```

Torch Export `.pt2` artifacts run through PyTorch and can be compiled with
`torch.compile`; ExecuTorch `.pte` artifacts run through the ExecuTorch
adapter on CPU.

## Measuring What Compression Cost You

Set `generate_report: true` in any step to get a report next to the
artifact: quantized-op coverage, on-disk size reduction, and the output
divergence between the compressed and the original float policy on real
batches. Check the divergence before trusting a compressed policy in
evaluation.

## Complete Examples

Assembled configs covering these combinations ship with VersatIL under
`end_to_end_ptq/`: `unstructured_prune_x86`, `pt2e_xnnpack`, and
`eager_xnnpack`. They run directly, for example:

```bash
python -m versatil.endpoints.post_training_compress \
    --config-name end_to_end_ptq/unstructured_prune_x86 \
    checkpoint_path=/path/to/training/checkpoint
```

## Known Limitations

- PT2E export and calibration run on CPU: `torch.export` bakes device
  metadata into the graph, so the prepared model must not be moved to CUDA.
- `torch.export` specializes `batch=1` to a constant; dynamic batch dims
  need `batch>=2`.
- The first inference call of a compiled artifact is slow while the inductor
  backend generates and compiles kernels.
- PT2E QAT is not supported yet; QAT conversion goes through the eager
  workflow.
