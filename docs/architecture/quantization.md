# Quantization

VersatIL implements quantization workflows built on the popular
[`torchao`](https://docs.pytorch.org/ao/main/) library from PyTorch. The
workflow owns the order of operations needed to load a checkpoint, optionally
prepare Quantization-Aware-Training modules before training or inference, export the policy, quantize
the full model or selected target modules, and return the graph that a
deployment backend can save or lower.

The two main workflows in `torchao` are:

- **Eager quantization**: mutates `nn.Module` layers before export. Linear layers only, 8-bits to 2-bits. Mainly dynamic quantization.
- **PyTorch 2 Export (PT2E) quantization**: quantizes the exported graph with PT2E. Linear and Convolutional layers. 8-bits only. Mainly static quantization.

For more information on the difference between
these two workflows, see the
[official torchao documentation](https://docs.pytorch.org/ao/main/). When
`quantization: null`, `NoQuantizationWorkflow` is used, which exports an
unquantized floating-point model.

Each policy can use a single type of quantization workflow; workflow types
cannot be mixed for the same policy.


!!! note

    `torchao` is in early-stage active development, so links and support may
    change with short notice.


## Package Layout

- **`src/versatil/quantization/workflows/`**: workflow implementations and the
  shared `BaseQuantizationWorkflow` contract.
- **`src/versatil/quantization/pt2e/`**: PT2E-only backend adapters that create
  torchao quantizers and configure PT2E-specific settings.
- **`src/versatil/quantization/calibration.py`**: calibration data provider for
  static quantization.
- **`src/versatil/post_training_compression/deployment_backends/`**: deployment
  artifact generation after the selected workflow has produced a graph.

## Workflow Contract

[`BaseQuantizationWorkflow`][versatil.quantization.workflows.base.BaseQuantizationWorkflow]
defines the common interface:

| Method or property | Role |
|--------------------|------|
| `quantization_mode` | Name of the mode: `none`, `eager`, or `pt2e`. |
| `is_qat` | Whether the workflow uses Quantization-Aware-Training (QAT). |
| `targets` | Module-level quantization targets owned by the workflow. |
| `prepare_model()` | Training-time QAT preparation hook. Raises when unsupported. |
| `load_policy_context()` | Loads the checkpoint shape required by the workflow. |
| `validate_targets()` | Validates target paths and rejects overlapping targets. |
| `quantize()` | Runs export and quantization, returning `QuantizedContext`. |

[`QuantizedContext`][versatil.quantization.workflows.base.QuantizedContext]
contains:

- `float_model`: exported float graph;
- `quantized_model`: exported or quantized graph selected by the workflow;
- `example_inputs`: positional tensor inputs used for export and lowering;
- `quantization_workflow`: metadata value stored in the compressed checkpoint.

PTC calls the selected workflow once, then passes the resulting context to the
deployment backend.

## Float Export

[`NoQuantizationWorkflow`][versatil.quantization.workflows.none.NoQuantizationWorkflow]
is selected when the config has `quantization: null`.

It loads a float checkpoint, builds example inputs, exports the policy, and
returns the same exported module as both `float_model` and `quantized_model`.
`prepare_model()` is a no-op, so training code can call it without special
handling when quantization is disabled.

## Quantizing Target Modules

Each module of a PyTorch model can be quantized with a specific quantization
configuration within the same workflow by defining a
[`QuantizationModuleTarget`][versatil.quantization.module_target.QuantizationModuleTarget].
For example, one eager workflow can use an int4 config for `decoder.head` and
an int8 dynamic config for `decoder.backbone`. Target paths must exist in the
policy and must not overlap. `module_path: ""` is the root policy target.

## Eager Quantization

[`EagerQuantizationWorkflow`][versatil.quantization.workflows.eager.EagerQuantizationWorkflow]
uses the torchao
[`quantize_()` API](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.quantize_.html#torchao.quantization.quantize_)
before export. The same class supports eager PTQ and eager QAT.
At the moment, `torchao` only supports eager quantization for `nn.Linear`
layers, with dynamic and static quantization schemes from 8-bit down to 2-bit.

### Eager PTQ

When `is_qat: false`, quantization is applied only after training:

```python
torchao.quantization.quantize_(model, quantize_config)
```

For the root target, the config is applied to the whole policy. For submodule
targets, it filters to `nn.Linear` modules under the
configured `module_path`.

### Eager Quantization-Aware-Training (QAT)
QAT trains the policy with fake-quantization layers that mimic inference conditions.
When `is_qat: true`, the workflow stores the same base torchao PTQ config but wraps
it in a `QATConfig`:

- Training calls `prepare_model()`, which applies
  `QATConfig(base_config=quantize_config, step="prepare")` to eligible
  `nn.Linear` modules selected by the workflow targets.
- Post-training compression reloads the checkpoint through the QAT policy-context path, then
  calls `convert_model()`, which applies
  `QATConfig(base_config=quantize_config, step="convert")`.

## PyTorch 2 Export Quantization

[`PT2EQuantizationWorkflow`][versatil.quantization.workflows.pt2e.PT2EQuantizationWorkflow]
quantizes the exported graph:

1. Load a float policy checkpoint.
2. Build positional example inputs from the policy metadata and tokenizer.
3. Export the policy with `torch.export`.
4. Create one PT2E quantizer per selected `PT2EQuantizationModuleTarget`.
5. Combine those quantizers with `ComposableQuantizer`.
6. Call `prepare_pt2e()`.
7. Calibrate with training batches when any selected PT2E backend is static.
8. Call `convert_pt2e()`.

Static PT2E requires calibration data. Dynamic PT2E skips calibration.
At the moment, `torchao` supports PT2E quantization for `nn.Conv2d` and
`nn.Linear` layers, only with 8-bit quantization schemes.

## PT2E Backends

PT2E backends configure which backend-specific environment settings are required
during PT2E conversion.

| Class | Module | Role |
|-------|--------|------|
| [`BasePT2EBackend`][versatil.quantization.pt2e.backends.base.BasePT2EBackend] | `src/versatil/quantization/pt2e/backends/base.py` | Interface for PT2E quantizer creation and environment setup. |
| [`X86InductorBackend`][versatil.quantization.pt2e.backends.x86_inductor.X86InductorBackend] | `src/versatil/quantization/pt2e/backends/x86_inductor.py` | Creates `X86InductorQuantizer` configs for x86 CPU PT2E quantization. |
| [`XNNPACKPT2EBackend`][versatil.quantization.pt2e.backends.xnnpack.XNNPACKPT2EBackend] | `src/versatil/quantization/pt2e/backends/xnnpack.py` | Creates `XNNPACKQuantizer` configs for ExecuTorch XNNPACK PT2E quantization. |

PT2E backend choice and deployment backend choice are coupled. Use
`X86InductorBackend` with `TorchInductorBackend` for `.pt2` artifacts, and
`XNNPACKPT2EBackend` with `ExecutorchXNNPACKBackend` for `.pte` artifacts.

## Hydra Examples

Float export:

```yaml
quantization: null
```

Eager PTQ:

```yaml
quantization:
  _target_: versatil.quantization.workflows.eager.EagerQuantizationWorkflow
  targets:
    - _target_: versatil.quantization.module_target.EagerQuantizationModuleTarget
      module_path: ""  # it means root, i.e. all modules are selected
      quantize_config:
        _target_: torchao.quantization.Int8DynamicActivationInt8WeightConfig
  is_qat: false
  auto_filter_incompatible_linears: true
```

Eager QAT:

```yaml
quantization:
  _target_: versatil.quantization.workflows.eager.EagerQuantizationWorkflow
  targets:
    - _target_: versatil.quantization.module_target.EagerQuantizationModuleTarget
      module_path: ""  # it means root, i.e. all modules are selected
      quantize_config:
        _target_: torchao.quantization.Int8DynamicActivationIntxWeightConfig
        weight_dtype: ${torch_dtype:int4}
        weight_granularity:
          _target_: torchao.quantization.PerGroup
          group_size: 32
  is_qat: true
  auto_filter_incompatible_linears: true
```

PT2E static x86:

```yaml
quantization:
  _target_: versatil.quantization.workflows.pt2e.PT2EQuantizationWorkflow
  targets:
    - _target_: versatil.quantization.module_target.PT2EQuantizationModuleTarget
      module_path: ""  # it means root, i.e. all modules are selected
      pt2e_backend:
        _target_: versatil.quantization.pt2e.backends.x86_inductor.X86InductorBackend
        is_dynamic: false
        is_qat: false
        reduce_range: false
```

PT2E static XNNPACK:

```yaml
quantization:
  _target_: versatil.quantization.workflows.pt2e.PT2EQuantizationWorkflow
  targets:
    - _target_: versatil.quantization.module_target.PT2EQuantizationModuleTarget
      module_path: ""  # it means root, i.e. all modules are selected
      pt2e_backend:
        _target_: versatil.quantization.pt2e.backends.xnnpack.XNNPACKPT2EBackend
        is_dynamic: false
        is_qat: false
        is_per_channel: true

deployment_backend:
  _target_: versatil.post_training_compression.deployment_backends.executorch_xnnpack.ExecutorchXNNPACKBackend
  max_batch_size: 32
```


## Compatibility Rules

- A compression run uses one unique workflow mode: `none`, `eager`, or `pt2e`.
- `none` is float export. `eager` and `pt2e` quantize the model (or parts of it).
- Quantization target module paths must exist in the policy and must not overlap.
- PT2E backend and deployment backend must be compatible: X86 Inductor writes
  `.pt2`, while XNNPACK writes ExecuTorch `.pte`.
- PT2E QAT is not supported in VersatIL yet.

## Relation To PTC

PTC resolves the configured workflow, calls `workflow.quantize()`, then passes
the returned `QuantizedContext` to the selected deployment backend. See
[Post-Training Compression](post_training_compression.md) for pruning, artifact
serialization, reports, and compressed runtime loading.
