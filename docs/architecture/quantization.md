# Quantization

## Background

**What is post-training quantization?**
Post-training quantization (PTQ) converts selected trained weights, and optionally
activations, to lower precision. The quantization configuration determines the
numerical format; the deployment backend determines which kernels execute it.
Memory use, inference time and prediction quality depend on both choices.
Static activation quantization estimates ranges from calibration data. Dynamic
activation quantization computes ranges during inference. Hybrid methods (e.g. SmoothQuant) can still require calibration, even when the final activation quantizer is dynamic.

**What is quantization-aware-training?**
Quantization-aware training (QAT) simulates quantization during training, allowing
the floating-point weights to adapt to rounding and clipping. Final conversion
must use the matching quantization configuration and trained quantization state.

## Workflows

VersatIL implements quantization workflows built on the
[`torchao`](https://docs.pytorch.org/ao/main/) library from PyTorch. The
workflow owns the order of operations needed to load a checkpoint, optionally
prepare Quantization-Aware-Training modules before training or inference, export the policy, quantize
the full model or selected target modules, and return the graph that a
deployment backend can save or lower.

The two main workflows in `torchao` are:

- **Eager quantization**: transforms selected
  `nn.Linear` or `nn.Embedding` modules through `quantize_()` before exporting
  the policy.
- **PyTorch 2 Export (PT2E) quantization**: annotates and transforms the exported
  graph using a backend-specific quantizer. VersatIL's current PT2E adapters
  provide INT8 linear and supported convolution quantization.


For more information on the difference between
these two workflows, see the
[official torchao documentation](https://docs.pytorch.org/ao/main/). When
`quantization: null`, `NoQuantizationWorkflow` is used, which exports an
unquantized floating-point model.

Each policy uses one quantization workflow.


!!! note

    Quantization APIs and supported formats depend on the installed TorchAO
    release. VersatIL's installation guide lists its supported dependency versions.


## Package Layout

- **`src/versatil/models/exportable/`**: policy adapters for graph capture and
  the input/output metadata saved with the artifact.
- **`src/versatil/quantization/workflows/`**: workflow implementations and the
  shared `BaseQuantizationWorkflow` contract.
- **`src/versatil/quantization/pt2e/`**: PT2E-only backend adapters that create
  torchao quantizers and configure PT2E-specific settings.
- **`src/versatil/quantization/module_target.py`**: layer selection, group-size
  filtering and metadata construction for each configured target.
- **`src/versatil/quantization/schemas/`**: TorchAO preparation/conversion settings and
  numerical, dtype, device and calibration checks for the eager workflow.
- **`src/versatil/quantization/metadata.py`**: data-only records of target settings,
  selected layers and converted weight types.
- **`src/versatil/quantization/calibration.py`**: representative observation
  batches and full-policy prediction for module observer calibration.
- **`src/versatil/post_training_compression/deployment_backends/`**: validation
  of workflow modes, PT2E pairings and eager representations, followed by artifact
  generation after the workflow produces a graph.

## Responsibilities

The policy supplies the prediction procedure. Its export adapter expresses that
procedure with tensor inputs and outputs. The selected workflow determines when
quantization and graph capture happen:

```text
Policy + export adapter
  ├─ Floating: export
  ├─ Eager: prepare using schema configuration → calibration when required
  │         → convert using schema configuration → export
  └─ PT2E: export → backend quantizer preparation → calibration when required
                   → graph conversion
        ↓
Deployment backend → artifact + export metadata + tokenizer + normalizer
        ↓
CompressedPolicyRuntime → reconstructed actions
```

The eager workflow also exports a floating reference before converting weights.
For PTQ, this initializes projections created during the first prediction. Layer
selection follows that initialization and precedes quantization preparation.
Its `eager` configuration name identifies module transformation through TorchAO's
`quantize_()` API. Device, numerical precision and artifact execution are separate
choices: the base quantization configuration defines the numerical format, while
the deployment backend produces `.pt2` or `.pte` artifacts.

### Workflow, target, schema and backend

The eager quantization components have four responsibilities:

| Component | Responsibility |
|---|---|
| `EagerQuantizationWorkflow` | Order initialization, preparation, calibration, conversion and export; retain QAT preparation state |
| `EagerQuantizationModuleTarget` | Select linear or embedding layers by scope and type, filter incompatible dimensions, and construct target metadata |
| `QuantizationSchema` | Specify preparation/conversion settings and check numerical settings, device, dtype and calibration statistics |
| `DeploymentBackend` | Validate workflow modes, PT2E backend pairings and eager representations; produce the deployment artifact |

Each eager target contains a schema. The schema's base TorchAO configuration
specifies weight and activation precision, granularity and tensor representation.

**Schema = preparation configuration + conversion configuration.**

The preparation configuration tells TorchAO which observers or fake-quantization
layers to insert. Observers collect activation statistics during calibration;
fake-quantization layers simulate quantization during training. The conversion
configuration tells TorchAO how to produce the quantized weights.
`QuantizationSchema` provides these through `preparation_config()` and
`conversion_config()`. The workflow applies the returned configurations.

For PTQ, `DirectQuantizationSchema` supplies the base TorchAO configuration as its
conversion configuration, with `None` for preparation. The workflow passes that
configuration to `quantize_()`, which computes quantization parameters from the
selected layers' weights and converts them. For QAT, the schema supplies
configurations that insert fake-quantization layers before training and convert
them after training.
`SmoothQuantSchema` supplies observer preparation and calibrated conversion
configurations for activation-aware weight rescaling.

For example, the same `EagerQuantizationWorkflow` executes both procedures:

| Schema | Base configuration | Workflow operations before export |
|--------|--------------------|--------------------------|
| `DirectQuantizationSchema` | `Int8WeightOnlyConfig` | Select layers, quantize weights |
| `SmoothQuantSchema` | `Int8DynamicActivationInt8WeightConfig` | Select layers, insert observers, run calibration, smooth and quantize |

The workflow passes these configurations to TorchAO's `quantize_()` function,
runs representative inference for calibration, and exports the resulting model.
QAT training runs in the trainer between preparation and checkpoint conversion.
Quantization schemas serve the eager workflow. PT2E uses a backend
quantizer to annotate the exported graph before observer insertion and conversion.

The schema package defines `QuantizationSchema` in `schemas/base.py`,
`DirectQuantizationSchema` in `schemas/direct.py`, and `SmoothQuantSchema` in
`schemas/smoothquant.py`. Import each class from its defining module. Hydra
registers the concrete schema configurations under `quantization/schema`.

### Validation and measurements

The workflow coordinates validation before weight conversion: targets check
their paths and eligible layers, schemas check quantization requirements, and
the deployment backend checks its supported workflow and representation.
Invalid settings raise an error identifying the target and conflicting
requirement. Configurations needing execution tests produce a logged warning.

Each target resolves layer selection once. The workflow reuses those names for
preparation and conversion, validating every target before quantization starts.
Its result includes per-target conversion metadata and the number
of calibration batches consumed. Compression saves these under
`quantization_targets` and `calibration_batches` in `compression_metadata.json`.
Each target record identifies its `schema` and `schema_parameters`, base TorchAO
configuration, selected and skipped layers, and converted weight classes.
`QuantizationReport` evaluates graph coverage, numerical differences, size and
execution time when `generate_report` is enabled.

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
| `validate_targets()` | Validates target paths and rejects overlapping layer selections. |
| `quantize()` | Runs export and quantization, returning `QuantizedContext`. |

[`QuantizedContext`][versatil.quantization.workflows.base.QuantizedContext]
contains:

- `float_model`: reference graph before conversion; floating for PTQ and
  fake-quantized for QAT;
- `quantized_model`: exported or quantized graph selected by the workflow;
- `example_inputs`: positional tensor inputs used for export and lowering;
- `quantization_workflow`: metadata value stored in the compressed checkpoint;
- `calibration_batches`: consumed observation batches, or `None` when the workflow
  leaves this count unspecified. Module quantization reports it, including zero for
  direct PTQ and QAT conversion;
- `quantization_targets`: `QuantizationTargetMetadata` records of actual eager
  selections and converted weight types, or `None` when the workflow leaves these
  details unspecified. `QuantizedLayerMetadata` records each selected layer's
  name, module type, weight shape, device and dtype.

PTC calls the selected workflow once, then passes the resulting context to the
deployment backend.
`save_compressed_model()` converts the metadata dataclasses to dictionaries when
writing `compression_metadata.json`.

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
policy. Eager targets with different `module_type` values can share a path;
targets of the same type require disjoint scopes. `module_path: ""` selects the
root policy scope.

## Policy Artifacts

An exported policy takes processed observations and produces either normalized
actions or action-token IDs. Compression saves this distinction in
`policy_export_metadata` in `compression_metadata.json` using `PolicyExportMetadata`.
The metadata also describes additional sampling inputs and their order.
Floating-point and quantized artifacts share this format. Native checkpoint
inference follows `Policy.predict_action()`.

The `models/exportable/` package contains three adapter modules:

- `base.py`: `ExportablePolicy`, for continuous-action prediction.
- `denoising.py`: `ExportableDenoisingPolicy`, for flow and diffusion with
  explicit noise inputs.
- `autoregressive.py`: `ExportableTokenPolicy`, for bounded token generation.

`factory.create_exportable_policy()` selects the adapter, and `metadata.py` defines
the serialized `PolicyExportMetadata`.

| Policy | Graph inputs after observations | Graph outputs | Runtime processing |
|--------|---------------------------------|---------------|--------------------|
| Continuous behavioral cloning | None | Normalized actions | Reverse action normalization |
| Flow matching or DDIM | Initial noise for each action component | Normalized actions | Sample noise, run graph, reverse action normalization |
| DDPM | Initial noise, then per-step noise for each action component | Normalized actions | Sample noise, run graph, reverse action normalization |
| OpenVLA, Pi0FAST or GPT with behavioral cloning | None | Integer token IDs | Decode the saved action tokenizer and token mappings, reverse action normalization |

### Autoregressive VLA policies

OpenVLA and Pi0FAST share `AutoregressiveVLADecoder` and the bounded generation
interface. Their model and tokenizer components differ:

| Preset | Vision-language backbone | Action representation |
|--------|--------------------------|-----------------------|
| `openvla` | Prismatic | Binned values mapped into the language vocabulary |
| `pi0_fast` | PaliGemma | FAST BPE tokens encoding DCT coefficients, mapped into the language vocabulary |

Export uses `BehavioralCloning`, greedy generation (`deterministic: true`) and a
fitted action tokenizer. The tokenizer's action dimension and prediction horizon
must match the policy. OpenVLA generates the fixed number of binned action
values; Pi0FAST uses its configured maximum token length and EOS stopping.
Binned training token capacity includes the action values and their EOS token;
the transformer context must also accommodate the generated action chunk.
`GPTActionTransformer` implements the same interface for smaller token models
and regression tests.

The graph contains the complete bounded generation sequence. Each invocation
creates a fresh cache; sequences that finish early repeat their end-of-sequence
token through the remaining positions. The compressed directory includes the
tokenizer assets needed to reconstruct actions. Inference validates the integer
token tensor and uses the same action decoding and normalization as the trained
policy. Binned decoding requires a complete action-value sequence. FAST decoding
uses the saved BPE vocabulary, coefficient scale and inverse DCT.

!!! note "Precision when validating token generation"

    Export keeps a fixed observation-prefix width, including masked padding.
    Native generation trims that padding. The different tensor shapes can select
    different attention kernels and change low-precision rounding. Near-tied
    logits can then produce different tokens and reconstructed actions. Compare
    native predictions, the export adapter and the loaded artifact using the same
    observations, precision and attention settings. Compare tokens through each
    sequence's EOS, then evaluate the reconstructed actions. Export repeats EOS in
    the remaining output positions.

Use the existing compression endpoint with the trained checkpoint, then load
the resulting directory through `CompressedPolicyRuntime` or the deployment
endpoint. Separate prefill/decode artifacts and sampled-token export are further
extensions. Checkpoint round-trip tests, quantized-kernel execution and
representative policy-quality evaluation establish different parts of deployment
compatibility; each model/backend combination requires its own evidence.

### Denoising policies

Flow and diffusion graphs accept noise tensors after their observations. The
runtime generates independent standard-normal inputs with the shapes recorded
in the export metadata. DDPM additionally receives a noise tensor for every
scheduled step. Export checks and numerical comparisons can reuse identical noise inputs
to compare the same sampling trajectory. Static PT2E calibration follows this
same input order.

!!! note "Conditional U-Net precision"

    `ConditionalActionUNet` uses mixed precision in the tested CPU paths. Eager
    INT8 weight-only conversion quantizes its timestep and conditioning linear
    weights. Static and dynamic x86 PT2E quantize those linear weights and their
    activation inputs. `Conv1d` and `ConvTranspose1d` weights remain FP32.
    Small flow, DDIM and DDPM U-Net policies pass complete prediction, export,
    save and reload with identical noise inputs. Convolution quantization,
    compiled-kernel coverage and trained-policy quality and speed require
    further validation.

Compression reports action-value differences for continuous outputs and token
disagreement for token outputs. Policy quality evaluation uses reconstructed
actions and representative observations or rollouts.

## Eager Quantization

[`EagerQuantizationWorkflow`][versatil.quantization.workflows.eager.EagerQuantizationWorkflow]
uses the torchao
[`quantize_()` API](https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.quantize_.html#torchao.quantization.quantize_)
before export. The same class supports eager PTQ and eager QAT.
VersatIL selects linear or embedding layers for this workflow. The base TorchAO
configuration specifies weight precision, activation precision and tensor
representation.

### Eager PTQ

When `is_qat: false`, quantization is applied only after training:

```python
torchao.quantization.quantize_(model, quantize_config)
```

Each target filters its `module_path` by `module_type`: `linear` selects
`nn.Linear` and `embedding` selects `nn.Embedding`. Embedding targets require
`IntxWeightOnlyConfig`. The root path selects the configured layer type throughout
the policy. Group size must divide the weight row width: `in_features` for
linears and `embedding_dim` for embeddings.

Targets can specify either `quantize_config` for direct conversion or
`schema` for preparation/conversion settings and calibration checks.
The `quantize_config` form creates a `DirectQuantizationSchema` internally
and retains its PTQ/QAT behavior. The target resolves layer names before
preparation; the workflow reuses those names for conversion.

### SmoothQuant

`SmoothQuantSchema` supplies configurations for TorchAO's SmoothQuant
observers and conversion. The eager workflow uses them to collect linear input
statistics, rescale input channels and weights, and apply
`Int8DynamicActivationInt8WeightConfig(version=2)`. W8A8 means
INT8 weights and INT8 activations at the selected linear operations; other policy
operations retain their existing precision. The converted weight stores the input
rescaling factors, so inference applies the same transformation.

Compress a trained policy with:

```bash
python -m versatil.endpoints.post_training_compress \
    --config-name end_to_end_ptq/smoothquant_int8 \
    checkpoint_path=/path/to/checkpoint \
    checkpoint_name=best.ckpt \
    calibration_steps=16
```

This preset selects linear layers under `decoder`, uses `alpha=0.5`, and writes
a Torch Export `.pt2` artifact through the `torch_inductor` deployment backend.
Compression and calibration use the existing CPU checkpoint-loading path.
The artifact requires PyTorch at inference; compilation is controlled by the
compressed runtime's existing settings.

Calibration uses the checkpoint's dataset, normalizer and tokenizer, with image
augmentation and batch shuffling disabled. `calibration_steps` counts observation
batches. Each batch runs the policy's full configured
prediction sequence: repeated denoising for flow/diffusion or generation for a
token decoder. All selected layers must execute before conversion; the error
identifies any layer missing calibration inputs.

The schema selects TorchAO's running-maximum observer. Each observed input updates
one absolute maximum per input channel, so inputs with different token lengths
contribute to the same statistics. The observer stores one maximum per selected
input channel throughout calibration.

Small flow, DDIM and DDPM regression tests cover SmoothQuant conversion, export
and reload. A separate binned-token decoder test covers SmoothQuant calibration,
conversion and generation. The binned-token artifact route is described under
[Policy Artifacts](#policy-artifacts). Evaluate trained-policy accuracy and
inference speed with representative observations.

CPU Inductor execution is tested by comparing the converted graph and its reloaded
artifact under the same compiler settings. Compare compiled predictions against
the original policy separately: small floating-point differences can change INT8
rounding and accumulate across denoising steps. Evaluate the resulting action
drift against the policy's accuracy requirements.

The SmoothQuant schema supports PTQ with dynamic W8A8. The Torch Export/Inductor
deployment backend accepts this representation.
TorchAO also provides static W8A8 configurations. Static conversion with the running
observer requires an upstream fix: the observer expects a dictionary, while the
base configuration supplies a typed parameter object.
See the tagged [TorchAO SmoothQuant implementation](https://github.com/pytorch/ao/blob/v0.18.0/torchao/prototype/smoothquant/api.py).

| Conversion procedure | Representative calibration | Training |
|---|---|---|
| Direct module PTQ | No | No |
| SmoothQuant module PTQ | Yes, full policy predictions | No |
| Existing module QAT | No separate calibration pass | Fake-quantized training before conversion |
| Static PT2E | Yes, prepared graph execution | No |
| Dynamic PT2E | No representative calibration | No |

### Eager Quantization-Aware-Training (QAT)
QAT trains the policy with fake-quantization layers that mimic inference conditions.
When `is_qat: true`, the workflow stores the same base torchao PTQ config but wraps
it in a `QATConfig`:

- Training calls `prepare_model()`, which applies
  `QATConfig(base_config=quantize_config, step="prepare")` to eligible
  linear or embedding modules selected by the workflow targets.
- Post-training compression restores the prepared layers and checkpoint weights,
  then applies
  `QATConfig(base_config=quantize_config, step="convert")`.

The same schema interface supplies these two QAT configurations. Each QAT schema
requires a defined fake-quantization procedure and matching conversion.

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
VersatIL's PT2E backend configurations provide INT8 linear and supported
convolution quantization. Their backend quantizers determine the accepted
operator patterns.

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
      module_path: ""  # Root policy
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
      module_path: ""  # Root policy
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
      module_path: ""  # Root policy
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
      module_path: ""  # Root policy
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
- Quantization target module paths must exist in the policy and select disjoint
  module scopes.
- PT2E backend and deployment backend must be compatible: X86 Inductor writes
  `.pt2`, while XNNPACK writes ExecuTorch `.pte`.
- QAT uses the eager workflow's matched preparation and conversion configurations.

## Relation To PTC

PTC resolves the configured workflow, calls `workflow.quantize()`, then passes
the returned `QuantizedContext` to the selected deployment backend. See
[Post-Training Compression](post_training_compression.md) for pruning, artifact
serialization, reports, and compressed runtime loading.
