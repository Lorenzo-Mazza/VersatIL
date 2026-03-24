# Post-Training Compression

VersatIL's post-training compression (PTC) pipeline reduces trained policy model size and improves CPU inference efficiency for deployment on edge or resource-constrained hardware, without retraining. It supports preparation (BatchNorm fusion), pruning (structured and unstructured), and quantization (PT2E static/dynamic and torchao quantize_() API).

## Architecture

The PTC pipeline spans two packages:

- **`post_training_compression/`**: Pipeline orchestration, model preparation, pruning, export, serialization, and reporting.
- **`quantization/`**: Bridge to [torchao](https://github.com/pytorch/ao) — quantization strategies, calibration, hardware backends, and patches for torch/torchao compatibility.

### Pipeline Flow

```
PostTrainingCompressor.compress()
│
├── _load_policy()                Load from training checkpoint (CPU)
├── resolve_modules()             Per-module targets or global fallback
├── validate()                    Check module paths + strategy compatibility
│
├── _prepare_and_prune()          Per target:
│   ├── prepare_batchnorms()        Replace FrozenBN → standard BN
│   ├── fuse_conv_batchnorm()       Fold BN weights into Conv2d
│   └── pruner.prune() × N          Apply pruners sequentially
│
├── _export_and_quantize()
│   ├── apply_quantize_api()        (if quantize_() targets) eager-mode quantization
│   ├── build_example_inputs()      From ObservationSpace + DataLoaderConfig metadata
│   ├── export_policy()             torch.export with dynamic batch dim
│   └── apply_pt2e_quantization()   (if PT2E targets) prepare → calibrate → convert
│
├── _resolve_output_directory()   Timestamped: compressed/<YYYYMMDD_HHMMSS>/
├── save_compressed_model()       .pt2 archive + normalizer + metadata + tokenizer
│
└── (optional) QuantizationReport   Operator coverage, size reduction, output divergence
```

### Key Classes

| Class | Module | Role |
|-------|--------|------|
| `PostTrainingCompressor` | `compressor.py` | Pipeline orchestrator. Owns `compress()` and all phase methods. |
| `CompressionTarget` | `compression_target.py` | Per-module config: module_path + preparation + pruning list + quantization strategy. |
| `PT2EStrategy` | `quantization/strategies.py` | Wraps a `BasePT2EBackend` for graph-based quantization. |
| `QuantizeApiStrategy` | `quantization/strategies.py` | Wraps a torchao quantization config for eager-mode quantization. |
| `X86InductorBackend` | `quantization/backends/x86_inductor.py` | Creates X86InductorQuantizer, manages inductor env vars, provides lowering. |
| `CalibrationDataProvider` | `quantization/calibration.py` | Yields observation tuples from the training dataloader for static calibration. |
| `ExportablePolicy` | `models/exportable_policy.py` | Wraps Policy with positional tensor I/O for torch.export compatibility. |
| `CompressedPolicyLoader` | `inference/policy_loading/compressed_loader.py` | Loads .pt2 archives, applies torch.compile, runs compiled inference. |

### Quantization Paths

**PT2E (PyTorch 2 Export)**

The graph-based path. The policy is exported to an FX graph via `torch.export`, then quantized using hardware-specific quantizers:

1. `export_policy()` — Runs an eager forward pass to materialize lazy modules, then exports with dynamic batch dimension.
2. `prepare_pt2e()` — Inserts observer modules into the FX graph at quantization points.
3. Calibration — Forward passes through training data to collect activation statistics.
4. `convert_pt2e()` — Replaces observers with quantized operations.

Supports per-module targeting via `ComposableQuantizer` — different modules can have different quantization configs (e.g., quantize vision backbones but skip the language encoder).

**quantize_() API**

The eager-mode path. Applies dynamic or weight-only quantization directly on the `nn.Module` before export:

```python
torchao.quantization.quantize_(model, Int8DynamicActivationInt8WeightConfig())
```

Simpler but less granular than PT2E. Only supports `nn.Linear` layers. The two paths cannot be combined in a single compression run.

### Pruning

Pruning is specified as a list of `BasePruner` instances, applied sequentially. This allows composing different strategies:

- **`UnstructuredPruner`**: Global L1 magnitude pruning. Zeros the lowest-magnitude weights across all targeted layers. Targets modules where `weight` is an `nn.Parameter` (not just `hasattr`).
- **`StructuredPruner`**: Per-layer Lp-norm channel pruning along a specified dimension. Defaults to targeting Conv1d, Conv2d, and Linear layers.

Both pruners remove the pruning reparametrization after application (weights become permanent zeros).

### Preparation

Pre-quantization model surgery to make the model quantization-friendly:

- **`prepare_batchnorms_for_quantization()`**: Replaces non-standard BatchNorm variants (FrozenBatchNorm2d, etc.) with standard `nn.BatchNorm2d` in eval mode with tracking disabled. Creates replacement modules on the same device as the original.
- **`fuse_all_conv_batchnorm_pairs()`**: Folds consecutive Conv2d + BatchNorm2d pairs into a single Conv2d with adjusted weights and bias. Creates the fused Conv2d on the same device as the input Conv2d. Replaces the BatchNorm with `nn.Identity` (or preserves a fused activation like ReLU).

### Compressed Checkpoints

A compressed checkpoint directory contains:

```
compressed/<timestamp>/
├── compressed_policy.pt2          # torch.export.save() archive
├── normalizer.pt                  # LinearNormalizer state_dict
├── compression_metadata.json      # Keys, versions, strategy, training path
├── quantization_config.yaml       # Hydra config used for compression
├── config.yaml                    # Training config (copied from source)
└── tokenizer/                     # Tokenizer files (copied from source)
```

`CompressedPolicyLoader` reads the metadata to determine the quantization strategy, loads the backend, and applies `torch.compile` with the backend's environment (e.g., `TORCHINDUCTOR_FREEZING=1`, `cpp_wrapper=True` for x86). The environment is activated permanently because `torch.compile` is lazy — actual kernel compilation happens on the first forward pass.

### Device Constraints

- **Export and calibration run on CPU.** `torch.export` bakes device metadata (`_to_copy(device='cpu')`, `_assert_tensor_metadata(device='cpu')`) into the FX graph. Moving the exported model to CUDA causes runtime device mismatches. This is a PyTorch limitation.

### Hydra Configuration

PTC uses its own Hydra config hierarchy under `hydra_configs/end_to_end_ptq/`:

```yaml
# Top-level fields serve as defaults for per-module targets
checkpoint_path: ???
device: cpu                     # Policy loading device
calibration_steps: 16           # Batches for static quantization
generate_report: false          # Optional size/speed/divergence report

preparation:
  replace_frozen_batchnorm: true
  fuse_conv_batchnorm: true

pruning:                        # List of pruners (composable)
  - _target_: versatil.post_training_compression.pruning.UnstructuredPruner
    amount: 0.5

quantization:                   # Global quantization strategy
  _target_: versatil.quantization.strategies.PT2EStrategy
  pt2e_backend:
    _target_: versatil.quantization.backends.x86_inductor.X86InductorBackend
    is_dynamic: false

modules: []                     # Empty = use global settings on root
```

When `modules` is non-empty, each entry targets a specific submodule and can override preparation, pruning, and quantization independently.

### Supported PT2E Backends

**PT2E backends** are hardware-specific quantizer configurations for the graph-based quantization path. Each backend provides a quantizer factory, environment setup, and operator lowering.

Currently supported:
- **X86InductorBackend**: Targets x86 CPUs via `X86InductorQuantizer` with Inductor operator fusion and lowering. Supports static, dynamic, and QAT quantization modes.

Additional torchao-supported backends (e.g., ARM, XNNPack, CUDA) can be added by implementing `BasePT2EBackend`. Each backend needs:
- `create_quantizer()`: Returns a hardware-specific torchao `Quantizer`.
- `environment_context()` / `activate_environment()`: Backend-specific env vars and config (e.g., `TORCHINDUCTOR_FREEZING`, `cpp_wrapper`).
- `lower()`: Backend-specific operator fusion and lowering pass.

**quantize_() API** is backend-agnostic — it operates on the eager model and supports any device, including CUDA. However, some torchao quantization configs have known batch size constraints on CUDA (see [pytorch/ao#2376](https://github.com/pytorch/ao/issues/2376)).