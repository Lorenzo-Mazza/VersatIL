# Training with Quantization Awareness

Quantization-aware training (QAT) trains the policy with fake-quantization
layers that mimic the numeric behavior of a quantized deployment, so the
weights learn to be robust to quantization noise. The result is a regular
checkpoint that converts to a quantized artifact with much less accuracy loss
than quantizing a float policy after the fact. For the background and the
workflow contract, see the [quantization architecture](../architecture/quantization.md).

QAT in VersatIL uses torchao's eager `QATConfig` for `nn.Linear` and
`nn.Embedding` layers. Each target selects one layer type. The
`auto_filter_incompatible_linears` setting filters both types by weight row width.

## Step 1: Pick a Quantization Scheme

The `/quantization` config group ships ready-to-compose presets:

| Preset | Scheme |
|--------|--------|
| `qat_int8_dynamic_intx_int4` | int8 dynamic activations, int4 weights (group size 32) |
| `qat_int4_weight_only` | int4 weight-only |
| `qat_int2_weight_only` | int2 weight-only |
| `qat_int4_embeddings_and_linears` | W4A8 linears and INT4 weight-only embeddings, group size 32 |

Add one to the defaults list of any end-to-end training config:

```yaml
defaults:
  - /end_to_end_training_runs/libero_lerobot/bcat_language
  - /quantization: qat_int8_dynamic_intx_int4
  - _self_
```

Ready-made examples live under
`end_to_end_training_runs/libero_lerobot/qat/`.

The presets are starting points, but the underlying torchao
configs are parametric, so any supported scheme composes the same way. For
example, `Int8DynamicActivationIntxWeightConfig` accepts any integer weight
width; int2 weights with a different group size is one override away:

```yaml
quantization:
  targets:
    - _target_: versatil.quantization.module_target.EagerQuantizationModuleTarget
      module_path: ""
      quantize_config:
        _target_: torchao.quantization.Int8DynamicActivationIntxWeightConfig
        weight_dtype: ${torch_dtype:int2}
        weight_granularity:
          _target_: torchao.quantization.PerGroup
          group_size: 64
  is_qat: true
  auto_filter_incompatible_linears: true
```

## Step 2: Train

```bash
python -m versatil.endpoints.train \
    --config-name end_to_end_training_runs/libero_lerobot/qat/bcat_language_qat_int8_dynamic_int4
```

Before the first optimizer step, the workflow applies
`QATConfig(base_config=..., step="prepare")` to the eligible layers of
the configured targets. Training then proceeds normally: the fake-quant
layers quantize and dequantize on the fly during the forward pass, gradients
flow through, and the checkpoint stores the QAT-prepared weights.

Everything else about the run (logging, callbacks, checkpointing, resuming)
behaves exactly like a float training run.

## Step 3: Convert for Deployment

A QAT checkpoint still contains fake-quant modules; converting it into a real
quantized artifact happens through post-training compression with
`quantization.is_qat=true`. See
[PTQ after QAT](post_training_compression.md#converting-a-qat-checkpoint)
in the compression tutorial.

## Limitations

- QAT uses the eager workflow. Each selected layer needs a matching TorchAO
  preparation and conversion configuration.
- Conversion uses the same quantization configuration saved during training.
- Embedding QAT applies fake quantization through the module's forward lookup.
  Select embedding modules used through this operation; direct `.weight` reads
  bypass fake quantization.
