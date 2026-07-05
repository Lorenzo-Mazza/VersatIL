# Synthetic Presets

Reusable synthetic task presets for Hydra composition.

These presets select the synthetic dataset schema, observation space, image
encoder, task length, and task-specific latent dimensionality used by retained
synthetic ablation configs.

Use them from another config with:

```yaml
defaults:
  - /synthetic_presets: sequential
```

Do not put model or loss choices here. Model definitions belong under
`hydra_configs/end_to_end_training_runs/synthetic/`.
