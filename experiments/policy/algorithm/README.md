# Algorithm Configurations

This directory contains Hydra configuration files for decoding algorithms.

## New Compositional Pattern (Recommended)

The new `VariationalAlgorithm` wrapper enables compositional design:

```yaml
_target_: refactoring.models.decoding.algorithm.variational.VariationalAlgorithm

base_algorithm:
  _target_: <any algorithm>  # BC, FlowMatching, Diffusion, etc.

posterior_encoder:
  _target_: <latent encoder>  # VAE, etc.

prior: <prior config or null>  # GaussianPrior (null), DiffusionPrior, etc.
```

## Available Configurations

### Pure Algorithms (Deterministic)

- **`behavioral_cloning.yaml`**: Pure BC without variational inference
  - Simple supervised learning
  - Deterministic, uni-modal predictions

### Variational Algorithms (Multi-modal)

- **`bc_with_vae_gaussian.yaml`**: BC + VAE + Gaussian prior
  - Replaces old `behavioral_cloning_vae.yaml`
  - Multi-modal action prediction
  - Simple N(0,I) prior

- **`bc_with_learned_prior.yaml`**: BC + VAE + Diffusion prior (NEW)
  - Multi-modal with learned prior p(z|s)
  - More expressive than Gaussian prior

- **`variational_diffusion.yaml`**: Diffusion + VAE + Diffusion prior (NEW)
  - Combines diffusion denoising with variational inference
  - Two-level hierarchy: latent diffusion + action diffusion

### Deprecated (Backward Compatibility)

- **`behavioral_cloning_vae.yaml`**: OLD API (deprecated)
  - Use `bc_with_vae_gaussian.yaml` instead
  - Still works for backward compatibility

## Migration Guide

### Old API:
```yaml
_target_: refactoring.models.decoding.algorithm.behavior_cloning.BehavioralCloning

latent_encoder:
  _target_: refactoring.models.decoding.latent.vae.VAETransformerEncoder
  ...
```

### New API:
```yaml
_target_: refactoring.models.decoding.algorithm.variational.VariationalAlgorithm

base_algorithm:
  _target_: refactoring.models.decoding.algorithm.behavior_cloning.BehavioralCloning

posterior_encoder:
  _target_: refactoring.models.decoding.latent.vae.VAETransformerEncoder
  ...

prior: null  # Auto-creates GaussianPrior
```

## Benefits of New Pattern

1. **Compositionality**: Mix any algorithm with any latent encoder and prior
2. **Code Reuse**: No need for algorithm-specific variational implementations
3. **New Combinations**: Enable previously impossible combinations (e.g., Variational Diffusion)
4. **Clearer Design**: Separates algorithm logic from variational inference
