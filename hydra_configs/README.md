# Hydra Configs + OmegaConf Validation

We use Hydra to manage configurations for training and inference experiments.
[Hydra](https://hydra.cc/) manages the launch of experiment configurations. This enables users to easily modify configurations via command-line arguments, do hyperparameter sweeps, and organize experiments hierarchically.

We use [OmegaConf](https://omegaconf.readthedocs.io/) to validate YAML configs against Python dataclasses in `src/versatil/configs/`. The main benefit of this approach is catching configuration errors early, before launching expensive training runs. This also allows users to write incomplete configs that inherit from base configs, the default values of which are defined in the dataclasses. All base configs to inherit from are registered as nodes in the OmegaConf store in `src/versatil/configs/__init__.py`. These nodes are the ones which we pass as defaults in the Hydra config files.

## Usage

```bash
python -m versatil.endpoints.train --config-name end_to_end_training_runs/bowel_retraction/act
```

## Structure

- `end_to_end_training_runs/` - Complete experiment training runs
- `experiment/`, `training/`, `inference/` - Base configs
- `task/` - Dataset schema definition, dataloader setup, task (action/observation spaces) definition
- `policy/` - Encoding pipeline, decoder, algorithm, loss components

## Environment Variables

Paths configured via `.env` (see `.env.example`).