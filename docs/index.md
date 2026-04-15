---
hide:
  - toc
---

<style>
  .md-typeset h1 { display: none; }
</style>

<div align="center" markdown>

![VersatIL Logo](media/VersatIL_transparent.png){ width="500" }

_Imitation Learning for Any Robot Policy._

[![pipeline status](https://gitlab.com/nct_tso_public/versatil/badges/main/pipeline.svg)](https://gitlab.com/nct_tso_public/versatil/-/commits/main)
[![coverage report](https://gitlab.com/nct_tso_public/versatil/badges/main/coverage.svg)](https://gitlab.com/nct_tso_public/versatil/-/commits/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/versatil.svg)](https://pypi.org/project/versatil/)

</div>

---

VersatIL is a composable PyTorch framework for training robot policies through Imitation Learning. It decouples **Data**, **Algorithm**, and **Architecture** so you can build, benchmark, and deploy any policy from config alone.

The key features are:

- **Composable**: A policy is assembled from an encoding pipeline, an algorithm, an action decoder, and a loss module. Swap any component with a config change.
- **Any encoder**: Use any vision backbone from [timm](https://github.com/huggingface/pytorch-image-models), any language model from [HF Transformers](https://github.com/huggingface/transformers), or custom geometric encoders for depth -- fuse them with attention, MLP, or concatenation.
- **Any algorithm**: Behavioral Cloning, Diffusion, Flow Matching, and a compositional Variational wrapper with pluggable prior-posterior schemes.
- **12 action decoders**: ACT, DiT, GPT, Free Transformer, LACT, MoDE-ACT, Phase-ACT, and more -- each with configurable positional encodings, normalization, and action heads.
- **Any dataset**: Ingest CSV, HDF5, or LeRobot formats into Zarr. Define observation/action spaces per task. Normalize, augment, and tokenize automatically.
- **Config-driven**: [Hydra](https://hydra.cc/) + typed [OmegaConf](https://omegaconf.readthedocs.io/) dataclasses. Every experiment is fully reproducible from a single YAML file. Errors are caught at startup, not mid-training.
- **Inference-ready**: Pluggable transport protocols (ZMQ), observation buffering, temporal aggregation, structured actions. One client for simulation and hardware. Post-training compression (pruning + quantization via [torchao](https://github.com/pytorch/ao)) for deployment on resource-constrained hardware.
- **Tested**: >90% test coverage. Ruff formatting and linting enforced via CI/CD and pre-commit hooks.

---

## Getting Started

<div class="grid cards" markdown>

-   **Installation**

    ---

    Set up Python, CUDA, and VersatIL with conda + uv.

    [:octicons-arrow-right-24: Install](getting-started/installation.md)

-   **First Training Run**

    ---

    Train a policy, override configs from CLI, resume from checkpoints.

    [:octicons-arrow-right-24: Train](getting-started/training.md)

-   **Configuration**

    ---

    Hydra composition, OmegaConf typed configs, interpolation.

    [:octicons-arrow-right-24: Configure](getting-started/configuration.md)

</div>

## Architecture

<div class="grid cards" markdown>

-   **Overview**

    ---

    Policy = EncodingPipeline + Algorithm + ActionDecoder + Loss.

    [:octicons-arrow-right-24: Overview](architecture/overview.md)

-   **Encoding Pipeline**

    ---

    Multi-modal encoders (RGB, depth, language, VLM) and fusion modules.

    [:octicons-arrow-right-24: Encoders](architecture/encoding.md)

-   **Algorithms**

    ---

    BC, Diffusion, Flow Matching, and VariationalAlgorithm.

    [:octicons-arrow-right-24: Algorithms](architecture/algorithms.md)

-   **Action Decoders**

    ---

    12 decoder architectures: ACT, DiT, GPT, Free Transformer, and more.

    [:octicons-arrow-right-24: Decoders](architecture/decoders.md)

-   **Data Pipeline**

    ---

    Zarr ingestion, normalization, augmentation, tokenization.

    [:octicons-arrow-right-24: Data](architecture/data.md)

-   **Inference**

    ---

    Transport protocols, preprocessing, temporal aggregation.

    [:octicons-arrow-right-24: Inference](architecture/inference.md)

-   **Post-Training Compression**

    ---

    Pruning, quantization (PT2E + quantize_() via torchao), compressed inference.

    [:octicons-arrow-right-24: Compression](architecture/post_training_compression.md)

</div>

## Reference

<div class="grid cards" markdown>

-   **API Reference**

    ---

    Auto-generated from source code docstrings.

    [:octicons-arrow-right-24: API](reference/SUMMARY.md)

-   **Changelog**

    ---

    Release history and notable changes.

    [:octicons-arrow-right-24: Changelog](changelog.md)

</div>
