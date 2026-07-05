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

[![CI](https://github.com/Lorenzo-Mazza/VersatIL/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Lorenzo-Mazza/VersatIL/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Lorenzo-Mazza/VersatIL/branch/main/graph/badge.svg)](https://codecov.io/gh/Lorenzo-Mazza/VersatIL)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Python 3.13/3.14](https://img.shields.io/badge/python-3.13%20%7C%203.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/versatil.svg)](https://pypi.org/project/versatil/)

</div>

---

VersatIL is a composable PyTorch framework for training robot policies through Imitation Learning. It decouples **Data**, **Algorithm**, **Architecture**, and **Loss** so you can build, benchmark, and deploy any policy from config alone.

The key features are:

- **Composable**: A policy is assembled from an encoding pipeline, an algorithm, an action decoder, and a loss module. Swap any component with a config change.
- **Any encoder**: Use any vision backbone from [timm](https://github.com/huggingface/pytorch-image-models), any language model from [HF Transformers](https://github.com/huggingface/transformers), or custom geometric encoders for depth -- fuse them with attention, MLP, or concatenation.
- **Any algorithm**: [`BehavioralCloning`][versatil.models.decoding.algorithm.behavior_cloning.BehavioralCloning], [`Diffusion`][versatil.models.decoding.algorithm.diffusion.Diffusion], [`FlowMatching`][versatil.models.decoding.algorithm.flow_matching.FlowMatching], and [`VariationalAlgorithm`][versatil.models.decoding.algorithm.variational.VariationalAlgorithm] with pluggable prior-posterior schemes.
- **Action decoders**: [`ACT`][versatil.models.decoding.decoders.factory.act.ACT], DiT, GPT, [`LACT`][versatil.models.decoding.decoders.factory.lact.LACT], MoDE-ACT, [`PhaseACT`][versatil.models.decoding.decoders.factory.phase_act.PhaseACT], [`AutoregressiveVLADecoder`][versatil.models.decoding.decoders.factory.autoregressive_vla.AutoregressiveVLADecoder], [`OpenVLAOFTDecoder`][versatil.models.decoding.decoders.factory.openvla_oft.OpenVLAOFTDecoder], [`Pi0Decoder`][versatil.models.decoding.decoders.factory.pi0.Pi0Decoder], and [`SmolVLADecoder`][versatil.models.decoding.decoders.factory.smolvla.SmolVLADecoder].
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

    [`Policy`][versatil.models.policy.Policy] = [`EncodingPipeline`][versatil.models.encoding.pipeline.EncodingPipeline] + Algorithm + [`ActionDecoder`][versatil.models.decoding.decoders.base.ActionDecoder] + Loss.

    [:octicons-arrow-right-24: Overview](architecture/overview.md)

-   **Encoding Pipeline**

    ---

    Multi-modal encoders (RGB, depth, language, VLM) and fusion modules.

    [:octicons-arrow-right-24: Encoders](architecture/encoding.md)

-   **Algorithms**

    ---

    BC, [`Diffusion`][versatil.models.decoding.algorithm.diffusion.Diffusion], [`FlowMatching`][versatil.models.decoding.algorithm.flow_matching.FlowMatching], and [`VariationalAlgorithm`][versatil.models.decoding.algorithm.variational.VariationalAlgorithm].

    [:octicons-arrow-right-24: Algorithms](architecture/algorithms.md)

-   **Action Decoders**

    ---

    Decoder architectures: [`ACT`][versatil.models.decoding.decoders.factory.act.ACT], DiT, GPT, [`AutoregressiveVLADecoder`][versatil.models.decoding.decoders.factory.autoregressive_vla.AutoregressiveVLADecoder], [`OpenVLAOFTDecoder`][versatil.models.decoding.decoders.factory.openvla_oft.OpenVLAOFTDecoder], [`Pi0Decoder`][versatil.models.decoding.decoders.factory.pi0.Pi0Decoder], and [`SmolVLADecoder`][versatil.models.decoding.decoders.factory.smolvla.SmolVLADecoder].

    [:octicons-arrow-right-24: Decoders](architecture/decoders.md)

-   **Data Pipeline**

    ---

    Zarr ingestion, normalization, augmentation, tokenization.

    [:octicons-arrow-right-24: Data](architecture/data.md)

-   **Inference**

    ---

    Transport protocols, preprocessing, temporal aggregation.

    [:octicons-arrow-right-24: Inference](architecture/inference.md)

-   **Explainability**

    ---

    Grad-CAM, Grad-CAM++, and Ablation-CAM heatmaps for any trained policy.

    [:octicons-arrow-right-24: Explainability](architecture/explainability.md)

-   **Quantization**

    ---

    Quantization-aware training and eager/PT2E workflows via torchao.

    [:octicons-arrow-right-24: Quantization](architecture/quantization.md)

-   **Post-Training Compression**

    ---

    Pruning, quantization, and export for compressed inference.

    [:octicons-arrow-right-24: Compression](architecture/post_training_compression.md)

</div>

## Reference

<div class="grid cards" markdown>

-   **API Reference**

    ---

    Auto-generated from source code docstrings.

    [:octicons-arrow-right-24: API](reference/)

-   **Changelog**

    ---

    Release history and notable changes.

    [:octicons-arrow-right-24: Changelog](changelog.md)

</div>
