"""Action-decoding algorithms for imitation learning.

This package provides composable algorithms that define how policies learn and predict:
- BehavioralCloning: Direct supervised learning from demonstrations
- Diffusion: Iterative denoising for action generation
- FlowMatching: Continuous normalizing flows
- VariationalAlgorithm: Compositional wrapper adding variational inference to any algorithm

Shared Components:
    The diffusion_process module provides reusable building blocks for diffusion-based
    algorithms, including scheduler configuration, noise addition, and timestep handling.
    These components are used by both the Diffusion algorithm and DiffusionPrior.
"""

from refactoring.models.decoding.algorithm.base import DecodingAlgorithm
from refactoring.models.decoding.algorithm.behavior_cloning import BehavioralCloning
from refactoring.models.decoding.algorithm.action_diffusion import Diffusion
from refactoring.models.decoding.algorithm.diffusion_process import (
    DiffusionSchedulerConfig,
    add_noise_to_tensor,
    create_noise_scheduler,
    sample_random_timesteps,
    setup_inference_timesteps,
)
from refactoring.models.decoding.algorithm.flow_matching import FlowMatching
from refactoring.models.decoding.algorithm.variational import VariationalAlgorithm

__all__ = [
    "DecodingAlgorithm",
    "BehavioralCloning",
    "Diffusion",
    "DiffusionSchedulerConfig",
    "FlowMatching",
    "VariationalAlgorithm",
    "add_noise_to_tensor",
    "create_noise_scheduler",
    "sample_random_timesteps",
    "setup_inference_timesteps",
]
