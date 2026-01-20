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

from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.algorithm.behavior_cloning import BehavioralCloning
from versatil.models.decoding.algorithm.diffusion import Diffusion
from versatil.models.decoding.algorithm.flow_matching import FlowMatching
from versatil.models.decoding.algorithm.variational import VariationalAlgorithm

__all__ = [
    "DecodingAlgorithm",
    "BehavioralCloning",
    "Diffusion",
    "FlowMatching",
    "VariationalAlgorithm",
]
