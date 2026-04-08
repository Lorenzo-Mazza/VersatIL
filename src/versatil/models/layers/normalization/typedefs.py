"""Type definitions for the normalization package."""

from versatil.models.layers.normalization.ada_norm import AdaNorm
from versatil.models.layers.normalization.unconditioned_norm import UnconditionedNorm

BlockNormalization = AdaNorm | UnconditionedNorm
