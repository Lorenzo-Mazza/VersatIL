"""Vector quantization modules for discrete latent variable models."""

from versatil.models.decoding.latent.vq.euclidean_codebook import EuclideanCodebook
from versatil.models.decoding.latent.vq.residual_vq import ResidualVQ
from versatil.models.decoding.latent.vq.vector_quantize import VectorQuantize

__all__ = ["EuclideanCodebook", "VectorQuantize", "ResidualVQ"]
