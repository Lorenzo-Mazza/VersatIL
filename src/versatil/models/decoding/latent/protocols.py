"""Structural protocols for latent variable component wiring.

Uses ``@runtime_checkable`` so priors don't need to import or inherit
this protocol — they just need to implement ``wire_posterior``.
"""

from typing import Protocol, runtime_checkable

from versatil.models.decoding.latent.posterior.base_posterior import (
    PosteriorLatentEncoder,
)


@runtime_checkable
class RequiresPosteriorWiring(Protocol):
    """Prior that needs a reference from the posterior encoder at init time.

    The VariationalAlgorithm calls ``wire_posterior`` after constructing
    both posterior and prior, allowing the prior to extract whatever
    shared state it needs (encoder reference, codebook, etc.).
    """

    def wire_posterior(self, posterior: PosteriorLatentEncoder) -> None:
        """Receive the constructed posterior encoder for shared-state extraction."""
        ...
