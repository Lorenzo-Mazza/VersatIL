"""Select a policy export adapter from its algorithm and decoder."""

from versatil.models.decoding.algorithm.diffusion import Diffusion
from versatil.models.decoding.algorithm.flow_matching import FlowMatching
from versatil.models.exportable.autoregressive import ExportableTokenPolicy
from versatil.models.exportable.base import ExportablePolicy
from versatil.models.exportable.denoising import ExportableDenoisingPolicy
from versatil.models.policy import Policy


def create_exportable_policy(policy: Policy) -> ExportablePolicy:
    """Create the export adapter matching the policy's prediction procedure.

    Args:
        policy: Initialized policy with its action tokenizer attached when required.

    Returns:
        Adapter for bounded autoregressive generation, explicit-noise denoising,
        or continuous-action prediction.

    Raises:
        ValueError: If token generation requires an unsupported algorithm, decoder,
            tokenizer or generation configuration.
    """
    if policy.decoder.requires_tokenized_actions:
        return ExportableTokenPolicy.from_policy(policy=policy)
    if isinstance(policy.algorithm, (Diffusion, FlowMatching)):
        return ExportableDenoisingPolicy.from_policy(policy=policy)
    return ExportablePolicy.from_policy(policy=policy)
