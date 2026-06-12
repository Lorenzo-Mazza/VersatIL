"""Batch-local graph context objects used by policy regularizers."""

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import torch

from versatil.common.tensor_ops import TensorTree
from versatil.data.metadata import ActionMetadata


class PolicyGraphInputDomain(StrEnum):
    """Policy graph boundary names where tensors can be replaced."""

    OBSERVATION = "observation"
    ENCODED_FEATURES = "encoded_features"
    DECODER_FEATURES = "decoder_features"


@dataclass(frozen=True)
class PolicyForwardContext:
    """Intermediate tensors from one normalized policy training forward pass.

    Attributes:
        observation: Normalized raw observations after metadata-only keys have
            been stripped. Tensor values are batched as ``(B, ...)``; metadata
            values such as raw language can remain non-tensor.
        encoded_features: Output tensors from the encoding pipeline. Values are
            batched as ``(B, ...)`` or ``(B, T, ...)`` depending on the encoder.
        decoder_features: Feature dictionary selected for the decoder contract,
            including decoder-requested raw observation tensors and encoded
            feature tensors. Values are batched as ``(B, ...)``.
        predictions: Output dictionary from ``algorithm.forward``. Values used by
            losses are batched as ``(B, ...)``.
        actions: Normalized training action dictionary, or ``None`` for
            action-free graph evaluations.
    """

    observation: dict[str, TensorTree]
    encoded_features: dict[str, torch.Tensor]
    decoder_features: dict[str, torch.Tensor]
    predictions: dict[str, torch.Tensor]
    actions: dict[str, torch.Tensor] | None


class PolicyGraphEvaluator(Protocol):
    """Re-runs the policy forward pass with selected tensors swapped out.

    ``Policy`` provides the implementation. A call executes the same
    encode -> select-features -> decode order as the original training
    forward, but starts from the boundary named by ``input_domain`` with the
    tensors in ``replacements`` substituted for the cached ones: observation
    replacements re-run encoding, encoded-feature replacements re-run feature
    selection and decoding, and decoder-feature replacements re-run only the
    algorithm and decoder. Every call replays the RNG snapshot captured when
    the graph was built, so stochastic sampling (algorithm timesteps, noise,
    dropout masks) is identical across calls and output differences reflect
    only the replaced tensors.
    """

    def __call__(
        self,
        input_domain: str,
        context: PolicyForwardContext,
        replacements: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return the prediction dictionary with ``replacements`` applied."""
        ...


class DeterministicScopeFactory(Protocol):
    """Builds a scope that silences decoder train-time stochasticity.

    With ``enabled=True`` the returned context manager runs decoder modules in
    eval mode (dropout off) for its duration and restores their training state
    on exit; with ``enabled=False`` it is a no-op.
    """

    def __call__(self, enabled: bool) -> AbstractContextManager[None]:
        """Return the context manager guarding perturbed graph evaluations."""
        ...


@dataclass(frozen=True)
class PolicyRegularizationGraph:
    """Policy-owned re-entry interface for regularizer forwards.

    A regularizer never calls the encoding pipeline, algorithm, or decoder
    directly. Instead, ``Policy`` packages everything a regularizer may touch
    for the current batch into this object: the tensors the original forward
    pass produced (``context``) and a callback that re-runs the policy on
    modified copies of those tensors (``evaluate_with_replacements``).

    A typical perturbation step looks like::

        features = graph.context.encoded_features
        perturbed = {"left_rgb": features["left_rgb"] + delta}
        predictions = graph.evaluate(
            input_domain=PolicyGraphInputDomain.ENCODED_FEATURES.value,
            context=graph.context,
            replacements=perturbed,
        )

    which decodes exactly as in training but with the perturbed ``left_rgb``
    tensor in place of the cached one. Because re-entries replay one RNG
    snapshot per batch, two calls with the same replacements return identical
    predictions even for stochastic algorithms such as flow matching.

    Attributes:
        context: Cached tensors from the original forward pass.
        training: Training mode of the policy when this graph was built.
        default_output_keys: Prediction keys used by the main loss. Regularizers
            use these when no explicit output keys are configured.
        evaluate_with_replacements: Policy-provided re-entry callback; call it
            through :meth:`evaluate`.
        deterministic_scope: Factory for the scope that runs decoder stochastic
            layers in eval mode during perturbed graph evaluations.
        action_metadata: Action-space metadata keyed by prediction name, for
            regularizers that need action semantics (e.g. position keys).
    """

    context: PolicyForwardContext
    training: bool
    default_output_keys: list[str]
    evaluate_with_replacements: PolicyGraphEvaluator
    deterministic_scope: DeterministicScopeFactory
    action_metadata: dict[str, ActionMetadata] = field(default_factory=dict)

    def evaluate(
        self,
        input_domain: str,
        context: PolicyForwardContext,
        replacements: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Evaluate the policy graph from a named domain with replacements.

        Args:
            input_domain: Boundary to re-enter. Valid values are ``"observation"``,
                ``"encoded_features"``, and ``"decoder_features"``.
            context: Forward context used as the base graph state.
            replacements: Tensors to replace at ``input_domain``. Each value must
                have the same shape as the matching tensor in ``context``.

        Returns:
            Prediction dictionary produced by the policy graph.
        """
        return self.evaluate_with_replacements(
            input_domain=input_domain,
            context=context,
            replacements=replacements,
        )
