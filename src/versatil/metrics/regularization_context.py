"""Batch-local graph context objects used by policy regularizers."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum

import torch


class PolicyGraphInputDomain(StrEnum):
    """Policy graph boundary names where tensors can be replaced."""

    OBSERVATION = "observation"
    ENCODED_FEATURES = "encoded_features"
    DECODER_FEATURES = "decoder_features"


@dataclass(frozen=True)
class PolicyForwardContext:
    """Intermediate tensors from one normalized policy training forward pass.

    Attributes:
        observation: Normalized raw observation tensors after metadata-only keys
            have been stripped. Values are batched as ``(B, ...)``.
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

    observation: dict[str, torch.Tensor]
    encoded_features: dict[str, torch.Tensor]
    decoder_features: dict[str, torch.Tensor]
    predictions: dict[str, torch.Tensor]
    actions: dict[str, torch.Tensor] | None


@dataclass(frozen=True)
class PolicyRegularizationGraph:
    """Policy-owned re-entry interface for regularizer forwards.

    A regularizer should not call the encoding pipeline, algorithm, or decoder
    directly. This object exposes the exact re-entry callback built by
    ``Policy`` for the current batch, so replacements follow the same operation
    order as normal training.

    Attributes:
        context: Cached tensors from the original forward pass.
        training: Training mode of the policy when this graph was built.
        default_output_keys: Prediction keys used by the main loss. Regularizers
            use these when no explicit output keys are configured.
        evaluate_with_replacements: Callback that evaluates the policy graph from
            a named input domain after replacing selected tensors. Replacement
            tensors must match the shapes of tensors in ``context``.
        deterministic_scope: Context-manager factory used to disable stochastic
            decoder behavior during perturbed graph evaluations.
    """

    context: PolicyForwardContext
    training: bool
    default_output_keys: list[str]
    evaluate_with_replacements: Callable[
        [str, PolicyForwardContext, dict[str, torch.Tensor]],
        dict[str, torch.Tensor],
    ]
    deterministic_scope: Callable[[bool], AbstractContextManager[None]]

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
