"""Policy prediction objectives used by visual attribution methods."""

import torch

from versatil.data.constants import SampleKey
from versatil.explainability.attribution.policy import (
    EncoderCacheDisabled,
    default_output_selector,
    prepare_policy_observation_for_explanation,
)
from versatil.explainability.typedefs import (
    ActionBatch,
    ObservationBatch,
    PolicyPredictionSelector,
)
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.policy import Policy


def resolve_actions_for_explanation(
    policy: Policy,
    observation: ObservationBatch,
    actions: ActionBatch | None,
    preprocess_observation: bool,
) -> ActionBatch | None:
    """Resolve action targets needed by the policy explanation objective.

    Decoders with ``requires_tokenized_actions=True`` need action token IDs to
    compute a differentiable likelihood score. Offline dataset batches can
    provide true tokenized actions. Online inference batches have no labels, so
    this function runs an unhooked inference pass first and uses generated
    action tokens as pseudo-targets for the subsequent hooked teacher-forced
    pass.

    Args:
        policy: Policy being explained.
        observation: Observation values keyed by observation-space names.
        actions: Optional action dictionary from the explanation source.
        preprocess_observation: Whether to normalize/tokenize observations
            before building policy features.

    Returns:
        ``None`` or the original action batch for decoders that do not require
        tokenized actions, existing tokenized labels when available, or
        generated pseudo-target actions for tokenized-action decoders.

    Raises:
        RuntimeError: If a tokenized-action decoder cannot produce action
            tokens for an unlabeled explanation batch.
    """
    if not _decoder_requires_tokenized_actions(policy=policy):
        return actions
    if actions is not None and SampleKey.TOKENIZED_ACTIONS.value in actions:
        return actions

    prepared_observation = prepare_policy_observation_for_explanation(
        policy=policy,
        observation=observation,
        preprocess_observation=preprocess_observation,
    )
    features = policy._build_algorithm_features(observation=prepared_observation)
    with torch.no_grad():
        predictions = policy.algorithm.predict(
            features=features, network=policy.decoder
        )
    if DecoderOutputKey.PREDICTED_ACTION_TOKENS.value not in predictions:
        raise RuntimeError(
            "Tokenized-action explanation without action labels requires "
            f"'{DecoderOutputKey.PREDICTED_ACTION_TOKENS.value}' from policy inference."
        )
    return {
        SampleKey.TOKENIZED_ACTIONS.value: predictions[
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value
        ].detach()
    }


def compute_policy_explanation_objective(
    policy: Policy,
    observation: ObservationBatch,
    actions: ActionBatch | None,
    preprocess_observation: bool,
    output_selector: PolicyPredictionSelector | None = None,
) -> torch.Tensor:
    """Compute the differentiable policy score used by attribution.

    Args:
        policy: Policy being explained.
        observation: Observation values keyed by observation-space names.
        actions: Optional action dictionary. Decoders with
            ``requires_tokenized_actions=True`` require ``tokenized_actions``;
            call ``resolve_actions_for_explanation`` before registering
            attribution hooks when labels are absent.
        preprocess_observation: Whether to normalize/tokenize observations
            before building policy features.
        output_selector: Optional selector for continuous prediction tensors.
            ``None`` scores the norm of all normalized action components.

    Returns:
        Tensor of per-sample or per-token scores. Attribution methods average
        this tensor for gradient backpropagation and compare score drops for
        ablation.

    Raises:
        ValueError: If a custom selector is passed for a decoder that requires
            tokenized actions.
        RuntimeError: If tokenized-action logits or target tokens are missing.
    """
    prepared_observation = prepare_policy_observation_for_explanation(
        policy=policy,
        observation=observation,
        preprocess_observation=preprocess_observation,
    )
    features = policy._build_algorithm_features(observation=prepared_observation)
    if _decoder_requires_tokenized_actions(policy=policy):
        if output_selector is not None:
            raise ValueError(
                "output_selector is only supported when the decoder does not "
                "require tokenized actions."
            )
        return _compute_tokenized_action_log_likelihood(
            policy=policy,
            features=features,
            actions=actions,
        )

    with EncoderCacheDisabled(decoder=policy.decoder):
        predictions = policy.algorithm.predict(
            features=features,
            network=policy.decoder,
        )
    selector = (
        output_selector if output_selector is not None else default_output_selector
    )
    return selector(predictions)


def repeat_action_batch(
    actions: ActionBatch | None, repeat_count: int
) -> ActionBatch | None:
    """Repeat action tensors along the batch axis for perturbation methods.

    Args:
        actions: Action tensors keyed by action component, or ``None`` when the
            attribution objective does not need action labels.
        repeat_count: Number of copies to concatenate.

    Returns:
        Repeated action batch, or ``None`` when ``actions`` is ``None``.

    Raises:
        ValueError: If ``repeat_count`` is less than one.
    """
    if actions is None:
        return None
    if repeat_count < 1:
        raise ValueError(f"repeat_count must be positive. Got: {repeat_count}")
    return {
        key: torch.cat([value.clone() for _ in range(repeat_count)], dim=0)
        for key, value in actions.items()
    }


def _decoder_requires_tokenized_actions(policy: Policy) -> bool:
    """Return whether the decoder needs tokenized action targets for scoring.

    The flag is needed as tokenized-action decoders need teacher-forced token
        likelihoods.
    """
    return policy.decoder.requires_tokenized_actions


def _compute_tokenized_action_log_likelihood(
    policy: Policy,
    features: dict[str, torch.Tensor],
    actions: ActionBatch | None,
) -> torch.Tensor:
    """Score teacher-forced action tokens under the decoder logits.

    Args:
        policy: Policy whose algorithm and decoder produce token logits.
        features: Prepared policy features from the current observation.
        actions: Action batch containing ``tokenized_actions`` and optionally
            ``is_pad_action``.

    Returns:
        Mean log-likelihood per sample after ignoring padded action tokens.

    Raises:
        RuntimeError: If tokenized action targets are missing.
        RuntimeError: If the decoder does not return action logits.
        RuntimeError: If logits, targets, and padding masks have incompatible
            shapes.
    """
    if actions is None or SampleKey.TOKENIZED_ACTIONS.value not in actions:
        raise RuntimeError(
            "Tokenized-action explanation requires tokenized action targets. "
            "Call resolve_actions_for_explanation() before attribution."
        )
    predictions = policy.algorithm.forward(
        features=features,
        actions=actions,
        network=policy.decoder,
    )
    if DecoderOutputKey.ACTION_LOGITS.value not in predictions:
        raise RuntimeError(
            "Tokenized-action explanation requires teacher-forced predictions with "
            f"'{DecoderOutputKey.ACTION_LOGITS.value}'."
        )
    logits = predictions[DecoderOutputKey.ACTION_LOGITS.value]
    target_tokens = actions[SampleKey.TOKENIZED_ACTIONS.value].long()
    if logits.shape[:2] != target_tokens.shape:
        raise RuntimeError(
            f"Action logits leading shape {tuple(logits.shape[:2])} must match "
            f"tokenized action shape {tuple(target_tokens.shape)}."
        )

    token_scores = logits.log_softmax(dim=-1).gather(
        dim=-1,
        index=target_tokens.unsqueeze(-1),
    )
    token_scores = token_scores.squeeze(-1)
    padding_mask = actions.get(SampleKey.IS_PAD_ACTION.value)
    if padding_mask is None:
        return token_scores.mean(dim=1)
    if padding_mask.shape != target_tokens.shape:
        raise RuntimeError(
            f"Action padding mask shape {tuple(padding_mask.shape)} must match "
            f"tokenized action shape {tuple(target_tokens.shape)}."
        )
    valid_tokens = (~padding_mask).sum(dim=1).clamp_min(1)
    return token_scores.masked_fill(padding_mask, 0.0).sum(dim=1) / valid_tokens
