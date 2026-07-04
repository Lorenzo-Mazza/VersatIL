"""Policy-facing helpers shared by attribution methods."""

from types import TracebackType

import torch

from versatil.data.processing.transform import (
    normalize_observation,
    tokenize_observation,
)
from versatil.explainability.typedefs import ObservationBatch
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.policy import Policy


class EncoderCacheDisabled:
    """Disable decoder encoder-prefix caches during attribution forwards.

    Some VLM decoders cache encoded visual prefixes for autoregressive
    inference. Attribution re-runs the policy with hooks and activation
    replacements, so reusing a cached prefix can detach the current visual
    target from the scored prediction. This context manager clears the cache,
    suppresses the cache toggles while attribution runs, then restores the
    decoder's prior cache state.
    """

    def __init__(self, decoder: ActionDecoder) -> None:
        """Store the decoder whose cache controls should be suppressed.

        Args:
            decoder: Policy decoder implementing the encoder-cache contract;
                decoders without a cache inherit the base no-op behavior.
        """
        self.decoder = decoder
        self._cache_was_enabled = False

    def __enter__(self) -> None:
        """Clear the cache and freeze its toggles."""
        self._cache_was_enabled = self.decoder.encoder_cache_enabled
        self.decoder.disable_encoder_cache()
        self.decoder.set_encoder_cache_suppressed(True)

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Unfreeze the toggles and restore the prior cache state.

        Args:
            exception_type: Exception class raised inside the context, if any.
            exception: Exception instance raised inside the context, if any.
            traceback: Traceback raised inside the context, if any.

        Returns:
            ``False`` so exceptions from the attribution forward pass propagate.
        """
        self.decoder.set_encoder_cache_suppressed(False)
        if self._cache_was_enabled:
            self.decoder.enable_encoder_cache()
        else:
            self.decoder.disable_encoder_cache()
        return False


def prepare_policy_observation_for_explanation(
    policy: Policy,
    observation: ObservationBatch,
    preprocess_observation: bool,
) -> dict[str, torch.Tensor]:
    """Prepare observations with the policy's inference preprocessing.

    Args:
        policy: Policy whose preprocessing stack should be reused.
        observation: Observation values keyed by observation-space names.
        preprocess_observation: Whether to apply the same normalization and
            tokenization path used by ``Policy.predict_action``. Use ``False``
            for batches already produced by ``EpisodicDataset``.

    Returns:
        Observation tensors ready for ``Policy._build_algorithm_features``.
    """
    if preprocess_observation:
        prepared_observation = normalize_observation(
            observation=observation,
            normalizer=policy.normalizer,
            observation_space=policy.observation_space,
        )
        prepared_observation = policy._strip_metadata_passthrough_observations(
            observation=prepared_observation
        )
        if (
            policy.tokenizer is not None
            and policy.tokenizer.observation_tokenizer is not None
        ):
            prepared_observation = tokenize_observation(
                observation=prepared_observation,
                obs_tokenizer=policy.tokenizer.observation_tokenizer,
                batched=True,
            )
    else:
        prepared_observation = policy._strip_metadata_passthrough_observations(
            observation=observation
        )
    return prepared_observation


def run_policy_for_explanation(
    policy: Policy,
    observation: ObservationBatch,
    preprocess_observation: bool,
) -> dict[str, torch.Tensor]:
    """Return normalized predictions without using decoder encoder caches.

    Args:
        policy: Policy being explained.
        observation: Observation values keyed by observation-space names.
        preprocess_observation: Whether to normalize/tokenize observations
            before building policy features.

    Returns:
        Policy predictions in the normalized action space used by attribution.
    """
    prepared_observation = prepare_policy_observation_for_explanation(
        policy=policy,
        observation=observation,
        preprocess_observation=preprocess_observation,
    )
    features = policy._build_algorithm_features(observation=prepared_observation)
    with EncoderCacheDisabled(decoder=policy.decoder):
        return policy.algorithm.predict(features=features, network=policy.decoder)


def default_output_selector(predictions: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the normalized action-vector norm used as default objective.

    Args:
        predictions: Normalized policy predictions keyed by action component.
            Each tensor must use its action-component dimension as the final
            axis and share all leading batch/horizon dimensions with the other
            tensors.

    Returns:
        Per-sample action norm after concatenating all prediction tensors along
        the final axis. Attribution methods average this tensor to obtain the
        scalar objective for gradients or ablation score drops.

    Raises:
        ValueError: If ``predictions`` is empty.
        ValueError: If a prediction tensor is scalar.
        ValueError: If prediction tensors do not share leading dimensions.
    """
    if not predictions:
        raise ValueError("Cannot select an explanation target from empty predictions.")

    action_chunks = []
    reference_key = ""
    reference_leading_shape: tuple[int, ...] | None = None
    for prediction_key, prediction in predictions.items():
        if prediction.dim() == 0:
            raise ValueError(
                f"Prediction '{prediction_key}' must have at least one dimension. "
                "Got scalar tensor."
            )
        leading_shape = tuple(prediction.shape[:-1])
        if reference_leading_shape is None:
            reference_key = prediction_key
            reference_leading_shape = leading_shape
        elif leading_shape != reference_leading_shape:
            raise ValueError(
                "All prediction tensors must share the same leading shape before "
                f"concatenation. Got {leading_shape} for '{prediction_key}' and "
                f"{reference_leading_shape} for '{reference_key}'."
            )
        action_chunks.append(prediction)

    action_prediction = torch.cat(action_chunks, dim=-1)
    return torch.linalg.vector_norm(action_prediction, dim=-1)
