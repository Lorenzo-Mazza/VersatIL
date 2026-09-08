"""Bounded autoregressive action-token generation for policy export."""

import torch

from versatil.data.tokenization.action_discretizer import (
    BinnedActionDiscretizer,
    FastActionDiscretizer,
)
from versatil.models.decoding.algorithm.behavior_cloning import BehavioralCloning
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders.autoregressive_mixin import (
    AutoregressiveDecoderMixin,
)
from versatil.models.exportable.base import ExportablePolicy
from versatil.models.exportable.metadata import PolicyExportMetadata, PredictionOutput
from versatil.models.policy import Policy


class ExportableTokenPolicy(ExportablePolicy):
    """Export bounded autoregressive action-token generation from observations.

    Note:
        Each invocation creates a fresh decoder cache and generates up to the
        tokenizer's token limit or the remaining decoder context capacity.
        Completed sequences repeat EOS through the remaining positions. The runtime
        reconstructs actions using the saved action tokenizer. The exported graph
        contains the complete generation sequence.
    """

    @classmethod
    def from_policy(cls, policy: Policy) -> "ExportableTokenPolicy":
        """Validate an autoregressive policy and describe its token output metadata.

        Args:
            policy: Initialized policy with its fitted action tokenizer attached.

        Returns:
            Wrapper sharing the policy's encoder, algorithm and decoder weights.

        Raises:
            ValueError: If the policy requires a different token-generation adapter
                or its fitted tokenizer is unavailable or has incompatible dimensions.
        """
        if not isinstance(policy.algorithm, BehavioralCloning) or not isinstance(
            policy.decoder, AutoregressiveDecoderMixin
        ):
            raise ValueError(
                "Tokenized export requires BehavioralCloning with an autoregressive token decoder."
            )
        tokenizer = (
            policy.tokenizer.action_tokenizer if policy.tokenizer is not None else None
        )
        if tokenizer is None or not isinstance(
            tokenizer.action_discretizer,
            (BinnedActionDiscretizer, FastActionDiscretizer),
        ):
            raise ValueError(
                "Tokenized export requires a fitted binned or FAST action tokenizer."
            )
        if not tokenizer.action_discretizer.is_fitted:
            raise ValueError(
                "Tokenized export requires a fitted binned or FAST action tokenizer."
            )
        expected_shape = (
            policy.prediction_horizon,
            policy.action_space.get_total_action_dim(),
        )
        tokenizer_shape = (
            tokenizer.action_discretizer.time_horizon,
            tokenizer.action_discretizer.action_dim,
        )
        if tokenizer_shape != expected_shape:
            raise ValueError(
                f"Action tokenizer decodes shape {tokenizer_shape}, "
                f"but the policy requires {expected_shape}."
            )
        required_tokens = expected_shape[0] * expected_shape[1]
        if (
            isinstance(tokenizer.action_discretizer, BinnedActionDiscretizer)
            and tokenizer.max_token_len < required_tokens + 1
        ):
            raise ValueError(
                f"Binned action export requires capacity for {required_tokens} action tokens and EOS, "
                f"got max_token_len={tokenizer.max_token_len}."
            )
        if not policy.decoder.deterministic:
            raise ValueError(
                "Action-token export requires greedy generation (deterministic=True)."
            )
        return cls(
            encoding_pipeline=policy.encoding_pipeline,
            algorithm=policy.algorithm,
            decoder=policy.decoder,
            observation_keys=policy.input_keys,
            action_keys=[DecoderOutputKey.PREDICTED_ACTION_TOKENS.value],
            export_metadata=PolicyExportMetadata(output=PredictionOutput.ACTION_TOKENS),
        )

    def _predict(
        self,
        features: dict[str, torch.Tensor],
        sampling_inputs: tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Generate fixed-width action tokens, retaining EOS after each sequence ends.

        Args:
            features: Observation features selected for the autoregressive decoder.
            sampling_inputs: Empty tuple for greedy token generation.

        Returns:
            Token IDs under the predicted-action-token key, shaped
            ``(batch, maximum_generation_steps)``.

        Raises:
            ValueError: If the export metadata supplies noise to greedy generation,
                or the decoder context holds fewer tokens than a complete binned
                action chunk.
        """
        if sampling_inputs:
            raise ValueError(
                "Greedy token generation requires an empty noise input list."
            )
        predictions = self.decoder(
            features=features, fixed_length=True
        )  # (batch, maximum_generation_steps)
        discretizer = self.decoder.tokenizer.action_discretizer
        if isinstance(discretizer, BinnedActionDiscretizer):
            required_tokens = discretizer.time_horizon * discretizer.action_dim
            token_count = predictions[
                DecoderOutputKey.PREDICTED_ACTION_TOKENS.value
            ].shape[1]
            if token_count < required_tokens:
                raise ValueError(
                    f"Binned action export requires {required_tokens} generated action tokens, "
                    f"but the decoder context allows {token_count}."
                )
        return predictions
