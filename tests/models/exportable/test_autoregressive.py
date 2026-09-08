"""Tests for versatil.models.exportable.autoregressive module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest

from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.exportable.autoregressive import ExportableTokenPolicy
from versatil.models.exportable.metadata import (
    PredictionOutput,
)


@pytest.mark.unit
class TestTokenizedExport:
    @pytest.mark.parametrize("tokenizer_horizon,tokenizer_dimension", [(1, 3), (2, 4)])
    def test_rejects_tokenizer_action_shape_mismatch(
        self,
        tokenized_export_policy_factory: Callable[..., MagicMock],
        tokenizer_horizon: int,
        tokenizer_dimension: int,
    ) -> None:
        policy = tokenized_export_policy_factory(
            tokenizer_horizon=tokenizer_horizon, tokenizer_dimension=tokenizer_dimension
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Action tokenizer decodes shape {(tokenizer_horizon, tokenizer_dimension)}, "
                "but the policy requires (2, 3)."
            ),
        ):
            ExportableTokenPolicy.from_policy(policy=policy)

    @pytest.mark.parametrize("max_token_len", [5, 6])
    def test_requires_capacity_for_a_complete_action_chunk(
        self,
        tokenized_export_policy_factory: Callable[..., MagicMock],
        max_token_len: int,
    ) -> None:
        policy = tokenized_export_policy_factory(max_token_len=max_token_len)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Binned action export requires capacity for 6 action tokens and EOS, got max_token_len={max_token_len}."
            ),
        ):
            ExportableTokenPolicy.from_policy(policy=policy)

    def test_requires_decoder_context_for_a_complete_binned_chunk(
        self,
        tokenized_export_policy_factory: Callable[..., MagicMock],
        token_generation_tensors_factory: Callable[..., tuple[MagicMock, MagicMock]],
    ) -> None:
        policy = tokenized_export_policy_factory(vlm=True, binned=True, max_token_len=7)
        observation, tokens = token_generation_tensors_factory(token_count=5)
        policy.decoder.return_value = {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: tokens
        }
        exportable = ExportableTokenPolicy.from_policy(policy=policy)
        with (
            patch(
                "versatil.models.exportable.base.build_algorithm_features",
                return_value={"encoded": observation},
            ),
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Binned action export requires 6 generated action tokens, but the decoder context allows 5."
                ),
            ),
        ):
            exportable(observation)  # (batch, maximum_tokens)

    def test_requires_greedy_generation(
        self,
        tokenized_export_policy_factory: Callable[..., MagicMock],
    ) -> None:
        policy = tokenized_export_policy_factory(deterministic=False)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Action-token export requires greedy generation (deterministic=True)."
            ),
        ):
            ExportableTokenPolicy.from_policy(policy=policy)

    @pytest.mark.parametrize("vlm,binned", [(False, True), (True, True), (True, False)])
    def test_records_token_output_and_calls_bounded_generation(
        self,
        tokenized_export_policy_factory: Callable[..., MagicMock],
        token_generation_tensors_factory: Callable[[], tuple[MagicMock, MagicMock]],
        vlm: bool,
        binned: bool,
    ) -> None:
        policy = tokenized_export_policy_factory(
            supported_algorithm=True,
            supported_decoder=True,
            has_tokenizer=True,
            binned=binned,
            fitted=True,
            vlm=vlm,
        )
        observation, tokens = token_generation_tensors_factory()
        features = {"encoded": observation}
        token_key = DecoderOutputKey.PREDICTED_ACTION_TOKENS.value
        policy.decoder.return_value = {token_key: tokens}
        exportable = ExportableTokenPolicy.from_policy(policy=policy)
        with patch(
            "versatil.models.exportable.base.build_algorithm_features",
            return_value=features,
        ):
            outputs = exportable(observation)  # (batch, maximum_tokens)
        assert policy.decoder.call_count == 1
        assert policy.decoder.call_args.kwargs == {
            "features": features,
            "fixed_length": True,
        }
        policy.algorithm.predict.assert_not_called()
        assert outputs == (tokens,)
        assert exportable.action_keys == [token_key]
        assert exportable.export_metadata.output == PredictionOutput.ACTION_TOKENS
        assert exportable.export_metadata.noise_inputs == ()

    @pytest.mark.parametrize(
        "supported_algorithm,supported_decoder", [(False, True), (True, False)]
    )
    def test_rejects_generation_without_a_matching_export_adapter(
        self,
        tokenized_export_policy_factory: Callable[..., MagicMock],
        supported_algorithm: bool,
        supported_decoder: bool,
    ) -> None:
        policy = tokenized_export_policy_factory(
            supported_algorithm=supported_algorithm,
            supported_decoder=supported_decoder,
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Tokenized export requires BehavioralCloning with an autoregressive token decoder."
            ),
        ):
            ExportableTokenPolicy.from_policy(policy=policy)

    @pytest.mark.parametrize(
        "has_tokenizer,binned,fitted",
        [(False, True, True), (True, False, False), (True, True, False)],
    )
    def test_requires_a_fitted_action_tokenizer(
        self,
        tokenized_export_policy_factory: Callable[..., MagicMock],
        has_tokenizer: bool,
        binned: bool,
        fitted: bool,
    ) -> None:
        policy = tokenized_export_policy_factory(
            has_tokenizer=has_tokenizer, binned=binned, fitted=fitted
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Tokenized export requires a fitted binned or FAST action tokenizer."
            ),
        ):
            ExportableTokenPolicy.from_policy(policy=policy)
