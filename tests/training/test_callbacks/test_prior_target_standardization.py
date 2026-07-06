"""Tests for versatil.training.callbacks.prior_target_standardization module."""

import re
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from versatil.data.constants import SampleKey
from versatil.models.decoding.constants import LatentKey
from versatil.models.decoding.latent.prior.latent_standardizer import (
    LatentStandardizer,
)
from versatil.training.callbacks.prior_target_standardization import (
    PriorTargetStandardizationCallback,
)
from versatil.training.constants import PrecisionType
from versatil.training.lightning_policy import LightningPolicy

LATENT_DIMENSION = 2


def _mock_trainer() -> MagicMock:
    trainer = MagicMock()
    trainer.precision = PrecisionType.FP32.value
    return trainer


class _ToyEncodingPipeline(torch.nn.Module):
    def forward(
        self,
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return precomputed conditioning features.

        Args:
            observations: Observation dictionary from the batch.

        Returns:
            Feature dictionary consumed by the posterior encoder.
        """
        return {"feature": observations["feature"]}


class _ToyPosteriorEncoder(torch.nn.Module):
    def __init__(self, trainable: bool) -> None:
        """Initialize the toy posterior encoder.

        Args:
            trainable: Whether the encoder parameter requires gradients.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=trainable)

    def encode(
        self,
        actions: dict[str, torch.Tensor],
        observations: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Encode targets into deterministic posterior outputs.

        Args:
            actions: Action dictionary containing ``target``.
            observations: Feature dictionary containing ``feature``.

        Returns:
            Posterior output with distinct ``mu`` and ``latent`` entries.
        """
        target = actions["target"] + observations["feature"]
        return {
            LatentKey.POSTERIOR_MU.value: target,
            LatentKey.POSTERIOR_LATENT.value: target + 10.0,
        }


class _ToyPrior(torch.nn.Module):
    def __init__(self, trainable: bool, enabled: bool = True) -> None:
        """Initialize the toy prior.

        Args:
            trainable: Whether the prior parameter requires gradients.
            enabled: Whether latent standardization is enabled.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=trainable)
        self.latent_standardizer = LatentStandardizer(
            latent_dimension=LATENT_DIMENSION,
            enabled=enabled,
        )

    def build_training_target(
        self,
        posterior_output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Select posterior mean as the prior target.

        Args:
            posterior_output: Posterior encoder output dictionary.

        Returns:
            Detached posterior mean target.
        """
        return posterior_output[LatentKey.POSTERIOR_MU.value].detach()


class _ToyAlgorithm(torch.nn.Module):
    def __init__(self, posterior_trainable: bool, prior_trainable: bool) -> None:
        """Initialize the toy variational algorithm.

        Args:
            posterior_trainable: Whether the posterior encoder is trainable.
            prior_trainable: Whether the prior is trainable.
        """
        super().__init__()
        self.posterior_encoder = _ToyPosteriorEncoder(trainable=posterior_trainable)
        self.prior = _ToyPrior(trainable=prior_trainable)


class _ToyPolicy(torch.nn.Module):
    def __init__(self, posterior_trainable: bool, prior_trainable: bool) -> None:
        """Initialize the toy policy.

        Args:
            posterior_trainable: Whether the posterior encoder is trainable.
            prior_trainable: Whether the prior is trainable.
        """
        super().__init__()
        self.encoding_pipeline = _ToyEncodingPipeline()
        self.algorithm = _ToyAlgorithm(
            posterior_trainable=posterior_trainable,
            prior_trainable=prior_trainable,
        )

    def _build_algorithm_features(
        self, observation: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return self.encoding_pipeline(observation)


def _batch(
    target: list[list[float]],
    feature: list[list[float]] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build a policy batch with observation and action dictionaries.

    Args:
        target: Target action rows.
        feature: Optional feature rows added by the posterior encoder.

    Returns:
        Batch dictionary using standard sample keys.
    """
    target_tensor = torch.tensor(target, dtype=torch.float32)
    if feature is None:
        feature_tensor = torch.zeros_like(target_tensor)
    else:
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
    return {
        SampleKey.OBSERVATION.value: {"feature": feature_tensor},
        SampleKey.ACTION.value: {"target": target_tensor},
    }


@pytest.fixture
def lightning_policy_for_standardization_factory(
    training_config_factory: Callable,
) -> Callable[..., LightningPolicy]:
    """Build a Lightning policy with toy variational modules.

    Args:
        training_config_factory: Shared training config fixture.

    Returns:
        Factory for configured ``LightningPolicy`` instances.
    """

    def factory(
        posterior_trainable: bool = False,
        prior_trainable: bool = True,
        batches: list[dict[str, dict[str, torch.Tensor]]] | None = None,
    ) -> LightningPolicy:
        policy = _ToyPolicy(
            posterior_trainable=posterior_trainable,
            prior_trainable=prior_trainable,
        )
        lightning_policy = LightningPolicy(
            policy=policy,
            training_config=training_config_factory(),
        )
        lightning_policy._train_dataloader = (
            batches
            if batches is not None
            else [_batch(target=[[1.0, 2.0], [3.0, 4.0]])]
        )
        return lightning_policy

    return factory


@pytest.mark.unit
class TestPriorTargetStandardizationCallback:
    def test_invalid_max_batches_raises(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "PriorTargetStandardizationCallback max_batches must be positive, "
                "got 0."
            ),
        ):
            PriorTargetStandardizationCallback(max_batches=0)

    def test_fits_when_prior_trainable_and_posterior_frozen(
        self,
        lightning_policy_for_standardization_factory: Callable[..., LightningPolicy],
    ) -> None:
        batches = [
            _batch(target=[[1.0, 2.0], [3.0, 4.0]]),
            _batch(target=[[5.0, 6.0]], feature=[[1.0, 1.0]]),
        ]
        lightning_policy = lightning_policy_for_standardization_factory(
            posterior_trainable=False,
            prior_trainable=True,
            batches=batches,
        )
        lightning_policy.policy.train()
        callback = PriorTargetStandardizationCallback()

        callback.on_train_start(trainer=_mock_trainer(), pl_module=lightning_policy)

        expected_targets = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [6.0, 7.0]],
            dtype=torch.float32,
        )
        standardizer = lightning_policy.policy.algorithm.prior.latent_standardizer
        torch.testing.assert_close(
            standardizer.mean,
            expected_targets.mean(dim=0),
        )
        torch.testing.assert_close(
            standardizer.std,
            expected_targets.std(dim=0, unbiased=False),
        )
        assert standardizer.is_fitted.item() is True
        assert lightning_policy.policy.training is True

    def test_respects_max_batches(
        self,
        lightning_policy_for_standardization_factory: Callable[..., LightningPolicy],
    ) -> None:
        batches = [
            _batch(target=[[1.0, 2.0], [3.0, 4.0]]),
            _batch(target=[[100.0, 200.0]]),
        ]
        lightning_policy = lightning_policy_for_standardization_factory(
            posterior_trainable=False,
            prior_trainable=True,
            batches=batches,
        )
        callback = PriorTargetStandardizationCallback(max_batches=1)

        callback.on_train_start(trainer=_mock_trainer(), pl_module=lightning_policy)

        expected_targets = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        standardizer = lightning_policy.policy.algorithm.prior.latent_standardizer
        torch.testing.assert_close(
            standardizer.mean,
            expected_targets.mean(dim=0),
        )
        torch.testing.assert_close(
            standardizer.std,
            expected_targets.std(dim=0, unbiased=False),
        )

    @pytest.mark.parametrize(
        "posterior_trainable, prior_trainable",
        [
            (True, True),
            (False, False),
        ],
    )
    def test_skips_until_posterior_frozen_and_prior_trainable(
        self,
        lightning_policy_for_standardization_factory: Callable[..., LightningPolicy],
        posterior_trainable: bool,
        prior_trainable: bool,
    ) -> None:
        lightning_policy = lightning_policy_for_standardization_factory(
            posterior_trainable=posterior_trainable,
            prior_trainable=prior_trainable,
        )
        posterior_encoder = lightning_policy.policy.algorithm.posterior_encoder
        callback = PriorTargetStandardizationCallback()

        with patch.object(
            posterior_encoder,
            "encode",
            wraps=posterior_encoder.encode,
        ) as encode_spy:
            callback.on_train_start(trainer=_mock_trainer(), pl_module=lightning_policy)

        standardizer = lightning_policy.policy.algorithm.prior.latent_standardizer
        assert standardizer.is_fitted.item() is False
        encode_spy.assert_not_called()

    def test_missing_dataloader_raises(
        self,
        lightning_policy_for_standardization_factory: Callable[..., LightningPolicy],
    ) -> None:
        lightning_policy = lightning_policy_for_standardization_factory(
            posterior_trainable=False,
            prior_trainable=True,
            batches=[],
        )
        lightning_policy._train_dataloader = None
        callback = PriorTargetStandardizationCallback()

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "PriorTargetStandardizationCallback requires a training dataloader "
                "before fitting latent statistics."
            ),
        ):
            callback.on_train_start(trainer=object(), pl_module=lightning_policy)

    def test_invalid_batch_keys_raise(
        self,
        lightning_policy_for_standardization_factory: Callable[..., LightningPolicy],
    ) -> None:
        lightning_policy = lightning_policy_for_standardization_factory(
            posterior_trainable=False,
            prior_trainable=True,
            batches=[{SampleKey.OBSERVATION.value: {"feature": torch.zeros(1, 2)}}],
        )
        callback = PriorTargetStandardizationCallback()

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "PriorTargetStandardizationCallback expects batches with "
                f"'{SampleKey.OBSERVATION.value}' and "
                f"'{SampleKey.ACTION.value}' keys."
            ),
        ):
            callback.on_train_start(trainer=_mock_trainer(), pl_module=lightning_policy)

    def test_non_lightning_policy_module_raises(self) -> None:
        callback = PriorTargetStandardizationCallback()
        not_a_lightning_policy = MagicMock()
        with pytest.raises(
            TypeError,
            match=re.escape(
                "PriorTargetStandardizationCallback requires a LightningPolicy module, "
                f"got {type(not_a_lightning_policy).__name__}."
            ),
        ):
            callback.on_train_start(
                trainer=_mock_trainer(), pl_module=not_a_lightning_policy
            )

    def test_target_latent_dimension_mismatch_raises(
        self,
        lightning_policy_for_standardization_factory: Callable[..., LightningPolicy],
    ) -> None:
        wrong_dimension = LATENT_DIMENSION + 1
        lightning_policy = lightning_policy_for_standardization_factory(
            posterior_trainable=False,
            prior_trainable=True,
            batches=[
                _batch(
                    target=[[1.0, 2.0, 3.0]],
                    feature=[[0.0, 0.0, 0.0]],
                )
            ],
        )
        callback = PriorTargetStandardizationCallback()

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Prior target latent dimension mismatch: expected "
                f"{LATENT_DIMENSION}, got {wrong_dimension}."
            ),
        ):
            callback.on_train_start(trainer=_mock_trainer(), pl_module=lightning_policy)

    def test_empty_dataloader_raises(
        self,
        lightning_policy_for_standardization_factory: Callable[..., LightningPolicy],
    ) -> None:
        lightning_policy = lightning_policy_for_standardization_factory(
            posterior_trainable=False,
            prior_trainable=True,
            batches=[],
        )
        callback = PriorTargetStandardizationCallback()

        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "PriorTargetStandardizationCallback could not fit latent statistics "
                "from an empty training dataloader."
            ),
        ):
            callback.on_train_start(trainer=_mock_trainer(), pl_module=lightning_policy)
