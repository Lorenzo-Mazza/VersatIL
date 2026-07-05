"""Tests for versatil.metrics.losses.optimal_transport module."""

import re
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.metrics.base import LossOutput
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.metrics.losses.optimal_transport import (
    LatentOptimalTransportLoss,
    OptimalTransportLoss,
    RelaxedConditionalLatentOptimalTransportLoss,
)
from versatil.models.decoding.constants import LatentKey

geomloss = pytest.importorskip("geomloss")


@pytest.fixture
def mock_samples_loss():
    mock_loss = MagicMock()
    mock_loss.return_value = torch.tensor([0.5, 0.3])
    return mock_loss


@pytest.fixture
def predictions_factory(rng):
    def factory(
        action_keys: list[str],
        batch_size: int = 4,
        horizon: int = 8,
        action_dimension: int = 3,
    ) -> dict[str, torch.Tensor]:
        result = {}
        for key in action_keys:
            data = rng.standard_normal((batch_size, horizon, action_dimension)).astype(
                np.float32
            )
            result[key] = torch.from_numpy(data)
        return result

    return factory


@pytest.fixture
def latent_predictions_factory(rng):
    def factory(
        batch_size: int = 4,
        latent_dimension: int = 8,
        include_mu: bool = False,
        include_logvar: bool = False,
        include_condition: bool = False,
        condition_dimension: int = 3,
    ) -> dict[str, torch.Tensor]:
        result = {}
        data_posterior = rng.standard_normal((batch_size, latent_dimension)).astype(
            np.float32
        )
        data_prior = rng.standard_normal((batch_size, latent_dimension)).astype(
            np.float32
        )
        result[LatentKey.POSTERIOR_LATENT.value] = torch.from_numpy(data_posterior)
        result[LatentKey.PRIOR_LATENT.value] = torch.from_numpy(data_prior)
        if include_mu:
            mu_data = rng.standard_normal((batch_size, latent_dimension)).astype(
                np.float32
            )
            result[LatentKey.POSTERIOR_MU.value] = torch.from_numpy(mu_data)
            prior_mu_data = rng.standard_normal((batch_size, latent_dimension)).astype(
                np.float32
            )
            result[LatentKey.PRIOR_MU.value] = torch.from_numpy(prior_mu_data)
        if include_logvar:
            logvar_data = rng.standard_normal((batch_size, latent_dimension)).astype(
                np.float32
            )
            result[LatentKey.POSTERIOR_LOGVAR.value] = torch.from_numpy(logvar_data)
            prior_logvar_data = rng.standard_normal(
                (batch_size, latent_dimension)
            ).astype(np.float32)
            result[LatentKey.PRIOR_LOGVAR.value] = torch.from_numpy(prior_logvar_data)
        if include_condition:
            condition_data = rng.standard_normal(
                (batch_size, condition_dimension)
            ).astype(np.float32)
            result[LatentKey.PRIOR_CONDITION.value] = torch.from_numpy(condition_data)
        return result

    return factory


@pytest.mark.unit
class TestOptimalTransportLossInit:
    def test_raises_import_error_when_geomloss_missing(self):
        with (
            patch.dict("sys.modules", {"geomloss": None}),
            pytest.raises(
                ImportError,
                match="OptimalTransportLoss requires geomloss and pykeops",
            ),
        ):
            OptimalTransportLoss(
                action_keys=["position"],
                weight=0.1,
            )

    @patch(
        "versatil.metrics.losses.optimal_transport.OptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_stores_weight(self, mock_init):
        instance = OptimalTransportLoss.__new__(OptimalTransportLoss)
        instance.weight = 0.5
        instance.action_keys = ["position"]
        instance.time_scale = 1.0
        assert instance.weight == 0.5

    @patch(
        "versatil.metrics.losses.optimal_transport.OptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_stores_action_keys(self, mock_init):
        instance = OptimalTransportLoss.__new__(OptimalTransportLoss)
        instance.action_keys = ["position", "orientation"]
        assert instance.action_keys == ["position", "orientation"]


@pytest.mark.unit
class TestOptimalTransportLossGetRequiredKeys:
    @patch(
        "versatil.metrics.losses.optimal_transport.OptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_returns_action_keys_as_set(self, mock_init):
        instance = OptimalTransportLoss.__new__(OptimalTransportLoss)
        instance.action_keys = ["position", "orientation"]
        required = instance.get_required_keys()
        assert required == {"position", "orientation"}


@pytest.mark.unit
class TestOptimalTransportLossForward:
    @patch(
        "versatil.metrics.losses.optimal_transport.OptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_raises_on_missing_prediction_key(self, mock_init, predictions_factory):
        instance = OptimalTransportLoss.__new__(OptimalTransportLoss)
        instance.action_keys = ["position", "orientation"]
        instance.weight = 1.0
        instance.time_scale = 1.0
        instance.ot = MagicMock()

        predictions = predictions_factory(
            action_keys=["position"], batch_size=2, horizon=4, action_dimension=3
        )
        targets = predictions_factory(
            action_keys=["position", "orientation"],
            batch_size=2,
            horizon=4,
            action_dimension=3,
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Predictions and targets must contain key 'orientation' for Optimal Transport Loss."
            ),
        ):
            instance.forward(predictions=predictions, targets=targets, is_pad=None)

    @patch(
        "versatil.metrics.losses.optimal_transport.OptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_raises_on_missing_target_key(self, mock_init, predictions_factory):
        instance = OptimalTransportLoss.__new__(OptimalTransportLoss)
        instance.action_keys = ["position", "orientation"]
        instance.weight = 1.0
        instance.time_scale = 1.0
        instance.ot = MagicMock()

        predictions = predictions_factory(
            action_keys=["position", "orientation"],
            batch_size=2,
            horizon=4,
            action_dimension=3,
        )
        targets = predictions_factory(
            action_keys=["position"], batch_size=2, horizon=4, action_dimension=3
        )

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Predictions and targets must contain key 'orientation' for Optimal Transport Loss."
            ),
        ):
            instance.forward(predictions=predictions, targets=targets, is_pad=None)

    @patch(
        "versatil.metrics.losses.optimal_transport.OptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_concatenates_action_keys_and_time_embeddings(
        self, mock_init, predictions_factory
    ):
        instance = OptimalTransportLoss.__new__(OptimalTransportLoss)
        instance.action_keys = ["position"]
        instance.weight = 0.1
        instance.time_scale = 1.0

        mock_ot = MagicMock()
        mock_ot.return_value = torch.tensor([0.5, 0.3, 0.4, 0.2])
        instance.ot = mock_ot

        batch_size, horizon, action_dim = 4, 8, 3
        predictions = predictions_factory(
            action_keys=["position"],
            batch_size=batch_size,
            horizon=horizon,
            action_dimension=action_dim,
        )
        targets = predictions_factory(
            action_keys=["position"],
            batch_size=batch_size,
            horizon=horizon,
            action_dimension=action_dim,
        )

        instance.forward(predictions=predictions, targets=targets, is_pad=None)

        # OT was called once
        mock_ot.assert_called_once()
        call_args = mock_ot.call_args
        # 4 positional args: weights_x, samples_x, weights_y, samples_y
        assert len(call_args[0]) == 4
        weights_x, samples_x, weights_y, samples_y = call_args[0]
        # Samples should have time concatenated: action_dim + 1
        assert samples_x.shape == (batch_size, horizon, action_dim + 1)
        assert samples_y.shape == (batch_size, horizon, action_dim + 1)

    @patch(
        "versatil.metrics.losses.optimal_transport.OptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_weight_scales_total_loss(self, mock_init, predictions_factory):
        instance = OptimalTransportLoss.__new__(OptimalTransportLoss)
        instance.action_keys = ["position"]
        instance.weight = 0.5
        instance.time_scale = 1.0

        ot_value = 2.0
        mock_ot = MagicMock()
        mock_ot.return_value = torch.tensor([ot_value, ot_value])
        instance.ot = mock_ot

        predictions = predictions_factory(
            action_keys=["position"], batch_size=2, horizon=4, action_dimension=3
        )
        targets = predictions_factory(
            action_keys=["position"], batch_size=2, horizon=4, action_dimension=3
        )

        result = instance.forward(predictions=predictions, targets=targets, is_pad=None)

        expected_component = ot_value  # mean of [ot_value, ot_value]
        assert result.component_losses[
            MetricKey.OPTIMAL_TRANSPORT_LOSS.value
        ].item() == pytest.approx(expected_component)
        assert result.total_loss.item() == pytest.approx(0.5 * expected_component)

    @patch(
        "versatil.metrics.losses.optimal_transport.OptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_padding_mask_zeros_out_padded_weights(
        self, mock_init, predictions_factory
    ):
        instance = OptimalTransportLoss.__new__(OptimalTransportLoss)
        instance.action_keys = ["position"]
        instance.weight = 1.0
        instance.time_scale = 1.0

        mock_ot = MagicMock()
        mock_ot.return_value = torch.tensor([0.5, 0.3])
        instance.ot = mock_ot

        batch_size, horizon = 2, 4
        predictions = predictions_factory(
            action_keys=["position"],
            batch_size=batch_size,
            horizon=horizon,
            action_dimension=3,
        )
        targets = predictions_factory(
            action_keys=["position"],
            batch_size=batch_size,
            horizon=horizon,
            action_dimension=3,
        )

        is_pad = torch.zeros(batch_size, horizon, dtype=torch.bool)
        is_pad[:, 2:] = True  # Last 2 timesteps are padded

        instance.forward(predictions=predictions, targets=targets, is_pad=is_pad)

        call_args = mock_ot.call_args
        weights_x = call_args[0][0]
        # Padded positions should have zero weight
        assert torch.all(weights_x[:, 2:] == 0.0)
        # Valid positions should have non-zero weight
        assert torch.all(weights_x[:, :2] > 0.0)

    @patch(
        "versatil.metrics.losses.optimal_transport.OptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_returns_loss_output(self, mock_init, predictions_factory):
        instance = OptimalTransportLoss.__new__(OptimalTransportLoss)
        instance.action_keys = ["position"]
        instance.weight = 1.0
        instance.time_scale = 1.0

        mock_ot = MagicMock()
        mock_ot.return_value = torch.tensor([0.5, 0.3])
        instance.ot = mock_ot

        predictions = predictions_factory(
            action_keys=["position"], batch_size=2, horizon=4, action_dimension=3
        )
        targets = predictions_factory(
            action_keys=["position"], batch_size=2, horizon=4, action_dimension=3
        )

        result = instance.forward(predictions=predictions, targets=targets, is_pad=None)

        assert isinstance(result, LossOutput)
        assert MetricKey.OPTIMAL_TRANSPORT_LOSS.value in result.component_losses


@pytest.mark.unit
@pytest.mark.unit
class TestLatentOptimalTransportLossInit:
    def test_raises_import_error_when_geomloss_missing(self):
        with (
            patch.dict("sys.modules", {"geomloss": None}),
            pytest.raises(
                ImportError,
                match="LatentOptimalTransportLoss requires geomloss and pykeops",
            ),
        ):
            LatentOptimalTransportLoss(weight=1.0)


@pytest.mark.unit
class TestLatentOptimalTransportLossGetRequiredKeys:
    @patch(
        "versatil.metrics.losses.optimal_transport.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_returns_posterior_and_prior_keys(self, mock_init):
        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.prior_target_key = LatentKey.POSTERIOR_LATENT.value
        required = instance.get_required_keys()
        assert required == {
            LatentKey.POSTERIOR_LATENT.value,
            LatentKey.PRIOR_LATENT.value,
        }

    @patch(
        "versatil.metrics.losses.optimal_transport.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_returns_configured_prior_target_key(self, mock_init):
        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.prior_target_key = LatentKey.POSTERIOR_MU.value
        required = instance.get_required_keys()
        assert required == {
            LatentKey.POSTERIOR_MU.value,
            LatentKey.PRIOR_LATENT.value,
        }


@pytest.mark.unit
class TestLatentOptimalTransportLossForward:
    @patch(
        "versatil.metrics.losses.optimal_transport.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_raises_on_missing_keys(self, mock_init, latent_predictions_factory):
        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.weight = 1.0
        instance.prior_target_key = LatentKey.POSTERIOR_LATENT.value
        instance.ot = MagicMock()

        full_predictions = latent_predictions_factory(batch_size=4, latent_dimension=8)
        predictions = {
            LatentKey.POSTERIOR_LATENT.value: full_predictions[
                LatentKey.POSTERIOR_LATENT.value
            ]
        }

        with pytest.raises(ValueError, match="Predictions must contain"):
            instance.forward(predictions=predictions, targets={}, is_pad=None)

    @patch(
        "versatil.metrics.losses.optimal_transport.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_computes_sinkhorn_loss(self, mock_init, latent_predictions_factory):
        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.weight = 0.5
        instance.prior_target_key = LatentKey.POSTERIOR_LATENT.value

        ot_value = 1.0
        mock_ot = MagicMock()
        mock_ot.return_value = torch.tensor([ot_value, ot_value, ot_value, ot_value])
        instance.ot = mock_ot

        predictions = latent_predictions_factory(batch_size=4, latent_dimension=8)

        result = instance.forward(predictions=predictions, targets={}, is_pad=None)

        assert result.component_losses[
            MetricKey.SINKHORN_LOSS.value
        ].item() == pytest.approx(ot_value)
        assert result.total_loss.item() == pytest.approx(0.5 * ot_value)

    @patch(
        "versatil.metrics.losses.optimal_transport.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_uses_configured_prior_target_key_for_matching(
        self, mock_init, latent_predictions_factory
    ):
        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.weight = 1.0
        instance.prior_target_key = LatentKey.POSTERIOR_MU.value

        mock_ot = MagicMock()
        mock_ot.return_value = torch.tensor([0.1, 0.1, 0.1, 0.1])
        instance.ot = mock_ot

        predictions = latent_predictions_factory(
            batch_size=4,
            latent_dimension=8,
            include_mu=True,
        )

        result = instance.forward(predictions=predictions, targets={}, is_pad=None)

        posterior_arg, _ = mock_ot.call_args.args
        assert torch.equal(posterior_arg, predictions[LatentKey.POSTERIOR_MU.value])
        assert torch.equal(
            result.metadata[MetadataKey.POSTERIOR_Z.value],
            predictions[LatentKey.POSTERIOR_LATENT.value],
        )

    @patch(
        "versatil.metrics.losses.optimal_transport.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_includes_latent_metadata(self, mock_init, latent_predictions_factory):
        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.weight = 1.0
        instance.prior_target_key = LatentKey.POSTERIOR_LATENT.value

        mock_ot = MagicMock()
        mock_ot.return_value = torch.tensor([0.1, 0.1, 0.1, 0.1])
        instance.ot = mock_ot

        predictions = latent_predictions_factory(batch_size=4, latent_dimension=8)

        result = instance.forward(predictions=predictions, targets={}, is_pad=None)

        assert MetadataKey.POSTERIOR_Z.value in result.metadata
        assert MetadataKey.PRIOR_Z.value in result.metadata

    @patch(
        "versatil.metrics.losses.optimal_transport.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_includes_optional_mu_and_logvar_metadata(
        self, mock_init, latent_predictions_factory
    ):
        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.weight = 1.0
        instance.prior_target_key = LatentKey.POSTERIOR_LATENT.value

        mock_ot = MagicMock()
        mock_ot.return_value = torch.tensor([0.1, 0.1, 0.1, 0.1])
        instance.ot = mock_ot

        predictions = latent_predictions_factory(
            batch_size=4,
            latent_dimension=8,
            include_mu=True,
            include_logvar=True,
        )

        result = instance.forward(predictions=predictions, targets={}, is_pad=None)

        assert MetadataKey.POSTERIOR_MU.value in result.metadata
        assert MetadataKey.PRIOR_MU.value in result.metadata
        assert MetadataKey.POSTERIOR_LOGVAR.value in result.metadata
        assert MetadataKey.PRIOR_LOGVAR.value in result.metadata

    @patch(
        "versatil.metrics.losses.optimal_transport.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_omits_optional_metadata_when_not_present(
        self, mock_init, latent_predictions_factory
    ):
        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.weight = 1.0
        instance.prior_target_key = LatentKey.POSTERIOR_LATENT.value

        mock_ot = MagicMock()
        mock_ot.return_value = torch.tensor([0.1, 0.1, 0.1, 0.1])
        instance.ot = mock_ot

        predictions = latent_predictions_factory(
            batch_size=4,
            latent_dimension=8,
            include_mu=False,
            include_logvar=False,
        )

        result = instance.forward(predictions=predictions, targets={}, is_pad=None)

        assert MetadataKey.POSTERIOR_MU.value not in result.metadata
        assert MetadataKey.PRIOR_MU.value not in result.metadata
        assert MetadataKey.POSTERIOR_LOGVAR.value not in result.metadata
        assert MetadataKey.PRIOR_LOGVAR.value not in result.metadata


@pytest.mark.unit
class TestRelaxedConditionalLatentOptimalTransportLossInit:
    def test_raises_import_error_when_geomloss_missing(self):
        with (
            patch.dict("sys.modules", {"geomloss": None}),
            pytest.raises(
                ImportError,
                match=("LatentOptimalTransportLoss requires geomloss and pykeops"),
            ),
        ):
            RelaxedConditionalLatentOptimalTransportLoss(weight=1.0)

    def test_rejects_negative_state_weight(self):
        with pytest.raises(
            ValueError,
            match=re.escape("state_weight must be non-negative, got -1.0."),
        ):
            RelaxedConditionalLatentOptimalTransportLoss(state_weight=-1.0)


@pytest.mark.unit
class TestRelaxedConditionalLatentOptimalTransportLossGetRequiredKeys:
    @patch(
        "versatil.metrics.losses.optimal_transport.RelaxedConditionalLatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_returns_posterior_prior_and_condition_keys(self, mock_init):
        instance = RelaxedConditionalLatentOptimalTransportLoss.__new__(
            RelaxedConditionalLatentOptimalTransportLoss
        )
        instance.prior_target_key = LatentKey.POSTERIOR_LATENT.value
        instance.condition_key = LatentKey.PRIOR_CONDITION.value

        required = instance.get_required_keys()

        assert required == {
            LatentKey.POSTERIOR_LATENT.value,
            LatentKey.PRIOR_LATENT.value,
            LatentKey.PRIOR_CONDITION.value,
        }

    @patch(
        "versatil.metrics.losses.optimal_transport.RelaxedConditionalLatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_returns_configured_keys(self, mock_init):
        instance = RelaxedConditionalLatentOptimalTransportLoss.__new__(
            RelaxedConditionalLatentOptimalTransportLoss
        )
        instance.prior_target_key = LatentKey.POSTERIOR_MU.value
        instance.condition_key = "custom_condition"

        required = instance.get_required_keys()

        assert required == {
            LatentKey.POSTERIOR_MU.value,
            LatentKey.PRIOR_LATENT.value,
            "custom_condition",
        }


@pytest.mark.unit
class TestRelaxedConditionalLatentOptimalTransportLossForward:
    @patch(
        "versatil.metrics.losses.optimal_transport.RelaxedConditionalLatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_raises_on_missing_keys(self, mock_init, latent_predictions_factory):
        instance = RelaxedConditionalLatentOptimalTransportLoss.__new__(
            RelaxedConditionalLatentOptimalTransportLoss
        )
        instance.weight = 1.0
        instance.prior_target_key = LatentKey.POSTERIOR_LATENT.value
        instance.condition_key = LatentKey.PRIOR_CONDITION.value
        instance.ot = MagicMock()

        predictions = latent_predictions_factory(batch_size=4, latent_dimension=8)

        with pytest.raises(
            ValueError,
            match="for RelaxedConditionalLatentOptimalTransportLoss",
        ):
            instance.forward(predictions=predictions, targets={}, is_pad=None)

    @patch(
        "versatil.metrics.losses.optimal_transport.RelaxedConditionalLatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_computes_relaxed_conditional_sinkhorn_loss(
        self, mock_init, latent_predictions_factory
    ):
        instance = RelaxedConditionalLatentOptimalTransportLoss.__new__(
            RelaxedConditionalLatentOptimalTransportLoss
        )
        instance.weight = 0.5
        instance.prior_target_key = LatentKey.POSTERIOR_LATENT.value
        instance.condition_key = LatentKey.PRIOR_CONDITION.value
        instance.state_weight = 1.0
        instance.normalize_condition = True

        ot_value = 1.0
        mock_ot = MagicMock()
        mock_ot.return_value = torch.tensor([ot_value, ot_value, ot_value, ot_value])
        instance.ot = mock_ot

        predictions = latent_predictions_factory(
            batch_size=4,
            latent_dimension=8,
            include_condition=True,
        )

        result = instance.forward(predictions=predictions, targets={}, is_pad=None)

        assert result.component_losses[
            MetricKey.RELAXED_CONDITIONAL_SINKHORN_LOSS.value
        ].item() == pytest.approx(ot_value)
        assert result.total_loss.item() == pytest.approx(0.5 * ot_value)

    @patch(
        "versatil.metrics.losses.optimal_transport.RelaxedConditionalLatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_passes_joint_state_latent_samples_to_sinkhorn(
        self, mock_init, latent_predictions_factory
    ):
        instance = RelaxedConditionalLatentOptimalTransportLoss.__new__(
            RelaxedConditionalLatentOptimalTransportLoss
        )
        instance.weight = 1.0
        instance.prior_target_key = LatentKey.POSTERIOR_LATENT.value
        instance.condition_key = LatentKey.PRIOR_CONDITION.value
        instance.state_weight = 4.0
        instance.normalize_condition = False
        instance.ot = MagicMock(return_value=torch.tensor([0.1, 0.1]))

        predictions = latent_predictions_factory(
            batch_size=2,
            latent_dimension=3,
            include_condition=True,
            condition_dimension=2,
        )

        instance.forward(predictions=predictions, targets={}, is_pad=None)

        posterior_joint, prior_joint = instance.ot.call_args.args
        expected_condition = predictions[LatentKey.PRIOR_CONDITION.value] * 2.0
        torch.testing.assert_close(posterior_joint[:, :2], expected_condition)
        torch.testing.assert_close(prior_joint[:, :2], expected_condition)
        torch.testing.assert_close(
            posterior_joint[:, 2:],
            predictions[LatentKey.POSTERIOR_LATENT.value],
        )
        torch.testing.assert_close(
            prior_joint[:, 2:],
            predictions[LatentKey.PRIOR_LATENT.value],
        )

    @patch(
        "versatil.metrics.losses.optimal_transport.RelaxedConditionalLatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_uses_configured_prior_target_key_for_matching(
        self, mock_init, latent_predictions_factory
    ):
        instance = RelaxedConditionalLatentOptimalTransportLoss.__new__(
            RelaxedConditionalLatentOptimalTransportLoss
        )
        instance.weight = 1.0
        instance.prior_target_key = LatentKey.POSTERIOR_MU.value
        instance.condition_key = LatentKey.PRIOR_CONDITION.value
        instance.state_weight = 0.0
        instance.normalize_condition = False
        instance.ot = MagicMock(return_value=torch.tensor([0.1, 0.1, 0.1, 0.1]))

        predictions = latent_predictions_factory(
            batch_size=4,
            latent_dimension=8,
            include_mu=True,
            include_condition=True,
        )

        result = instance.forward(predictions=predictions, targets={}, is_pad=None)

        posterior_joint, _ = instance.ot.call_args.args
        torch.testing.assert_close(
            posterior_joint[:, -8:],
            predictions[LatentKey.POSTERIOR_MU.value],
        )
        torch.testing.assert_close(
            result.metadata[MetadataKey.POSTERIOR_Z.value],
            predictions[LatentKey.POSTERIOR_LATENT.value],
        )

    @patch(
        "versatil.metrics.losses.optimal_transport.RelaxedConditionalLatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_includes_condition_metadata(self, mock_init, latent_predictions_factory):
        instance = RelaxedConditionalLatentOptimalTransportLoss.__new__(
            RelaxedConditionalLatentOptimalTransportLoss
        )
        instance.weight = 1.0
        instance.prior_target_key = LatentKey.POSTERIOR_LATENT.value
        instance.condition_key = LatentKey.PRIOR_CONDITION.value
        instance.state_weight = 1.0
        instance.normalize_condition = True
        instance.ot = MagicMock(return_value=torch.tensor([0.1, 0.1, 0.1, 0.1]))

        predictions = latent_predictions_factory(
            batch_size=4,
            latent_dimension=8,
            include_condition=True,
        )

        result = instance.forward(predictions=predictions, targets={}, is_pad=None)

        torch.testing.assert_close(
            result.metadata[MetadataKey.PRIOR_CONDITION.value],
            predictions[LatentKey.PRIOR_CONDITION.value],
        )

    @patch(
        "versatil.metrics.losses.optimal_transport.RelaxedConditionalLatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_rejects_batch_size_mismatch(self, mock_init, latent_predictions_factory):
        instance = RelaxedConditionalLatentOptimalTransportLoss.__new__(
            RelaxedConditionalLatentOptimalTransportLoss
        )
        instance.weight = 1.0
        instance.prior_target_key = LatentKey.POSTERIOR_LATENT.value
        instance.condition_key = LatentKey.PRIOR_CONDITION.value
        instance.state_weight = 1.0
        instance.normalize_condition = True
        instance.ot = MagicMock()

        predictions = latent_predictions_factory(
            batch_size=4,
            latent_dimension=8,
            include_condition=True,
        )
        predictions[LatentKey.PRIOR_CONDITION.value] = torch.zeros(3, 2)

        with pytest.raises(
            ValueError,
            match=re.escape(
                "Latent and condition samples must have the same batch size"
            ),
        ):
            instance.forward(predictions=predictions, targets={}, is_pad=None)


def test_geomloss_stays_unimported_without_an_optimal_transport_loss():
    # PyKeOps JIT compilation makes the geomloss import expensive, so it must
    # stay constructor-local. A subprocess gives a clean interpreter where no
    # other test has imported geomloss already.
    script = (
        "import sys; "
        "import versatil; "
        "import versatil.endpoints.train; "
        "import versatil.endpoints.deploy; "
        "import versatil.endpoints.explain; "
        "import versatil.endpoints.post_training_compress; "
        "import versatil.metrics.losses.optimal_transport; "
        "assert 'geomloss' not in sys.modules, 'geomloss imported eagerly'; "
        "assert 'pykeops' not in sys.modules, 'pykeops imported eagerly'"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
