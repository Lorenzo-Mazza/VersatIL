"""Tests for versatil.metrics.ot_loss module."""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from versatil.metrics.base import LossOutput
from versatil.metrics.constants import MetadataKey, MetricKey
from versatil.models.decoding.constants import LatentKey


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
            prior_mu_data = rng.standard_normal(
                (batch_size, latent_dimension)
            ).astype(np.float32)
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
        return result

    return factory


@pytest.mark.unit
class TestOptimalTransportLossInit:
    def test_raises_import_error_when_geomloss_missing(self):
        with patch.dict("sys.modules", {"geomloss": None}):
            from versatil.metrics.ot_loss import OptimalTransportLoss

            with pytest.raises(
                ImportError,
                match="OptimalTransportLoss requires geomloss and pykeops",
            ):
                OptimalTransportLoss(
                    action_keys=["position"],
                    weight=0.1,
                )

    @patch("versatil.metrics.ot_loss.OptimalTransportLoss.__init__", return_value=None)
    def test_stores_weight(self, mock_init):
        from versatil.metrics.ot_loss import OptimalTransportLoss

        instance = OptimalTransportLoss.__new__(OptimalTransportLoss)
        instance.weight = 0.5
        instance.action_keys = ["position"]
        instance.time_scale = 1.0
        assert instance.weight == 0.5

    @patch("versatil.metrics.ot_loss.OptimalTransportLoss.__init__", return_value=None)
    def test_stores_action_keys(self, mock_init):
        from versatil.metrics.ot_loss import OptimalTransportLoss

        instance = OptimalTransportLoss.__new__(OptimalTransportLoss)
        instance.action_keys = ["position", "orientation"]
        assert instance.action_keys == ["position", "orientation"]


@pytest.mark.unit
class TestOptimalTransportLossGetRequiredKeys:
    @patch("versatil.metrics.ot_loss.OptimalTransportLoss.__init__", return_value=None)
    def test_returns_action_keys_as_set(self, mock_init):
        from versatil.metrics.ot_loss import OptimalTransportLoss

        instance = OptimalTransportLoss.__new__(OptimalTransportLoss)
        instance.action_keys = ["position", "orientation"]
        required = instance.get_required_keys()
        assert required == {"position", "orientation"}


@pytest.mark.unit
class TestOptimalTransportLossForward:
    @patch("versatil.metrics.ot_loss.OptimalTransportLoss.__init__", return_value=None)
    def test_raises_on_missing_prediction_key(self, mock_init, predictions_factory):
        from versatil.metrics.ot_loss import OptimalTransportLoss

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
            instance.forward(
                predictions=predictions, targets=targets, is_pad=None
            )

    @patch("versatil.metrics.ot_loss.OptimalTransportLoss.__init__", return_value=None)
    def test_raises_on_missing_target_key(self, mock_init, predictions_factory):
        from versatil.metrics.ot_loss import OptimalTransportLoss

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
            instance.forward(
                predictions=predictions, targets=targets, is_pad=None
            )

    @patch("versatil.metrics.ot_loss.OptimalTransportLoss.__init__", return_value=None)
    def test_concatenates_action_keys_and_time_embeddings(
        self, mock_init, predictions_factory
    ):
        from versatil.metrics.ot_loss import OptimalTransportLoss

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

        result = instance.forward(predictions=predictions, targets=targets, is_pad=None)

        # OT was called once
        mock_ot.assert_called_once()
        call_args = mock_ot.call_args
        # 4 positional args: weights_x, samples_x, weights_y, samples_y
        assert len(call_args[0]) == 4
        weights_x, samples_x, weights_y, samples_y = call_args[0]
        # Samples should have time concatenated: action_dim + 1
        assert samples_x.shape == (batch_size, horizon, action_dim + 1)
        assert samples_y.shape == (batch_size, horizon, action_dim + 1)

    @patch("versatil.metrics.ot_loss.OptimalTransportLoss.__init__", return_value=None)
    def test_weight_scales_total_loss(self, mock_init, predictions_factory):
        from versatil.metrics.ot_loss import OptimalTransportLoss

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
        assert result.component_losses[MetricKey.OPTIMAL_TRANSPORT_LOSS.value].item() == pytest.approx(
            expected_component
        )
        assert result.total_loss.item() == pytest.approx(0.5 * expected_component)

    @patch("versatil.metrics.ot_loss.OptimalTransportLoss.__init__", return_value=None)
    def test_padding_mask_zeros_out_padded_weights(
        self, mock_init, predictions_factory
    ):
        from versatil.metrics.ot_loss import OptimalTransportLoss

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

    @patch("versatil.metrics.ot_loss.OptimalTransportLoss.__init__", return_value=None)
    def test_returns_loss_output(self, mock_init, predictions_factory):
        from versatil.metrics.ot_loss import OptimalTransportLoss

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
        with patch.dict("sys.modules", {"geomloss": None}):
            from versatil.metrics.ot_loss import LatentOptimalTransportLoss

            with pytest.raises(
                ImportError,
                match="LatentOptimalTransportLoss requires geomloss and pykeops",
            ):
                LatentOptimalTransportLoss(weight=1.0)


@pytest.mark.unit
class TestLatentOptimalTransportLossGetRequiredKeys:
    @patch(
        "versatil.metrics.ot_loss.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_returns_posterior_and_prior_keys(self, mock_init):
        from versatil.metrics.ot_loss import LatentOptimalTransportLoss

        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        required = instance.get_required_keys()
        assert required == {
            LatentKey.POSTERIOR_LATENT.value,
            LatentKey.PRIOR_LATENT.value,
        }


@pytest.mark.unit
class TestLatentOptimalTransportLossForward:
    @patch(
        "versatil.metrics.ot_loss.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_raises_on_missing_keys(self, mock_init, latent_predictions_factory):
        from versatil.metrics.ot_loss import LatentOptimalTransportLoss

        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.weight = 1.0
        instance.ot = MagicMock()

        full_predictions = latent_predictions_factory(batch_size=4, latent_dimension=8)
        predictions = {LatentKey.POSTERIOR_LATENT.value: full_predictions[LatentKey.POSTERIOR_LATENT.value]}

        with pytest.raises(ValueError, match="Predictions must contain"):
            instance.forward(predictions=predictions, targets={}, is_pad=None)

    @patch(
        "versatil.metrics.ot_loss.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_computes_sinkhorn_loss(self, mock_init, latent_predictions_factory):
        from versatil.metrics.ot_loss import LatentOptimalTransportLoss

        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.weight = 0.5

        ot_value = 1.0
        mock_ot = MagicMock()
        mock_ot.return_value = torch.tensor([ot_value, ot_value, ot_value, ot_value])
        instance.ot = mock_ot

        predictions = latent_predictions_factory(batch_size=4, latent_dimension=8)

        result = instance.forward(predictions=predictions, targets={}, is_pad=None)

        assert result.component_losses[MetricKey.SINKHORN_LOSS.value].item() == pytest.approx(
            ot_value
        )
        assert result.total_loss.item() == pytest.approx(0.5 * ot_value)

    @patch(
        "versatil.metrics.ot_loss.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_includes_latent_metadata(self, mock_init, latent_predictions_factory):
        from versatil.metrics.ot_loss import LatentOptimalTransportLoss

        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.weight = 1.0

        mock_ot = MagicMock()
        mock_ot.return_value = torch.tensor([0.1, 0.1, 0.1, 0.1])
        instance.ot = mock_ot

        predictions = latent_predictions_factory(batch_size=4, latent_dimension=8)

        result = instance.forward(predictions=predictions, targets={}, is_pad=None)

        assert MetadataKey.POSTERIOR_Z.value in result.metadata
        assert MetadataKey.PRIOR_Z.value in result.metadata

    @patch(
        "versatil.metrics.ot_loss.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_includes_optional_mu_and_logvar_metadata(
        self, mock_init, latent_predictions_factory
    ):
        from versatil.metrics.ot_loss import LatentOptimalTransportLoss

        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.weight = 1.0

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
        "versatil.metrics.ot_loss.LatentOptimalTransportLoss.__init__",
        return_value=None,
    )
    def test_omits_optional_metadata_when_not_present(
        self, mock_init, latent_predictions_factory
    ):
        from versatil.metrics.ot_loss import LatentOptimalTransportLoss

        instance = LatentOptimalTransportLoss.__new__(LatentOptimalTransportLoss)
        instance.weight = 1.0

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
