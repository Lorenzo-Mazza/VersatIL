"""Unit tests for the VersatIL DDP strategy."""

from unittest.mock import MagicMock, patch

import pytest
from torch.nn import Module

from versatil.training.ddp_strategy import DefaultStreamDDPStrategy


@pytest.mark.unit
class TestSetupModel:
    def test_uses_default_cuda_stream_for_ordinary_training(self) -> None:
        model = MagicMock(spec=Module)
        strategy = DefaultStreamDDPStrategy()
        default_stream = MagicMock()
        stream_context = MagicMock()
        wrapped_model = MagicMock()

        with (
            patch.object(
                strategy,
                "determine_ddp_device_ids",
                return_value=[0],
            ),
            patch(
                "versatil.training.ddp_strategy.torch.cuda.is_current_stream_capturing",
                return_value=False,
            ),
            patch(
                "versatil.training.ddp_strategy.torch.cuda.default_stream",
                return_value=default_stream,
            ) as mock_default_stream,
            patch(
                "versatil.training.ddp_strategy.torch.cuda.Stream"
            ) as mock_side_stream,
            patch(
                "versatil.training.ddp_strategy.torch.cuda.stream",
                return_value=stream_context,
            ) as mock_stream_context,
            patch(
                "versatil.training.ddp_strategy.DistributedDataParallel",
                return_value=wrapped_model,
            ) as mock_ddp,
        ):
            result = strategy._setup_model(model=model)

        assert result is wrapped_model
        mock_default_stream.assert_called_once_with()
        mock_side_stream.assert_not_called()
        mock_stream_context.assert_called_once_with(default_stream)
        mock_ddp.assert_called_once_with(module=model, device_ids=[0])

    def test_uses_side_stream_during_cuda_graph_capture(self) -> None:
        model = MagicMock(spec=Module)
        strategy = DefaultStreamDDPStrategy(find_unused_parameters=True)
        side_stream = MagicMock()

        with (
            patch.object(
                strategy,
                "determine_ddp_device_ids",
                return_value=[1],
            ),
            patch(
                "versatil.training.ddp_strategy.torch.cuda.is_current_stream_capturing",
                return_value=True,
            ),
            patch(
                "versatil.training.ddp_strategy.torch.cuda.Stream",
                return_value=side_stream,
            ),
            patch(
                "versatil.training.ddp_strategy.torch.cuda.stream",
                return_value=MagicMock(),
            ) as mock_stream_context,
            patch(
                "versatil.training.ddp_strategy.torch.autograd.graph."
                "set_warn_on_accumulate_grad_stream_mismatch"
            ) as mock_set_warning,
            patch("versatil.training.ddp_strategy.DistributedDataParallel") as mock_ddp,
        ):
            strategy._setup_model(model=model)

        mock_stream_context.assert_called_once_with(side_stream)
        mock_set_warning.assert_called_once_with(False)
        mock_ddp.assert_called_once_with(
            module=model,
            device_ids=[1],
            find_unused_parameters=True,
        )

    def test_skips_cuda_stream_selection_for_cpu(self) -> None:
        model = MagicMock(spec=Module)
        strategy = DefaultStreamDDPStrategy()

        with (
            patch.object(
                strategy,
                "determine_ddp_device_ids",
                return_value=None,
            ),
            patch(
                "versatil.training.ddp_strategy.torch.cuda.stream"
            ) as mock_stream_context,
            patch("versatil.training.ddp_strategy.DistributedDataParallel") as mock_ddp,
        ):
            strategy._setup_model(model=model)

        mock_stream_context.assert_not_called()
        mock_ddp.assert_called_once_with(module=model, device_ids=None)
