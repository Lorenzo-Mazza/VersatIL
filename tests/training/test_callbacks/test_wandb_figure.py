"""Tests for versatil.training.callbacks.wandb_figure module."""

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import pytest

from versatil.training.callbacks.wandb_figure import figure_to_wandb_image


@pytest.mark.unit
@patch("versatil.training.callbacks.wandb_figure.wandb")
def test_returns_wandb_image_wrapping_the_figure(mock_wandb):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    mock_wandb.Image.return_value = MagicMock()

    result = figure_to_wandb_image(fig)

    mock_wandb.Image.assert_called_once()
    assert result == mock_wandb.Image.return_value
    plt.close(fig)
