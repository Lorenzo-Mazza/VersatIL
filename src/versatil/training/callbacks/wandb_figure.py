"""Helper for converting matplotlib figures into wandb images."""

import io

import matplotlib.pyplot as plt
from PIL import Image

import wandb

plt.set_loglevel("warning")


def figure_to_wandb_image(fig: plt.Figure, dpi: int = 100) -> wandb.Image:
    """Convert matplotlib figure to WandB image.

    Args:
        fig: Matplotlib figure
        dpi: Resolution for the rasterized PNG buffer

    Returns:
        WandB image object
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    pil_img = Image.open(buf)
    return wandb.Image(pil_img)
