"""Logging utilities for VersatIL endpoints."""

import logging

LOG_FORMAT = "%(asctime)s %(module)s %(levelname)s %(message)s"


def override_log_format() -> None:
    """Replace the log formatter on all root handlers.

    Call after Hydra's ``@hydra.main`` has configured logging,
    so the format shows the source module instead of ``root``.
    """
    formatter = logging.Formatter(LOG_FORMAT)
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)
