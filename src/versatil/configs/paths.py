"""Filesystem locations of packaged configuration resources."""

from importlib.resources import files
from pathlib import Path


def get_hydra_configs_dir() -> Path:
    """Return the packaged Hydra config directory as a filesystem path.

    The configs ship inside the ``versatil`` package, so this resolves
    correctly for source checkouts, editable installs, and wheels alike.
    """
    return Path(str(files("versatil") / "hydra_configs"))
