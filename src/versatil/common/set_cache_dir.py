"""Configuration of HuggingFace and Torch cache directories."""

import os
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "versatil"


def resolve_cache_directory() -> Path:
    """Return the model cache root from ``VERSATIL_CACHE_DIR`` or the default."""
    return Path(os.environ.get("VERSATIL_CACHE_DIR", str(DEFAULT_CACHE_DIR)))


def setup_cache_directories(cache_dir: str | Path) -> None:
    """Configure cache directories for model downloads before any run."""
    cache_path = Path(cache_dir)

    os.environ["HF_HOME"] = str(cache_path / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_path / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(cache_path / "torch")

    for dir_path in [
        cache_path / "huggingface" / "transformers",
        cache_path / "huggingface" / "hub",
        cache_path / "torch" / "hub",
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
