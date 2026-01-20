import os
from pathlib import Path


def setup_cache_directories(cache_dir: str | Path):
    """Configure cache directories for model downloads before any run."""
    # Convert to Path for consistent handling
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
