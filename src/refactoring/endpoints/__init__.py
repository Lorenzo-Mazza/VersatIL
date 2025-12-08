"""Endpoints for model training, inference and explainability."""
import os
from pathlib import Path

CACHE_DIR = Path("/mnt/cluster/workspaces/mazzalore/pretrained_models")

def setup_cache_directories():
    """Configure cache directories for model downloads before any test run."""
    os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(CACHE_DIR / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(CACHE_DIR / "torch")
    for cache_path in [
        CACHE_DIR / "huggingface" / "transformers",
        CACHE_DIR / "huggingface" / "hub",
        CACHE_DIR / "torch" / "hub",
    ]:
        cache_path.mkdir(parents=True, exist_ok=True)
setup_cache_directories()
