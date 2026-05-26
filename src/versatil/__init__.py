"""VersatIL library."""

import logging
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Set cache dirs before importing transformers/timm — they read
CACHE_DIR = Path(os.environ["VERSATIL_CACHE_DIR"])


def setup_cache_directories():
    """Configure cache directories for model downloads."""
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

logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

import transformers

from versatil.quantization.torch_patches import patch_pt2e_python314

patch_pt2e_python314()

logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)
transformers.logging.set_verbosity_error()

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic._internal._generate_schema",
)
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="timm")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
warnings.filterwarnings("ignore", category=UserWarning, module="hydra")
warnings.filterwarnings("ignore", message="Trying to infer the `batch_size`")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="geomloss")
warnings.filterwarnings("ignore", category=SyntaxWarning, module="torchao")
warnings.filterwarnings(
    "ignore",
    message="The given buffer is not writable",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*self.log.*self.trainer.*not registered.*",
    module="pytorch_lightning",
)
warnings.filterwarnings(
    "ignore",
    message=".*tensorboardX.*has been removed.*",
    module="pytorch_lightning",
)
warnings.filterwarnings(
    "ignore",
    message=".*isinstance.*LeafSpec.*is deprecated.*",
    module="pytorch_lightning",
)
warnings.filterwarnings(
    "ignore",
    message="Checkpoint directory.*exists and is not empty",
    module="pytorch_lightning",
)
warnings.filterwarnings(
    "ignore",
    message="crc32c usage is deprecated",
    module="numcodecs.*",
)
