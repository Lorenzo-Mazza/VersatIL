"""VersatIL library."""

import logging
import warnings

from dotenv import load_dotenv

from versatil.common.set_cache_dir import (
    resolve_cache_directory,
    setup_cache_directories,
)

load_dotenv()

CACHE_DIR = resolve_cache_directory()

setup_cache_directories(cache_dir=CACHE_DIR)

logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

import transformers

from versatil.quantization.torch_patches import register_torchao_patches

register_torchao_patches()

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
