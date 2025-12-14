"""Quick and dirty CVAE for conditional image generation.

Proof of concept reusing existing modules:
- CNNEncoder for image features
- Embedder + LanguageEncoder for text features (HuggingFace transformers)
- TransformerEncoder for posterior q(z|x, c) and prior p(z|c)
- BidirectionalDecoder for image reconstruction
- MaximumMeanDiscrepancyLoss from metrics.components

Training: MMD loss + reconstruction loss
Evaluation: FID score
"""


import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import wandb

# Reuse existing modules
from refactoring.models.encoding.encoders.rgb.cnn import CNNEncoder
from refactoring.models.encoding.encoders.constants import PoolingMethod, RGBBackboneType
from refactoring.models.encoding.encoders.language.embedder import Embedder
from refactoring.models.encoding.encoders.language.language import LanguageEncoder
from refactoring.models.layers.transformer import BidirectionalDecoder
from refactoring.models.layers.detr_transformer import TransformerEncoder, TransformerEncoderLayer
from refactoring.models.decoding.latent.reparametrize import reparametrize
from refactoring.models.decoding.constants import (
    LATENT_KEY, MU_KEY, LOGVAR_KEY,
    PRIOR_MU_KEY, PRIOR_LOGVAR_KEY, PRIOR_LATENT_KEY, CLASS_TOKEN_KEY
)
from refactoring.models.layers.transformer_input_builder import TransformerInputBuilder
from refactoring.models.layers.positional_encoding.sinusoidal import SinusoidalPositionalEncoding1D, SinusoidalPositionalEncoding2D
from refactoring.metrics.components import MaximumMeanDiscrepancyLoss, RegressionLoss
from refactoring.metrics.base import LossOutput
from refactoring.data.constants import TOKENIZED_OBSERVATIONS_KEY, IS_PAD_OBSERVATION_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CVAEConfig:
    """Configuration for CVAE model."""
    # Image settings
    image_size: int = 224
    image_channels: int = 3

    # Architecture
    embedding_dim: int = 512
    latent_dim: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 6
    feedforward_dim: int = 2048
    dropout: float = 0.1

    # Image encoder
    image_backbone: str = RGBBackboneType.RESNET18.value
    image_pooling: str = PoolingMethod.NONE.value  # Keep spatial features

    # Text encoder
    text_model: str = "google-bert/bert-base-uncased"
    max_caption_length: int = 32  # Short captions (~15-20 words)

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 100
    mmd_weight: float = 1.0
    recon_weight: float = 1.0

    # Augmentation
    augmentation_strength: str = "strong"  # "none", "light", "medium", "strong"
    use_horizontal_flip: bool = True  # Standard for diffusion models

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Image Augmentation Pipeline (Albumentations)
# =============================================================================
def get_color_augmentation(strength: str = "strong") -> A.Compose:
    """Get color augmentation pipeline.

    Inspired by strong_color.yaml and modern diffusion model practices.

    Args:
        strength: "light", "medium", or "strong"

    Returns:
        Albumentations Compose pipeline for color augmentations
    """
    if strength == "light":
        return A.Compose([
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
                p=0.4
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            ),
            A.RandomGamma(gamma_limit=(90, 110), p=0.2),
        ])
    elif strength == "medium":
        return A.Compose([
            A.ColorJitter(
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.1,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=0.6
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.RandomShadow(p=0.4),
            A.ImageCompression(
                quality_range=(50, 100),
                p=0.2
            ),
        ])
    else:  # strong
        return A.Compose([
            # Core color jittering (diffusion model standard)
            A.ColorJitter(
                brightness=0.4,
                contrast=0.5,
                saturation=0.6,
                hue=0.2,
                p=0.7
            ),
            # Brightness/contrast (from strong_color.yaml)
            A.RandomBrightnessContrast(
                brightness_limit=0.5,
                contrast_limit=0.5,
                p=0.7
            ),
            # Gamma adjustment
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            # CLAHE for local contrast enhancement
            A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=0.5),
            # Lighting effects (from strong_color.yaml)
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                src_color=(255, 255, 255),
                p=0.3
            ),
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),
                num_shadows_limit=(1, 2),
                p=0.5
            ),
            # Image quality degradation
            A.ImageCompression(
                quality_range=(30, 100),
                p=0.3
            ),
            # Additional color transforms for diffusion
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.4
            ),
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.3),
            # Occasional grayscale (common in diffusion)
            A.ToGray(p=0.05),
            # Channel manipulation
            A.ChannelShuffle(p=0.05),
        ])


def get_spatial_augmentation(
    strength: str = "strong",
    image_size: int = 224,
) -> A.Compose:
    """Get spatial/geometric augmentation pipeline.

    Inspired by strong_spatial.yaml and modern diffusion model practices.

    Args:
        strength: "light", "medium", or "strong"
        image_size: Target image size

    Returns:
        Albumentations Compose pipeline for spatial augmentations
    """
    if strength == "light":
        return A.Compose([
            # Mild blur
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            # Light noise (std_range in albumentations 2.x)
            A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
            # Mild scale/shift (use Affine in 2.x)
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=0,
                border_mode=0,
                p=0.3
            ),
        ])
    elif strength == "medium":
        return A.Compose([
            # Blur
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            # Noise (std_range in albumentations 2.x)
            A.GaussNoise(std_range=(0.05, 0.2), p=0.3),
            # Cutout/dropout (albumentations 2.x uses fraction-based ranges)
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(0.03, 0.06),  # ~7-14 pixels at 224px
                hole_width_range=(0.03, 0.06),
                fill=0,
                p=0.3
            ),
            # Scale/shift (use Affine in 2.x)
            A.Affine(
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                scale=(0.5, 1.6),
                rotate=0,
                border_mode=0,
                p=0.5
            ),
        ])
    else:  # strong
        return A.Compose([
            # Blur (from strong_spatial.yaml)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.6),
            # Noise (std_range in albumentations 2.x)
            A.OneOf([
                A.GaussNoise(std_range=(0.1, 0.3), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
            ], p=0.4),
            # Cutout/dropout (albumentations 2.x uses fraction-based ranges)
            A.CoarseDropout(
                num_holes_range=(1, 12),
                hole_height_range=(0.04, 0.08),  # ~9-18 pixels at 224px
                hole_width_range=(0.04, 0.08),
                fill=0,
                p=0.5
            ),
            # Scale/shift/rotate (use Affine in 2.x)
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.4, 1.8),
                rotate=(-15, 15),  # Add rotation for image generation
                border_mode=0,
                p=0.6
            ),
            # Perspective transform (common in diffusion models)
            A.Perspective(scale=(0.02, 0.08), p=0.3),
            # Elastic transform (no alpha_affine in 2.x)
            A.ElasticTransform(
                alpha=50,
                sigma=2.5,
                border_mode=0,
                p=0.2
            ),
            # Grid distortion
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
            # Optical distortion (lens effects)
            A.OpticalDistortion(distort_limit=(-0.1, 0.1), p=0.2),
        ])


def get_diffusion_augmentation(
    image_size: int = 224,
    strength: str = "strong",
    use_horizontal_flip: bool = True,
) -> A.Compose:
    """Get full augmentation pipeline for diffusion-style image generation.

    Combines:
    - RandomResizedCrop (critical for diffusion models)
    - Horizontal flip (standard for diffusion)
    - Color augmentations
    - Spatial augmentations
    - Final resize and normalization

    Args:
        image_size: Target image size
        strength: "none", "light", "medium", or "strong"
        use_horizontal_flip: Whether to use horizontal flip

    Returns:
        Albumentations Compose pipeline
    """
    transforms = []

    # Random resized crop (critical for diffusion models - provides scale augmentation)
    if strength != "none":
        transforms.append(
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0) if strength == "light" else (0.6, 1.0) if strength == "medium" else (0.5, 1.0),
                ratio=(0.9, 1.1) if strength == "light" else (0.75, 1.33),
                p=0.8 if strength != "light" else 0.5
            )
        )
    else:
        transforms.append(A.Resize(image_size, image_size))

    # Horizontal flip (standard for diffusion models)
    if use_horizontal_flip and strength != "none":
        transforms.append(A.HorizontalFlip(p=0.5))

    # Color augmentations
    if strength != "none":
        transforms.append(get_color_augmentation(strength))

    # Spatial augmentations
    if strength != "none":
        transforms.append(get_spatial_augmentation(strength, image_size))

    # Ensure final size
    transforms.append(A.Resize(image_size, image_size))

    # Normalize to ImageNet stats (standard for pretrained encoders)
    transforms.append(
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        )
    )

    # Convert to tensor
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def get_val_transform(image_size: int = 224) -> A.Compose:
    """Get validation/inference transform (no augmentation).

    Args:
        image_size: Target image size

    Returns:
        Albumentations Compose pipeline for validation
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


# =============================================================================
# Dataset
# =============================================================================
class ImageCaptionDataset(Dataset):
    """Dataset for image-caption pairs with albumentations augmentation.

    Supports two formats:
    1. JSON format: captions_file is a JSON mapping image_name -> caption string
    2. TXT format (CelebA-HQ style): captions_dir contains .txt files with same
       name as images, each having multiple caption lines (uses first line)

    Args:
        image_dir: Path to directory containing images (.jpg, .png, etc.)
        captions_file: Path to JSON file OR directory with .txt caption files
        image_size: Target image size
        tokenizer_model: HuggingFace tokenizer model name
        max_caption_length: Maximum caption token length
        augmentation_strength: "none", "light", "medium", or "strong"
        use_horizontal_flip: Whether to use horizontal flip augmentation
        is_training: If True, use augmentation; if False, use validation transform
        caption_line_idx: Which line to use from multi-line caption files (default: 0)
    """

    def __init__(
        self,
        image_dir: str,
        captions_file: str,
        image_size: int = 224,
        tokenizer_model: str = "google-bert/bert-base-uncased",
        max_caption_length: int = 32,  # Short captions (~15-20 words)
        augmentation_strength: str = "strong",
        use_horizontal_flip: bool = True,
        is_training: bool = True,
        caption_line_idx: int = 0,
    ):
        self.image_dir = Path(image_dir)
        self.captions_path = Path(captions_file)
        self.image_size = image_size
        self.max_caption_length = max_caption_length
        self._use_dummy = False
        self.is_training = is_training
        self.augmentation_strength = augmentation_strength
        self.caption_line_idx = caption_line_idx
        self._caption_mode = "json"  # or "txt_dir"

        # Load HuggingFace tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

        # Determine caption format and load data
        self.captions = {}
        self.image_names = []

        if self.captions_path.is_dir():
            # TXT directory mode (CelebA-HQ style)
            self._caption_mode = "txt_dir"
            self._load_txt_dir_format()
        elif self.captions_path.exists() and self.captions_path.suffix == ".json":
            # JSON file mode
            self._caption_mode = "json"
            self._load_json_format()
        else:
            logger.warning(f"Captions path not found or unsupported: {captions_file}. Using dummy data.")
            self._use_dummy = True

        if not self._use_dummy:
            logger.info(f"Loaded {len(self.image_names)} image-caption pairs ({self._caption_mode} mode)")

        # Set up albumentations transforms
        if is_training and augmentation_strength != "none":
            self.transform = get_diffusion_augmentation(
                image_size=image_size,
                strength=augmentation_strength,
                use_horizontal_flip=use_horizontal_flip,
            )
            logger.info(f"Using {augmentation_strength} augmentation for training")
        else:
            self.transform = get_val_transform(image_size=image_size)
            logger.info("Using validation transform (no augmentation)")

        self.inverse_transform = transforms.Compose([
            transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            ),
        ])

    def _load_json_format(self):
        """Load captions from JSON file mapping image_name -> caption."""
        with open(self.captions_path, 'r') as f:
            self.captions = json.load(f)
        self.image_names = list(self.captions.keys())
        # Filter to only existing images
        self.image_names = [n for n in self.image_names if (self.image_dir / n).exists()]
        if not self.image_names:
            logger.warning("No matching images found for JSON captions. Using dummy data.")
            self._use_dummy = True

    def _load_txt_dir_format(self):
        """Load captions from directory of .txt files (CelebA-HQ style).

        Each .txt file has the same stem as the image file and contains
        multiple caption lines. Uses caption_line_idx to select which line.
        """
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        image_files = [
            f for f in self.image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            logger.warning(f"No images found in {self.image_dir}. Using dummy data.")
            self._use_dummy = True
            return

        # Match images to caption files
        matched = 0
        for img_path in image_files:
            img_stem = img_path.stem  # filename without extension
            caption_file = self.captions_path / f"{img_stem}.txt"

            if caption_file.exists():
                self.image_names.append(img_path.name)
                matched += 1

        if not self.image_names:
            logger.warning(f"No matching caption files found in {self.captions_path}. Using dummy data.")
            self._use_dummy = True
        else:
            logger.info(f"Matched {matched}/{len(image_files)} images to caption files")

    def _get_caption(self, image_name: str) -> str:
        """Get caption for an image based on the caption mode."""
        if self._caption_mode == "json":
            return self.captions[image_name]
        else:  # txt_dir mode
            img_stem = Path(image_name).stem
            caption_file = self.captions_path / f"{img_stem}.txt"
            with open(caption_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Get the specified line (default: first line)
            line_idx = min(self.caption_line_idx, len(lines) - 1)
            return lines[line_idx].strip()

    def __len__(self) -> int:
        if self._use_dummy:
            return 1000
        return len(self.image_names)

    def __getitem__(self, idx: int) -> dict:
        if self._use_dummy:
            # Generate random dummy data (skip augmentation for dummy)
            image = torch.randn(3, self.image_size, self.image_size)
            # Dummy caption tokens
            tokens = self.tokenizer(
                "a random generated test image caption",
                padding="max_length",
                truncation=True,
                max_length=self.max_caption_length,
                return_tensors="pt",
            )
            return {
                "image": image,
                "input_ids": tokens["input_ids"].squeeze(0),
                "attention_mask": tokens["attention_mask"].squeeze(0),
            }

        image_name = self.image_names[idx]
        image_path = self.image_dir / image_name

        # Load image as numpy array for albumentations
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)  # (H, W, C) uint8 [0-255]

        # Apply albumentations transform
        transformed = self.transform(image=image_np)
        image_tensor = transformed["image"]  # From ToTensorV2: (C, H, W) float

        # Get caption using appropriate method
        caption = self._get_caption(image_name)
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_caption_length,
            return_tensors="pt",
        )

        return {
            "image": image_tensor,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }


# =============================================================================
# Simple CNN Image Encoder (TIMM-based, outputs spatial tokens like ACT)
# =============================================================================
class SimpleCNNEncoder(nn.Module):
    """Simple CNN encoder using TIMM backbone - outputs spatial feature map (B, C, H, W).

    NO POOLING - keeps all spatial tokens for transformer input.
    """

    def __init__(
        self,
        backbone: str = "timm/resnet18.a1_in1k",
        pretrained: bool = True,
        use_group_norm: bool = True,
    ):
        super().__init__()
        # Use TIMM via transformers
        from transformers import TimmBackbone, TimmBackboneConfig
        from refactoring.models.layers.convert_layers import replace_batchnorm_with_groupnorm

        config = TimmBackboneConfig(backbone, use_pretrained_backbone=pretrained, features_only=True)
        self.backbone = TimmBackbone(config=config)

        if use_group_norm:
            self.backbone = replace_batchnorm_with_groupnorm(self.backbone)

        self.feature_dim = self.backbone.num_features[-1]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to spatial feature map.

        Args:
            images: (B, C, H, W)

        Returns:
            Spatial features (B, feature_dim, H', W') - NO POOLING
        """
        outputs = self.backbone(images)
        features = outputs.feature_maps[-1]  # (B, C, H', W')
        return features


# =============================================================================
# Text Encoder (Using HuggingFace Transformers - outputs all tokens)
# =============================================================================
class TextEncoder(nn.Module):
    """Text encoder using HuggingFace transformers - outputs ALL sequence tokens.

    NO POOLING - keeps all text tokens for transformer input (like ACT).
    """

    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        pretrained: bool = True,
        frozen: bool = True,
    ):
        super().__init__()
        self.model_name = model_name

        # Load HuggingFace model
        config = AutoConfig.from_pretrained(model_name)
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name, attn_implementation="sdpa")
        else:
            self.encoder = AutoModel.from_config(config, attn_implementation="sdpa")

        self.hidden_size = config.hidden_size

        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text to sequence of token embeddings.

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len), 1 for valid, 0 for padding

        Returns:
            Tuple of:
                - Text token embeddings (B, seq_len, hidden_size) - ALL tokens, NO POOLING
                - Padding mask (B, seq_len) where True = padded (inverted from attention_mask)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # Return ALL tokens, not just CLS
        # Invert attention_mask: attention_mask=1 means valid, we want padding_mask=True for padded
        padding_mask = (attention_mask == 0)
        return outputs.last_hidden_state, padding_mask


# =============================================================================
# Posterior Encoder q(z|x, c) - ACT-style with all tokens + 2D PE for images
# =============================================================================
class PosteriorEncoder(nn.Module):
    """Posterior encoder q(z|x, c) using TransformerEncoder.

    Takes ALL image tokens + ALL text tokens, concatenates them with a CLS token,
    and uses transformer encoder to produce latent distribution.
    Follows ACT pattern - NO POOLING of inputs.

    Uses 2D positional encoding for image tokens (spatial) and 1D PE for text tokens.
    """

    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int,
        image_feature_dim: int,
        text_feature_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

        # Project image and text features to common embedding dim
        self.image_proj = nn.Linear(image_feature_dim, embedding_dim)
        self.text_proj = nn.Linear(text_feature_dim, embedding_dim)

        # CLS token for aggregation (appended at END like in ACT)
        self.cls_token = nn.Embedding(1, embedding_dim)

        # 2D positional encoding for image tokens (spatial features)
        self.image_pos_encoding = SinusoidalPositionalEncoding2D(
            embedding_dimension=embedding_dim,
            normalize=True,
        )

        # 1D positional encoding for text tokens + CLS
        self.text_pos_encoding = SinusoidalPositionalEncoding1D(
            embedding_dimension=embedding_dim,
            maximum_length=256,  # text_tokens + CLS
        )

        # TransformerEncoder (reusing existing module)
        self.encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                embedding_dimension=embedding_dim,
                number_of_heads=num_heads,
                feedforward_dimension=feedforward_dim,
                dropout=dropout,
                normalize_before=False,
            ),
            number_of_layers=num_layers,
            normalization=nn.LayerNorm(embedding_dim),
        )

        # Project CLS output to latent stats (mu, logvar)
        self.to_latent = nn.Linear(embedding_dim, latent_dim * 2)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_spatial_shape: tuple[int, int],
        text_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode to latent distribution using ALL tokens.

        Args:
            image_tokens: Image spatial tokens (B, H*W, image_feature_dim)
            text_tokens: Text sequence tokens (B, seq_len, text_feature_dim)
            image_spatial_shape: Tuple of (H, W) for computing 2D PE
            text_padding_mask: Text padding mask (B, seq_len), True = padded

        Returns:
            Dict with LATENT_KEY, MU_KEY, LOGVAR_KEY
        """
        B = image_tokens.shape[0]
        num_image_tokens = image_tokens.shape[1]
        H, W = image_spatial_shape

        # Project to common embedding dim
        image_emb = self.image_proj(image_tokens)  # (B, H*W, emb_dim)
        text_emb = self.text_proj(text_tokens)  # (B, seq_len, emb_dim)

        # CLS token (at the end like ACT)
        cls = self.cls_token.weight.unsqueeze(0).expand(B, 1, -1)  # (B, 1, emb_dim)

        # Compute 2D positional encoding for image tokens
        # SinusoidalPositionalEncoding2D expects (B, C, H, W), outputs (B, emb_dim, H, W)
        dummy_spatial = torch.zeros(1, 1, H, W, device=image_tokens.device)
        image_pe_2d = self.image_pos_encoding(dummy_spatial)  # (1, emb_dim, H, W)
        image_pe_flat = image_pe_2d.flatten(2).transpose(1, 2)  # (1, H*W, emb_dim)
        image_pe = image_pe_flat.expand(B, -1, -1)  # (B, H*W, emb_dim)

        # Compute 1D positional encoding for text tokens + CLS
        num_text_and_cls = text_tokens.shape[1] + 1
        dummy_text = torch.zeros(1, num_text_and_cls, device=text_tokens.device)
        text_pe = self.text_pos_encoding(dummy_text)  # (1, seq_len+1, emb_dim)
        text_pe = text_pe.expand(B, -1, -1)  # (B, seq_len+1, emb_dim)

        # Build sequence: [image_tokens, text_tokens, CLS]
        sequence = torch.cat([image_emb, text_emb, cls], dim=1)  # (B, H*W + seq_len + 1, emb_dim)

        # Build combined positional encoding
        pos_enc = torch.cat([image_pe, text_pe], dim=1)  # (B, H*W + seq_len + 1, emb_dim)

        # Build padding mask: image tokens are never padded, text tokens may be, CLS never padded
        if text_padding_mask is not None:
            image_mask = torch.zeros(B, num_image_tokens, dtype=torch.bool, device=image_tokens.device)
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=image_tokens.device)
            padding_mask = torch.cat([image_mask, text_padding_mask, cls_mask], dim=1)
        else:
            padding_mask = None

        # Encode with transformer
        encoded = self.encoder(
            sequence,
            positional_encoding=pos_enc,
            source_key_padding_mask=padding_mask,
        )

        # Extract CLS token output (last position)
        cls_output = encoded[:, -1, :]  # (B, emb_dim)

        # Get latent stats
        stats = self.to_latent(cls_output)
        mu, logvar = stats.chunk(2, dim=-1)

        # Reparameterize
        z = reparametrize(mu, logvar)

        return {
            LATENT_KEY: z,
            MU_KEY: mu,
            LOGVAR_KEY: logvar,
        }


# =============================================================================
# Prior Encoder p(z|c) - ACT-style with all text tokens
# =============================================================================
class PriorEncoder(nn.Module):
    """Prior encoder p(z|c) using TransformerEncoder.

    Takes ALL text tokens (no image), concatenates with CLS token,
    and produces latent distribution for generation.
    Follows ACT pattern - NO POOLING.
    """

    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int,
        text_feature_dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

        # Project text features to embedding dim
        self.text_proj = nn.Linear(text_feature_dim, embedding_dim)

        # CLS token (at the end like ACT)
        self.cls_token = nn.Embedding(1, embedding_dim)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding1D(
            embedding_dimension=embedding_dim,
            maximum_length=128,  # text_tokens + CLS
        )

        # TransformerEncoder
        self.encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                embedding_dimension=embedding_dim,
                number_of_heads=num_heads,
                feedforward_dimension=feedforward_dim,
                dropout=dropout,
                normalize_before=False,
            ),
            number_of_layers=num_layers,
            normalization=nn.LayerNorm(embedding_dim),
        )

        # Project to latent stats
        self.to_latent = nn.Linear(embedding_dim, latent_dim * 2)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode to prior latent distribution using ALL text tokens.

        Args:
            text_tokens: Text sequence tokens (B, seq_len, text_feature_dim)
            text_padding_mask: Text padding mask (B, seq_len), True = padded

        Returns:
            Dict with PRIOR_LATENT_KEY, PRIOR_MU_KEY, PRIOR_LOGVAR_KEY
        """
        B = text_tokens.shape[0]
        num_text_tokens = text_tokens.shape[1]

        # Project to embedding dim
        text_emb = self.text_proj(text_tokens)  # (B, seq_len, emb_dim)

        # CLS token (at the end)
        cls = self.cls_token.weight.unsqueeze(0).expand(B, 1, -1)  # (B, 1, emb_dim)

        # Build sequence: [text_tokens, CLS]
        sequence = torch.cat([text_emb, cls], dim=1)  # (B, seq_len + 1, emb_dim)

        # Build padding mask
        if text_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=text_tokens.device)
            padding_mask = torch.cat([text_padding_mask, cls_mask], dim=1)
        else:
            padding_mask = None

        # Add positional encoding
        pos_enc = self.pos_encoding(sequence)

        # Encode
        encoded = self.encoder(
            sequence,
            positional_encoding=pos_enc,
            source_key_padding_mask=padding_mask,
        )

        # Extract CLS token output (last position)
        cls_output = encoded[:, -1, :]

        # Get latent stats
        stats = self.to_latent(cls_output)
        mu, logvar = stats.chunk(2, dim=-1)

        # Reparameterize
        z = reparametrize(mu, logvar)

        return {
            PRIOR_LATENT_KEY: z,
            PRIOR_MU_KEY: mu,
            PRIOR_LOGVAR_KEY: logvar,
        }

    def sample(
        self,
        text_tokens: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample from prior."""
        return self.forward(text_tokens, text_padding_mask)[PRIOR_LATENT_KEY]


# =============================================================================
# Image Decoder p(x|z, c) - Cross-attention to ALL text tokens
# =============================================================================
class ImageDecoder(nn.Module):
    """Image decoder using BidirectionalDecoder + CNN upsampler.

    Decodes latent z conditioned on ALL text tokens (not pooled).
    Uses cross-attention from patch queries to [latent, text_tokens] context.
    """

    def __init__(
        self,
        embedding_dim: int,
        latent_dim: int,
        text_feature_dim: int,
        image_channels: int = 3,
        image_size: int = 224,
        num_heads: int = 8,
        num_layers: int = 6,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.embedding_dim = embedding_dim

        # Project latent to embedding
        self.latent_proj = nn.Linear(latent_dim, embedding_dim)

        # Project text features to embedding dim
        self.text_proj = nn.Linear(text_feature_dim, embedding_dim)

        # Learnable query tokens for image patches (14x14 = 196 for 224px with 16px patches)
        self.patch_size = 16
        self.num_patches_per_side = image_size // self.patch_size
        self.num_patches = self.num_patches_per_side ** 2
        self.patch_queries = nn.Parameter(torch.randn(1, self.num_patches, embedding_dim))

        # BidirectionalDecoder (reusing existing module)
        self.decoder = BidirectionalDecoder(
            number_of_layers=num_layers,
            embedding_dimension=embedding_dim,
            number_of_heads=num_heads,
            number_of_key_value_heads=num_heads // 2,
            feedforward_dimension=feedforward_dim,
            dropout=dropout,
            attention_dropout=0.0,
            normalization_type="rmsnorm",
            attention_type="gqa",
        )

        # Project to patch pixels
        self.patch_proj = nn.Linear(embedding_dim, self.patch_size * self.patch_size * image_channels)

        # CNN refinement upsampler
        self.refiner = nn.Sequential(
            nn.Conv2d(image_channels, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, image_channels, 3, 1, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        text_tokens: torch.Tensor,
        text_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode latent to image using cross-attention to ALL text tokens.

        Args:
            z: Latent vector (B, latent_dim)
            text_tokens: Text sequence tokens (B, seq_len, text_feature_dim)
            text_padding_mask: Text padding mask (B, seq_len), True = padded

        Returns:
            Reconstructed image (B, C, H, W)
        """
        B = z.shape[0]

        # Project latent to embedding
        z_proj = self.latent_proj(z).unsqueeze(1)  # (B, 1, emb_dim)

        # Project text tokens to embedding dim
        text_emb = self.text_proj(text_tokens)  # (B, seq_len, emb_dim)

        # Build context: [latent, text_tokens] for cross-attention
        context = torch.cat([z_proj, text_emb], dim=1)  # (B, 1 + seq_len, emb_dim)

        # Build memory padding mask (latent never padded, text tokens may be)
        if text_padding_mask is not None:
            latent_mask = torch.zeros(B, 1, dtype=torch.bool, device=z.device)
            memory_padding_mask = torch.cat([latent_mask, text_padding_mask], dim=1)
        else:
            memory_padding_mask = None

        # Expand queries
        queries = self.patch_queries.expand(B, -1, -1)

        # Decode with cross-attention to context
        decoded = self.decoder(
            hidden_states=queries,
            encoded_features=context,
            memory_padding_mask=memory_padding_mask,
        )  # (B, num_patches, emb_dim)

        # Project to patch pixels
        patches = self.patch_proj(decoded)  # (B, num_patches, patch_size^2 * C)

        # Reshape to image
        h = w = self.num_patches_per_side
        C = 3
        patches = patches.view(B, h, w, self.patch_size, self.patch_size, C)
        patches = patches.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        image = patches.reshape(B, C, self.image_size, self.image_size)

        # Refine
        image = self.refiner(image)

        return image


# =============================================================================
# Full CVAE Model - ACT-style with all tokens
# =============================================================================
class ConditionalVAE(nn.Module):
    """Conditional VAE for image generation.

    Architecture follows ACT pattern - NO POOLING anywhere:
        - Image encoder: CNN outputs spatial features (B, C, H, W) → flatten to (B, H*W, C)
        - Text encoder: BERT outputs ALL tokens (B, seq_len, hidden_size)
        - Posterior: q(z|x, c) - TransformerEncoder over [image_tokens, text_tokens, CLS]
        - Prior: p(z|c) - TransformerEncoder over [text_tokens, CLS]
        - Decoder: p(x|z, c) - Cross-attention from patch queries to [latent, text_tokens]
    """

    def __init__(self, config: CVAEConfig):
        super().__init__()
        self.config = config

        # Image encoder - outputs spatial features (B, C, H', W')
        self.image_encoder = SimpleCNNEncoder(
            backbone=config.image_backbone,
            pretrained=True,
            use_group_norm=True,
        )

        # Text encoder - outputs ALL tokens (B, seq_len, hidden_size)
        self.text_encoder = TextEncoder(
            model_name=config.text_model,
            pretrained=True,
            frozen=True,
        )

        # Get feature dimensions
        image_feature_dim = self.image_encoder.feature_dim
        text_feature_dim = self.text_encoder.hidden_size

        # Posterior encoder q(z|x, c) - takes ALL tokens
        self.posterior = PosteriorEncoder(
            embedding_dim=config.embedding_dim,
            latent_dim=config.latent_dim,
            image_feature_dim=image_feature_dim,
            text_feature_dim=text_feature_dim,
            num_heads=config.num_heads,
            num_layers=config.num_encoder_layers,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
        )

        # Prior encoder p(z|c) - takes ALL text tokens
        self.prior = PriorEncoder(
            embedding_dim=config.embedding_dim,
            latent_dim=config.latent_dim,
            text_feature_dim=text_feature_dim,
            num_heads=config.num_heads,
            num_layers=config.num_encoder_layers,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
        )

        # Image decoder p(x|z, c) - cross-attention to ALL text tokens
        self.decoder = ImageDecoder(
            embedding_dim=config.embedding_dim,
            latent_dim=config.latent_dim,
            text_feature_dim=text_feature_dim,
            image_channels=config.image_channels,
            image_size=config.image_size,
            num_heads=config.num_heads,
            num_layers=config.num_decoder_layers,
            feedforward_dim=config.feedforward_dim,
            dropout=config.dropout,
        )

    def encode_image(self, images: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Encode image to spatial tokens.

        Args:
            images: (B, C, H, W)

        Returns:
            Tuple of:
                - Image tokens (B, H'*W', feature_dim) - flattened spatial features
                - Spatial shape (H', W') for 2D positional encoding
        """
        features = self.image_encoder(images)  # (B, C, H', W')
        B, C, H, W = features.shape
        # Flatten spatial dims and transpose to (B, H*W, C)
        tokens = features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        return tokens, (H, W)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode text to sequence tokens.

        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)

        Returns:
            Tuple of:
                - Text tokens (B, seq_len, hidden_size)
                - Padding mask (B, seq_len), True = padded
        """
        return self.text_encoder(input_ids, attention_mask)

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for training.

        Args:
            images: Input images (B, C, H, W)
            input_ids: Caption token IDs (B, seq_len)
            attention_mask: Caption attention mask (B, seq_len)

        Returns:
            Dict with reconstructions and latent outputs
        """
        # Encode image to tokens (B, H*W, C) + spatial shape for 2D PE
        image_tokens, image_spatial_shape = self.encode_image(images)

        # Encode text to tokens (B, seq_len, hidden_size) + padding mask
        text_tokens, text_padding_mask = self.encode_text(input_ids, attention_mask)

        # Posterior q(z|x, c) - uses ALL tokens with 2D PE for image
        posterior_out = self.posterior(image_tokens, text_tokens, image_spatial_shape, text_padding_mask)

        # Prior p(z|c) - uses ALL text tokens
        prior_out = self.prior(text_tokens, text_padding_mask)

        # Decode using posterior sample - cross-attention to ALL text tokens
        z = posterior_out[LATENT_KEY]
        reconstructed = self.decoder(z, text_tokens, text_padding_mask)

        # Combine outputs
        outputs = {}
        outputs.update(posterior_out)
        outputs.update(prior_out)
        outputs["reconstructed"] = reconstructed

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Generate images from captions.

        Args:
            input_ids: Caption token IDs (B, seq_len)
            attention_mask: Caption attention mask (B, seq_len)
            num_samples: Number of samples per caption

        Returns:
            Generated images (B * num_samples, C, H, W)
        """
        self.eval()

        # Encode text to tokens
        text_tokens, text_padding_mask = self.encode_text(input_ids, attention_mask)

        if num_samples > 1:
            text_tokens = text_tokens.repeat_interleave(num_samples, dim=0)
            text_padding_mask = text_padding_mask.repeat_interleave(num_samples, dim=0)

        # Sample from prior using ALL text tokens
        z = self.prior.sample(text_tokens, text_padding_mask)

        # Decode with cross-attention to ALL text tokens
        images = self.decoder(z, text_tokens, text_padding_mask)

        return images


# =============================================================================
# Loss Functions (Using existing MaximumMeanDiscrepancyLoss from components.py)
# =============================================================================
class CVAELoss(nn.Module):
    """Combined loss for CVAE training: MMD + Reconstruction.

    Uses existing MaximumMeanDiscrepancyLoss from refactoring.metrics.components.
    """

    def __init__(
        self,
        mmd_weight: float = 1.0,
        recon_weight: float = 1.0,
    ):
        super().__init__()
        self.mmd_weight = mmd_weight
        self.recon_weight = recon_weight

        # Reuse existing MMD loss from refactoring.metrics.components
        self.mmd_loss_fn = MaximumMeanDiscrepancyLoss(weight=mmd_weight)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute CVAE loss.

        Args:
            outputs: Model outputs containing reconstructed, z, prior_z, mu, logvar, etc.
            target_images: Ground truth images

        Returns:
            Dict with total_loss and component losses
        """
        reconstructed = outputs["reconstructed"]

        # Reconstruction loss (L1 + MSE)
        recon_l1 = F.l1_loss(reconstructed, target_images)
        recon_mse = F.mse_loss(reconstructed, target_images)
        recon_loss = recon_l1 + 0.1 * recon_mse

        # MMD loss using existing component
        # MaximumMeanDiscrepancyLoss expects predictions dict with LATENT_KEY, PRIOR_LATENT_KEY, etc.
        mmd_output: LossOutput = self.mmd_loss_fn(
            predictions=outputs,
            targets={},  # MMD doesn't need targets, only latent samples
        )
        mmd_loss = mmd_output.total_loss

        # Total loss
        total_loss = self.recon_weight * recon_loss + mmd_loss  # mmd already weighted

        return LossOutput(
            total_loss=total_loss,
            component_losses={
                "recon_loss": recon_loss,
                "recon_l1": recon_l1,
                "recon_mse": recon_mse,
                "mmd_loss": mmd_loss,
            },
        )


# =============================================================================
# Evaluation Metrics (FID)
# =============================================================================
class FIDCalculator:
    """FID score calculator using Inception features."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._inception = None

    def _get_inception(self):
        if self._inception is None:
            try:
                from torchvision.models import inception_v3, Inception_V3_Weights
                self._inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
                self._inception.fc = nn.Identity()
                self._inception.eval()
                self._inception.to(self.device)
            except Exception as e:
                logger.warning(f"Could not load Inception: {e}")
                return None
        return self._inception

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor | None:
        inception = self._get_inception()
        if inception is None:
            return None

        # Resize to 299x299
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)

        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        with torch.no_grad():
            features = inception(images)
        return features

    def compute_fid(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
    ) -> float:
        """Compute FID between real and generated images."""
        real_features = self._extract_features(real_images)
        gen_features = self._extract_features(generated_images)

        if real_features is None or gen_features is None:
            return float('nan')

        # Compute statistics
        mu_real = real_features.mean(dim=0)
        mu_gen = gen_features.mean(dim=0)

        diff = mu_real - mu_gen
        fid_approx = diff.dot(diff).item()

        return fid_approx


# =============================================================================
# Training Loop
# =============================================================================
def train_epoch(
    model: ConditionalVAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CVAELoss,
    device: str,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_losses = {"total_loss": 0.0, "recon_loss": 0.0, "mmd_loss": 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()

        outputs = model(images, input_ids, attention_mask)
        loss_output = loss_fn(outputs, images)

        loss_output.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_losses["total_loss"] += loss_output.total_loss.item()
        total_losses["recon_loss"] += loss_output.component_losses["recon_loss"].item()
        total_losses["mmd_loss"] += loss_output.component_losses["mmd_loss"].item()
        num_batches += 1

        pbar.set_postfix({
            "loss": f"{loss_output.total_loss.item():.4f}",
            "recon": f"{loss_output.component_losses['recon_loss'].item():.4f}",
            "mmd": f"{loss_output.component_losses['mmd_loss'].item():.4f}",
        })

    for k in total_losses:
        total_losses[k] /= num_batches

    return total_losses


@torch.no_grad()
def evaluate(
    model: ConditionalVAE,
    dataloader: DataLoader,
    loss_fn: CVAELoss,
    device: str,
    fid_calculator: FIDCalculator | None = None,
) -> dict[str, float]:
    """Evaluate model."""
    model.eval()
    total_losses = {"total_loss": 0.0, "recon_loss": 0.0, "mmd_loss": 0.0}
    num_batches = 0

    real_for_fid = []
    gen_for_fid = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(images, input_ids, attention_mask)
        loss_output = loss_fn(outputs, images)

        total_losses["total_loss"] += loss_output.total_loss.item()
        total_losses["recon_loss"] += loss_output.component_losses["recon_loss"].item()
        total_losses["mmd_loss"] += loss_output.component_losses["mmd_loss"].item()
        num_batches += 1

        if fid_calculator and len(real_for_fid) < 32:
            real_for_fid.append(images.cpu())
            gen_for_fid.append(outputs["reconstructed"].cpu())

    for k in total_losses:
        total_losses[k] /= num_batches

    if fid_calculator and real_for_fid:
        real_all = torch.cat(real_for_fid, dim=0)[:512].to(device)
        gen_all = torch.cat(gen_for_fid, dim=0)[:512].to(device)
        total_losses["fid"] = fid_calculator.compute_fid(real_all, gen_all)

    return total_losses


def save_samples(
    model: ConditionalVAE,
    dataloader: DataLoader,
    device: str,
    output_dir: str,
    num_samples: int = 8,
    epoch: int | None = None,
    use_wandb: bool = False,
) -> dict[str, torch.Tensor] | None:
    """Generate and save sample images, optionally logging to wandb.

    Args:
        model: CVAE model
        dataloader: Data loader
        device: Device string
        output_dir: Directory to save images
        num_samples: Number of samples to generate
        epoch: Current epoch (for wandb logging)
        use_wandb: Whether to log to wandb

    Returns:
        Dict with denormalized images if use_wandb, else None
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    batch = next(iter(dataloader))
    images = batch["image"][:num_samples].to(device)
    input_ids = batch["input_ids"][:num_samples].to(device)
    attention_mask = batch["attention_mask"][:num_samples].to(device)

    with torch.no_grad():
        outputs = model(images, input_ids, attention_mask)
        reconstructed = outputs["reconstructed"]
        generated = model.generate(input_ids, attention_mask, num_samples=1)

    # Denormalize for visualization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    images = images * std + mean
    reconstructed = reconstructed * std + mean
    generated = generated * std + mean

    images = images.clamp(0, 1)
    reconstructed = reconstructed.clamp(0, 1)
    generated = generated.clamp(0, 1)

    # Save to files
    save_image(images, os.path.join(output_dir, "real.png"), nrow=4)
    save_image(reconstructed, os.path.join(output_dir, "reconstructed.png"), nrow=4)
    save_image(generated, os.path.join(output_dir, "generated.png"), nrow=4)

    logger.info(f"Saved samples to {output_dir}")

    # Log to wandb
    if use_wandb and wandb.run is not None:
        # Create image grids for wandb
        from torchvision.utils import make_grid

        real_grid = make_grid(images, nrow=4)
        recon_grid = make_grid(reconstructed, nrow=4)
        gen_grid = make_grid(generated, nrow=4)

        wandb_images = {
            "samples/real": wandb.Image(real_grid.permute(1, 2, 0).cpu().numpy(), caption="Real Images"),
            "samples/reconstructed": wandb.Image(recon_grid.permute(1, 2, 0).cpu().numpy(), caption="Reconstructed"),
            "samples/generated": wandb.Image(gen_grid.permute(1, 2, 0).cpu().numpy(), caption="Generated from Prior"),
        }
        wandb.log(wandb_images, step=epoch)

    return {"real": images, "reconstructed": reconstructed, "generated": generated}


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train CVAE for conditional image generation")
    parser.add_argument("--image_dir", type=str, default="/mnt/cluster/workspaces/mazzalore/celeb/image/images")
    parser.add_argument("--captions_file", type=str, default="/mnt/cluster/workspaces/mazzalore/celeb/text/celeba-caption")
    parser.add_argument("--output_dir", type=str, default="./outputs/cvae")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--mmd_weight", type=float, default=1.0)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--text_model", type=str, default="google-bert/bert-base-uncased")
    # Augmentation arguments
    parser.add_argument("--augmentation", type=str, default="strong",
                        choices=["none", "light", "medium", "strong"],
                        help="Augmentation strength for training")
    parser.add_argument("--no_horizontal_flip", action="store_true",
                        help="Disable horizontal flip augmentation")
    parser.add_argument("--caption_line_idx", type=int, default=0,
                        help="Which caption line to use for multi-line caption files (default: 0)")
    # WandB arguments
    parser.add_argument("--wandb_project", type=str, default="cvae-image-generation",
                        help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="WandB entity (team/username)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging")
    args = parser.parse_args()

    config = CVAEConfig(
        image_size=args.image_size,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        mmd_weight=args.mmd_weight,
        recon_weight=args.recon_weight,
        text_model=args.text_model,
        augmentation_strength=args.augmentation,
        use_horizontal_flip=not args.no_horizontal_flip,
    )

    logger.info(f"Config: {config}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize WandB
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                "image_size": config.image_size,
                "latent_dim": config.latent_dim,
                "embedding_dim": config.embedding_dim,
                "num_encoder_layers": config.num_encoder_layers,
                "num_decoder_layers": config.num_decoder_layers,
                "num_heads": config.num_heads,
                "feedforward_dim": config.feedforward_dim,
                "dropout": config.dropout,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "mmd_weight": config.mmd_weight,
                "recon_weight": config.recon_weight,
                "text_model": config.text_model,
                "augmentation_strength": config.augmentation_strength,
                "use_horizontal_flip": config.use_horizontal_flip,
                "image_backbone": config.image_backbone,
            },
        )
        logger.info(f"WandB initialized: {wandb.run.url}")

    # Dataset & DataLoader (training with augmentation)
    train_dataset = ImageCaptionDataset(
        image_dir=args.image_dir,
        captions_file=args.captions_file,
        image_size=config.image_size,
        tokenizer_model=config.text_model,
        max_caption_length=config.max_caption_length,
        augmentation_strength=config.augmentation_strength,
        use_horizontal_flip=config.use_horizontal_flip,
        is_training=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = ConditionalVAE(config).to(config.device)
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    if use_wandb:
        wandb.log({"model/num_parameters": num_params})

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    # Loss
    loss_fn = CVAELoss(mmd_weight=config.mmd_weight, recon_weight=config.recon_weight)

    # FID
    fid_calculator = FIDCalculator(device=config.device)

    # Training
    best_loss = float("inf")

    for epoch in range(config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        train_losses = train_epoch(model, train_loader, optimizer, loss_fn, config.device)
        logger.info(f"Train - Loss: {train_losses['total_loss']:.4f}, "
                    f"Recon: {train_losses['recon_loss']:.4f}, MMD: {train_losses['mmd_loss']:.4f}")

        # Log training metrics to wandb
        if use_wandb:
            wandb.log({
                "train/total_loss": train_losses["total_loss"],
                "train/recon_loss": train_losses["recon_loss"],
                "train/mmd_loss": train_losses["mmd_loss"],
                "train/learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch + 1,
            }, step=epoch + 1)

        scheduler.step()

        # Evaluation every eval_every epochs
        if (epoch + 1) % args.eval_every == 0:
            eval_losses = evaluate(model, train_loader, loss_fn, config.device, fid_calculator)
            fid_str = f", FID: {eval_losses.get('fid', float('nan')):.2f}"
            logger.info(f"Eval - Loss: {eval_losses['total_loss']:.4f}, "
                        f"Recon: {eval_losses['recon_loss']:.4f}, MMD: {eval_losses['mmd_loss']:.4f}{fid_str}")

            # Log eval metrics to wandb
            if use_wandb:
                eval_log = {
                    "eval/total_loss": eval_losses["total_loss"],
                    "eval/recon_loss": eval_losses["recon_loss"],
                    "eval/mmd_loss": eval_losses["mmd_loss"],
                }
                if "fid" in eval_losses and not np.isnan(eval_losses["fid"]):
                    eval_log["eval/fid"] = eval_losses["fid"]
                wandb.log(eval_log, step=epoch + 1)

            if eval_losses["total_loss"] < best_loss:
                best_loss = eval_losses["total_loss"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                }, os.path.join(args.output_dir, "best_model.pt"))
                logger.info(f"Saved best model (loss: {best_loss:.4f})")

                if use_wandb:
                    wandb.log({"eval/best_loss": best_loss}, step=epoch + 1)

        # Generate and save samples every 10 epochs (or save_every)
        if (epoch + 1) % args.save_every == 0:
            save_samples(
                model,
                train_loader,
                config.device,
                os.path.join(args.output_dir, f"samples_epoch_{epoch + 1}"),
                epoch=epoch + 1,
                use_wandb=use_wandb,
            )

    # Save final model
    torch.save({
        "epoch": config.num_epochs,
        "model_state_dict": model.state_dict(),
        "config": config,
    }, os.path.join(args.output_dir, "final_model.pt"))

    # Final wandb logging
    if use_wandb:
        wandb.log({"final/best_loss": best_loss})
        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()