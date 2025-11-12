"""Configuration classes for observation encoders of different data modalities."""
from dataclasses import dataclass, field

from omegaconf import MISSING

from refactoring.data.constants import Cameras
from refactoring.models.encoding.encoders.constants import LanguageEncoderType


@dataclass
class EncoderConfig:
    """Base encoder configuration."""
    _target_: str = MISSING
    #: What keys to read from the sample observation dict
    input_keys: str | list[str] = MISSING
    pretrained: bool = False
    frozen : bool = False


@dataclass
class DepthEncoderConfig(EncoderConfig):
    """Depth CNN encoder configuration."""
    _target_: str = "refactoring.models.encoding.encoders.depth.cnn.DepthCNNEncoder"
    backbone: str = MISSING
    use_group_norm: bool = True
    spatial_softmax: bool = True


@dataclass
class DFormerEncoderConfig(EncoderConfig):
    """DFormer RGB+Depth encoder configuration."""
    _target_: str = "refactoring.models.encoding.encoders.depth.dformerv2.DFormerEncoder"
    input_keys: list[str] = field(default_factory=lambda: [Cameras.LEFT.value, Cameras.DEPTH.value])
    variant: str = "S"
    checkpoint: str | None = None


@dataclass
class StateEncoderConfig(EncoderConfig):
    """State encoder configuration for proprioceptive data."""
    _target_: str = "refactoring.models.encoding.encoders.proprioceptive.base.ProprioceptiveEncoder"
    hidden_dims: list[int] = field(default_factory=lambda: [128])
    activation: str = "relu"
    dropout: float = 0.1


@dataclass
class LanguageEncoderConfig(EncoderConfig):
    """Language encoder configuration."""
    _target_: str = "refactoring.models.encoding.encoders.language.language.LanguageEncoder"
    model_name: str = LanguageEncoderType.BERT_BASE.value
    max_seq_length: int = 128
    use_pooler: bool = True


@dataclass
class LanguageProprioTokenizerEncoderConfig(EncoderConfig):
    """Language + Proprioceptive Tokenizer Encoder configuration.

    Tokenizes language instruction + discretized proprio state and returns
    embeddings for FAST-style autoregressive models.
    """
    _target_: str = "refactoring.models.encoding.encoders.multimodal.language_proprio_tokenizer.LanguageProprioTokenizerEncoder"
    language_model_name: str = "google/gemma-2b"
    max_token_len: int = 512
    device: str = "cpu"



