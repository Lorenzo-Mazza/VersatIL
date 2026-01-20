from dataclasses import dataclass, field
from versatil.data.constants import TokenizerType
from versatil.models.encoding.encoders.constants import LanguageEncoderType


@dataclass
class ObservationTokenizationConfig:
    # Language tokenizer model name
    tokenizer_model: str = LanguageEncoderType.GEMMA_2B.value
    # Observation keys to include in prompt (order preserved in prompt construction)
    # Example: ["language", "proprio_robot_frame", "proprio_camera_frame"]
    observation_keys: list[str] = field(default_factory=list)
    # Whether to bin continuous observations into quantiles before string conversion
    bin_continuous_data: bool = True
    num_bins: int = 256
    # Maximum token length for the prompt
    max_token_len: int = 256


@dataclass
class ActionTokenizationConfig:
    # Chain of tokenizers to apply in sequence
    tokenizer_chain: list[str] = field(
        default_factory=lambda: [TokenizerType.FAST.value]
    )
    # For FAST tokenizer
    use_pretrained_fast: bool = True
    # For language tokenizer in chain (if TokenizerType.LANGUAGE.value in tokenizer_chain)
    language_tokenizer_model: str | None = None
    max_token_len: int = 128


@dataclass
class TokenizationConfig:
    tokenize_observations: bool = False
    observation_tokenizer: ObservationTokenizationConfig | None = None
    tokenize_actions: bool = False
    action_tokenizer: ActionTokenizationConfig | None = None
