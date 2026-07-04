from dataclasses import dataclass, field

from versatil.data.constants import (
    ActionDiscretizerType,
    ActionTokenIdMappingType,
    BinningStrategy,
    TokenPaddingStrategy,
)
from versatil.models.encoding.encoders.constants import LanguageEncoderType


@dataclass
class ObservationTokenizationConfig:
    """Configuration for converting observations into text token IDs.

    Attributes:
        tokenizer_model: Language tokenizer model name.
        observation_keys: Observation keys to include in prompt (order preserved in
            prompt construction) Example: ["language", "proprio_robot_frame",
            "proprio_camera_frame"].
        bin_continuous_data: Whether to bin continuous observations into quantiles
            before string conversion.
        num_bins: Number of discretization bins for continuous observations.
        max_token_len: Maximum token length for the prompt.
        raw_text: Pass language text through unformatted (no "Task:" prefix, no
            lowercasing). Use for VLM policies (SmolVLA, Pi0) that expect raw text.
        prompt_template: Optional template wrapped around the language instruction in
            raw-text mode, with an "{instruction}" placeholder. The instruction is
            lowercased and stripped before insertion (OpenVLA convention).
        padding_strategy: Padding strategy: "max_length" pads all sequences to
            max_token_len, "longest" pads to the longest sequence in the batch.
        trust_remote_code: Allow tokenizers that ship custom HuggingFace code.
    """

    # Language tokenizer model name
    tokenizer_model: str = LanguageEncoderType.BERT_BASE.value
    # Observation keys to include in prompt (order preserved in prompt construction)
    # Example: ["language", "proprio_robot_frame", "proprio_camera_frame"]
    observation_keys: list[str] = field(default_factory=list)
    # Whether to bin continuous observations into quantiles before string conversion
    bin_continuous_data: bool = True
    num_bins: int = 256
    # Maximum token length for the prompt
    max_token_len: int = 256
    # Pass language text through unformatted (no "Task:" prefix, no lowercasing).
    # Use for VLM policies (SmolVLA, Pi0) that expect raw text.
    raw_text: bool = False
    # Optional template wrapped around the language instruction in raw-text
    # mode, with an "{instruction}" placeholder. The instruction is lowercased
    # and stripped before insertion (OpenVLA convention).
    prompt_template: str | None = None
    # Padding strategy: "max_length" pads all sequences to max_token_len,
    # "longest" pads to the longest sequence in the batch.
    padding_strategy: str = TokenPaddingStrategy.MAX_LENGTH.value
    # Allow tokenizers that ship custom HuggingFace code
    trust_remote_code: bool = False


@dataclass
class ActionDiscretizerConfig:
    """Configuration for discretizing continuous action chunks.

    Attributes:
        type: Strategy that turns continuous action chunks into local discrete action
            IDs.
        use_pretrained: FAST-specific options.
        tokenizer_model: FAST tokenizer model identifier.
        num_bins: Binned discretizer options. Uniform binning places equal-width bins
            over [min_value, max_value]; quantile binning adapts edges to the action
            distribution and ignores the range bounds.
        binning_strategy: Strategy mapping continuous values to bins.
        min_value: Lower bound of the discretized value range.
        max_value: Upper bound of the discretized value range.
    """

    # Strategy that turns continuous action chunks into local discrete action IDs.
    type: str = ActionDiscretizerType.FAST.value
    # FAST-specific options.
    use_pretrained: bool = True
    tokenizer_model: str = "physical-intelligence/fast"
    # Binned discretizer options. Uniform binning places equal-width bins
    # over [min_value, max_value]; quantile binning adapts edges to the
    # action distribution and ignores the range bounds.
    num_bins: int = 256
    binning_strategy: str = BinningStrategy.UNIFORM.value
    min_value: float = -1.0
    max_value: float = 1.0


@dataclass
class ActionTokenIdMappingConfig:
    """Configuration for mapping action-local IDs into model token IDs.

    Attributes:
        type: Mapping from local action IDs into the model token-id space.
        language_tokenizer_model: Language-tokenizer mapping options.
        num_special_tokens_to_skip: Vocabulary offset reserved for special tokens.
    """

    # Mapping from local action IDs into the model token-id space.
    type: str = ActionTokenIdMappingType.IDENTITY.value
    # Language-tokenizer mapping options.
    language_tokenizer_model: str | None = None
    num_special_tokens_to_skip: int = 128


@dataclass
class ActionTokenizationConfig:
    """Configuration for action tokenization.

    Attributes:
        action_discretizer: Discretizer turning continuous actions into bins.
        token_id_mapping: Mapping between action bins and vocabulary token ids.
        max_token_len: Maximum action token sequence length.
    """

    action_discretizer: ActionDiscretizerConfig = field(
        default_factory=ActionDiscretizerConfig
    )
    token_id_mapping: ActionTokenIdMappingConfig = field(
        default_factory=ActionTokenIdMappingConfig
    )
    max_token_len: int = 128


@dataclass
class TokenizationConfig:
    """Top-level observation/action tokenization configuration.

    Attributes:
        tokenize_observations: Whether observations are tokenized into prompts.
        observation_tokenizer: Observation tokenizer settings.
        tokenize_actions: Whether actions are tokenized.
        action_tokenizer: Action tokenizer settings.
    """

    tokenize_observations: bool = False
    observation_tokenizer: ObservationTokenizationConfig | None = None
    tokenize_actions: bool = False
    action_tokenizer: ActionTokenizationConfig | None = None
