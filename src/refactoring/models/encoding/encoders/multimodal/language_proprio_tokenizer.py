"""Language + Proprioceptive Tokenizer Encoder.

This encoder:
1. Discretizes proprioceptive state into bins using pre-fitted BinningTokenizer
2. Converts discretized state to string representation (robot/camera frame)
3. Optionally concatenates with language instruction
4. Tokenizes the combined text and returns token IDs

Outputs token ID sequences.
"""
import logging

import torch
from transformers import AutoTokenizer

from refactoring.data.constants import (
    LANGUAGE_KEY,
    PROPRIO_OBS_CAMERA_FRAME_KEY,
    PROPRIO_OBS_ROBOT_FRAME_KEY,
)
from refactoring.data.tokenize.binning_tokenizer import BinningTokenizer
from refactoring.models.encoding.encoders.base import EncoderInput, EncoderOutput
from refactoring.models.encoding.encoders.constants import EncoderOutputKeys, LanguageEncoderType
from refactoring.models.encoding.encoders.unconditional import Encoder




class LanguageProprioTokenizerEncoder(Encoder):
    """Tokenizes language instruction + discretized proprioceptive state.

    This encoder is designed to tokenize input data strings.

    Architecture:
        1. Receive pre-fitted BinningTokenizer via set_tokenizer()
        2. Discretize proprio state → bins (robot/camera frame)
        3. Convert bins to string with frame labels
        4. Create prefix: "Task: {language}, Proprioceptive state in robot frame: {state_robot}, ...in camera frame: {state_camera};\n"
        5. Tokenize prefix
    """

    def __init__(
        self,
        input_keys: str | list[str],
        pretrained: bool,
        frozen: bool,
        language_model_name: str =LanguageEncoderType.GEMMA_2B.value,
        max_token_len: int = 256,
    ):
        """Initialize language + proprio tokenizer encoder.

        Args:
            input_keys: Keys for language and proprio inputs
            pretrained: Whether to use pretrained weights
            frozen: Whether to freeze encoder weights
            language_model_name: HuggingFace model name for language model
            max_token_len: Maximum token sequence length
        """
        # Language is optional, but at least one proprio frame must be present
        specification = EncoderInput(
            keys=input_keys,
            at_least_one_of_groups=[[PROPRIO_OBS_ROBOT_FRAME_KEY, PROPRIO_OBS_CAMERA_FRAME_KEY]],
        )
        super().__init__(
            input_specification=specification,
            pretrained=pretrained,
            frozen=frozen,
        )

        self.max_token_len = max_token_len
        self.lm_model_name = language_model_name
        self.language_tokenizer = AutoTokenizer.from_pretrained(language_model_name)
        if self.language_tokenizer.pad_token is None:
            self.language_tokenizer.pad_token = self.language_tokenizer.eos_token

        self.vocabulary_size = self.language_tokenizer.vocab_size
        self.binning_tokenizer_robot: BinningTokenizer | None = None
        self.binning_tokenizer_camera: BinningTokenizer | None = None

    def set_tokenizer(self, tokenizer) -> None:
        """Set the fitted binning tokenizers for proprioceptive state.

        This is called by the Policy after the tokenizer has been fitted on the dataset.

        Args:
            tokenizer: Tokenizer object containing fitted BinningTokenizers for proprio keys
        """
        if PROPRIO_OBS_ROBOT_FRAME_KEY in tokenizer.tokenizers:
            self.binning_tokenizer_robot = tokenizer.tokenizers[PROPRIO_OBS_ROBOT_FRAME_KEY]
        if PROPRIO_OBS_CAMERA_FRAME_KEY in tokenizer.tokenizers:
            self.binning_tokenizer_camera = tokenizer.tokenizers[PROPRIO_OBS_CAMERA_FRAME_KEY]
        if self.binning_tokenizer_robot is None and self.binning_tokenizer_camera is None:
            raise ValueError(
                f"Tokenizer must contain at least one of '{PROPRIO_OBS_ROBOT_FRAME_KEY}' or "
                f"'{PROPRIO_OBS_CAMERA_FRAME_KEY}'. Available keys: {list(tokenizer.tokenizers.keys())}"
            )

    def forward(
        self, inputs: dict[str, torch.Tensor | list[str] | list[list[str]]]
    ) -> dict[str, torch.Tensor]:
        """Forward pass to tokenize language + proprio.

        Args:
            inputs: Dict containing:
                - LANGUAGE_KEY (optional): Language instructions
                    - Single timestep: List[str] with length B
                    - Temporal: List[List[str]] with shape (B, T)
                - PROPRIO_OBS_ROBOT_FRAME_KEY (optional): Proprio tensor (B, D) or (B, T, D)
                - PROPRIO_OBS_CAMERA_FRAME_KEY (optional): Proprio tensor (B, D) or (B, T, D)

        Returns:
            Dict with tokenized sequences:
                - "language": token IDs (B, max_token_len)
                - "is_pad_token": padded token mask (B, max_token_len)
        """
        if self.binning_tokenizer_robot is None and self.binning_tokenizer_camera is None:
            raise RuntimeError(
                "No binning tokenizer set. Call set_tokenizer() before forward pass."
            )

        language_instructions = inputs.get(LANGUAGE_KEY, None)
        proprio_robot = inputs.get(PROPRIO_OBS_ROBOT_FRAME_KEY, None)
        proprio_camera = inputs.get(PROPRIO_OBS_CAMERA_FRAME_KEY, None)

        if proprio_robot is not None:
            batch_size = proprio_robot.shape[0]
        elif proprio_camera is not None:
            batch_size = proprio_camera.shape[0]
        else:
            raise ValueError("At least one proprio frame must be present in inputs")

        T = None
        if proprio_robot is not None:
            if proprio_robot.dim() == 2:
                proprio_robot = proprio_robot.unsqueeze(1)
            T = proprio_robot.shape[1]

        if proprio_camera is not None:
            if proprio_camera.dim() == 2:
                proprio_camera = proprio_camera.unsqueeze(1)
            T_cam = proprio_camera.shape[1]
            if T is not None and T != T_cam:
                raise ValueError(f"Robot and camera proprio must have same T. Got T_robot={T}, T_camera={T_cam}")
            T = T_cam

        if language_instructions is not None and isinstance(language_instructions[0], list):
            T_lang = len(language_instructions[0])
            if T is not None and T != T_lang:
                raise ValueError(f"Language and proprio must have same T. Got T_proprio={T}, T_language={T_lang}")
            T = T_lang

        T = 1 if T is None else T
        discretized_robot = None
        discretized_camera = None
        if proprio_robot is not None:
            if self.binning_tokenizer_robot is None:
                raise RuntimeError(f"Binning tokenizer for {PROPRIO_OBS_ROBOT_FRAME_KEY} not set")
            if not self.binning_tokenizer_robot._is_fitted:
                raise RuntimeError(f"Binning tokenizer for {PROPRIO_OBS_ROBOT_FRAME_KEY} not fitted")
            discretized_robot = self.binning_tokenizer_robot.encode(
                proprio_robot.view(batch_size * T, -1)
            )
            discretized_robot = discretized_robot.view(batch_size, T, -1)

        if proprio_camera is not None:
            if self.binning_tokenizer_camera is None:
                raise RuntimeError(f"Binning tokenizer for {PROPRIO_OBS_CAMERA_FRAME_KEY} not set")
            if not self.binning_tokenizer_camera._is_fitted:
                raise RuntimeError(f"Binning tokenizer for {PROPRIO_OBS_CAMERA_FRAME_KEY} not fitted")
            discretized_camera = self.binning_tokenizer_camera.encode(
                proprio_camera.view(batch_size * T, -1)
            )
            discretized_camera = discretized_camera.view(batch_size, T, -1)

        prefixes = []
        for i in range(batch_size):
            time_parts = []
            for t in range(T):
                if language_instructions is not None:
                    if isinstance(language_instructions[i], list):
                        lang_instruction = language_instructions[i][t]
                    else:
                        lang_instruction = language_instructions[i]
                else:
                    lang_instruction = None

                cleaned_text = lang_instruction.lower().strip().replace("_", " ") if lang_instruction else None
                task_part = f"Task: {cleaned_text}" if cleaned_text else None

                robot_part = ""
                if discretized_robot is not None:
                    robot_str = " ".join(map(str, discretized_robot[i, t].numpy().tolist()))
                    robot_part = f"State at t={t} in robot frame: {robot_str}"

                camera_part = ""
                if discretized_camera is not None:
                    camera_str = " ".join(map(str, discretized_camera[i, t].numpy().tolist()))
                    camera_part = f"State at t={t} in camera frame: {camera_str}"

                parts = [p for p in [task_part, robot_part, camera_part] if p]
                time_parts.append(", ".join(parts))

            prefix = ", ".join(time_parts) + ";\n"
            prefixes.append(prefix)

        tokenized = self.language_tokenizer(
            prefixes,
            padding_side='left',
            add_special_tokens=True,
            return_tensors="pt",
            max_length=self.max_token_len,
            truncation=True,
            padding="max_length",
        )  # (B, max_token_len)
        tokens = tokenized.data['input_ids']
        is_pad_mask = ~tokenized.data['attention_mask'].to(torch.bool)
        if tokens.shape[-1] > self.max_token_len:
            logging.warning(msg="Tokenized sequence length exceeds max_token_len; truncating. If this occurs frequently, consider increasing max_token_len.")
            tokens = tokens[:, -self.max_token_len :]
            is_pad_mask = is_pad_mask[:, -self.max_token_len :]
        return {
            EncoderOutputKeys.LANGUAGE.value: tokens,
            EncoderOutputKeys.TOKEN_MASK.value: is_pad_mask,
        }

    def get_output_specification(self) -> EncoderOutput:
        """Get output specification.

        Returns encoder output keys and dimensions.
        """
        return EncoderOutput(
            features=[EncoderOutputKeys.LANGUAGE.value, EncoderOutputKeys.TOKEN_MASK.value],
            dimensions={
                EncoderOutputKeys.LANGUAGE.value: (self.max_token_len,),
                EncoderOutputKeys.TOKEN_MASK.value: (self.max_token_len,),
            },
        )