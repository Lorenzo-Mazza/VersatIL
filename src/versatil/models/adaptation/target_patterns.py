"""VLM module-name specific patterns for PEFT target selection."""

LLAMA_ATTENTION_AND_FEEDFORWARD_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LLAMA_QUERY_VALUE_MODULES = [
    "q_proj",
    "v_proj",
]
VLM_TEXT_MODEL_ATTENTION_AND_FEEDFORWARD_PATTERN = (
    r".*(language_model|text_model)\..*\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)
VLM_TEXT_MODEL_QUERY_VALUE_PATTERN = (
    r".*(language_model|text_model)\..*\.self_attn\.(q_proj|v_proj)$"
)
