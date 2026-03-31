"""Convert a LeRobot SmolVLA checkpoint to VersatIL format.

Loads every key from the LeRobot safetensors, maps each to the VersatIL
state dict structure, and raises if any key on either side is unmatched.

Usage:
    python scripts/convert_smolvla_checkpoint.py \
        --input /path/to/smolvla_libero/model.safetensors \
        --output /path/to/versatil_smolvla.pt
"""

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

VLM_PREFIX = "model.vlm_with_expert.vlm."
EXPERT_PREFIX = "model.vlm_with_expert.lm_expert."

# LeRobot wraps the VLM in AutoModelForVision2Seq which adds a .model.
# level. VersatIL uses AutoModel directly, so strip the extra .model. prefix.
VLM_UNWRAP_PREFIX = "model."

DECODER_PROJECTION_MAP = {
    "model.action_in_proj.": "action_input_projection.",
    "model.action_out_proj.": "action_output_projection.",
    "model.action_time_mlp_in.": "action_time_fusion_input.",
    "model.action_time_mlp_out.": "action_time_fusion_output.",
}

DEFAULT_PROPRIO_FEATURE_KEY = "robot_state_proprio"

NORMALIZER_PREFIXES = [
    "normalize_inputs.",
    "normalize_targets.",
    "unnormalize_outputs.",
]

# Keys present in LeRobot but not in VersatIL (AutoModel vs AutoModelForVision2Seq)
SKIPPED_VLM_SUFFIXES = [
    "lm_head.",
]

MMDIT_EXPERT_KEY_MAP = {
    "self_attn.q_proj.weight": "joint_attention.query_projection_action.weight",
    "self_attn.k_proj.weight": "joint_attention.key_projection_action.weight",
    "self_attn.v_proj.weight": "joint_attention.value_projection_action.weight",
    "self_attn.o_proj.weight": "joint_attention.output_projection_action.weight",
    "input_layernorm.weight": "attention_normalization_action.weight",
    "post_attention_layernorm.weight": "feedforward_normalization_action.weight",
    "mlp.gate_proj.weight": "feedforward_action.0.gate_proj.weight",
    "mlp.up_proj.weight": "feedforward_action.0.value_proj.weight",
    "mlp.down_proj.weight": "feedforward_action.2.weight",
}

CROSS_ATTN_EXPERT_KEY_MAP = {
    "self_attn.q_proj.weight": "layer.cross_attention.query_projection.weight",
    "self_attn.k_proj.weight": "key_bridge.weight",
    "self_attn.v_proj.weight": "value_bridge.weight",
    "self_attn.o_proj.weight": "layer.cross_attention.output_projection.weight",
    "input_layernorm.weight": "layer.pre_attention_normalization.weight",
    "post_attention_layernorm.weight": "layer.pre_feedforward_normalization.weight",
    "mlp.gate_proj.weight": "layer.feedforward.0.gate_proj.weight",
    "mlp.up_proj.weight": "layer.feedforward.0.value_proj.weight",
    "mlp.down_proj.weight": "layer.feedforward.2.weight",
}


def determine_layer_types(
    num_layers: int,
    self_attn_every_n_layers: int,
) -> list[str]:
    """Determine which expert layers are MMDiT (self-attention) vs cross-attention."""
    layer_types = []
    for layer_index in range(num_layers):
        if self_attn_every_n_layers > 0 and layer_index % self_attn_every_n_layers == 0:
            layer_types.append("self_attention")
        else:
            layer_types.append("cross_attention")
    return layer_types


def convert_smolvla_checkpoint(
    input_path: Path,
    output_path: Path,
    self_attn_every_n_layers: int = 2,
    proprioceptive_feature_key: str = DEFAULT_PROPRIO_FEATURE_KEY,
    action_dim: int = 7,
) -> None:
    """Convert LeRobot SmolVLA safetensors to VersatIL format.

    LeRobot pads actions to max_action_dim (default 32). This function
    slices action projection weights to the actual action_dim.

    Raises:
        ValueError: If any source key has no mapping.
    """
    source_tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(input_path), framework="pt") as f:
        for key in f.keys():
            source_tensors[key] = f.get_tensor(key)

    encoder_state_dict: dict[str, torch.Tensor] = {}
    decoder_state_dict: dict[str, torch.Tensor] = {}
    normalizer_state_dict: dict[str, torch.Tensor] = {}
    matched_source_keys: set[str] = set()

    expert_layer_indices: list[int] = sorted(
        {
            int(key[len(EXPERT_PREFIX + "layers.") :].split(".")[0])
            for key in source_tensors
            if key.startswith(EXPERT_PREFIX + "layers.")
        }
    )
    num_expert_layers = len(expert_layer_indices)
    layer_types = determine_layer_types(
        num_layers=num_expert_layers,
        self_attn_every_n_layers=self_attn_every_n_layers,
    )
    print(f"Expert layers: {num_expert_layers}")
    print(
        f"Layer types: {layer_types[:6]}... "
        f"(self_attn_every_n={self_attn_every_n_layers})"
    )

    for key, tensor in source_tensors.items():
        if any(key.startswith(prefix) for prefix in NORMALIZER_PREFIXES):
            normalizer_state_dict[key] = tensor
            matched_source_keys.add(key)
            continue

        if key.startswith(VLM_PREFIX):
            vlm_suffix = key[len(VLM_PREFIX) :]
            if any(vlm_suffix.startswith(s) for s in SKIPPED_VLM_SUFFIXES):
                matched_source_keys.add(key)
                continue
            if vlm_suffix.startswith(VLM_UNWRAP_PREFIX):
                vlm_suffix = vlm_suffix[len(VLM_UNWRAP_PREFIX) :]
            encoder_state_dict["vlm." + vlm_suffix] = tensor
            matched_source_keys.add(key)
            continue

        for lerobot_prefix, versatil_prefix in DECODER_PROJECTION_MAP.items():
            if key.startswith(lerobot_prefix):
                decoder_state_dict[versatil_prefix + key[len(lerobot_prefix) :]] = (
                    tensor
                )
                matched_source_keys.add(key)
                break
        if key in matched_source_keys:
            continue

        if key.startswith("model.state_proj."):
            param_name = key[len("model.state_proj.") :]
            versatil_key = (
                f"proprioceptive_projection.linear_projections."
                f"{proprioceptive_feature_key}.{param_name}"
            )
            decoder_state_dict[versatil_key] = tensor
            matched_source_keys.add(key)
            continue

        if key.startswith(EXPERT_PREFIX + "layers."):
            remainder = key[len(EXPERT_PREFIX + "layers.") :]
            source_layer_idx = int(remainder.split(".", 1)[0])
            layer_suffix = remainder.split(".", 1)[1]
            versatil_layer_idx = expert_layer_indices.index(source_layer_idx)
            layer_type = layer_types[versatil_layer_idx]
            key_map = (
                MMDIT_EXPERT_KEY_MAP
                if layer_type == "self_attention"
                else CROSS_ATTN_EXPERT_KEY_MAP
            )
            if layer_suffix in key_map:
                versatil_key = (
                    f"expert_layers.{versatil_layer_idx}.{key_map[layer_suffix]}"
                )
                decoder_state_dict[versatil_key] = tensor
                matched_source_keys.add(key)
            continue

        if key == EXPERT_PREFIX + "norm.weight":
            decoder_state_dict["expert_final_norm.weight"] = tensor
            matched_source_keys.add(key)
            continue

    unmatched = set(source_tensors.keys()) - matched_source_keys
    if unmatched:
        print(f"\nUNMATCHED SOURCE KEYS ({len(unmatched)}):")
        for key in sorted(unmatched):
            print(f"  {key}: {source_tensors[key].shape}")
        raise ValueError(f"{len(unmatched)} source keys have no mapping. See above.")

    # Slice action projections from LeRobot's max_action_dim to actual action_dim
    source_action_dim = decoder_state_dict["action_input_projection.weight"].shape[1]
    if source_action_dim != action_dim:
        print(f"\nSlicing action projections: {source_action_dim} → {action_dim}")
        w = decoder_state_dict["action_input_projection.weight"]
        decoder_state_dict["action_input_projection.weight"] = w[:, :action_dim]
        b = decoder_state_dict["action_input_projection.bias"]
        decoder_state_dict["action_input_projection.bias"] = b
        w = decoder_state_dict["action_output_projection.weight"]
        decoder_state_dict["action_output_projection.weight"] = w[:action_dim, :]
        b = decoder_state_dict["action_output_projection.bias"]
        decoder_state_dict["action_output_projection.bias"] = b[:action_dim]

    # Build Lightning-compatible flat state_dict with policy.* prefix
    # Encoder keys: policy.encoding_pipeline.encoders.vlm.<key>
    # Decoder keys: policy.decoder.<key>
    # Normalizer keys: policy.decoder.normalizer.<key>
    state_dict: dict[str, torch.Tensor] = {}
    for key, tensor in encoder_state_dict.items():
        state_dict[f"policy.encoding_pipeline.encoders.vlm.{key}"] = tensor
    for key, tensor in decoder_state_dict.items():
        state_dict[f"policy.decoder.{key}"] = tensor
    for key, tensor in normalizer_state_dict.items():
        state_dict[f"policy.decoder.normalizer.{key}"] = tensor

    checkpoint = {"state_dict": state_dict}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, str(output_path))
    print(f"\nEncoder: {len(encoder_state_dict)} keys")
    print(f"Decoder: {len(decoder_state_dict)} keys")
    print(f"Normalizer: {len(normalizer_state_dict)} keys")
    print(f"Total state_dict: {len(state_dict)} keys")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot SmolVLA checkpoint to VersatIL format"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-attn-every-n", type=int, default=2)
    parser.add_argument(
        "--proprio-key",
        type=str,
        default=DEFAULT_PROPRIO_FEATURE_KEY,
        help="Feature key for proprioceptive state in the VersatIL decoder.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="Actual action dimension (LeRobot pads to max_action_dim=32).",
    )
    args = parser.parse_args()
    convert_smolvla_checkpoint(
        input_path=args.input,
        output_path=args.output,
        self_attn_every_n_layers=args.self_attn_every_n,
        proprioceptive_feature_key=args.proprio_key,
        action_dim=args.action_dim,
    )


if __name__ == "__main__":
    main()
