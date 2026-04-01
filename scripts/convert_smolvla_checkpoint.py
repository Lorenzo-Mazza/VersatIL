"""Convert a LeRobot SmolVLA checkpoint to a complete VersatIL inference directory.

Produces:
- ``versatil_smolvla.pt``: Lightning-format state dict (model + normalizer)
- ``config.yaml``: Resolved Hydra config for PolicyLoader
- ``tokenizer/``: Fitted observation tokenizer

Usage::

    python scripts/convert_smolvla_checkpoint.py \
        --input /path/to/model.safetensors \
        --output-dir /path/to/output \
        --hydra-config-dir /absolute/path/to/hydra_configs \
        --config-name end_to_end_training_runs/libero_lerobot/smolvla
"""

import argparse
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from safetensors import safe_open

sys.path.insert(0, "src")

from versatil.configs import register_configs
from versatil.data.dataloader import get_dataloaders
from versatil.data.tokenization.observation_tokenizer import ObservationTokenizer
from versatil.data.tokenization.tokenizer import Tokenizer

VLM_PREFIX = "model.vlm_with_expert.vlm."
EXPERT_PREFIX = "model.vlm_with_expert.lm_expert."
VLM_UNWRAP_PREFIX = "model."

DECODER_PROJECTION_MAP = {
    "model.action_in_proj.": "action_input_projection.",
    "model.action_out_proj.": "action_output_projection.",
    "model.action_time_mlp_in.": "action_time_fusion_input.",
    "model.action_time_mlp_out.": "action_time_fusion_output.",
}

SKIPPED_VLM_SUFFIXES = ["lm_head."]

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


def convert_weights(
    source_tensors: dict[str, torch.Tensor],
    self_attn_every_n_layers: int,
    action_dim: int,
    state_dim: int,
) -> dict[str, torch.Tensor]:
    """Convert LeRobot weight keys to VersatIL Lightning state dict format."""
    encoder_sd: dict[str, torch.Tensor] = {}
    proprio_sd: dict[str, torch.Tensor] = {}
    decoder_sd: dict[str, torch.Tensor] = {}
    matched: set[str] = set()

    expert_layer_indices = sorted(
        {
            int(k[len(EXPERT_PREFIX + "layers.") :].split(".", 1)[0])
            for k in source_tensors
            if k.startswith(EXPERT_PREFIX + "layers.")
        }
    )
    layer_types = determine_layer_types(
        num_layers=len(expert_layer_indices),
        self_attn_every_n_layers=self_attn_every_n_layers,
    )
    print(f"Expert layers: {len(expert_layer_indices)}, types: {layer_types[:6]}...")

    for key, tensor in source_tensors.items():
        if key.startswith(VLM_PREFIX):
            vlm_suffix = key[len(VLM_PREFIX) :]
            if any(vlm_suffix.startswith(s) for s in SKIPPED_VLM_SUFFIXES):
                matched.add(key)
                continue
            if vlm_suffix.startswith(VLM_UNWRAP_PREFIX):
                vlm_suffix = vlm_suffix[len(VLM_UNWRAP_PREFIX) :]
            encoder_sd["vlm." + vlm_suffix] = tensor
            matched.add(key)
            continue

        for lr_prefix, vt_prefix in DECODER_PROJECTION_MAP.items():
            if key.startswith(lr_prefix):
                decoder_sd[vt_prefix + key[len(lr_prefix) :]] = tensor
                matched.add(key)
                break
        if key in matched:
            continue

        if key.startswith("model.state_proj."):
            param_name = key[len("model.state_proj.") :]
            if param_name == "weight":
                tensor = tensor[:, :state_dim]
            proprio_sd[f"network.layers.0.{param_name}"] = tensor
            matched.add(key)
            continue

        if key.startswith(EXPERT_PREFIX + "layers."):
            remainder = key[len(EXPERT_PREFIX + "layers.") :]
            source_idx = int(remainder.split(".", 1)[0])
            layer_suffix = remainder.split(".", 1)[1]
            versatil_idx = expert_layer_indices.index(source_idx)
            key_map = (
                MMDIT_EXPERT_KEY_MAP
                if layer_types[versatil_idx] == "self_attention"
                else CROSS_ATTN_EXPERT_KEY_MAP
            )
            if layer_suffix in key_map:
                decoder_sd[f"expert_layers.{versatil_idx}.{key_map[layer_suffix]}"] = (
                    tensor
                )
                matched.add(key)
            continue

        if key == EXPERT_PREFIX + "norm.weight":
            decoder_sd["expert_final_norm.weight"] = tensor
            matched.add(key)
            continue

    unmatched = set(source_tensors.keys()) - matched
    if unmatched:
        print(f"\nUNMATCHED SOURCE KEYS ({len(unmatched)}):")
        for k in sorted(unmatched):
            print(f"  {k}: {source_tensors[k].shape}")
        raise ValueError(f"{len(unmatched)} source keys have no mapping.")

    src_action_dim = decoder_sd["action_input_projection.weight"].shape[1]
    if src_action_dim != action_dim:
        print(f"Slicing action projections: {src_action_dim} → {action_dim}")
        decoder_sd["action_input_projection.weight"] = decoder_sd[
            "action_input_projection.weight"
        ][:, :action_dim]
        decoder_sd["action_output_projection.weight"] = decoder_sd[
            "action_output_projection.weight"
        ][:action_dim, :]
        decoder_sd["action_output_projection.bias"] = decoder_sd[
            "action_output_projection.bias"
        ][:action_dim]

    state_dict: dict[str, torch.Tensor] = {}
    for k, v in encoder_sd.items():
        state_dict[f"policy.encoding_pipeline.encoders.vlm.{k}"] = v
    for k, v in proprio_sd.items():
        state_dict[f"policy.encoding_pipeline.encoders.robot_state.{k}"] = v
    for k, v in decoder_sd.items():
        state_dict[f"policy.decoder.{k}"] = v

    print(
        f"Encoder: {len(encoder_sd)}, Proprio: {len(proprio_sd)}, Decoder: {len(decoder_sd)}"
    )
    return state_dict


def build_normalizer(
    hydra_config_dir: str,
    config_name: str,
) -> dict[str, torch.Tensor]:
    """Compute normalizer from the dataset via Hydra config and get_dataloaders."""
    register_configs()
    with initialize_config_dir(config_dir=hydra_config_dir, version_base=None):
        config = compose(config_name=config_name)

    config = instantiate(config)
    _, _, normalizer, _, _ = get_dataloaders(config)

    normalizer_sd: dict[str, torch.Tensor] = {}
    for k, v in normalizer.state_dict().items():
        normalizer_sd[f"policy.normalizer.{k}"] = v

    print(f"Normalizer: {len(normalizer_sd)} keys")
    return normalizer_sd


def save_tokenizer(
    output_dir: Path,
    hydra_config_dir: str,
    config_name: str,
) -> None:
    """Create and save a fitted observation tokenizer from the Hydra config."""
    register_configs()
    with initialize_config_dir(config_dir=hydra_config_dir, version_base=None):
        config = compose(config_name=config_name)

    tok_config = config.task.dataloader.tokenization
    if not tok_config.tokenize_observations or tok_config.observation_tokenizer is None:
        print("Tokenization disabled in config, skipping tokenizer.")
        return

    obs_tok_config = tok_config.observation_tokenizer
    obs_tokenizer = ObservationTokenizer(
        tokenizer_model=obs_tok_config.tokenizer_model,
        observation_keys=list(obs_tok_config.observation_keys),
        max_token_len=obs_tok_config.max_token_len,
        raw_text=obs_tok_config.get("raw_text", False),
    )
    obs_tokenizer.fit(observation_data={})

    tokenizer = Tokenizer(
        observation_tokenizer=obs_tokenizer,
        action_tokenizer=None,
    )
    tokenizer_path = output_dir / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_path))
    print(
        f"Tokenizer saved to {tokenizer_path} (vocab={obs_tokenizer.vocab_size}, raw_text={obs_tokenizer.raw_text})"
    )


def save_config(output_dir: Path, hydra_config_dir: str, config_name: str) -> None:
    """Generate fully resolved config.yaml from Hydra config."""
    register_configs()
    with initialize_config_dir(config_dir=hydra_config_dir, version_base=None):
        cfg = compose(config_name=config_name)

    resolved = OmegaConf.to_container(cfg, resolve=True)
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(resolved)))
    print(f"Config saved to {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LeRobot SmolVLA checkpoint to VersatIL inference directory."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to LeRobot model.safetensors."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory."
    )
    parser.add_argument(
        "--hydra-config-dir",
        type=str,
        required=True,
        help="Absolute path to hydra_configs.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="end_to_end_training_runs/libero_lerobot/smolvla",
    )
    parser.add_argument("--self-attn-every-n", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert weights
    print("=" * 60)
    print("Step 1: Converting weights")
    print("=" * 60)
    source_tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(args.input), framework="pt") as f:
        for key in f.keys():
            source_tensors[key] = f.get_tensor(key)

    state_dict = convert_weights(
        source_tensors=source_tensors,
        self_attn_every_n_layers=args.self_attn_every_n,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
    )

    # Step 2: Build normalizer from dataset
    print("\n" + "=" * 60)
    print("Step 2: Computing normalizer from dataset")
    print("=" * 60)
    normalizer_sd = build_normalizer(
        hydra_config_dir=args.hydra_config_dir,
        config_name=args.config_name,
    )
    state_dict.update(normalizer_sd)

    # Step 3: Save checkpoint
    checkpoint_path = args.output_dir / "versatil_smolvla.pt"
    torch.save({"state_dict": state_dict}, str(checkpoint_path))
    print(f"Checkpoint saved to {checkpoint_path} ({len(state_dict)} keys)")

    # Step 4: Save tokenizer
    print("\n" + "=" * 60)
    print("Step 3: Saving tokenizer")
    print("=" * 60)
    save_tokenizer(
        output_dir=args.output_dir,
        hydra_config_dir=args.hydra_config_dir,
        config_name=args.config_name,
    )

    # Step 5: Save config
    print("\n" + "=" * 60)
    print("Step 4: Generating config.yaml")
    print("=" * 60)
    save_config(
        output_dir=args.output_dir,
        hydra_config_dir=args.hydra_config_dir,
        config_name=args.config_name,
    )

    print("\n" + "=" * 60)
    print(f"Done. Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
