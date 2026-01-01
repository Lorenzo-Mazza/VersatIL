"""Debug script to compare model inference vs validation data.

This script loads ONE episode from zarr (like the offline validation server)
and tests model predictions against ground truth actions.

Run with:
    python debug_inference_vs_val.py [checkpoint_path] [checkpoint_name] [episode_idx]
"""

import logging
import os
import sys

import hydra
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf

from refactoring.configs import MainConfig
from refactoring.data.constants import Cameras, ProprioKey
from refactoring.models.policy import Policy
from refactoring.training.lightning_policy import LightningPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_model_from_checkpoint(checkpoint_path: str, checkpoint_name: str, device: torch.device) -> tuple[Policy, MainConfig]:
    """Load policy and config from checkpoint."""
    config_path = os.path.join(checkpoint_path, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    logging.info(f"Loading config from {config_path}")
    config: MainConfig = hydra.utils.instantiate(OmegaConf.load(config_path))

    checkpoint_file = os.path.join(checkpoint_path, checkpoint_name)
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_file}")

    logging.info(f"Loading model from {checkpoint_file}")

    policy: Policy = config.policy
    policy.to(device).eval()

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    lightning_module = LightningPolicy(policy=policy, training_config=config.training)
    lightning_module.load_state_dict(checkpoint['state_dict'], strict=False)

    return policy, config


def load_episode_from_zarr(zarr_path: str, episode_idx: int) -> dict:
    """Load a single episode from zarr dataset."""
    logging.info(f"Loading zarr from {zarr_path}")
    root = zarr.open(store=zarr_path, mode='r')
    data = root['data']
    meta = root['meta']

    episode_ends = np.array(meta['episode_ends'])
    n_episodes = len(episode_ends)

    if episode_idx >= n_episodes:
        raise ValueError(f"Episode {episode_idx} out of range (max {n_episodes-1})")

    ep_start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    ep_end = int(episode_ends[episode_idx])

    logging.info(f"Episode {episode_idx}: indices [{ep_start}, {ep_end}), length={ep_end - ep_start}")

    # Load episode data
    episode = {
        'agentview_rgb': np.array(data['agentview_rgb'][ep_start:ep_end]),
        'eye_in_hand_rgb': np.array(data['eye_in_hand_rgb'][ep_start:ep_end]),
        'ee_pos': np.array(data['ee_pos'][ep_start:ep_end]),
        'ee_ori': np.array(data['ee_ori'][ep_start:ep_end]),
        'gripper_state_obs': np.array(data['gripper_state_obs'][ep_start:ep_end]),
        'ee_pos_action': np.array(data['ee_pos_action'][ep_start:ep_end]),
        'ee_ori_action': np.array(data['ee_ori_action'][ep_start:ep_end]),
        'gripper_state_action': np.array(data['gripper_state_action'][ep_start:ep_end]),
    }

    # Get language instruction
    lang = data['language_instruction'][ep_start]
    if isinstance(lang, np.ndarray):
        lang = lang[0] if lang.ndim > 0 else str(lang)
    episode['language_instruction'] = str(lang)

    logging.info(f"Task: {episode['language_instruction']}")

    return episode


def preprocess_image(img: np.ndarray, target_size: int = 128) -> torch.Tensor:
    """Preprocess image like LiberoClient does."""
    import cv2

    # Ensure uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)

    # Resize
    img = cv2.resize(img, (target_size, target_size))

    # Convert to tensor: HWC -> CHW, normalize to [0, 1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    return img_tensor


def build_observation(episode: dict, timestep: int, obs_horizon: int, target_size: int, device: torch.device) -> dict:
    """Build observation dict for a single timestep, matching LiberoClient format."""
    obs = {}

    # Build observation history
    agentview_list = []
    eye_in_hand_list = []

    for t in range(obs_horizon):
        # Clamp to valid indices (pad with first frame if needed)
        idx = max(0, timestep - obs_horizon + 1 + t)

        agentview_list.append(preprocess_image(episode['agentview_rgb'][idx], target_size))
        eye_in_hand_list.append(preprocess_image(episode['eye_in_hand_rgb'][idx], target_size))

    # Stack to (obs_horizon, C, H, W) then add batch dim -> (1, obs_horizon, C, H, W)
    obs[Cameras.AGENTVIEW.value] = torch.stack(agentview_list).unsqueeze(0).to(device)
    obs[Cameras.EYE_IN_HAND.value] = torch.stack(eye_in_hand_list).unsqueeze(0).to(device)

    return obs


def run_episode_test(policy: Policy, episode: dict, config: MainConfig, device: torch.device) -> None:
    """Run through episode and compare predictions to GT."""
    obs_horizon = config.task.observation_horizon
    target_size = config.task.dataloader.target_resolution

    episode_length = len(episode['ee_pos_action'])

    logging.info(f"\n{'='*60}")
    logging.info(f"Running inference on episode with {episode_length} timesteps")
    logging.info(f"obs_horizon={obs_horizon}, target_size={target_size}")
    logging.info(f"{'='*60}")

    pos_errors = []
    ori_errors = []
    grip_matches = []
    all_pred_pos = []

    for t in range(min(episode_length, 50)):  # Test first 50 timesteps
        # Build observation
        obs = build_observation(episode, t, obs_horizon, target_size, device)

        # Get GT action
        gt_pos = episode['ee_pos_action'][t].flatten()
        gt_ori = episode['ee_ori_action'][t].flatten()
        gt_grip = episode['gripper_state_action'][t].flatten()[0]

        # Predict
        with torch.no_grad():
            predicted_actions = policy.predict_action(obs)

        # Extract predictions
        pred_pos = predicted_actions[ProprioKey.EE_POS_ACTION.value][0, 0].cpu().numpy()
        pred_ori = predicted_actions[ProprioKey.EE_ORI_ACTION.value][0, 0].cpu().numpy()
        pred_grip = predicted_actions[ProprioKey.GRIPPER_STATE_ACTION.value][0, 0].cpu().numpy()

        # Calculate errors
        pos_error = np.abs(pred_pos - gt_pos).mean()
        ori_error = np.abs(pred_ori - gt_ori).mean()
        grip_match = (pred_grip > 0) == (gt_grip > 0)

        pos_errors.append(pos_error)
        ori_errors.append(ori_error)
        grip_matches.append(grip_match)
        all_pred_pos.append(pred_pos)

        if t < 5:  # Log first 5 timesteps in detail
            logging.info(f"\n[t={t}] pos_mae={pos_error:.4f}, ori_mae={ori_error:.4f}, grip={'MATCH' if grip_match else 'MISMATCH'}")
            logging.info(f"  Pred pos: {pred_pos}")
            logging.info(f"  GT pos:   {gt_pos}")
            logging.info(f"  Pred ori: {pred_ori}")
            logging.info(f"  GT ori:   {gt_ori}")
            logging.info(f"  Pred grip: {pred_grip}, GT grip: {gt_grip}")

    # Summary
    logging.info(f"\n{'='*60}")
    logging.info("SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Timesteps tested: {len(pos_errors)}")
    logging.info(f"Position MAE: mean={np.mean(pos_errors):.4f}, std={np.std(pos_errors):.4f}")
    logging.info(f"Orientation MAE: mean={np.mean(ori_errors):.4f}, std={np.std(ori_errors):.4f}")
    logging.info(f"Gripper match: {sum(grip_matches)}/{len(grip_matches)} ({100*sum(grip_matches)/len(grip_matches):.1f}%)")

    # Check if predictions are constant
    all_pred_pos = np.array(all_pred_pos)
    pos_std = all_pred_pos.std(axis=0)
    logging.info(f"Predicted pos std per dim: {pos_std}")

    if np.all(pos_std < 0.01):
        logging.warning("WARNING: Model outputs are nearly constant! This indicates a major problem.")

    logging.info(f"{'='*60}")


def analyze_normalizer(policy: Policy) -> None:
    """Analyze normalizer state."""
    logging.info(f"\n{'='*60}")
    logging.info("NORMALIZER ANALYSIS")
    logging.info(f"{'='*60}")

    normalizer = policy.normalizer

    logging.info(f"Normalizer type: {type(normalizer)}")
    logging.info(f"Number of keys: {len(normalizer.params_dict)}")

    if len(normalizer.params_dict) == 0:
        logging.error("CRITICAL: Normalizer is EMPTY! Model will not work correctly.")
        return

    for key in list(normalizer.params_dict.keys()):
        params = normalizer.params_dict[key]
        logging.info(f"\n  {key}:")
        if hasattr(params, 'keys'):
            for param_name in params.keys():
                param_val = params[param_name]
                if isinstance(param_val, torch.Tensor):
                    vals = param_val.cpu().numpy().flatten()
                    logging.info(f"    {param_name}: shape={param_val.shape}, values={vals[:5]}...")


def main():
    # Parse args
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/cluster/workspaces/mazzalore/bowel_retraction/action_transformer_libero/20251229_010655"
    checkpoint_name = sys.argv[2] if len(sys.argv) > 2 else "best-90-0.1118.ckpt"
    episode_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 450
    zarr_path = "/mnt/cluster/workspaces/mazzalore/libero_10.zarr"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model
    policy, config = load_model_from_checkpoint(checkpoint_path, checkpoint_name, device)

    # Analyze normalizer
    analyze_normalizer(policy)

    # Load episode
    episode = load_episode_from_zarr(zarr_path, episode_idx)

    # Run test
    run_episode_test(policy, episode, config, device)


if __name__ == "__main__":
    main()