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

    # Check what keys are in checkpoint vs model
    ckpt_keys = set(checkpoint['state_dict'].keys())
    model_keys = set(lightning_module.state_dict().keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    if missing:
        logging.warning(f"Missing keys in checkpoint ({len(missing)}):")
        for k in sorted(missing):
            logging.warning(f"  MISSING: {k}")
    if unexpected:
        logging.warning(f"Unexpected keys in checkpoint ({len(unexpected)}):")
        for k in sorted(unexpected):
            logging.warning(f"  UNEXPECTED: {k}")

    # Load and check if weights actually changed
    sample_key = list(model_keys)[0]
    before = lightning_module.state_dict()[sample_key].clone()

    lightning_module.load_state_dict(checkpoint['state_dict'], strict=False)

    after = lightning_module.state_dict()[sample_key]
    if torch.allclose(before, after):
        logging.error(f"CRITICAL: Weight '{sample_key}' unchanged after loading checkpoint!")
    else:
        logging.info(f"Checkpoint loaded - weights changed for '{sample_key}'")

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
    pred_horizon = config.task.prediction_horizon
    target_size = 128  # LIBERO images are 128x128

    episode_length = len(episode['ee_pos_action'])

    logging.info(f"\n{'='*60}")
    logging.info(f"Running inference on episode with {episode_length} timesteps")
    logging.info(f"obs_horizon={obs_horizon}, pred_horizon={pred_horizon}, target_size={target_size}")
    logging.info(f"{'='*60}")

    # ========== TEST 1: compute_loss ==========
    logging.info("\n--- TEST 1: compute_loss ---")
    # Build a batch with observations and GT actions
    t = 10  # Pick a timestep with enough history and future
    obs = build_observation(episode, t, obs_horizon, target_size, device)

    # Build action chunk (pred_horizon actions starting from t)
    action_chunk_pos = torch.tensor(episode['ee_pos_action'][t:t+pred_horizon]).float().unsqueeze(0).to(device)
    action_chunk_ori = torch.tensor(episode['ee_ori_action'][t:t+pred_horizon]).float().unsqueeze(0).to(device)
    action_chunk_grip = torch.tensor(episode['gripper_state_action'][t:t+pred_horizon]).float().unsqueeze(0).to(device)

    # Normalize actions like the dataset does
    action_chunk_pos_norm = policy.normalizer[ProprioKey.EE_POS_ACTION.value].normalize(action_chunk_pos)
    action_chunk_ori_norm = policy.normalizer[ProprioKey.EE_ORI_ACTION.value].normalize(action_chunk_ori)

    # Normalize observations like the dataset does
    obs_norm = {}
    for k, v in obs.items():
        if k in policy.normalizer.params_dict:
            obs_norm[k] = policy.normalizer[k].normalize(v)
        else:
            obs_norm[k] = v

    batch = {
        'observation': obs_norm,
        'action': {
            ProprioKey.EE_POS_ACTION.value: action_chunk_pos_norm,
            ProprioKey.EE_ORI_ACTION.value: action_chunk_ori_norm,
            ProprioKey.GRIPPER_STATE_ACTION.value: action_chunk_grip,
        }
    }

    with torch.no_grad():
        loss_output = policy.compute_loss(batch)

    logging.info(f"compute_loss result: {loss_output.total_loss.item():.4f}")
    for k, v in loss_output.component_losses.items():
        logging.info(f"  {k}: {v:.4f}")

    # ========== TEST 2: predict_action ==========
    logging.info("\n--- TEST 2: predict_action on same observation ---")
    with torch.no_grad():
        predicted_actions = policy.predict_action(obs)

    pred_pos = predicted_actions[ProprioKey.EE_POS_ACTION.value][0, 0].cpu().numpy()
    pred_ori = predicted_actions[ProprioKey.EE_ORI_ACTION.value][0, 0].cpu().numpy()
    gt_pos = episode['ee_pos_action'][t].flatten()
    gt_ori = episode['ee_ori_action'][t].flatten()

    logging.info(f"  Pred pos: {pred_pos}")
    logging.info(f"  GT pos:   {gt_pos}")
    logging.info(f"  Pred ori: {pred_ori}")
    logging.info(f"  GT ori:   {gt_ori}")
    logging.info(f"  Position MAE: {np.abs(pred_pos - gt_pos).mean():.4f}")
    logging.info(f"  Orientation MAE: {np.abs(pred_ori - gt_ori).mean():.4f}")
    # ========== TEST 2b: Simulate predict_action step by step ==========
    logging.info("\n--- TEST 2b: Simulating predict_action step-by-step ---")
    from refactoring.data.transform import normalize_observation
    from refactoring.common.tensor_ops import to_device

    obs_2b = to_device(obs, policy.device)
    logging.info(f"  Input obs keys: {list(obs_2b.keys())}")
    logging.info(f"  observation_space.observations_metadata keys: {list(policy.observation_space.observations_metadata.keys())}")
    logging.info(f"  normalizer.params_dict keys: {list(policy.normalizer.params_dict.keys())}")

    # This is what normalize_observation does
    normalized_obs_2b = normalize_observation(
        observation=obs_2b,
        normalizer=policy.normalizer,
        observation_space=policy.observation_space
    )
    logging.info(f"  Normalized obs keys: {list(normalized_obs_2b.keys())}")

    # Check if images were normalized
    for k in obs_2b.keys():
        if k in normalized_obs_2b:
            orig_val = obs_2b[k][0, 0, 0, 0, 0].item()
            norm_val = normalized_obs_2b[k][0, 0, 0, 0, 0].item()
            logging.info(f"  Key '{k}': orig[0,0,0,0,0]={orig_val:.4f}, normalized[0,0,0,0,0]={norm_val:.4f}")
            if abs(orig_val - norm_val) < 0.001:
                logging.warning(f"    WARNING: Key '{k}' was NOT normalized (values unchanged)!")

    # Now encode and predict
    with torch.no_grad():
        features_2b = policy.encoding_pipeline(normalized_obs_2b)
        predictions_2b = policy.algorithm.predict(features=features_2b, network=policy.decoder)

    pred_pos_2b_norm = predictions_2b[ProprioKey.EE_POS_ACTION.value][0, 0].cpu().numpy()
    logging.info(f"  Predicted pos (normalized): {pred_pos_2b_norm}")
    logging.info(f"  GT pos (normalized): {action_chunk_pos_norm[0, 0].cpu().numpy()}")

    # ========== TEST 3: forward pass directly ==========
    logging.info("\n--- TEST 3: forward() on normalized batch ---")
    with torch.no_grad():
        forward_output = policy.forward(batch)

    fwd_pos = forward_output[ProprioKey.EE_POS_ACTION.value][0, 0].cpu().numpy()
    fwd_ori = forward_output[ProprioKey.EE_ORI_ACTION.value][0, 0].cpu().numpy()
    gt_pos_norm = action_chunk_pos_norm[0, 0].cpu().numpy()
    gt_ori_norm = action_chunk_ori_norm[0, 0].cpu().numpy()

    logging.info(f"  Forward pos (norm): {fwd_pos}")
    logging.info(f"  GT pos (norm):      {gt_pos_norm}")
    logging.info(f"  Forward ori (norm): {fwd_ori}")
    logging.info(f"  GT ori (norm):      {gt_ori_norm}")
    logging.info(f"  Position MAE (norm): {np.abs(fwd_pos - gt_pos_norm).mean():.4f}")

    # ========== TEST 3b: MANUALLY UNNORMALIZE forward() output and compare with predict_action ==========
    logging.info("\n--- TEST 3b: MANUALLY UNNORMALIZE forward() output vs predict_action ---")
    # Unnormalize forward output by hand
    fwd_pos_tensor = forward_output[ProprioKey.EE_POS_ACTION.value]
    fwd_ori_tensor = forward_output[ProprioKey.EE_ORI_ACTION.value]
    fwd_grip_tensor = forward_output[ProprioKey.GRIPPER_STATE_ACTION.value]

    fwd_pos_unnorm = policy.normalizer[ProprioKey.EE_POS_ACTION.value].unnormalize(fwd_pos_tensor)
    fwd_ori_unnorm = policy.normalizer[ProprioKey.EE_ORI_ACTION.value].unnormalize(fwd_ori_tensor)

    fwd_pos_unnorm_np = fwd_pos_unnorm[0, 0].cpu().numpy()
    fwd_ori_unnorm_np = fwd_ori_unnorm[0, 0].cpu().numpy()

    logging.info(f"  Forward pos (UNNORM by hand): {fwd_pos_unnorm_np}")
    logging.info(f"  predict_action pos:           {pred_pos}")
    logging.info(f"  GT pos (raw):                 {gt_pos}")
    logging.info(f"  Forward ori (UNNORM by hand): {fwd_ori_unnorm_np}")
    logging.info(f"  predict_action ori:           {pred_ori}")
    logging.info(f"  GT ori (raw):                 {gt_ori}")

    forward_unnorm_mae = np.abs(fwd_pos_unnorm_np - gt_pos).mean()
    predict_action_mae = np.abs(pred_pos - gt_pos).mean()

    logging.info(f"  Forward UNNORM pos MAE vs GT: {forward_unnorm_mae:.4f}")
    logging.info(f"  predict_action pos MAE vs GT: {predict_action_mae:.4f}")

    if forward_unnorm_mae < 0.1 and predict_action_mae > 0.2:
        logging.error("DIAGNOSIS: forward() works but predict_action() fails → ERROR IS IN PREDICT BRANCH (prior vs posterior)")
    elif forward_unnorm_mae > 0.2 and predict_action_mae > 0.2:
        logging.error("DIAGNOSIS: Both forward() and predict_action() fail → ERROR IS IN MODEL LOADING")
    elif forward_unnorm_mae < 0.1 and predict_action_mae < 0.1:
        logging.info("DIAGNOSIS: Both forward() and predict_action() work correctly!")
    else:
        logging.warning(f"DIAGNOSIS: Unclear - forward MAE={forward_unnorm_mae:.4f}, predict MAE={predict_action_mae:.4f}")

    # ========== TEST 5: Compare POSTERIOR vs PRIOR latent distributions ==========
    logging.info("\n--- TEST 5: Compare POSTERIOR vs PRIOR latent distributions ---")
    from refactoring.models.decoding.constants import LATENT_KEY, PRIOR_LATENT_KEY, MU_KEY, LOGVAR_KEY, PRIOR_MU_KEY, PRIOR_LOGVAR_KEY

    # Get variational algorithm
    variational_algo = policy.algorithm

    # Get features from encoding pipeline (on normalized observations)
    with torch.no_grad():
        # Build observations for batch
        obs_for_latent = {
            Cameras.AGENTVIEW.value: batch['observation'][Cameras.AGENTVIEW.value],
            Cameras.EYE_IN_HAND.value: batch['observation'][Cameras.EYE_IN_HAND.value],
        }
        features = policy.encoding_pipeline(obs_for_latent)

        # Get actions
        actions = batch['action']

        # Call _variational_step to get both posterior and prior
        posterior_output, prior_output = variational_algo._variational_step(features, actions)

    z_posterior = posterior_output[LATENT_KEY]
    z_prior = prior_output[PRIOR_LATENT_KEY]
    mu_posterior = posterior_output[MU_KEY]
    mu_prior = prior_output[PRIOR_MU_KEY]
    logvar_posterior = posterior_output[LOGVAR_KEY]
    logvar_prior = prior_output[PRIOR_LOGVAR_KEY]

    logging.info(f"  Posterior z: mean={z_posterior.mean().item():.4f}, std={z_posterior.std().item():.4f}")
    logging.info(f"  Prior z:     mean={z_prior.mean().item():.4f}, std={z_prior.std().item():.4f}")
    logging.info(f"  Posterior mu: mean={mu_posterior.mean().item():.4f}, std={mu_posterior.std().item():.4f}")
    logging.info(f"  Prior mu:     mean={mu_prior.mean().item():.4f}, std={mu_prior.std().item():.4f}")
    logging.info(f"  Posterior logvar: mean={logvar_posterior.mean().item():.4f}, std={logvar_posterior.std().item():.4f}")
    logging.info(f"  Prior logvar:     mean={logvar_prior.mean().item():.4f}, std={logvar_prior.std().item():.4f}")

    # Compute metrics
    z_mae = (z_posterior - z_prior).abs().mean().item()
    mu_mae = (mu_posterior - mu_prior).abs().mean().item()
    logvar_mae = (logvar_posterior - logvar_prior).abs().mean().item()
    z_cosine_sim = torch.nn.functional.cosine_similarity(z_posterior, z_prior, dim=-1).mean().item()

    logging.info(f"  Z MAE (posterior vs prior): {z_mae:.4f}")
    logging.info(f"  Mu MAE (posterior vs prior): {mu_mae:.4f}")
    logging.info(f"  Logvar MAE (posterior vs prior): {logvar_mae:.4f}")
    logging.info(f"  Z cosine similarity: {z_cosine_sim:.4f}")

    # Compute MMD between posterior and prior
    def compute_mmd(x, y):
        dim = x.size(1)
        x = x.unsqueeze(1)  # (N, 1, D)
        y = y.unsqueeze(0)  # (1, M, D)
        mean_sq_diff_xx = (x - x.transpose(0, 1)).pow(2).mean(dim=2)
        mean_sq_diff_yy = (y - y.transpose(0, 1)).pow(2).mean(dim=2)
        mean_sq_diff_xy = (x - y).pow(2).mean(dim=2)
        k_xx = torch.exp(-mean_sq_diff_xx / dim).mean()
        k_yy = torch.exp(-mean_sq_diff_yy / dim).mean()
        k_xy = torch.exp(-mean_sq_diff_xy / dim).mean()
        return (k_xx + k_yy - 2 * k_xy).item()

    mmd = compute_mmd(z_posterior, z_prior)
    logging.info(f"  MMD(posterior, prior): {mmd:.6f}")

    if z_cosine_sim < 0.5:
        logging.error("  DIAGNOSIS: Prior and posterior are NOT aligned (cosine_sim < 0.5)")
        logging.error("  → This is the ROOT CAUSE: prior doesn't match posterior distribution")
        logging.error("  → Solutions: Increase MMD weight, train longer, use stronger prior architecture")
    elif z_mae > 1.0:
        logging.warning("  DIAGNOSIS: Prior and posterior have high MAE - partial alignment")
    else:
        logging.info("  DIAGNOSIS: Prior and posterior appear reasonably aligned")

    # ========== TEST 5b: Decode with POSTERIOR z vs PRIOR z ==========
    logging.info("\n--- TEST 5b: Decode with POSTERIOR z vs PRIOR z ---")
    with torch.no_grad():
        # Decode with posterior latent (training path)
        features_with_posterior = {**features, LATENT_KEY: z_posterior}
        pred_with_posterior = variational_algo.base_algorithm.forward(
            network=policy.decoder,
            features=features_with_posterior,
            actions=None,
        )

        # Decode with prior latent (inference path)
        features_with_prior = {**features, LATENT_KEY: z_prior}
        pred_with_prior = variational_algo.base_algorithm.forward(
            network=policy.decoder,
            features=features_with_prior,
            actions=None,
        )

    pos_key = ProprioKey.EE_POS_ACTION.value
    posterior_pred_pos = pred_with_posterior[pos_key][0, 0].cpu().numpy()
    prior_pred_pos = pred_with_prior[pos_key][0, 0].cpu().numpy()

    logging.info(f"  Pred pos (with POSTERIOR z): {posterior_pred_pos}")
    logging.info(f"  Pred pos (with PRIOR z):     {prior_pred_pos}")
    logging.info(f"  GT pos (normalized):         {gt_pos_norm}")

    posterior_mae = np.abs(posterior_pred_pos - gt_pos_norm).mean()
    prior_mae = np.abs(prior_pred_pos - gt_pos_norm).mean()

    logging.info(f"  MAE with POSTERIOR z: {posterior_mae:.4f}")
    logging.info(f"  MAE with PRIOR z:     {prior_mae:.4f}")

    if posterior_mae < 0.1 and prior_mae > 0.2:
        logging.error("  CONFIRMED: Posterior z works, Prior z fails")
        logging.error("  → Prior is not learning to match posterior distribution")
        logging.error("  → This is a TRAINING issue, not a loading/inference issue")

    logging.info("\n--- TEST 4: Loop through timesteps ---")

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
    logging.info(f"Gripper match: {sum(grip_matches)}/{len(grip_matches)}")

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

    # Check observation_space keys
    logging.info(f"\n--- observation_space.observations_metadata keys ---")
    for key in policy.observation_space.observations_metadata.keys():
        logging.info(f"  {key}")

    # Check normalizer keys
    logging.info(f"\n--- normalizer.params_dict keys ---")
    for key in normalizer.params_dict.keys():
        logging.info(f"  {key}")

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
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/cluster/workspaces/mazzalore/bowel_retraction/act_libero/20251230_221343"
    checkpoint_name = sys.argv[2] if len(sys.argv) > 2 else "best-76-0.0981.ckpt"
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