import logging
import time  # Added for timing
import argparse
import numpy as np
import torch
from imitation_learning_toolkit.sockets.model_client import AbstractModelClient, Action
from legacy_config import load_config
from constants import PolicyType, Cameras, ROBOT_STATE_KEY, ImageNormalizationType,IMAGENET_RGB_MEAN, IMAGENET_RGB_STD
from workspace import DiffusionWorkspace, FlowMatchingWorkspace, ACTWorkspace, TaskWorkspace, PhaseACTWorkspace
import albumentations as A
from albumentations.pytorch import ToTensorV2


class InferenceClient(AbstractModelClient):
    def __init__(self,
                 device: torch.device,
                 checkpoint_path: str,
                 temporal_agg: bool = True, # Whether to use temporal aggregation for actions
                 update_rate_hz: float | None = None,
                 **kwargs):
        self.checkpoint_path = checkpoint_path
        self.config = load_config(checkpoint_path)
        self.config.device = device
        self.temporal_agg = temporal_agg
        update_rate_hz = self.config.update_rate_hz if update_rate_hz is None else update_rate_hz
        super().__init__(observation_buffer_size=self.config.obs_horizon, request_depth=self.config.use_depth,
                         request_rectified_images=self.config.use_rectified, request_gripper_state=self.config.predict_gripper_action,
                         predicts_in_camera_frame=self.config.predict_in_camera_frame,
                         predicts_delta=self.config.deltas_as_actions, obs_robot_frame=self.config.obs_robot_frame,
                         obs_camera_frame=self.config.obs_camera_frame,
                         device=device,
                         update_rate_hz=update_rate_hz,
                         enable_logging=False,
                         **kwargs)
        additional_targets = {"right_image": "image"}
        if self.config.use_depth:
            additional_targets["depth"] = "mask"
        self.transform = A.Compose([
            A.Resize(height=self.config.image_height, width=self.config.image_width),
            ToTensorV2(), ],
            additional_targets=additional_targets,
        )
        self.max_timesteps = 10000
        self.action_dim = self.config.action_dim
        self.action_horizon = self.config.action_horizon
        self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps + self.action_horizon, self.action_dim]).to(self.device)
        self.timestep = 0
        self.current_all_actions = None

    def _load_model(self) -> TaskWorkspace:
        if self.config.policy_name == PolicyType.DIFFUSION_POLICY.value:
            workspace = DiffusionWorkspace(config=self.config, is_inference=True)
        elif self.config.policy_name == PolicyType.FLOW_MATCHING.value:
            workspace = FlowMatchingWorkspace(config=self.config, is_inference=True)
        elif self.config.policy_name == PolicyType.ACT.value:
            workspace = ACTWorkspace(config=self.config, is_inference=True)
        elif self.config.policy_name == PolicyType.PHASE_ACT.value:
            workspace = PhaseACTWorkspace(config=self.config, is_inference=True)
        else:
            raise ValueError(f"Unknown policy type: {self.config.policy_name}")
        workspace.load_checkpoint(path=f"{self.checkpoint_path}/latest.pt")
        return workspace

    def imagenet_norm_back(self, img, mean, std):
        return img*std + mean

    def get_actions_from_model(self) -> list[Action]:
        """Computes the next actions using the trained policy model.
            Without temporal_agg (False):
            - Model predicts a chunk of `self.action_horizon` actions.
            - Executes them sequentially without waiting for new observations.
            - Replans only after full chunk, following fixed short trajectories for efficiency but less reactivity.
            With temporal_agg (True):
            - Predicts new chunk every step.
            - For current t, averages overlapping predictions for t from recent chunks (up to `self.action_horizon`).
            - Uses exponential weights (k=0.01, slight bias to older preds, nearly uniform) for smoothing.
            - Adapts every step while reducing jitter by ensembling predictions.
        """
        total_start_time = time.time()
        print(f"\n=== TIMESTEP {self.timestep} - Starting get_actions_from_model ===")
        
        # ========== INPUT PREPROCESSING START ==========
        preprocessing_start_time = time.time()
        print(f"[TIMING] Input preprocessing started at: {preprocessing_start_time:.6f}")
        
        if self.obs_camera_frame and self.obs_robot_frame:
            state_dim = 6
        elif self.obs_camera_frame or self.obs_robot_frame:
            state_dim = 3
        else:
            state_dim = 0

        if state_dim> 0:
            last_states = self.robot_state_buffer[-self.observation_buffer_size:]
            qpos = np.array([state[:state_dim] for state in last_states])
            qpos_tensor = torch.tensor(qpos, dtype=torch.float32).unsqueeze(0) # Shape: (1, observation_buffer_size, state_dim)
        else:
            qpos_tensor = None

        left_img_list = self.left_image_buffer[-self.observation_buffer_size:]
        right_img_list = self.right_image_buffer[-self.observation_buffer_size:]

        # Save images for debugging
        #image_save_start = time.time()
        #for left_img, right_img in zip(left_img_list, right_img_list):
        #    Image.fromarray(left_img).save(f'/mnt/cluster/temp/needle_driving/inference_imgs/left_img_{self.timestep}.png')
        #    Image.fromarray(right_img).save(f'/mnt/cluster/temp/needle_driving/inference_imgs/right_img_{self.timestep}.png')
        #print(f"[TIMING] Image saving took: {time.time() - image_save_start:.6f} seconds")

        # Process depth images if needed
        depth_processing_start = time.time()
        depth_imgs = None
        if self.request_depth:
            depth_img_list = self.depth_buffer[-self.observation_buffer_size:]
            print(len(depth_img_list))
            print(depth_img_list[0].shape)
            print(depth_img_list[0].max())
            
            transformed = [self.transform(image=left_np, right_image=right_np, depth=depth_np)
                          for left_np, right_np, depth_np in zip(left_img_list, right_img_list, depth_img_list)]
            depth_tensors = [t['depth'] for t in transformed]
            depth_imgs = torch.stack(depth_tensors).unsqueeze(0).unsqueeze(-3) # Shape: (1, observation_buffer_size, 1, H, W)
            print("Min:", depth_imgs.min().item())
            print("Max:", depth_imgs.max().item())
            print("Mean:", depth_imgs.mean().item())
            print("Std:", depth_imgs.std().item())
            max_depth = 9.352702140808105
            depth_imgs = torch.clamp(depth_imgs, min=0.0, max=max_depth)
            
            # Save each depth map as a greyscale image
            depth_imgs_normalized = (depth_imgs / max_depth) * 255.0
            depth_imgs_uint8 = depth_imgs_normalized.to(torch.uint8)
            #for i in range(depth_imgs_uint8.shape[1]): # observation_buffer_size
            #    vutils.save_image(depth_imgs_uint8[0, i].float() / 255.0, f'/mnt/cluster/temp/needle_driving/depth_maps/depth_map_{i}.png') # save_image expects [0,1] for float
        else:
            transformed = [self.transform(image=left_np, right_image=right_np)
                          for left_np, right_np in zip(left_img_list, right_img_list)]
        
        print(f"[TIMING] Depth plus RGB transform took: {time.time() - depth_processing_start:.6f} seconds")

        # Process RGB images
        rgb_processing_start = time.time()
        left_tensors = [t['image']/255.0 for t in transformed]
        right_tensors = [t['right_image']/255.0 for t in transformed]

        if self.config.image_norm_type == ImageNormalizationType.IMAGENET.value:
            mean = torch.tensor(IMAGENET_RGB_MEAN)
            std = torch.tensor(IMAGENET_RGB_STD)
            left_tensors = [self.imagenet_norm_back(image=left, mean=mean, std=std)
                           for left in left_tensors]
            right_tensors = [self.imagenet_norm_back(image=right, mean=mean, std=std)
                            for right in right_tensors]

        left_imgs = torch.stack(left_tensors).unsqueeze(0) # Shape: (1, observation_buffer_size, 3, H, W)
        right_imgs = torch.stack(right_tensors).unsqueeze(0) # Shape: (1, observation_buffer_size, 3, H, W)
        print(f"[TIMING] RGB processing took: {time.time() - rgb_processing_start:.6f} seconds")

        # Save processed images for debugging
        #processed_save_start = time.time()
        #for i in range(left_imgs.shape[1]): # observation_buffer_size
        #    vutils.save_image(left_imgs[0, i].float(), f'/mnt/cluster/temp/needle_driving/inference_imgs/left_{i}_timestep_{self.timestep}.png') # save_image expects [0,1] for float
        #    vutils.save_image(right_imgs[0, i].float(), f'/mnt/cluster/temp/needle_driving/inference_imgs/right_{i}_timestep_{self.timestep}.png') # save_image expects [0,1] for float
        #print(f"[TIMING] Processed image saving took: {time.time() - processed_save_start:.6f} seconds")

        # Create observation dictionary
        obs_dict = {
            Cameras.LEFT.value: left_imgs,
            Cameras.RIGHT.value: right_imgs,
        }

        if state_dim>0:
            obs_dict[ROBOT_STATE_KEY] = qpos_tensor

        if self.request_depth:
            obs_dict[Cameras.DEPTH.value] = depth_imgs

        # Get current robot position for action conversion
        if self.predicts_in_camera_frame and self.obs_camera_frame and self.obs_robot_frame:
            # If the model predicts in camera frame, and we use both robot and camera frame observations,
            # we need to get the current robot position in camera frame, which is the last 3 elements of the state
            current_robot_position = self.robot_state_buffer[-1][3:6]
        else:
            current_robot_position = self.robot_state_buffer[-1][:3]

        preprocessing_end_time = time.time()
        preprocessing_duration = preprocessing_end_time - preprocessing_start_time
        print(f"[TIMING] Input preprocessing completed in: {preprocessing_duration:.6f} seconds")
        # ========== INPUT PREPROCESSING END ==========

        # ========== MODEL INFERENCE START ==========
        inference_start_time = time.time()
        print(f"[TIMING] Model inference started at: {inference_start_time:.6f}")
        
        self.current_all_actions = self.model.predict_actions(obs_dict=obs_dict)
        
        inference_end_time = time.time()
        inference_duration = inference_end_time - inference_start_time
        print(f"[TIMING] Model inference completed in: {inference_duration:.6f} seconds")
        # ========== MODEL INFERENCE END ==========

        # ========== POST-PROCESSING START ==========
        postprocessing_start_time = time.time()
        print(f"[TIMING] Post-processing started at: {postprocessing_start_time:.6f}")

        if self.temporal_agg:
            temporal_agg_start = time.time()
            raw_action = self.get_exponential_averaged_action()
            print(f"[TIMING] Temporal aggregation took: {time.time() - temporal_agg_start:.6f} seconds")
            
            raw_action = raw_action.cpu().detach().numpy()
            raw_position_action = raw_action[:3]
            raw_gripper_action = raw_action[-1] if self.config.predict_gripper_action else None

            if not self.predicts_delta:
                raw_position_action = raw_position_action - current_robot_position

            robot_action = np.concatenate((raw_position_action[:3], [0.])) # we don't predict roll
            gripper_action = raw_gripper_action > 0.5 if self.config.predict_gripper_action else None
            actions = [Action(robot_action=robot_action, gripper_action=gripper_action)]
        else:
            actions = []
            for i in range(self.config.action_horizon):
                raw_action = self.current_all_actions[0, i].cpu().detach().numpy()
                raw_position_action = raw_action[:3]
                raw_gripper_action = raw_action[-1] if self.config.predict_gripper_action else None

                if not self.predicts_delta:
                    raw_position_action = raw_position_action - current_robot_position

                robot_action = np.concatenate((raw_position_action[:3], [0.]))
                gripper_action = raw_gripper_action > 0.5 if self.config.predict_gripper_action else None
                actions.append(Action(robot_action=robot_action, gripper_action=gripper_action))

        postprocessing_end_time = time.time()
        postprocessing_duration = postprocessing_end_time - postprocessing_start_time
        print(f"[TIMING] Post-processing completed in: {postprocessing_duration:.6f} seconds")
        # ========== POST-PROCESSING END ==========

        self.timestep += 1

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        # ========== SUMMARY TIMING ==========
        print(f"\n[TIMING SUMMARY] Timestep {self.timestep - 1}:")
        print(f"  - Preprocessing: {preprocessing_duration:.6f}s ({preprocessing_duration/total_duration*100:.1f}%)")
        print(f"  - Model inference: {inference_duration:.6f}s ({inference_duration/total_duration*100:.1f}%)")
        print(f"  - Post-processing: {postprocessing_duration:.6f}s ({postprocessing_duration/total_duration*100:.1f}%)")
        print(f"  - TOTAL: {total_duration:.6f}s")
        print(f"  - Effective FPS: {1.0/total_duration:.2f}")
        print(f"=== TIMESTEP {self.timestep - 1} COMPLETE ===\n")

        if self.enable_logging:
            logging.log(level=logging.INFO, msg=f"{actions=}")
        print(actions)
        return actions

    def get_exponential_averaged_action(self) -> torch.Tensor:
        """Averages exponentially the actions predicted for the current timestep from recent chunks."""
        self.all_time_actions[[self.timestep], self.timestep:self.timestep + self.config.action_horizon] = self.current_all_actions[:, :, :self.action_dim]
        actions_for_curr_step = self.all_time_actions[:, self.timestep]
        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / exp_weights.sum()
        exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0)
        return raw_action


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Diffusion Policy/Flow Matching Model Client")
    parser.add_argument(
        "--model-server-address",
        type=str,
        default="localhost",
        help="Address of the model server"
    )
    parser.add_argument(
        "--model-server-port",
        type=int,
        default=5555,
        help="Port of the model server"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/mnt/cluster/workspaces/mazzalore/iros/diffusion_policy_checkpoints/experiment_20250515_113430/latest.pt",
        help="Checkpoint path for the diffusion policy model"
    )
    parser.add_argument(
        "--temporal-agg",
        type=int,
        default=1,
        choices=(0, 1),
        help="1 = use temporal aggregation for actions, 0 = no temporal aggregation"
    )
    parser.add_argument(
        "--update-frequency",
        type=float,
        default=None,
        help="Update frequency in Hz (overrides config file)"
    )
    args = parser.parse_args()
    args.temporal_agg = bool(args.temporal_agg)
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        print("Warning: Running on CPU, this may be slow or go OOM. Consider using a GPU for better performance.")

    client = InferenceClient(
        model_server_address=args.model_server_address,
        model_server_port=args.model_server_port,
        checkpoint_path=args.checkpoint_path,
        temporal_agg=args.temporal_agg,
        device=device,
        update_rate_hz=args.update_frequency,
    )

    try:
        client.update_loop()
    except KeyboardInterrupt:
        print("Shutting down client...")
        client.shutdown()
    except Exception as e:
        print(f"Error: {e}")
        client.shutdown()


if __name__ == "__main__":
    main()