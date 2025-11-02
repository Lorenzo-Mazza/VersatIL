import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from diffusers import DDPMScheduler
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from legacy_constants import ROBOT_STATE_KEY, OBSERVATION_KEY, SHAPE_KEY, ACTION_KEY, POSITION_ACTION_KEY, DiffusionArchitecture, IS_PAD_KEY, GRIPPER_ACTION_KEY, \
    ExplanationType
from metrics import Metrics, DiffusionFlowMetrics
from model.diffusion.conditional_unet_1d import ConditionalUnet1D
from model.diffusion.transformer import TransformerForDiffusion
from model.diffusion.vision_encoder.encoder import MultiImageObsEncoder
from model.common.normalizer import LinearNormalizer
from model.diffusion.mask_generator import LowdimMaskGenerator
from model.explainer import compute_gradcam_custom, compute_saliency_maps, compute_integrated_grad_maps
from pytorch_utils import dict_apply


class FlowMatchingPolicy(nn.Module):
    def __init__(self,
                 shape_meta: dict,
                 architecture: str,
                 horizon: int,
                 n_action_steps: int,
                 n_obs_steps: int,
                 camera_names: List[str],
                 backbone: str,
                 pretrained_backbone: bool = False,
                 depth_fusion_strategy: Optional[str] = None,
                 num_inference_steps: int = 100,
                 obs_as_global_cond: bool = True,
                 crop_size: tuple[int, int] = (212, 212),
                 random_crop: bool = True,
                 imagenet_norm: bool = False,
                 diffusion_step_embed_dim: int = 128,
                 down_dims = (256, 512, 1024),
                 kernel_size: int = 5,
                 n_groups: int = 8,
                 use_group_norm: bool = True,
                 sigma: float = 0.0,
                 predict_gripper_action: bool = False,
                freeze_dformer: bool = False,
                 dformer_checkpoint_path: Optional[str] = None,
                 **kwargs):
        super().__init__()

        # Parse shapes
        action_shape = shape_meta[ACTION_KEY][SHAPE_KEY]
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        # Convert shape_meta to format expected by MultiImageObsEncoder
        # MultiImageObsEncoder expects: {'key': shape_tuple, ...}
        # where shape_tuple is either (3,H,W) for RGB, (1,H,W) for Depth Map or (D,) for state
        obs_shapes = {}
        if ROBOT_STATE_KEY in shape_meta[OBSERVATION_KEY]:
            obs_shapes[ROBOT_STATE_KEY] = shape_meta[OBSERVATION_KEY][ROBOT_STATE_KEY][SHAPE_KEY]  # should be (3,) or (6,)
        for camera_name in camera_names:
            obs_shapes[camera_name] = shape_meta[OBSERVATION_KEY][camera_name][SHAPE_KEY]

        obs_encoder = MultiImageObsEncoder(
            obs_shapes=obs_shapes,
            resize_shape=None,
            crop_shape=crop_size,
            random_crop=random_crop,
            use_group_norm=use_group_norm,
            imagenet_norm=imagenet_norm,
            backbone=backbone,
            pretrained=pretrained_backbone,
            depth_fusion_strategy=depth_fusion_strategy,
            dformer_checkpoint_path=dformer_checkpoint_path,
            freeze_dformer=freeze_dformer,
        )
        obs_feature_dim = obs_encoder.get_output_dim()

        # Create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        match architecture:
            case DiffusionArchitecture.UNET.value:
                model = ConditionalUnet1D(
                    input_dim=input_dim,
                    global_cond_dim=global_cond_dim,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=down_dims,
                    kernel_size=kernel_size,
                    n_groups=n_groups
                )
            case DiffusionArchitecture.TRANSFORMER.value:
                model = TransformerForDiffusion(
                    input_dim=input_dim,
                    output_dim=input_dim,
                    horizon=horizon,
                    n_obs_steps=n_obs_steps,
                    cond_dim= obs_feature_dim,
                    n_layer=8,
                    n_cond_layers=0,  # >0: use transformer encoder for cond, otherwise use MLP
                    n_head=4,
                    n_emb=256,
                    p_drop_emb=0.0,
                    p_drop_attn=0.3,
                    causal_attn=True,
                    time_as_cond=True, # if false, use BERT like encoder only arch, time as input
                    obs_as_cond=obs_as_global_cond,
                )
            case _:
                raise ValueError(f"Unknown diffusion architecture: {architecture}")

        self.obs_encoder = obs_encoder
        self.model = model
        #: Create conditional flow matcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=sigma)

        # Save parameters
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.predict_gripper_action = predict_gripper_action

        if num_inference_steps is None:
            raise ValueError("inference steps number cannot be None")
        self.num_inference_steps = num_inference_steps
        self.normalizer = LinearNormalizer()
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )



    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set normalizer for observations and actions"""
        self.normalizer.load_state_dict(normalizer.state_dict())


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        # Normalize observations.
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, _ = value.shape[:2]
        Da = self.action_dim
        To = self.n_obs_steps  # Number of observation steps to use for global conditioning.

        # Compute global conditioning features from the first To time steps.
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs, is_train=False)
        global_cond = nobs_features.reshape(B, -1)

        # For global conditioning, we do not override any trajectory elements.
        # Directly sample the action trajectory.
        sample = self.conditional_sample(global_cond, device, dtype=global_cond.dtype)

        # Extract the predicted actions.
        naction_pred = sample[..., :Da]
        if self.predict_gripper_action:
            position_pred = self.normalizer[POSITION_ACTION_KEY].unnormalize(naction_pred[:, :, :-1])
            gripper_norm = naction_pred[:, :, -1:]
            gripper_pred = (gripper_norm + 1) / 2  # float in [0,1]
            action_pred = torch.cat([position_pred, gripper_pred], dim=-1)
        else:
            action_pred = self.normalizer[POSITION_ACTION_KEY].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        return action_pred[:, start:end]

    def _reduce_padded_loss(self, loss_all: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        return (loss_all * ~is_pad.unsqueeze(-1)).sum() / (~is_pad).sum()


    def compute_loss(self, batch, is_train: Optional[bool] = None, gripper_positive_class_weight: Optional[torch.Tensor]=None) -> DiffusionFlowMetrics:
        nobs = self.normalizer.normalize(batch[OBSERVATION_KEY])
        nactions = self.normalizer[POSITION_ACTION_KEY].normalize(batch[POSITION_ACTION_KEY])
        if self.predict_gripper_action:
            gripper = batch[GRIPPER_ACTION_KEY]
            ngripper = 2 * gripper - 1  # This converts {0,1} to {-1,1}, which is the expected range of Diffusion model outputs
            nactions = torch.cat([nactions, ngripper], dim=-1)

        batch_size = nactions.shape[0]

        # Compute global conditioning features.
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs, is_train=is_train)
        global_cond = nobs_features.reshape(batch_size, -1)

        trajectory = nactions
        noise = torch.randn_like(trajectory, device=trajectory.device)
        timestep, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0=noise, x1=trajectory)
        pred = self.model(xt, timestep, global_cond=global_cond)
        is_pad = batch[IS_PAD_KEY]
        all_mse = Metrics.MSE_LOSS.to_metric()(ut, pred, reduction='none')
        loss = self._reduce_padded_loss(all_mse, is_pad=is_pad)
        return DiffusionFlowMetrics(loss=loss)


    def conditional_sample(self, global_cond, device, dtype):
        # Initialize the trajectory with random noise.
        # We use a shape that matches the action trajectory.
        # Since input_dim = action_dim, we create a tensor of that shape.
        dummy_shape = (global_cond.shape[0], self.horizon, self.action_dim)
        trajectory = torch.randn(dummy_shape, device=device, dtype=dtype)
        for t in range(self.num_inference_steps):
            timestep = torch.tensor([t / self.num_inference_steps]).to(device)
            vt = self.model(trajectory, timestep, global_cond=global_cond)
            trajectory = (vt * 1 / self.num_inference_steps + trajectory)
        return trajectory


    def explain_predictions(self, explanation_types: list[str], obs_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:

        class FlowMatchingWrapper(nn.Module):

            def __init__(self, encoder, denoising_model, n_obs_steps, architecture):
                super().__init__()
                self.obs_encoder = encoder
                self.denoising_model = denoising_model
                self.n_obs_steps = n_obs_steps
                self.architecture = architecture


            def forward(self, observation, device):
                value = next(iter(observation.values()))
                batch_size, _ = value.shape[:2]
                this_nobs = dict_apply(observation, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
                nobs_features = self.obs_encoder(this_nobs, is_train=False)
                if isinstance(self.denoising_model, TransformerForDiffusion):
                    global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
                else:
                    global_cond = nobs_features.reshape(batch_size, -1)
                # Fixed zero noise for deterministic sampling in explanations
                dummy_shape = (batch_size, self.architecture.horizon, self.architecture.action_dim)
                trajectory = torch.zeros(dummy_shape, device=device, dtype=global_cond.dtype)
                for t in range(self.architecture.num_inference_steps):
                    timestep = torch.tensor([t / self.architecture.num_inference_steps]).to(device)
                    vt = self.denoising_model(trajectory, timestep, global_cond=global_cond)
                    trajectory = (vt * 1 / self.architecture.num_inference_steps + trajectory)
                return trajectory.unsqueeze(0)


            def target_layers_getter(self, camera: str) -> List[nn.Module]:
                """Get target layers for Grad-CAM style heatmaps based on camera name."""
                return [self.obs_encoder.key_encoders[camera].layer4[-1]]


        input_channels = {cam: self.obs_encoder.key_encoders[cam].conv1.in_channels for cam in self.obs_encoder.image_keys}
        target_action_dim = None
        target_chunk_idx = 0
        normalized_obs = self.normalizer.normalize(obs_dict)
        device = next(iter(normalized_obs.values())).device


        def output_selector(predicted_actions: torch.Tensor) -> torch.Tensor:
            if target_action_dim is not None:
                return predicted_actions[:, target_chunk_idx, target_action_dim]
            else:
                return predicted_actions.mean(dim=(1, 2))


        model_to_explain = FlowMatchingWrapper(encoder=self.obs_encoder, denoising_model=self.model, n_obs_steps=self.n_obs_steps, architecture=self)
        requires_grad_states = [param.requires_grad for param in model_to_explain.parameters()]
        for param in model_to_explain.parameters():
            param.requires_grad_(True)

        explanation_maps = {}
        for explanation in explanation_types:
            match explanation:
                case ExplanationType.GRADCAM_PLUS_PLUS.value:
                    explanation_maps[ExplanationType.GRADCAM_PLUS_PLUS.value] = compute_gradcam_custom(
                        model=model_to_explain,
                        explanation_type=ExplanationType.GRADCAM_PLUS_PLUS.value,
                        observation=normalized_obs,
                        camera_names=self.obs_encoder.image_keys,
                        input_channels=input_channels,
                        target_layers_getter=model_to_explain.target_layers_getter,
                        output_selector=output_selector,
                        target_camera=None,
                        eigen_smooth=False,
                        device=device,  # forward_kwargs
                    )
                case ExplanationType.SALIENCY_MAP.value:
                    explanation_maps[ExplanationType.SALIENCY_MAP.value] = compute_saliency_maps(
                        model=model_to_explain,
                        observation=normalized_obs,
                        camera_names=self.obs_encoder.image_keys,
                        output_selector=output_selector,
                        target_camera=None,
                        smooth=False,
                        device=device,
                    )
                case ExplanationType.INTEGRATED_GRADIENT.value:
                    explanation_maps[ExplanationType.INTEGRATED_GRADIENT.value] = compute_integrated_grad_maps(
                        model=model_to_explain,
                        observation=normalized_obs,
                        camera_names=self.obs_encoder.image_keys,
                        output_selector=output_selector,
                        target_camera=None,
                        num_steps=50,
                        baseline=None,
                        smooth=False,
                        device=device,
                    )
                case ExplanationType.GRADCAM.value:
                    explanation_maps[ExplanationType.GRADCAM.value] = compute_gradcam_custom(
                        model=model_to_explain,
                        explanation_type=ExplanationType.GRADCAM.value,
                        observation=normalized_obs,
                        camera_names=self.obs_encoder.image_keys,
                        input_channels=input_channels,
                        target_layers_getter=model_to_explain.target_layers_getter,
                        output_selector=output_selector,
                        target_camera=None,
                        eigen_smooth=False,
                        device=device,  # forward_kwargs
                    )
                case ExplanationType.ABLATION_CAM.value:
                    explanation_maps[ExplanationType.ABLATION_CAM.value] = compute_gradcam_custom(
                        model=model_to_explain,
                        explanation_type=ExplanationType.ABLATION_CAM.value,
                        observation=normalized_obs,
                        camera_names=self.obs_encoder.image_keys,
                        input_channels=input_channels,
                        target_layers_getter=model_to_explain.target_layers_getter,
                        output_selector=output_selector,
                        target_camera=None,
                        eigen_smooth=False,
                        device=device,  # forward_kwargs
                    )
                case _:
                    raise ValueError(f"Unknown explanation type: {explanation}")

        for param, state in zip(model_to_explain.parameters(), requires_grad_states):
            param.requires_grad_(state)
        return explanation_maps