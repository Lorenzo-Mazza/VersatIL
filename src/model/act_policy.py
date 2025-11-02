import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Optional

from legacy_config import ACTConfig
from legacy_constants import OBSERVATION_KEY, IS_PAD_KEY, ExplanationType, POSITION_ACTION_KEY, GRIPPER_ACTION_KEY
from metrics import Metrics, ActEpochMetrics
from model.common.normalizer import LinearNormalizer
from model.detr.detr_vae import DETRVAE
from model.explainer import compute_saliency_maps, compute_integrated_grad_maps, compute_gradcam_custom
from pytorch_utils import dict_apply


class ACTPolicy(nn.Module):
    def __init__(self,
                 config: ACTConfig
                 ):
        super().__init__()
        self.model = DETRVAE(
            camera_names=config.camera_names,
            device=config.device,
            backbone=config.backbone,
            position_embedding=config.position_embedding,
            lr_backbone=config.lr_backbone,
            dilation=config.dilation,
            enc_layers=config.enc_layers,
            dec_layers=config.dec_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            nheads=config.nheads,
            pre_norm=config.pre_norm,
            obs_dimension=config.state_dim,
            hidden_dimension=config.hidden_dim,
            #TODO: pass a `shape_meta` with all shapes to prevent action_dimension mis-specification.
            action_dimension=config.action_dim, 
            chunk_size=config.action_horizon,
            depth_fusion_strategy=config.depth_fusion,
            use_fake_proprio=config.use_fake_proprio,
            predict_gripper_action=config.predict_gripper_action,
            freeze_dformer=config.freeze_dformer,
            dformer_checkpoint_path=config.dformer_checkpoint_path

        ).to(device=torch.device(config.device))
        self.normalizer = LinearNormalizer()
        self.predict_gripper_action = config.predict_gripper_action
        
        self.sinkhorn_weight = config.sinkhorn_weight
        self.mse_weight = config.mse_weight
        self.length_weight = config.length_weight
        self.bce_weight = config.bce_weight
        self.l1_weight = config.l1_weight
        self.kl_weight = config.kl_weight


    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set normalizer for observations and actions"""
        self.normalizer.load_state_dict(normalizer.state_dict())


    def _reduce_padded_loss(self, loss_all: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        return (loss_all * ~is_pad.unsqueeze(-1)).sum() / (~is_pad).sum()

    def compute_loss(self, 
                     batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], 
                     is_train: Optional[bool]=None, 
                     gripper_positive_class_weight: Optional[torch.Tensor]=None) -> ActEpochMetrics:
        """ Compute the loss for the given batch.

        Args:
            batch (Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]): A dictionary containing the batch data.
                It should contain keys `OBSERVATION_KEY`, `POSITION_ACTION_KEY`, and `IS_PAD_KEY`. If gripper action is predicted, it should also contain key `GRIPPER_ACTION_KEY`.
            is_train (Optional[bool]): A flag indicating whether the model is in training mode. Not used in this method.
            pos_weight (Optional[torch.Tensor]): A weight for the positive class of the gripper action. Used only if gripper action is predicted.
        Returns:
            ActEpochMetrics: An object containing the computed losses.
        """
        normalized_obs = self.normalizer.normalize(batch[OBSERVATION_KEY])
        # Remove the observation dimension because in ACT we use a single observation horizon.
        normalized_obs = dict_apply(normalized_obs, lambda x: x.squeeze(1))
        target_position_actions = self.normalizer[POSITION_ACTION_KEY].normalize(batch[POSITION_ACTION_KEY])
        target_gripper_actions = batch[GRIPPER_ACTION_KEY] if self.predict_gripper_action else None
        is_pad = batch[IS_PAD_KEY]
        position_prediction, gripper_prediction, is_pad_hat, (mu, logvar) = self.model(observation=normalized_obs, position_actions=target_position_actions, gripper_actions=target_gripper_actions, is_pad=is_pad)
        
        total_kld, _, _ = Metrics.KL_DIVERGENCE.to_metric()(mu, logvar)
        l1_loss_all = Metrics.L1_LOSS.to_metric()(position_prediction, target_position_actions, reduction='none')
        l1_loss = self._reduce_padded_loss(l1_loss_all, is_pad)

        mse_loss_all = Metrics.MSE_LOSS.to_metric()(position_prediction, target_position_actions, reduction='none')
        mse_loss = self._reduce_padded_loss(mse_loss_all, is_pad)
        sinkhorn_loss_all = Metrics.SINKHORN_LOSS.to_metric()(position_prediction.contiguous(), target_position_actions.contiguous())
        sinkhorn_loss = self._reduce_padded_loss(sinkhorn_loss_all, is_pad)
        bce_loss = 0
        if self.predict_gripper_action:
            bce_loss = Metrics.BINARY_CROSS_ENTROPY_WITH_LOGITS.to_metric()(gripper_prediction, target_gripper_actions, pos_weight=gripper_positive_class_weight, reduction='none')
            bce_loss = self._reduce_padded_loss(bce_loss, is_pad)

        position_prediction_padded = position_prediction * ~is_pad.unsqueeze(-1)
        target_position_actions_padded = target_position_actions * ~is_pad.unsqueeze(-1)
        pred_length = torch.norm(position_prediction_padded[:, 1:] - position_prediction_padded[:, :-1], dim=-1).mean()
        target_length = torch.norm(target_position_actions_padded[:, 1:] - target_position_actions_padded[:, :-1], dim=-1).mean()
        length_loss = (pred_length - target_length)**2

        loss = mse_loss*self.mse_weight + sinkhorn_loss*self.sinkhorn_weight + bce_loss*self.bce_weight + length_loss*self.length_weight + l1_loss*self.l1_weight + total_kld[0]*self.kl_weight

        return ActEpochMetrics(mse_loss= mse_loss,
                        sinkhorn_loss= sinkhorn_loss,
                        length_loss= length_loss,
                        loss = loss,
                        binary_cross_entropy=bce_loss,
                        l1_loss=l1_loss,
                        kl_divergence=total_kld[0])

    def predict_action(self, obs_dict: Dict[str, Dict[str, torch.Tensor]], device: torch.device
                       ) -> torch.Tensor:
        """ Predict actions based on the given batch of observations.
        Args:
            obs_dict: A dictionary containing the batch data.
            device: The device to which the model is moved. Unused here.
        Returns:
            torch.Tensor: The unnormalized prediction.
        """
        normalized_obs = self.normalizer.normalize(obs_dict)
        # Remove the observation dimension because in ACT we use a single observation horizon.
        normalized_obs = dict_apply(normalized_obs, lambda x: x.squeeze(1))
        position_prediction, gripper_prediction, _, (_, _) = self.model(observation=normalized_obs)  # sample from prior
        unnormalized_prediction = self.normalizer[POSITION_ACTION_KEY].unnormalize(position_prediction)
        
        if self.predict_gripper_action:
            gripper_prediction = F.sigmoid(gripper_prediction)
            unnormalized_prediction = torch.cat([unnormalized_prediction, gripper_prediction], axis=-1)
        
        return unnormalized_prediction


    def explain_predictions(self, explanation_types: list[str], obs_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:

        input_channels = {cam: self.model.backbones[cam][0].body.conv1.in_channels for cam in self.model.camera_names}


        target_action_dim = None
        target_chunk_idx = 0
        normalized_obs = self.normalizer.normalize(obs_dict)
        # Remove the observation dimension because in ACT we use a single observation horizon.
        normalized_obs = dict_apply(normalized_obs, lambda x: x.squeeze(1))


        def output_selector(predicted_actions: torch.Tensor) -> torch.Tensor:
            if target_action_dim is not None:
                return predicted_actions[:, target_chunk_idx, target_action_dim]
            else:
                return predicted_actions.mean(dim=(1, 2))

        explanation_maps = {}
        for explanation in explanation_types:
            match explanation:
                case ExplanationType.GRADCAM_PLUS_PLUS.value:
                    explanation_maps[ExplanationType.GRADCAM_PLUS_PLUS.value] = compute_gradcam_custom(
                        model=self.model,
                        explanation_type=ExplanationType.GRADCAM_PLUS_PLUS.value,
                        observation=normalized_obs,
                        camera_names=self.model.camera_names,
                        input_channels=input_channels,
                        target_layers_getter=self.model.target_layers_getter,
                        output_selector=output_selector,
                        target_camera=None,
                        eigen_smooth=False,
                        position_actions=None,  # forward_kwargs
                        gripper_actions=None,  # forward_kwargs
                        is_pad=None# forward_kwargs
                    )
                case ExplanationType.SALIENCY_MAP.value:
                    explanation_maps[ExplanationType.SALIENCY_MAP.value] = compute_saliency_maps(
                        model=self.model,
                        observation=normalized_obs,
                        camera_names=self.model.camera_names,
                        output_selector=output_selector,
                        target_camera=None,
                        smooth=False,
                        position_actions=None,  # forward_kwargs
                        gripper_actions=None,  # forward_kwargs
                        is_pad=None# forward_kwargs
                    )
                case ExplanationType.INTEGRATED_GRADIENT.value:
                    explanation_maps[ExplanationType.INTEGRATED_GRADIENT.value] = compute_integrated_grad_maps(
                        model=self.model,
                        observation=normalized_obs,
                        camera_names=self.model.camera_names,
                        output_selector=output_selector,
                        target_camera=None,
                        num_steps=500,
                        baseline=None,
                        smooth=False,
                        position_actions=None,  # forward_kwargs
                        gripper_actions=None,  # forward_kwargs
                        is_pad=None# forward_kwargs
                    )
                case ExplanationType.GRADCAM.value:
                    explanation_maps[ExplanationType.GRADCAM.value] = compute_gradcam_custom(
                        model=self.model,
                        explanation_type=ExplanationType.GRADCAM.value,
                        observation=normalized_obs,
                        camera_names=self.model.camera_names,
                        input_channels=input_channels,
                        target_layers_getter=self.model.target_layers_getter,
                        output_selector=output_selector,
                        target_camera=None,
                        eigen_smooth=False,
                        position_actions=None,  # forward_kwargs
                        gripper_actions=None,  # forward_kwargs
                        is_pad=None# forward_kwargs
                    )
                case ExplanationType.ABLATION_CAM.value:
                    explanation_maps[ExplanationType.ABLATION_CAM.value] = compute_gradcam_custom(
                        model=self.model,
                        explanation_type=ExplanationType.ABLATION_CAM.value,
                        observation=normalized_obs,
                        camera_names=self.model.camera_names,
                        input_channels=input_channels,
                        target_layers_getter=self.model.target_layers_getter,
                        output_selector=output_selector,
                        target_camera=None,
                        eigen_smooth=False,
                        position_actions=None,  # forward_kwargs
                        gripper_actions=None,  # forward_kwargs
                        is_pad=None# forward_kwargs
                    )
                case _:
                    raise ValueError(f"Unknown explanation type: {explanation}")

        return explanation_maps





