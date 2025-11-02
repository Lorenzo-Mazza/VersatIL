import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Optional

from legacy_config import PhaseACTConfig
from legacy_constants import OBSERVATION_KEY, IS_PAD_KEY, ExplanationType, POSITION_ACTION_KEY, GRIPPER_ACTION_KEY, PHASE_LABEL_KEY
from metrics import Metrics, ActEpochMetrics, PhaseActEpochMetrics
from model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from model.detr.phase_detr_vae import PhaseDETRVAE
from model.explainer import compute_saliency_maps, compute_integrated_grad_maps, compute_gradcam_custom
from pytorch_utils import dict_apply


class PhaseACTPolicy(nn.Module):

    def __init__(self,
                 config: PhaseACTConfig
                 ):
        super().__init__()
        self.model = PhaseDETRVAE(
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
            # TODO: pass a `shape_meta` with all shapes to prevent action_dimension mis-specification.
            action_dimension=config.action_dim,
            chunk_size=config.action_horizon,
            depth_fusion_strategy=config.depth_fusion,
            use_fake_proprio=config.use_fake_proprio,
            predict_gripper_action=config.predict_gripper_action,
            freeze_dformer=config.freeze_dformer,
            dformer_checkpoint_path=config.dformer_checkpoint_path,
            n_phases=config.n_phases,
            phase_learnable_temperature=config.phase_learnable_temperature,
        ).to(device=torch.device(config.device))
        self.normalizer = LinearNormalizer()
        self.predict_gripper_action = config.predict_gripper_action

        self.sinkhorn_weight = config.sinkhorn_weight
        self.mse_weight = config.mse_weight
        self.length_weight = config.length_weight
        self.bce_weight = config.bce_weight
        self.l1_weight = config.l1_weight
        self.kl_weight = config.kl_weight
        self.phase_ce_weight = config.phase_ce_weight
        self.entropy_weight = config.entropy_weight

    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set normalizer for observations and actions"""
        self.normalizer.load_state_dict(normalizer.state_dict())


    def _reduce_padded_loss(self, loss_all: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        return (loss_all * ~is_pad.unsqueeze(-1)).sum() / (~is_pad).sum()


    def compute_loss(self,
                     batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]],
                     is_train: Optional[bool] = None,
                     gripper_positive_class_weight: Optional[torch.Tensor] = None) -> ActEpochMetrics:
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
        phase_labels = batch[PHASE_LABEL_KEY]  # (B, chunk)

        position_preds_list, gripper_preds_list, is_pad_hat, (mu, logvar), phase_logits, phase_probs = self.model(observation=normalized_obs,
                                                                                                                  position_actions=target_position_actions,
                                                                                       gripper_actions=target_gripper_actions, is_pad=is_pad)

        weights = phase_probs.unsqueeze(-1)  # (B, chunk, n_phases, 1)
        stacked_pos_preds = torch.stack(position_preds_list, dim=-1).transpose(2, 3)  # (B, chunk, 3, n_phases) -> transpose to (B, chunk, n_phases, 3)
        position_prediction = torch.sum(stacked_pos_preds * weights, dim=2)  # (B, chunk, 3)
        if self.predict_gripper_action:
            stacked_grip_preds = torch.stack(gripper_preds_list, dim=-1).transpose(2, 3)  # (B, chunk, n_phases, 1)
            gripper_prediction = torch.sum(stacked_grip_preds * weights, dim=2)  # (B, chunk, 1)
        


        if phase_logits.shape[1] == self.model.num_queries and phase_logits.shape[2] == self.model.n_phases:
            # Swap chunk dimension and n_phases dimension for torch cross_entropy.
            phase_logits = phase_logits.transpose(1, 2)
        if phase_logits.dim() == 3 and phase_labels.dim() == 3:
            # Phase logits dimension is (B, n_phases, chunk), but phase_labels dimension is (B, chunk, 1).
            # We need to unsqueeze phase_labels to B, n_phases, chunk, 1).
            phase_logits = phase_logits.unsqueeze(-1)
        phase_ce_loss_all = Metrics.CROSS_ENTROPY.to_metric()(phase_logits, phase_labels, label_smoothing=0.2, reduction='none')
        phase_ce_loss = self._reduce_padded_loss(phase_ce_loss_all, is_pad)

        phase_entropy_all = Metrics.ENTROPY.to_metric()(phase_probs, 1e-8).unsqueeze(-1)  # Shape: (B, chunk, 1)
        phase_entropy = -self._reduce_padded_loss(phase_entropy_all, is_pad) # We want to maximize entropy in the loss

        l1_loss_all = Metrics.L1_LOSS.to_metric()(position_prediction, target_position_actions, reduction='none')
        l1_loss = self._reduce_padded_loss(l1_loss_all, is_pad)

        mse_loss_all = Metrics.MSE_LOSS.to_metric()(position_prediction, target_position_actions, reduction='none')
        mse_loss = self._reduce_padded_loss(mse_loss_all, is_pad)

        sinkhorn_loss_all = Metrics.SINKHORN_LOSS.to_metric()(position_prediction.contiguous(), target_position_actions.contiguous())
        sinkhorn_loss = self._reduce_padded_loss(sinkhorn_loss_all, is_pad)

        bce_loss = 0
        if self.predict_gripper_action:
            bce_loss_all = Metrics.BINARY_CROSS_ENTROPY_WITH_LOGITS.to_metric()(
                gripper_prediction, target_gripper_actions, pos_weight=gripper_positive_class_weight, reduction='none'
            )
            bce_loss = self._reduce_padded_loss(bce_loss_all, is_pad)

        position_prediction_padded = position_prediction * ~is_pad.unsqueeze(-1)
        target_position_actions_padded = target_position_actions * ~is_pad.unsqueeze(-1)
        pred_length = torch.norm(position_prediction_padded[:, 1:] - position_prediction_padded[:, :-1], dim=-1).mean()
        target_length = torch.norm(target_position_actions_padded[:, 1:] - target_position_actions_padded[:, :-1], dim=-1).mean()
        length_loss = (pred_length - target_length) ** 2

        total_kld, _, _ = Metrics.KL_DIVERGENCE.to_metric()(mu, logvar)


        loss = (mse_loss * self.mse_weight + sinkhorn_loss * self.sinkhorn_weight + bce_loss * self.bce_weight +
                length_loss * self.length_weight + l1_loss * self.l1_weight + total_kld[0] * self.kl_weight +
                phase_ce_loss * self.phase_ce_weight + phase_entropy*self.entropy_weight)
        metrics = PhaseActEpochMetrics(mse_loss=mse_loss,
                               sinkhorn_loss=sinkhorn_loss,
                               length_loss=length_loss,
                               loss=loss,
                               binary_cross_entropy=bce_loss,
                               l1_loss=l1_loss,
                               kl_divergence=total_kld[0],
                               phase_cross_entropy=phase_ce_loss,
                               phase_entropy=phase_entropy
                               )
        metrics.add_phase_predictions(phase_probs, phase_labels.squeeze(-1), is_pad)
        return metrics



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
        position_preds_list, gripper_preds_list, _, (_, _), _, phase_probs = self.model(observation=normalized_obs)
        weights = phase_probs.unsqueeze(-1)  # (B, chunk, n_phases, 1)
        stacked_pos = torch.stack(position_preds_list, dim=2)  # (B, chunk, n_phases, 3)
        position_prediction = torch.sum(stacked_pos * weights, dim=2)  # (B, chunk, 3)
        print(f"Phase probabilities : {phase_probs}")
        if self.predict_gripper_action:
            stacked_grip = torch.stack(gripper_preds_list, dim=2)  # (B, chunk, n_phases, 1)
            gripper_prediction = torch.sum(stacked_grip * weights, dim=2)  # (B, chunk, 1)
            gripper_prediction = F.sigmoid(gripper_prediction)
            unnormalized_prediction = torch.cat([
                self.normalizer[POSITION_ACTION_KEY].unnormalize(position_prediction), gripper_prediction
            ], dim=-1)
        else:
            unnormalized_prediction = self.normalizer[POSITION_ACTION_KEY].unnormalize(position_prediction)
        return unnormalized_prediction


    def explain_predictions(self, explanation_types: list[str], obs_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:

        input_channels = {cam: self.model.backbones[cam][0].body.conv1.in_channels for cam in self.model.camera_names}

        target_action_dim = None
        target_chunk_idx = 0
        normalized_obs = self.normalizer.normalize(obs_dict)
        # Remove the observation dimension because in ACT we use a single observation horizon.
        normalized_obs = dict_apply(normalized_obs, lambda x: x.squeeze(1))


        class WeightedWrapper(nn.Module):

            def __init__(self, model):
                super().__init__()
                self.model = model


            def forward(self, observation, **kwargs):
                position_preds_list, gripper_preds_list, is_pad_hat, (mu, logvar), phase_logits, phase_probs = self.model(observation=observation, **kwargs)
                weights = phase_probs.unsqueeze(-1)
                stacked_pos = torch.stack(position_preds_list, dim=2)
                predicted_actions = torch.sum(stacked_pos * weights, dim=2)
                return predicted_actions, None, None, [None, None], None, None  # Mock single tensor


        wrapped_model = WeightedWrapper(self.model)
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
                        model=wrapped_model,
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
                        model=wrapped_model,
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
                        model=wrapped_model,
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
                        model=wrapped_model,
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
                        model=wrapped_model,
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



if __name__ == "__main__":

    # Dummy config
    class DummyConfig:
        camera_names = ['left', 'right']
        device = 'cpu'
        backbone = 'resnet18'
        position_embedding = 'sine'
        lr_backbone = 1e-5
        dilation = False
        enc_layers = 4
        dec_layers = 6
        dim_feedforward = 2048
        dropout = 0.1
        nheads = 8
        pre_norm = False
        state_dim = 0
        hidden_dim = 256
        action_dim = 4
        action_horizon = 30
        depth_fusion = None
        use_fake_proprio = False
        predict_gripper_action = True
        freeze_dformer = False
        dformer_checkpoint_path = None
        n_phases = 5
        sinkhorn_weight = 1.0
        mse_weight = 1.0
        length_weight = 1.0
        bce_weight = 1.0
        l1_weight = 1.0
        kl_weight = 1.0
        phase_ce_weight = 1.0


    config = DummyConfig()


    # Instantiate model
    model = PhaseACTPolicy(config)

    # Dummy batch (adjust shapes)
    B, C, H, W, chunk = 2, 3, 224, 224, 30
    dummy_obs_data = {'left': torch.randn(100, C, H, W), 'right': torch.randn(100, C, H, W)}  # Multiple samples for fit
    dummy_action_data = torch.randn(100, chunk, 3)
    dummy_gripper_data = torch.randn(100, chunk, 1)
    # Dummy normalizer
    normalizer = LinearNormalizer()
    normalizer.fit(dummy_obs_data)

    pos_norm = SingleFieldLinearNormalizer.create_identity()
    pos_norm.fit(dummy_action_data, last_n_dims=1)
    normalizer[POSITION_ACTION_KEY] = pos_norm

    grip_norm = SingleFieldLinearNormalizer.create_identity()
    grip_norm.fit(dummy_gripper_data, last_n_dims=1)
    normalizer[GRIPPER_ACTION_KEY] = grip_norm
    model.set_normalizer(normalizer)
    # Dummy batch
    batch = {
        OBSERVATION_KEY: {'left': torch.randn(B, C, H, W), 'right': torch.randn(B, C, H, W)},
        POSITION_ACTION_KEY: torch.randn(B, chunk, 3),
        GRIPPER_ACTION_KEY: torch.randn(B, chunk, 1),
        IS_PAD_KEY: torch.zeros(B, chunk, dtype=torch.bool),
        PHASE_LABEL_KEY: torch.randint(0, config.n_phases, (B, chunk))
    }
    # Test loss
    metrics = model.compute_loss(batch)
    print("Loss computation: OK", metrics.loss.item())

    # Test predict
    obs_dict = {'left': torch.randn(B, C, H, W), 'right': torch.randn(B, C, H, W)}
    action = model.predict_action(obs_dict, torch.device('cpu'))
    print("Prediction shape:", action.shape)  # Expect (B, chunk, 4) with gripper
    explanation_types = [ExplanationType.GRADCAM.value]  # Example type
    explanations = model.explain_predictions(explanation_types, obs_dict)
    print("Explanation computation: OK", list(explanations.keys()))
    print("Test passed!")
