"""
DETRVAE: A Variational Autoencoder based on DETR for imitation learning.
# Processes images (multi-camera + optional depth), robot state (qpos), and predicts action sequences.
# Uses VAE for latent representation of actions during training.
"""
from typing import Dict, Optional, List

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from legacy_constants import ROBOT_STATE_KEY, Cameras, DepthFusionStrategy
from model.common.dformer_depth_encoder import DFormerDepthEncoder
from model.common.spatial_softmax import SpatialSoftmax2d
from model.detr.backbone import build_backbones
from model.detr.position_encoding import build_position_encoding
from model.detr.transformer import build_transformer, TransformerEncoderLayer, TransformerEncoder
import torch.nn.functional as F


# Helper functions
def reparametrize(mu, logvar):
    """Reparametrization trick for VAE: sample from N(mu, var)."""
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    """Generate sinusoidal position encodings for transformer inputs."""


    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]


    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class PhaseDETRVAE(nn.Module):

    def __init__(
            self,
            camera_names,  # List of camera names (e.g., ['left', 'right', 'depth']).
            device,  # Device to run the model on (e.g., 'cuda' or 'cpu').
            backbone,  # Backbone type (e.g., 'resnet18').
            position_embedding,  # Positional encoding type.
            lr_backbone,  # Learning rate for backbone.
            dilation,  # Dilation in backbone.
            enc_layers = 4,  # Encoder layers.
            dec_layers = 6,  # Decoder layers.
            dim_feedforward = 2048,  # FFN dim.
            dropout = 0.1,  # Dropout rate.
            nheads = 8,  # Attention heads.
            pre_norm = False,  # Pre-normalization.
            obs_dimension = 3,  # Robot state dim (qpos).
            hidden_dimension = 256,  # Model hidden dim.
            action_dimension = 3,  # Action dim (e.g., action_embedding,y,z).
            chunk_size = 30,  # Num actions to predict (queries).
            depth_fusion_strategy = None,  # Strategy for depth fusion (if applicable).
            use_fake_proprio = False,  # Use identity as proprio input (for testing without robot state).
            predict_gripper_action = False,  # Whether to predict gripper action.
            freeze_dformer = False,  # Whether to freeze Dformer model or fine-tune last layer, used only if depth fusion is geometric attention.
            dformer_checkpoint_path = "my_pretrained_dformer.pth ",
            n_phases: int = 5,  # Number of phases for multi-phase head
            phase_learnable_temperature: float = 100.0 ,
    ):
        super().__init__()
        self.use_fake_proprio = use_fake_proprio
        if use_fake_proprio:
            obs_dimension = 4
        self.num_queries = chunk_size  # Actions per prediction.
        self.camera_names = camera_names
        self.use_robot_state = obs_dimension > 0  # Use robot state?
        self.hidden_dim = hidden_dimension
        self.action_dim = action_dimension
        self.device = torch.device(device)
        self.predict_gripper_action = predict_gripper_action
        self.n_phases = n_phases
        # Build backbones for each camera (CNN feature extractors).
        self.backbones = nn.ModuleDict(build_backbones(
            camera_names=camera_names,
            backbone=backbone,
            hidden_dim=hidden_dimension,
            position_embedding=position_embedding,
            lr_backbone=lr_backbone,
            dilation=dilation
        ))

        self.phase_head = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension // 2, n_phases)
        ) # Phase classification head

        self.phase_temperature = nn.Parameter(torch.tensor(phase_learnable_temperature, dtype=torch.float32), requires_grad=True)

        # Transformer (encoder + decoder).
        self.transformer = build_transformer(
            enc_layers=enc_layers, dec_layers=dec_layers,
            dim_feedforward=dim_feedforward, hidden_dim=hidden_dimension,
            dropout=dropout, nheads=nheads, pre_norm=pre_norm
        )

        # Separate encoder for VAE latent (actions -> latent).
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dimension, nhead=nheads,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation="relu", normalize_before=pre_norm
        )
        self.vae_encoder = TransformerEncoder(
            encoder_layer, num_layers=enc_layers,
            norm=nn.LayerNorm(hidden_dimension) if pre_norm else None
        )

        # Heads: predict actions and padding.
        position_action_dimension = action_dimension - 1 if predict_gripper_action else action_dimension

        self.position_action_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dimension),
                nn.Linear(hidden_dimension, hidden_dimension // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dimension // 2, 3),
            ) for _ in range(n_phases)
        ])

        self.is_pad_head = nn.Linear(hidden_dimension, 1)  # Binary pad prediction.

        if self.predict_gripper_action:
            self.gripper_action_heads = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(hidden_dimension),
                    nn.Linear(hidden_dimension, hidden_dimension // 4),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dimension // 4, 1),
                ) for _ in range(n_phases)
            ])

        # Query embeddings for decoder.
        self.query_embed = nn.Embedding(chunk_size, hidden_dimension)

        # Project image features to hidden dim (assumes same for all cams).
        self.image_feature_proj = nn.Conv2d(
            self.backbones[camera_names[0]].num_channels,  # Use first cam's channels.
            hidden_dimension, kernel_size=1
        )
        if Cameras.DEPTH.value in camera_names:
            self.depth_fusion_strategy = depth_fusion_strategy
            # Project fusion of left+depth to hidden dim.
            self.fusion_proj = nn.Conv2d(hidden_dimension * 2, hidden_dimension, kernel_size=1)

            if depth_fusion_strategy == DepthFusionStrategy.ATTENTION.value:
                self.attention_fusion = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=hidden_dimension, nhead=8, batch_first=True),
                    num_layers=1
                )
                self.spatial_pool = SpatialSoftmax2d(normalize=True, temperature=1.0, learnable_temperature=False)
                self.view_fc = nn.Linear(hidden_dimension * 2, hidden_dimension)
                self.position_encoding = build_position_encoding(hidden_dim=hidden_dimension, position_embedding=position_embedding)
            elif depth_fusion_strategy == DepthFusionStrategy.GEOMETRIC_ATTENTION.value:
                self.dformer_encoder = DFormerDepthEncoder(
                    variant='S',
                    out_dim=512,  # Default
                    pretrained=dformer_checkpoint_path,
                )
                self.dformer_encoder.requires_grad_(False)
                if not freeze_dformer:
                    for param in self.dformer_encoder.backbone.layers[-1].parameters():
                        param.requires_grad = True  # Unfreeze last layer
                self.dformer_pos = build_position_encoding(hidden_dim=hidden_dimension, position_embedding=position_embedding)
                self.dformer_proj = nn.Conv2d(self.dformer_encoder.backbone.num_features, hidden_dimension, kernel_size=1)

        if self.use_robot_state:
            # Project robot state to hidden dim.
            self.input_proj_robot_state = nn.Linear(obs_dimension, hidden_dimension)

        # VAE-specific:
        self.latent_dim = 32  # Latent space size.
        self.class_token_embedding = nn.Embedding(1, hidden_dimension)  # CLS token for encoder.
        self.vae_position_action_proj = nn.Linear(position_action_dimension, hidden_dimension)  # Actions to embed.
        if self.predict_gripper_action:
            self.vae_gripper_action_proj = nn.Linear(1, hidden_dimension)  # Gripper action to embed.
            self.encoder_action_proj_resized = nn.Linear(hidden_dimension * 2, hidden_dimension)
        self.vae_robot_state_proj = nn.Linear(obs_dimension, hidden_dimension) if self.use_robot_state else None  # qpos to embed.
        self.latent_proj = nn.Linear(hidden_dimension, self.latent_dim * 2)  # To mu/logvar.
        # Pos encodings for encoder input: CLS + [qpos] + actions.
        extra_slots = 1 + (1 if self.use_robot_state else 0) + chunk_size
        self.register_buffer('pos_table', get_sinusoid_encoding_table(extra_slots, hidden_dimension))

        # Decoder extras: latent to embed, pos for proprio/latent.
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dimension)
        self.additional_pos_embed = nn.Embedding(2, hidden_dimension)  # For proprio and latent.


    def forward(self, observation: Dict[str, torch.tensor],
                position_actions: Optional[torch.tensor] = None,
                gripper_actions: Optional[torch.tensor] = None,
                is_pad: Optional[torch.tensor] = None
                ):
        """Forward pass of the DETRVAE model.

        Args:
            observation (Dict[str, torch.tensor]): Dictionary containing:
                - 'robot_state': (B, obs_dim) Robot state (qpos) if used.
                - 'left': (B, C, H, W) Left camera RGB image.
                - 'right': (B, C, H, W) Right camera RGB image.
                - 'depth_image': (B, 1, H, W) Depth image if used.
            actions (Optional[torch.tensor]): (B, chunk_size, action_dim) Actions for training.
            is_pad (Optional[torch.tensor]): (B, chunk_size) Padding mask for actions.
        """
        if Cameras.LEFT.value in observation.keys():
            batch_size = observation[Cameras.LEFT.value].shape[0]
        elif Cameras.RIGHT.value in observation.keys():
            batch_size = observation[Cameras.RIGHT.value].shape[0]
        else:
            raise ValueError("No valid camera observations found in input.")
        if self.use_robot_state:
            if self.use_fake_proprio:
                robot_state = torch.tensor([1., 0., 0., 0.], device=self.device).repeat(batch_size, 1)
            else:
                robot_state = observation[ROBOT_STATE_KEY]
        else:
            robot_state = None

        is_training = position_actions is not None or gripper_actions is not None

        # VAE Encoding (only in training): Actions (+ robot state) -> latent z.
        if is_training:
            # Embed actions, add CLS token.
            action_embeddings = self.vae_position_action_proj(position_actions)  # (B, seq, hidden)
            if self.predict_gripper_action:
                gripper_action_embeddings = self.vae_gripper_action_proj(gripper_actions)  # (B, seq, hidden)
                action_embeddings = torch.cat([action_embeddings, gripper_action_embeddings], axis=-1)
                action_embeddings = self.encoder_action_proj_resized(action_embeddings)

            class_token_embedding = self.class_token_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 1, hidden)

            if self.use_robot_state:
                robot_state_embedding = self.vae_robot_state_proj(robot_state).unsqueeze(1)  # (B, 1, hidden)
                encoder_input = torch.cat([class_token_embedding, robot_state_embedding, action_embeddings], dim=1)  # (B, 1+1+seq, hidden)
                non_action_pad_mask = torch.full((batch_size, 2), False, device=self.device)  # No pad for class token+robot state.
            else:
                encoder_input = torch.cat([class_token_embedding, action_embeddings], dim=1)  # (B, 1+seq, hidden)
                non_action_pad_mask = torch.full((batch_size, 1), False, device=self.device)

            # Transpose for transformer: (seq+1/2, B, hidden)
            encoder_input = encoder_input.permute(1, 0, 2)
            encoder_pad_mask = torch.cat([non_action_pad_mask, is_pad], dim=1)  # (B, seq+)

            # Add pos encodings.
            encoder_positional_embedding = self.pos_table.clone().permute(1, 0, 2)  # (seq+, 1, hidden)

            # Run encoder, take CLS output.
            encoder_output = self.vae_encoder(encoder_input,
                                              pos=encoder_positional_embedding,
                                              src_key_padding_mask=encoder_pad_mask)[0]

            # Get mu/logvar, sample latent.
            latent_statistics = self.latent_proj(encoder_output)
            mu, logvar = latent_statistics.chunk(2, dim=1)
            z_sample = reparametrize(mu, logvar)
            latent_embedding = self.latent_out_proj(z_sample)
        else:
            # Inference: zero latent.
            mu = logvar = None
            z_sample = torch.zeros(batch_size, self.latent_dim, device=self.device)
            latent_embedding = self.latent_out_proj(z_sample)

        # Process images: Extract features per camera.
        image_features, all_pos_embeddings = {}, {}
        for cam in self.camera_names:
            image = observation[cam]
            features, pos_embedding = self.backbones[cam](image)
            image_features[cam] = self.image_feature_proj(features[0])  # Get last layer and project to hidden dimension.
            all_pos_embeddings[cam] = pos_embedding[0]

        # proprioception features
        if self.use_robot_state:
            proprio_input = self.input_proj_robot_state(robot_state)
        else:
            proprio_input = None

        has_depth = Cameras.DEPTH.value in self.camera_names
        if has_depth:
            if self.depth_fusion_strategy == DepthFusionStrategy.ATTENTION.value:
                view_feats = []
                for cam in self.camera_names:
                    feat_map = image_features[cam]
                    pooled = self.spatial_pool(feat_map)
                    feat = self.view_fc(pooled)
                    view_feats.append(feat.unsqueeze(1))  # (B, 1, hidden)
                stacked = torch.cat(view_feats, dim=1)  # (B, num_views, hidden)
                fused = self.attention_fusion(stacked)
                fused_feat = fused.mean(dim=1)  # (B, hidden)
                fused_image_features = fused_feat.unsqueeze(2).unsqueeze(3)  # (B, hidden, 1, 1)
                fused_positional_embeddings = self.position_encoding(fused_image_features)

            elif self.depth_fusion_strategy == DepthFusionStrategy.LEFT_CHANNEL_WISE.value:
                fused_left_features = torch.cat([image_features[Cameras.LEFT.value], image_features[Cameras.DEPTH.value]], dim=1)
                fused_left_features = self.fusion_proj(fused_left_features)
                fused_image_features = torch.cat([fused_left_features, image_features[Cameras.RIGHT.value]], dim=3)
                fused_positional_embeddings = torch.cat([all_pos_embeddings[Cameras.LEFT.value], all_pos_embeddings[Cameras.RIGHT.value]], dim=3)
            elif self.depth_fusion_strategy == DepthFusionStrategy.SEPARATE.value:
                sequences = []
                pos_sequences = []
                for cam in self.camera_names:
                    feat = image_features[cam].flatten(2).transpose(1, 2)  # (B, H*W, hidden)
                    pos = all_pos_embeddings[cam].flatten(2).transpose(1, 2)
                    sequences.append(feat)
                    pos_sequences.append(pos)
                fused_image_features = torch.cat(sequences, dim=1)  # (B, total_tokens, hidden)
                fused_positional_embeddings = torch.cat(pos_sequences, dim=1)
            elif self.depth_fusion_strategy == DepthFusionStrategy.GEOMETRIC_ATTENTION.value:
                image_features = {}
                all_pos_embeddings = {}
                other_cams = [c for c in self.camera_names if c not in [Cameras.LEFT.value, Cameras.DEPTH.value]]
                for cam in other_cams:
                    image = observation[cam]
                    features, pos_embedding = self.backbones[cam](image)
                    image_features[cam] = self.image_feature_proj(features[0])
                    all_pos_embeddings[cam] = pos_embedding[0]
                rgb_left = observation[Cameras.LEFT.value]
                depth = observation[Cameras.DEPTH.value]

                # Fuse left RGB and depth using DFormerDepthEncoder, returning feature map
                fused_map = self.dformer_encoder(rgb_left, depth, return_feature_map=True)  # (B, C, Hs, Ws)
                fused_map = self.dformer_proj(fused_map)  # Project to hidden_dim
                fused_pos = self.dformer_pos(fused_map)  # Positional encoding for fused map
                # Collect features and positions from other cameras
                other_features = [image_features[cam] for cam in other_cams]
                other_pos = [all_pos_embeddings[cam] for cam in other_cams]

                # Concatenate fused left-depth map with other cameras along width dimension
                fused_image_features = torch.cat([fused_map] + other_features, dim=3)
                fused_positional_embeddings = torch.cat([fused_pos] + other_pos, dim=3)
            else:
                # Concat features/pos across cameras (fold into width dim).
                fused_image_features = torch.cat(list(image_features.values()), dim=3)  # (B, hidden, H, W_total)
                fused_positional_embeddings = torch.cat(list(all_pos_embeddings.values()), dim=3)
        else:
            # No depth: just concat across cameras.
            fused_image_features = torch.cat(list(image_features.values()), dim=3)  # (B, hidden, H, W_total)
            fused_positional_embeddings = torch.cat(list(all_pos_embeddings.values()), dim=3)

        if fused_image_features.dim() == 3:
            fused_positional_embeddings = fused_positional_embeddings.squeeze(0)

        decoder_output = self.transformer(input_features=fused_image_features,
                                          attention_mask=None,  # No mask for now.
                                          query_embedding=self.query_embed.weight,
                                          positional_embedding=fused_positional_embeddings,
                                          latent_input=latent_embedding,
                                          proprio_input=proprio_input,
                                          additional_positional_embedding=self.additional_pos_embed.weight)[0]  # (B, num_queries, hidden)

        # Phase prediction: pool decoder_output to get per-batch features
        phase_logits = self.phase_head(decoder_output) / self.phase_temperature  # (B, num_queries, n_phases)
        phase_probs = F.softmax(phase_logits, dim=-1)  # Apply temperature for learning the phase prediction confidence
        predicted_positions = [head(decoder_output) for head in self.position_action_heads]  # List[n_phases] of (B, num_queries, 3)
        predicted_gripper_actions = None
        if self.predict_gripper_action:
            predicted_gripper_actions = [head(decoder_output) for head in self.gripper_action_heads]  # List[n_phases] of (B, num_queries, 1)
        predicted_pad_mask = self.is_pad_head(decoder_output)
        return predicted_positions, predicted_gripper_actions, predicted_pad_mask, [mu, logvar], phase_logits, phase_probs

    def target_layers_getter(self, camera: str) -> List[nn.Module]:
        """Get target layers for Grad-CAM style heatmaps based on camera name."""
        return [self.backbones[camera][0].body['layer4'][-1]]



