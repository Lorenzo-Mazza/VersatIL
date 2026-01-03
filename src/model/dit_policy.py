# Copyright (c) Sudeep Dasari, 2023, modified by Lorenzo Mazza, 2025.
# Heavy inspiration taken from DETR by Meta AI (Carion et. al.): https://github.com/facebookresearch/detr
# and DiT by Meta AI (Peebles and Xie): https://github.com/facebookresearch/DiT

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from typing import Callable, List, Optional, Dict, Union


def get_activation_function(activation_name: str) -> Callable:
    """Returns an activation function given its name.

    Args:
      activation_name: Name of the activation function (e.g., "relu", "gelu").

    Returns:
      The corresponding activation function.

    Raises:
      RuntimeError: If the activation name is not supported.
    """
    if activation_name == "relu":
        return F.relu
    if activation_name == "gelu":
        return nn.GELU(approximate="tanh")
    if activation_name == "glu":
        return F.glu
    raise RuntimeError(f"Activation should be relu/gelu/glu, not {activation_name}.")


def add_positional_embedding(
    tensor: torch.Tensor, positional_embedding: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Adds positional embedding to the tensor if provided.

    Args:
      tensor: Input tensor.
      positional_embedding: Positional embedding tensor to add (optional).

    Returns:
      Tensor with positional embedding added if provided, otherwise the original tensor.
    """
    return tensor if positional_embedding is None else tensor + positional_embedding


class PositionalEncoding(nn.Module):
    """Implements sinusoidal positional encoding as described in the Transformer paper.

    This module adds positional information to input sequences to help the model
    understand the order of elements.
    """

    def __init__(self, embedding_dim: int, max_sequence_length: int = 5000) -> None:
        """Initializes the PositionalEncoding module.

        Args:
          embedding_dim: Dimensionality of the embeddings.
          max_sequence_length: Maximum length of the input sequence.
        """
        super().__init__()
        # Precompute positional encodings in log space for efficiency.
        positional_encodings = torch.zeros(max_sequence_length, embedding_dim)
        positions = torch.arange(0, max_sequence_length, dtype=torch.float).unsqueeze(1)
        div_terms = torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float)
            * -(math.log(10000.0) / embedding_dim)
        )
        positional_encodings[:, 0::2] = torch.sin(positions * div_terms)
        positional_encodings[:, 1::2] = torch.cos(positions * div_terms)
        positional_encodings = positional_encodings.unsqueeze(0).transpose(0, 1)
        self.register_buffer("positional_encodings", positional_encodings)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Adds positional encodings to the input tensor.

        Args:
          input_tensor: Input tensor of shape (sequence_length, batch_size, embedding_dim).

        Returns:
          Input tensor with positional encodings added.
        """
        positional_encodings_slice = self.positional_encodings[: input_tensor.shape[0]]
        positional_encodings_slice = positional_encodings_slice.repeat(
            (1, input_tensor.shape[1], 1)
        )
        return positional_encodings_slice.detach().clone()


class TimestepEmbeddingNetwork(nn.Module):
    """Embeds timesteps using sinusoidal encodings followed by an MLP.

    This is used to condition the model on diffusion timesteps.
    """

    def __init__(
        self,
        timestep_embedding_dim: int,
        output_dim: int,
        learnable_frequencies: bool = False,
    ) -> None:
        """Initializes the TimestepEmbeddingNetwork.

        Args:
          timestep_embedding_dim: Dimensionality of the sinusoidal embedding (must be even).
          output_dim: Output dimensionality after the MLP.
          learnable_frequencies: Whether the frequency parameters are learnable.
        """
        assert timestep_embedding_dim % 2 == 0, "timestep_embedding_dim must be even!"
        super().__init__()
        half_dim = timestep_embedding_dim // 2

        w = np.log(10000) / (half_dim - 1)
        frequencies = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter(
            "frequencies",
            nn.Parameter(frequencies, requires_grad=learnable_frequencies),
        )
        self.output_network = nn.Sequential(
            nn.Linear(timestep_embedding_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embeds the timesteps.

        Args:
          timesteps: 1D tensor of timesteps (batch_size,).

        Returns:
          Embedded timesteps of shape (batch_size, output_dim).
        """
        assert len(timesteps.shape) == 1, "Assumes 1D input timestep array."
        # Compute position * frequency for sinusoidal inputs.
        scaled_timesteps = timesteps[:, None] * self.frequencies[None]
        sinusoidal_embeddings = torch.cat(
            (torch.cos(scaled_timesteps), torch.sin(scaled_timesteps)), dim=1
        )
        return self.output_network(sinusoidal_embeddings)


class SelfAttentionEncoderBlock(nn.Module):
    """A single self-attention encoder block similar to Transformer encoder layers."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 8,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        activation_name: str = "gelu",
    ) -> None:
        """Initializes the SelfAttentionEncoderBlock.

        Args:
          embedding_dim: Dimensionality of the embeddings.
          num_heads: Number of attention heads.
          feedforward_dim: Hidden dimensionality in the feedforward network.
          dropout: Dropout rate.
          activation_name: Name of the activation function.
        """
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout
        )
        # Feedforward subnetwork.
        self.linear1 = nn.Linear(embedding_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, embedding_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_function(activation_name)

    def forward(
        self, source_tensor: torch.Tensor, positional_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Applies the self-attention encoder block.

        Args:
          source_tensor: Input tensor (sequence_length, batch_size, embedding_dim).
          positional_embedding: Positional embedding to add.

        Returns:
          Processed tensor.
        """
        # Add positional embedding to queries and keys for self-attention.
        query = key = add_positional_embedding(source_tensor, positional_embedding)
        attended_tensor, _ = self.self_attention(
            query, key, value=source_tensor, need_weights=False
        )
        # Residual connection and layer norm after attention.
        source_tensor = source_tensor + self.dropout1(attended_tensor)
        source_tensor = self.norm1(source_tensor)
        # Feedforward subnetwork with residual and norm.
        feedforward_tensor = self.linear2(
            self.dropout2(self.activation(self.linear1(source_tensor)))
        )
        source_tensor = source_tensor + self.dropout3(feedforward_tensor)
        source_tensor = self.norm2(source_tensor)
        return source_tensor

    def reset_parameters(self) -> None:
        """Resets parameters using Xavier uniform initialization."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)


class ShiftScaleModulation(nn.Module):
    """Applies shift and scale modulation conditioned on an input vector.

    This is similar to adaptive layer normalization but with shift and scale.
    """

    def __init__(self, modulation_dim: int) -> None:
        """Initializes the ShiftScaleModulation.

        Args:
          modulation_dim: Dimensionality of the input and output.
        """
        super().__init__()
        self.activation = nn.SiLU()
        self.scale_linear = nn.Linear(modulation_dim, modulation_dim)
        self.shift_linear = nn.Linear(modulation_dim, modulation_dim)

    def forward(
        self, input_tensor: torch.Tensor, conditioning_vector: torch.Tensor
    ) -> torch.Tensor:
        """Applies modulation.

        Args:
          input_tensor: Tensor to modulate (sequence_length, batch_size, dim).
          conditioning_vector: Conditioning vector (batch_size, dim).

        Returns:
          Modulated tensor.
        """
        activated_conditioning = self.activation(conditioning_vector)
        return (
            input_tensor * self.scale_linear(activated_conditioning)[None]
            + self.shift_linear(activated_conditioning)[None]
        )

    def reset_parameters(self) -> None:
        """Resets parameters with Xavier init and zero bias."""
        nn.init.xavier_uniform_(self.scale_linear.weight)
        nn.init.xavier_uniform_(self.shift_linear.weight)
        nn.init.zeros_(self.scale_linear.bias)
        nn.init.zeros_(self.shift_linear.bias)


class ZeroScaleModulation(nn.Module):
    """Applies scale modulation (starting from zero) conditioned on an input vector."""

    def __init__(self, modulation_dim: int) -> None:
        """Initializes the ZeroScaleModulation.

        Args:
          modulation_dim: Dimensionality of the input and output.
        """
        super().__init__()
        self.activation = nn.SiLU()
        self.scale_linear = nn.Linear(modulation_dim, modulation_dim)

    def forward(
        self, input_tensor: torch.Tensor, conditioning_vector: torch.Tensor
    ) -> torch.Tensor:
        """Applies modulation.

        Args:
          input_tensor: Tensor to modulate (sequence_length, batch_size, dim).
          conditioning_vector: Conditioning vector (batch_size, dim).

        Returns:
          Modulated tensor.
        """
        activated_conditioning = self.activation(conditioning_vector)
        return input_tensor * self.scale_linear(activated_conditioning)[None]

    def reset_parameters(self) -> None:
        """Resets parameters to zeros."""
        nn.init.zeros_(self.scale_linear.weight)
        nn.init.zeros_(self.scale_linear.bias)


class DiTDecoderBlock(nn.Module):
    """A decoder block with self-attention and modulation, inspired by DiT.

    This block applies self-attention and feedforward layers with conditional modulation
    from timestep and encoder outputs.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        activation_name: str = "gelu",
    ) -> None:
        """Initializes the DiTDecoderBlock.

        Args:
          embedding_dim: Dimensionality of the embeddings.
          num_heads: Number of attention heads.
          feedforward_dim: Hidden dimensionality in the feedforward network.
          dropout: Dropout rate.
          activation_name: Name of the activation function.
        """
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout
        )
        # Feedforward subnetwork.
        self.linear1 = nn.Linear(embedding_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, embedding_dim)

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = get_activation_function(activation_name)

        # Modulation layers for attention and MLP, conditioned on timestep + condition mean.
        self.attention_mod1 = ShiftScaleModulation(embedding_dim)
        self.attention_mod2 = ZeroScaleModulation(embedding_dim)
        self.mlp_mod1 = ShiftScaleModulation(embedding_dim)
        self.mlp_mod2 = ZeroScaleModulation(embedding_dim)

    def forward(
        self,
        input_tensor: torch.Tensor,
        timestep_embedding: torch.Tensor,
        condition_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Applies the decoder block.

        Args:
          input_tensor: Input tensor (sequence_length, batch_size, embedding_dim).
          timestep_embedding: Timestep embedding (batch_size, embedding_dim).
          condition_tensor: Condition tensor from encoder (sequence_length, batch_size, embedding_dim).

        Returns:
          Processed tensor.
        """
        # Combine condition by taking mean over sequence length and adding to timestep.
        condition_mean = torch.mean(condition_tensor, dim=0)
        combined_condition = condition_mean + timestep_embedding

        # Attention branch with modulation.
        normalized_input = self.attention_mod1(
            self.norm1(input_tensor), combined_condition
        )
        attended_tensor, _ = self.self_attention(
            normalized_input, normalized_input, normalized_input, need_weights=False
        )
        input_tensor = (
            self.attention_mod2(self.dropout1(attended_tensor), combined_condition)
            + input_tensor
        )

        # MLP branch with modulation.
        normalized_input = self.mlp_mod1(self.norm2(input_tensor), combined_condition)
        feedforward_tensor = self.linear2(
            self.dropout2(self.activation(self.linear1(normalized_input)))
        )
        feedforward_tensor = self.mlp_mod2(
            self.dropout3(feedforward_tensor), combined_condition
        )
        return input_tensor + feedforward_tensor

    def reset_parameters(self) -> None:
        """Resets parameters using Xavier uniform initialization."""
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

        for mod in (
            self.attention_mod1,
            self.attention_mod2,
            self.mlp_mod1,
            self.mlp_mod2,
        ):
            mod.reset_parameters()


class FinalPredictionLayer(nn.Module):
    """Final layer that predicts noise (epsilon) with adaptive LN modulation.

    This layer normalizes, modulates with condition, and projects to output dim.
    """

    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        """Initializes the FinalPredictionLayer.

        Args:
          hidden_dim: Input hidden dimensionality.
          output_dim: Output dimensionality (action_dim).
        """
        super().__init__()
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_linear = nn.Linear(hidden_dim, output_dim, bias=True)
        self.adaptive_ln_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        timestep_embedding: torch.Tensor,
        condition_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts the output.

        Args:
          input_tensor: Input tensor (sequence_length, batch_size, hidden_dim).
          timestep_embedding: Timestep embedding (batch_size, hidden_dim).
          condition_tensor: Condition tensor (sequence_length, batch_size, hidden_dim).

        Returns:
          Predicted tensor (batch_size, sequence_length, output_dim).
        """
        # Combine condition by mean over sequence and add to timestep.
        condition_mean = torch.mean(condition_tensor, dim=0)
        combined_condition = condition_mean + timestep_embedding

        # Compute shift and scale for modulation.
        shift, scale = self.adaptive_ln_modulation(combined_condition).chunk(2, dim=1)
        # Apply modulation after norm (though norm is before in code, but it's affine=False).
        input_tensor = input_tensor * scale[None] + shift[None]
        output = self.output_linear(input_tensor)
        return output.transpose(0, 1)  # To (batch_size, sequence_length, output_dim)

    def reset_parameters(self) -> None:
        """Resets parameters to zeros."""
        for param in self.parameters():
            nn.init.zeros_(param)


class TransformerEncoder(nn.Module):
    """Stack of self-attention encoder blocks."""

    def __init__(self, base_block: nn.Module, num_layers: int) -> None:
        """Initializes the TransformerEncoder.

        Args:
          base_block: The base SelfAttentionEncoderBlock to copy.
          num_layers: Number of layers.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(base_block) for _ in range(num_layers)]
        )

        for layer in self.layers:
            layer.reset_parameters()

    def forward(
        self, source_tensor: torch.Tensor, positional_embedding: torch.Tensor
    ) -> List[torch.Tensor]:
        """Applies the encoder layers.

        Args:
          source_tensor: Input tensor (sequence_length, batch_size, embedding_dim).
          positional_embedding: Positional embedding.

        Returns:
          List of outputs from each layer.
        """
        current_tensor = source_tensor
        layer_outputs = []
        for layer in self.layers:
            current_tensor = layer(current_tensor, positional_embedding)
            layer_outputs.append(current_tensor)
        return layer_outputs


class TransformerDecoder(TransformerEncoder):
    """Stack of DiT decoder blocks.

    Inherits from TransformerEncoder but overrides forward for decoder logic.
    """

    def forward(
        self,
        source_tensor: torch.Tensor,
        timestep_embedding: torch.Tensor,
        all_condition_tensors: List[torch.Tensor],
    ) -> torch.Tensor:
        """Applies the decoder layers.

        Args:
          source_tensor: Input tensor (sequence_length, batch_size, embedding_dim).
          timestep_embedding: Timestep embedding (batch_size, embedding_dim).
          all_condition_tensors: List of condition tensors from encoder layers.

        Returns:
          Final output tensor after all layers.
        """
        current_tensor = source_tensor
        for layer, condition in zip(self.layers, all_condition_tensors):
            current_tensor = layer(current_tensor, timestep_embedding, condition)
        return current_tensor


class DiTNoisePredictionNetwork(nn.Module):
    """The core noise prediction network using encoder-decoder Transformer with DiT modulation.

    This network predicts noise in the diffusion process conditioned on observations.
    """

    def __init__(
        self,
        action_dim: int,
        prediction_horizon: int,
        timestep_embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_blocks: int = 6,
        dropout: float = 0.1,
        feedforward_dim: int = 2048,
        num_heads: int = 8,
        activation_name: str = "gelu",
    ) -> None:
        """Initializes the DiTNoisePredictionNetwork.

        Args:
          action_dim: Dimensionality of actions.
          prediction_horizon: Length of action sequence to predict.
          timestep_embedding_dim: Dim for timestep sinusoidal embedding.
          hidden_dim: Main hidden dimensionality.
          num_blocks: Number of encoder/decoder blocks.
          dropout: Dropout rate.
          feedforward_dim: Feedforward hidden dim.
          num_heads: Number of attention heads.
          activation_name: Activation function name.
        """
        super().__init__()

        # Positional encodings for encoder and decoder.
        self.encoder_positional_encoding = PositionalEncoding(hidden_dim)
        self.register_parameter(
            "decoder_positional_encoding",
            nn.Parameter(
                torch.empty(prediction_horizon, 1, hidden_dim), requires_grad=True
            ),
        )
        nn.init.xavier_uniform_(self.decoder_positional_encoding.data)

        # Projections for inputs.
        self.timestep_embedding_network = TimestepEmbeddingNetwork(
            timestep_embedding_dim, hidden_dim
        )
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(action_dim, hidden_dim),
        )

        # Encoder: stack of self-attention blocks for observation tokens.
        encoder_base_block = SelfAttentionEncoderBlock(
            hidden_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            activation_name=activation_name,
        )
        self.encoder = TransformerEncoder(encoder_base_block, num_blocks)

        # Decoder: stack of modulated self-attention blocks for action tokens.
        decoder_base_block = DiTDecoderBlock(
            hidden_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            activation_name=activation_name,
        )
        self.decoder = TransformerDecoder(decoder_base_block, num_blocks)

        # Final layer to predict noise (epsilon).
        self.noise_output_layer = FinalPredictionLayer(hidden_dim, action_dim)

        print(
            "Number of diffusion parameters: {:e}".format(
                sum(p.numel() for p in self.parameters())
            )
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        observation_encoding: torch.Tensor,
        encoder_cache: Optional[List[torch.Tensor]] = None,
    ) -> tuple[List[torch.Tensor], torch.Tensor]:
        """Predicts noise given noisy actions, timesteps, and observation encoding.

        Args:
          noisy_actions: Noisy action tensor (batch_size, prediction_horizon, action_dim).
          timesteps: Diffusion timesteps (batch_size,).
          observation_encoding: Encoded observations (batch_size, num_tokens, hidden_dim).
          encoder_cache: Precomputed encoder outputs (optional).

        Returns:
          encoder_cache: Encoder layer outputs.
          noise_prediction: Predicted noise (batch_size, prediction_horizon, action_dim).
        """
        if encoder_cache is None:
            encoder_cache = self.forward_encoder(observation_encoding)
        return encoder_cache, self.forward_decoder(
            noisy_actions, timesteps, encoder_cache
        )

    def forward_encoder(self, observation_encoding: torch.Tensor) -> List[torch.Tensor]:
        """Encodes the observation tokens.

        Args:
          observation_encoding: (batch_size, num_tokens, hidden_dim).

        Returns:
          List of encoder layer outputs, each (num_tokens, batch_size, hidden_dim).
        """
        # Transpose to (sequence_length, batch_size, embedding_dim) for MultiheadAttention.
        observation_encoding = observation_encoding.transpose(0, 1)
        positional_embedding = self.encoder_positional_encoding(observation_encoding)
        encoder_cache = self.encoder(observation_encoding, positional_embedding)
        return encoder_cache

    def forward_decoder(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_cache: List[torch.Tensor],
    ) -> torch.Tensor:
        """Decodes to predict noise using action tokens and encoder cache.

        Args:
          noisy_actions: (batch_size, prediction_horizon, action_dim).
          timesteps: (batch_size,).
          encoder_cache: List of encoder outputs.

        Returns:
          Predicted noise (batch_size, prediction_horizon, action_dim).
        """
        timestep_embedding = self.timestep_embedding_network(timesteps)

        # Project noisy actions to tokens and add positional encoding.
        action_tokens = self.action_projection(noisy_actions)
        action_tokens = action_tokens.transpose(
            0, 1
        )  # To (prediction_horizon, batch_size, hidden_dim)
        decoder_input = action_tokens + self.decoder_positional_encoding

        # Apply decoder blocks conditioned on encoder layers.
        decoder_output = self.decoder(decoder_input, timestep_embedding, encoder_cache)

        # Predict noise epsilon.
        return self.noise_output_layer(
            decoder_output, timestep_embedding, encoder_cache[-1]
        )


class BatchNorm1DHelper(nn.BatchNorm1d):
    """Helper wrapper for BatchNorm1d to handle 3D inputs by transposing."""

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Applies batch norm.

        Args:
          input_tensor: Input tensor (batch_size, ..., channels) or 3D.

        Returns:
          Normalized tensor.
        """
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.transpose(1, 2)
            output = super().forward(input_tensor)
            return output.transpose(1, 2)
        return super().forward(input_tensor)


class DiTPolicy(nn.Module):
    """Diffusion Transformer Policy for action prediction using vision and optional proprioception.

    This model uses a diffusion process to generate action sequences conditioned on images and obs.
    """

    def __init__(
        self,
        visual_feature_extractor: nn.Module,
        observation_dim: int,
        num_cameras: int,
        use_proprioception: Union[str, bool],
        action_dim: int,
        prediction_horizon: int,
        training_diffusion_steps: int,
        evaluation_diffusion_steps: int,
        images_per_camera: int = 1,
        dropout: float = 0.0,
        share_camera_features: bool = False,
        early_fusion: bool = False,
        feature_normalization: Optional[str] = None,
        token_dim: int = 512,
    ) -> None:
        """Initializes the DiTPolicy.

        Args:
          visual_feature_extractor: Module to extract features from images.
          observation_dim: Dimensionality of proprioceptive observations.
          num_cameras: Number of camera views.
          use_proprioception: Strategy to incorporate proprio ("add_token", "pad_img_tokens", or False).
          action_dim: Dimensionality of actions.
          prediction_horizon: Length of predicted action sequence.
          training_diffusion_steps: Number of diffusion steps during training.
          evaluation_diffusion_steps: Number of diffusion steps during evaluation.
          images_per_camera: Number of images per camera (for temporal stacking).
          dropout: Dropout rate.
          share_camera_features: Whether to share feature extractor weights across cameras.
          early_fusion: Whether to concatenate temporal images before feature extraction.
          feature_normalization: Normalization type ("batch_norm", "layer_norm", or None).
          token_dim: Optional projection dim for tokens.
        """
        super().__init__()
        noise_network_kwargs = {
            "time_dim": 256,
            "hidden_dim": 512,
            "num_blocks": 6,
            "dim_feedforward": 2048,
            "dropout": dropout,
            "nhead": 8,
            "activation": "gelu",
        }
        # Handle visual feature extractors (shared or per-camera).
        self.share_camera_features = share_camera_features
        if self.share_camera_features:
            self.visual_feature_extractors = visual_feature_extractor
        else:
            feature_list = [visual_feature_extractor] + [
                copy.deepcopy(visual_feature_extractor) for _ in range(1, num_cameras)
            ]
            self.visual_feature_extractors = nn.ModuleList(feature_list)

        self.early_fusion = early_fusion
        effective_images_per_camera = 1 if early_fusion else images_per_camera
        self.token_dim = visual_feature_extractor.embed_dim
        self.num_tokens = (
            effective_images_per_camera
            * num_cameras
            * visual_feature_extractor.n_tokens
        )
        self.num_cameras = num_cameras

        # Handle proprioceptive observation incorporation strategy.
        if use_proprioception == "add_token":
            self.proprio_strategy = "add_token"
            self.num_tokens += 1
            self.proprio_processor = nn.Sequential(
                nn.Dropout(p=0.2), nn.Linear(observation_dim, self.token_dim)
            )
        elif use_proprioception == "pad_img_tokens":
            self.proprio_strategy = "pad_img_tokens"
            self.token_dim += observation_dim
            self.proprio_processor = nn.Dropout(p=0.2)
        else:
            assert not use_proprioception
            self.proprio_strategy = None

        # Optional linear projection for token dim.
        linear_projection = nn.Identity()
        if token_dim is not None and token_dim != self.token_dim:
            linear_projection = nn.Linear(self.token_dim, token_dim)
            self.token_dim = token_dim

        # Feature normalization layer.
        if feature_normalization == "batch_norm":
            normalization_layer = BatchNorm1DHelper(self.token_dim)
        elif feature_normalization == "layer_norm":
            normalization_layer = nn.LayerNorm(self.token_dim)
        else:
            assert feature_normalization is None
            normalization_layer = nn.Identity()

        # Post-processing for tokens: projection, norm, dropout.
        self.token_post_processor = nn.Sequential(
            linear_projection, normalization_layer, nn.Dropout(dropout)
        )

        self.noise_prediction_network = DiTNoisePredictionNetwork(
            action_dim=action_dim,
            prediction_horizon=prediction_horizon,
            **noise_network_kwargs,
        )
        self.action_dim, self.prediction_horizon = action_dim, prediction_horizon

        assert (
            evaluation_diffusion_steps <= training_diffusion_steps
        ), "Can't evaluate with more steps than training!"
        self.training_diffusion_steps = training_diffusion_steps
        self.evaluation_diffusion_steps = evaluation_diffusion_steps
        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=training_diffusion_steps,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",
        )

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        proprio_observations: torch.Tensor,
        flattened_actions: torch.Tensor,
        flattened_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Computes training loss for the noise prediction network.

        Args:
          images: Dict of images per camera, each (batch_size, [time,] channels, height, width).
          proprio_observations: Proprioceptive obs (batch_size, observation_dim).
          flattened_actions: Flattened actions (batch_size, prediction_horizon * action_dim).
          flattened_masks: Masks for actions (batch_size, prediction_horizon * action_dim).

        Returns:
          Mean MSE loss between predicted and true noise.
        """
        # Tokenize observations (images + optional proprio).
        batch_size, device = proprio_observations.shape[0], proprio_observations.device
        observation_tokens = self.tokenize_observations(images, proprio_observations)

        # Sample diffusion timesteps.
        timesteps = torch.randint(
            low=0, high=self.training_diffusion_steps, size=(batch_size,), device=device
        ).long()

        # Reshape to (batch_size, prediction_horizon, action_dim) for diffusion logic.
        action_masks = flattened_masks.reshape(
            (batch_size, self.prediction_horizon, self.action_dim)
        )
        actions = flattened_actions.reshape(
            (batch_size, self.prediction_horizon, self.action_dim)
        )
        noise = torch.randn_like(actions)

        # Add noise to actions according to diffusion schedule.
        noisy_actions = self.diffusion_scheduler.add_noise(actions, noise, timesteps)
        _, noise_prediction = self.noise_prediction_network(
            noisy_actions, timesteps, observation_tokens
        )

        # Compute masked MSE loss (only on valid actions).
        loss = nn.functional.mse_loss(noise_prediction, noise, reduction="none")
        loss = (loss * action_masks).sum(
            1
        )  # Sum over horizon and dim, average over batch later.
        return loss.mean()

    def get_actions(
        self,
        images: Dict[str, torch.Tensor],
        proprio_observations: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generates actions via iterative diffusion denoising.

        Args:
          images: Dict of images per camera.
          proprio_observations: Proprioceptive obs (batch_size, observation_dim).
          num_steps: Optional number of diffusion steps (overrides evaluation_diffusion_steps).

        Returns:
          Predicted actions (batch_size, prediction_horizon, action_dim).
        """
        batch_size, device = proprio_observations.shape[0], proprio_observations.device
        observation_tokens = self.tokenize_observations(images, proprio_observations)
        encoder_cache = None
        noisy_actions = torch.randn(
            batch_size, self.prediction_horizon, self.action_dim, device=device
        )

        # Set diffusion steps for evaluation.
        eval_steps = self.evaluation_diffusion_steps
        if num_steps is not None:
            assert (
                num_steps <= self.training_diffusion_steps
            ), f"Can't exceed {self.training_diffusion_steps} steps."
            eval_steps = num_steps

        # Precompute encoder cache once.
        encoder_cache = self.noise_prediction_network.forward_encoder(
            observation_tokens
        )

        # Set scheduler timesteps and denoise iteratively.
        self.diffusion_scheduler.set_timesteps(eval_steps)
        self.diffusion_scheduler.alphas_cumprod = (
            self.diffusion_scheduler.alphas_cumprod.to(device)
        )
        for timestep in self.diffusion_scheduler.timesteps:
            batched_timestep = timestep.unsqueeze(0).repeat(batch_size).to(device)
            noise_prediction = self.noise_prediction_network.forward_decoder(
                noisy_actions, batched_timestep, encoder_cache
            )
            noisy_actions = self.diffusion_scheduler.step(
                model_output=noise_prediction, timestep=timestep, sample=noisy_actions
            ).prev_sample

        return noisy_actions

    def tokenize_observations(
        self,
        images: Dict[str, torch.Tensor],
        proprio_observations: torch.Tensor,
        flatten: bool = False,
    ) -> torch.Tensor:
        """Tokenizes images and proprio into a unified token sequence.

        Args:
          images: Dict of images per camera.
          proprio_observations: Proprio obs.
          flatten: Whether to flatten the tokens into (batch_size, num_tokens * token_dim).

        Returns:
          Token tensor (batch_size, num_tokens, token_dim) or flattened.
        """
        # Extract image tokens first.
        image_tokens = self.embed_images(images)

        # Incorporate proprio based on strategy.
        if self.proprio_strategy == "add_token":
            proprio_token = self.proprio_processor(proprio_observations)[:, None]
            tokens = torch.cat((image_tokens, proprio_token), 1)
        elif self.proprio_strategy == "pad_img_tokens":
            processed_proprio = self.proprio_processor(proprio_observations)
            processed_proprio = processed_proprio[:, None].repeat(
                (1, image_tokens.shape[1], 1)
            )
            tokens = torch.cat((processed_proprio, image_tokens), 2)
        else:
            assert self.proprio_strategy is None
            tokens = image_tokens

        tokens = self.token_post_processor(tokens)
        if flatten:
            return tokens.reshape((tokens.shape[0], -1))
        return tokens

    def embed_images(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Embeds images using feature extractors.

        Handles temporal stacking if not early_fusion.

        Args:
          images: Dict 'cam{i}' -> tensor (batch_size, [time,] channels, height, width).

        Returns:
          Concatenated tokens (batch_size, total_tokens, embed_dim).
        """

        def embed_helper(
            extractor: nn.Module, image_tensor: torch.Tensor
        ) -> torch.Tensor:
            # If early_fusion, concatenate time dimension into channels.
            if self.early_fusion and len(image_tensor.shape) == 5:
                time_steps = image_tensor.shape[1]
                image_tensor = torch.cat(
                    [image_tensor[:, t] for t in range(time_steps)], 1
                )
                return extractor(image_tensor)
            # If temporal but not early_fusion, flatten batch*time and reshape back.
            elif len(image_tensor.shape) == 5:
                batch_size, time_steps, channels, height, width = image_tensor.shape
                embeds = extractor(
                    image_tensor.reshape(
                        (batch_size * time_steps, channels, height, width)
                    )
                )
                embeds = embeds.reshape((batch_size, -1, extractor.embed_dim))
                return embeds

            # Single image case.
            assert len(image_tensor.shape) == 4
            return extractor(image_tensor)

        if self.share_camera_features:
            embeds = [
                embed_helper(self.visual_feature_extractors, images[f"cam{i}"])
                for i in range(self.num_cameras)
            ]
        else:
            embeds = [
                embed_helper(extractor, images[f"cam{i}"])
                for i, extractor in enumerate(self.visual_feature_extractors)
            ]
        return torch.cat(embeds, dim=1)
