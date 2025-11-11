"""FAST Decoder for tokenized action prediction.

DETR-style transformer encoder (ACT-like, no history supported) and GPT-style autoregressive transformer decoder
 specifically designed for FAST (Frequency-space Action Sequence Tokenization) action prediction.

Reference: https://arxiv.org/abs/2501.09747
"""
import logging

import torch
import torch.nn as nn

from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.data.constants import (
    ACTION_KEY,
    GRIPPER_ACTION_KEY,
    IS_PAD_KEY,
    ORIENTATION_ACTION_KEY,
    POSITION_ACTION_KEY, GripperType,
)
from refactoring.data.tokenize import ActionTokenizer
from refactoring.data.tokenize.tokenizer import Tokenizer
from refactoring.models.decoding.action_heads import ActionHead
from refactoring.models.decoding.constants import (
    ACTION_LOGITS_KEY,
    ACTION_TOKENS_KEY,
    LATENT_KEY,
    LOGVAR_KEY,
    MU_KEY,
)
from refactoring.models.decoding.decoders.base import ActionDecoder, DecoderInput, FeatureType
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.detr_transformer.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer, generate_causal_mask,
)
from refactoring.models.layers.detr_transformer.transformer_encoder import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from refactoring.models.layers.feature_projection import (
    FeatureProjection,
    SpatialFeatureConcatenator,
)
from refactoring.models.layers.positional_encoding.sinusoidal import (
    SinusoidalPositionalEncoding1D,
    SinusoidalPositionalEncoding2D,
)


class FASTDETRDecoder(ActionDecoder):
    """FAST DETR decoder for tokenized action prediction.

    Uses DETR-style transformer encoder and GPT-style decoder to generate
    variable-length sequences of discrete action tokens (FAST tokenization) autoregressively.
    """

    supports_tokenized_actions: bool = True

    def __init__(
        self,
        input_keys: list[str],
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        observation_horizon: int,
        prediction_horizon: int,
        device: str,
        vocab_size: int = 2048,
        max_seq_len: int = 512,
        embedding_dimension: int = 256,
        number_of_heads: int = 8,
        feedforward_dimension: int = 512,
        number_of_encoder_layers: int = 6,
        number_of_decoder_layers: int = 6,
        activation: str = ActivationFunction.RELU.value,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        deterministic: bool = True,
    ):
        """Initialize FAST decoder.

        Args:
            input_keys: Feature keys expected from encoder pipeline
            action_space: Action space configuration
            observation_space: Observation space configuration
            observation_horizon: Number of observation timesteps
            prediction_horizon: Max sequence length for generation
            device: Device to run model on
            vocab_size: Vocabulary size (default 2048 for pretrained FAST)
            max_seq_len: Maximum sequence length for positional encoding
            embedding_dimension: Transformer hidden dimension
            number_of_heads: Number of attention heads
            feedforward_dimension: FFN hidden dimension
            number_of_encoder_layers: Number of encoder layers for visual features
            number_of_decoder_layers: Number of decoder layers
            activation: Activation function
            dropout_rate: Dropout probability
            normalize_before: Use pre-normalization
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            temperature: Initial temperature for sampling (not used in greedy decoding)
            learnable_temperature: If True, make temperature a learnable parameter
            deterministic: If True, use greedy decoding during inference
        """
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.embedding_dimension = embedding_dimension
        self.deterministic = deterministic

        action_heads = {
            ACTION_LOGITS_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=vocab_size,
                blocks=None,
            )
        }

        decoder_input = DecoderInput(
            keys=input_keys,
            required_types=[FeatureType.SPATIAL.value],
            requires_actions=True,
        )

        super().__init__(
            decoder_input=decoder_input,
            action_space=action_space,
            action_heads=action_heads,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
        )

        self.temperature = nn.Parameter(
            torch.tensor(temperature, dtype=torch.float32),
            requires_grad=learnable_temperature,
        )

        # Feature projection for handling spatial and flat features
        self.spatial_feature_concatenator = SpatialFeatureConcatenator(
            target_channels=embedding_dimension,
            concat_dim=3,  # Concatenate along width for multi-camera
            warn_on_projection=True,
        )
        self.flat_feature_projection = FeatureProjection(
            embedding_dim=embedding_dimension,
            warn_on_projection=False,
            raise_on_mismatch=False,
        )
        # Final projection for concatenated flat features to embedding_dimension
        # Using LazyLinear since we don't know total flat feature dimension a priori
        self.flat_feature_final_projection = nn.LazyLinear(embedding_dimension)

        self.token_embedding = nn.Embedding(vocab_size, embedding_dimension)

        # Positional encoding for token sequences (1D)
        self.token_positional_encoding = SinusoidalPositionalEncoding1D(
            embedding_dimension=embedding_dimension,
            maximum_length=max_seq_len,
        )

        # Positional encoding for image features (2D)
        self.image_positional_encoding = SinusoidalPositionalEncoding2D(
            embedding_dimension=embedding_dimension,
            normalize=True,
        )

        # Learnable positional embeddings for additional tokens (latent + proprio)
        # Index 0: latent, Index 1: proprio
        self.additional_pos_embed = nn.Embedding(2, embedding_dimension)

        encoder_layer = TransformerEncoderLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout_rate,
            activation=activation,
            normalize_before=normalize_before,
        )
        encoder_norm = nn.LayerNorm(embedding_dimension) if normalize_before else None
        self.visual_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            number_of_layers=number_of_encoder_layers,
            normalization=encoder_norm,
        )

        decoder_layer = TransformerDecoderLayer(
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout_rate,
            activation=activation,
            normalize_before=normalize_before,
        )
        decoder_norm = nn.LayerNorm(embedding_dimension) if normalize_before else None
        self.action_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            number_of_layers=number_of_decoder_layers,
            normalization=decoder_norm,
            return_intermediate=False,
        )

        self.dropout = nn.Dropout(dropout_rate)

        # Move all components to device
        self.to(self.device)

        logging.info(
            f"FASTDecoder initialized: vocab_size={vocab_size}, "
            f"max_seq_len={max_seq_len}, embedding_dim={embedding_dimension}"
        )

    def _prepare_flat_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Extract and project flat features (proprioceptive, language, etc.).

        Args:
            features: Dictionary of encoded features

        Returns:
            Projected flat features (B, embedding_dimension) or None if no flat features
        """
        flat_features_dict = {}
        for key, feature in features.items():
            if len(feature.shape) == 3:
                if self.has_history:
                    feature = feature[:, -1]  # Use most recent timestep
                elif feature.shape[1] == 1:
                    feature = feature.squeeze(1)  # Squeeze temporal dimension when T=1
                else:
                    raise ValueError(
                        f"Feature {key} has temporal dimension T={feature.shape[1]}, "
                        f"but FASTDecoder expects single-frame observation"
                    )
            if len(feature.shape) == 2:
                flat_features_dict[key] = feature
        if len(flat_features_dict) == 0:
            return None

        # First project each feature to embedding_dimension, then concatenate
        concatenated = self.flat_feature_projection.project_and_concatenate(
            flat_features_dict,
            concatenation_dimension=-1,
        )

        # Then project concatenated features back to embedding_dimension for use as single token
        return self.flat_feature_final_projection(concatenated)

    def _prepare_image_features(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """Collect and concatenate spatial features from encoder pipeline.

        Args:
            features: Dictionary of encoded features

        Returns:
            Concatenated spatial features (B, embedding_dimension, H, W_total)

        Raises:
            ValueError: If no spatial features are found
        """
        spatial_features_dict = {}

        for key, feature in sorted(features.items()):
            if len(feature.shape) == 5:
                if self.has_history:
                    feature = feature[:, -1]  # Use most recent timestep
                elif feature.shape[1] == 1:
                    feature = feature.squeeze(1)  # Squeeze temporal dimension when T=1
                else:
                    raise ValueError(
                        f"Feature {key} has temporal dimension T={feature.shape[1]}, "
                        f"but FASTDecoder expects single-frame observation"
                    )
            if len(feature.shape) == 4:  # Spatial features (B, C, H, W)
                spatial_features_dict[key] = feature

        if len(spatial_features_dict) == 0:
            raise ValueError(
                f"No spatial features found. Available keys: {list(features.keys())}"
            )
        elif len(spatial_features_dict) == 1:
            # Single spatial feature - still need to project if channel dimension mismatches
            feature_name, feature_tensor = list(spatial_features_dict.items())[0]
            return self.spatial_feature_concatenator({feature_name: feature_tensor})  # type: ignore[no-any-return]
        else:
            # Multiple spatial features - concatenate with automatic projection
            return self.spatial_feature_concatenator(spatial_features_dict)  # type: ignore[no-any-return]


    def set_tokenizer(self, tokenizer: Tokenizer):
        """Set tokenizer and validate vocab size."""
        super().set_tokenizer(tokenizer)
        if ACTION_KEY not in tokenizer.tokenizers:
            raise ValueError(
                f"Tokenizer must have '{ACTION_KEY}' tokenizer fitted. "
                f"Available: {list(tokenizer.tokenizers.keys())}"
            )
        action_tokenizer: ActionTokenizer = tokenizer.tokenizers[ACTION_KEY]
        tokenizer_vocab_size = action_tokenizer.processor.vocab_size
        if tokenizer_vocab_size != self.vocab_size:
            raise ValueError(
                f"Decoder vocab_size ({self.vocab_size}) doesn't match "
                f"tokenizer vocab_size ({tokenizer_vocab_size})"
            )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Training: Teacher forcing with ground truth tokens
        Inference: Autoregressive generation

        Args:
            features: Encoded features from pipeline
                May optionally contain LATENT_KEY from algorithm's latent encoder
            actions: Ground truth actions (training) or None (inference)

        Returns:
            Dict with ACTION_LOGITS_KEY (training) or continuous actions (inference)
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set - call set_tokenizer() before forward()")

        # Extract latent embedding if provided by algorithm
        latent_embedding = features.get(LATENT_KEY, None)

        # Prepare observation features (excluding latent-related keys)
        observation_features = {
            k: v for k, v in features.items()
            if k not in {LATENT_KEY, MU_KEY, LOGVAR_KEY}
        }

        encoder_output = self._encode_visual_features(observation_features, latent_embedding)

        if actions is not None:
            predictions = self._forward_training(encoder_output, actions)
        else:
            predictions = self._forward_inference(encoder_output)

        # Preserve latent-related outputs from algorithm (e.g., mu, logvar for loss computation)
        for key in [MU_KEY, LOGVAR_KEY]:
            if key in features:
                predictions[key] = features[key]

        return predictions

    def _encode_visual_features(
        self,
        features: dict[str, torch.Tensor],
        latent_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode visual features with positional encoding.

        Following original DETR implementation: prepends latent and/or proprio tokens
        to encoder input with learnable positional embeddings.

        Args:
            features: Dict with spatial and flat features
            latent_embedding: Optional VAE latent embedding (B, embedding_dimension)

        Returns:
            - encoded_features: (seq_len, B, embedding_dimension)
        """
        # Collect and concatenate all spatial features
        spatial_features = self._prepare_image_features(features)  # (B, C, H, W)
        flat_features = self._prepare_flat_features(features)  # (B, D) or None
        batch_size, _, height, width = spatial_features.shape

        # Compute 2D positional encoding for image
        positional_encoding = self.image_positional_encoding(
            torch.zeros(1, 1, height, width, device=self.device)
        )[0]  # (embedding_dimension, H, W)

        # Flatten spatial features: (B, C, H, W) -> (H*W, B, C)
        flattened_features = spatial_features.flatten(2).permute(2, 0, 1)

        # Flatten positional encoding and repeat for batch
        positional_encoding_flat = (
            positional_encoding.flatten(1)
            .permute(1, 0)
            .unsqueeze(1)
            .repeat(1, batch_size, 1)
        )  # (H*W, B, embedding_dimension)

        # Prepare learnable positional embeddings for additional tokens
        additional_pos_embed = self.additional_pos_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # (2, B, D)

        # Prepend latent and/or proprio tokens (following original DETR)
        if flat_features is None:
            if latent_embedding is not None:
                # Only latent: prepend 1 token
                addition_input = latent_embedding.unsqueeze(0)  # (1, B, D)
                positional_encoding = torch.cat([additional_pos_embed[0].unsqueeze(0), positional_encoding_flat], dim=0)
                encoder_input = torch.cat([addition_input, flattened_features], dim=0)
            else:
                # No additional tokens
                positional_encoding = positional_encoding_flat
                encoder_input = flattened_features
        else:
            if latent_embedding is not None:
                # Both latent and proprio: prepend 2 tokens
                addition_input = torch.stack([latent_embedding, flat_features], dim=0)  # (2, B, D)
                positional_encoding = torch.cat([additional_pos_embed, positional_encoding_flat], dim=0)
            else:
                # Only proprio: prepend 1 token
                addition_input = flat_features.unsqueeze(0)  # (1, B, D)
                positional_encoding = torch.cat([additional_pos_embed[1].unsqueeze(0), positional_encoding_flat], dim=0)
            encoder_input = torch.cat([addition_input, flattened_features], dim=0)

        # Encode through transformer encoder with positional encoding as separate parameter
        # DETR-style: positional encoding is added to Q/K inside the encoder, not to values
        encoded = self.visual_encoder(
            source=encoder_input,
            positional_encoding=positional_encoding
        )  # (seq_len, B, D)

        return encoded

    def _forward_training(
        self,
        encoder_output: torch.Tensor,
        actions: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Training forward with teacher forcing.

        Args:
            encoder_output: Encoded visual features (seq_len, B, D)
            actions: Ground truth actions

        Returns:
            Dict with ACTION_LOGITS_KEY and tokenized targets
        """
        tokenized_actions = self._tokenize_actions(actions)
        token_ids = tokenized_actions[ACTION_TOKENS_KEY]  # (B, seq_len)
        batch_size, seq_len = token_ids.shape
        # Shift: Input embeds for tokens 0 to seq-2, to predict 1 to seq-1 (EOS)
        input_ids = token_ids[:, :-1]
        token_embeddings = self.token_embedding(input_ids)  # (B, seq_len-1, D)
        token_embeddings = self.dropout(token_embeddings)
        token_embeddings = self.token_positional_encoding(token_embeddings)
        token_embeddings = token_embeddings.transpose(0, 1)  # (seq_len-1, B, D)
        causal_mask = generate_causal_mask(seq_len - 1, token_embeddings.device)
        decoder_output = self.action_decoder(
            target=token_embeddings,
            memory=encoder_output,
            target_mask=causal_mask,
            memory_mask=None,
            target_key_padding_mask=tokenized_actions.get(IS_PAD_KEY, None)[:, :-1] if IS_PAD_KEY in tokenized_actions else None,
            memory_key_padding_mask=None,
        )  # (1, seq_len-1, B, D)
        decoder_output = decoder_output[0].transpose(0, 1)  # (B, seq_len-1, D)
        head = self.action_heads[ACTION_LOGITS_KEY]
        logits = head(decoder_output)  # (B, seq_len-1, vocab_size)
        # Shifted targets
        target_ids = token_ids[:, 1:]
        target_is_pad = tokenized_actions[IS_PAD_KEY][:, 1:] if IS_PAD_KEY in tokenized_actions else None
        return {
            ACTION_TOKENS_KEY: logits,  # Predictions: logits over vocabulary
            f"{ACTION_TOKENS_KEY}_target": target_ids,  # Targets: ground truth token IDs (shifted)
            IS_PAD_KEY: target_is_pad
        }


    def _forward_inference(
        self,
        encoder_output: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Inference with autoregressive generation.

        FAST paper doesn't use BOS token - generation starts from empty sequence.

        Args:
            encoder_output: Encoded visual features

        Returns:
            Dict with continuous action predictions
        """
        batch_size = encoder_output.shape[1]
        device = encoder_output.device

        # Start with empty token sequence (no BOS)
        generated_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=device)

        for step in range(self.max_seq_len):
            if generated_tokens.shape[1] == 0:
                # First token: use a zero embedding to get initial prediction
                token_embeddings = torch.zeros(
                    batch_size, 1, self.embedding_dimension, device=device
                )
                token_embeddings = self.token_positional_encoding(token_embeddings)
            else:
                # Subsequent tokens: use generated tokens
                token_embeddings = self.token_embedding(generated_tokens)
                token_embeddings = self.token_positional_encoding(token_embeddings)

            token_embeddings = token_embeddings.transpose(0, 1)  # (T, B, D)

            seq_len = token_embeddings.shape[0]
            causal_mask = generate_causal_mask(seq_len, device)

            decoder_output = self.action_decoder(
                target=token_embeddings,
                memory=encoder_output,
                target_mask=causal_mask,
                memory_mask=None,
            )[0].transpose(0, 1)  # (B, T, D)

            head = self.action_heads[ACTION_LOGITS_KEY]
            logits = head(decoder_output[:, -1:, :])  # (B, 1, vocab_size)
            logits_scaled = logits / self.temperature.clamp(min=0.01)  # Prevent division by zero
            probs = torch.softmax(logits_scaled, dim=-1)  # (B, 1, vocab_size)
            if self.deterministic:
                next_token = torch.argmax(logits, dim=-1)  # (B, 1) # Deterministic greedy decoding
            else:
                next_token = torch.multinomial(probs.squeeze(1), num_samples=1)  # (B, 1) - stochastic sampling
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
            # Add early stopping check
            with torch.no_grad():
                detokenized_actions = self._detokenize_predictions(generated_tokens)
                # Take min over batch
                decoded_lengths = [v.shape[1] for v in detokenized_actions.values()]
                min_decoded_len = min(decoded_lengths) if decoded_lengths else 0
                if min_decoded_len >= self.prediction_horizon:
                    break

            if (next_token == self.eos_token_id).all():
                break

        return self._detokenize_predictions(generated_tokens)

    def _tokenize_actions(self, actions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Tokenize ground truth actions.

        Args:
            actions: Dict with position_action, orientation_action, gripper_action, is_pad

        Returns:
            Dict with ACTION_TOKENS_KEY and IS_PAD_KEY
        """
        action_components = []
        if self.action_space.has_position and POSITION_ACTION_KEY in actions:
            action_components.append(actions[POSITION_ACTION_KEY])
        if self.action_space.has_orientation and ORIENTATION_ACTION_KEY in actions:
            action_components.append(actions[ORIENTATION_ACTION_KEY])
        if self.action_space.has_gripper and GRIPPER_ACTION_KEY in actions:
            gripper = actions[GRIPPER_ACTION_KEY]
            if self.action_space.gripper_type == GripperType.BINARY.value:
                # Remap binary {0,1} to continuous {-1,1}
                gripper_continuous = 2.0 * gripper - 1.0
            else:
                gripper_continuous = gripper
            action_components.append(gripper_continuous)

        action_chunks = torch.cat(action_components, dim=-1)  # (B, T, action_dim)
        batch_size = action_chunks.shape[0]

        if IS_PAD_KEY in actions:
            is_pad = actions[IS_PAD_KEY].squeeze(-1)  # (B, T)
            action_chunks_list = []
            for i in range(batch_size):
                valid_mask = ~is_pad[i]
                valid_actions = action_chunks[i, valid_mask]
                action_chunks_list.append(valid_actions)
        else:
            action_chunks_list = [action_chunks[i] for i in range(batch_size)]

        tokens_list_of_lists = []
        for valid_actions in action_chunks_list:
            valid_actions_batch = valid_actions.unsqueeze(0)
            tokens_dict = self.tokenizer.tokenize({ACTION_KEY: valid_actions_batch})
            tokens = tokens_dict[ACTION_KEY][0]
            # Add EOS token only
            tokens = tokens + [self.eos_token_id]
            tokens_list_of_lists.append(tokens)

        max_token_len = max(len(tokens) for tokens in tokens_list_of_lists)
        token_ids = torch.full(
            (batch_size, max_token_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=action_chunks.device,
        )
        token_is_pad = torch.ones(
            (batch_size, max_token_len), dtype=torch.bool, device=action_chunks.device
        )

        for i, tokens in enumerate(tokens_list_of_lists):
            token_len = len(tokens)
            token_ids[i, :token_len] = torch.tensor(tokens, dtype=torch.long)
            token_is_pad[i, :token_len] = False

        return {
            ACTION_TOKENS_KEY: token_ids,
            IS_PAD_KEY: token_is_pad,
        }

    def _detokenize_predictions(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Detokenize generated tokens to continuous actions.

        Args:
            token_ids: Generated token IDs (B, seq_len)

        Returns:
            Dict with continuous action predictions
        """
        batch_size = token_ids.shape[0]

        tokens_list_of_lists = []
        for i in range(batch_size):
            sample_tokens = token_ids[i]
            # Filter out EOS and PAD tokens
            valid_tokens = sample_tokens[
                (sample_tokens != self.eos_token_id)
                & (sample_tokens != self.pad_token_id)
            ]
            tokens_list_of_lists.append(valid_tokens.cpu().tolist())

        tokens_dict = {ACTION_KEY: tokens_list_of_lists}
        decoded_dict = self.tokenizer.detokenize(tokens_dict)
        decoded_actions = decoded_dict[ACTION_KEY]
        decoded_actions = torch.from_numpy(decoded_actions).to(token_ids.device)

        result = {}
        idx = 0

        if self.action_space.has_position:
            position_dim = self.action_space.position_dim
            result[POSITION_ACTION_KEY] = decoded_actions[..., idx : idx + position_dim]
            idx += position_dim

        if self.action_space.has_orientation:
            orientation_dim = self.action_space.orientation_dim
            result[ORIENTATION_ACTION_KEY] = decoded_actions[..., idx : idx + orientation_dim]
            idx += orientation_dim

        if self.action_space.has_gripper:
            gripper_dim = self.action_space.gripper_dim
            result[GRIPPER_ACTION_KEY] = decoded_actions[..., idx : idx + gripper_dim]
            idx += gripper_dim

        return result