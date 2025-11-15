"""FAST GPT Decoder for tokenized action prediction.

Similarly to the FAST-pi0 model, it uses a GPT-style autoregressive decoder (only self-attention)
to generate sequences of tokenized actions. Unlike FASTDETRDecoder, this uses pure self-attention
without a separate visual encoder or cross-attention.
"""
import torch
import torch.nn as nn

from refactoring.configs.task.task import ActionSpace, ObservationSpace
from refactoring.data.constants import (
    ACTION_KEY,
    GRIPPER_ACTION_KEY,
    IS_PAD_KEY,
    ORIENTATION_ACTION_KEY,
    POSITION_ACTION_KEY,
    GripperType,
)
from refactoring.data.tokenize import ActionTokenizer
from refactoring.data.tokenize.tokenizer import Tokenizer
from refactoring.models.decoding.action_heads import ActionHead
from refactoring.models.decoding.constants import (
    ACTION_LOGITS_KEY,
    ACTION_TOKENS_KEY,
    LATENT_KEY,
    LOGVAR_KEY,
    MU_KEY, FeatureType,
)
from refactoring.models.decoding.decoders.base import ActionDecoder, DecoderInput
from refactoring.models.encoding.encoders.constants import EncoderOutputKeys
from refactoring.models.layers.activation import ActivationFunction
from refactoring.models.layers.constants import AttentionType, NormalizationType, PositionalEncodingType
from refactoring.models.layers.gpt_transformer.gpt_decoder import GPTDecoder
from refactoring.models.layers.feature_projection import FeatureProjection


class FASTGPTDecoder(ActionDecoder):
    """FAST GPT decoder for tokenized action prediction.

    Uses pure GPT-style transformer with self-attention only (no cross-attention).
    Visual/proprioceptive features are concatenated as prefix tokens, followed by
    action token embeddings for autoregressive generation.

    Architecture:
        [visual_tokens] + [proprio_tokens] + [latent_token] + [action_tokens] -> GPT -> next_token

    This is similar to Pi0 FAST but adapted to work with any feature encoder.
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
        action_vocabulary_size: int = 2048,
        max_seq_len: int = 512,
        embedding_dimension: int = 256,
        number_of_heads: int = 8,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        number_of_layers: int = 6,
        activation: str = ActivationFunction.SWIGLU.value,
        normalization_type: str = NormalizationType.RMS_NORM.value,
        attention_type: str = AttentionType.MULTI_HEAD.value,
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        positional_encoding_type: str | None = PositionalEncodingType.SINUSOIDAL.value,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        deterministic: bool = True,
    ):
        """Initialize FAST GPT decoder.

        Args:
            input_keys: Feature keys expected from encoder pipeline
            action_space: Action space configuration
            observation_space: Observation space configuration
            observation_horizon: Number of observation timesteps
            prediction_horizon: Max action horizon for generation
            device: Device to run model on
            action_vocabulary_size: Vocabulary size (default 2048 for pretrained FAST)
            max_seq_len: Maximum sequence length for GPT (features + action tokens)
            embedding_dimension: Transformer hidden dimension
            number_of_heads: Number of query attention heads
            number_of_key_value_heads: Number of K/V heads for GQA (None = same as heads = MHA)
            feedforward_dimension: FFN hidden dimension (default: 4 * embedding_dimension)
            number_of_layers: Number of transformer layers
            activation: Activation function (swiglu, gelu, relu, silu)
            normalization_type: Normalization type (rmsnorm, layernorm)
            attention_type: Attention type (gqa, mha)
            dropout_rate: Dropout probability
            attention_dropout: Attention dropout probability
            positional_encoding_type: Type of positional encoding (sinusoidal, rope, None)
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            temperature: Initial temperature for sampling (not used in greedy decoding)
            learnable_temperature: If True, make temperature a learnable parameter
            deterministic: If True, use greedy decoding during inference
        """
        self.vocab_size = action_vocabulary_size
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.embedding_dimension = embedding_dimension
        self.deterministic = deterministic

        action_heads = {
            ACTION_LOGITS_KEY: ActionHead(
                input_dim=embedding_dimension,
                output_dim=action_vocabulary_size,
                blocks=None,
            )
        }

        decoder_input = DecoderInput(
            keys=input_keys,
            required=[],
            requires_actions=True,
            raises_for_types=[FeatureType.SPATIAL.value]
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
        # Feature projection to convert all features to embedding_dimension
        self.feature_projection = FeatureProjection(
            embedding_dim=embedding_dimension,
            warn_on_projection=False,
            raise_on_mismatch=False,
        )
        # Token embedding for action tokens
        self.token_embedding = nn.Embedding(action_vocabulary_size, embedding_dimension)
        self.gpt_decoder = GPTDecoder(
            number_of_layers=number_of_layers,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            dropout=dropout_rate,
            attention_dropout=attention_dropout,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            use_cross_attention=False,  # Pure GPT - no cross-attention
            positional_encoding_type=positional_encoding_type,
            maximum_sequence_length=max_seq_len,
        )
        self.to(self.device)


    def _prepare_feature_tokens(
            self, features: dict[str, torch.Tensor], latent_embedding: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Convert features to tokens and extract masks (None if absent)."""
        feature_tokens_list = []
        mask_segments_list = []  # Build mask in same order as tokens

        language_token_mask = None
        for key, feature in features.items():
            if EncoderOutputKeys.TOKEN_MASK.value in key:
                language_token_mask = feature
                features.pop(key)
                break

        for key, feature in sorted(features.items()):
            if EncoderOutputKeys.LANGUAGE.value in key:
                # Add corresponding mask for these language tokens
                batch_size, seq_len = feature.shape[0], feature.shape[1]
                if language_token_mask is not None:
                    # Find actual sequence length (first False in mask)
                    actual_lengths = language_token_mask.sum(dim=1)
                    max_len = actual_lengths.max().item()
                    # Truncate both tokens and mask to actual content
                    feature = feature[:, -max_len:, :]  # Keep only non-padding, tokens are left-padded so slice from the end
                    mask_segment = language_token_mask[:, -max_len:]
                    mask_segments_list.append(mask_segment)
                else:
                    # No mask provided - assume all language tokens valid
                    batch_size, seq_len = feature.shape[0], feature.shape[1]
                    mask_segments_list.append(
                        torch.ones((batch_size, seq_len), dtype=torch.bool, device=feature.device)
                    )
                # Shape: (B, max_token_len, embed_dim)
                if feature.shape[-1] != self.embedding_dimension:
                    feature = self.feature_projection({key: feature})[key]
                feature_tokens_list.append(feature)
                continue

            if len(feature.shape) >= 4:
                raise ValueError("FASTGPTDecoder doesn't accept spatial features.")
            elif len(feature.shape) == 3:  # Sequential (B, T, D)
                feature = self.feature_projection({key: feature})[key]
                feature_tokens_list.append(feature)
                # Sequential features are always valid
                batch_size, seq_len = feature.shape[0], feature.shape[1]
                mask_segments_list.append(
                    torch.ones((batch_size, seq_len), dtype=torch.bool, device=feature.device)
                )
            elif len(feature.shape) == 2:  # Flat (B, D)
                feature = self.feature_projection({key: feature})[key]
                feature_tokens_list.append(feature.unsqueeze(1))  # (B, 1, D)
                # Flat features are always valid (single token)
                batch_size = feature.shape[0]
                mask_segments_list.append(
                    torch.ones((batch_size, 1), dtype=torch.bool, device=feature.device)
                )
            else:
                raise ValueError(
                    f"Unsupported feature shape for key '{key}': {feature.shape}. "
                    f"Expected 2D (B, D), 3D (B, T, D), but got {len(feature.shape)}D."
                )

        if latent_embedding is not None:
            if len(latent_embedding.shape) == 2:
                latent_embedding = latent_embedding.unsqueeze(1)
            if latent_embedding.shape[-1] != self.embedding_dimension:
                latent_embedding = self.feature_projection({LATENT_KEY: latent_embedding})[LATENT_KEY]
            feature_tokens_list.append(latent_embedding)
            # Latent is always valid
            batch_size, seq_len = latent_embedding.shape[0], latent_embedding.shape[1]
            mask_segments_list.append(
                torch.ones((batch_size, seq_len), dtype=torch.bool, device=latent_embedding.device)
            )

        if len(feature_tokens_list) == 0:
            raise ValueError("FASTGPTDecoder requires at least one feature input.")

        feature_tokens = torch.cat(feature_tokens_list, dim=1)  # (B, num_tokens, D)

        # Concatenate all mask segments in the same order as tokens
        if len(mask_segments_list) > 0:
            full_mask = torch.cat(mask_segments_list, dim=1)  # (B, total_tokens)
            return feature_tokens, full_mask

        return feature_tokens, None

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
                f"Decoder action_vocabulary_size ({self.vocab_size}) doesn't match "
                f"tokenizer action_vocabulary_size ({tokenizer_vocab_size})"
            )

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Training: Teacher forcing with ground truth tokens
        Inference: Autoregressive generation with KV caching

        Args:
            features: Encoded features from pipeline
                May optionally contain LATENT_KEY from algorithm's latent encoder
            actions: Ground truth actions (training) or None (inference)

        Returns:
            Dict with ACTION_LOGITS_KEY (training) or continuous actions (inference)
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set - call set_tokenizer() before forward()")

        latent_embedding = features.get(LATENT_KEY, None)
        observation_features = {
            k: v for k, v in features.items() if k not in {LATENT_KEY, MU_KEY, LOGVAR_KEY}
        }
        feature_tokens, feature_token_mask = self._prepare_feature_tokens(observation_features, latent_embedding)
        if actions is not None:
            predictions = self._forward_training(feature_tokens=feature_tokens, feature_token_mask=feature_token_mask,
                                                 actions=actions)
        else:
            predictions = self._forward_inference(feature_tokens=feature_tokens, feature_token_mask=feature_token_mask)

        for key in [MU_KEY, LOGVAR_KEY]:
            if key in features:
                predictions[key] = features[key]

        return predictions


    def _make_attention_mask(self,
                             feature_tokens: torch.Tensor,
                             action_tokens: torch.Tensor,
                             feature_token_mask: torch.Tensor| None = None
                             ) -> torch.Tensor:
        """Compute attention mask with bidirectional prefix and causal actions.

        Note: True indicates valid tokens, False indicates padding. This is the convention used in
          torch.nn.scaled_dot_product_attention.
        """
        prefix_len = feature_tokens.shape[1]
        action_input_mask = action_tokens != self.pad_token_id  # (B, seq_len-1)
        if feature_token_mask is not None:
            full_input_mask = torch.cat([feature_token_mask, action_input_mask], dim=1)
        else:
            # Assume all feature tokens are valid if no mask is provided
            batch_size = feature_tokens.shape[0]
            prefix_mask = torch.ones((batch_size, prefix_len), dtype=torch.bool, device=feature_tokens.device)
            full_input_mask = torch.cat([prefix_mask, action_input_mask], dim=1)

        # Create AR mask: 0 for bidirectional (prefix), 1 for causal (actions)
        autoregressive_mask = torch.zeros_like(full_input_mask, dtype=torch.float32)
        autoregressive_mask[:, prefix_len:] = 1.0
        cumsum = torch.cumsum(autoregressive_mask, dim=1)
        # Broadcasting: cumsum[:, None, :] = (B, 1, total_len), cumsum[:, :, None]= (B, total_len, 1)
        attention_allowed = (cumsum[:, None, :] <= cumsum[:, :, None]).to(torch.bool) # (B, total_len, total_len)
        valid_mask = full_input_mask[:, None, :] & full_input_mask[:, :, None]
        final_attention_allowed = attention_allowed & valid_mask
        return final_attention_allowed.unsqueeze(1)  # (B, 1, total_len, total_len)


    def _forward_training(
        self,
        actions: dict[str, torch.Tensor],
        feature_tokens: torch.Tensor,
        feature_token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Training forward with teacher forcing.

        Args:
            feature_tokens: Feature token embeddings (B, num_features, D)
            feature_token_mask: Feature token mask (B, num_features)
            actions: Ground truth actions

        Returns:
            Dict with ACTION_LOGITS_KEY and tokenized targets
        """
        tokenized_actions = self._tokenize_actions(actions)
        token_ids = tokenized_actions[ACTION_TOKENS_KEY]  # (B, seq_len)
        # Embed action tokens (all but last for teacher forcing)
        input_ids = token_ids[:, :-1]  # (B, seq_len-1)
        action_token_embeddings = self.token_embedding(input_ids)  # (B, seq_len-1, D)
        full_sequence = torch.cat([feature_tokens, action_token_embeddings], dim=1)  # (B, prefix+seq_len-1, D)
        full_attention_mask = self._make_attention_mask(
            feature_tokens=feature_tokens,
            action_tokens=input_ids,
            feature_token_mask=feature_token_mask,
        )  # (B, prefix+seq_len-1, prefix+seq_len-1)
        decoder_output, _ = self.gpt_decoder(
            hidden_states=full_sequence,
            encoded_features=None,  # No cross-attention
            cross_attention_mask=None,
            decoder_cache=None,
            use_cache=False,
            self_attention_mask=full_attention_mask,
        )  # (B, prefix+seq_len-1, D)

        # Extract action token outputs (skip feature prefix)
        prefix_len = feature_tokens.shape[1]
        action_outputs = decoder_output[:, prefix_len:, :]  # (B, seq_len-1, D)
        head = self.action_heads[ACTION_LOGITS_KEY]
        logits = head(action_outputs)  # (B, seq_len-1, action_vocabulary_size)

        target_ids = token_ids[:, 1:]  # (B, seq_len-1)
        target_is_pad = (
            tokenized_actions[IS_PAD_KEY][:, 1:] if IS_PAD_KEY in tokenized_actions else None
        )

        return {
            ACTION_TOKENS_KEY: logits,  # Predictions: logits over vocabulary
            f"{ACTION_TOKENS_KEY}_target": target_ids,  # Targets: ground truth token IDs
            IS_PAD_KEY: target_is_pad,
        }

    def _forward_inference(
        self,
        feature_tokens: torch.Tensor,
        feature_token_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Inference with autoregressive generation and KV caching.

        Args:
            feature_tokens: Feature token embeddings (B, num_features, D) or None
            feature_token_mask: Feature token mask (B, num_features) or None

        Returns:
            Dict with continuous action predictions
        """
        batch_size = feature_tokens.shape[0]
        prefix_len = feature_tokens.shape[1]
        current_sequence = feature_tokens
        if feature_token_mask is None:
            feature_token_mask = torch.ones(batch_size, prefix_len, dtype=torch.bool, device=self.device)
        autoregressive_mask = torch.zeros(batch_size, prefix_len, device=self.device)
        cumsum = torch.cumsum(autoregressive_mask, dim=1)
        attention_allowed = (cumsum[:, None, :] <= cumsum[:, :, None])
        valid_mask = feature_token_mask[:, None, :] & feature_token_mask[:, :, None]
        self_attention_mask = attention_allowed & valid_mask
        self_attention_mask = self_attention_mask.unsqueeze(1)  # Add head dimension
        decoder_output, decoder_cache = self.gpt_decoder(
            hidden_states=current_sequence,
            encoded_features=None,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=None,
            decoder_cache=None,
            use_cache=True,
        )
        generated_tokens = []
        next_token_embedding = None
        for step in range(self.max_seq_len - prefix_len):
            if step > 0:
                decoder_output, decoder_cache = self.gpt_decoder(
                    hidden_states=next_token_embedding,
                    self_attention_mask=None,  # Standard causal
                    decoder_cache=decoder_cache,
                    use_cache=True,
                )

            # Get last token's output
            last_output = decoder_output[:, -1:, :]  # (B, 1, D)
            head = self.action_heads[ACTION_LOGITS_KEY]
            logits = head(last_output)  # (B, 1, action_vocabulary_size)
            logits_scaled = logits / self.temperature.clamp(min=0.01)
            if self.deterministic:
                next_token = torch.argmax(logits, dim=-1)  # (B, 1)
            else:
                probs = torch.softmax(logits_scaled, dim=-1)
                next_token = torch.multinomial(probs.squeeze(1), num_samples=1)  # (B, 1)
            generated_tokens.append(next_token)

            if (next_token == self.eos_token_id).all():
                break

            # Early stopping if we've generated enough actions
            with torch.no_grad():
                current_tokens = torch.cat(generated_tokens, dim=1)  # (B, current_len)
                detokenized_actions = self._detokenize_predictions(current_tokens)
                decoded_lengths = [v.shape[1] for v in detokenized_actions.values()]
                min_decoded_len = min(decoded_lengths) if decoded_lengths else 0
                if min_decoded_len >= self.prediction_horizon:
                    break
            next_token_embedding = self.token_embedding(next_token)  # (B, 1, D)

        generated_token_ids = torch.cat(generated_tokens, dim=1)  # (B, num_tokens)
        return self._detokenize_predictions(generated_token_ids)

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
        device = action_chunks.device

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
            # Add EOS token
            tokens = tokens + [self.eos_token_id]
            tokens_list_of_lists.append(tokens)

        # Pad to max length
        max_token_len = max(len(tokens) for tokens in tokens_list_of_lists)
        token_ids = torch.full(
            (batch_size, max_token_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
            device=action_chunks.device,
        )
        token_is_pad = torch.ones((batch_size, max_token_len), dtype=torch.bool, device=device)

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
        device = token_ids.device

        tokens_list_of_lists = []
        for i in range(batch_size):
            sample_tokens = token_ids[i]
            # Filter out EOS and PAD tokens
            valid_tokens = sample_tokens[
                (sample_tokens != self.eos_token_id) & (sample_tokens != self.pad_token_id)
            ]
            tokens_list_of_lists.append(valid_tokens.cpu().tolist())

        tokens_dict = {ACTION_KEY: tokens_list_of_lists}
        decoded_dict = self.tokenizer.detokenize(tokens_dict)
        decoded_actions = decoded_dict[ACTION_KEY]
        decoded_actions = torch.from_numpy(decoded_actions).to(device)

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
