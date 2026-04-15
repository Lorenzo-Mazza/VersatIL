"""A MoE action decoder which utilizes the latent layer of the Free Transformer as gating for multiple action heads."""

import torch
from torch import nn

from versatil.data.constants import SampleKey
from versatil.data.task import ActionSpace, ObservationSpace
from versatil.data.tokenization import Tokenizer
from versatil.models.decoding.action_heads import ActionHead
from versatil.models.decoding.action_heads.moe import MoEHead
from versatil.models.decoding.action_masking import make_attention_mask
from versatil.models.decoding.constants import DecoderOutputKey
from versatil.models.decoding.decoders import ActionDecoder
from versatil.models.decoding.decoders.factory.free_action_transformer import (
    FreeActionTransformer,
)


class MoEFreeActionTransformer(FreeActionTransformer):
    """A Mixture-of-Experts (MoE) action decoder utilizing the Free Transformer architecture.

    This decoder extends the Free Transformer by incorporating MoE action heads.
    It leverages the latent representations from the Free Transformer as gating signals
    to route inputs to multiple expert action heads.

    During the forward pass:
        1. The Free Transformer processes input features to produce action embeddings.
        2. Each MoE action head uses the latent layer outputs as routing weights to select experts.
        3. Each expert specializes in different aspects of action prediction.
    """

    def __init__(
        self,
        action_heads: dict[str, ActionHead],
        input_keys: list[str],
        action_space: ActionSpace,
        observation_space: ObservationSpace,
        observation_horizon: int,
        prediction_horizon: int,
        device: str,
        max_seq_len: int = 512,
        embedding_dimension: int = 256,
        number_of_heads: int = 8,
        number_of_key_value_heads: int | None = None,
        feedforward_dimension: int | None = None,
        number_of_decoder_layers: int = 6,
        number_of_encoder_layers: int = 1,
        latent_bits: int = 16,
        activation: str = "swiglu",
        normalization_type: str = "rmsnorm",
        attention_type: str = "mha",
        dropout_rate: float = 0.1,
        attention_dropout: float = 0.0,
        positional_encoding_type: str | None = "rope",
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        deterministic: bool = True,
        use_global_latent: bool = True,
    ):
        """Initialize MoeFreeTransformer decoder.

        Args:
            action_heads: Action heads for different action components.
            input_keys: List of feature keys required from encoding pipeline.
            action_space: Action space configuration.
            observation_space: Observation space configuration.
            observation_horizon: Number of observation timesteps.
            prediction_horizon: Number of action timesteps to predict.
            device: Device for computation.
            max_seq_len: Maximum input token sequence length.
            embedding_dimension: Model embedding dimension.
            number_of_heads: Number of attention heads.
            number_of_key_value_heads: Number of K/V heads for GQA.
            feedforward_dimension: FFN hidden dimension.
            number_of_decoder_layers: Total decoder layers.
            number_of_encoder_layers: Number of latent encoder layers.
            latent_bits: Number of bits for latent codes.
            activation: Activation function name.
            normalization_type: Normalization type name.
            attention_type: Attention type name.
            dropout_rate: Dropout probability.
            attention_dropout: Attention dropout probability.
            positional_encoding_type: Type of positional encoding.
            temperature: Initial temperature for sampling.
            learnable_temperature: If True, make temperature a learnable parameter.
            deterministic: If True, use greedy decoding during inference.
            use_global_latent: If True, use a single latent code for the entire action sequence.
        """
        super().__init__(
            action_heads=action_heads,
            input_keys=input_keys,
            action_space=action_space,
            observation_space=observation_space,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
            device=device,
            max_seq_len=max_seq_len,
            embedding_dimension=embedding_dimension,
            number_of_heads=number_of_heads,
            number_of_key_value_heads=number_of_key_value_heads,
            feedforward_dimension=feedforward_dimension,
            number_of_decoder_layers=number_of_decoder_layers,
            number_of_encoder_layers=number_of_encoder_layers,
            latent_bits=latent_bits,
            activation=activation,
            normalization_type=normalization_type,
            attention_type=attention_type,
            dropout_rate=dropout_rate,
            attention_dropout=attention_dropout,
            positional_encoding_type=positional_encoding_type,
            temperature=temperature,
            learnable_temperature=learnable_temperature,
            deterministic=deterministic,
            use_global_latent=use_global_latent,
        )
        self.moe_action_head: MoEHead = self.action_heads[
            DecoderOutputKey.ACTION_LOGITS.value
        ]
        self.expert_gating_projection = None

    def get_auxiliary_output_keys(self) -> set[str]:
        """MoE free transformer adds routing weights to free transformer's auxiliary keys."""
        keys = super().get_auxiliary_output_keys()
        keys.add(DecoderOutputKey.ROUTING_WEIGHTS.value)
        return keys

    def set_tokenizer(self, tokenizer: Tokenizer | None = None):
        if tokenizer is None or tokenizer.action_tokenizer is None:
            raise ValueError(
                "MoEFreeActionTransformer requires a tokenizer for tokenized action prediction."
            )
        device = self.temperature.device
        self.vocab_size = tokenizer.action_tokenizer.vocab_size
        self.moe_action_head.output_dim = self.vocab_size
        token_input_embedding = nn.Embedding(
            self.vocab_size, self.embedding_dimension
        ).to(device)
        nn.init.normal_(
            token_input_embedding.weight,
            mean=0.0,
            std=self.free_transformer.initializer_range,
        )
        self.token_embedding = token_input_embedding
        for expert in self.moe_action_head.experts:
            expert: ActionHead
            output_block_in_features = expert.output_proj.in_features
            expert_out = nn.Linear(
                output_block_in_features, self.vocab_size, bias=True, device=device
            )
            nn.init.kaiming_uniform_(expert_out.weight, nonlinearity="linear")
            nn.init.zeros_(expert_out.bias)
            expert.output_dim = self.vocab_size
            expert.output_proj = expert_out  # Replace final projection with expert head
        expert_gating_projection = nn.Linear(
            self.free_transformer.embedding_dimension,
            self.moe_action_head.num_experts,
            bias=False,
            device=device,
        )
        nn.init.normal_(
            expert_gating_projection.weight,
            mean=0.0,
            std=self.free_transformer.initializer_range,
        )
        self.expert_gating_projection = expert_gating_projection
        ActionDecoder.set_tokenizer(
            self, tokenizer
        )  # Call action decoder base, free transformer base would raise error

    def _forward_training(
        self,
        actions: dict[str, torch.Tensor],
        feature_tokens: torch.Tensor,
        feature_token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prefix_len = feature_tokens.shape[1]
        target_token_ids = actions[SampleKey.TOKENIZED_ACTIONS.value]
        action_token_embeddings = self.token_embedding(target_token_ids)
        full_attention_mask, full_key_padding_mask = make_attention_mask(
            feature_tokens=feature_tokens,
            action_tokens=action_token_embeddings,
            feature_token_mask=feature_token_mask,
        )
        full_token_sequence = torch.cat(
            [feature_tokens, action_token_embeddings], dim=1
        )
        if full_token_sequence.shape[1] > self.max_seq_len:
            raise ValueError(
                f"Input token length {full_token_sequence.shape[1]} > max_seq_len {self.max_seq_len}."
            )

        (
            decoder_output,
            bit_logits,
            latent_codes,
            latent_embeddings,
            _,
        ) = self.free_transformer(
            hidden_states=full_token_sequence,
            key_padding_mask=full_key_padding_mask,
            self_attention_mask=full_attention_mask,
            is_inference=False,
            return_latent_embeddings=True,
        )
        # Shift alignment: grabs outputs from the last feature to the penultimate action so step t predicts target t+1.
        action_outputs = decoder_output[
            :, prefix_len - 1 : -1, :
        ]  # (B, action_token_len, D)
        gating_logits = self.expert_gating_projection(
            latent_embeddings
        )  # Global latent (B, 1, num_experts)
        logits_dict = self.moe_action_head(
            features=action_outputs, gating_feature=gating_logits
        )
        logits = logits_dict[SampleKey.ACTION.value]
        expert_usage = logits_dict[DecoderOutputKey.ROUTING_WEIGHTS.value]
        return {
            DecoderOutputKey.ACTION_LOGITS.value: logits,
            DecoderOutputKey.BINARY_LOGITS.value: bit_logits,
            DecoderOutputKey.LATENT_CODES.value: latent_codes,
            DecoderOutputKey.ROUTING_WEIGHTS.value: expert_usage,
        }

    def _forward_inference(
        self,
        feature_tokens: torch.Tensor,
        feature_token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = feature_tokens.shape[0]
        prefix_len = feature_tokens.shape[1]
        current_sequence = feature_tokens
        prefix_self_mask = torch.zeros(
            batch_size, 1, prefix_len, prefix_len, dtype=torch.bool, device=self.device
        )
        generation_cache = self.free_transformer.create_empty_generation_cache(
            batch_size=batch_size, device=self.device, dtype=feature_tokens.dtype
        )
        (
            decoder_output,
            _,
            latent_codes,
            latent_embeddings,
            generation_cache,
        ) = self.free_transformer(
            hidden_states=current_sequence,
            key_padding_mask=feature_token_mask,
            self_attention_mask=prefix_self_mask,
            generation_cache=generation_cache,
            is_inference=True,
            return_latent_embeddings=True,
        )
        generated_tokens = []
        expert_usages = []
        next_token_embedding = None
        for step in range(self.max_seq_len - prefix_len):
            if step > 0:
                (
                    decoder_output,
                    _,
                    latent_codes,
                    latent_embeddings,
                    generation_cache,
                ) = self.free_transformer(
                    hidden_states=next_token_embedding,
                    key_padding_mask=None,  # Cached mask handles prefix padding; new token is always valid
                    self_attention_mask=None,  # Causal mask handled internally
                    generation_cache=generation_cache,
                    is_inference=True,
                    return_latent_embeddings=True,
                )
            last_output = decoder_output[:, -1:, :]  # (B, 1, embedding_dimension)
            gating_logits = self.expert_gating_projection(
                latent_embeddings
            )  # (B, 1, num_experts)
            logits_dict = self.moe_action_head(
                features=last_output, gating_feature=gating_logits
            )
            logits = logits_dict[SampleKey.ACTION.value]  # (B, 1, vocab_size)
            logits_scaled = logits / self.temperature.clamp(min=0.01)
            if self.deterministic:
                next_token = torch.argmax(logits, dim=-1)  # (B, 1)
            else:
                probs = torch.softmax(logits_scaled, dim=-1)
                next_token = torch.multinomial(
                    probs.squeeze(1), num_samples=1
                )  # (B, 1)
            expert_usage = logits_dict[
                DecoderOutputKey.ROUTING_WEIGHTS.value
            ]  # (B, 1, num_experts)
            expert_usages.append(expert_usage)
            generated_tokens.append(next_token)
            if (next_token == self.tokenizer.eos_token_id).all():
                break
            next_token_embedding = self.token_embedding(
                next_token
            )  # (B, 1, embedding_dimension)

        return {
            DecoderOutputKey.PREDICTED_ACTION_TOKENS.value: torch.cat(
                generated_tokens, dim=1
            ),  # (B, num_generated_tokens)
            f"{DecoderOutputKey.ACTION_LOGITS.value}_{DecoderOutputKey.ROUTING_WEIGHTS.value}": torch.cat(
                expert_usages, dim=1
            ),  # (B, num_generated_tokens, num_experts)
        }
