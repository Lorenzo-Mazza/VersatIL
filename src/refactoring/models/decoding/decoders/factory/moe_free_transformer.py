"""A MoE action decoder which utilizes the latent layer of the Free Transformer as gating for multiple action heads."""

import torch

from refactoring.data.constants import TOKENIZED_ACTIONS_KEY, ACTION_KEY, IS_PAD_ACTION_KEY
from refactoring.data.tokenization import Tokenizer
from refactoring.models.decoding.action_heads.moe import MoEHead
from refactoring.models.decoding.action_masking import make_attention_mask
from refactoring.models.decoding.constants import ROUTING_WEIGHT, ACTION_LOGITS_KEY, MoERoutingType, LATENT_KEY, BINARY_LOGITS_KEY, LATENT_CODES, \
    PREDICTED_ACTION_TOKENS_KEY, EXPERT_USAGE
from refactoring.models.decoding.decoders.factory.free_transformer import FreeTransformerDecoder
from refactoring.models.layers.activation import ActivationFunction


class MoEFreeTransformer(FreeTransformerDecoder):
    """A Mixture-of-Experts (MoE) action decoder utilizing the Free Transformer architecture.

    This decoder extends the Free Transformer by incorporating MoE action heads.
    It leverages the latent representations from the Free Transformer as gating signals
    to route inputs to multiple expert action heads.

    During the forward pass:
        1. The Free Transformer processes input features to produce action embeddings.
        2. Each MoE action head uses the latent layer outputs as routing weights to select experts.
        3. Each expert specializes in different aspects of action prediction.
    """

    def __init__(self,
                 *args,
                 num_experts: int,
                 gating_network_dims: list[int]|None=None,
                 routing_type: str = MoERoutingType.SOFT.value,
                 gating_activation: str = ActivationFunction.SILU.value,
                 top_k: int = 2,
                 expert_temperature: float = 1.0,
                 learnable_expert_temperature: bool = False,
                 gating_dropout: float = 0.1,
                 gating_normalization: bool = True,
                 **kwargs):
        """Initialize MoeFreeTransformer decoder.

        Args:
            *args, **kwargs: Arguments passed to the base FreeTransformer decoder.
        """
        self.num_experts = num_experts
        self.gating_network_dims = gating_network_dims
        self.routing_type = routing_type
        self.gating_activation = gating_activation
        self.top_k = top_k
        self.expert_temperature = expert_temperature
        self.learnable_expert_temperature = learnable_expert_temperature
        self.gating_dropout = gating_dropout
        self.gating_normalization = gating_normalization
        self.moe_head = None  # Will be set in set_tokenizer
        super().__init__(*args, **kwargs)


    def set_tokenizer(self, tokenizer: Tokenizer | None = None):
        super().set_tokenizer(tokenizer)  # Call base first (sets token_embedding, base action_heads, vocab_size)
        device = self.device
        vocab_size = self.vocab_size
        base_expert = self.action_heads[ACTION_LOGITS_KEY]
        self.moe_head = MoEHead(
            output_dim=vocab_size,
            device=device,
            base_expert=base_expert,
            num_experts=self.num_experts,
            gating_input_dim=self.free_transformer.embedding_dimension,
            gating_hidden_dims=self.gating_network_dims,
            gating_activation=self.gating_activation,
            routing_type=self.routing_type,
            top_k=self.top_k,
            temperature=self.expert_temperature,
            learnable_temperature=self.learnable_expert_temperature,
            gating_dropout=self.gating_dropout,
            gating_normalization=self.gating_normalization,
            gating_feature_key=None,
        )
        self.action_heads[ACTION_LOGITS_KEY] = self.moe_head  # Replaces base


    def _forward_training(
            self,
            actions: dict[str, torch.Tensor],
            feature_tokens: torch.Tensor,
            feature_token_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        prefix_len = feature_tokens.shape[1]
        target_token_ids = actions[TOKENIZED_ACTIONS_KEY]
        action_token_embeddings = self.token_embedding(target_token_ids)
        full_attention_mask = make_attention_mask(
            feature_tokens=feature_tokens,
            action_tokens=action_token_embeddings,
            feature_token_mask=feature_token_mask,
        )
        full_token_sequence = torch.cat([feature_tokens, action_token_embeddings], dim=1)
        if full_token_sequence.shape[1] > self.max_seq_len:
            raise ValueError(f"Input token length {full_token_sequence.shape[1]} > max_seq_len {self.max_seq_len}.")

        decoder_output, bit_logits, latent_codes, _ = self.free_transformer(
            hidden_states=full_token_sequence,
            key_padding_mask=feature_token_mask,
            decoder_cache=None,
            use_cache=False,
            self_attention_mask=full_attention_mask,
            is_inference=False
        )
        padding_action_mask = actions.get(IS_PAD_ACTION_KEY, None)
        action_outputs = decoder_output[:, prefix_len:, :]  # (B, action_len, emb_dim)
        latent_weights = self.free_transformer.latent_encoder(mid_features=action_outputs, mid_features_mask=padding_action_mask) # (B, action_len, latent_dim)
        logits_dict = self.moe_head(
            expert_features=action_outputs,
            gating_features=latent_weights
        )
        logits = logits_dict[ACTION_KEY]
        expert_usage = logits_dict[ROUTING_WEIGHT]
        return {
            ACTION_LOGITS_KEY: logits,
            BINARY_LOGITS_KEY: bit_logits,
            LATENT_CODES: latent_codes,
            ROUTING_WEIGHT: expert_usage,
        }


    def _forward_inference(
            self,
            feature_tokens: torch.Tensor,
            feature_token_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        batch_size = feature_tokens.shape[0]
        prefix_len = feature_tokens.shape[1]
        current_sequence = feature_tokens
        prefix_self_mask = torch.zeros(batch_size, 1, prefix_len, prefix_len, dtype=torch.bool, device=self.device)
        decoder_output, _, latent_codes,  decoder_cache = self.free_transformer(
            hidden_states=current_sequence,
            key_padding_mask=feature_token_mask,
            self_attention_mask=prefix_self_mask,
            decoder_cache=None,
            use_cache=True,
            is_inference=True
        )
        generated_tokens = []
        expert_usages = []
        next_token_embedding = None
        for step in range(self.max_seq_len - prefix_len):
            if step > 0:
                decoder_output, _, latent_codes, decoder_cache = self.free_transformer(
                    hidden_states=next_token_embedding,
                    key_padding_mask=feature_token_mask,
                    self_attention_mask=None, # Causal mask handled internally
                    decoder_cache=decoder_cache,
                    use_cache=True,
                    is_inference=True
                )
            last_output = decoder_output[:, -1:, :]  # (B, 1, embedding_dimension)
            latent_weights = self.free_transformer.latent_encoder(mid_features=last_output, mid_features_mask=None) # (B, 1, latent_dim)
            logits_dict = self.moe_head(
                expert_features=last_output,
                gating_features=latent_weights
            )
            logits = logits_dict[ACTION_KEY] #(B, 1, vocab_size)
            logits_scaled = logits / self.temperature.clamp(min=0.01)
            if self.deterministic:
                next_token = torch.argmax(logits, dim=-1)  # (B, 1)
            else:
                probs = torch.softmax(logits_scaled, dim=-1)
                next_token = torch.multinomial(probs.squeeze(-1), num_samples=1)  # (B, 1)
            expert_usage = logits_dict[ROUTING_WEIGHT]  # (B, 1, num_experts)
            expert_usages.append(expert_usage)
            next_token_embedding = self.token_embedding(next_token)  # (B, 1, embedding_dimension)
            generated_tokens.append(next_token)

        return {
            PREDICTED_ACTION_TOKENS_KEY: torch.cat(generated_tokens, dim=1),  # (B, max_seq_len)
            ROUTING_WEIGHT: torch.cat(expert_usages, dim=1),  # (B, max_seq_len, num_experts)
        }

