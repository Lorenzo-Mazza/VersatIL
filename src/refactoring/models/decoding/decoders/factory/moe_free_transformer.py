"""A MoE action decoder which utilizes the latent layer of the Free Transformer as gating for multiple action heads."""

import torch
from torch import nn

from refactoring.data.constants import TOKENIZED_ACTIONS_KEY, ACTION_KEY, IS_PAD_ACTION_KEY
from refactoring.data.tokenization import Tokenizer
from refactoring.models.decoding.action_heads import ActionHead
from refactoring.models.decoding.action_heads.moe import MoEHead
from refactoring.models.decoding.action_masking import make_attention_mask
from refactoring.models.decoding.constants import ROUTING_WEIGHT, ACTION_LOGITS_KEY, BINARY_LOGITS_KEY, LATENT_CODES, \
    PREDICTED_ACTION_TOKENS_KEY
from refactoring.models.decoding.decoders import ActionDecoder
from refactoring.models.decoding.decoders.factory.free_transformer import FreeTransformerDecoder
from refactoring.models.layers.swiglu import SwiGLU


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
                 **kwargs):
        """Initialize MoeFreeTransformer decoder.

        Args:
            *args, **kwargs: Arguments passed to the base FreeTransformer decoder.
        """
        super().__init__(*args,**kwargs)
        self.moe_action_head: MoEHead = self.action_heads[ACTION_LOGITS_KEY]
        self.expert_gating_projection = None


    def set_tokenizer(self, tokenizer: Tokenizer | None = None):
        if tokenizer is None or tokenizer.action_tokenizer is None:
            raise ValueError("FreeTransformerDecoder requires a tokenizer for tokenized action prediction.")
        device = self.temperature.device
        self.vocab_size = tokenizer.action_tokenizer.vocab_size
        self.moe_action_head.output_dim = self.vocab_size
        token_input_embedding = nn.Embedding(self.vocab_size, self.embedding_dimension).to(device)
        nn.init.normal_(token_input_embedding.weight, mean=0.0, std=self.free_transformer.initializer_range)
        self.token_embedding = token_input_embedding
        for expert in self.moe_action_head.experts:
            expert:ActionHead
            output_block_in_features = expert.output_proj.in_features
            expert_out = nn.Linear(output_block_in_features, self.vocab_size, bias=True, device=device)
            nn.init.kaiming_uniform_(expert_out.weight, nonlinearity='linear')
            nn.init.zeros_(expert_out.bias)
            expert.output_dim = self.vocab_size
            expert.output_proj = expert_out  # Replace final projection with expert head
        expert_gating_projection = nn.Linear(
            self.free_transformer.embedding_dimension,
            self.moe_action_head.num_experts,
            bias=False,
            device=device
        )
        nn.init.normal_(expert_gating_projection.weight, mean=0.0, std=self.free_transformer.initializer_range)
        self.expert_gating_projection = expert_gating_projection
        ActionDecoder.set_tokenizer(self, tokenizer)  # Call action decoder base, free transformer base would raise error



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

        decoder_output, bit_logits, latent_codes, latent_embeddings, _ = self.free_transformer(
            hidden_states=full_token_sequence,
            key_padding_mask=feature_token_mask,
            decoder_cache=None,
            use_cache=False,
            self_attention_mask=full_attention_mask,
            is_inference=False,
            return_latent_embeddings=True,
        )
        latent_action_embeddings = latent_embeddings[:, prefix_len:, :]  # (B, action_len, emb_dim)
        action_outputs = decoder_output[:, prefix_len:, :]  # (B, action_len, emb_dim)
        latent_weights = self.expert_gating_projection(latent_action_embeddings) # (B, action_len, num_experts)
        routing_weights = torch.softmax(latent_weights, dim=-1) # (B, action_len, num_experts)
        logits_dict = self.moe_action_head(
            features=action_outputs,
            routing_weights=routing_weights
        )
        logits = logits_dict[ACTION_KEY]
        expert_usage = logits_dict[ROUTING_WEIGHT]
        return {
            ACTION_LOGITS_KEY: logits,
            BINARY_LOGITS_KEY: bit_logits,
            LATENT_CODES: latent_codes,
            f"{ACTION_LOGITS_KEY}_{ROUTING_WEIGHT}": expert_usage,
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
        decoder_output, _, latent_codes, latent_embeddings, decoder_cache = self.free_transformer(
            hidden_states=current_sequence,
            key_padding_mask=feature_token_mask,
            self_attention_mask=prefix_self_mask,
            decoder_cache=None,
            use_cache=True,
            is_inference=True,
            return_latent_embeddings=True,
        )
        generated_tokens = []
        expert_usages = []
        next_token_embedding = None
        for step in range(self.max_seq_len - prefix_len):
            if step > 0:
                decoder_output, _, latent_codes, latent_embeddings, decoder_cache = self.free_transformer(
                    hidden_states=next_token_embedding,
                    key_padding_mask=feature_token_mask,
                    self_attention_mask=None, # Causal mask handled internally
                    decoder_cache=decoder_cache,
                    use_cache=True,
                    is_inference=True
                )
            last_output = decoder_output[:, -1:, :]  # (B, 1, embedding_dimension)
            latent_action_embeddings = latent_embeddings[:, -1:, :]  # (B, 1, emb_dim)
            latent_weights = self.expert_gating_projection(latent_action_embeddings)  # (B, 1, num_experts)
            routing_weights = torch.softmax(latent_weights, dim=-1) # (B, 1, num_experts)
            logits_dict = self.moe_action_head(
                features=last_output,
                routing_weights=routing_weights
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
            f"{ACTION_LOGITS_KEY}_{ROUTING_WEIGHT}": torch.cat(expert_usages, dim=1),  # (B, max_seq_len, num_experts)
        }

