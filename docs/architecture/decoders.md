# Action Decoders

Action decoders are the neural network architectures that transform encoded observation features into action predictions. All decoders inherit from [`ActionDecoder`][versatil.models.decoding.decoders.base.ActionDecoder] and are paired with an algorithm at runtime -- the decoder defines **what** network processes the features, while the algorithm defines **how** training and inference are orchestrated.

```python
class ActionDecoder(nn.Module, ABC):
    supports_tokenized_actions: bool = False

    def __init__(
        self,
        decoder_input: DecoderInput,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        action_heads: dict,
        device: str,
        observation_horizon: int,
        prediction_horizon: int,
    ): ...

    @abstractmethod
    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]: ...
```

## [`DecoderInput`][versatil.models.decoding.decoders.base.DecoderInput] Specification

Each decoder declares its input requirements via a [`DecoderInput`][versatil.models.decoding.decoders.base.DecoderInput] dataclass. This enables validation during experiment setup -- [`ExperimentValidator`][versatil.validation.ExperimentValidator] checks that the encoding pipeline produces compatible features before training begins.

```python
@dataclass
class DecoderInput:
    keys: list[str]                                        # Feature keys required by the decoder
    required_types: list[str] = []                         # Must have at least one of these feature types
    raises_for_types: list[str] = []                       # Reject these feature types
    requires_actions: bool = False                         # Whether forward() needs ground-truth actions
    requires_vlm_backbone: bool = False                    # Whether the decoder needs VLM layers (Pi0/SmolVLA)
    conditioning_key: str | None = None                    # For conditional decoders
    conditioning_required: list[str] = []                  # Required conditioning keys
    conditioning_one_of_groups: list[list[str]] = []       # Exactly one from each group required
```

Feature types are `"spatial"` (C, H, W), `"sequential"` (S, D), or `"flat"` (D,).

---

## Available Decoders

### Non-Autoregressive Transformer Decoders

These decoders predict all action timesteps in parallel using bidirectional or DETR-style attention.

| Decoder | Class | Description |
|---------|-------|-------------|
| **[`ActionTransformer`][versatil.models.decoding.decoders.factory.action_transformer.ActionTransformer]** | [`ActionTransformer`][versatil.models.decoding.decoders.factory.action_transformer.ActionTransformer] | Bidirectional transformer decoder with cross-attention to observation tokens. Configurable positional encoding, normalization, and activation. |
| **[`ACT`][versatil.models.decoding.decoders.factory.act.ACT]** | [`ACT`][versatil.models.decoding.decoders.factory.act.ACT] | Action Chunking Transformer ([Zhao et al., 2023](https://arxiv.org/abs/2304.13705)). DETR-style non-autoregressive decoder with learnable queries. Optionally accepts a latent embedding from [`VariationalAlgorithm`][versatil.models.decoding.algorithm.variational.VariationalAlgorithm]. |
| **[`LACT`][versatil.models.decoding.decoders.factory.lact.LACT]** | [`LACT`][versatil.models.decoding.decoders.factory.lact.LACT] | Latent Action Transformer. Extends [`ActionTransformer`][versatil.models.decoding.decoders.factory.action_transformer.ActionTransformer] with latent-conditioned decoding via AdaLN or FiLM modulation at each layer. Uses PixArt-style cross-attention DiT architecture. |
| **[`PhaseACT`][versatil.models.decoding.decoders.factory.phase_act.PhaseACT]** | [`PhaseACT`][versatil.models.decoding.decoders.factory.phase_act.PhaseACT] | Phase-aware ACT with surgical phase prediction ([Mazza et al., 2026](https://arxiv.org/abs/2601.21971)). Extends [`ACT`][versatil.models.decoding.decoders.factory.act.ACT] with a phase classifier head whose predictions route through MoE action heads. |
| **MODE-ACT** | [`MixtureOfDensitiesActionTransformer`][versatil.models.decoding.decoders.factory.mode_act.MixtureOfDensitiesActionTransformer] | Mixture Density Network Transformer. Predicts K mixture components per action using multiple Gaussian expert heads and a mode query token for routing. |

### Tokenized Decoders

These decoders operate on discrete tokenized actions. Autoregressive decoders (GPT, Free, MoE Free) generate action tokens sequentially, while non-autoregressive decoders (DiscreteDETR) predict all tokens in parallel.

| Decoder | Class | Description |
|---------|-------|-------------|
| **[`GPTActionTransformer`][versatil.models.decoding.decoders.factory.gpt_action_transformer.GPTActionTransformer]** | [`GPTActionTransformer`][versatil.models.decoding.decoders.factory.gpt_action_transformer.GPTActionTransformer] | Autoregressive GPT-style decoder with self-attention only. Observation features are concatenated as prefix tokens, followed by action token embeddings. Inspired by [Pi0-FAST](https://www.physicalintelligence.company/blog/pi0-fast). |
| **[`FreeActionTransformer`][versatil.models.decoding.decoders.factory.free_action_transformer.FreeActionTransformer]** | [`FreeActionTransformer`][versatil.models.decoding.decoders.factory.free_action_transformer.FreeActionTransformer] | Autoregressive Free Transformer ([Fleuret, 2025](https://arxiv.org/abs/2510.17558)). Encodes trajectory style/mode in a discrete latent variable and generates action tokens sequentially. |
| **[`MoEFreeActionTransformer`][versatil.models.decoding.decoders.factory.moe_free_action_transformer.MoEFreeActionTransformer]** | [`MoEFreeActionTransformer`][versatil.models.decoding.decoders.factory.moe_free_action_transformer.MoEFreeActionTransformer] | Autoregressive. Extends [`FreeActionTransformer`][versatil.models.decoding.decoders.factory.free_action_transformer.FreeActionTransformer] with MoE action heads. Uses the Free Transformer's latent layer outputs as gating signals to route through multiple expert heads. |
| **[`DiscreteDETRActionTransformer`][versatil.models.decoding.decoders.factory.discrete_detr_action_transformer.DiscreteDETRActionTransformer]** | [`DiscreteDETRActionTransformer`][versatil.models.decoding.decoders.factory.discrete_detr_action_transformer.DiscreteDETRActionTransformer] | Non-autoregressive DETR-style decoder ([Carion et al., 2020](https://arxiv.org/abs/2005.12872)) adapted for tokenized action prediction. Uses [`ACT`][versatil.models.decoding.decoders.factory.act.ACT]-style parallel decoding with discrete action tokens. |

!!! note "Tokenized decoders"
    Decoders with `supports_tokenized_actions = True` require a [`Tokenizer`][versatil.data.tokenization.tokenizer.Tokenizer] to be set via `set_tokenizer()`. The tokenizer maps continuous actions to discrete vocabulary indices. Both the decoder **and** the algorithm must support tokenization.

### Diffusion / Flow Matching Decoders

These decoders are designed for iterative denoising algorithms. They accept noisy actions and timestep conditioning as input.

| Decoder | Class | Description |
|---------|-------|-------------|
| **[`ConditionalActionUNet`][versatil.models.decoding.decoders.factory.conditional_action_unet.ConditionalActionUNet]** | [`ConditionalActionUNet`][versatil.models.decoding.decoders.factory.conditional_action_unet.ConditionalActionUNet] | U-Net decoder for Diffusion Policy ([Chi et al., 2023](https://arxiv.org/abs/2303.04137)). Uses FiLM conditioning from pooled observation features. Accepts global and optional local (sequence-aligned) conditioning. |
| **[`DiTBlockActionTransformer`][versatil.models.decoding.decoders.factory.dit_block_action_transformer.DiTBlockActionTransformer]** | [`DiTBlockActionTransformer`][versatil.models.decoding.decoders.factory.dit_block_action_transformer.DiTBlockActionTransformer] | DiT-Block Policy ([Block et al., 2024](https://arxiv.org/html/2410.10088v1)). Processes observation tokens through an encoder with mean pooling, then conditions the decoder via AdaLN (pooled vector + timestep embedding). Supports encoder caching during inference. |
| **[`DiffusionActionTransformer`][versatil.models.decoding.decoders.factory.diffusion_action_transformer.DiffusionActionTransformer]** | [`DiffusionActionTransformer`][versatil.models.decoding.decoders.factory.diffusion_action_transformer.DiffusionActionTransformer] | Diffusion action transformer supporting two sub-architectures: **[`CrossAttentionDiT`][versatil.models.layers.diffusion_transformer.cross_attention_dit.CrossAttentionDiT]** (PixArt-style cross-attention to unpooled observation tokens) and **MMDiT** (SD3-style joint attention between observation and action streams). |

### VLA (Vision-Language-Action) Decoders

These decoders borrow pretrained layers from a generative VLM encoder and pair them with learned expert layers for interleaved processing. They require `requires_vlm_backbone=True` in their [`DecoderInput`][versatil.models.decoding.decoders.base.DecoderInput] and implement `set_backbone()` to receive the VLM layers at initialization.

| Decoder | Class | Description |
|---------|-------|-------------|
| **Pi0** | [`Pi0Decoder`][versatil.models.decoding.decoders.factory.pi0.Pi0Decoder] | Interleaved VLM-expert joint attention. Each VLM layer is paired 1:1 with an expert layer. Pi0 fuses timestep via concat-MLP; Pi0.5 modulates via adaptive normalization. |
| **SmolVLA** | [`SmolVLADecoder`][versatil.models.decoding.decoders.factory.smolvla.SmolVLADecoder] | Alternates between joint self-attention (expert attends alongside VLM tokens) and cross-attention (expert attends to VLM key/values) layers. |

References: [Pi0](https://arxiv.org/abs/2410.24164), [Pi0.5](https://arxiv.org/abs/2504.16054), [SmolVLA](https://arxiv.org/abs/2506.01844).

Both decoders accept noisy actions and timestep conditioning, making them compatible with generative algorithms (Flow Matching, Diffusion).

#### [`Pi0Decoder`][versatil.models.decoding.decoders.factory.pi0.Pi0Decoder]

Expert layers are fully configurable (hidden size, intermediate size, heads, K/V heads, head dimension). The number of expert layers must match the VLM layer count.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `time_conditioning` | `"concat_mlp"` | Timestep fusion mode: `"concat_mlp"` (Pi0) or `"adanorm"` (Pi0.5) |
| `expert_hidden_size` | -- | Expert network hidden dimension |
| `expert_number_of_layers` | -- | Must match VLM layer count |
| `proprioceptive_feature_key` | `None` | Proprioceptive feature prepended to VLM prefix |

#### [`SmolVLADecoder`][versatil.models.decoding.decoders.factory.smolvla.SmolVLADecoder]

Expert dimensions are derived from the VLM via `expert_width_multiplier`. The layer routing alternates between joint self-attention and cross-attention at a configurable period.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `expert_width_multiplier` | `0.75` | Expert hidden size as fraction of VLM hidden size |
| `num_expert_layers` | `-1` | Number of expert layers (`-1` matches VLM count) |
| `num_vlm_layers` | `16` | VLM layers to use (truncates if fewer available) |
| `self_attention_every_n_layers` | `2` | Period for joint self-attention (`0` = all cross-attention) |
| `freeze_vlm` | `True` | Whether to freeze VLM layer parameters |
| `proprioceptive_feature_key` | `None` | Proprioceptive feature prepended to VLM prefix |

!!! info "VLM backbone wiring"
    When `requires_vlm_backbone=True`, [`Policy`][versatil.models.policy.Policy] automatically calls `decoder.set_backbone()` at initialization, passing the VLM encoder's transformer layers, rotary embedding, hidden dimension, and text config. The VLM encoder must use `use_embeddings_only=True` so its LM layers remain available for the decoder.

!!! info "Action masking for VLA"
    VLA decoders use `make_attention_mask()` with `causal_actions` and `causal_prefix_suffix_length` parameters to construct prefix-suffix attention patterns where the prefix (observations) is bidirectional and the suffix (actions) is optionally causal.

### MoE Decoder Wrapper

[`MoEDecoder`][versatil.models.decoding.decoders.moe.MoEDecoder] is a general-purpose Mixture of Experts wrapper applicable on top of **any** [`ActionDecoder`][versatil.models.decoding.decoders.base.ActionDecoder]. It deep-copies a base decoder into `num_experts` independent experts and learns a gating network that routes inputs based on a specified feature key. Supports soft routing (weighted sum) and top-k routing.

---

## Action Heads

Action heads are the final projection layers that convert decoder embeddings into action predictions. Each decoder has one action head per action component (position, orientation, gripper).

### [`ActionHead`][versatil.models.decoding.action_heads.single_output.ActionHead] (Single Output)

The default head. A linear projection from decoder embedding dimension to action dimension, optionally preceded by composable blocks.

```python
class ActionHead(BaseActionHead):
    def forward(self, action_embedding: torch.Tensor) -> torch.Tensor:
        # (B, prediction_horizon, embedding_dim) -> (B, prediction_horizon, action_dim)
```

### [`GaussianHead`][versatil.models.decoding.action_heads.gaussian.GaussianHead]

Outputs Gaussian distribution parameters (mean and log-variance) instead of point predictions. Used by [`MixtureOfDensitiesActionTransformer`][versatil.models.decoding.decoders.factory.mode_act.MixtureOfDensitiesActionTransformer] and other mixture density approaches.

```python
class GaussianHead(BaseActionHead):
    def forward(self, action_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        # Returns {"mean": ..., "logvar": ...}
```

The `logvar` output is clamped between `min_logvar` (default -10.0) and `max_logvar` (default 4.0) for training stability.

### [`MoEHead`][versatil.models.decoding.action_heads.moe.MoEHead] (Mixture of Experts)

Wraps multiple expert [`ActionHead`][versatil.models.decoding.action_heads.single_output.ActionHead] instances with a learned gating network. Each expert produces an independent action prediction, and the final output is a weighted combination.

```python
class MoEHead(BaseMixtureOfExperts):
    def forward(
        self,
        features: torch.Tensor,
        gating_feature: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Returns {"action": ..., "routing_weights": ..., "expert_outputs": ...}
```

**Initialization modes:**

| Mode | Usage |
|------|-------|
| Explicit expert list | Pass pre-instantiated `experts` |
| Base expert cloning | Pass `base_expert` + `num_experts` (deep-copies with re-initialized weights) |
| Lazy initialization | Pass `base_expert` only; call `set_num_experts()` later (used by [`PhaseACT`][versatil.models.decoding.decoders.factory.phase_act.PhaseACT]) |

**Routing strategies:** `"soft"` (weighted sum of all experts) or `"top_k"` (select top-k experts).

### Action Head Blocks

Action heads support composable building blocks inserted before the final projection:

| Block | Description |
|-------|-------------|
| [`MLPBlock`][versatil.models.decoding.action_heads.blocks.MLPBlock] | LayerNorm + MLP with configurable hidden dims, activation, and dropout |
| [`AttentionBlock`][versatil.models.decoding.action_heads.blocks.AttentionBlock] | Self-attention with residual connection across the prediction horizon |
| [`ResidualBlock`][versatil.models.decoding.action_heads.blocks.ResidualBlock] | Wraps any block with a residual connection |

```python
ActionHead(
    input_dim=256,
    blocks=[
        MLPBlock(input_dim=256, hidden_dims=[512], output_dim=256),
        AttentionBlock(embedding_dimension=256, num_heads=8),
    ],
)
```

---

## Positional Encoding

All transformer decoder factories follow a unified positional encoding (PE) pattern when using the transformer package (`versatil.models.layers.transformer`).

### How it works

1. **[`TransformerInputBuilder`][versatil.models.decoding.transformer_input_builder.TransformerInputBuilder]** projects input features to a common embedding dimension and computes additive PE: 2D sinusoidal for spatial features, 1D sinusoidal for flat/sequential features. When the observation horizon is greater than 1, an additional learned temporal PE layer is added so that the policy receives distinct temporal representations across timesteps. The builder returns `(input_tokens, pos_encodings, padding_mask)`.
2. **Always pre-add** the PE to tokens before the transformer call: `hidden_states = input_tokens + pos_encodings`. This ensures cross-attention keys carry absolute position information regardless of the transformer's internal PE setting.
3. **`positional_encoding_type`** on the transformer controls self-attention PE only:
    - `None`: no internal PE. Position info comes entirely from the pre-added additive PE.
    - `rope`: RoPE (rotary) applied to Q/K inside self-attention layers, complementing the pre-added additive PE.
    - `sinusoidal` or `learned`: an additional absolute PE is applied inside the transformer. In the context of action decoders this is redundant with the pre-added PE from step 1, but it preserves the self-contained correctness of the transformer module when used independently.
4. **Cross-attention** never applies RoPE. Keys receive position info solely from the pre-added additive PE. This avoids position-space collisions between query and key sequences from different modalities.


### Exceptions

Two decoder families intentionally diverge from the contract above. They are correct by design within their own attention scheme; the unified contract does not apply.

**DETR-style decoders** ([`ACT`][versatil.models.decoding.decoders.factory.act.ACT], [`DiscreteDETRActionTransformer`][versatil.models.decoding.decoders.factory.discrete_detr_action_transformer.DiscreteDETRActionTransformer]) use `versatil.models.layers.detr_transformer.Transformer`, which takes `source_positional_encoding` as a separate argument and re-adds it at every attention layer (the original DETR pattern). They do **not** pre-add PE to the tokens before the transformer call. New decoders should follow the standard pre-add contract; the DETR variant is preserved for parity with the published architectures.

**VLA decoders** ([`Pi0Decoder`][versatil.models.decoding.decoders.factory.pi0.Pi0Decoder], [`SmolVLADecoder`][versatil.models.decoding.decoders.factory.smolvla.SmolVLADecoder]) bypass [`TransformerInputBuilder`][versatil.models.decoding.transformer_input_builder.TransformerInputBuilder] entirely. They consume pre-embedded VLM prefix tokens and apply RoPE jointly across the concatenated `[prefix, action]` sequence using position IDs derived from a shared validity mask (see `pi0.py:321-328`, `smolvla.py:380-386`). This means RoPE *is* used across what would normally be cross-attention boundaries — it works because the VLM and expert layers share the same position space by construction. If a VLA decoder is ever paired with a non-VLM encoder whose outputs do not carry positional information, the prefix keys will be position-blind and the contract breaks; document this explicitly in any new VLA-style decoder.

---

## Adding a New Decoder

### 1. Implement the decoder class

Create a new file in `src/versatil/models/decoding/decoders/factory/`:

```python
from versatil.models.decoding.decoders.base import ActionDecoder, DecoderInput
from versatil.models.feature_meta import FeatureType


class MyDecoder(ActionDecoder):

    def __init__(
        self,
        input_keys: list[str],
        action_space: ActionSpace,
        action_heads: dict[str, ActionHead],
        observation_space: ObservationSpace,
        observation_horizon: int,
        prediction_horizon: int,
        device: str,
        # ... custom parameters
    ):
        decoder_input = DecoderInput(
            keys=input_keys,
            required_types=[FeatureType.FLAT.value],
            requires_actions=False,
        )
        super().__init__(
            decoder_input=decoder_input,
            observation_space=observation_space,
            action_space=action_space,
            action_heads=action_heads,
            device=device,
            observation_horizon=observation_horizon,
            prediction_horizon=prediction_horizon,
        )
        # Build your architecture here

    def forward(
        self,
        features: dict[str, torch.Tensor],
        actions: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        # Process features, apply action heads, return predictions
        ...
```

### 2. Define the config dataclass

Add a config in `src/versatil/configs/decoding/decoder.py`:

```python
from dataclasses import dataclass

from versatil.configs.decoding.decoder import DecodingNetworkConfig


@dataclass
class MyDecoderConfig(DecodingNetworkConfig):
    _target_: str = "versatil.models.decoding.decoders.factory.my_decoder.MyDecoder"
    # ... custom parameters with defaults
```

### 3. Register in ConfigStore

Add the config to the Hydra ConfigStore in `src/versatil/configs/__init__.py`.

### 4. Create YAML config

Add a reusable config in `hydra_configs/policy/decoder/`:

```yaml
# hydra_configs/policy/decoder/my_decoder.yaml
defaults:
  - heads@action_heads: bowel_retraction

input_keys:
  - left_rgb
# ... parameter overrides
```

### 5. Add tests

Create tests in `tests/models/decoding/` following the patterns in `tests/CLAUDE.md`.

### 6. Checklist for new decoders

When implementing a new decoder factory:

- Always call `self.input_sequence_builder(features)` and pre-add the returned `pos_encodings` to the tokens.
- Accept `positional_encoding_type` as an `__init__` parameter and pass it through to the transformer constructor.
- Do **not** hardcode `positional_encoding_type=None` — let the user configure it via Hydra.

---

## Source Locations

| Component | Path |
|-----------|------|
| [`ActionDecoder`][versatil.models.decoding.decoders.base.ActionDecoder] base class | `src/versatil/models/decoding/decoders/base.py` |
| [`DecoderInput`][versatil.models.decoding.decoders.base.DecoderInput] | `src/versatil/models/decoding/decoders/base.py` |
| Decoder factories | `src/versatil/models/decoding/decoders/factory/` |
| [`ActionHead`][versatil.models.decoding.action_heads.single_output.ActionHead] | `src/versatil/models/decoding/action_heads/single_output.py` |
| [`GaussianHead`][versatil.models.decoding.action_heads.gaussian.GaussianHead] | `src/versatil/models/decoding/action_heads/gaussian.py` |
| [`MoEHead`][versatil.models.decoding.action_heads.moe.MoEHead] | `src/versatil/models/decoding/action_heads/moe.py` |
| Action head blocks | `src/versatil/models/decoding/action_heads/blocks.py` |
| Decoder configs | `src/versatil/configs/decoding/decoder.py` |
