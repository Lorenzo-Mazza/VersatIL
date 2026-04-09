from dataclasses import dataclass

import torch


@dataclass
class ConditioningLayerCache:
    """Per-layer K/V cache for static conditioning (computed once, reused).

    Stores precomputed projections for conditioning information that does not
    change between forward calls. Populated once before generation starts.

    Attributes:
        keys: Precomputed keys (B, kv_heads, S, head_dim).
        values: Precomputed values (B, kv_heads, S, head_dim).
        queries: Precomputed queries (B, heads, S, head_dim). Populated when
            conditioning participates bidirectionally (both streams attend to
            each other). None when conditioning is only attended to.
    """

    keys: torch.Tensor
    values: torch.Tensor
    queries: torch.Tensor | None = None


@dataclass
class ConditioningCache:
    """Multi-layer conditioning cache for static context.

    Attributes:
        layers: One ConditioningLayerCache per decoder layer.
    """

    layers: list[ConditioningLayerCache]
