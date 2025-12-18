import torch
from torch import nn


class DynamicFeatureEmbedding(nn.Module):
    """Learned embeddings for features, created on-demand at runtime."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.ParameterDict()


    def forward(self, name: str, device: torch.device) -> torch.Tensor:
        """Get or create a learned embedding for the given feature name."""
        key = name.replace(".", "_")
        if key not in self.embeddings:
            self.embeddings[key] = nn.Parameter(
                torch.randn(1, 1, self.embedding_dim, device=device) * 0.02
            )
        return self.embeddings[key]