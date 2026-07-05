import torch
from torch import nn

from versatil.common.module_attr_mixin import ModuleAttrMixin


class DynamicFeatureEmbedding(ModuleAttrMixin):
    """Learned embeddings for features, created on-demand at runtime.

    This module uses lazy initialization - embeddings are created on first access.
    To support loading checkpoints, it overrides _load_from_state_dict to create
    embeddings dynamically from the state dict.
    """

    def __init__(self, embedding_dimension: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.embeddings = nn.ParameterDict()

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        error_msgs: list,
    ) -> None:
        """Load state dict with dynamic embedding creation.

        This method creates embedding parameters on-the-fly from checkpoint values,
        enabling proper loading even when embeddings don't exist yet due to lazy initialization.
        """
        embeddings_prefix = prefix + "embeddings."
        device = self.device
        dtype = self.dtype
        for key, value in state_dict.items():
            if key.startswith(embeddings_prefix):
                feature_name = key[len(embeddings_prefix) :]
                if feature_name not in self.embeddings:
                    # Create parameter with correct shape on the correct device
                    self.embeddings[feature_name] = nn.Parameter(
                        torch.zeros(value.shape, device=device, dtype=dtype)
                    )
        # Now parent can load weights into the newly created embeddings
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, name: str, device: torch.device) -> torch.Tensor:
        """Get or create a learned embedding for the given feature name."""
        key = name.replace(".", "_")
        if key not in self.embeddings:
            self.embeddings[key] = nn.Parameter(
                torch.randn(
                    1,
                    1,
                    self.embedding_dimension,
                    device=device,
                    dtype=self.dtype,
                )
                * 0.02
            )
        return self.embeddings[key]
