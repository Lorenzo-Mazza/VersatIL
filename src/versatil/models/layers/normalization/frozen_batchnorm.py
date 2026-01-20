"""Frozen BatchNorm2d implementation taken from DETR."""
import torch


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    """

    def __init__(self, dimension: int):
        """Initialize with dimension equal to the channel dimension."""
        super().__init__()
        self.register_buffer("weight", torch.ones(dimension))
        self.register_buffer("bias", torch.zeros(dimension))
        self.register_buffer("running_mean", torch.zeros(dimension))
        self.register_buffer("running_var", torch.ones(dimension))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor):
        """Forward pass for input tensor with shape (B, C, H, W)."""
        w = self.weight.reshape(1, -1, 1, 1)  # Shape: (1, C, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
