import enum

from torch import nn

from versatil.models.layers.gated_linear_unit import GeGLU, SwiGLU


class ActivationFunction(enum.StrEnum):
    """Available activation functions."""

    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    LINEAR = "linear"
    MISH = "mish"

    @property
    def is_gated(self) -> bool:
        """Whether this activation uses a gated linear unit (two projections)."""
        return self in (ActivationFunction.SWIGLU, ActivationFunction.GEGLU)

    def to_torch_activation(self) -> type[nn.Module]:
        """Convert to corresponding PyTorch activation module.

        Gated activations (SwiGLU, GeGLU) require ``input_dim`` and
        ``hidden_dim`` constructor args. Standard activations are zero-arg.
        """
        mapping = {
            ActivationFunction.RELU.value: nn.ReLU,
            ActivationFunction.GELU.value: nn.GELU,
            ActivationFunction.SILU.value: nn.SiLU,
            ActivationFunction.SWIGLU.value: SwiGLU,
            ActivationFunction.GEGLU.value: GeGLU,
            ActivationFunction.SIGMOID.value: nn.Sigmoid,
            ActivationFunction.TANH.value: nn.Tanh,
            ActivationFunction.LEAKY_RELU.value: nn.LeakyReLU,
            ActivationFunction.LINEAR.value: nn.Identity,
            ActivationFunction.MISH.value: nn.Mish,
        }
        if self.value not in mapping:
            raise ValueError(
                f"Unsupported activation: {self.value}. "
                f"Choose from {[e.value for e in ActivationFunction]}"
            )
        return mapping[self.value]
