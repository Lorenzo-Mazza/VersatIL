import enum

from torch import nn


class ActivationFunction(str, enum.Enum):
    """Available activation functions."""
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    LEAKY_RELU = "leaky_relu"
    LINEAR = "linear"
    MISH = "mish"

    def to_torch_activation(self) -> type[nn.Module]:
        """Convert to corresponding PyTorch activation module."""
        ACTIVATION_MAPPING = {
            ActivationFunction.RELU.value: nn.ReLU,
            ActivationFunction.GELU.value: nn.GELU,
            ActivationFunction.SILU.value: nn.SiLU,
            ActivationFunction.TANH.value: nn.Tanh,
            ActivationFunction.SIGMOID.value: nn.Sigmoid,
            ActivationFunction.LEAKY_RELU.value: nn.LeakyReLU,
            ActivationFunction.LINEAR.value: nn.Identity,
            ActivationFunction.MISH.value: nn.Mish,
        }
        if self.value not in ACTIVATION_MAPPING:
            raise ValueError(
                f"Unsupported activation: {self.value}. "
                f"Choose from {[e.value for e in ActivationFunction]}"
            )
        return ACTIVATION_MAPPING[self.value]

