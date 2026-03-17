import torch

from versatil.models.encoding.encoders.base import EncoderInput, EncoderOutput
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.layers.activation import ActivationFunction
from versatil.models.layers.mlp import MLP


class ProprioceptiveEncoder(Encoder):
    """Encoding proprioceptive state (robot joint positions, velocities, etc.) with a Feedforward Fully-Connected NN."""

    def __init__(
        self,
        input_keys: str | list[str],
        output_dim: int,
        hidden_dims: list[int] | None = None,
        activation: str = ActivationFunction.RELU.value,
        dropout: float = 0.0,
        pretrained: bool = False,
        frozen: bool = False,
    ):
        """Initialize proprioceptive encoder.

        Args:
            input_keys: Keys for proprioceptive inputs
            output_dim: Output feature dimension
            hidden_dims: Hidden layer dimensions. If None or [], creates simple linear layer.
                        If [128], creates one hidden layer. If [256, 128], creates two hidden layers.
            activation: Activation function from ActivationFunction enum
            dropout: Dropout rate between layers
            pretrained: Whether to use pretrained weights (unused for proprio encoder)
            frozen: Whether to freeze encoder weights
        """
        specification = EncoderInput(keys=input_keys)
        super().__init__(
            input_specification=specification, pretrained=pretrained, frozen=frozen
        )
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation_fn = ActivationFunction(activation).to_torch_activation()
        self.network: MLP | None = None
        self.frozen = frozen

    def _build_network(self, input_dim: int):
        """Build MLP network."""
        self.network = MLP(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation_function=self.activation_fn,
            dropout=self.dropout,
        )
        if self.frozen:
            super()._freeze_weights()

    def forward(
        self,
        inputs: dict[str, torch.Tensor],
        is_train: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Forward pass to encode proprioceptive state.

        Args:
            inputs: Dict with state tensors
            is_train: Whether in training mode

        Returns:
            Dict containing PROPRIO_FEATURES key with encoded features.
            Shape: (batch_size, output_dim) or (batch_size, time_steps, output_dim)
        """
        # Concatenate the whole proprioceptive data along the last dimension, to concatenate e.g. robot frame obs and camera frame obs
        state = torch.cat(
            [inputs[key] for key in self.input_specification.keys], dim=-1
        )
        input_dim = state.shape[-1]
        batch_size = state.shape[0]
        time_steps = None
        # Build network if not already built, and move to same device as input
        if self.network is None:
            self._build_network(input_dim)
            if self.network is None:
                raise RuntimeError("Network should be built by _build_network")
            self.network = self.network.to(state.device)
        has_time = False
        if state.dim() == 3:
            time_steps = state.shape[1]
            state = state.reshape(batch_size * time_steps, -1)
            has_time = True
        if self.network is None:
            raise RuntimeError("network must be built before forward pass")
        features = self.network(state)
        if has_time:
            features = features.reshape(batch_size, time_steps, -1)
        return {EncoderOutputKeys.PROPRIOCEPTIVE.value: features}

    def get_output_dims(self) -> dict[str, int]:
        """Get output dimensions."""
        return {EncoderOutputKeys.PROPRIOCEPTIVE.value: self.output_dim}

    def get_output_specification(self) -> EncoderOutput:
        return EncoderOutput(
            features=[EncoderOutputKeys.PROPRIOCEPTIVE.value],
            dimensions={EncoderOutputKeys.PROPRIOCEPTIVE.value: self.output_dim},
        )
