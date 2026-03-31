"""MLP-based proprioceptive state encoder."""

import torch

from versatil.data.metadata import BaseMetadata, CameraMetadata
from versatil.models.encoding.encoders.base import EncoderInput
from versatil.models.encoding.encoders.constants import EncoderOutputKeys
from versatil.models.encoding.encoders.unconditional import Encoder
from versatil.models.feature_meta import FeatureMetadata, FeatureType
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
        model_dtype: str | None = None,
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
            model_dtype: Precision string from experiment config (e.g. ``"bf16-mixed"``).
        """
        specification = EncoderInput(keys=input_keys)
        super().__init__(
            input_specification=specification,
            pretrained=pretrained,
            frozen=frozen,
            model_dtype=model_dtype,
        )
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation_fn = ActivationFunction(activation).to_torch_activation()
        self.network: MLP | None = None

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
        """Build the MLP from checkpoint weights before loading."""
        network_prefix = prefix + "network."
        has_network_keys = any(k.startswith(network_prefix) for k in state_dict)
        if has_network_keys and self.network is None:
            first_weight = state_dict[network_prefix + "layers.0.weight"]
            self._build_network(input_dim=first_weight.shape[1])
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Encode proprioceptive state.

        Args:
            inputs: Dict with state tensors, each as (B, D).

        Returns:
            Dict with proprioceptive features.
        """
        state = torch.cat(
            [inputs[key] for key in self.input_specification.keys], dim=-1
        )
        input_dimension = state.shape[-1]
        if self.network is None:
            self._build_network(input_dimension)
            if self.network is None:
                raise RuntimeError("Network should be built by _build_network")
            self.network = self.network.to(state.device)
        features = self.network(state)
        return {EncoderOutputKeys.PROPRIOCEPTIVE.value: features}

    def get_output_dims(self) -> dict[str, int]:
        """Get output dimensions."""
        return {EncoderOutputKeys.PROPRIOCEPTIVE.value: self.output_dim}

    def validate_input_metadata(self, key: str, metadata: BaseMetadata) -> str | None:
        """Validate that input metadata is not camera metadata.

        Args:
            key: Observation key being validated.
            metadata: Metadata from the observation space.

        Returns:
            Error message if incompatible, None if valid.
        """
        if isinstance(metadata, CameraMetadata):
            return (
                f"ProprioceptiveEncoder cannot process image data for '{key}'. "
                f"Got CameraMetadata, expected proprioceptive state input."
            )
        return None

    def get_output_specification(self) -> list[FeatureMetadata]:
        """Get structured output specification with feature names and dimensions.

        Returns:
            List of FeatureMetadata with proprioceptive feature and dimension.
        """
        return [
            FeatureMetadata(
                key=EncoderOutputKeys.PROPRIOCEPTIVE.value,
                feature_type=FeatureType.FLAT.value,
                dimension=(self.output_dim,),
            )
        ]
