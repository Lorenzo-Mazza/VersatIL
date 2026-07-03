"""Wrapper making VersatIL Policy compatible with torch.export.export()."""

import torch
import torch.nn as nn

from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.encoding.pipeline import EncodingPipeline
from versatil.models.policy import Policy, build_algorithm_features


class ExportablePolicy(nn.Module):
    """Wraps Policy components for torch.export with positional tensor I/O.

    torch.export requires positional tensor args and tuple returns.
    This wrapper converts between Policy's dict-based interface and
    the positional interface needed for quantization export.
    """

    def __init__(
        self,
        encoding_pipeline: EncodingPipeline,
        algorithm: DecodingAlgorithm,
        decoder: ActionDecoder,
        observation_keys: list[str],
        action_keys: list[str],
    ) -> None:
        """Initialize with policy components and key orderings.

        Args:
            encoding_pipeline: The policy's encoding pipeline.
            algorithm: The policy's decoding algorithm.
            decoder: The policy's action decoder.
            observation_keys: Sorted list of observation dict keys.
            action_keys: Action output keys in action-space metadata order.
        """
        super().__init__()
        self.encoding_pipeline = encoding_pipeline
        self.algorithm = algorithm
        self.decoder = decoder
        self._observation_keys = observation_keys
        self._action_keys = action_keys

    @property
    def observation_keys(self) -> list[str]:
        """Get observation key ordering."""
        return list(self._observation_keys)

    @property
    def action_keys(self) -> list[str]:
        """Get action key ordering."""
        return list(self._action_keys)

    def forward(self, *observation_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass with positional tensor I/O.

        Args:
            *observation_tensors: Tensors in the same order as observation_keys.

        Returns:
            Tuple of action tensors in the same order as action_keys.
        """
        if len(observation_tensors) != len(self._observation_keys):
            raise ValueError(
                f"Expected {len(self._observation_keys)} observation tensors "
                f"matching keys {self._observation_keys}, "
                f"got {len(observation_tensors)}"
            )
        observation_dict = dict(zip(self._observation_keys, observation_tensors))
        features = build_algorithm_features(
            observation=observation_dict,
            encoding_pipeline=self.encoding_pipeline,
            decoder=self.decoder,
        )
        predictions = self.algorithm.predict(features=features, network=self.decoder)
        return tuple(predictions[key] for key in self._action_keys)

    @classmethod
    def from_policy(cls, policy: Policy) -> "ExportablePolicy":
        """Create ExportablePolicy from a loaded Policy.

        Delegates key derivation to Policy.input_keys and
        Policy.output_keys, keeping the single source of truth.

        Args:
            policy: A loaded, initialized Policy instance.

        Returns:
            ExportablePolicy wrapping the policy's components.

        Raises:
            ValueError: If the policy predicts action tokens; exported
                forward passes return raw tensors and never detokenize, so
                the exported outputs would not match the action space.
        """
        if policy.decoder.requires_tokenized_actions:
            raise ValueError(
                "Policies with tokenized-action decoders cannot be exported: "
                "the exported forward returns raw action tokens without "
                "detokenization."
            )
        return cls(
            encoding_pipeline=policy.encoding_pipeline,
            algorithm=policy.algorithm,
            decoder=policy.decoder,
            observation_keys=policy.input_keys,
            action_keys=policy.output_keys,
        )

    def get_example_inputs(
        self,
        observation_shapes: dict[str, tuple[int, ...]],
        batch_size: int = 1,
        observation_dtypes: dict[str, torch.dtype] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Generate example inputs with correct shapes for torch.export.

        Args:
            observation_shapes: Mapping from observation key to shape tuple
                (excluding batch dimension). Must cover all observation_keys.
            batch_size: Batch dimension size.
            observation_dtypes: Optional mapping from observation key to
                torch dtype. Defaults to torch.float32 for all keys.

        Returns:
            Tuple of tensors matching observation_keys order.
        """
        if observation_dtypes is None:
            observation_dtypes = {}

        example_tensors = []
        for key in self._observation_keys:
            if key not in observation_shapes:
                raise ValueError(
                    f"No shape provided for observation key {key!r}. "
                    f"observation_shapes must cover all observation_keys. "
                    f"Missing keys: {set(self._observation_keys) - set(observation_shapes.keys())}"
                )
            shape = (batch_size, *observation_shapes[key])
            dtype = observation_dtypes.get(key, torch.float32)
            example_tensors.append(torch.zeros(shape, dtype=dtype))

        return tuple(example_tensors)
