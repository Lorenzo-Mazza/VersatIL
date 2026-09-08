"""Tensor input and output adapter for continuous-action policy export."""

import torch
import torch.nn as nn

from versatil.models.decoding.algorithm.base import DecodingAlgorithm
from versatil.models.decoding.decoders.base import ActionDecoder
from versatil.models.encoding.pipeline import EncodingPipeline
from versatil.models.exportable.metadata import PolicyExportMetadata
from versatil.models.policy import Policy, build_algorithm_features


class ExportablePolicy(nn.Module):
    """Expose ordered tensor inputs and outputs for policy export."""

    def __init__(
        self,
        encoding_pipeline: EncodingPipeline,
        algorithm: DecodingAlgorithm,
        decoder: ActionDecoder,
        observation_keys: list[str],
        action_keys: list[str],
        export_metadata: PolicyExportMetadata | None = None,
    ) -> None:
        """Initialize with policy components and key orderings.

        Args:
            encoding_pipeline: The policy's encoding pipeline.
            algorithm: The policy's decoding algorithm.
            decoder: The policy's action decoder.
            observation_keys: Sorted list of observation dict keys.
            action_keys: Graph output keys in their returned tensor order.
            export_metadata: Output meaning and additional graph inputs saved
                with the artifact. Defaults to normalized continuous actions.
        """
        super().__init__()
        self.encoding_pipeline = encoding_pipeline
        self.algorithm = algorithm
        self.decoder = decoder
        self._observation_keys = observation_keys
        self._action_keys = action_keys
        self.export_metadata = export_metadata or PolicyExportMetadata()

    @property
    def observation_keys(self) -> list[str]:
        """Get observation key ordering."""
        return list(self._observation_keys)

    @property
    def action_keys(self) -> list[str]:
        """Get action key ordering."""
        return list(self._action_keys)

    def forward(self, *observation_tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Encode observations and produce outputs in the saved key order.

        Args:
            *observation_tensors: Observations in ``observation_keys`` order,
                followed by the noise tensors listed in the export metadata.

        Returns:
            Action tensors or token IDs in ``action_keys`` order.

        Raises:
            ValueError: If the input count differs from the export metadata.
        """
        observation_count = len(self._observation_keys)
        noise_count = len(self.export_metadata.noise_inputs)
        if len(observation_tensors) != observation_count + noise_count:
            raise ValueError(
                f"Expected {observation_count + noise_count} policy input tensors "
                f"({observation_count} observations and {noise_count} noise inputs), "
                f"got {len(observation_tensors)}"
            )
        observation_dict = dict(
            zip(
                self._observation_keys,
                observation_tensors[:observation_count],
                strict=True,
            )
        )
        features = build_algorithm_features(
            observation=observation_dict,
            encoding_pipeline=self.encoding_pipeline,
            decoder=self.decoder,
            algorithm_injected_keys=self.algorithm.injected_feature_keys(),
        )  # (batch, ...)
        predictions = self._predict(
            features=features, sampling_inputs=observation_tensors[observation_count:]
        )  # (batch, ...)
        return tuple(predictions[key] for key in self._action_keys)  # (batch, ...)

    def _predict(
        self,
        features: dict[str, torch.Tensor],
        sampling_inputs: tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Run the policy algorithm on encoded observation features.

        Args:
            features: Observation features selected for the action decoder.
            sampling_inputs: Additional noise tensors required by the adapter.

        Returns:
            Normalized action tensors keyed by action component.

        Raises:
            ValueError: If noise inputs require a denoising adapter.
        """
        if sampling_inputs:
            raise ValueError("Noise inputs require a denoising export adapter.")
        return self.algorithm.predict(
            features=features, network=self.decoder
        )  # (batch, horizon, action_dimension)

    @classmethod
    def from_policy(cls, policy: Policy) -> "ExportablePolicy":
        """Create the continuous-action adapter from a loaded policy.

        Note:
            Input and output ordering follows the policy's key properties.

        Args:
            policy: Initialized policy whose algorithm returns continuous actions.

        Returns:
            ExportablePolicy wrapping the policy's components.
        """
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
            Observation tensors in key order, followed by required noise inputs.

        Raises:
            ValueError: If a required observation key has no shape description.
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
            example_tensors.append(
                torch.zeros(shape, dtype=dtype)
            )  # (batch, *observation_shape)

        return self.export_metadata.prepare_inputs(
            observations=tuple(example_tensors)
        )  # (batch, ...)
