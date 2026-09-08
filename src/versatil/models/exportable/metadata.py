"""Tensor inputs and prediction types shared by export and policy runtimes."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import torch


class PredictionOutput(StrEnum):
    """Meaning of a policy graph's output tensors."""

    ACTIONS = "actions"
    ACTION_TOKENS = "action_tokens"


class SamplingInput(StrEnum):
    """Names of the initial and per-step noise inputs to an exported policy."""

    INITIAL_NOISE = "initial_noise"
    STEP_NOISE = "step_noise"


@dataclass(frozen=True)
class NoiseInput:
    """Name and dimensions of a noise tensor required by an exported policy.

    Attributes:
        name: Nonempty input name, such as ``initial_noise`` or ``step_noise``.
        shape: Positive integer tensor dimensions after the batch dimension.

    Note:
        Values are sampled from a normal distribution with mean zero and standard
        deviation one.

    Raises:
        ValueError: If the name is empty or the dimensions contain invalid values.
    """

    name: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate the input name and noise tensor dimensions."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                f"Noise input names must be nonempty strings, got {self.name!r}."
            )
        if not self.shape or any(
            type(dimension) is not int or dimension <= 0 for dimension in self.shape
        ):
            raise ValueError(
                f"Noise input {self.name!r} requires a nonempty shape of positive "
                f"integers, got {self.shape}."
            )


@dataclass(frozen=True)
class PolicyExportMetadata:
    """Describe the output format and additional inputs of an exported policy.

    Note:
        Outputs contain normalized actions or action-token IDs. Diffusion and flow
        policies can also require noise tensors after the observation inputs;
        ``noise_inputs`` lists those tensors in graph argument order.
        The policy runtime decodes action tokens for tokenized policies and reverses
        action normalization.

    Attributes:
        output: Whether the graph returns normalized actions or action-token IDs.
        noise_inputs: Optional noise input descriptions for diffusion and flow,
            in graph input order after observations.
        version: Version of the metadata format used to save these fields.

    Raises:
        ValueError: If the format version is unsupported or noise input names repeat.
    """

    output: PredictionOutput = PredictionOutput.ACTIONS
    noise_inputs: tuple[NoiseInput, ...] = field(default_factory=tuple)
    version: int = 1

    def __post_init__(self) -> None:
        """Validate the metadata version and noise input names."""
        if self.version != 1:
            raise ValueError(
                f"Unsupported policy export metadata version {self.version}."
            )
        names = tuple(noise.name for noise in self.noise_inputs)
        if len(names) != len(set(names)):
            raise ValueError(
                f"Policy export noise input names must be unique, got {names}."
            )

    def prepare_inputs(
        self, observations: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, ...]:
        """Sample the required noise tensors and append them to the observations.

        Args:
            observations: Preprocessed tensors in the saved observation-key order.

        Returns:
            Observations followed by independent standard-normal noise tensors.
            Each noise tensor has shape ``(batch, *noise_input.shape)``.

        Raises:
            ValueError: If noise inputs require an observation tensor and the
                observation tuple is empty, or the reference tensor is scalar.

        Note:
            Noise uses the first floating-point observation's batch size, device
            and dtype. When all observations have integer or boolean dtypes,
            noise uses the first observation's batch size and device with float32
            dtype.
        """
        if not self.noise_inputs:
            return observations
        if not observations:
            raise ValueError(
                "Policy export noise inputs require at least one observation tensor."
            )
        reference = next(
            (tensor for tensor in observations if tensor.is_floating_point()),
            observations[0],
        )
        if reference.ndim == 0:
            raise ValueError(
                "Policy export noise inputs require an observation "
                "tensor with a batch dimension."
            )
        noise_dtype = (
            reference.dtype if reference.is_floating_point() else torch.float32
        )
        return observations + tuple(
            torch.randn(
                (reference.shape[0], *noise.shape),
                device=reference.device,
                dtype=noise_dtype,
            )  # (batch, *noise.shape)
            for noise in self.noise_inputs
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the output type, noise input descriptions and format version as a dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, metadata: dict[str, Any]) -> "PolicyExportMetadata":
        """Read the output type and noise input descriptions from saved metadata.

        Args:
            metadata: Dictionary produced by ``to_dict()`` or read from JSON.
                Missing fields use the defaults for a policy that returns actions
                and requires no noise inputs.

        Returns:
            Input and output metadata for the exported policy.

        Raises:
            ValueError: If the version or output type is unsupported, or noise input
                names or shapes are invalid.
        """
        return cls(
            output=PredictionOutput(metadata.get("output", PredictionOutput.ACTIONS)),
            noise_inputs=tuple(
                NoiseInput(name=entry["name"], shape=tuple(entry["shape"]))
                for entry in metadata.get("noise_inputs", [])
            ),
            version=metadata.get("version", 1),
        )
