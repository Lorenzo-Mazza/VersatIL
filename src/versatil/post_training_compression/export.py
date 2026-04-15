"""Model export utilities for post-training compression."""

import logging

import torch
import torch.nn as nn

from versatil.data.constants import SampleKey
from versatil.data.task import ObservationSpace
from versatil.data.tokenization.tokenizer import Tokenizer
from versatil.models.exportable_policy import ExportablePolicy


def _export_with_dynamic_batch(
    model: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    dynamic_shapes_key: str | None = None,
) -> torch.export.ExportedProgram:
    """Export model with dynamic batch dimension on all inputs.

    Args:
        model: Model to export.
        example_inputs: Example input tensors for tracing.
        dynamic_shapes_key: If provided, wraps shapes in a dict
            with this key (for models with named *args). Otherwise
            uses a positional tuple.

    Returns:
        ExportedProgram with dynamic batch dimension.
    """
    batch_dim = torch.export.Dim("batch", min=1)
    per_input_shapes = tuple({0: batch_dim} for _ in range(len(example_inputs)))
    if dynamic_shapes_key is not None:
        dynamic_shapes = {dynamic_shapes_key: per_input_shapes}
    else:
        dynamic_shapes = per_input_shapes
    return torch.export.export(
        model,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )


def export_policy(
    exportable: ExportablePolicy,
    example_inputs: tuple[torch.Tensor, ...],
) -> nn.Module:
    """Export an ExportablePolicy with dynamic batch dimension.

    Runs one eager forward pass before tracing to materialize any
    lazily-initialized modules (e.g. FeatureProjection layers).
    torch.export's FX tracer silently drops nn.ModuleDict mutations,
    so all projection layers must exist before tracing begins.

    Args:
        exportable: The ExportablePolicy wrapping the policy.
        example_inputs: Example input tensors for tracing.

    Returns:
        Exported FX GraphModule.
    """
    logging.info("Materializing lazy modules with eager forward pass...")
    with torch.no_grad():
        exportable(*example_inputs)

    return _export_with_dynamic_batch(
        model=exportable,
        example_inputs=example_inputs,
        dynamic_shapes_key="observation_tensors",
    ).module()


def build_example_inputs(
    exportable: ExportablePolicy,
    observation_space: ObservationSpace,
    observation_horizon: int,
    tokenizer: Tokenizer | None = None,
) -> tuple[torch.Tensor, ...]:
    """Build example inputs from observation space metadata.

    Args:
        exportable: ExportablePolicy defining required observation keys.
        observation_space: Observation space with camera/proprio metadata.
        observation_horizon: Number of temporal observation frames.
        tokenizer: Tokenizer for language token sequence length.

    Returns:
        Tuple of example input tensors matching exportable.observation_keys.
    """
    observation_shapes: dict[str, tuple[int, ...]] = {}
    observation_dtypes: dict[str, torch.dtype] = {}

    for key, camera_meta in observation_space.cameras.items():
        observation_shapes[key] = (
            observation_horizon,
            camera_meta.channels,
            camera_meta.image_height,
            camera_meta.image_width,
        )

    for key, proprio_meta in observation_space.proprioceptive_observations.items():
        observation_shapes[key] = (observation_horizon, proprio_meta.dimension)

    if tokenizer is not None and tokenizer.observation_tokenizer is not None:
        token_length = tokenizer.observation_tokenizer.max_token_len
        observation_shapes[SampleKey.TOKENIZED_OBSERVATIONS.value] = (
            observation_horizon,
            token_length,
        )
        observation_dtypes[SampleKey.TOKENIZED_OBSERVATIONS.value] = torch.long
        observation_shapes[SampleKey.IS_PAD_OBSERVATION.value] = (
            observation_horizon,
            token_length,
        )
        observation_dtypes[SampleKey.IS_PAD_OBSERVATION.value] = torch.bool

    return exportable.get_example_inputs(
        observation_shapes=observation_shapes,
        observation_dtypes=observation_dtypes,
        batch_size=2,
    )
