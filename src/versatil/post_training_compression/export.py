"""Model export utilities for post-training compression."""

import logging

import torch
import torch.nn as nn

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
    policy: nn.Module,
    exportable: ExportablePolicy,
) -> tuple[torch.Tensor, ...]:
    """Build example inputs from policy encoder specifications.

    Args:
        policy: Policy with encoding_pipeline.
        exportable: ExportablePolicy for shape inference.

    Returns:
        Tuple of example input tensors.
    """
    observation_shapes = {}
    for encoder in policy.encoding_pipeline.encoders.values():
        spec = encoder.input_specification
        for key in spec.keys:
            observation_shapes[key] = tuple(spec.shape)
    for encoder in policy.encoding_pipeline.conditional_encoders.values():
        spec = encoder.input_specification
        for key in spec.keys:
            observation_shapes[key] = tuple(spec.shape)
    return exportable.get_example_inputs(
        observation_shapes=observation_shapes,
        batch_size=2,
    )
