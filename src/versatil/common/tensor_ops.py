"""Tensor container utilities used across VersatIL."""

from collections import OrderedDict
from collections.abc import Callable

import torch

type NestedDictionary[Leaf] = dict[str, Leaf | NestedDictionary[Leaf]]
type TensorTree = (
    torch.Tensor
    | str
    | None
    | dict[str, TensorTree]
    | OrderedDict[str, TensorTree]
    | list[TensorTree]
    | tuple[TensorTree, ...]
)
type TensorTreeHandlerMap = dict[type, Callable[[TensorTree], TensorTree]]


def dict_apply[LeafInput, LeafOutput](
    data: NestedDictionary[LeafInput],
    transform: Callable[[LeafInput], LeafOutput],
) -> NestedDictionary[LeafOutput]:
    """Apply a transform to every leaf in a nested dictionary.

    Args:
        data: Nested dictionary whose leaves should be transformed.
        transform: Function applied to each leaf value.

    Returns:
        Nested dictionary with the same keys and transformed leaf values.
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = dict_apply(data=value, transform=transform)
        else:
            result[key] = transform(value)
    return result


def tensor_to_str(tensor: torch.Tensor) -> str:
    """Format a tensor as a compact numeric list for logs.

    Args:
        tensor: Tensor to format. Scalars are treated as one-element tensors.

    Returns:
        String with each value formatted to three significant digits.
    """
    flat_tensor = tensor.flatten()
    return "[" + ", ".join([f"{value.item():.3g}" for value in flat_tensor]) + "]"


def clone_tensor_dictionary_with_replacements(
    values: dict[str, torch.Tensor],
    replacements: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Copy a tensor dictionary and replace selected entries.

    Args:
        values: Source tensor dictionary. Values are usually batched tensors with
            shape ``(B, ...)``.
        replacements: Tensor values to overwrite in the copy. Replacement shapes
            are not changed or validated by this helper.

    Returns:
        Shallow copy of ``values`` with ``replacements`` applied.
    """
    result = dict(values)
    result.update(replacements)
    return result


def detach_floating_tensor_dictionary(
    values: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Detach floating-point tensor values from their autograd graph.

    Args:
        values: Tensor dictionary to detach. Values can have any shape.

    Returns:
        Tensor dictionary where floating tensors are detached and non-floating
        tensors are preserved.
    """
    return {
        key: value.detach() if torch.is_floating_point(value) else value
        for key, value in values.items()
    }


def slice_tensor_dictionary(
    values: dict[str, torch.Tensor] | None,
    max_batch_size: int | None,
) -> dict[str, torch.Tensor] | None:
    """Slice the leading batch dimension of tensor dictionary values.

    Args:
        values: Tensor dictionary to slice, or ``None``. Every tensor is expected
            to be batched with shape ``(B, ...)``.
        max_batch_size: Maximum number of leading batch elements to keep.

    Returns:
        Sliced tensor dictionary, or the original ``None`` value.
    """
    if values is None or max_batch_size is None:
        return values
    return {
        key: value[:max_batch_size] if value.shape[0] > max_batch_size else value
        for key, value in values.items()
    }


def reshape_batch_scale_for_broadcast(
    scale: torch.Tensor,
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Reshape a batch vector so it broadcasts over a reference tensor.

    Args:
        scale: Batch vector with shape ``(B,)`` or ``(B, 1)``.
        tensor: Reference tensor with shape ``(B, ...)`` whose non-batch
            dimensions define broadcasting.

    Returns:
        Reshaped scale tensor with shape ``(B, 1, ..., 1)`` and rank matching
        ``tensor``.
    """
    return scale.reshape(tensor.shape[0], *([1] * (tensor.ndim - 1)))


def batch_rms(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute per-sample RMS over all non-batch dimensions.

    Args:
        tensor: Batched tensor with shape ``(B, ...)``.
        eps: Lower bound for the returned RMS values.

    Returns:
        Per-sample RMS vector with shape ``(B,)``.

    Raises:
        ValueError: If ``tensor`` is scalar.
    """
    if tensor.ndim == 0:
        raise ValueError("Expected a batched tensor, got a scalar tensor.")
    flattened = tensor.reshape(tensor.shape[0], -1)
    return flattened.pow(2).mean(dim=1).sqrt().clamp_min(eps)


def combined_batch_rms(
    tensors: list[torch.Tensor],
    eps: float,
) -> torch.Tensor:
    """Compute per-sample RMS after concatenating tensor values.

    Args:
        tensors: Batched tensors with matching leading batch dimension ``B``.
            Each tensor may have its own trailing shape ``(B, ...)``.
        eps: Lower bound for the returned RMS values.

    Returns:
        Per-sample RMS vector with shape ``(B,)`` over the concatenated
        non-batch dimensions.

    Raises:
        ValueError: If ``tensors`` is empty.
    """
    if not tensors:
        raise ValueError("Expected at least one tensor for RMS computation.")
    flattened = [tensor.reshape(tensor.shape[0], -1) for tensor in tensors]
    return torch.cat(flattened, dim=1).pow(2).mean(dim=1).sqrt().clamp_min(eps)


def normalize_tensor_tuple(
    tensors: tuple[torch.Tensor, ...],
    eps: float,
) -> tuple[torch.Tensor, ...]:
    """Normalize a tuple of tensors as one product-space vector.

    Args:
        tensors: Tensor tuple to normalize jointly. Shapes are preserved.
        eps: Lower bound for the product-space norm.

    Returns:
        Tensor tuple divided by the shared product-space norm.
    """
    squared_norm = sum(tensor.pow(2).sum() for tensor in tensors)
    norm = torch.sqrt(squared_norm).clamp_min(eps)
    return tuple(tensor / norm for tensor in tensors)


def recursive_dict_list_tuple_apply(
    data: TensorTree,
    type_handler_map: TensorTreeHandlerMap,
) -> TensorTree:
    """Apply type-specific handlers to leaves inside nested containers.

    Args:
        data: Tensor tree containing dictionaries, sequences, and supported leaves.
        type_handler_map: Mapping from leaf type to its transform function.

    Returns:
        Tensor tree with handlers applied to matching leaves.

    Raises:
        ValueError: If a container type is passed as a leaf handler.
        NotImplementedError: If a leaf value has no matching handler.
    """
    if isinstance(data, (dict, OrderedDict)):
        if dict in type_handler_map:
            raise ValueError("dict cannot be handled by type_func_dict")
        return OrderedDict(
            [
                (
                    key,
                    recursive_dict_list_tuple_apply(
                        data=value,
                        type_handler_map=type_handler_map,
                    ),
                )
                for key, value in data.items()
            ]
        )

    if isinstance(data, (list, tuple)):
        if list in type_handler_map or tuple in type_handler_map:
            raise ValueError("list/tuple cannot be handled by type_func_dict")
        return type(data)(
            [
                recursive_dict_list_tuple_apply(
                    data=value,
                    type_handler_map=type_handler_map,
                )
                for value in data
            ]
        )

    for leaf_type, handler in type_handler_map.items():
        if isinstance(data, leaf_type):
            return handler(data)

    raise NotImplementedError(f"Unsupported type: {type(data)}")


def to_device(data: TensorTree, device: torch.device | str) -> TensorTree:
    """Move every tensor in a nested container to a device.

    Args:
        data: Tensor tree containing tensors and pass-through string or None leaves.
        device: Destination device.

    Returns:
        Tensor tree with all tensor leaves moved to the destination device.
    """
    return recursive_dict_list_tuple_apply(
        data=data,
        type_handler_map={
            torch.Tensor: lambda tensor: tensor.to(device),
            str: lambda value: value,
            type(None): lambda value: value,
        },
    )
