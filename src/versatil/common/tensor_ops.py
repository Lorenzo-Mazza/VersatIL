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
