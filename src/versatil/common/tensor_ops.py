"""
A collection of utilities for working with nested tensor structures consisting
of numpy arrays and torch tensors.
"""
import collections
import functools
from typing import Dict, Callable, List

import numpy as np
import torch
from torch import nn


def dict_apply(
    x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def pad_remaining_dims(x, target):
    if x.shape != target.shape[: len(x.shape)]:
        raise ValueError(
            f"Shape mismatch: x.shape {x.shape} is not a prefix of "
            f"target.shape {target.shape}"
        )
    return x.reshape(x.shape + (1,) * (len(target.shape) - len(x.shape)))


def dict_apply_split(
    x: Dict[str, torch.Tensor],
    split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results


def dict_apply_reduce(
    x: List[Dict[str, torch.Tensor]],
    reduce_func: Callable[[List[torch.Tensor]], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def replace_submodules(
    root_module: nn.Module,
    predicate: Callable[[nn.Module], bool],
    func: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule(".".join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all matching modules are replaced
    remaining = [
        k.split(".")
        for k, m in root_module.named_modules(remove_duplicate=True)
        if predicate(m)
    ]
    if len(remaining) != 0:
        raise RuntimeError(
            f"Failed to replace all matching submodules. "
            f"{len(remaining)} modules still match the predicate."
        )
    return root_module


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer


def tensor_to_str(t: torch.Tensor) -> str:
    """Convert tensor to clean string without torch metadata."""
    arr = t.detach().cpu().numpy()
    if arr.ndim == 0:
        return f"{arr.item():.4f}"
    return "[" + ", ".join(f"{x:.4f}" for x in arr) + "]"


def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to a nested dictionary or list or tuple, given a dictionary of
    {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        type_func_dict (dict): a mapping from data types to the functions to be
            applied for each data type.

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    if list in type_func_dict:
        raise TypeError(
            "type_func_dict must not contain 'list' as a key. "
            "Lists are traversed recursively."
        )
    if tuple in type_func_dict:
        raise TypeError(
            "type_func_dict must not contain 'tuple' as a key. "
            "Tuples are traversed recursively."
        )
    if dict in type_func_dict:
        raise TypeError(
            "type_func_dict must not contain 'dict' as a key. "
            "Dicts are traversed recursively."
        )

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = (
            collections.OrderedDict() if isinstance(x, collections.OrderedDict) else {}
        )
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    elif isinstance(x, (list, tuple)):
        ret: list | tuple = [
            recursive_dict_list_tuple_apply(v, type_func_dict) for v in x
        ]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            raise NotImplementedError(f"Cannot handle data type {str(type(x))}")


def map_tensor(x, func):
    """
    Apply function @func to torch.Tensor objects in a nested dictionary or
    list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        func (function): function to apply to each tensor

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: func,
            type(None): lambda x: x,
        },
    )


def map_ndarray(x, func):
    """
    Apply function @func to np.ndarray objects in a nested dictionary or
    list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        func (function): function to apply to each array

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            np.ndarray: func,
            type(None): lambda x: x,
        },
    )


def map_tensor_ndarray(x, tensor_func, ndarray_func):
    """
    Apply function @tensor_func to torch.Tensor objects and @ndarray_func to
    np.ndarray objects in a nested dictionary or list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        tensor_func (function): function to apply to each tensor
        ndarray_Func (function): function to apply to each array

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: tensor_func,
            np.ndarray: ndarray_func,
            type(None): lambda x: x,
        },
    )


def clone(x):
    """
    Clones all torch tensors and numpy arrays in nested dictionary or list
    or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.clone(),
            np.ndarray: lambda x: x.copy(),
            type(None): lambda x: x,
        },
    )


def detach(x):
    """
    Detaches all torch tensors in nested dictionary or list
    or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.detach(),
        },
    )


def to_batch(x):
    """
    Introduces a leading batch dimension of 1 for all torch tensors and numpy
    arrays in nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x[None, ...],
            np.ndarray: lambda x: x[None, ...],
            type(None): lambda x: x,
        },
    )


def to_sequence(x):
    """
    Introduces a time dimension of 1 at dimension 1 for all torch tensors and numpy
    arrays in nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x[:, None, ...],
            np.ndarray: lambda x: x[:, None, ...],
            type(None): lambda x: x,
        },
    )


def index_at_time(x, ind):
    """
    Indexes all torch tensors and numpy arrays in dimension 1 with index @ind in
    nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        ind (int): index

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x[:, ind, ...],
            np.ndarray: lambda x: x[:, ind, ...],
            type(None): lambda x: x,
        },
    )


def unsqueeze(x, dim):
    """
    Adds dimension of size 1 at dimension @dim in all torch tensors and numpy arrays
    in nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        dim (int): dimension

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.unsqueeze(dim=dim),
            np.ndarray: lambda x: np.expand_dims(x, axis=dim),
            type(None): lambda x: x,
        },
    )


def contiguous(x):
    """
    Makes all torch tensors and numpy arrays contiguous in nested dictionary or
    list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.contiguous(),
            np.ndarray: lambda x: np.ascontiguousarray(x),
            type(None): lambda x: x,
        },
    )


def to_device(x, device):
    """
    Sends all torch tensors in nested dictionary or list or tuple to device
    @device, and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        device (torch.Device): device to send tensors to

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, d=device: x.to(d),
            str: lambda x: x,
            type(None): lambda x: x,
        },
    )


def to_tensor(x):
    """
    Converts all numpy arrays in nested dictionary or list or tuple to
    torch tensors (and leaves existing torch Tensors as-is), and returns
    a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x,
            np.ndarray: lambda x: torch.from_numpy(x),
            type(None): lambda x: x,
        },
    )


def to_numpy(x):
    """
    Converts all torch tensors in nested dictionary or list or tuple to
    numpy (and leaves existing numpy arrays as-is), and returns
    a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """

    def f(tensor):
        if tensor.is_cuda:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()

    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: f,
            np.ndarray: lambda x: x,
            type(None): lambda x: x,
        },
    )


def to_list(x):
    """
    Converts all torch tensors and numpy arrays in nested dictionary or list
    or tuple to a list, and returns a new nested structure. Useful for
    json encoding.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """

    def f(tensor):
        if tensor.is_cuda:
            return tensor.detach().cpu().numpy().tolist()
        else:
            return tensor.detach().numpy().tolist()

    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: f,
            np.ndarray: lambda x: x.tolist(),
            type(None): lambda x: x,
        },
    )


def to_float(x):
    """
    Converts all torch tensors and numpy arrays in nested dictionary or list
    or tuple to float type entries, and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.float(),
            np.ndarray: lambda x: x.astype(np.float32),
            type(None): lambda x: x,
        },
    )


def to_uint8(x):
    """
    Converts all torch tensors and numpy arrays in nested dictionary or list
    or tuple to uint8 type entries, and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.byte(),
            np.ndarray: lambda x: x.astype(np.uint8),
            type(None): lambda x: x,
        },
    )


def to_torch(x, device):
    """
    Converts all numpy arrays and torch tensors in nested dictionary or list or tuple to
    torch tensors on device @device and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        device (torch.Device): device to send tensors to

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return to_device(to_float(to_tensor(x)), device)


def to_one_hot_single(tensor, num_class):
    """
    Convert tensor to one-hot representation, assuming a certain number of total class labels.

    Args:
        tensor (torch.Tensor): tensor containing integer labels
        num_class (int): number of classes

    Returns:
        action_embedding (torch.Tensor): tensor containing one-hot representation of labels
    """
    x = torch.zeros(tensor.size() + (num_class,)).to(tensor.device)
    x.scatter_(-1, tensor.unsqueeze(-1), 1)
    return x


def to_one_hot(tensor, num_class):
    """
    Convert all tensors in nested dictionary or list or tuple to one-hot representation,
    assuming a certain number of total class labels.

    Args:
        tensor (dict or list or tuple): a possibly nested dictionary or list or tuple
        num_class (int): number of classes

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return map_tensor(tensor, func=lambda x, nc=num_class: to_one_hot_single(x, nc))


def flatten_single(x, begin_axis=1):
    """
    Flatten a tensor in all dimensions from @begin_axis onwards.

    Args:
        x (torch.Tensor): tensor to flatten
        begin_axis (int): which axis to flatten from

    Returns:
        y (torch.Tensor): flattened tensor
    """
    fixed_size = x.size()[:begin_axis]
    _s = list(fixed_size) + [-1]
    return x.reshape(*_s)


def flatten(x, begin_axis=1):
    """
    Flatten all tensors in nested dictionary or list or tuple, from @begin_axis onwards.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): which axis to flatten from

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis: flatten_single(x, begin_axis=b),
        },
    )


def reshape_dimensions_single(x, begin_axis, end_axis, target_dims):
    """
    Reshape selected dimensions in a tensor to a target dimension.

    Args:
        x (torch.Tensor): tensor to reshape
        begin_axis (int): begin dimension
        end_axis (int): end dimension
        target_dims (tuple or list): target shape for the range of dimensions
            (@begin_axis, @end_axis)

    Returns:
        y (torch.Tensor): reshaped tensor
    """
    if begin_axis > end_axis:
        raise ValueError(
            f"begin_axis ({begin_axis}) must be <= end_axis ({end_axis})"
        )
    if begin_axis < 0:
        raise ValueError(
            f"begin_axis ({begin_axis}) must be >= 0"
        )
    if end_axis >= len(x.shape):
        raise ValueError(
            f"end_axis ({end_axis}) must be < number of dimensions ({len(x.shape)})"
        )
    if not isinstance(target_dims, (tuple, list)):
        raise TypeError(
            f"target_dims must be a tuple or list, got {type(target_dims).__name__}"
        )
    s = x.shape
    final_s: list[int] = []
    for i in range(len(s)):
        if i == begin_axis:
            final_s.extend(target_dims)
        elif i < begin_axis or i > end_axis:
            final_s.append(s[i])
    return x.reshape(*final_s)


def reshape_dimensions(x, begin_axis, end_axis, target_dims):
    """
    Reshape selected dimensions for all tensors in nested dictionary or list or tuple
    to a target dimension.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): begin dimension
        end_axis (int): end dimension
        target_dims (tuple or list): target shape for the range of dimensions
            (@begin_axis, @end_axis)

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis, e=end_axis, t=target_dims: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=t
            ),
            np.ndarray: lambda x, b=begin_axis, e=end_axis, t=target_dims: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=t
            ),
            type(None): lambda x: x,
        },
    )


def join_dimensions(x, begin_axis, end_axis):
    """
    Joins all dimensions between dimensions (@begin_axis, @end_axis) into a flat dimension, for
    all tensors in nested dictionary or list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): begin dimension
        end_axis (int): end dimension

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis, e=end_axis: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=[-1]
            ),
            np.ndarray: lambda x, b=begin_axis, e=end_axis: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=[-1]
            ),
            type(None): lambda x: x,
        },
    )


def expand_at_single(x, size, dim):
    """
    Expand a tensor at a single dimension @dim by @size

    Args:
        x (torch.Tensor): input tensor
        size (int): size to expand
        dim (int): dimension to expand

    Returns:
        y (torch.Tensor): expanded tensor
    """
    if dim >= x.ndimension():
        raise ValueError(
            f"dim ({dim}) must be < number of dimensions ({x.ndimension()})"
        )
    if x.shape[dim] != 1:
        raise ValueError(
            f"Dimension {dim} must have size 1 for expansion, got {x.shape[dim]}"
        )
    expand_dims = [-1] * x.ndimension()
    expand_dims[dim] = size
    return x.expand(*expand_dims)


def expand_at(x, size, dim):
    """
    Expand all tensors in nested dictionary or list or tuple at a single
    dimension @dim by @size.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        size (int): size to expand
        dim (int): dimension to expand

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return map_tensor(x, lambda t, s=size, d=dim: expand_at_single(t, s, d))


def unsqueeze_expand_at(x, size, dim):
    """
    Unsqueeze and expand a tensor at a dimension @dim by @size.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        size (int): size to expand
        dim (int): dimension to unsqueeze and expand

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    x = unsqueeze(x, dim)
    return expand_at(x, size, dim)


def repeat_by_expand_at(x, repeats, dim):
    """
    Repeat a dimension by combining expand and reshape operations.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        repeats (int): number of times to repeat the target dimension
        dim (int): dimension to repeat on

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    x = unsqueeze_expand_at(x, repeats, dim + 1)
    return join_dimensions(x, dim, dim + 1)


def named_reduce_single(x, reduction, dim):
    """
    Reduce tensor at a dimension by named reduction functions.

    Args:
        x (torch.Tensor): tensor to be reduced
        reduction (str): one of ["sum", "max", "mean", "flatten"]
        dim (int): dimension to be reduced (or begin axis for flatten)

    Returns:
        y (torch.Tensor): reduced tensor
    """
    if x.ndimension() <= dim:
        raise ValueError(
            f"Tensor has {x.ndimension()} dimensions, "
            f"but dim ({dim}) requires at least {dim + 1}"
        )
    valid_reductions = ["sum", "max", "mean", "flatten"]
    if reduction not in valid_reductions:
        raise ValueError(
            f"reduction must be one of {valid_reductions}, got '{reduction}'"
        )
    if reduction == "flatten":
        x = flatten(x, begin_axis=dim)
    elif reduction == "max":
        x = torch.max(x, dim=dim)[0]  # [B, D]
    elif reduction == "sum":
        x = torch.sum(x, dim=dim)
    else:
        x = torch.mean(x, dim=dim)
    return x


def named_reduce(x, reduction, dim):
    """
    Reduces all tensors in nested dictionary or list or tuple at a dimension
    using a named reduction function.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        reduction (str): one of ["sum", "max", "mean", "flatten"]
        dim (int): dimension to be reduced (or begin axis for flatten)

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return map_tensor(
        x, func=lambda t, r=reduction, d=dim: named_reduce_single(t, r, d)
    )


def gather_along_dim_with_dim_single(x, target_dim, source_dim, indices):
    """
    This function indexes out a target dimension of a tensor in a structured way,
    by allowing a different value to be selected for each member of a flat index
    tensor (@indices) corresponding to a source dimension. This can be interpreted
    as moving along the source dimension, using the corresponding index value
    in @indices to select values for all other dimensions outside of the
    source and target dimensions. A common use case is to gather values
    in target dimension 1 for each batch member (target dimension 0).

    Args:
        x (torch.Tensor): tensor to gather values for
        target_dim (int): dimension to gather values along
        source_dim (int): dimension to hold constant and use for gathering values
            from the other dimensions
        indices (torch.Tensor): flat index tensor with same shape as tensor @action_embedding along
            @source_dim

    Returns:
        y (torch.Tensor): gathered tensor, with dimension @target_dim indexed out
    """
    if len(indices.shape) != 1:
        raise ValueError(
            f"indices must be 1D, got shape {indices.shape}"
        )
    if x.shape[source_dim] != indices.shape[0]:
        raise ValueError(
            f"x.shape[{source_dim}] ({x.shape[source_dim]}) must match "
            f"indices.shape[0] ({indices.shape[0]})"
        )

    # unsqueeze in all dimensions except the source dimension
    new_shape = [1] * x.ndimension()
    new_shape[source_dim] = -1
    indices = indices.reshape(*new_shape)

    # repeat in all dimensions - but preserve shape of source dimension,
    # and make sure target_dimension has singleton dimension
    expand_shape = list(x.shape)
    expand_shape[source_dim] = -1
    expand_shape[target_dim] = 1
    indices = indices.expand(*expand_shape)

    out = x.gather(dim=target_dim, index=indices)
    return out.squeeze(target_dim)


def gather_along_dim_with_dim(x, target_dim, source_dim, indices):
    """
    Apply @gather_along_dim_with_dim_single to all tensors in a nested
    dictionary or list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        target_dim (int): dimension to gather values along
        source_dim (int): dimension to hold constant and use for gathering values
            from the other dimensions
        indices (torch.Tensor): flat index tensor with same shape as tensor @action_embedding along
            @source_dim

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return map_tensor(
        x,
        lambda y, t=target_dim, s=source_dim, i=indices: gather_along_dim_with_dim_single(
            y, t, s, i
        ),
    )


def gather_sequence_single(seq, indices):
    """
    Given a tensor with leading dimensions [B, T, ...], gather an element from each sequence in
    the batch given an index for each sequence.

    Args:
        seq (torch.Tensor): tensor with leading dimensions [B, T, ...]
        indices (torch.Tensor): tensor indices of shape [B]

    Return:
        y (torch.Tensor): indexed tensor of shape [B, ....]
    """
    return gather_along_dim_with_dim_single(
        seq, target_dim=1, source_dim=0, indices=indices
    )


def gather_sequence(seq, indices):
    """
    Given a nested dictionary or list or tuple, gathers an element from each sequence of the batch
    for tensors with leading dimensions [B, T, ...].

    Args:
        seq (dict or list or tuple): a possibly nested dictionary or list or tuple with tensors
            of leading dimensions [B, T, ...]
        indices (torch.Tensor): tensor indices of shape [B]

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple with tensors of shape [B, ...]
    """
    return gather_along_dim_with_dim(seq, target_dim=1, source_dim=0, indices=indices)


def pad_sequence_single(seq, padding, batched=False, pad_same=True, pad_values=None):
    """
    Pad input tensor or array @seq in the time dimension (dimension 1).

    Args:
        seq (np.ndarray or torch.Tensor): sequence to be padded
        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and end of the sequence by 1
        batched (bool): if sequence has the batch dimension
        pad_same (bool): if pad by duplicating
        pad_values (scalar or (ndarray, Tensor)): values to be padded if not pad_same

    Returns:
        padded sequence (np.ndarray or torch.Tensor)
    """
    if not isinstance(seq, (np.ndarray, torch.Tensor)):
        raise TypeError(
            f"seq must be np.ndarray or torch.Tensor, got {type(seq).__name__}"
        )
    if not pad_same and pad_values is None:
        raise ValueError(
            "pad_values must be provided when pad_same is False"
        )
    if pad_values is not None and not isinstance(pad_values, float):
        raise TypeError(
            f"pad_values must be a float, got {type(pad_values).__name__}"
        )
    repeat_func = np.repeat if isinstance(seq, np.ndarray) else torch.repeat_interleave
    concat_func = np.concatenate if isinstance(seq, np.ndarray) else torch.cat
    ones_like_func = np.ones_like if isinstance(seq, np.ndarray) else torch.ones_like
    seq_dim = 1 if batched else 0

    begin_pad = []
    end_pad = []

    if padding[0] > 0:
        if batched:
            first_element = seq[:, [0]]
        else:
            first_element = seq[[0]]
        pad = first_element if pad_same else ones_like_func(first_element) * pad_values  # type: ignore[arg-type]
        begin_pad.append(repeat_func(pad, padding[0], seq_dim))  # type: ignore[arg-type]
    if padding[1] > 0:
        if batched:
            last_element = seq[:, [-1]]
        else:
            last_element = seq[[-1]]
        pad = last_element if pad_same else ones_like_func(last_element) * pad_values  # type: ignore[arg-type]
        end_pad.append(repeat_func(pad, padding[1], seq_dim))  # type: ignore[arg-type]

    return concat_func(begin_pad + [seq] + end_pad, seq_dim)  # type: ignore[arg-type]


def pad_sequence(seq, padding, batched=False, pad_same=True, pad_values=None):
    """
    Pad a nested dictionary or list or tuple of sequence tensors in the time dimension (dimension 1).

    Args:
        seq (dict or list or tuple): a possibly nested dictionary or list or tuple with tensors
            of leading dimensions [B, T, ...]
        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and end of the sequence by 1
        batched (bool): if sequence has the batch dimension
        pad_same (bool): if pad by duplicating
        pad_values (scalar or (ndarray, Tensor)): values to be padded if not pad_same

    Returns:
        padded sequence (dict or list or tuple)
    """
    return recursive_dict_list_tuple_apply(
        seq,
        {
            torch.Tensor: lambda x, p=padding, b=batched, ps=pad_same, pv=pad_values: pad_sequence_single(
                x, p, b, ps, pv
            ),
            np.ndarray: lambda x, p=padding, b=batched, ps=pad_same, pv=pad_values: pad_sequence_single(
                x, p, b, ps, pv
            ),
            type(None): lambda x: x,
        },
    )


def assert_size_at_dim_single(x, size, dim, msg):
    """
    Ensure that array or tensor @action_embedding has size @size in dim @dim.

    Args:
        x (np.ndarray or torch.Tensor): input array or tensor
        size (int): size that tensors should have at @dim
        dim (int): dimension to check
        msg (str): text to display if assertion fails
    """
    if x.shape[dim] != size:
        raise ValueError(msg)


def assert_size_at_dim(x, size, dim, msg):
    """
    Ensure that arrays and tensors in nested dictionary or list or tuple have
    size @size in dim @dim.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        size (int): size that tensors should have at @dim
        dim (int): dimension to check
    """
    map_tensor(x, lambda t, s=size, d=dim, m=msg: assert_size_at_dim_single(t, s, d, m))


def get_shape(x):
    """
    Get all shapes of arrays and tensors in nested dictionary or list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple that contains each array or
            tensor's shape
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.shape,
            np.ndarray: lambda x: x.shape,
            type(None): lambda x: x,
        },
    )


def list_of_flat_dict_to_dict_of_list(list_of_dict):
    """
    Helper function to go from a list of flat dictionaries to a dictionary of lists.
    By "flat" we mean that none of the values are dictionaries, but are numpy arrays,
    floats, etc.

    Args:
        list_of_dict (list): list of flat dictionaries

    Returns:
        dict_of_list (dict): dictionary of lists
    """
    if not isinstance(list_of_dict, list):
        raise TypeError(
            f"Expected a list, got {type(list_of_dict).__name__}"
        )
    dic: collections.OrderedDict[str, list] = collections.OrderedDict()
    for i in range(len(list_of_dict)):
        for k in list_of_dict[i]:
            if k not in dic:
                dic[k] = []
            dic[k].append(list_of_dict[i][k])
    return dic


def flatten_nested_dict_list(d, parent_key="", sep="_", item_key=""):
    """
    Flatten a nested dict or list to a list.

    For example, given a dict
    {
        a: 1
        b: {
            c: 2
        }
        c: 3
    }

    the function would return [(a, 1), (b_c, 2), (c, 3)]

    Args:
        d (dict, list): a nested dict or list to be flattened
        parent_key (str): recursion helper
        sep (str): separator for nesting keys
        item_key (str): recursion helper
    Returns:
        list: a list of (key, value) tuples
    """
    items = []
    if isinstance(d, (tuple, list)):
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        for i, v in enumerate(d):
            items.extend(flatten_nested_dict_list(v, new_key, sep=sep, item_key=str(i)))
        return items
    elif isinstance(d, dict):
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        for k, v in d.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"Dict keys must be strings, got {type(k).__name__}"
                )
            items.extend(flatten_nested_dict_list(v, new_key, sep=sep, item_key=k))
        return items
    else:
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        return [(new_key, d)]


def time_distributed(
    inputs, op, activation=None, inputs_as_kwargs=False, inputs_as_args=False, **kwargs
):
    """
    Apply function @op to all tensors in nested dictionary or list or tuple @inputs in both the
    batch (B) and time (T) dimension, where the tensors are expected to have shape [B, T, ...].
    Will do this by reshaping tensors to [B * T, ...], passing through the op, and then reshaping
    outputs to [B, T, ...].

    Args:
        inputs (list or tuple or dict): a possibly nested dictionary or list or tuple with tensors
            of leading dimensions [B, T, ...]
        op: a layer op that accepts inputs
        activation: activation to apply at the output
        inputs_as_kwargs (bool): whether to feed input as a kwargs dict to the op
        inputs_as_args (bool) whether to feed input as a args list to the op
        kwargs (dict): other kwargs to supply to the op

    Returns:
        outputs (dict or list or tuple): new nested dict-list-tuple with tensors of leading dimension [B, T].
    """
    batch_size, seq_len = flatten_nested_dict_list(inputs)[0][1].shape[:2]
    inputs = join_dimensions(inputs, 0, 1)
    if inputs_as_kwargs:
        outputs = op(**inputs, **kwargs)
    elif inputs_as_args:
        outputs = op(*inputs, **kwargs)
    else:
        outputs = op(inputs, **kwargs)

    if activation is not None:
        outputs = map_tensor(outputs, activation)
    outputs = reshape_dimensions(
        outputs, begin_axis=0, end_axis=0, target_dims=(batch_size, seq_len)
    )
    return outputs


def get_module_by_path(module: nn.Module, path: list[int | str]) -> nn.Module:
    """
    Traverse the module structure using the given path to retrieve a nested module.

    Args:
        module (nn.Module): The starting module (e.g., self.backbone).
        path (List[Union[int, str]]): List of keys/indices to traverse (e.g., ['stem', 0]).

    Returns:
        nn.Module: The module at the specified path.
    """
    return functools.reduce(lambda m, p: m[p] if isinstance(p, int) else getattr(m, p), path, module)  # type: ignore[index]


def set_module_by_path(
    module: nn.Module, path: list[int | str], value: nn.Module
) -> None:
    """
    Traverse the module structure using the given path and set a new value at that location.

    Args:
        module (nn.Module): The starting module (e.g., self.backbone).
        path (List[Union[int, str]]): List of keys/indices to traverse (e.g., ['stem', 0]).
        value (nn.Module): The new module to set at the path.

    Raises:
        ValueError: If the path is empty.
    """
    if not path:
        raise ValueError("Path cannot be empty")
    parent_path = path[:-1]
    key = path[-1]
    parent = get_module_by_path(module, parent_path) if parent_path else module
    if isinstance(key, int):
        parent[key] = value
    else:
        setattr(parent, key, value)
