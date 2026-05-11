"""Tests for versatil.common.tensor_ops module."""

import collections
import re

import pytest
import torch

from versatil.common.tensor_ops import (
    dict_apply,
    recursive_dict_list_tuple_apply,
    tensor_to_str,
    to_device,
)

pytestmark = pytest.mark.unit


def test_dict_apply_recurses_over_nested_dicts():
    data = {
        "a": torch.tensor([1, 2]),
        "nested": {"b": torch.tensor([3, 4])},
    }

    result = dict_apply(data, lambda tensor: tensor + 1)

    torch.testing.assert_close(result["a"], torch.tensor([2, 3]))
    torch.testing.assert_close(result["nested"]["b"], torch.tensor([4, 5]))


def test_tensor_to_str_formats_tensor_values():
    assert tensor_to_str(torch.tensor([1.2345, 0.00123])) == "[1.23, 0.00123]"


def test_tensor_to_str_formats_scalar_tensor_values():
    assert tensor_to_str(torch.tensor(1.2345)) == "[1.23]"


class TestRecursiveDictListTupleApply:
    def test_applies_handler_inside_nested_containers(self):
        data = collections.OrderedDict(
            [
                ("a", torch.tensor([1])),
                ("b", [torch.tensor([2]), (torch.tensor([3]),)]),
            ]
        )

        result = recursive_dict_list_tuple_apply(
            data=data,
            type_handler_map={torch.Tensor: lambda tensor: tensor + 1},
        )

        torch.testing.assert_close(result["a"], torch.tensor([2]))
        torch.testing.assert_close(result["b"][0], torch.tensor([3]))
        torch.testing.assert_close(result["b"][1][0], torch.tensor([4]))

    @pytest.mark.parametrize(
        ("handler_type", "data", "expected_message"),
        [
            (
                dict,
                {"a": torch.tensor(1)},
                "dict cannot be handled by type_func_dict",
            ),
            (
                list,
                [torch.tensor(1)],
                "list/tuple cannot be handled by type_func_dict",
            ),
            (
                tuple,
                (torch.tensor(1),),
                "list/tuple cannot be handled by type_func_dict",
            ),
        ],
    )
    def test_rejects_container_handlers(
        self,
        handler_type: type,
        data,
        expected_message: str,
    ):
        with pytest.raises(ValueError, match=rf"^{re.escape(expected_message)}$"):
            recursive_dict_list_tuple_apply(
                data=data,
                type_handler_map={handler_type: lambda value: value},
            )

    def test_rejects_unsupported_leaf_type(self):
        expected_message = "Unsupported type: <class 'int'>"

        with pytest.raises(
            NotImplementedError, match=rf"^{re.escape(expected_message)}$"
        ):
            recursive_dict_list_tuple_apply(
                data=1,
                type_handler_map={torch.Tensor: lambda tensor: tensor},
            )


def test_to_device_recurses_and_preserves_non_tensor_values():
    data = {"a": torch.tensor([1]), "b": [None, "keep"]}

    result = to_device(data, "cpu")

    torch.testing.assert_close(result["a"], torch.tensor([1]))
    assert result["a"].device.type == "cpu"
    assert result["b"] == [None, "keep"]
