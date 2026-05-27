"""Tests for versatil.common.tensor_ops module."""

import collections
import re

import pytest
import torch

from versatil.common.tensor_ops import (
    batch_rms,
    clone_tensor_dictionary_with_replacements,
    combined_batch_rms,
    detach_floating_tensor_dictionary,
    dict_apply,
    normalize_tensor_tuple,
    recursive_dict_list_tuple_apply,
    reshape_batch_scale_for_broadcast,
    slice_tensor_dictionary,
    tensor_to_str,
    to_device,
)


@pytest.mark.unit
def test_dict_apply_recurses_over_nested_dicts():
    data = {
        "a": torch.tensor([1, 2]),
        "nested": {"b": torch.tensor([3, 4])},
    }

    result = dict_apply(data, lambda tensor: tensor + 1)

    torch.testing.assert_close(result["a"], torch.tensor([2, 3]))
    torch.testing.assert_close(result["nested"]["b"], torch.tensor([4, 5]))


@pytest.mark.unit
def test_tensor_to_str_formats_tensor_values():
    assert tensor_to_str(torch.tensor([1.2345, 0.00123])) == "[1.23, 0.00123]"


@pytest.mark.unit
def test_tensor_to_str_formats_scalar_tensor_values():
    assert tensor_to_str(torch.tensor(1.2345)) == "[1.23]"


@pytest.mark.unit
class TestTensorDictionaryUtilities:
    def test_clone_with_replacements_preserves_source_dictionary(self):
        source = {"feature": torch.tensor([1.0]), "other": torch.tensor([2.0])}
        replacement = torch.tensor([3.0])

        result = clone_tensor_dictionary_with_replacements(
            values=source,
            replacements={"feature": replacement},
        )

        torch.testing.assert_close(result["feature"], replacement)
        torch.testing.assert_close(source["feature"], torch.tensor([1.0]))
        assert result["other"] is source["other"]

    def test_detach_floating_tensor_dictionary_preserves_integer_tensors(self):
        floating_tensor = torch.tensor([1.0], requires_grad=True)
        integer_tensor = torch.tensor([1])

        result = detach_floating_tensor_dictionary(
            values={"floating": floating_tensor, "integer": integer_tensor}
        )

        assert result["floating"].requires_grad is False
        assert result["integer"] is integer_tensor

    def test_slice_tensor_dictionary_keeps_leading_batch_items(self):
        values = {
            "feature": torch.arange(12).reshape(3, 4),
            "short": torch.arange(4).reshape(1, 4),
        }

        result = slice_tensor_dictionary(values=values, max_batch_size=2)

        torch.testing.assert_close(result["feature"], values["feature"][:2])
        assert result["short"] is values["short"]


@pytest.mark.unit
class TestBatchTensorUtilities:
    def test_reshape_batch_scale_for_broadcast_matches_reference_tensor_rank(self):
        scale = torch.tensor([1.0, 2.0])
        tensor = torch.zeros(2, 3, 4)

        result = reshape_batch_scale_for_broadcast(scale=scale, tensor=tensor)

        assert result.shape == (2, 1, 1)
        torch.testing.assert_close(result[:, 0, 0], scale)

    def test_batch_rms_computes_per_sample_values(self):
        tensor = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

        result = batch_rms(tensor=tensor, eps=1e-6)

        expected = torch.tensor([(9.0 + 16.0) / 2.0, 1e-12]).sqrt()
        torch.testing.assert_close(result, expected)

    def test_batch_rms_rejects_scalar_tensor(self):
        expected_message = "Expected a batched tensor, got a scalar tensor."

        with pytest.raises(ValueError, match=rf"^{re.escape(expected_message)}$"):
            batch_rms(tensor=torch.tensor(1.0), eps=1e-6)

    def test_combined_batch_rms_concatenates_tensors_per_sample(self):
        first = torch.tensor([[3.0], [0.0]])
        second = torch.tensor([[4.0], [0.0]])

        result = combined_batch_rms(tensors=[first, second], eps=1e-6)

        expected = torch.tensor([(9.0 + 16.0) / 2.0, 1e-12]).sqrt()
        torch.testing.assert_close(result, expected)

    def test_combined_batch_rms_rejects_empty_input(self):
        expected_message = "Expected at least one tensor for RMS computation."

        with pytest.raises(ValueError, match=rf"^{re.escape(expected_message)}$"):
            combined_batch_rms(tensors=[], eps=1e-6)

    def test_normalize_tensor_tuple_uses_shared_norm(self):
        first = torch.tensor([3.0])
        second = torch.tensor([4.0])

        result_first, result_second = normalize_tensor_tuple(
            tensors=(first, second),
            eps=1e-6,
        )

        torch.testing.assert_close(result_first, torch.tensor([0.6]))
        torch.testing.assert_close(result_second, torch.tensor([0.8]))


@pytest.mark.unit
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


@pytest.mark.unit
def test_to_device_recurses_and_preserves_non_tensor_values():
    data = {"a": torch.tensor([1]), "b": [None, "keep"]}

    result = to_device(data, "cpu")

    torch.testing.assert_close(result["a"], torch.tensor([1]))
    assert result["a"].device.type == "cpu"
    assert result["b"] == [None, "keep"]
