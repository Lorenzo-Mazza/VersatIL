"""Tests for versatil.common.tensor_ops module."""

import collections
import re
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn

from versatil.common.tensor_ops import (
    clone,
    contiguous,
    detach,
    dict_apply,
    dict_apply_reduce,
    dict_apply_split,
    expand_at_single,
    flatten_nested_dict_list,
    flatten_single,
    gather_along_dim_with_dim_single,
    gather_sequence_single,
    get_module_by_path,
    get_shape,
    index_at_time,
    join_dimensions,
    list_of_flat_dict_to_dict_of_list,
    map_ndarray,
    map_tensor,
    map_tensor_ndarray,
    named_reduce_single,
    optimizer_to,
    pad_remaining_dims,
    pad_sequence_single,
    recursive_dict_list_tuple_apply,
    repeat_by_expand_at,
    replace_submodules,
    reshape_dimensions_single,
    set_module_by_path,
    tensor_to_str,
    time_distributed,
    to_batch,
    to_device,
    to_float,
    to_list,
    to_numpy,
    to_one_hot,
    to_one_hot_single,
    to_sequence,
    to_tensor,
    to_torch,
    to_uint8,
    unsqueeze,
)


@pytest.fixture
def tensor_factory(rng: np.random.Generator) -> Callable[..., torch.Tensor]:
    def factory(
        shape: tuple[int, ...] = (3, 4),
    ) -> torch.Tensor:
        return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))

    return factory


@pytest.fixture
def ndarray_factory(rng: np.random.Generator) -> Callable[..., np.ndarray]:
    def factory(
        shape: tuple[int, ...] = (3, 4),
    ) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    return factory


@pytest.fixture
def nested_dict_factory(
    tensor_factory: Callable[..., torch.Tensor],
) -> Callable[..., dict[str, torch.Tensor]]:
    def factory(
        shape: tuple[int, ...] = (2, 3),
    ) -> dict[str, torch.Tensor]:
        return {
            "alpha": tensor_factory(shape=shape),
            "beta": tensor_factory(shape=shape),
        }

    return factory


@pytest.mark.unit
class TestDictApply:
    def test_applies_function_to_each_value(
        self,
        nested_dict_factory: Callable[..., dict[str, torch.Tensor]],
    ):
        input_dict = nested_dict_factory(shape=(2, 3))
        result = dict_apply(input_dict, func=lambda x: x * 2.0)
        torch.testing.assert_close(result["alpha"], input_dict["alpha"] * 2.0)
        torch.testing.assert_close(result["beta"], input_dict["beta"] * 2.0)

    def test_recurses_into_nested_dicts(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        tensor = tensor_factory(shape=(2,))
        nested = {"outer": {"inner": tensor}}
        result = dict_apply(nested, func=lambda x: x + 1.0)
        torch.testing.assert_close(result["outer"]["inner"], tensor + 1.0)


@pytest.mark.unit
class TestPadRemainingDims:
    def test_pads_to_match_higher_dimensional_target(self):
        source = torch.tensor([2.0, 3.0])
        target = torch.zeros(2, 4, 5)
        result = pad_remaining_dims(source, target)
        assert result.shape == (2, 1, 1)
        assert result[0, 0, 0].item() == pytest.approx(2.0)
        assert result[1, 0, 0].item() == pytest.approx(3.0)

    def test_no_padding_when_same_ndim(self):
        source = torch.ones(2, 3)
        target = torch.zeros(2, 3)
        result = pad_remaining_dims(source, target)
        assert result.shape == (2, 3)

    def test_raises_on_shape_prefix_mismatch(self):
        source = torch.tensor([1.0, 2.0, 3.0])
        target = torch.zeros(2, 4)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Shape mismatch: x.shape {source.shape} is not a prefix of "
                f"target.shape {target.shape}"
            ),
        ):
            pad_remaining_dims(source, target)


@pytest.mark.unit
class TestDictApplySplit:
    def test_splits_dict_values_into_sub_dicts(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        tensor_a = tensor_factory(shape=(4, 6))
        tensor_b = tensor_factory(shape=(4, 6))
        input_dict = {"a": tensor_a, "b": tensor_b}

        def split_in_half(value: torch.Tensor) -> dict[str, torch.Tensor]:
            first_half, second_half = value.chunk(2, dim=0)
            return {"first": first_half, "second": second_half}

        result = dict_apply_split(input_dict, split_func=split_in_half)
        assert result["first"]["a"].shape == (2, 6)
        torch.testing.assert_close(result["first"]["a"], tensor_a[:2])
        torch.testing.assert_close(result["second"]["b"], tensor_b[2:])


@pytest.mark.unit
class TestDictApplyReduce:
    def test_reduces_list_of_dicts_by_key(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        dict_1 = {"x": tensor_factory(shape=(3,))}
        dict_2 = {"x": tensor_factory(shape=(3,))}
        result = dict_apply_reduce(
            [dict_1, dict_2],
            reduce_func=lambda tensors: torch.stack(tensors).sum(dim=0),
        )
        torch.testing.assert_close(result["x"], dict_1["x"] + dict_2["x"])


@pytest.mark.unit
class TestRecursiveDictListTupleApply:
    def test_applies_to_dict_values(self):
        data = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
        result = recursive_dict_list_tuple_apply(
            data, {torch.Tensor: lambda x: x * 10.0}
        )
        torch.testing.assert_close(result["a"], torch.tensor([10.0, 20.0]))
        torch.testing.assert_close(result["b"], torch.tensor([30.0]))

    def test_preserves_ordered_dict_type(self):
        data = collections.OrderedDict(
            [
                ("z", torch.tensor(1.0)),
                ("a", torch.tensor(2.0)),
            ]
        )
        result = recursive_dict_list_tuple_apply(data, {torch.Tensor: lambda x: x})
        assert isinstance(result, collections.OrderedDict)
        assert list(result.keys()) == ["z", "a"]

    def test_traverses_lists_and_preserves_type(self):
        data = [torch.tensor(1.0), torch.tensor(2.0)]
        result = recursive_dict_list_tuple_apply(
            data, {torch.Tensor: lambda x: x + 100.0}
        )
        assert isinstance(result, list)
        assert result[0].item() == pytest.approx(101.0)
        assert result[1].item() == pytest.approx(102.0)

    def test_traverses_tuples_and_preserves_type(self):
        data = (torch.tensor(5.0),)
        result = recursive_dict_list_tuple_apply(
            data, {torch.Tensor: lambda x: x * 2.0}
        )
        assert isinstance(result, tuple)
        assert result[0].item() == pytest.approx(10.0)

    @pytest.mark.parametrize(
        "forbidden_type, label",
        [
            (list, "list"),
            (tuple, "tuple"),
            (dict, "dict"),
        ],
    )
    def test_raises_when_container_type_in_func_dict(
        self,
        forbidden_type: type,
        label: str,
    ):
        with pytest.raises(
            TypeError,
            match=re.escape(f"type_func_dict must not contain '{label}' as a key. "),
        ):
            recursive_dict_list_tuple_apply({}, {forbidden_type: lambda x: x})

    def test_raises_for_unhandled_type(self):
        with pytest.raises(
            NotImplementedError,
            match=re.escape("Cannot handle data type <class 'int'>"),
        ):
            recursive_dict_list_tuple_apply(42, {torch.Tensor: lambda x: x})


@pytest.mark.unit
class TestMapTensor:
    def test_applies_func_to_tensors_and_skips_none(self):
        data = {"tensor": torch.tensor([1.0, 2.0]), "none_val": None}
        result = map_tensor(data, func=lambda x: x * 3.0)
        torch.testing.assert_close(result["tensor"], torch.tensor([3.0, 6.0]))
        assert result["none_val"] is None

    def test_handles_nested_list_with_tensors(self):
        data = {"items": [torch.tensor(5.0), torch.tensor(10.0)]}
        result = map_tensor(data, func=lambda x: x - 1.0)
        assert result["items"][0].item() == pytest.approx(4.0)
        assert result["items"][1].item() == pytest.approx(9.0)


@pytest.mark.unit
class TestClone:
    def test_cloned_tensor_is_independent_copy(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        original = {"data": tensor_factory(shape=(3,))}
        cloned = clone(original)
        torch.testing.assert_close(cloned["data"], original["data"])
        cloned["data"][0] = 9999.0
        assert original["data"][0].item() != 9999.0

    def test_cloned_ndarray_is_independent_copy(
        self,
        ndarray_factory: Callable[..., np.ndarray],
    ):
        original = {"data": ndarray_factory(shape=(3,))}
        cloned = clone(original)
        np.testing.assert_array_equal(cloned["data"], original["data"])
        cloned["data"][0] = 9999.0
        assert original["data"][0] != 9999.0


@pytest.mark.unit
class TestDetach:
    def test_detaches_tensor_from_computation_graph(self):
        tensor = torch.tensor([1.0, 2.0], requires_grad=True)
        result = detach({"x": tensor})
        assert not result["x"].requires_grad
        torch.testing.assert_close(result["x"], torch.tensor([1.0, 2.0]))


@pytest.mark.unit
class TestToBatch:
    def test_adds_leading_batch_dim_to_tensor(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        tensor = tensor_factory(shape=(3, 4))
        result = to_batch({"x": tensor})
        assert result["x"].shape == (1, 3, 4)
        torch.testing.assert_close(result["x"][0], tensor)

    def test_adds_leading_batch_dim_to_ndarray(
        self,
        ndarray_factory: Callable[..., np.ndarray],
    ):
        array = ndarray_factory(shape=(3, 4))
        result = to_batch({"x": array})
        assert result["x"].shape == (1, 3, 4)
        np.testing.assert_array_equal(result["x"][0], array)


@pytest.mark.unit
class TestToSequence:
    def test_inserts_time_dim_at_position_1(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        tensor = tensor_factory(shape=(2, 5))
        result = to_sequence({"x": tensor})
        assert result["x"].shape == (2, 1, 5)
        torch.testing.assert_close(result["x"][:, 0, :], tensor)


@pytest.mark.unit
class TestIndexAtTime:
    def test_selects_specific_time_step(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        tensor = tensor_factory(shape=(2, 3, 4))
        result = index_at_time({"x": tensor}, ind=1)
        assert result["x"].shape == (2, 4)
        torch.testing.assert_close(result["x"], tensor[:, 1, :])


@pytest.mark.unit
class TestUnsqueeze:
    @pytest.mark.parametrize(
        "dim, expected_shape",
        [
            (0, (1, 3, 4)),
            (1, (3, 1, 4)),
            (2, (3, 4, 1)),
        ],
    )
    def test_inserts_dimension_at_correct_position(
        self,
        tensor_factory: Callable[..., torch.Tensor],
        dim: int,
        expected_shape: tuple[int, ...],
    ):
        tensor = tensor_factory(shape=(3, 4))
        result = unsqueeze({"x": tensor}, dim=dim)
        assert result["x"].shape == expected_shape


@pytest.mark.unit
class TestContiguous:
    def test_makes_non_contiguous_tensor_contiguous(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        tensor = tensor_factory(shape=(3, 4)).t()
        assert not tensor.is_contiguous()
        result = contiguous({"x": tensor})
        assert result["x"].is_contiguous()
        torch.testing.assert_close(result["x"], tensor)


@pytest.mark.unit
class TestToDevice:
    def test_moves_tensor_to_target_device(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        tensor = tensor_factory(shape=(2, 3))
        result = to_device({"x": tensor}, device=torch.device("cpu"))
        assert result["x"].device.type == "cpu"

    def test_preserves_string_values(self):
        data = {"text": "hello", "tensor": torch.tensor(1.0)}
        result = to_device(data, device=torch.device("cpu"))
        assert result["text"] == "hello"


@pytest.mark.unit
class TestToFloat:
    def test_converts_int_tensor_to_float32(self):
        tensor = torch.tensor([1, 2, 3])
        result = to_float({"x": tensor})
        assert result["x"].dtype == torch.float32
        torch.testing.assert_close(result["x"], torch.tensor([1.0, 2.0, 3.0]))

    def test_converts_int_ndarray_to_float32(self):
        array = np.array([1, 2, 3], dtype=np.int64)
        result = to_float({"x": array})
        assert result["x"].dtype == np.float32


@pytest.mark.unit
class TestToUint8:
    def test_converts_tensor_to_uint8(self):
        tensor = torch.tensor([0.0, 128.0, 255.0])
        result = to_uint8({"x": tensor})
        assert result["x"].dtype == torch.uint8
        assert result["x"].tolist() == [0, 128, 255]


@pytest.mark.unit
class TestToTensor:
    def test_converts_ndarray_to_torch_tensor(self):
        array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = to_tensor({"x": array})
        assert isinstance(result["x"], torch.Tensor)
        torch.testing.assert_close(result["x"], torch.tensor([1.0, 2.0, 3.0]))

    def test_leaves_existing_tensor_unchanged(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        tensor = tensor_factory(shape=(3,))
        result = to_tensor({"x": tensor})
        torch.testing.assert_close(result["x"], tensor)


@pytest.mark.unit
class TestToNumpy:
    def test_converts_tensor_to_ndarray(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy({"x": tensor})
        assert isinstance(result["x"], np.ndarray)
        np.testing.assert_array_almost_equal(result["x"], [1.0, 2.0, 3.0])


@pytest.mark.unit
class TestToList:
    def test_converts_tensor_to_python_list(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = to_list({"x": tensor})
        assert result["x"] == [1.0, 2.0, 3.0]

    def test_converts_ndarray_to_python_list(self):
        array = np.array([4.0, 5.0, 6.0])
        result = to_list({"x": array})
        assert result["x"] == [4.0, 5.0, 6.0]


@pytest.mark.unit
class TestTensorToStr:
    def test_scalar_tensor_formatted_with_4_decimals(self):
        tensor = torch.tensor(3.14159)
        result = tensor_to_str(tensor)
        assert result == "3.1416"

    def test_1d_tensor_formatted_as_bracketed_list(self):
        tensor = torch.tensor([1.0, 2.5, 0.0])
        result = tensor_to_str(tensor)
        assert result == "[1.0000, 2.5000, 0.0000]"


@pytest.mark.unit
class TestToOneHotSingle:
    def test_produces_correct_one_hot_vectors(self):
        labels = torch.tensor([0, 2, 1])
        result = to_one_hot_single(labels, num_class=3)
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        torch.testing.assert_close(result, expected)

    def test_handles_batched_labels(self):
        labels = torch.tensor([[0, 1], [2, 0]])
        result = to_one_hot_single(labels, num_class=4)
        assert result.shape == (2, 2, 4)
        # [0, 0]: class 0 -> [1, 0, 0, 0]
        torch.testing.assert_close(result[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
        # [1, 0]: class 2 -> [0, 0, 1, 0]
        torch.testing.assert_close(result[1, 0], torch.tensor([0.0, 0.0, 1.0, 0.0]))


@pytest.mark.unit
class TestFlattenSingle:
    @pytest.mark.parametrize(
        "shape, begin_axis, expected_shape",
        [
            ((2, 3, 4), 1, (2, 12)),
            ((2, 3, 4), 0, (24,)),
            ((2, 3, 4, 5), 2, (2, 3, 20)),
        ],
    )
    def test_flattens_from_begin_axis(
        self,
        tensor_factory: Callable[..., torch.Tensor],
        shape: tuple[int, ...],
        begin_axis: int,
        expected_shape: tuple[int, ...],
    ):
        tensor = tensor_factory(shape=shape)
        result = flatten_single(tensor, begin_axis=begin_axis)
        assert result.shape == expected_shape
        assert result.numel() == tensor.numel()


@pytest.mark.unit
class TestReshapeDimensionsSingle:
    def test_splits_dimension_into_multiple(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        # (6, 4) -> reshape dim 0:0 to (2, 3) -> (2, 3, 4)
        tensor = tensor_factory(shape=(6, 4))
        result = reshape_dimensions_single(
            tensor, begin_axis=0, end_axis=0, target_dims=[2, 3]
        )
        assert result.shape == (2, 3, 4)

    def test_joins_dimensions_into_one(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        # (2, 3, 4) -> reshape dims 0:1 to (-1,) -> (6, 4)
        tensor = tensor_factory(shape=(2, 3, 4))
        result = reshape_dimensions_single(
            tensor, begin_axis=0, end_axis=1, target_dims=[-1]
        )
        assert result.shape == (6, 4)

    def test_raises_when_begin_axis_greater_than_end_axis(self):
        tensor = torch.zeros(2, 3, 4)
        with pytest.raises(
            ValueError,
            match=re.escape("begin_axis (2) must be <= end_axis (1)"),
        ):
            reshape_dimensions_single(tensor, begin_axis=2, end_axis=1, target_dims=[6])

    def test_raises_when_begin_axis_negative(self):
        tensor = torch.zeros(2, 3)
        with pytest.raises(
            ValueError,
            match=re.escape("begin_axis (-1) must be >= 0"),
        ):
            reshape_dimensions_single(
                tensor, begin_axis=-1, end_axis=0, target_dims=[2]
            )

    def test_raises_when_end_axis_exceeds_ndim(self):
        tensor = torch.zeros(2, 3)
        with pytest.raises(
            ValueError,
            match=re.escape("end_axis (5) must be < number of dimensions (2)"),
        ):
            reshape_dimensions_single(tensor, begin_axis=0, end_axis=5, target_dims=[2])

    def test_raises_when_target_dims_not_sequence(self):
        tensor = torch.zeros(2, 3)
        with pytest.raises(
            TypeError,
            match=re.escape("target_dims must be a tuple or list, got int"),
        ):
            reshape_dimensions_single(tensor, begin_axis=0, end_axis=0, target_dims=2)


@pytest.mark.unit
class TestJoinDimensions:
    def test_joins_consecutive_dimensions(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        # (2, 3, 4) -> join dims 0:1 -> (6, 4)
        tensor = tensor_factory(shape=(2, 3, 4))
        result = join_dimensions({"x": tensor}, begin_axis=0, end_axis=1)
        assert result["x"].shape == (6, 4)

    def test_preserves_element_order(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        tensor = tensor_factory(shape=(2, 3, 4))
        result = join_dimensions({"x": tensor}, begin_axis=0, end_axis=1)
        torch.testing.assert_close(result["x"][0], tensor[0, 0])


@pytest.mark.unit
class TestExpandAtSingle:
    def test_expands_singleton_dimension(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        tensor = tensor_factory(shape=(2, 1, 4))
        result = expand_at_single(tensor, size=5, dim=1)
        assert result.shape == (2, 5, 4)
        torch.testing.assert_close(result[:, 0, :], result[:, 3, :])

    def test_raises_when_dim_exceeds_ndim(self):
        tensor = torch.zeros(2, 3)
        with pytest.raises(
            ValueError,
            match=re.escape("dim (2) must be < number of dimensions (2)"),
        ):
            expand_at_single(tensor, size=5, dim=2)

    def test_raises_when_dim_not_singleton(self):
        tensor = torch.zeros(2, 3)
        with pytest.raises(
            ValueError,
            match=re.escape("Dimension 1 must have size 1 for expansion, got 3"),
        ):
            expand_at_single(tensor, size=5, dim=1)


@pytest.mark.unit
class TestRepeatByExpandAt:
    def test_repeats_dimension_correctly(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        # (2, 3) -> repeat dim 0 by 4 -> (8, 3)
        # unsqueeze_expand_at inserts dim 1 and expands: (2, 4, 3)
        # join_dimensions joins 0:1 -> (8, 3)
        tensor = tensor_factory(shape=(2, 3))
        result = repeat_by_expand_at({"x": tensor}, repeats=4, dim=0)
        assert result["x"].shape == (8, 3)
        # Elements [0..3] should all equal tensor[0]
        torch.testing.assert_close(result["x"][0], tensor[0])
        torch.testing.assert_close(result["x"][3], tensor[0])
        # Elements [4..7] should all equal tensor[1]
        torch.testing.assert_close(result["x"][4], tensor[1])
        torch.testing.assert_close(result["x"][7], tensor[1])


@pytest.mark.unit
class TestNamedReduceSingle:
    def test_sum_along_dimension(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = named_reduce_single(tensor, reduction="sum", dim=1)
        torch.testing.assert_close(result, torch.tensor([3.0, 7.0]))

    def test_mean_along_dimension(self):
        tensor = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        result = named_reduce_single(tensor, reduction="mean", dim=1)
        torch.testing.assert_close(result, torch.tensor([3.0, 7.0]))

    def test_max_along_dimension(self):
        tensor = torch.tensor([[1.0, 5.0], [3.0, 2.0]])
        result = named_reduce_single(tensor, reduction="max", dim=1)
        torch.testing.assert_close(result, torch.tensor([5.0, 3.0]))

    def test_flatten_from_dimension(self):
        tensor = torch.zeros(2, 3, 4)
        result = named_reduce_single(tensor, reduction="flatten", dim=1)
        assert result.shape == (2, 12)

    def test_raises_for_invalid_reduction_name(self):
        tensor = torch.zeros(2, 3)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "reduction must be one of ['sum', 'max', 'mean', 'flatten'], "
                "got 'invalid'"
            ),
        ):
            named_reduce_single(tensor, reduction="invalid", dim=0)

    def test_raises_when_dim_exceeds_ndim(self):
        tensor = torch.zeros(2, 3)
        with pytest.raises(
            ValueError,
            match=re.escape("Tensor has 2 dimensions, but dim (2) requires at least 3"),
        ):
            named_reduce_single(tensor, reduction="sum", dim=2)


@pytest.mark.unit
class TestGatherAlongDimWithDimSingle:
    def test_gathers_correct_elements_from_3d_tensor(self):
        # (B=3, T=4, D=2): gather time steps per batch
        tensor = torch.arange(24).reshape(3, 4, 2).float()
        indices = torch.tensor([0, 2, 1])
        result = gather_along_dim_with_dim_single(
            tensor, target_dim=1, source_dim=0, indices=indices
        )
        assert result.shape == (3, 2)
        # batch 0, time 0: [0, 1]
        torch.testing.assert_close(result[0], torch.tensor([0.0, 1.0]))
        # batch 1, time 2: [12, 13]
        torch.testing.assert_close(result[1], torch.tensor([12.0, 13.0]))
        # batch 2, time 1: [18, 19]
        torch.testing.assert_close(result[2], torch.tensor([18.0, 19.0]))

    def test_raises_for_non_1d_indices(self):
        tensor = torch.zeros(2, 3)
        indices = torch.zeros(2, 1).long()
        with pytest.raises(
            ValueError,
            match=re.escape(f"indices must be 1D, got shape {indices.shape}"),
        ):
            gather_along_dim_with_dim_single(
                tensor, target_dim=1, source_dim=0, indices=indices
            )

    def test_raises_for_source_dim_shape_mismatch(self):
        tensor = torch.zeros(3, 4)
        indices = torch.tensor([0, 1])  # length 2, but source dim 0 has size 3
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"x.shape[0] ({tensor.shape[0]}) must match "
                f"indices.shape[0] ({indices.shape[0]})"
            ),
        ):
            gather_along_dim_with_dim_single(
                tensor, target_dim=1, source_dim=0, indices=indices
            )


@pytest.mark.unit
class TestGatherSequenceSingle:
    def test_gathers_per_batch_time_step(self):
        # [B=2, T=3, D=4]
        sequence = torch.arange(24).reshape(2, 3, 4).float()
        indices = torch.tensor([2, 0])
        result = gather_sequence_single(sequence, indices=indices)
        assert result.shape == (2, 4)
        # batch 0, time 2
        torch.testing.assert_close(result[0], sequence[0, 2])
        # batch 1, time 0
        torch.testing.assert_close(result[1], sequence[1, 0])


@pytest.mark.unit
class TestPadSequenceSingle:
    def test_unbatched_pad_same_replicates_boundaries(self):
        sequence = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = pad_sequence_single(
            sequence, padding=(1, 1), batched=False, pad_same=True
        )
        assert result.shape == (5, 2)
        torch.testing.assert_close(result[0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(result[1:4], sequence)
        torch.testing.assert_close(result[4], torch.tensor([5.0, 6.0]))

    def test_unbatched_pad_with_constant_value(self):
        sequence = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = pad_sequence_single(
            sequence,
            padding=(2, 0),
            batched=False,
            pad_same=False,
            pad_values=0.0,
        )
        assert result.shape == (4, 2)
        torch.testing.assert_close(result[0], torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(result[1], torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(result[2:], sequence)

    def test_batched_pad_same_indexes_correct_time_dimension(self):
        # [B=2, T=3, D=2]
        batch = torch.tensor(
            [
                [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
                [[40.0, 41.0], [50.0, 51.0], [60.0, 61.0]],
            ]
        )
        result = pad_sequence_single(batch, padding=(1, 1), batched=True, pad_same=True)
        assert result.shape == (2, 5, 2)
        # Begin pad: first time step of each batch
        torch.testing.assert_close(result[0, 0], torch.tensor([10.0, 11.0]))
        torch.testing.assert_close(result[1, 0], torch.tensor([40.0, 41.0]))
        # End pad: last time step of each batch
        torch.testing.assert_close(result[0, 4], torch.tensor([30.0, 31.0]))
        torch.testing.assert_close(result[1, 4], torch.tensor([60.0, 61.0]))
        # Original data preserved in the middle
        torch.testing.assert_close(result[:, 1:4, :], batch)

    def test_batched_pad_with_constant_value(self):
        batch = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        )
        result = pad_sequence_single(
            batch,
            padding=(0, 2),
            batched=True,
            pad_same=False,
            pad_values=-1.0,
        )
        assert result.shape == (2, 4, 2)
        torch.testing.assert_close(result[:, :2, :], batch)
        torch.testing.assert_close(result[:, 2:, :], torch.full((2, 2, 2), -1.0))

    def test_numpy_unbatched_pad_same(self):
        sequence = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = pad_sequence_single(
            sequence, padding=(1, 0), batched=False, pad_same=True
        )
        assert isinstance(result, np.ndarray)
        assert result.shape == (4, 2)
        np.testing.assert_array_equal(result[0], [1.0, 2.0])

    def test_zero_padding_returns_original(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        tensor = tensor_factory(shape=(3, 4))
        result = pad_sequence_single(
            tensor, padding=(0, 0), batched=False, pad_same=True
        )
        torch.testing.assert_close(result, tensor)

    def test_raises_for_invalid_sequence_type(self):
        with pytest.raises(
            TypeError,
            match=re.escape("seq must be np.ndarray or torch.Tensor, got list"),
        ):
            pad_sequence_single([1, 2, 3], padding=(1, 1), batched=False, pad_same=True)

    def test_raises_when_pad_same_false_without_pad_values(self):
        tensor = torch.zeros(3, 4)
        with pytest.raises(
            ValueError,
            match=re.escape("pad_values must be provided when pad_same is False"),
        ):
            pad_sequence_single(tensor, padding=(1, 0), batched=False, pad_same=False)

    def test_raises_for_non_float_pad_values(self):
        tensor = torch.zeros(3, 4)
        with pytest.raises(
            TypeError,
            match=re.escape("pad_values must be a float, got int"),
        ):
            pad_sequence_single(
                tensor,
                padding=(1, 0),
                batched=False,
                pad_same=False,
                pad_values=0,
            )


@pytest.mark.unit
class TestGetShape:
    def test_returns_shapes_for_nested_tensors(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        data = {
            "tensor": tensor_factory(shape=(2, 3)),
            "none": None,
        }
        shapes = get_shape(data)
        assert shapes["tensor"] == torch.Size([2, 3])
        assert shapes["none"] is None


@pytest.mark.unit
class TestListOfFlatDictToDictOfList:
    def test_transposes_list_to_dict(self):
        list_of_dicts = [
            {"x": 1, "y": 2},
            {"x": 3, "y": 4},
        ]
        result = list_of_flat_dict_to_dict_of_list(list_of_dicts)
        assert result["x"] == [1, 3]
        assert result["y"] == [2, 4]

    def test_result_is_ordered_dict(self):
        result = list_of_flat_dict_to_dict_of_list([{"a": 1}])
        assert isinstance(result, collections.OrderedDict)

    def test_handles_missing_keys_in_some_dicts(self):
        list_of_dicts = [{"x": 1}, {"x": 2, "y": 3}]
        result = list_of_flat_dict_to_dict_of_list(list_of_dicts)
        assert result["x"] == [1, 2]
        assert result["y"] == [3]

    def test_raises_for_non_list_input(self):
        with pytest.raises(
            TypeError,
            match=re.escape("Expected a list, got dict"),
        ):
            list_of_flat_dict_to_dict_of_list({"a": 1})


@pytest.mark.unit
class TestFlattenNestedDictList:
    def test_flattens_simple_dict(self):
        data = {"a": 1, "b": 2}
        result = flatten_nested_dict_list(data)
        assert ("a", 1) in result
        assert ("b", 2) in result

    def test_flattens_nested_dict_with_underscore_separator(self):
        data = {"a": 1, "b": {"c": 2}}
        result = dict(flatten_nested_dict_list(data))
        assert result["a"] == 1
        assert result["b_c"] == 2

    def test_flattens_list_with_index_as_key(self):
        data = [10, 20, 30]
        result = flatten_nested_dict_list(data)
        assert result == [("0", 10), ("1", 20), ("2", 30)]

    def test_custom_separator(self):
        data = {"a": {"b": 1}}
        result = dict(flatten_nested_dict_list(data, sep="."))
        assert result["a.b"] == 1

    def test_raises_for_non_string_dict_keys(self):
        data = {42: "value"}
        with pytest.raises(
            TypeError,
            match=re.escape("Dict keys must be strings, got int"),
        ):
            flatten_nested_dict_list(data)


@pytest.mark.unit
class TestTimeDistributed:
    def test_applies_operation_across_batch_and_time(self):
        linear = nn.Linear(4, 8, bias=False)
        tensor = torch.ones(2, 3, 4)
        result = time_distributed(
            {"data": tensor},
            op=lambda x: linear(x["data"]),
        )
        assert result.shape == (2, 3, 8)

    def test_preserves_batch_time_structure(
        self,
        tensor_factory: Callable[..., torch.Tensor],
    ):
        tensor = tensor_factory(shape=(2, 3, 4))
        result = time_distributed(
            {"data": tensor},
            op=lambda x: x["data"],
        )
        assert result.shape == (2, 3, 4)


@pytest.mark.unit
class TestReplaceSubmodules:
    def test_replaces_all_matching_modules(self):
        model = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.Linear(4, 2),
            nn.BatchNorm1d(2),
        )
        replaced = replace_submodules(
            root_module=model,
            predicate=lambda m: isinstance(m, nn.BatchNorm1d),
            func=lambda m: nn.Identity(),
        )
        batchnorm_count = sum(
            1 for m in replaced.modules() if isinstance(m, nn.BatchNorm1d)
        )
        identity_count = sum(
            1
            for m in replaced.modules()
            if isinstance(m, nn.Identity) and m is not replaced
        )
        assert batchnorm_count == 0
        assert identity_count == 2

    def test_replaces_root_module_when_matching(self):
        model = nn.ReLU()
        result = replace_submodules(
            root_module=model,
            predicate=lambda m: isinstance(m, nn.ReLU),
            func=lambda m: nn.GELU(),
        )
        assert isinstance(result, nn.GELU)

    def test_raises_when_replacement_still_matches_predicate(self):
        model = nn.Sequential(nn.Linear(4, 4))
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Failed to replace all matching submodules. "
                "1 modules still match the predicate."
            ),
        ):
            replace_submodules(
                root_module=model,
                predicate=lambda m: isinstance(m, nn.Linear),
                func=lambda m: nn.Linear(4, 4),
            )


@pytest.mark.unit
class TestGetModuleByPath:
    def test_retrieves_module_by_attribute_name(self):
        container = nn.Module()
        container.backbone = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        retrieved = get_module_by_path(container, path=["backbone", 0])
        assert isinstance(retrieved, nn.Linear)
        assert retrieved.in_features == 4

    def test_empty_path_returns_root(self):
        model = nn.Linear(4, 8)
        retrieved = get_module_by_path(model, path=[])
        assert retrieved is model


@pytest.mark.unit
class TestSetModuleByPath:
    def test_sets_module_at_integer_index(self):
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        new_module = nn.GELU()
        set_module_by_path(model, path=[1], value=new_module)
        assert isinstance(model[1], nn.GELU)

    def test_sets_module_at_named_attribute(self):
        container = nn.Module()
        container.layer = nn.Linear(4, 8)
        new_layer = nn.Linear(4, 16)
        set_module_by_path(container, path=["layer"], value=new_layer)
        assert container.layer.out_features == 16

    def test_raises_on_empty_path(self):
        model = nn.Linear(4, 8)
        with pytest.raises(
            ValueError,
            match=re.escape("Path cannot be empty"),
        ):
            set_module_by_path(model, path=[], value=nn.ReLU())


@pytest.mark.unit
class TestOptimizerTo:
    def test_moves_optimizer_state_tensors_to_device(self):
        model = nn.Linear(4, 4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # Run a step to create optimizer state
        output = model(torch.ones(2, 4))
        output.sum().backward()
        optimizer.step()
        result = optimizer_to(optimizer, device=torch.device("cpu"))
        for state in result.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    assert value.device.type == "cpu"

    def test_returns_same_optimizer_instance(self):
        model = nn.Linear(4, 4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        output = model(torch.ones(2, 4))
        output.sum().backward()
        optimizer.step()
        result = optimizer_to(optimizer, device=torch.device("cpu"))
        assert result is optimizer


@pytest.mark.unit
class TestMapNdarray:
    def test_applies_function_to_ndarrays_and_skips_none(
        self,
        ndarray_factory: Callable[..., np.ndarray],
    ):
        array = ndarray_factory(shape=(3,))
        data = {"array": array, "none_val": None}
        result = map_ndarray(data, func=lambda x: x * 2.0)
        np.testing.assert_array_almost_equal(result["array"], array * 2.0)
        assert result["none_val"] is None

    def test_handles_nested_structure_with_ndarrays(
        self,
        ndarray_factory: Callable[..., np.ndarray],
    ):
        array = ndarray_factory(shape=(2,))
        data = {"items": [array]}
        result = map_ndarray(data, func=lambda x: x + 10.0)
        np.testing.assert_array_almost_equal(result["items"][0], array + 10.0)


@pytest.mark.unit
class TestMapTensorNdarray:
    def test_applies_separate_functions_to_tensors_and_ndarrays(
        self,
        tensor_factory: Callable[..., torch.Tensor],
        ndarray_factory: Callable[..., np.ndarray],
    ):
        tensor = tensor_factory(shape=(3,))
        array = ndarray_factory(shape=(3,))
        data = {"tensor": tensor, "array": array, "none_val": None}
        result = map_tensor_ndarray(
            data,
            tensor_func=lambda x: x * 2.0,
            ndarray_func=lambda x: x * 3.0,
        )
        torch.testing.assert_close(result["tensor"], tensor * 2.0)
        np.testing.assert_array_almost_equal(result["array"], array * 3.0)
        assert result["none_val"] is None


@pytest.mark.unit
class TestToTorch:
    def test_converts_ndarray_to_float_tensor_on_device(
        self,
        ndarray_factory: Callable[..., np.ndarray],
    ):
        array = ndarray_factory(shape=(3,))
        result = to_torch({"x": array}, device=torch.device("cpu"))
        assert isinstance(result["x"], torch.Tensor)
        assert result["x"].dtype == torch.float32
        assert result["x"].device.type == "cpu"
        np.testing.assert_array_almost_equal(
            result["x"].numpy(), array.astype(np.float32)
        )


@pytest.mark.unit
class TestToOneHot:
    def test_converts_nested_dict_labels_to_one_hot(self):
        labels = {"a": torch.tensor([0, 2]), "b": torch.tensor([1])}
        result = to_one_hot(labels, num_class=3)
        expected_a = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        expected_b = torch.tensor([[0.0, 1.0, 0.0]])
        torch.testing.assert_close(result["a"], expected_a)
        torch.testing.assert_close(result["b"], expected_b)
