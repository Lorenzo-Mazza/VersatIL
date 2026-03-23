"""Tests for versatil.quantization.torch_patches module."""

import sys
import types
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
import torch
from packaging.version import Version
from torch.fx.graph import Graph
from torch.fx.passes.utils.source_matcher_utils import SourcePartition

from versatil.quantization.torch_patches import (
    _VERSATIL_PATCHED_SENTINEL,
    _is_patch_needed,
    _make_patched_get_source_partitions,
    patch_get_source_partitions,
)


@pytest.fixture
def mock_graph_with_torch_fn_metadata() -> Callable[..., Graph]:
    """Factory for FX graphs with torch_fn metadata simulating the bug."""

    def factory(
        torch_fn_value: Any = ("linear.default", "aten"),
    ) -> Graph:
        graph = Graph()
        input_node = graph.placeholder("x")
        input_node.meta["torch_fn"] = None

        linear_node = graph.call_function(
            torch.ops.aten.linear.default, args=(input_node,)
        )
        linear_node.meta["torch_fn"] = torch_fn_value

        output_node = graph.output(linear_node)
        output_node.meta["torch_fn"] = None

        return graph

    return factory


@pytest.fixture
def original_function_returning_empty() -> Callable[
    ..., dict[Any, list[SourcePartition]]
]:
    """Simulates the buggy get_source_partitions that always returns empty."""

    def empty_original(
        graph: Graph,
        wanted_sources: list[Any],
        filter_fn: Callable | None = None,
    ) -> dict[Any, list[SourcePartition]]:
        return {}

    return empty_original


@pytest.fixture
def original_function_returning_results() -> Callable[
    ..., dict[Any, list[SourcePartition]]
]:
    """Simulates a working get_source_partitions that returns non-empty results."""

    def nonempty_original(
        graph: Graph,
        wanted_sources: list[Any],
        filter_fn: Callable | None = None,
    ) -> dict[Any, list[SourcePartition]]:
        sentinel_partition = SourcePartition(nodes=[], source=wanted_sources[0])
        return {wanted_sources[0]: [sentinel_partition]}

    return nonempty_original


@pytest.fixture
def clean_source_matcher():
    """Yield the source_matcher module and restore it after test."""
    module = torch.fx.passes.utils.source_matcher_utils
    original = module.get_source_partitions
    yield module
    module.get_source_partitions = original
    if hasattr(module, _VERSATIL_PATCHED_SENTINEL):
        delattr(module, _VERSATIL_PATCHED_SENTINEL)


@pytest.mark.unit
class TestIsPatchNeeded:
    @pytest.mark.parametrize(
        "torch_version, torchao_return, expected",
        [
            ("2.11.0", Version("0.16.0"), False),
            ("2.10.0", None, False),
            ("2.10.0", Version("0.17.0"), False),
            ("2.10.0", Version("0.16.0"), True),
            ("2.10.0+cu128", Version("0.16.0"), True),
        ],
        ids=[
            "torch_too_new",
            "torchao_missing",
            "torchao_too_new",
            "both_affected",
            "cuda_suffix",
        ],
    )
    def test_version_check(self, torch_version, torchao_return, expected):
        with (
            patch(
                "versatil.quantization.torch_patches.torch.__version__",
                torch_version,
            ),
            patch(
                "versatil.quantization.torch_patches._get_torchao_version",
                return_value=torchao_return,
            ),
        ):
            assert _is_patch_needed() is expected


@pytest.mark.unit
class TestMakePatchedGetSourcePartitions:
    def test_delegates_to_original_when_original_returns_results(
        self,
        original_function_returning_results,
        mock_graph_with_torch_fn_metadata,
    ):
        patched = _make_patched_get_source_partitions(
            original_function_returning_results
        )
        graph = mock_graph_with_torch_fn_metadata()

        result = patched(graph, [torch.nn.Linear])

        assert torch.nn.Linear in result
        assert len(result[torch.nn.Linear]) == 1

    def test_fallback_matches_string_name_against_class_name(
        self,
        original_function_returning_empty,
        mock_graph_with_torch_fn_metadata,
    ):
        patched = _make_patched_get_source_partitions(original_function_returning_empty)
        graph = mock_graph_with_torch_fn_metadata()

        result = patched(graph, [torch.nn.Linear])

        assert torch.nn.Linear in result
        partition = result[torch.nn.Linear][0]
        assert partition.source is torch.nn.Linear
        assert len(partition.nodes) == 1

    @pytest.mark.parametrize(
        "torch_fn_value",
        [
            ("linear.default", "aten"),
            "linear.default",
            ("Linear.default", "aten"),
        ],
        ids=["tuple", "string", "case_insensitive"],
    )
    def test_fallback_handles_torch_fn_formats(
        self,
        original_function_returning_empty,
        mock_graph_with_torch_fn_metadata,
        torch_fn_value,
    ):
        patched = _make_patched_get_source_partitions(original_function_returning_empty)
        graph = mock_graph_with_torch_fn_metadata(
            torch_fn_value=torch_fn_value,
        )

        result = patched(graph, [torch.nn.Linear])

        assert torch.nn.Linear in result

    def test_fallback_strips_deduplication_suffix(
        self,
        original_function_returning_empty,
        mock_graph_with_torch_fn_metadata,
    ):
        patched = _make_patched_get_source_partitions(original_function_returning_empty)
        graph = mock_graph_with_torch_fn_metadata(
            torch_fn_value=("linear_1.default", "aten"),
        )

        result = patched(graph, [torch.nn.Linear])

        assert torch.nn.Linear in result

    def test_fallback_returns_empty_when_no_class_matches(
        self,
        original_function_returning_empty,
        mock_graph_with_torch_fn_metadata,
    ):
        patched = _make_patched_get_source_partitions(original_function_returning_empty)
        graph = mock_graph_with_torch_fn_metadata(
            torch_fn_value=("conv2d.default", "aten"),
        )

        result = patched(graph, [torch.nn.Linear])

        assert result == {}

    def test_fallback_skips_nodes_without_torch_fn(
        self,
        original_function_returning_empty,
    ):
        graph = Graph()
        input_node = graph.placeholder("x")
        graph.output(input_node)

        patched = _make_patched_get_source_partitions(original_function_returning_empty)

        assert patched(graph, [torch.nn.Linear]) == {}

    def test_fallback_returns_empty_when_wanted_sources_has_no_classes(
        self,
        original_function_returning_empty,
        mock_graph_with_torch_fn_metadata,
    ):
        patched = _make_patched_get_source_partitions(original_function_returning_empty)
        graph = mock_graph_with_torch_fn_metadata()

        assert patched(graph, [torch.nn.functional.linear]) == {}

    def test_fallback_respects_filter_fn(
        self,
        original_function_returning_empty,
        mock_graph_with_torch_fn_metadata,
    ):
        patched = _make_patched_get_source_partitions(original_function_returning_empty)
        graph = mock_graph_with_torch_fn_metadata()

        result = patched(graph, [torch.nn.Linear], filter_fn=lambda node: False)

        assert result == {}

    def test_partition_tracks_input_and_output_nodes(
        self,
        original_function_returning_empty,
        mock_graph_with_torch_fn_metadata,
    ):
        patched = _make_patched_get_source_partitions(original_function_returning_empty)
        graph = mock_graph_with_torch_fn_metadata()
        nodes = list(graph.nodes)
        input_node = nodes[0]
        linear_node = nodes[1]

        result = patched(graph, [torch.nn.Linear])

        partition = result[torch.nn.Linear][0]
        assert input_node in partition.input_nodes
        assert linear_node in partition.output_nodes

    def test_multiple_matching_nodes_produce_separate_partitions(
        self,
        original_function_returning_empty,
    ):
        graph = Graph()
        x = graph.placeholder("x")
        x.meta["torch_fn"] = None
        linear1 = graph.call_function(torch.ops.aten.linear.default, args=(x,))
        linear1.meta["torch_fn"] = ("linear.default", "aten")
        linear2 = graph.call_function(torch.ops.aten.linear.default, args=(linear1,))
        linear2.meta["torch_fn"] = ("linear_1.default", "aten")
        output = graph.output(linear2)
        output.meta["torch_fn"] = None

        patched = _make_patched_get_source_partitions(original_function_returning_empty)

        result = patched(graph, [torch.nn.Linear])

        assert torch.nn.Linear in result
        assert len(result[torch.nn.Linear]) == 2


@pytest.mark.unit
class TestPatchGetSourcePartitions:
    def test_patch_is_idempotent(self):
        source_matcher_module = torch.fx.passes.utils.source_matcher_utils
        with patch.object(
            source_matcher_module,
            _VERSATIL_PATCHED_SENTINEL,
            True,
            create=True,
        ):
            original_before = source_matcher_module.get_source_partitions
            patch_get_source_partitions()
            assert source_matcher_module.get_source_partitions is original_before

    def test_skips_patch_when_versions_not_affected(self):
        source_matcher_module = torch.fx.passes.utils.source_matcher_utils
        with (
            patch(
                "versatil.quantization.torch_patches._is_patch_needed",
                return_value=False,
            ),
            patch.object(
                source_matcher_module,
                _VERSATIL_PATCHED_SENTINEL,
                False,
                create=True,
            ),
        ):
            original_before = source_matcher_module.get_source_partitions
            patch_get_source_partitions()
            assert source_matcher_module.get_source_partitions is original_before

    def test_applies_patch_when_versions_affected(
        self,
        clean_source_matcher,
    ):
        original_function = clean_source_matcher.get_source_partitions
        with (
            patch(
                "versatil.quantization.torch_patches._is_patch_needed",
                return_value=True,
            ),
            patch.object(
                clean_source_matcher,
                _VERSATIL_PATCHED_SENTINEL,
                False,
                create=True,
            ),
        ):
            patch_get_source_partitions()
            patched_function = clean_source_matcher.get_source_partitions

            assert patched_function is not original_function
            assert getattr(clean_source_matcher, _VERSATIL_PATCHED_SENTINEL)

    def test_patches_cross_module_bindings(
        self,
        clean_source_matcher,
    ):
        original_function = clean_source_matcher.get_source_partitions
        fake_module = types.ModuleType("_test_versatil_fake_consumer")
        fake_module.get_source_partitions = original_function
        sys.modules[fake_module.__name__] = fake_module
        try:
            with (
                patch(
                    "versatil.quantization.torch_patches._is_patch_needed",
                    return_value=True,
                ),
                patch.object(
                    clean_source_matcher,
                    _VERSATIL_PATCHED_SENTINEL,
                    False,
                    create=True,
                ),
            ):
                patch_get_source_partitions()
                assert fake_module.get_source_partitions is not original_function
                assert (
                    fake_module.get_source_partitions
                    is clean_source_matcher.get_source_partitions
                )
        finally:
            del sys.modules[fake_module.__name__]
