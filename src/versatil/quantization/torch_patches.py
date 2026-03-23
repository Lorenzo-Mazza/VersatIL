"""Monkey-patches for torch.fx source matching bugs in torch 2.10 + torchao 0.16.

Note:
    Addresses pytorch/ao#3914: X86InductorQuantizer silently quantizes 0 ops
    because get_source_partitions compares source_fn_name strings against class
    objects in wanted_sources, which never matches.
Link:
 https://github.com/pytorch/ao/issues/3914
"""

import importlib.metadata
import sys
from collections.abc import Callable
from typing import Any

import torch
from packaging.version import Version
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx.passes.utils.source_matcher_utils import SourcePartition

_VERSATIL_PATCHED_SENTINEL = "_versatil_patched"

_TORCH_MAX_VERSION = Version("2.10.99")
_TORCHAO_MAX_VERSION = Version("0.16.99")


def _get_torchao_version() -> Version | None:
    """Return the installed torchao version, or None if not installed."""
    torchao_distributions = importlib.metadata.packages_distributions().get(
        "torchao", None
    )
    if torchao_distributions is None:
        return None
    package_name = torchao_distributions[0]
    return Version(importlib.metadata.version(package_name))


def _is_patch_needed() -> bool:
    """Check whether the patch should be applied based on package versions.

    Returns:
        True if torch <= 2.10.x and torchao <= 0.16.x are installed,
        meaning the bug is present and needs patching.
    """
    torch_version = Version(torch.__version__.split("+")[0])
    if torch_version > _TORCH_MAX_VERSION:
        return False

    torchao_version = _get_torchao_version()
    if torchao_version is None:
        return False

    return torchao_version <= _TORCHAO_MAX_VERSION


def _make_patched_get_source_partitions(
    original_function: Callable[..., dict[Any, list[SourcePartition]]],
) -> Callable[..., dict[Any, list[SourcePartition]]]:
    """Create a patched version of get_source_partitions.

    The original function's fallback path extracts a source_fn_name string
    (e.g. "linear") from node.meta["torch_fn"] and checks membership in
    wanted_sources. But wanted_sources contains class objects (e.g.
    torch.nn.Linear), so the string-vs-class comparison never matches.

    The patched version first tries the original function. If it returns
    empty partitions, it falls back to matching source_fn_name against
    each class's __name__ (case-insensitive).

    Args:
        original_function: The unpatched get_source_partitions function.

    Returns:
        Patched function with the same signature.
    """

    def patched_get_source_partitions(
        graph: Graph,
        wanted_sources: list[Any],
        filter_fn: Callable | None = None,
    ) -> dict[Any, list[SourcePartition]]:
        result = original_function(graph, wanted_sources, filter_fn=filter_fn)

        if result:
            return result
        # Fallback: match torch_fn string names against class __name__
        name_to_source: dict[str, Any] = {}
        for source in wanted_sources:
            if isinstance(source, type):
                name_to_source[source.__name__.lower()] = source

        if not name_to_source:
            return result

        modules: dict[Any, dict[str, list[Node]]] = {}

        for node in graph.nodes:
            torch_fn = node.meta.get("torch_fn", None)
            if torch_fn is None:
                continue

            if isinstance(torch_fn, tuple) and len(torch_fn) >= 1:
                source_fn_raw = torch_fn[0]
            elif isinstance(torch_fn, str):
                source_fn_raw = torch_fn
            else:
                continue
            # FX tracing appends deduplication suffixes like "_1", "_2"
            # to repeated ops. Strip trailing digits + underscore to
            # recover the base operator name (e.g. "linear_1" -> "linear").
            base_name = source_fn_raw.split(".")[0].lower()
            source_fn_name = base_name.rstrip("0123456789").rstrip("_")
            matched_source = name_to_source.get(source_fn_name)
            if matched_source is None:
                continue

            diff_modules = modules.setdefault(matched_source, {})
            node_name = node.name if hasattr(node, "name") else str(node)
            partition = diff_modules.setdefault(node_name, [])
            partition.append(node)

        fallback_result: dict[Any, list[SourcePartition]] = {}
        for source_type, name_to_nodes in modules.items():
            partitions = []
            for nodes in name_to_nodes.values():
                input_nodes = set()
                output_nodes = set()
                params = set()
                for node in nodes:
                    for argument in node.args:
                        if isinstance(argument, Node) and argument not in nodes:
                            input_nodes.add(argument)
                    if node.op == "get_attr":
                        params.add(node)
                    for user in node.users:
                        if user not in nodes:
                            output_nodes.add(node)

                if filter_fn is not None and not all(filter_fn(node) for node in nodes):
                    continue

                partitions.append(
                    SourcePartition(
                        nodes=nodes,
                        source=source_type,
                        input_nodes=list(input_nodes),
                        output_nodes=list(output_nodes),
                        params=list(params),
                    )
                )
            if partitions:
                fallback_result[source_type] = partitions

        return fallback_result

    return patched_get_source_partitions


def patch_get_source_partitions() -> None:
    """Apply the monkey-patch to torch.fx get_source_partitions.

    The patch is idempotent: calling it multiple times has no additional
    effect. The patch is skipped if torch or torchao versions indicate
    the upstream bug has been fixed.
    """
    source_matcher_module = torch.fx.passes.utils.source_matcher_utils
    if getattr(source_matcher_module, _VERSATIL_PATCHED_SENTINEL, False):
        return

    if not _is_patch_needed():
        return

    original_function = source_matcher_module.get_source_partitions
    patched_function = _make_patched_get_source_partitions(original_function)

    source_matcher_module.get_source_partitions = patched_function
    # Patch all modules that imported get_source_partitions by name,
    # since `from module import func` creates a local binding that
    # won't see the canonical patch.
    for module in list(sys.modules.values()):
        if module is None or module is source_matcher_module:
            continue
        if getattr(module, "get_source_partitions", None) is original_function:
            module.get_source_partitions = patched_function

    setattr(source_matcher_module, _VERSATIL_PATCHED_SENTINEL, True)
