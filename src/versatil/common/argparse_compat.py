"""Python 3.14 argparse compatibility shim for Hydra 1.3.2.

Python 3.14 added ``_ActionsContainer._check_help`` which eagerly validates
``help=`` arguments at ``add_argument`` time by calling
``formatter._expand_help(action)``. That path performs ``'%' in help_string``,
which assumes ``help`` is a ``str``. Hydra 1.3.2 passes a
``LazyCompletionHelp`` instance (defined inside
``hydra._internal.utils.get_args_parser``) that only implements ``__repr__``,
so the eager check raises ``ValueError('badly formed help string')`` and
prevents any Hydra endpoint from starting on Python 3.14.

This module restores the pre-3.14 deferred behavior only for non-string
``help`` values: string help strings still go through the stock check, so
genuinely malformed strings are still caught.

Import this module once before ``import hydra``.

Upstream status
---------------
- Hydra issue: https://github.com/facebookresearch/hydra/issues/3121
  (open, same traceback, unresolved in 1.3.2)
- Hydra fix: https://github.com/facebookresearch/hydra/pull/3090
  (merged 2025-10-28 into ``main``, targets ``1.4.0.dev``, not on PyPI).
  The upstream fix uses the same approach — temporarily disabling
  ``argparse.ArgumentParser._check_help``.
- Release tracker: https://github.com/facebookresearch/hydra/issues/3125
  (asking for a 1.4 release; no release yet)
- CPython: the eager ``_check_help`` was added in CPython PR #141940 and is
  not flagged as a breaking change in the Python 3.14 "What's New" notes.

This shim can be removed once ``hydra-core >= 1.4`` is released on PyPI
and the project bumps the pin in ``pyproject.toml``.
"""

import argparse


def _install() -> None:
    original_check_help = argparse._ActionsContainer._check_help

    def _check_help_string_only(
        self: argparse._ActionsContainer, action: argparse.Action
    ) -> None:
        if action.help is None or isinstance(action.help, str):
            original_check_help(self, action)

    argparse._ActionsContainer._check_help = _check_help_string_only  # type: ignore[method-assign]


_install()
