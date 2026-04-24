# Known Issues

Compatibility patches for upstream bugs in Hydra and torchao. Each patch is version-gated and can be removed once the upstream fix ships on PyPI.

---

## Hydra 1.3.2 + Python 3.14

**Error:** `ValueError: badly formed help string` at startup of any `@hydra.main` endpoint.

**Root cause:** Python 3.14 added eager help-string validation in `argparse._ActionsContainer._check_help`. Hydra 1.3.2's `LazyCompletionHelp` only defines `__repr__`, not `__contains__`, so the `'%' in help_string` check raises `TypeError`.

**Upstream:**

- Bug report: [facebookresearch/hydra#3121](https://github.com/facebookresearch/hydra/issues/3121)
- Fix (merged to `main`, targets 1.4.0.dev): [facebookresearch/hydra#3090](https://github.com/facebookresearch/hydra/pull/3090)
- Release request: [facebookresearch/hydra#3125](https://github.com/facebookresearch/hydra/issues/3125)

**Workaround:** `src/versatil/common/argparse_compat.py` monkey-patches `_check_help` to skip the eager validation for non-string help values. Imported in `src/versatil/endpoints/train.py` and `src/versatil/endpoints/post_training_compress.py` before `import hydra`.

**Remove when:** `hydra-core >= 1.4` ships on PyPI.

---

## torchao 0.16 + Python 3.14: PT2E import crash

**Error:** `TypeError: cannot set '__module__' attribute of immutable type 'types.UnionType'` when importing `torchao.quantization.pt2e`.

**Root cause:** Python 3.14 merged `typing.Union` with `types.UnionType`, making Union objects immutable. torchao 0.16 assigns `__module__` to Union aliases at module import time.

**Upstream:**

- Bug report: [pytorch/ao#3619](https://github.com/pytorch/ao/issues/3619)
- Fix: [pytorch/ao#3657](https://github.com/pytorch/ao/pull/3657)

**Workaround:** `src/versatil/quantization/torch_patches.py` (`patch_pt2e_python314`) patches the installed torchao `.py` files on disk, replacing the crashing `__module__` assignments with `pass`. Called automatically at import time, before any PT2E code runs. Idempotent — skips files already patched.

**Remove when:** the repository dependency is bumped past `torchao==0.16.0` and the import succeeds without the patch.

---

## torchao 0.16: X86InductorQuantizer silently quantizes 0 ops

**Error:** No error — the quantizer runs successfully but quantizes nothing. The compressed model has identical size and performance to the original.

**Root cause:** `torch.fx.passes.utils.source_matcher_utils.get_source_partitions` compares `source_fn_name` strings (e.g. `"linear"`) against class objects (e.g. `torch.nn.Linear`) in `wanted_sources`. The string-vs-class comparison never matches, so zero partitions are found and zero ops are quantized.

**Upstream:**

- Bug report: [pytorch/ao#3914](https://github.com/pytorch/ao/issues/3914)

**Workaround:** `src/versatil/quantization/torch_patches.py` (`patch_get_source_partitions`) monkey-patches `get_source_partitions` with a fallback that matches `source_fn_name` against each class's `__name__` (case-insensitive). Applied automatically, idempotent, version-gated to torch <= 2.10 + torchao <= 0.16.

**Remove when:** the repository dependency is bumped past `torchao==0.16.0` or `torch==2.10.0` and PT2E quantization still covers the expected operators without the patch.
