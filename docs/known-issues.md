# Known Issues

Compatibility patches for upstream bugs in torchao.

---

## torchao 0.17 + Python 3.14: PT2E import crash

**Error:** `TypeError: cannot set '__module__' attribute of immutable type 'types.UnionType'` when importing `torchao.quantization.pt2e`.

**Root cause:** Python 3.14 makes `typing.Union` objects immutable. torchao 0.17 still assigns `__module__` to Union aliases at module import time.

**Upstream:**

- Bug report: [pytorch/ao#3619](https://github.com/pytorch/ao/issues/3619)
- Fix: [pytorch/ao#3657](https://github.com/pytorch/ao/pull/3657)

**Workaround:** `src/versatil/quantization/torch_patches.py` (`patch_pt2e_python314`) patches the installed torchao `.py` files on disk, replacing the crashing `__module__` assignments with `pass`. Called automatically at import time, before any PT2E code runs. Idempotent and skips files already patched.

**Remove when:** a clean torchao wheel imports `torchao.quantization.pt2e` on Python 3.14 without the patch.
