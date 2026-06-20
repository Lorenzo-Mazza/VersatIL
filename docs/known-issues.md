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

---

## torchao 0.17: QAT INT4 group size is ignored

**Error:** `RuntimeError: shape '[..., -1, 128]' is invalid` during a QAT training forward prepared from `Int4WeightOnlyConfig(group_size=32)` or `group_size=64`.

**Root cause:** torchao 0.17 infers `Int4WeightFakeQuantizeConfig(group_size=128)` for `Int4WeightOnlyConfig(version=2)`, ignoring the user-provided `Int4WeightOnlyConfig.group_size`.

**Upstream:**

- Bug report: [pytorch/ao#3572](https://github.com/pytorch/ao/issues/3572)
- Fix: [pytorch/ao#4518](https://github.com/pytorch/ao/pull/4518)

**Workaround:** `src/versatil/quantization/torch_patches.py` (`patch_qat_int4_group_size`) patches the installed torchao QAT fake-quant config on disk so it uses `base_config.group_size`. Called automatically before QAT imports. Idempotent and skips wheels that already include the upstream fix.

**Remove when:** the pinned torchao wheel includes the fix from `pytorch/ao#4518`.
