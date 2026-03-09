# Testing Guidelines

**Read this file first before writing or modifying any test.**

These guidelines define how tests in this codebase must be written. Reference implementations:
- `tests/data/augmentation/test_augmentation_pipeline.py` — mocking and factory fixtures
- `tests/data/test_task.py` — using conftest factories, module-level fixtures, explicit value testing
- `tests/data/conftest.py` — factory fixtures with type hints, RNG-based data generators
- `tests/data/test_action_processor.py` — testing all code paths including logging/plotting, denoising logic
- `tests/data/tokenization/test_observation_tokenizer.py` — semantic dict factories, parametrized configs, observation_dict_factory pattern
- `tests/data/tokenization/conftest.py` — shared semantic fixtures (action_chunk_factory, pad_mask_factory, binning_tokenizer_factory)
- `tests/data/test_dataloader.py` — `does_not_raise()` parametrization pattern for validation tests, mock schema factories with `spec=` for `isinstance` dispatch

## Principles

1. **Tests expose bugs, not confirm happy paths.** The goal is to verify that the codebase works correctly and to safeguard against potential issues. Never hand-craft a test to "make it pass" — that defeats the purpose. If a test reveals a bug in the source code, report it; do not adjust the test to hide it.

2. **Tests are self-contained.** Each test verifies one functionality (or a small cluster of closely related behaviors). If a construct is not the object under test, mock or patch it. Do not let unrelated interfaces leak into the test.

3. **Test names explain what is being tested and why.** The name should make the test's purpose obvious without reading the body. Use descriptive names like `test_eval_mode_disables_training_augmentations`, not `test_pipeline_1`.

## Structure

4. **One testing module per source module.** Each module in `src/versatil/` should have a corresponding test module. Do not bundle tests for multiple source modules into one test file.

5. **Organize tests in classes by the class or component they test.** Group related tests under a class like `TestAugmentationPipelineInitialization` or `TestApplyRGBAugmentations`.

6. **Shared fixtures go in conftest.** Data or objects shared across multiple test modules in a package belong in a package-level `conftest.py`. Objects shared across packages belong in the top-level `tests/conftest.py`.

## Fixtures and Parametrization

7. **Fixtures should be customizable via factories.** Return factory functions so each test can tweak the fixture to its needs. See how `mock_color_augmentation` and `mock_resize_transform` are defined in the reference file — they return factories that each test can configure. Module-level fixtures must be defined at the top of the file, after imports and before the first test class — never at the bottom.

8. **Use `pytest.mark.parametrize` for multiple conditions and edge cases.** When a test applies to several inputs or configurations, parametrize it instead of duplicating code. Only parametrize meaningful conditions — do not test impossible or meaningless edge cases. Aim for 2–3 parametrized cases per test to cover different realistic configurations (e.g., different shapes, different keys, different config values). When testing validation functions, use `does_not_raise()` from `contextlib` to combine valid and invalid cases into a single parametrized test instead of splitting them into separate "passes" and "raises" tests:
   ```python
   from contextlib import nullcontext as does_not_raise

   @pytest.mark.parametrize("value, expectation", [
       (1, does_not_raise()),
       (0, pytest.raises(ValueError, match="must be positive")),
       (-1, pytest.raises(ValueError, match="must be positive")),
   ])
   def test_field_validation(self, factory, value, expectation):
       config = factory(field=value)
       with expectation:
           validate(config)
   ```
   Reference: `tests/data/test_dataloader.py`.

9. **Always set values explicitly — never test defaults implicitly.** When verifying that a parameter is stored correctly, pass the value explicitly rather than relying on the default. Testing `factory()` and asserting `.field is True` is fragile because it assumes knowledge of the default value and will silently pass even if the default changes. Instead, use `factory(field=True)` and assert against the explicit value, or better, parametrize useful values to check several cases if it makes sense. Take as reference `tests/data/test_task.py`.

10. **Use the `rng` fixture for all random data generation.** Never call `np.random.rand`, `np.random.randn`, `np.random.randint`, or `torch.randn` directly. Instead, use the `rng` fixture from `tests/conftest.py` (a `np.random.Generator` seeded per test). This ensures reproducibility and test isolation without global seed mutation. For torch tensors, convert from numpy: `torch.from_numpy(rng.standard_normal((shape,)).astype(np.float32))`. Wrap `rng` calls in semantic factory fixtures (see below) rather than calling `rng` directly in test bodies.

10b. **Create semantic factory fixtures for data objects.** Do not construct raw arrays/tensors inline in tests. Instead, define a factory fixture per semantic object (e.g., `observation_dict_factory`, `action_chunk_factory`, `pad_mask_factory`, `training_data_factory`). Each factory should return the complete semantic object (e.g., a full observations dict with optional language + proprio keys), be configurable via parameters (batch_size, observation_dim, as_torch, etc.), and use the `rng` fixture internally. This keeps tests readable and avoids duplicating data construction logic. Use kwargs when calling project code with >1 argument; this rule does NOT apply to external library functions (numpy, torch, pytest).

## Code Style

11. **Follow the project coding guidelines from the root `CLAUDE.md`.** In particular:
   - No inline imports — all imports at the top of the module.
   - No abbreviations in variable names — use full English words.
   - No `**kwargs` or `*args` — always explicit named parameters.
   - No section separator comments (e.g., `# ------`).
   - No qualitative words in docstrings or comments ("robust", "powerful", etc.).
   - Inline comments only when the code is not self-explanatory or when verifying something non-obvious from the test name.
   - Use double quotes for strings.
   - Use Google-style docstrings when docstrings are needed. Usually docstrings are not needed for tests, as tests should be as minimal and straightforward as possible.
   - When comparing tensor devices, use `.device.type` (e.g., `assert tensor.device.type == device.type`) because `torch.device("cuda") != torch.device("cuda", 0)`. Only use direct `==` for stored attribute comparisons (e.g., `tokenizer.device == device`), never for tensor device checks.

## What Not To Do

12. **Never use legacy tests as reference.** Tests not listed as references above are outdated and do not follow these guidelines. Always refer to the reference implementations listed at the top and to this document for the rules.

13.**Never write tests just to increase coverage.** Every test must verify a meaningful condition. If a test does not guard against a real failure mode, it should not exist.

14.**Every code path must be tested.** If a function exists in the codebase and is used, it must be tested — regardless of whether it contains "business logic". Logging functions, plotting functions, and utility functions all have logic that can break. Untested code paths are unacceptable.

15.**Verify source code correctness while writing tests.** When writing tests for a module, audit the source code for violations of project rules (e.g., `assert` statements that should be `raise`, missing kwargs, bare assertions). If a test reveals a bug or rule violation in the source code, fix the source code — do not adjust the test to work around it.