# Testing Guidelines

**Read this file first before writing or modifying any test.**

## Unit vs integration — the contract

**Unit tests** verify a single function/method in complete isolation. Every external dependency (other modules, IO, databases, rendering, network, real submodules) is replaced with a mock via `patch()` at the import path where the dependency is used. The test controls all inputs and verifies two things:

1. The function's return value or state change is correct.
2. The function called its dependencies with the correct arguments, shapes, and types.

When any dependency changes its contract, the unit test breaks immediately and points to the exact mismatch. Use `pytest.mark.parametrize` to sweep edge cases (boundary values, optional fields present/absent, error paths via `pytest.raises` alongside `does_not_raise()`) without duplicating logic. Use fixture factories (`Callable[..., T]`) so each test can customise its inputs. No real IO, no real computation from dependencies — only the unit's own logic executes.

**Integration tests** run multiple real components together to verify they compose correctly: real rendering, real file writes, real data pipelines, real model forward passes. They're slower, flakier, harder to diagnose, but catch composition bugs unit tests miss (e.g. two modules both respecting a contract individually but misinterpreting it in combination). Mark them `@pytest.mark.integration` and run them less frequently.

The failure mode of skipping real unit testing and writing "unit tests" that call real dependencies: tests pass silently when internal contracts break, edge cases go uncovered because parametrising over real components is slow, and when something does fail you get a five-module traceback instead of a pinpointed mock assertion.

**DL modules need both.** For layers, encoders, decoders, and fusion modules, write:
1. Strict **unit tests** that mock every submodule (including `nn.Linear`/`nn.Conv2d` child modules) and assert on the mock call args — this is where you catch "my layer is passing the wrong shape to its downstream collaborator". These are still the first line of defense and must exist.
2. Separate **behavioural/integration tests** (marked `@pytest.mark.integration` when they need real pretrained weights) that instantiate the real submodule stack *with small dimensions* and verify value-level properties: causal masking, cache equivalence, gradient flow, conditioning sensitivity. These catch the composition bugs that pure mock-based unit tests can't see.

Don't conflate the two. A test that calls a real `nn.Linear` forward pass is a behavioural test, not a unit test, even if it's fast and marked `@pytest.mark.unit`. Both kinds of test are required for DL building blocks.

## Principles

1. **Tests expose bugs, not confirm happy paths.** If a test reveals a bug in the source code, report it and fix the source — never adjust the test to hide it.

2. **Tests are self-contained.** Each test verifies one functionality. If a construct is not the object under test, mock it.

3. **Names explain what and why.** `test_eval_mode_disables_training_augmentations`, not `test_pipeline_1`.

## Structure

4. **One test module per source module.** Mirror `src/versatil/` structure under `tests/`.

5. **Group tests in classes by component, but only when there are ≥2 tests.** `TestAugmentationPipelineInitialization`, `TestApplyRGBAugmentations`. A class with a single test method should be a module-level `test_*` function — classes exist for grouping, not as a default wrapper.

6. **Shared fixtures go in conftest.** Package-level conftest for package-local fixtures, top-level `tests/conftest.py` for global.

## Fixtures and Parametrization

7. **Fixtures are factories.** Return callables so each test can customise. Module-level fixtures go at the top of the file, after imports and before the first test class.

8. **Parametrize edge cases.** 2–3 realistic cases per parameter; don't test impossible configurations. Combine valid + invalid into one parametrised test via `does_not_raise()`:
   ```python
   @pytest.mark.parametrize("value, expectation", [
       (1, does_not_raise()),
       (0, pytest.raises(ValueError, match="must be positive")),
   ])
   def test_field_validation(self, factory, value, expectation):
       with expectation:
           validate(factory(field=value))
   ```

8b. **Consolidate attribute-storage tests into one cross-product.** Stacked `@pytest.mark.parametrize` on a single `test_stores_configuration` that asserts every storable field. Keep validation tests (enum coverage, error paths) separate. Reference: `tests/models/encoding/encoders/rgb/test_spatial.py`.

9. **Set values explicitly — never rely on defaults.** `factory(field=True)` + assert against the explicit value, or parametrize. `factory()` + assert default silently passes when defaults change.

10. **Use the `rng` fixture for all random data.** Never call `np.random.*` or `torch.randn` directly. Wrap `rng` calls in semantic factory fixtures rather than using it inline.

10b. **Semantic factory fixtures per data object.** One factory per concept (`observation_dict_factory`, `action_chunk_factory`, `pad_mask_factory`), configurable via kwargs, `rng` used internally. Use kwargs for project-code calls with >1 argument (external libraries exempt).

## Code Style

11. **Follow root `AGENTS.md` coding rules.** In particular:
   - No inline imports.
   - No abbreviations in variable names.
   - No `**kwargs` / `*args`.
   - No section-separator comments.
   - Double quotes for strings.
   - Inline comments only when non-obvious.
   - Compare tensor devices via `.device.type` (never `== torch.device("cuda")`), except when comparing stored attrs directly.

11b. **`pytest.raises(match=...)` must reproduce the full error message.** No lazy partial matches like `match="requires"`. Reconstruct the exact message via f-string using the actual values that would appear. Use `re.escape()` for regex metacharacters.
   ```python
   with pytest.raises(
       ValueError,
       match=f"Input token length {expected_length} > max_seq_len {max_seq_len}",
   ):
   ```

11c. **Reuse conftest fixtures — never duplicate locally.** Before writing a fixture, check the conftest hierarchy. When calling conftest fixtures, use their *exact* parameter names (e.g. `feature_dimension=`, not `feature_dim=`) — mismatched kwargs are silently ignored and defaults kick in.

11d. **No docstrings on test functions or test classes.** Module-level docstring `"""Tests for versatil.{module_path} module."""` is required on every test file.

11e. **Assertions must verify what the test name claims.** A test named "injects zero padding" that only checks output keys is a lie. Assert the specific behaviour the name promises.

11f. **Unit tests assert on dependency call args; behavioural tests assert on computation correctness.** These are different, complementary tools:
   - **Unit tests** (`tests/data/`, most of `tests/configs/`, `tests/inference/`, etc.) assert `mock.assert_called_once_with(...)` to check that the unit called its collaborators with the right arguments/shapes/types. The mock pinpoints contract breaks when dependencies change.
   - **Behavioural tests** (DL building blocks in `tests/models/`, mathematical modules) assert value-level properties with controlled inputs: causal masking (modify a middle token, verify earlier predictions unchanged and later ones changed), conditioning sensitivity (different conditioning → different outputs), cache equivalence (cached vs uncached forward produce identical results), weight tying (mutate one tensor, verify tied tensor reflects the change), routing correctness (force routing weights, verify routed output matches the chosen expert). Shape-only assertions on the output are insufficient for these tests — they pass even if the computation is garbage — but they're fine *in addition* to behavioural assertions.
   
   Never use `is` / `isinstance` / `is not None` as the primary assertion — those test the Python mechanism, not the behaviour. Test the *consequence* of the behaviour being correct.

11g. **`wraps=` spy pattern for inspecting internal call arguments.** When you need to verify a method receives correct arguments without changing behaviour, use `patch.object(..., wraps=original)` and inspect `call_args`:
   ```python
   with patch.object(decoder, "_run_training_forward", wraps=decoder._run_training_forward) as spy:
       decoder(features=features, actions=actions)
   mask = spy.call_args.kwargs["attention_mask"]
   assert mask[0, 0, :action_len, action_len:].all()
   ```
   The method runs normally; you just observe what it received.

11h. **`patcher.start()` / `patcher.stop()` inside yield fixtures for patches that must outlive factory construction.** `with patch(...)` only holds the patch inside the `with` block. If a factory fixture builds an object and the test later calls a method that re-enters the patched boundary (e.g. `encoder.set_image_size()` → `timm.create_model()`), the patch has already expired and the real dependency gets called. Persistent patches use the stdlib `patcher = patch(...)`, `patcher.start()`, …, `patcher.stop()` pattern, wrapped in a yield fixture so teardown runs on test exit even if the test raises:

   ```python
   @pytest.fixture
   def mock_timm_backend():
       class _Backend:
           def __init__(self):
               self.cfg = MagicMock()
               self.cfg.fixed_input_size = False

           def configure(self, *, fixed_input_size: bool | None = None) -> None:
               if fixed_input_size is not None:
                   self.cfg.fixed_input_size = fixed_input_size

           def _side_effect(self, *args, **kwargs) -> MagicMock:
               return _make_mock_backbone(img_size=kwargs.get("img_size"))

       backend = _Backend()
       cfg_patcher = patch("versatil.X.timm.get_pretrained_cfg", return_value=backend.cfg)
       model_patcher = patch("versatil.X.timm.create_model", side_effect=backend._side_effect)
       cfg_patcher.start()
       backend.create_model_mock = model_patcher.start()
       yield backend
       cfg_patcher.stop()
       model_patcher.stop()
   ```

   **When required:** any time an object's later method calls re-trigger the same library boundary you patched at construction time (lazy builders, rebuildable backbones, anything that calls the patched function more than once). **When not:** one-shot construction tests — plain `with patch(...)` is shorter.

## What Not To Do

12. **Never write tests just to increase coverage.** Every test must guard against a real failure mode.

13. **Every used code path must be tested.** Logging, plotting, utilities — all have logic that can break. No exemptions for "not business logic".

14. **Verify source correctness while writing tests.** If a test reveals a bug or rule violation (`assert` that should be `raise`, missing kwargs, bare asserts), fix the *source*, not the test.

## Conftest hierarchy

Check the layered conftests before writing any fixture:

```
tests/conftest.py                 ← rng, device, batch_size, temporal_length, image_size,
                                     loss_output_factory, padding_mask_factory,
                                     action_tensor_factory, get_test_device
tests/models/conftest.py          ← input_tensor_factory, feature_dictionary_factory,
                                     action_dictionary_factory, batch_dictionary_factory,
                                     policy_factory, vision_encoder_factory
tests/models/layers/conftest.py   ← flat_tensor_factory, sequence_tensor_factory,
                                     nchw_tensor_factory, conv1d_tensor_factory,
                                     condition_factory, timestep_factory,
                                     attention_mask_factory
tests/models/decoding/conftest.py ← mock_action_space_factory, spatial_feature_factory,
                                     flat_feature_factory, action_head_factory
tests/models/encoding/conftest.py ← encoder_mock_factory, conditional_encoder_mock_factory,
                                     fusion_module_mock_factory
```

Subpackage conftests (`denoising/conftest.py`, `detr_transformer/conftest.py`, …) exist only for domain-specific fixtures shared across multiple test files in that subpackage and not used elsewhere.

## Reference implementations

- `tests/data/test_dataloader.py` — `does_not_raise()` parametrization, mock schema with `spec=` for `isinstance` dispatch.
- `tests/data/tokenization/test_observation_tokenizer.py` + `conftest.py` — semantic dict factories, parametrized configs.
- `tests/models/encoding/encoders/rgb/test_spatial.py` — consolidated `test_stores_configuration`, stacked parametrize, validation tests separated, mocked backbone via `patch.object`.
- `tests/models/encoding/encoders/rgb/test_flat.py` — yield-fixture persistent-patch pattern (`mock_timm_backend`) for rebuildable backbones.
- `tests/models/decoding/decoders/factory/test_dit_block_action_transformer.py` — behavioural assertions (conditioning sensitivity, timestep independence at init, gradient flow).
- `tests/models/layers/denoising/test_ode_solvers.py` — mathematical correctness (convergence order, exact integration for known ODEs).
- `tests/models/decoding/generative_language_models/vision_language/test_base.py` — fixture-level `params=` auto-parametrising over multiple implementations, session-scoped real-model factories.
