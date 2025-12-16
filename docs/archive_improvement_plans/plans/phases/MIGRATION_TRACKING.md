# CLI Migration Tracking

> **Started:** December 10, 2024  
> **Target Completion:** January 21, 2025 (6 weeks)  
> **Current Phase:** 2 (Rendering Consolidation) ✅

## Phase Status

| Phase | Name | Status | Started | Completed | Notes |
|-------|------|--------|---------|-----------|-------|
| 0 | Preparation | ✅ Complete | Dec 10, 2024 | Dec 10, 2024 | Baseline captured |
| 1 | Foundation Layer | ✅ Complete | Dec 10, 2024 | Dec 10, 2024 | `HandlerContext`, `bootstrap_cli()` |
| 2 | Rendering Consolidation | ✅ Complete | Dec 10, 2024 | Dec 10, 2024 | Single `UnifiedRenderer` |
| 3 | Handler Migration | ⚪ Not Started | | | All handlers use new context |
| 4 | Registry Unification | ⚪ Not Started | | | Single `OperationRegistry` |
| 5 | Command Decorator | ⚪ Not Started | | | `@cli_command` + migrate commands |
| 6 | Legacy Cleanup | ⚪ Not Started | | | Delete superseded files |

## Key Metrics

| Metric | Baseline | Current | Target |
|--------|----------|---------|--------|
| Test count | 362 | 444 | >= 362 |
| Test pass rate | 97.2% (352/362) | 97.7% (434/444) | 100% |
| CLI coverage | ~48% | ~48% | >= 48% |
| Handler files migrated | 0/13 | 0/13 | 13/13 |
| Command files migrated | 0/14 | 0/14 | 14/14 |
| Legacy files deleted | 0/15 | 1/15 | 15/15 |

## Baseline Artifacts

| Artifact | Location | Status |
|----------|----------|--------|
| Test baseline output | `artifacts/test_baseline_output.txt` | ✅ Created |
| Test baseline report | `artifacts/test_baseline_report.md` | ✅ Created |
| Coverage JSON | `artifacts/coverage_baseline.json` | ✅ Created |
| Coverage HTML | `artifacts/htmlcov_baseline/` | ✅ Created |
| Handler inventory | `artifacts/handler_inventory.md` | ✅ Created |
| Command inventory | `artifacts/command_inventory.md` | ✅ Created |
| Known test issues | `artifacts/known_test_issues.md` | ✅ Created |

## Daily Log

### December 10, 2024

**Phase 0 Completed:**
- Created artifacts directory structure
- Ran full CLI test suite (352 passed, 9 failed, 1 error)
- Generated CLI-specific coverage report
- Created comprehensive handler inventory (13 files, 40+ handlers)
- Created comprehensive command inventory (14 command files)
- Documented 10 known pre-existing test issues
- Created migration tracking document
- Created feature flag module for gradual rollout

**Key Findings:**
- All handlers use `EnhancedHandlerContext` from `handlers/protocol.py`
- 11 of 13 handler files have local param helper functions to consolidate
- Most commands follow consistent `__call__` pattern with `command_context`
- Test failures are pre-existing and don't block migration

**Phase 1 Completed:**
- Created `src/codeintel/cli/handlers/context.py` with unified `HandlerContext`
- Created `src/codeintel/cli/handlers/_lazy_resources.py` for lazy imports
- Created `src/codeintel/cli/execution/bootstrap.py` with `bootstrap_cli()`
- Created `HandlerContextOptions` dataclass for clean API
- Added all typed parameter accessors (`param_str`, `param_int`, `param_bool`, etc.)
- Added require methods (`require_str`, `require_int`, `require_path`)
- Added lazy resource properties (`runtime`, `gateway`, `graph_runtime`)
- Added context manager protocol with automatic cleanup
- Added `from_enhanced_context()` adapter for gradual migration
- Created comprehensive unit tests (72 new tests)
- All quality checks pass (ruff, pyright, pyrefly)
- No regressions in existing tests

**Phase 2 Completed:**
- Added `get_renderer()` factory function to `rendering/service.py`
- Added `render_cli_result()` convenience function to `rendering/service.py`
- Verified `rendering/specs.py` already existed with table specs
- Updated `rendering/__init__.py` to import from `service.py`
- Updated `execution/executor.py` to use `UnifiedRenderer` type
- Verified `execution/adapter.py` imports via `__init__.py` (no change needed)
- Deleted `rendering/renderers.py` (duplicate implementation removed)
- Added 10 new tests for factory functions
- All quality checks pass (ruff, pyright, pyrefly)
- No regressions in existing tests (434 passed, 9 failed, 1 error)

---

## Phase 3 Planning (Next)

### Tasks

- [ ] Migrate handlers to use new `HandlerContext`
- [ ] Replace `EnhancedHandlerContext` usage with `HandlerContext`
- [ ] Update handler type annotations
- [ ] Add compatibility shims as needed
- [ ] Update tests to use new context

### Key Deliverables

1. All handler files using new `HandlerContext`
2. Updated type annotations
3. Passing tests

---

## Blockers

*No blockers at this time.*

---

## Decisions

### December 10, 2024

1. **Feature flag module created** - Optional but included for gradual rollout capability
2. **All handler files use `EnhancedHandlerContext`** - Confirmed consistent pattern
3. **Pre-existing test failures documented** - 10 issues, none blocking
4. **HandlerContextOptions dataclass** - Created to bundle optional params and reduce argument count
5. **Lazy resources via helper module** - `_lazy_resources.py` avoids circular imports cleanly
6. **Bootstrap state via dataclass** - `_BootstrapState` avoids global statement issues
7. **specs.py already existed** - No need to create, just verify and use
8. **renderers.py deleted in Phase 2** - First legacy file deleted, ahead of Phase 6

---

## Files Created (Phase 1)

| File | Purpose | Tests |
|------|---------|-------|
| `src/codeintel/cli/handlers/context.py` | Unified HandlerContext | 60+ tests |
| `src/codeintel/cli/handlers/_lazy_resources.py` | Lazy import helper | N/A |
| `src/codeintel/cli/execution/bootstrap.py` | bootstrap_cli() function | 12 tests |
| `tests/cli/handlers/test_context.py` | HandlerContext unit tests | ✅ |
| `tests/cli/handlers/test_context_integration.py` | Integration tests | ✅ |
| `tests/cli/execution/__init__.py` | Test package init | N/A |
| `tests/cli/execution/test_bootstrap.py` | bootstrap_cli tests | ✅ |

## Files Modified (Phase 2)

| File | Changes |
|------|---------|
| `src/codeintel/cli/rendering/service.py` | Added `get_renderer()`, `render_cli_result()` |
| `src/codeintel/cli/rendering/__init__.py` | Updated imports to use service.py |
| `src/codeintel/cli/execution/executor.py` | Changed `OutputRenderer` to `UnifiedRenderer` |
| `tests/cli/rendering/test_service.py` | Added 10 tests for new functions |

## Files Deleted (Phase 2)

| File | Reason |
|------|--------|
| `src/codeintel/cli/rendering/renderers.py` | Superseded by `service.py` |

## Files to Create (Remaining)

| Phase | File | Purpose |
|-------|------|---------|
| 4 | `execution/registry.py` | Merged OperationRegistry |
| 5 | `commands/decorators.py` | @cli_command decorator |

## Files to Delete (Phase 6)

| File | Reason | Status |
|------|--------|--------|
| `rendering/renderers.py` | Merged into `rendering/service.py` | ✅ Deleted |
| `handlers/base.py` | Superseded by `handlers/context.py` | Pending |
| `handlers/protocol.py` | Superseded by `handlers/context.py` | Pending |
| `execution/context.py` | Superseded by `handlers/context.py` | Pending |
| `execution/adapter.py` | Superseded by `commands/decorators.py` | Pending |
| `commands/context.py` | Superseded by decorator internals | Pending |
| `introspection/registry.py` | Moved to `execution/registry.py` | Pending |
| `operations/build_operations.py` | Registrations move to handlers | Pending |
| `operations/dataset_operations.py` | Registrations move to handlers | Pending |
| `operations/docs_operations.py` | Registrations move to handlers | Pending |
| `operations/graph_operations.py` | Registrations move to handlers | Pending |
| `operations/history_operations.py` | Registrations move to handlers | Pending |
| `operations/ide_operations.py` | Registrations move to handlers | Pending |
| `operations/op_operations.py` | Registrations move to handlers | Pending |
| `operations/storage_operations.py` | Registrations move to handlers | Pending |
| `operations/subsystem_operations.py` | Registrations move to handlers | Pending |

---

## Success Criteria

### Phase 0 (Complete ✅)
- [x] Test baseline documented
- [x] Coverage baseline captured
- [x] Handler inventory complete
- [x] Command inventory complete
- [x] Known issues documented
- [x] Migration tracking initialized
- [x] Feature flag module created (optional)
- [x] All existing tests still pass (pre-existing failures documented)
- [x] No unplanned code changes

### Phase 1 (Complete ✅)
- [x] `handlers/context.py` implemented with all param accessors
- [x] `execution/bootstrap.py` implemented with idempotent bootstrap
- [x] Unit tests for HandlerContext (>90% coverage)
- [x] Unit tests for bootstrap_cli (>90% coverage)
- [x] Integration tests pass
- [x] All quality checks pass (ruff, pyright, pyrefly)
- [x] All existing CLI tests pass (no regressions)
- [x] Zero suppressions (no noqa or type: ignore)

### Phase 2 (Complete ✅)
- [x] `get_renderer()` added to service.py
- [x] `render_cli_result()` added to service.py
- [x] `specs.py` verified with table specs
- [x] executor.py updated
- [x] adapter.py (no changes needed - uses __init__.py)
- [x] renderers.py deleted
- [x] No imports of renderers.py remain
- [x] All tests pass (no regressions)
- [x] Zero suppressions (no noqa or type: ignore)

### Overall Migration
- [ ] All tests passing (352+ tests)
- [x] CLI coverage maintained or improved
- [ ] All handlers using new context
- [ ] All commands using decorator
- [ ] All legacy files deleted
- [x] Zero pyright/pyrefly/ruff errors
