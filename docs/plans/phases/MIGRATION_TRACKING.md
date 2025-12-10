# CLI Migration Tracking

> **Started:** December 10, 2024  
> **Target Completion:** January 21, 2025 (6 weeks)  
> **Current Phase:** 0 (Preparation) ✅

## Phase Status

| Phase | Name | Status | Started | Completed | Notes |
|-------|------|--------|---------|-----------|-------|
| 0 | Preparation | ✅ Complete | Dec 10, 2024 | Dec 10, 2024 | Baseline captured |
| 1 | Foundation Layer | ⚪ Not Started | | | `HandlerContext`, `bootstrap_cli()` |
| 2 | Rendering Consolidation | ⚪ Not Started | | | Single `UnifiedRenderer` |
| 3 | Handler Migration | ⚪ Not Started | | | All handlers use new context |
| 4 | Registry Unification | ⚪ Not Started | | | Single `OperationRegistry` |
| 5 | Command Decorator | ⚪ Not Started | | | `@cli_command` + migrate commands |
| 6 | Legacy Cleanup | ⚪ Not Started | | | Delete superseded files |

## Key Metrics

| Metric | Baseline | Current | Target |
|--------|----------|---------|--------|
| Test count | 362 | 362 | >= 362 |
| Test pass rate | 97.2% (352/362) | 97.2% | 100% |
| CLI coverage | ~48% | ~48% | >= 48% |
| Handler files migrated | 0/13 | 0/13 | 13/13 |
| Command files migrated | 0/14 | 0/14 | 14/14 |
| Legacy files deleted | 0/15 | 0/15 | 15/15 |

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

---

## Phase 1 Planning (Next)

### Tasks

- [ ] Create unified `HandlerContext` in `handlers/context.py`
- [ ] Implement `bootstrap_cli()` in `execution/bootstrap.py`
- [ ] Add typed parameter accessor methods
- [ ] Add lazy resource resolution (gateway, runtime, graph_runtime)
- [ ] Create context manager support
- [ ] Write comprehensive unit tests
- [ ] Update documentation

### Key Deliverables

1. `src/codeintel/cli/handlers/context.py` - New unified HandlerContext
2. `src/codeintel/cli/execution/bootstrap.py` - Bootstrap function
3. Unit tests for new components
4. Updated handler inventory with migration progress

---

## Blockers

*No blockers at this time.*

---

## Decisions

### December 10, 2024

1. **Feature flag module created** - Optional but included for gradual rollout capability
2. **All handler files use `EnhancedHandlerContext`** - Confirmed consistent pattern
3. **Pre-existing test failures documented** - 10 issues, none blocking

---

## Files to Create (New)

| Phase | File | Purpose |
|-------|------|---------|
| 1 | `handlers/context.py` | Unified HandlerContext |
| 1 | `execution/bootstrap.py` | bootstrap_cli() function |
| 4 | `execution/registry.py` | Merged OperationRegistry |
| 5 | `commands/decorators.py` | @cli_command decorator |

## Files to Delete (Phase 6)

| File | Reason |
|------|--------|
| `handlers/base.py` | Superseded by `handlers/context.py` |
| `handlers/protocol.py` | Superseded by `handlers/context.py` |
| `execution/context.py` | Superseded by `handlers/context.py` |
| `execution/adapter.py` | Superseded by `commands/decorators.py` |
| `commands/context.py` | Superseded by decorator internals |
| `rendering/renderers.py` | Merged into `rendering/service.py` |
| `introspection/registry.py` | Moved to `execution/registry.py` |
| `operations/build_operations.py` | Registrations move to handlers |
| `operations/dataset_operations.py` | Registrations move to handlers |
| `operations/docs_operations.py` | Registrations move to handlers |
| `operations/graph_operations.py` | Registrations move to handlers |
| `operations/history_operations.py` | Registrations move to handlers |
| `operations/ide_operations.py` | Registrations move to handlers |
| `operations/op_operations.py` | Registrations move to handlers |
| `operations/storage_operations.py` | Registrations move to handlers |
| `operations/subsystem_operations.py` | Registrations move to handlers |

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

### Overall Migration
- [ ] All tests passing (352+ tests)
- [ ] CLI coverage maintained or improved
- [ ] All handlers using new context
- [ ] All commands using decorator
- [ ] All legacy files deleted
- [ ] Zero pyright/pyrefly/ruff errors
