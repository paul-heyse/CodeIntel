# CLI Consolidation Phase 4: Cleanup

> **Status**: Draft  
> **Depends On**: Phases 1, 2, 3  
> **Enables**: Maintenance mode  
> **Risk Level**: Low (mostly deletions)  
> **Reference**: [CLI_CONSOLIDATION_ARCHITECTURE.md](../CLI_CONSOLIDATION_ARCHITECTURE.md)

## Objective

Remove all deprecated code, backward compatibility shims, and legacy files. This phase completes the consolidation by establishing the target architecture as the only implementation.

**This phase contains breaking changes.** It should coincide with a major version bump if the CLI is versioned separately.

---

## Scope

### Files to Delete

| File | Reason |
|------|--------|
| `cli/common_handlers.py` | Replaced by `resolution/` + `handlers/protocol.py` |
| `cli/cli_render.py` | Replaced by `rendering/service.py` |
| `cli/ide_handlers.py` | Moved to `handlers/ide.py` |
| `cli/subsystem_handlers.py` | Moved to `handlers/subsystem.py` |
| `cli/build_handlers.py` | Moved to `handlers/build.py` |
| `cli/storage_handlers.py` | Moved to `handlers/storage.py` |
| `cli/history_handlers.py` | Moved to `handlers/history.py` |
| `cli/datasets_handlers.py` | Moved to `handlers/datasets.py` |
| `cli/docs_handlers.py` | Moved to `handlers/docs.py` |
| `cli/graphs_handlers.py` | Moved to `handlers/graphs.py` |
| `cli/ops_handlers.py` | Moved to `handlers/ops.py` |

### Code to Remove

| Location | Code |
|----------|------|
| `cli/cyclopts_common.py` | `build_runtime_from_cli()` deprecated function |
| `cli/cli_types.py` | `RuntimeOptions` class (replaced by `RuntimeParams`) |
| Various | All `# type: ignore` comments for removed code |
| Various | All `warnings.warn()` calls for deprecations |

### Aliases to Remove

| Location | Alias | Replacement |
|----------|-------|-------------|
| `cli/cyclopts_common.py` | `RuntimeCliOptions = RuntimeParams` | Direct use of `RuntimeParams` |
| Various imports | Legacy handler imports | New `handlers/` imports |

---

## Pre-Cleanup Verification

Before deleting anything, verify no external usage:

### 1. Search for Imports

```bash
# Check for imports of legacy handler modules
rg "from codeintel.cli import (ide_handlers|subsystem_handlers|build_handlers|...)"
rg "from codeintel.cli.common_handlers import"
rg "from codeintel.cli.cli_render import"

# Check for RuntimeCliOptions usage (should only be in deprecated code)
rg "RuntimeCliOptions" --type py

# Check for build_runtime_from_cli usage
rg "build_runtime_from_cli" --type py
```

### 2. Verify Test Coverage

All tests should be passing with the new implementations. Run:

```bash
pytest tests/cli/ -v --tb=short
```

### 3. Grep for Deprecation Warnings

Ensure deprecation warnings are coming from the files we're about to delete:

```bash
rg "DeprecationWarning" src/codeintel/cli/
```

---

## Deletion Order

Delete in dependency order to catch any remaining usages:

### Step 1: Legacy Handler Files

```bash
# These re-export from handlers/ so safe to delete
rm src/codeintel/cli/ide_handlers.py
rm src/codeintel/cli/subsystem_handlers.py
rm src/codeintel/cli/build_handlers.py
rm src/codeintel/cli/storage_handlers.py
rm src/codeintel/cli/history_handlers.py
rm src/codeintel/cli/datasets_handlers.py
rm src/codeintel/cli/docs_handlers.py
rm src/codeintel/cli/graphs_handlers.py
rm src/codeintel/cli/ops_handlers.py
```

Run tests after each deletion to catch import errors.

### Step 2: common_handlers.py

```bash
rm src/codeintel/cli/common_handlers.py
```

This is a critical file. Verify:
- All functionality moved to `resolution/` and `handlers/protocol.py`
- No remaining imports

### Step 3: cli_render.py

```bash
rm src/codeintel/cli/cli_render.py
```

Verify:
- All functionality moved to `rendering/service.py`
- No remaining imports

### Step 4: Clean Up cli_types.py

Remove `RuntimeOptions` class, keeping only types that are still needed.

### Step 5: Clean Up cyclopts_common.py

Remove:
- `build_runtime_from_cli()` function
- `RuntimeCliOptions` alias
- Any remaining deprecated helpers

Keep:
- `RuntimeCLI` dataclass (Cyclopts integration)
- `OutputCLI` dataclass
- `runtime_field()` helper
- `output_field()` helper
- `command_context()` helper

### Step 6: Update __init__.py Exports

Update `cli/__init__.py` to export only the new API:

```python
"""CodeIntel CLI package."""

from codeintel.cli.config import ConfigService, CliConfig, load_config
from codeintel.cli.resolution import RuntimeParams, RuntimeResolver, ResolvedRuntime
from codeintel.cli.rendering import UnifiedRenderer, RenderContext, OutputFormat
from codeintel.cli.handlers import HandlerContext, HandlerProtocol
from codeintel.cli.results import CliResult
from codeintel.cli.cli_errors import ProblemDetail

__all__ = [
    # Config
    "ConfigService",
    "CliConfig", 
    "load_config",
    # Resolution
    "RuntimeParams",
    "RuntimeResolver",
    "ResolvedRuntime",
    # Rendering
    "UnifiedRenderer",
    "RenderContext",
    "OutputFormat",
    # Handlers
    "HandlerContext",
    "HandlerProtocol",
    # Results
    "CliResult",
    "ProblemDetail",
]
```

---

## Import Updates

After deletions, update any remaining imports in the codebase:

### Find and Replace Patterns

| Old Import | New Import |
|------------|------------|
| `from codeintel.cli.common_handlers import ...` | `from codeintel.cli.resolution import ...` |
| `from codeintel.cli.cli_render import RichRenderer` | `from codeintel.cli.rendering import UnifiedRenderer` |
| `from codeintel.cli.cli_render import PlainRenderer` | `from codeintel.cli.rendering import UnifiedRenderer` |
| `from codeintel.cli.ide_handlers import ...` | `from codeintel.cli.handlers.ide import ...` |
| `from codeintel.cli.build_handlers import ...` | `from codeintel.cli.handlers.build import ...` |
| (etc. for all handler modules) | |

### Update Test Imports

Tests may still import from old locations. Update:

```bash
rg "from codeintel.cli.(ide|subsystem|build|storage|history|datasets|docs|graphs|ops)_handlers" tests/
```

---

## Documentation Updates

### Update CLI_USAGE.md

- Remove references to legacy modules
- Update import examples
- Document new handler pattern

### Update AGENTS.md

If any CLI examples reference old patterns, update them.

### Create MIGRATION_GUIDE.md (Optional)

For external users, document:
- What changed
- How to update their code
- New patterns to follow

---

## CHANGELOG Entry

```markdown
## [2.0.0] - YYYY-MM-DD

### Removed

- **BREAKING**: `codeintel.cli.common_handlers` module removed
  - Use `codeintel.cli.resolution` for runtime resolution
  - Use `codeintel.cli.handlers.protocol` for handler context
  
- **BREAKING**: `codeintel.cli.cli_render` module removed
  - Use `codeintel.cli.rendering.UnifiedRenderer` instead
  
- **BREAKING**: Legacy handler modules removed
  - `ide_handlers.py` → `handlers/ide.py`
  - `subsystem_handlers.py` → `handlers/subsystem.py`
  - `build_handlers.py` → `handlers/build.py`
  - `storage_handlers.py` → `handlers/storage.py`
  - `history_handlers.py` → `handlers/history.py`
  - `datasets_handlers.py` → `handlers/datasets.py`
  - `docs_handlers.py` → `handlers/docs.py`
  - `graphs_handlers.py` → `handlers/graphs.py`
  - `ops_handlers.py` → `handlers/ops.py`

- **BREAKING**: `RuntimeCliOptions` removed
  - Use `RuntimeParams` from `codeintel.cli.resolution`
  
- **BREAKING**: `build_runtime_from_cli()` removed
  - Use `RuntimeResolver.resolve(RuntimeParams)` instead

### Changed

- All CLI handlers now use unified `HandlerContext` pattern
- All CLI output goes through `UnifiedRenderer`
- Configuration loading unified through `ConfigService`
```

---

## Acceptance Criteria

1. **No legacy files remain** — All listed files deleted
2. **No deprecated code** — All `DeprecationWarning` calls removed
3. **No broken imports** — `pyright` and `pyrefly` pass
4. **All tests pass** — Full test suite green
5. **CLI functional** — All commands work correctly
6. **Documentation updated** — No references to deleted modules
7. **CHANGELOG updated** — Breaking changes documented

---

## Cleanup Checklist

- [ ] Verify no external imports of legacy modules
- [ ] Delete ide_handlers.py
- [ ] Delete subsystem_handlers.py  
- [ ] Delete build_handlers.py
- [ ] Delete storage_handlers.py
- [ ] Delete history_handlers.py
- [ ] Delete datasets_handlers.py
- [ ] Delete docs_handlers.py
- [ ] Delete graphs_handlers.py
- [ ] Delete ops_handlers.py
- [ ] Delete common_handlers.py
- [ ] Delete cli_render.py
- [ ] Remove RuntimeOptions from cli_types.py
- [ ] Remove deprecated functions from cyclopts_common.py
- [ ] Update cli/__init__.py exports
- [ ] Update all imports in codebase
- [ ] Update all imports in tests
- [ ] Run full test suite
- [ ] Run pyright
- [ ] Run pyrefly
- [ ] Update CLI_USAGE.md
- [ ] Update CHANGELOG
- [ ] Manual smoke test of key commands

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| External code depends on deleted modules | Low | High | Pre-cleanup verification, major version bump |
| Missed import somewhere | Medium | Low | pyright/pyrefly will catch |
| Behavior regression | Low | Medium | Comprehensive testing before deletion |

---

## Estimated Effort

| Task | Effort |
|------|--------|
| Pre-cleanup verification | 1 hour |
| File deletions | 30 min |
| Import updates | 1-2 hours |
| Test fixes | 1-2 hours |
| Documentation | 1 hour |
| **Total** | **~5-6 hours** |

---

## Dependencies

### From Phase 3

- All handlers migrated to `handlers/` package
- All `cyclopts_*.py` files updated to use new handlers
- All handlers return `CliResult[T]`
- Zero `sys.stdout.write()` calls in handler code

### Verification Before Starting

Confirm Phase 3 is complete:
```bash
# No RuntimeCliOptions in handler files (except as deprecated alias)
rg "class RuntimeCliOptions" src/codeintel/cli/*_handlers.py

# No direct stdout writes in handlers
rg "sys\.stdout\.write" src/codeintel/cli/handlers/

# All handlers use HandlerContext
rg "HandlerContext" src/codeintel/cli/handlers/*.py
```

---

## Post-Cleanup State

After Phase 4, the CLI package structure is:

```
cli/
├── __init__.py                # Clean public API
├── cyclopts_app.py           # Root app (uses ConfigService)
├── cyclopts_common.py        # RuntimeCLI, OutputCLI, command_context
├── cyclopts_*.py             # Command wiring (thin)
│
├── config/                   # Configuration (SSoT)
│   ├── __init__.py
│   ├── service.py           # ConfigService
│   ├── model.py             # CliConfig
│   └── ...
│
├── resolution/               # Runtime resolution (SSoT)
│   ├── __init__.py
│   ├── runtime.py           # RuntimeParams, RuntimeResolver
│   ├── gateway.py           # GatewayManager
│   └── ...
│
├── rendering/                # Output rendering (SSoT)
│   ├── __init__.py
│   ├── service.py           # UnifiedRenderer
│   ├── types.py             # RenderContext, OutputFormat
│   └── ...
│
├── handlers/                 # Business logic
│   ├── __init__.py
│   ├── protocol.py          # HandlerContext, HandlerProtocol
│   ├── build.py
│   ├── datasets.py
│   ├── docs.py
│   ├── graphs.py
│   ├── history.py
│   ├── health.py
│   ├── ide.py
│   ├── ops.py
│   ├── storage.py
│   └── subsystem.py
│
├── execution/                # Existing
├── operations/               # Existing
├── plugins/                  # Existing
├── completions/              # Existing
│
├── results.py               # CliResult
├── cli_types.py             # Remaining types
├── cli_errors.py            # ProblemDetail, errors
├── pipelines.py             # Refactored to use rendering/
└── ...                      # Other utility files
```

**No more:**
- `*_handlers.py` at top level
- `common_handlers.py`
- `cli_render.py`
- Duplicate `RuntimeCliOptions` definitions
- Multiple `build_runtime_from_cli()` implementations
- Direct `sys.stdout.write()` in business logic

