# CLI Consolidation Phase 3: Handler Migration

> **Status**: Draft  
> **Depends On**: Phase 1 (Foundation Layer), Phase 2 (Config Integration)  
> **Enables**: Phase 4 (Cleanup)  
> **Risk Level**: Medium-High (touches all handler code)  
> **Reference**: [CLI_CONSOLIDATION_ARCHITECTURE.md](../CLI_CONSOLIDATION_ARCHITECTURE.md)

## Objective

Migrate all CLI handlers from the legacy pattern (per-module `RuntimeCliOptions`, direct stdout writes) to the unified pattern (`HandlerContext`, `CliResult`, `UnifiedRenderer`).

Upon completion, all handlers:
- Accept `HandlerContext` as their primary input
- Return `CliResult[T]` (never write to stdout directly)
- Use `RuntimeParams` via `RuntimeResolver.resolve()`

---

## Migration Strategy: Three Waves

Handlers are migrated in order of increasing complexity. Each wave validates the pattern before moving to more complex handlers.

| Wave | Handlers | Complexity | stdout.write Calls |
|------|----------|------------|-------------------|
| **Wave 1** | ide, subsystem | Low | 0 |
| **Wave 2** | build, storage, history, health | Medium | 0 |
| **Wave 3** | datasets, docs, graphs, ops | High | 21 total |

Each wave is a self-contained deliverable that can be merged independently.

---

## Handler Migration Pattern

### Before (Legacy Pattern)

```python
# xxx_handlers.py

@dataclass(frozen=True)
class RuntimeCliOptions:
    project_root: Path | None = None
    # ... varying fields per module

@dataclass(frozen=True)
class MyCommandOptions:
    param1: str
    runtime_options: RuntimeCliOptions
    verbose: int = 0

def my_command_handler(options: MyCommandOptions) -> CliResult[dict[str, Any]]:
    setup_logging(options.verbose)
    runtime = build_runtime_from_cli(options.runtime_options)
    gateway = open_gateway(runtime.paths.db_path)
    
    # Business logic
    result = do_stuff(gateway, options.param1)
    
    # Direct output (anti-pattern)
    sys.stdout.write(json.dumps(result))
    
    gateway.close()
    return CliResult.ok(result)
```

### After (Unified Pattern)

```python
# cli/handlers/xxx.py

def my_command_handler(ctx: HandlerContext) -> CliResult[MyCommandData]:
    """Execute my command.
    
    Parameters
    ----------
    ctx
        Handler context. Expects ctx.params["param1"].
        
    Returns
    -------
    CliResult[MyCommandData]
        Command result for rendering.
    """
    param1 = ctx.params.get("param1")
    if not param1:
        return CliResult.fail(
            ProblemDetail(
                type="urn:codeintel:cli:validation/missing-param",
                title="Missing Parameter",
                detail="param1 is required",
            )
        )
    
    # Business logic using ctx.gateway (lazy), ctx.logger
    ctx.logger.info("Processing with param1=%s", param1)
    result = do_stuff(ctx.gateway, str(param1))
    
    # Return result, never write to stdout
    return CliResult.ok(MyCommandData(value=result))


# cyclopts_xxx.py (wiring layer)

@xxx_app.command()
def my_command(
    param1: str,
    runtime: RuntimeCLI = runtime_field(),
    output: OutputCLI = output_field(),
) -> int:
    """Execute my command."""
    with command_context(runtime, output, {"param1": param1}) as (ctx, renderer):
        result = my_command_handler(ctx)
        return renderer.render_result(result)
```

---

## Wave 1: Simple Handlers

### Scope

| File | New Location | Handlers |
|------|--------------|----------|
| `ide_handlers.py` | `handlers/ide.py` | `ide_hints_handler` |
| `subsystem_handlers.py` | `handlers/subsystem.py` | ~5 handlers |
| `cyclopts_ide.py` | (update) | Command wiring |
| `cyclopts_subsystem.py` | (update) | Command wiring |

### Key Changes

1. **Delete `RuntimeCliOptions`** from both files
2. **Update handlers** to accept `HandlerContext`
3. **Update Cyclopts commands** to use `command_context()` helper
4. **Move handler functions** to `handlers/` package

### Acceptance Criteria (Wave 1)

- [ ] `codeintel ide hints <path>` works with new pattern
- [ ] `codeintel subsystem *` commands work with new pattern
- [ ] No `RuntimeCliOptions` in ide_handlers.py or subsystem_handlers.py
- [ ] Handlers in `handlers/ide.py` and `handlers/subsystem.py`
- [ ] All tests pass

---

## Wave 2: Medium Handlers

### Scope

| File | New Location | Handlers | Notes |
|------|--------------|----------|-------|
| `build_handlers.py` | `handlers/build.py` | ~8 handlers | Already uses common_handlers |
| `storage_handlers.py` | `handlers/storage.py` | ~5 handlers | |
| `history_handlers.py` | `handlers/history.py` | ~4 handlers | |
| `health.py` | `handlers/health.py` | ~3 handlers | |

### Key Changes

1. **Migrate `build_handlers.py`** (already closer to target pattern)
2. **Update all handler signatures** to `HandlerContext -> CliResult[T]`
3. **Create typed data classes** for handler return types
4. **Update corresponding `cyclopts_*.py`** files

### Build Handlers Specifics

`build_handlers.py` imports from `common_handlers.py`. The migration should:
- Replace `RuntimeCliOptions` (alias) with `RuntimeParams`
- Replace `build_runtime_from_cli()` with `RuntimeResolver.resolve()`
- Remove direct gateway management, use `ctx.gateway`

### Acceptance Criteria (Wave 2)

- [ ] `codeintel build *` commands work with new pattern
- [ ] `codeintel storage *` commands work with new pattern
- [ ] `codeintel history *` commands work with new pattern
- [ ] `codeintel health *` commands work with new pattern
- [ ] All handlers in `handlers/` package
- [ ] All tests pass

---

## Wave 3: Complex Handlers

### Scope

| File | New Location | Handlers | stdout.write Calls |
|------|--------------|----------|-------------------|
| `datasets_handlers.py` | `handlers/datasets.py` | ~15 handlers | 0 |
| `docs_handlers.py` | `handlers/docs.py` | ~10 handlers | 9 |
| `graphs_handlers.py` | `handlers/graphs.py` | ~8 handlers | 12 |
| `ops_handlers.py` | `handlers/ops.py` | ~6 handlers | 0 |

### stdout.write Remediation

Files with direct `sys.stdout.write()` calls require special attention:

**`graphs_handlers.py`** (12 instances):
- Replace with `return CliResult.ok(data)` + JSON formatting in renderer
- For streaming graph output, use `StreamingEmitter` if batch is large

**`docs_handlers.py`** (9 instances):
- Review each call: some may be legitimate streaming
- Convert to `CliResult` return or `StreamingEmitter`

### datasets_handlers.py Specifics

This is the largest handler file (2,117 lines). Migration approach:
1. Start with the `RuntimeCliOptions` class removal
2. Update one handler at a time
3. Each handler gets a typed return dataclass
4. Preserve all nested option classes as `ctx.params` entries

### Acceptance Criteria (Wave 3)

- [ ] `codeintel datasets *` commands work with new pattern
- [ ] `codeintel docs *` commands work with new pattern  
- [ ] `codeintel graphs *` commands work with new pattern
- [ ] `codeintel ops *` commands work with new pattern
- [ ] **Zero** `sys.stdout.write()` calls in handler files
- [ ] All handlers in `handlers/` package
- [ ] All tests pass

---

## Command Context Helper

The `command_context()` helper simplifies Cyclopts wiring:

```python
# cli/cyclopts_common.py

@contextmanager
def command_context(
    runtime: RuntimeCLI,
    output: OutputCLI,
    params: dict[str, object],
) -> Iterator[tuple[HandlerContext, UnifiedRenderer]]:
    """Standard context manager for command wiring."""
    config_service = ConfigService.load()
    resolved = RuntimeResolver.resolve(RuntimeParams.from_cyclopts(runtime))
    
    ctx = HandlerContext(
        config=config_service.config,
        runtime=resolved,
        params=params,
        verbosity=runtime.verbose,
    )
    
    setup_logging(ctx.verbosity, config=ctx.config)
    
    render_ctx = RenderContext.auto_detect(
        format_override=OutputFormat.JSON if output.json else output.output_format,
    )
    renderer = UnifiedRenderer(render_ctx)
    
    try:
        yield ctx, renderer
    finally:
        ctx.close()
```

Commands that don't need full runtime resolution can use a simpler pattern or `RuntimeParams.minimal()`.

---

## Handler Return Types

Each handler should define a typed return dataclass:

```python
# handlers/build.py

@dataclass(frozen=True)
class BuildStatusData:
    """Data returned by build_status_handler."""
    
    targets: list[TargetStatus]
    total_count: int
    completed_count: int
    failed_count: int


def build_status_handler(ctx: HandlerContext) -> CliResult[BuildStatusData]:
    ...
```

For handlers returning tabular data:

```python
@dataclass(frozen=True)
class DatasetListData:
    """Data returned by dataset_list_handler."""
    
    rows: list[DatasetRow]
    table_spec: TableSpec = field(default=DATASETS_TABLE)
```

---

## File Organization

### handlers/ Package Structure (End State)

```
cli/handlers/
├── __init__.py          # Public API exports
├── protocol.py          # HandlerContext, HandlerProtocol
├── build.py             # Build command handlers
├── datasets.py          # Dataset command handlers
├── docs.py              # Documentation command handlers
├── graphs.py            # Graph command handlers
├── history.py           # History command handlers
├── health.py            # Health check handlers
├── ide.py               # IDE integration handlers
├── ops.py               # Operation handlers
├── storage.py           # Storage command handlers
└── subsystem.py         # Subsystem command handlers
```

### Legacy Files (Keep Until Phase 4)

During migration, keep legacy handler files with deprecation notices:

```python
# ide_handlers.py (during Phase 3)
"""Legacy IDE handlers.

.. deprecated:: 2.0
    This module is deprecated. Use codeintel.cli.handlers.ide instead.
"""
from __future__ import annotations

import warnings

warnings.warn(
    "codeintel.cli.ide_handlers is deprecated. "
    "Use codeintel.cli.handlers.ide instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for compatibility
from codeintel.cli.handlers.ide import *  # noqa: F401, F403
```

---

## Testing Strategy

### Unit Tests (Per Handler)

```python
def test_handler_success(test_ctx: TestContext) -> None:
    """Handler returns success result with valid data."""
    test_ctx.require(CORE_PACK)
    
    ctx = HandlerContext(
        config=test_ctx.config,
        runtime=test_ctx.resolved_runtime,
        params={"param1": "value"},
    )
    
    result = my_handler(ctx)
    
    assert result.success
    assert result.data is not None
    # Assert on specific data fields


def test_handler_missing_param(test_ctx: TestContext) -> None:
    """Handler returns failure on missing required param."""
    ctx = HandlerContext(
        config=test_ctx.config,
        runtime=test_ctx.resolved_runtime,
        params={},  # Missing param1
    )
    
    result = my_handler(ctx)
    
    assert not result.success
    assert result.error.type == "urn:codeintel:cli:validation/missing-param"
```

### Integration Tests (Per Command)

```python
def test_command_e2e(tmp_path: Path, cli_runner: CliRunner) -> None:
    """Command works end-to-end."""
    # Setup test project
    setup_project(tmp_path)
    
    result = cli_runner.invoke(["build", "status", "--root", str(tmp_path)])
    
    assert result.exit_code == 0
    # Assert on output
```

---

## Acceptance Criteria (Overall Phase 3)

1. **All handlers migrated** — Every handler in `handlers/` package
2. **Unified context** — All handlers accept `HandlerContext`
3. **Unified results** — All handlers return `CliResult[T]`
4. **No direct stdout** — Zero `sys.stdout.write()` in handler code
5. **No RuntimeCliOptions variants** — Only `RuntimeParams` used
6. **All tests pass** — Including new handler unit tests
7. **CLI works** — All commands function correctly

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Large scope causes merge conflicts | Merge each wave as separate PR |
| Subtle behavior changes | Extensive testing, review output diffs |
| Performance regression in rendering | Benchmark before/after for key commands |
| Breaking external scripts | Ensure JSON output structure unchanged |

---

## Estimated Effort

| Wave | Handler Files | Cyclopts Files | Estimated Lines |
|------|--------------|----------------|-----------------|
| Wave 1 | 2 | 2 | ~400 |
| Wave 2 | 4 | 4 | ~800 |
| Wave 3 | 4 | 4 | ~1500 |
| Tests | — | — | ~800 |
| **Total** | **10** | **10** | **~3500** |

---

## Dependencies

### From Phase 1

- `RuntimeParams` with all factory methods
- `RuntimeResolver.resolve()` accepting `RuntimeParams`
- `HandlerContext` and `HandlerProtocol`
- `UnifiedRenderer` with all rendering methods
- `RenderContext.auto_detect()`

### From Phase 2

- `ConfigService.load()` integrated into Cyclopts app
- `build_runtime_from_cli()` has deprecation warning

---

## Exit Criteria for Phase 4

Phase 4 can begin when:
1. All three waves completed and merged
2. All acceptance criteria met
3. No handler code outside `handlers/` package (except deprecated re-exports)
4. CI passes on all platforms

