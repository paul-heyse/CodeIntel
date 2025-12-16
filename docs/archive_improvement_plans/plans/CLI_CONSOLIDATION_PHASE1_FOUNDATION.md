# CLI Consolidation Phase 1: Foundation Layer

> **Status**: Draft  
> **Depends On**: None  
> **Enables**: Phases 2, 3, 4  
> **Risk Level**: Low (purely additive)  
> **Reference**: [CLI_CONSOLIDATION_ARCHITECTURE.md](../CLI_CONSOLIDATION_ARCHITECTURE.md)

## Objective

Create all new infrastructure components required by the target architecture **without modifying any existing code**. This phase establishes the foundation that subsequent phases will migrate to.

Upon completion, the new infrastructure exists alongside the old, fully tested, and ready for adoption.

---

## Scope

### In Scope

| Component | Location | Purpose |
|-----------|----------|---------|
| `ConfigService` | `cli/config/service.py` | Unified config loading with Cyclopts integration |
| `RuntimeParams` | `cli/resolution/runtime.py` | Canonical runtime parameters type |
| `RuntimeResolver` enhancements | `cli/resolution/runtime.py` | Accept `RuntimeParams`, factory methods |
| `RenderContext` | `cli/rendering/types.py` | Output context with format/color/TTY |
| `UnifiedRenderer` | `cli/rendering/service.py` | Single rendering implementation |
| `TableSpec`, `ColumnSpec` | `cli/rendering/table.py` | Table specification types |
| `HandlerContext` | `cli/handlers/protocol.py` | Unified handler context |
| `HandlerProtocol` | `cli/handlers/protocol.py` | Handler contract definition |
| Pre-built table specs | `cli/rendering/specs.py` | Standard table definitions |

### Out of Scope

- Modifying existing handler files
- Modifying `cyclopts_*.py` files
- Removing any existing code
- Deprecation warnings (Phase 2+)

---

## Deliverables

### 1. Configuration Service (`cli/config/service.py`)

**Requirements:**
- Implement `ConfigService` dataclass per architecture spec §3.1
- `load()` class method with precedence: CLI flags > env vars > config file > defaults
- `get_cyclopts_config_chain()` returning callables compatible with `cyclopts.App.config`
- `with_overrides()` for testing scenarios
- Support both `codeintel.toml` (Cyclopts style) and `~/.codeintel/config.yaml` (config package style)
- Thread-safe (stateless load, immutable result)

**Key Decisions for Implementer:**
- Determine how to merge Cyclopts' TOML loading with the existing `cli/config/loader.py`
- Decide whether `ConfigService` wraps `load_config()` or replaces its internals
- Handle config file search order when both TOML and YAML exist

### 2. Resolution Enhancements (`cli/resolution/`)

**`RuntimeParams` type:**
- Implement per architecture spec §3.2
- All fields optional with sensible defaults
- Factory methods: `from_context()`, `from_cyclopts()`, `minimal()`
- Immutable (`frozen=True`)

**`RuntimeResolver` updates:**
- Add `resolve(params: RuntimeParams, *, allow_fallback: bool = True)` overload
- Preserve existing `resolve(ctx: ExecutionContext)` for backward compatibility
- Both methods should share core resolution logic (DRY)

**Key Decisions for Implementer:**
- Decide internal structure: single implementation with adapters, or branching logic
- Determine how `BackendFlags` integrates with existing `GraphBackendConfig`
- Handle the `RuntimeCLI` → `RuntimeParams` conversion edge cases (verbose field, etc.)

### 3. Rendering Package (`cli/rendering/`)

**Package structure:**
```
cli/rendering/
├── __init__.py      # Public API exports
├── types.py         # RenderContext, OutputFormat enum, JustifyMethod
├── table.py         # TableSpec, ColumnSpec
├── service.py       # UnifiedRenderer, RenderingService protocol
├── specs.py         # Pre-built table specs (OPERATIONS_TABLE, etc.)
└── streaming.py     # StreamingEmitter for JSONL mode (optional, can defer)
```

**`UnifiedRenderer` requirements:**
- Implement `RenderingService` protocol per §3.3
- `render_result(result: CliResult[T]) -> int` as primary entry point
- `render_table()` with format negotiation (Rich vs plain vs JSON)
- `render_error()` with RFC 9457 Problem Details
- `render_message()` for simple status messages
- TTY detection via `RenderContext.auto_detect()`
- `RenderContext.for_testing()` returning captured StringIO streams

**Key Decisions for Implementer:**
- Determine how much of `cli_render.py`'s `RichRenderer`/`PlainRenderer` to reuse vs rewrite
- Decide Rich console lifecycle (create per render vs cache)
- Handle edge cases: empty tables, None data, nested dicts

### 4. Handler Protocol (`cli/handlers/protocol.py`)

**Requirements:**
- `HandlerContext` dataclass per §3.4
- Lazy `gateway` and `graph_runtime` properties via `GatewayManager`
- `logger` property returning appropriately named logger
- `close()` method for resource cleanup
- `HandlerProtocol[T]` as `Protocol` for type checking

**`GatewayManager` updates (if needed):**
- Ensure `cli/resolution/gateway.py` can be instantiated with `ResolvedRuntime`
- Add `graph_runtime` lazy property if not present

**Key Decisions for Implementer:**
- Determine `_operation_name` derivation strategy for logger naming
- Decide if `HandlerContext` should hold a back-reference to `ExecutionContext`
- Handle the case where handler doesn't need gateway (avoid opening connection)

---

## Testing Requirements

### Unit Tests

| Component | Test Coverage |
|-----------|--------------|
| `ConfigService.load()` | Default loading, explicit path, env overrides, CLI overrides, validation errors |
| `ConfigService.get_cyclopts_config_chain()` | Returns valid callables, integrates with mock App |
| `RuntimeParams` factories | `from_context()`, `from_cyclopts()`, `minimal()` with various inputs |
| `RuntimeResolver.resolve()` | Project discovery success, fallback to params, missing params error |
| `RenderContext.auto_detect()` | TTY vs non-TTY, format overrides |
| `RenderContext.for_testing()` | Returns captured streams |
| `UnifiedRenderer.render_result()` | Success case, failure case, warnings, metadata |
| `UnifiedRenderer.render_table()` | TEXT/JSON/JSONL formats, empty table |
| `HandlerContext` properties | Lazy gateway access, logger naming, close() |

### Integration Tests

- `ConfigService` loads real config files (TOML and YAML)
- `RuntimeResolver` discovers actual project in test fixtures
- `UnifiedRenderer` produces valid JSON output parseable by `jq`

---

## Acceptance Criteria

1. **All new code compiles** — `pyright` and `pyrefly` pass with zero errors
2. **No existing code modified** — `git diff` shows only new files
3. **Test coverage** — New code has ≥90% line coverage
4. **Public API documented** — All public classes/functions have NumPy docstrings
5. **Importable** — The following imports work without error:
   ```python
   from codeintel.cli.config import ConfigService
   from codeintel.cli.resolution import RuntimeParams, RuntimeResolver
   from codeintel.cli.rendering import UnifiedRenderer, RenderContext, TableSpec
   from codeintel.cli.handlers import HandlerContext, HandlerProtocol
   ```
6. **No circular imports** — Import graph remains acyclic

---

## Implementation Notes

### File Creation Order (Suggested)

1. `cli/rendering/types.py` — No dependencies on other new code
2. `cli/rendering/table.py` — Depends only on types.py
3. `cli/rendering/specs.py` — Depends on table.py
4. `cli/rendering/service.py` — Depends on types.py, table.py, existing `cli_errors.py`
5. `cli/rendering/__init__.py` — Re-exports
6. `cli/handlers/protocol.py` — Depends on rendering, resolution (existing)
7. `cli/config/service.py` — Depends on existing config package
8. Resolution enhancements — Extends existing code

### Patterns to Follow

- Use `from __future__ import annotations` in all new files
- Follow existing patterns in `cli/config/model.py` for frozen dataclasses
- Use `TYPE_CHECKING` guards for heavy imports per AGENTS.md typing gates
- Match existing error patterns in `cli/cli_errors.py`

### Patterns to Avoid

- Don't create circular dependencies between new packages
- Don't import from `common_handlers.py` (will be deleted in Phase 4)
- Don't add optional parameters that aren't immediately useful
- Don't over-engineer — keep implementations minimal for this phase

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cyclopts config chain complexity | Medium | Low | Start with simple pass-through, enhance later |
| Rich console thread safety | Low | Medium | Create new console per `UnifiedRenderer` instance |
| Gateway lifecycle edge cases | Medium | Medium | Extensive testing of lazy init + close patterns |

---

## Estimated Effort

| Component | Lines of Code | Complexity |
|-----------|--------------|------------|
| `ConfigService` | ~150 | Medium |
| `RuntimeParams` + factories | ~100 | Low |
| `RuntimeResolver` enhancements | ~50 | Low |
| Rendering package | ~400 | Medium |
| `HandlerContext` + protocol | ~150 | Low |
| Tests | ~500 | Medium |
| **Total** | **~1350** | **Medium** |

---

## Exit Criteria for Phase 2

Phase 2 can begin when:
1. All acceptance criteria above are met
2. PR merged to main branch
3. CI passes on all platforms

