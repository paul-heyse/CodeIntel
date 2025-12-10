# CLI Consolidation Phase 2: Config Integration

> **Status**: Draft  
> **Depends On**: Phase 1 (Foundation Layer)  
> **Enables**: Phase 3 (Handler Migration)  
> **Risk Level**: Low (isolated changes)  
> **Reference**: [CLI_CONSOLIDATION_ARCHITECTURE.md](../CLI_CONSOLIDATION_ARCHITECTURE.md)

## Objective

Integrate `ConfigService` (created in Phase 1) into the Cyclopts application root, removing duplicate configuration loading code from `cyclopts_common.py` and `cyclopts_config.py`.

Upon completion, all CLI commands load configuration through a single path with consistent precedence rules.

---

## Scope

### In Scope

| File | Changes |
|------|---------|
| `cli/cyclopts_app.py` | Update `make_root_app()` to use `ConfigService` |
| `cli/cyclopts_common.py` | Remove config loading code, add deprecation warnings |
| `cli/cyclopts_config.py` | Remove duplicate constants and functions |
| `cli/handlers/base.py` | Update `setup_logging()` to accept `CliConfig` if needed |

### Out of Scope

- Handler migration (Phase 3)
- Rendering changes (Phase 3)
- Deleting files (Phase 4)

---

## Current State: Duplicate Code Inventory

### Constants (to remove)

| Location | Code |
|----------|------|
| `cyclopts_common.py:41` | `CONFIG_ENV_PREFIX = "CODEINTEL_"` |
| `cyclopts_common.py:42` | `CONFIG_PATH_ENV_VAR = "CODEINTEL_CONFIG_PATH"` |
| `cyclopts_common.py:43` | `DEFAULT_CONFIG_PATH = Path("codeintel.toml")` |
| `cyclopts_config.py:24` | `CONFIG_ENV_PREFIX = "CODEINTEL_"` (duplicate) |

### Functions (to deprecate/remove)

| Location | Function |
|----------|----------|
| `cyclopts_common.py:60-85` | `_optional_toml_config()` |
| `cyclopts_common.py:45` | `_ENV_CONFIG = cyclopts_config.Env(...)` |
| `cyclopts_config.py:29-63` | `_resolve_config_path()` |
| `cyclopts_config.py:65-95` | `_get_env_overrides()` |

---

## Target State

### App Construction (after)

```python
# cyclopts_app.py (target)
from codeintel.cli.config import ConfigService

def make_root_app() -> App:
    """Construct root Cyclopts application with unified config."""
    config_service = ConfigService.load(validate=False)
    
    return App(
        name="codeintel",
        help="CodeIntel unified CLI.",
        config=config_service.get_cyclopts_config_chain(),
        default_parameter=Parameter(show_default=True),
        result_action=["call_if_callable", "return_value"],
        print_error=True,
    )
```

### cyclopts_common.py (after)

Remove entirely:
- `CONFIG_ENV_PREFIX`
- `CONFIG_PATH_ENV_VAR`
- `DEFAULT_CONFIG_PATH`
- `_ENV_CONFIG`
- `_optional_toml_config()`

Keep with deprecation warnings:
- `RuntimeCLI` dataclass
- `OutputCLI` dataclass
- `runtime_field()` helper
- `output_field()` helper
- `build_runtime_from_cli()` — add `DeprecationWarning`

---

## Deliverables

### 1. Update `make_root_app()` in `cyclopts_app.py`

**Changes:**
- Import `ConfigService` from `cli.config`
- Replace custom config chain with `config_service.get_cyclopts_config_chain()`
- Add module-level config service cache if needed for downstream access

**Verification:**
- `codeintel --help` works
- `codeintel build status` loads config correctly
- Environment variables override config file values
- `CODEINTEL_CONFIG_PATH` still works

### 2. Remove Duplicates from `cyclopts_common.py`

Remove all config-related constants and the `_optional_toml_config()` function.

Add deprecation warning to `build_runtime_from_cli()`:

```python
import warnings

def build_runtime_from_cli(options: RuntimeCliOptions) -> ProjectRuntime:
    """Build runtime from CLI options.
    
    .. deprecated:: 2.0
        Use RuntimeResolver.resolve(RuntimeParams.from_cyclopts(...)) instead.
    """
    warnings.warn(
        "build_runtime_from_cli is deprecated. "
        "Use RuntimeResolver.resolve(RuntimeParams) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # ... existing implementation
```

### 3. Remove Duplicates from `cyclopts_config.py`

Remove:
- `CONFIG_ENV_PREFIX` constant
- `_resolve_config_path()` function
- `_get_env_overrides()` function

Grep for usages before removing; update callers to use `ConfigService`.

### 4. Update `handlers/base.py` if Needed

If `setup_logging()` loads config separately, update to:
- Accept `CliConfig` parameter, or
- Call `ConfigService.load().config` internally

---

## Testing Strategy

### Regression Tests

```python
def test_config_precedence_preserved():
    """CLI flags > env vars > config file > defaults."""
    # Setup and verify precedence chain

def test_toml_config_still_works():
    """codeintel.toml in current directory is loaded."""
    
def test_yaml_config_still_works():
    """~/.codeintel/config.yaml is loaded."""

def test_env_var_override():
    """CODEINTEL_* env vars override file config."""
```

### Deprecation Warning Tests

```python
def test_build_runtime_from_cli_deprecation_warning():
    """build_runtime_from_cli emits DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="RuntimeResolver"):
        build_runtime_from_cli(RuntimeCliOptions())
```

---

## Acceptance Criteria

1. **`make_root_app()` uses `ConfigService`** — No direct config loading in cyclopts_app.py
2. **Duplicates removed** — `CONFIG_ENV_PREFIX` exists only in `cli/config/`
3. **Deprecation warnings added** — `build_runtime_from_cli()` warns on use
4. **All existing tests pass** — No regression in CLI behavior
5. **Config precedence preserved** — CLI > env > file > default
6. **Both TOML and YAML work** — Either config format is loaded

---

## Migration Checklist

- [ ] Verify ConfigService.get_cyclopts_config_chain() works with real App
- [ ] Update cyclopts_app.py make_root_app()
- [ ] Remove CONFIG_ENV_PREFIX from cyclopts_common.py
- [ ] Remove CONFIG_PATH_ENV_VAR from cyclopts_common.py
- [ ] Remove DEFAULT_CONFIG_PATH from cyclopts_common.py
- [ ] Remove _ENV_CONFIG from cyclopts_common.py
- [ ] Remove _optional_toml_config() from cyclopts_common.py
- [ ] Remove duplicates from cyclopts_config.py
- [ ] Add deprecation warning to build_runtime_from_cli()
- [ ] Update any internal callers of removed functions
- [ ] Run full test suite
- [ ] Manual smoke test: codeintel build status
- [ ] Manual smoke test: codeintel --help
- [ ] Verify env var override still works

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cyclopts config chain API mismatch | Medium | High | Test with mock App before integration |
| Breaking existing env var users | Low | High | Ensure ConfigService respects same env var names |
| Config file search order change | Medium | Medium | Document changes, test both locations |

---

## Estimated Effort

| Task | Lines Changed | Complexity |
|------|--------------|------------|
| Update `cyclopts_app.py` | ~20 | Low |
| Clean `cyclopts_common.py` | ~-80 (removal) | Low |
| Clean `cyclopts_config.py` | ~-60 (removal) | Low |
| Deprecation warnings | ~10 | Low |
| Tests | ~100 | Medium |
| **Total** | **~-10 net** | **Low** |

---

## Dependencies on Phase 1

This phase requires from Phase 1:

- `ConfigService.load()` — Must be working
- `ConfigService.get_cyclopts_config_chain()` — Must return valid Cyclopts config callables
- `ConfigService` must support `CODEINTEL_CONFIG_PATH` env var for path override

---

## Exit Criteria for Phase 3

Phase 3 can begin when:
1. All acceptance criteria above are met
2. PR merged to main branch
3. No deprecation warnings in core CLI operations (only in deprecated functions)

