# Test Migration Guide: Hexagonal Architecture

This guide provides recipes for migrating existing tests to the new hexagonal test architecture.

## Quick Reference

| Old Pattern | New Pattern | Fixture/Helper |
|-------------|-------------|----------------|
| `fresh_gateway` + manual inserts | `core_ctx` / `graph_ctx` | Pre-seeded context |
| `open_ingestion_gateway()` | `test_ctx` fixture | TestContext |
| `gateway.con.execute("INSERT...")` | `ctx.require(PACK)` | Seed packs |
| `insert_modules()`, `insert_goids()` | `CORE_PACK` | Automatic seeding |
| `SpanTestEnv` | `SpanEnvAdapter` | Env adapters |
| `PipelineEnv` | `PipelineEnvAdapter` | Env adapters |
| `CoverageEdgeEnv` | `CoverageEnvAdapter` | Env adapters |

## Available Seed Packs

| Pack | Tables Seeded | Use Case |
|------|---------------|----------|
| `CORE_PACK` | `core.modules`, `core.goids`, `core.repo_map` | Basic module/function data |
| `GRAPH_PACK` | `graph.call_graph_*`, `graph.import_graph_*`, `graph.cfg_*`, `graph.dfg_*` | Call/import graphs, CFG/DFG |
| `COVERAGE_PACK` | `analytics.test_catalog`, `analytics.test_coverage_edges`, `analytics.coverage_functions` | Test coverage data |
| `METRICS_PACK` | `analytics.function_metrics`, `analytics.goid_risk_factors`, `analytics.typedness`, etc. | Function metrics and risk |
| `DOCSTRING_PACK` | `core.docstrings` | Parsed docstring data |
| `SUBSYSTEM_PACK` | `analytics.subsystems`, `analytics.subsystem_modules` | Architectural groupings |
| `SYMBOL_PACK` | `graph.symbol_use_edges` | Symbol usage relationships |
| `CONFIG_PACK` | `analytics.config_values` | Configuration references |

## Available Fixtures

| Fixture | Seeds Applied | Use Case |
|---------|---------------|----------|
| `test_ctx` | None | Minimal context, apply seeds manually |
| `core_ctx` | `CORE_PACK` | Tests needing basic module/goid data |
| `graph_ctx` | `GRAPH_PACK` (includes CORE_PACK) | Graph analysis tests |
| `coverage_ctx` | `COVERAGE_PACK` (includes CORE_PACK) | Coverage computation tests |
| `metrics_ctx` | `METRICS_PACK` (includes CORE_PACK) | Metrics analysis tests |
| `full_ctx` | All packs | Integration tests |
| `scenario_builder` | Configurable | Complex custom scenarios |

---

## Recipe A: `fresh_gateway` to Context Fixture

### Before

```python
from codeintel.storage.gateway import StorageGateway

def test_something(fresh_gateway: StorageGateway) -> None:
    fresh_gateway.core.insert_modules([("mod", "mod.py", "repo", "commit")])
    fresh_gateway.core.insert_goids([...])
    
    result = some_function(fresh_gateway)
    assert result is not None
```

### After

```python
from tests._helpers.context import TestContext

def test_something(core_ctx: TestContext) -> None:
    # Seeds already applied via fixture
    result = some_function(core_ctx.gateway)
    assert result is not None
```

### When to Use Which Fixture

- **`test_ctx`**: Need gateway but will apply custom seeds
- **`core_ctx`**: Need basic modules/goids
- **`graph_ctx`**: Need call graphs, import graphs, CFG/DFG
- **`coverage_ctx`**: Need test coverage data
- **`metrics_ctx`**: Need function metrics, risk factors
- **`full_ctx`**: Need everything for integration tests

---

## Recipe B: Raw SQL to Seed Packs

### Before

```python
def test_something(fresh_gateway: StorageGateway) -> None:
    con = fresh_gateway.con
    now = datetime.now(UTC).isoformat()
    
    con.execute(
        "INSERT INTO core.modules (module, path, repo, commit, language) "
        "VALUES ('mod', 'mod.py', 'repo', 'commit', 'python')"
    )
    con.execute(
        "INSERT INTO core.goids (goid_h128, urn, repo, commit, ...) "
        "VALUES (1, 'urn:mod.func', 'repo', 'commit', ...)"
    )
    
    result = some_function(fresh_gateway)
```

### After

```python
from tests._helpers.context import TestContext
from tests._helpers.seeds import CORE_PACK

def test_something(test_ctx: TestContext) -> None:
    test_ctx.require(CORE_PACK)
    
    result = some_function(test_ctx.gateway)
```

### Or Use Pre-Seeded Fixture

```python
def test_something(core_ctx: TestContext) -> None:
    # CORE_PACK already applied
    result = some_function(core_ctx.gateway)
```

---

## Recipe C: `open_ingestion_gateway` to TestContext

### Before

```python
from tests._helpers.gateway import open_ingestion_gateway

def test_something(tmp_path: Path) -> None:
    gateway = open_ingestion_gateway(
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
    )
    try:
        insert_modules(gateway, [...])
        insert_goids(gateway, [...])
        
        result = some_function(gateway)
        assert result is not None
    finally:
        gateway.con.close()
```

### After

```python
from tests._helpers.context import TestContext

def test_something(graph_ctx: TestContext) -> None:
    # Gateway already opened with schema, seeds applied
    result = some_function(graph_ctx.gateway)
    assert result is not None
    # Context manager handles cleanup
```

### For Custom Setup

```python
from tests._helpers.scenarios import TestScenario
from tests._helpers.seeds import CORE_PACK, GRAPH_PACK

def test_something(tmp_path: Path) -> None:
    ctx = (
        TestScenario()
        .with_repo("custom/repo")
        .with_commit("custom123")
        .with_seeds(CORE_PACK, GRAPH_PACK)
        .build(tmp_path)
    )
    
    with ctx:
        result = some_function(ctx.gateway)
        assert result is not None
```

---

## Recipe D: Env Classes to Adapters

### Before (SpanTestEnv)

```python
from tests._helpers.graph_env import SpanTestEnv, create_span_test_env

def test_something(tmp_path: Path, fresh_gateway: StorageGateway) -> None:
    env = create_span_test_env(tmp_path, fresh_gateway)
    
    result = some_function(env.gateway, env.expected_goid)
    assert result is not None
```

### After

```python
from tests._helpers.env_adapters import SpanEnvAdapter, create_span_env_from_context

def test_something(tmp_path: Path) -> None:
    env = create_span_env_from_context(tmp_path)
    
    result = some_function(env.gateway, env.expected_goid)
    assert result is not None
```

### Before (CoverageEdgeEnv)

```python
from tests._helpers.coverage_env import CoverageEdgeEnv, create_coverage_edge_env

def test_something(tmp_path: Path) -> None:
    env = create_coverage_edge_env(tmp_path)
    
    result = compute_coverage(env.gateway, env.function_goid)
```

### After

```python
from tests._helpers.env_adapters import (
    CoverageEnvAdapter,
    CoverageEnvConfig,
    create_coverage_env_from_context,
)

def test_something(tmp_path: Path) -> None:
    # With defaults
    env = create_coverage_env_from_context(tmp_path)
    
    # Or with custom config
    config = CoverageEnvConfig(
        module_import="custom.module",
        function_name="custom_func",
    )
    env = create_coverage_env_from_context(tmp_path, config=config)
    
    result = compute_coverage(env.gateway, env.function_goid)
```

---

## Recipe E: Direct Insert Helpers to Seed Packs

### Before

```python
from tests._helpers.builders import (
    ModuleRow,
    GoidRow,
    insert_modules,
    insert_goids,
)

def test_something(fresh_gateway: StorageGateway) -> None:
    insert_modules(fresh_gateway, [
        ModuleRow(module="mod", path="mod.py", repo="repo", commit="commit"),
    ])
    insert_goids(fresh_gateway, [
        GoidRow(goid_h128=1, urn="urn:mod.func", ...),
    ])
```

### After (Using Seed Pack)

```python
def test_something(core_ctx: TestContext) -> None:
    # Standard test data already seeded
    # Access seeded values via constants
    from tests._helpers.seeds.core import GOID_FUNC_A, MOD_A_FQN
    
    assert core_ctx.query_count("core.modules") > 0
```

### After (Custom Data Still Needed)

```python
from tests._helpers.builders import ModuleRow, insert_modules
from tests._helpers.context import TestContext

def test_something(test_ctx: TestContext) -> None:
    # Apply standard seeds first
    test_ctx.require(CORE_PACK)
    
    # Add custom data on top
    insert_modules(test_ctx.gateway, [
        ModuleRow(
            module="custom.mod",
            path="custom/mod.py",
            repo=test_ctx.repo,
            commit=test_ctx.commit,
        ),
    ])
```

---

## Recipe F: Query Helpers

### Before

```python
def test_something(fresh_gateway: StorageGateway) -> None:
    # ... setup ...
    
    rows = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM core.modules WHERE repo = ? AND commit = ?",
        [repo, commit],
    ).fetchone()
    count = rows[0] if rows else 0
    assert count == 1
```

### After

```python
def test_something(core_ctx: TestContext) -> None:
    # Use query helpers
    count = core_ctx.query_count("core.modules")
    assert count > 0
    
    # Or for custom queries
    rows = core_ctx.query("SELECT module, path FROM core.modules")
    assert len(rows) > 0
    assert rows[0]["module"] is not None  # Dict-like access
    assert rows[0][0] is not None  # Index access
```

---

## Recipe G: TestScenario for Complex Setup

### Before

```python
def test_complex_scenario(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "mod.py").write_text("def func(): pass")
    
    gateway = open_ingestion_gateway(...)
    insert_modules(gateway, [...])
    insert_goids(gateway, [...])
    insert_call_graph_nodes(gateway, [...])
    # ... more setup ...
```

### After

```python
from tests._helpers.scenarios import TestScenario
from tests._helpers.seeds import CORE_PACK, GRAPH_PACK

def test_complex_scenario(tmp_path: Path) -> None:
    ctx = (
        TestScenario()
        .with_repo("test/repo")
        .with_commit("abc123")
        .with_seeds(CORE_PACK, GRAPH_PACK)
        .with_files({
            "mod.py": "def func(): pass",
            "util.py": "def helper(): pass",
        })
        .build(tmp_path)
    )
    
    with ctx:
        # All setup done, run test
        result = analyze(ctx.gateway, ctx.repo_root)
        assert result.success
```

---

## Validation Checklist

After migrating each file, verify:

```bash
# Quality checks (must all pass with zero errors)
uv run ruff check {file}
uv run pyright {file}
uv run pyrefly check {file}

# Tests pass
uv run pytest {file} -v
```

**Zero tolerance for:**
- pyright errors
- pyrefly errors  
- ruff errors
- `# type: ignore` comments
- `# noqa` suppressions

---

## Testing Charter Compliance

Migrated tests must comply with the Testing Charter in AGENTS.md:

1. **No monkeypatching** - Use DI and configuration instead
2. **Same stack, different instances** - Real DuckDB, isolated instances
3. **Realistic data** - Seed packs provide production-like data shapes
4. **Public entry points** - Test through public APIs
5. **Parallel-safe** - Use unique temp directories per test

---

## Getting Help

- **Seed pack details**: See `tests/_helpers/seeds/*.py`
- **Context API**: See `tests/_helpers/context.py`
- **Scenario builder**: See `tests/_helpers/scenarios.py`
- **Env adapters**: See `tests/_helpers/env_adapters.py`
- **Example tests**: See `tests/analytics/test_hexagonal_demo.py`

