# CLI Test Architecture Migration Plan

## Executive Summary

This plan outlines a comprehensive strategy for migrating CLI tests to align with the hexagonal test architecture used in `tests/analytics/` and `tests/graphs/`. The goal is to maximize test realness, reduce code duplication, and create a unified test architecture following the Testing Charter from AGENTS.md.

---

## Current State Analysis

### What Analytics/Graphs Tests Do Well (Patterns to Emulate)

1. **Unified Context System**
   - `TestContext` provides gateway, snapshot, build_paths in one object
   - `SeedPack` protocol enables composable data seeding
   - Contexts are created once and seeded incrementally

2. **Harness Pattern**
   - `PluginTestHarness` wraps TestContext with plugin execution methods
   - `AnalyticsPluginHarness` provides `execute_plugin()` convenience
   - `PluginHarnessFactory` offers pre-seeded harness variants

3. **Builder Pattern**
   - `ExecutionContextBuilder` fluently constructs execution contexts
   - `TestScenario` enables declarative test setup
   - Builders handle complex wiring internally

4. **Real Infrastructure**
   - Tests use real DuckDB gateways (in-memory for speed)
   - Real schema/macros applied
   - No mocking of core infrastructure

5. **Conftest Structure**
   - Domain-specific fixtures (`analytics_gateway`, `graph_plugin_context`)
   - Mock fixtures for optional test doubles (`mock_graph_runtime`)
   - Automatic cleanup with `Iterator` pattern

### CLI Test Issues

| Issue | Impact | Location |
|-------|--------|----------|
| Extensive `unittest.mock.patch` usage | Violates Testing Charter "no monkeypatching" | `test_graphs.py`, `test_storage.py`, `test_ops.py` |
| Duplicate `_build_test_context()` functions | Code duplication | Multiple handler test files |
| `MagicMock(spec=StorageGateway)` instead of real gateways | Misses real behavior | Handler tests |
| Autouse monkeypatching in conftest | Violates charter | `conftest.py` lines 197-245 |
| No seed pack integration | Tests create data manually | Throughout |
| Inconsistent context types | Confusion, duplication | `CLIContext` vs `CLIProjectContext` vs `CommandContext` |
| Test doubles don't match production protocols | Fakes may drift from real behavior | `_doubles/contexts.py` |

---

## Architecture Vision

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CLI Test Layer                                 │
├─────────────────────────────────────────────────────────────────────┤
│  CliHandlerHarness  │  CliOperationHarness  │  CliIntegrationHarness │
├─────────────────────────────────────────────────────────────────────┤
│                     CliTestContext                                  │
│         (wraps TestContext + CommandContext building)               │
├─────────────────────────────────────────────────────────────────────┤
│  CLI-Specific Seed Packs  │  Handler Context Builder                │
├─────────────────────────────────────────────────────────────────────┤
│                    Shared Test Helpers (_helpers/)                   │
│    TestContext  │  ExecutionContextBuilder  │  SeedPacks            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Foundation - CLI Test Context & Seed Packs

#### 1.1 Create CLI-Specific Seed Packs

**File:** `tests/_helpers/seeds/cli.py`

```python
"""CLI-specific seed packs for handler and operation tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers.seeds.core import CORE_PACK

if TYPE_CHECKING:
    from tests._helpers.context import TestContext


@dataclass(frozen=True)
class CliCorePack:
    """Seed pack for CLI core infrastructure testing.
    
    Seeds minimal data needed for CLI handlers to function:
    - Core tables (modules, functions, goids)
    - Snapshot metadata
    """
    
    name: str = "cli_core"
    dependencies: tuple = (CORE_PACK,)
    
    def apply(self, ctx: TestContext) -> None:
        """Apply CLI core seeds to context."""
        # Core pack already handles base tables
        # Add CLI-specific metadata if needed
        pass


@dataclass(frozen=True)
class OperationRegistryPack:
    """Seed pack for operation registry testing.
    
    Seeds sample operations for op list/describe handlers.
    """
    
    name: str = "operation_registry"
    dependencies: tuple = ()
    
    def apply(self, ctx: TestContext) -> None:
        """Seed operation metadata (if persisted)."""
        # Operations are typically in-memory, but if CLI needs
        # persisted op metadata, seed it here
        pass


@dataclass(frozen=True)
class StorageProfilePack:
    """Seed pack for storage handler testing.
    
    Seeds macro metadata and storage profile data.
    """
    
    name: str = "storage_profile"
    dependencies: tuple = (CORE_PACK,)
    
    def apply(self, ctx: TestContext) -> None:
        """Seed storage profile data."""
        # Seed ingest.datasets, macro metadata, etc.
        pass


CLI_CORE_PACK = CliCorePack()
OPERATION_REGISTRY_PACK = OperationRegistryPack()
STORAGE_PROFILE_PACK = StorageProfilePack()
```

#### 1.2 Create CliTestContext

**File:** `tests/_helpers/cli_context.py`

```python
"""Unified CLI test context integrating TestContext with CommandContext building."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Self

from codeintel.cli.context import CommandContext, CommandContextBuilder
from tests._helpers.context import SeedPack, TestContext, create_test_context
from tests._helpers.repo import write_canonical_repo

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


@dataclass
class CliTestContext:
    """Unified context for CLI handler and operation tests.
    
    Wraps TestContext and provides CommandContext building methods.
    Enables seed pack composition while offering CLI-specific conveniences.
    
    Attributes
    ----------
    test_ctx : TestContext
        Underlying test context with gateway, snapshot, and query methods.
    operation_id : str
        Default operation ID for command contexts.
    """
    
    __test__ = False  # Prevent pytest collection
    
    test_ctx: TestContext
    operation_id: str = "cli.test"
    _command_contexts: list[CommandContext] = field(default_factory=list)
    
    @property
    def gateway(self) -> StorageGateway:
        """Return the underlying storage gateway."""
        return self.test_ctx.gateway
    
    @property
    def repo(self) -> str:
        """Return repository identifier."""
        return self.test_ctx.repo
    
    @property
    def commit(self) -> str:
        """Return commit identifier."""
        return self.test_ctx.commit
    
    @property
    def repo_root(self) -> Path:
        """Return repository root path."""
        return self.test_ctx.repo_root
    
    @property
    def build_dir(self) -> Path:
        """Return build directory path."""
        return self.test_ctx.build_dir
    
    def require(self, *seed_packs: SeedPack) -> Self:
        """Apply seed packs to the underlying test context.
        
        Returns
        -------
        Self
            Self for method chaining.
        """
        self.test_ctx.require(*seed_packs)
        return self
    
    def query_count(self, table: str, where: str | None = None) -> int:
        """Count rows in a table."""
        return self.test_ctx.query_count(table, where)
    
    @contextmanager
    def command_context(
        self,
        params: dict[str, object] | None = None,
        *,
        operation_id: str | None = None,
    ) -> Iterator[CommandContext]:
        """Create a CommandContext backed by the real gateway.
        
        Parameters
        ----------
        params
            Handler parameters.
        operation_id
            Operation identifier (defaults to self.operation_id).
        
        Yields
        ------
        CommandContext
            Configured command context.
        """
        builder = (
            CommandContextBuilder()
            .with_params(params or {})
            .with_operation_id(operation_id or self.operation_id)
            .with_injected_gateway(self.test_ctx.gateway)
        )
        with builder.build() as ctx:
            yield ctx
    
    def build_command_context(
        self,
        params: dict[str, object] | None = None,
        *,
        operation_id: str | None = None,
    ) -> CommandContext:
        """Build a CommandContext (caller manages lifecycle).
        
        For most tests, prefer the context manager `command_context()`.
        """
        builder = (
            CommandContextBuilder()
            .with_params(params or {})
            .with_operation_id(operation_id or self.operation_id)
            .with_injected_gateway(self.test_ctx.gateway)
        )
        ctx_mgr = builder.build()
        ctx = ctx_mgr.__enter__()
        self._command_contexts.append(ctx)
        return ctx
    
    def close(self) -> None:
        """Close all resources."""
        self.test_ctx.close()


def create_cli_test_context(
    tmp_path: Path,
    *,
    operation_id: str = "cli.test",
    with_repo_files: bool = False,
) -> CliTestContext:
    """Create a CliTestContext with real gateway.
    
    Parameters
    ----------
    tmp_path
        Temporary directory for test isolation.
    operation_id
        Default operation ID for command contexts.
    with_repo_files
        If True, write canonical Python files to repo_root.
    
    Returns
    -------
    CliTestContext
        Configured CLI test context.
    """
    test_ctx = create_test_context(tmp_path)
    if with_repo_files:
        write_canonical_repo(test_ctx.repo_root)
    return CliTestContext(test_ctx=test_ctx, operation_id=operation_id)
```

#### 1.3 Create CLI Handler Harness

**File:** `tests/_helpers/harnesses/cli.py`

```python
"""CLI handler harness for production-parity testing."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.context import CommandContext
from codeintel.cli.core import HandlerResult
from tests._helpers.cli_context import CliTestContext, create_cli_test_context
from tests._helpers.context import SeedPack
from tests._helpers.seeds import CORE_PACK

if TYPE_CHECKING:
    from collections.abc import Callable
    
    Handler = Callable[[CommandContext], HandlerResult]


@dataclass
class CliHandlerHarness:
    """Harness for testing CLI handlers with real infrastructure.
    
    Provides a fluent interface for:
    - Creating command contexts with params
    - Executing handlers
    - Asserting on results and database state
    """
    
    ctx: CliTestContext
    
    @property
    def gateway(self):
        """Access the underlying gateway for assertions."""
        return self.ctx.gateway
    
    def with_params(self, params: dict[str, object]) -> CommandContext:
        """Create a command context with the given params.
        
        Returns
        -------
        CommandContext
            Context configured for handler execution.
        """
        return self.ctx.build_command_context(params)
    
    def execute(
        self,
        handler: Handler,
        params: dict[str, object] | None = None,
    ) -> HandlerResult:
        """Execute a handler with the given parameters.
        
        Parameters
        ----------
        handler
            Handler function to execute.
        params
            Handler parameters.
        
        Returns
        -------
        HandlerResult
            Result of handler execution.
        """
        with self.ctx.command_context(params) as cmd_ctx:
            return handler(cmd_ctx)
    
    def close(self) -> None:
        """Close underlying resources."""
        self.ctx.close()


@contextmanager
def cli_handler_harness(
    tmp_path: Path,
    *packs: SeedPack,
) -> Iterator[CliHandlerHarness]:
    """Create a CLI handler harness with seed packs applied.
    
    Parameters
    ----------
    tmp_path
        Pytest temporary path.
    packs
        Seed packs to apply (defaults to CORE_PACK).
    
    Yields
    ------
    CliHandlerHarness
        Configured harness with cleanup.
    """
    ctx = create_cli_test_context(tmp_path)
    packs_to_apply = packs or (CORE_PACK,)
    ctx.require(*packs_to_apply)
    harness = CliHandlerHarness(ctx=ctx)
    try:
        yield harness
    finally:
        harness.close()


@contextmanager
def storage_handler_harness(tmp_path: Path) -> Iterator[CliHandlerHarness]:
    """Harness pre-seeded for storage handler testing."""
    from tests._helpers.seeds.cli import STORAGE_PROFILE_PACK
    
    with cli_handler_harness(tmp_path, CORE_PACK, STORAGE_PROFILE_PACK) as harness:
        yield harness


@contextmanager
def graph_handler_harness(tmp_path: Path) -> Iterator[CliHandlerHarness]:
    """Harness pre-seeded for graph handler testing."""
    from tests._helpers.seeds import GRAPH_PACK
    
    with cli_handler_harness(tmp_path, CORE_PACK, GRAPH_PACK) as harness:
        yield harness
```

---

### Phase 2: Migrate Handler Tests

#### 2.1 Pattern for Handler Test Migration

**Before (Current Pattern with Mocking):**

```python
from unittest.mock import MagicMock, patch

def test_graph_plugins_list_handler_success(mock_list_plugins: MagicMock) -> None:
    mock_plugin = MagicMock()
    mock_plugin.metadata.name = "test_plugin"
    # ... more mock setup
    
    mock_list_plugins.return_value = [mock_plugin]
    ctx = _make_mock_context({})
    
    result = graph_plugins_list_handler(ctx)
    # assertions
```

**After (Production-Parity Pattern):**

```python
from tests._helpers.harnesses.cli import graph_handler_harness

def test_graph_plugins_list_handler_success(tmp_path: Path) -> None:
    with graph_handler_harness(tmp_path) as harness:
        # Real plugins from registry, no mocking
        result = harness.execute(graph_plugins_list_handler, params={})
        
        expect_true(result.success)
        # Assertions against real data
```

#### 2.2 Handler Test Migration Order

**Priority 1: Most Mocking, Most Value**
1. `tests/cli/handlers/test_graphs.py` - Heavy patching
2. `tests/cli/handlers/test_storage.py` - Gateway mocking
3. `tests/cli/handlers/test_ops.py` - Operation registry mocking

**Priority 2: Moderate Complexity**
4. `tests/cli/handlers/test_datasets.py`
5. `tests/cli/handlers/test_subsystem.py`
6. `tests/cli/handlers/test_docs.py`

**Priority 3: Lower Complexity**
7. `tests/cli/handlers/test_ide.py`
8. `tests/cli/handlers/test_deprecation_warnings.py`

#### 2.3 Migration Template

For each handler test file:

1. **Remove mock imports**
   ```diff
   - from unittest.mock import MagicMock, patch
   + from tests._helpers.harnesses.cli import cli_handler_harness
   ```

2. **Replace `_build_test_context` with harness**
   ```diff
   - def _build_test_context(params: dict[str, object]) -> CommandContext:
   -     mock_gateway = MagicMock(spec=StorageGateway)
   -     builder = CommandContextBuilder()...
   -     return ctx_manager.__enter__()
   
   # Use harness in test instead
   ```

3. **Replace patched tests with real-data tests**
   ```diff
   - @patch("codeintel.cli.handlers.graphs.list_graph_plugins")
   - def test_handler(mock_list_plugins: MagicMock) -> None:
   -     mock_list_plugins.return_value = [mock_plugin]
   + def test_handler(tmp_path: Path) -> None:
   +     with graph_handler_harness(tmp_path) as harness:
   +         result = harness.execute(handler, params={})
   ```

---

### Phase 3: Update CLI Conftest

#### 3.1 Remove Monkeypatching Fixtures

**Current (Violates Charter):**
```python
@pytest.fixture(autouse=True)
def _disable_contract_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(...)

@pytest.fixture(autouse=True) 
def _track_and_close_gateways(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(...)
```

**Proposed (Charter-Compliant):**
```python
# Remove monkeypatching fixtures entirely
# Use dependency injection via CommandContextBuilder.with_injected_gateway()

@pytest.fixture
def cli_handler_ctx(tmp_path: Path) -> Iterator[CliTestContext]:
    """Provide CLI test context with real gateway and cleanup."""
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK)
    try:
        yield ctx
    finally:
        ctx.close()
```

#### 3.2 New Fixture Structure

**File:** `tests/cli/conftest.py` (rewritten)

```python
"""CLI test fixtures following the Testing Charter.

This module provides fixtures for CLI testing with production parity:
- Real gateways (in-memory for speed)
- No monkeypatching
- Composable seed packs
- Automatic cleanup
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests._helpers.cli_context import CliTestContext, create_cli_test_context
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.harnesses.cli import CliHandlerHarness, cli_handler_harness
from tests._helpers.seeds import CORE_PACK, GRAPH_PACK
from tests.cli._harness import CliTestHarness, GoldenFileAssertion, OperationTestHarness


# =============================================================================
# Core CLI Fixtures
# =============================================================================


@pytest.fixture
def cli_test_ctx(tmp_path: Path) -> Iterator[CliTestContext]:
    """Provide CLI test context with real gateway.
    
    Yields
    ------
    CliTestContext
        Context with gateway, snapshot, and command context building.
    """
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def cli_handler_harness_fixture(tmp_path: Path) -> Iterator[CliHandlerHarness]:
    """Provide handler harness with core seeds.
    
    Yields
    ------
    CliHandlerHarness
        Harness for handler execution and assertions.
    """
    with cli_handler_harness(tmp_path, CORE_PACK) as harness:
        yield harness


@pytest.fixture
def graph_cli_ctx(tmp_path: Path) -> Iterator[CliTestContext]:
    """Provide CLI context with graph seeds applied.
    
    Yields
    ------
    CliTestContext
        Context ready for graph handler tests.
    """
    ctx = create_cli_test_context(tmp_path)
    ctx.require(CORE_PACK, GRAPH_PACK)
    try:
        yield ctx
    finally:
        ctx.close()


# =============================================================================
# Harness Fixtures (Existing, Retained)
# =============================================================================


@pytest.fixture
def cli() -> CliTestHarness:
    """Provide CLI test harness for full CLI invocation."""
    return CliTestHarness()


@pytest.fixture
def golden(request: pytest.FixtureRequest) -> GoldenFileAssertion:
    """Provide golden file assertion helper."""
    import os
    test_dir = request.path.parent
    golden_dir = test_dir / "_golden"
    update_mode = os.environ.get("UPDATE_GOLDEN", "").lower() in {"1", "true"}
    return GoldenFileAssertion(golden_dir=golden_dir, update_mode=update_mode)


@pytest.fixture
def op_harness() -> OperationTestHarness:
    """Provide operation test harness."""
    return OperationTestHarness(render=False)
```

---

### Phase 4: Create CLI Assertion Helpers

**File:** `tests/_helpers/assertions/cli.py`

```python
"""CLI-specific assertion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.cli.core import HandlerResult


def expect_handler_success(result: HandlerResult) -> None:
    """Assert handler returned success."""
    if not result.success:
        error_detail = result.error.detail if result.error else "Unknown error"
        msg = f"Expected handler success, got error: {error_detail}"
        raise AssertionError(msg)


def expect_handler_error(
    result: HandlerResult,
    *,
    error_type: str | None = None,
    status: int | None = None,
) -> None:
    """Assert handler returned expected error."""
    if result.success:
        msg = "Expected handler failure, but it succeeded"
        raise AssertionError(msg)
    
    if error_type and result.error:
        if result.error.type != error_type:
            msg = f"Expected error type {error_type}, got {result.error.type}"
            raise AssertionError(msg)
    
    if status and result.error:
        if result.error.status != status:
            msg = f"Expected status {status}, got {result.error.status}"
            raise AssertionError(msg)


def expect_handler_data_count(result: HandlerResult, key: str, expected: int) -> None:
    """Assert handler data contains expected count."""
    expect_handler_success(result)
    if result.data is None:
        msg = "Expected handler data, got None"
        raise AssertionError(msg)
    
    data_dict = result.data.to_dict() if hasattr(result.data, "to_dict") else {}
    actual = data_dict.get(key)
    if actual != expected:
        msg = f"Expected {key}={expected}, got {actual}"
        raise AssertionError(msg)
```

---

### Phase 5: Integration with Existing _helpers

#### 5.1 Update `tests/_helpers/__init__.py`

Add exports for new CLI helpers:

```python
from tests._helpers.cli_context import CliTestContext, create_cli_test_context
from tests._helpers.harnesses.cli import (
    CliHandlerHarness,
    cli_handler_harness,
    graph_handler_harness,
    storage_handler_harness,
)
from tests._helpers.seeds.cli import (
    CLI_CORE_PACK,
    OPERATION_REGISTRY_PACK,
    STORAGE_PROFILE_PACK,
)

__all__ = [
    # ... existing exports ...
    "CLI_CORE_PACK",
    "CliHandlerHarness",
    "CliTestContext",
    "OPERATION_REGISTRY_PACK",
    "STORAGE_PROFILE_PACK",
    "cli_handler_harness",
    "create_cli_test_context",
    "graph_handler_harness",
    "storage_handler_harness",
]
```

#### 5.2 Update `tests/_helpers/harnesses/__init__.py`

```python
from tests._helpers.harnesses.cli import (
    CliHandlerHarness,
    cli_handler_harness,
    graph_handler_harness,
    storage_handler_harness,
)

__all__ = [
    # ... existing exports ...
    "CliHandlerHarness",
    "cli_handler_harness",
    "graph_handler_harness",
    "storage_handler_harness",
]
```

---

## Migration Checklist

### Phase 1: Foundation
- [ ] Create `tests/_helpers/seeds/cli.py` with CLI-specific seed packs
- [ ] Create `tests/_helpers/cli_context.py` with `CliTestContext`
- [ ] Create `tests/_helpers/harnesses/cli.py` with `CliHandlerHarness`
- [ ] Create `tests/_helpers/assertions/cli.py` with CLI assertions
- [ ] Update `tests/_helpers/__init__.py` exports
- [ ] Update `tests/_helpers/harnesses/__init__.py` exports
- [ ] Update `tests/_helpers/seeds/__init__.py` exports

### Phase 2: Handler Test Migration
- [ ] Migrate `tests/cli/handlers/test_graphs.py`
- [ ] Migrate `tests/cli/handlers/test_storage.py`
- [ ] Migrate `tests/cli/handlers/test_ops.py`
- [ ] Migrate `tests/cli/handlers/test_datasets.py`
- [ ] Migrate `tests/cli/handlers/test_subsystem.py`
- [ ] Migrate `tests/cli/handlers/test_docs.py`
- [ ] Migrate `tests/cli/handlers/test_ide.py`
- [ ] Migrate `tests/cli/handlers/conftest.py` (remove FakeGraphEngine etc.)

### Phase 3: Conftest Cleanup
- [ ] Remove `_disable_contract_validation` autouse fixture
- [ ] Remove `_track_and_close_gateways` monkeypatching
- [ ] Replace with proper gateway lifecycle fixtures
- [ ] Add new harness-based fixtures
- [ ] Remove deprecated fixture exports from `__all__`

### Phase 4: Cleanup Legacy Doubles
- [ ] Evaluate `tests/cli/_doubles/` - remove or refactor
- [ ] Ensure all doubles implement production protocols
- [ ] Move useful doubles to `tests/_helpers/fakes/` if reusable

### Phase 5: Verification
- [ ] Run `uv run pytest tests/cli/ -q` - all tests pass
- [ ] Run `uv run ruff check tests/cli/` - no violations
- [ ] Run `uv run pyright tests/cli/` - no type errors
- [ ] Verify no `unittest.mock.patch` in handler tests
- [ ] Verify no `monkeypatch` in conftest autouse fixtures

---

## Example Migrated Test

### Before

```python
# tests/cli/handlers/test_ops.py (current)
from unittest.mock import MagicMock, patch

def test_op_list_handler_returns_ok() -> None:
    mock_op = MagicMock()
    mock_op.id = "test-op"
    mock_op.category = "test"
    mock_op.summary = "Test operation"
    
    ctx = _build_test_context(params={})
    
    with patch("codeintel.cli.handlers.ops.iter_operations", return_value=[mock_op]):
        result = op_list_handler(ctx)
    
    expect_true(result.success)
```

### After

```python
# tests/cli/handlers/test_ops.py (migrated)
from pathlib import Path

from codeintel.cli.handlers.ops import op_list_handler
from tests._helpers.assertions.cli import expect_handler_success
from tests._helpers.harnesses.cli import cli_handler_harness
from tests._helpers.seeds import CORE_PACK

def test_op_list_handler_returns_ok(tmp_path: Path) -> None:
    """Handler returns success with real operation registry."""
    with cli_handler_harness(tmp_path, CORE_PACK) as harness:
        result = harness.execute(op_list_handler, params={})
        
        expect_handler_success(result)
        # Real operations from registry, no mocking needed
        # If we need specific operations, use a seed pack
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing tests during migration | Migrate one file at a time, run full test suite after each |
| Performance regression (real vs mocked) | Use in-memory DuckDB, share gateways across tests in same file |
| Missing test coverage after removing mocks | Ensure seed packs provide equivalent data scenarios |
| Complex handler dependencies | Create specialized seed packs (e.g., `GRAPH_HANDLER_PACK`) |

---

## Success Criteria

1. **Zero `unittest.mock.patch` in handler tests** - All handlers tested with real infrastructure
2. **Zero autouse monkeypatching in CLI conftest** - Proper lifecycle management
3. **100% test pass rate** - All existing test cases continue to pass
4. **Consistent patterns with analytics/graphs** - CLI tests follow same architecture
5. **Reduced code duplication** - No more per-file `_build_test_context` functions
6. **Type-clean** - Passes pyright strict and pyrefly

---

## Timeline Estimate

| Phase | Effort | Duration |
|-------|--------|----------|
| Phase 1: Foundation | Medium | 1-2 days |
| Phase 2: Handler Migration | High | 3-4 days |
| Phase 3: Conftest Cleanup | Medium | 1 day |
| Phase 4: Legacy Cleanup | Low | 0.5 days |
| Phase 5: Verification | Low | 0.5 days |
| **Total** | | **6-8 days** |

---

## Appendix: Files to Create

1. `tests/_helpers/seeds/cli.py` - CLI-specific seed packs
2. `tests/_helpers/cli_context.py` - CliTestContext class
3. `tests/_helpers/harnesses/cli.py` - CliHandlerHarness and factories
4. `tests/_helpers/assertions/cli.py` - CLI assertion helpers

## Appendix: Files to Modify

1. `tests/_helpers/__init__.py` - Add CLI exports
2. `tests/_helpers/harnesses/__init__.py` - Add CLI harness exports
3. `tests/_helpers/seeds/__init__.py` - Add CLI pack exports
4. `tests/cli/conftest.py` - Rewrite without monkeypatching
5. `tests/cli/handlers/conftest.py` - Remove FakeGraphEngine, use harnesses
6. `tests/cli/handlers/test_*.py` - Migrate each handler test file

## Appendix: Files to Evaluate for Removal

1. `tests/cli/_doubles/` - May be superseded by `tests/_helpers/fakes/`
2. Duplicate `_build_test_context` functions in handler tests
