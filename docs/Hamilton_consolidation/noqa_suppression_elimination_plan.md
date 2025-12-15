# NOQA Suppression Elimination Plan

## Executive Summary

This document provides a root cause analysis of all 62 `noqa` suppressions found in the CodeIntel codebase and proposes structural design changes to eliminate them entirely. The goal is not merely to silence linter warnings, but to implement best-in-class design patterns that enhance functionality, robustness, extensibility, and maintainability.

**Current State**: 62 suppressions across 7 rule categories  
**Target State**: Zero suppressions through architectural remediation

---

## Table of Contents

1. [Root Cause Analysis by Rule Type](#root-cause-analysis-by-rule-type)
2. [Architectural Design Proposals](#architectural-design-proposals)
3. [Implementation Plan](#implementation-plan)
4. [Migration Strategy](#migration-strategy)
5. [Success Criteria](#success-criteria)

---

## Root Cause Analysis by Rule Type

### 1. PLC0415 — Import Outside Top-Level (44 occurrences)

#### Root Cause

The codebase has **circular import dependencies** between modules. To work around these, developers use deferred imports inside functions. This is a symptom of:

1. **Tightly coupled modules** — Contract providers, registries, and schema modules depend on each other
2. **God modules** — Large modules that aggregate too much functionality
3. **Lack of dependency inversion** — Concrete implementations are imported directly instead of through abstractions
4. **Missing interface layers** — No clean separation between API contracts and implementations

#### Affected Modules

| Module | Import Pattern | Dependency Issue |
|--------|---------------|------------------|
| `contract_provider.py` | Imports `build.registry`, `row_registry`, `contracts`, `composites` | Bidirectional dependency with schemas |
| `registrations.py` | Imports plugins and registry targets | Plugin registration requires targets that import registrations |
| `provider_unified.py` | Imports `build.registry`, `provider_hamilton` | Schema provider needs registry, registry needs schemas |
| `schemas/__init__.py` | Dynamic import via `importlib` | Late binding for optional components |

#### Design Smell

The current architecture violates the **Dependency Inversion Principle (DIP)**:
- High-level modules (schemas, contracts) depend on low-level modules (registrations)
- Low-level modules depend on high-level modules (circular)

---

### 2. PLW0603 — Global Statement (6 occurrences)

#### Root Cause

The codebase uses **module-level mutable singletons** with explicit `global` statements for lazy initialization:

```python
_deferred_registry_module: ModuleType | None = None

def _registry_module() -> ModuleType:
    global _deferred_registry_module  # noqa: PLW0603
    if _deferred_registry_module is None:
        import codeintel.build.registry as mod
        _deferred_registry_module = mod
    return _deferred_registry_module
```

This pattern exists because:

1. **Lazy initialization requirement** — Modules must not import heavy dependencies at load time
2. **Caching requirement** — Repeated calls should return cached instances
3. **Thread safety not considered** — No synchronization for concurrent access
4. **Testing difficulty** — Global state is hard to reset between tests

#### Affected Modules

| Module | Global Variables |
|--------|-----------------|
| `contract_provider.py` | `_deferred_registry_module`, `_deferred_row_registry_module`, `_deferred_contracts_module`, `_deferred_composites_module`, `_contract_provider_instance` |
| `unified_registry.py` | `_UNIFIED_REGISTRY` |

#### Design Smell

The pattern conflates three concerns:
1. Lazy instantiation
2. Caching/memoization
3. Singleton lifecycle management

---

### 3. BLE001 — Blind Except (4 occurrences)

#### Root Cause

Functions catch bare `Exception` to provide graceful degradation when optional components fail:

```python
try:
    from codeintel.build.schemas.json_schema_registry import get_json_schema_for_dataset_name
    return get_json_schema_for_dataset_name(schema_name)
except Exception:  # noqa: BLE001
    return None
```

This pattern exists because:

1. **Defensive programming** — Optional features should not crash the system
2. **Unknown failure modes** — Import errors, missing schemas, configuration issues
3. **No structured error taxonomy** — Different error types not distinguished
4. **Silent failures** — Errors are swallowed without proper logging or metrics

#### Affected Locations

| File | Context | Risk |
|------|---------|------|
| `exports/validation.py:44` | Schema lookup failure | Medium — may hide real bugs |
| `storage/validation/conformance.py:73` | Schema conformance check | Medium — validation bypassed |
| `exports/common.py:290` | JSON schema digest computation | Low — non-critical metadata |
| `hamilton/native/executor.py:207` | Target execution | High — any error captured |

#### Design Smell

The broad exception catching violates **explicit error handling** principles:
- Catches `KeyboardInterrupt`, `SystemExit`, `MemoryError` unintentionally
- No distinction between recoverable and fatal errors
- No structured error reporting

---

### 4. PLR0911 — Too Many Return Statements (2 occurrences)

#### Root Cause

Type coercion functions have many early-return branches for different input types:

```python
def optional_int(value: object | None) -> int | None:  # noqa: PLR0911
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip()) if value.strip() else None
        except ValueError:
            return None
    return _int_from_object(value)
```

This pattern exists because:

1. **Type dispatch** — Each input type requires different handling
2. **Fail-fast design** — Early returns simplify control flow
3. **No polymorphism** — Runtime type checking instead of method dispatch

#### Affected Functions

| Function | Return Statements | Purpose |
|----------|------------------|---------|
| `optional_int()` | 6 | Convert any object to int or None |
| `optional_float()` | 6 | Convert any object to float or None |

#### Design Smell

The functions are essentially **pattern matching** on types, which Python handles poorly without structural dispatch mechanisms.

---

### 5. PLC2801 — Unnecessary Dunder Call (2 occurrences)

#### Root Cause

Tests verify frozen dataclass immutability by attempting to mutate via `object.__setattr__`:

```python
with pytest.raises(AttributeError):
    object.__setattr__(cast("Any", meta), "name", "changed")  # noqa: PLC2801
```

This pattern exists because:

1. **Immutability testing** — Need to verify `frozen=True` enforcement
2. **Bypassing Python's attribute setting** — Direct dunder call bypasses descriptor protocol
3. **No dedicated immutability test utility** — Tests inline this pattern

#### Affected Locations

| File | Test Purpose |
|------|--------------|
| `test_hamilton_phase0.py:138` | Verify `CanonicalPluginMeta` immutability |
| `test_hamilton_phase1.py:292` | Verify `DatasetRef` immutability |

#### Design Smell

Testing immutability is a valid concern, but the implementation:
- Uses cast to `Any` to bypass type checking
- Directly invokes magic methods

---

### 6. PLR2004 — Magic Value Comparison (1 occurrence)

#### Root Cause

Code compares against a well-known constant (SHA-256 digest length) without naming it:

```python
if len(digest) != 64:  # noqa: PLR2004
```

This is a minor issue, but indicates:
1. **Missing named constants** — Magic numbers embedded in logic
2. **Domain knowledge not captured** — SHA-256 → 64 hex chars not documented

---

### 7. PLR0913 — Too Many Arguments (1 occurrence)

#### Root Cause

The `create_run_record()` function accepts many parameters:

```python
def create_run_record(  # noqa: PLR0913
    target: OutputTarget,
    status: Literal["succeeded", "skipped", "failed"],
    input_hash: str,
    *,
    env: BuildEnv | None = None,
    run: NativeRunInfo | None = None,
    error: Exception | None = None,
) -> TargetRunRecord:
```

This exists because:
1. **Factory function** — Creates different record types based on status
2. **Conditional parameters** — Some params required for some statuses
3. **No builder pattern** — All configuration via function arguments

---

## Architectural Design Proposals

### Proposal 1: Dependency Injection Container

**Eliminates**: PLC0415 (44 imports), PLW0603 (6 globals)

#### Problem Statement

The current architecture has bidirectional dependencies between:
- `build.schemas` ↔ `build.registry`
- `config.datasets.contracts` ↔ `build.schemas`
- `build.registrations` ↔ `build.plugins`

These are resolved through deferred imports and global singletons.

#### Solution: Protocol-Based Dependency Injection

Introduce a **dependency injection container** that:
1. Defines **protocol interfaces** for all cross-cutting concerns
2. **Lazily resolves** implementations on first access
3. Uses **`functools.lru_cache`** instead of mutable globals
4. Provides **test fixtures** for dependency override

##### New Module: `codeintel.core.di`

```python
"""Dependency injection protocols and container.

This module defines protocol interfaces for cross-cutting concerns and provides
a lazy-resolving container that eliminates circular imports.

Example
-------
>>> from codeintel.core.di import Container, get_container
>>> container = get_container()
>>> registry = container.target_registry()
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.contract_primitives import DatasetContract

__all__ = [
    "Container",
    "SchemaProviderProtocol",
    "TargetRegistryProtocol",
    "ContractProviderProtocol",
    "get_container",
    "reset_container",
]


@runtime_checkable
class SchemaProviderProtocol(Protocol):
    """Protocol for schema lookup operations."""

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        """Get schema for a table key."""
        ...

    def require_table_schema(self, table_key: str) -> TableSchema:
        """Get schema or raise KeyError."""
        ...

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        """Iterate all known schemas."""
        ...


@runtime_checkable
class TargetRegistryProtocol(Protocol):
    """Protocol for target registration and lookup."""

    def get_target(self, name: str) -> object:
        """Get target by name."""
        ...

    def __contains__(self, name: str) -> bool:
        """Check if target exists."""
        ...

    def __iter__(self) -> Iterable[str]:
        """Iterate target names."""
        ...


@runtime_checkable
class ContractProviderProtocol(Protocol):
    """Protocol for dataset contract lookup."""

    def get_contract(self, table_key: str) -> DatasetContract:
        """Get contract for a table key."""
        ...

    def iter_contracts(self) -> Iterable[DatasetContract]:
        """Iterate all known contracts."""
        ...


class Container:
    """Lazy dependency injection container.
    
    Uses lru_cache for automatic memoization without mutable global state.
    Each resolver function is cached independently.
    
    Examples
    --------
    >>> container = Container()
    >>> schema_provider = container.schema_provider()
    >>> target_registry = container.target_registry()
    """

    @lru_cache(maxsize=1)
    def schema_provider(self) -> SchemaProviderProtocol:
        """Get the schema provider implementation.
        
        Returns
        -------
        SchemaProviderProtocol
            Lazy-loaded schema provider.
        """
        from codeintel.build.schemas.provider_unified import UnifiedSchemaProvider
        return UnifiedSchemaProvider()

    @lru_cache(maxsize=1)
    def target_registry(self) -> TargetRegistryProtocol:
        """Get the target registry implementation.
        
        Returns
        -------
        TargetRegistryProtocol
            Lazy-loaded unified registry.
        """
        from codeintel.build.unified_registry import UnifiedRegistry
        from codeintel.build.registrations import register_all_targets
        
        registry = UnifiedRegistry()
        register_all_targets(registry)
        return registry

    @lru_cache(maxsize=1)
    def contract_provider(self) -> ContractProviderProtocol:
        """Get the contract provider implementation.
        
        Returns
        -------
        ContractProviderProtocol
            Lazy-loaded contract provider.
        """
        from codeintel.build.schemas.contract_provider import ContractProvider
        return ContractProvider()

    def clear_caches(self) -> None:
        """Clear all cached instances (for testing)."""
        self.schema_provider.cache_clear()
        self.target_registry.cache_clear()
        self.contract_provider.cache_clear()


# Module-level container instance (immutable reference)
_CONTAINER: Container = Container()


def get_container() -> Container:
    """Get the singleton container instance.
    
    Returns
    -------
    Container
        The dependency injection container.
    """
    return _CONTAINER


def reset_container() -> None:
    """Reset the container (for testing).
    
    Clears all cached dependencies so they will be re-resolved on next access.
    """
    _CONTAINER.clear_caches()
```

##### Benefits

1. **No `global` statements** — Uses `lru_cache` on instance methods
2. **No deferred imports at call sites** — Imports happen inside container methods
3. **Protocol-based** — Consumers depend on protocols, not implementations
4. **Testable** — `reset_container()` enables clean test isolation
5. **Type-safe** — Protocols are runtime-checkable

##### Migration Path

Replace all occurrences of:
```python
global _deferred_registry_module  # noqa: PLW0603
if _deferred_registry_module is None:
    import codeintel.build.registry as mod  # noqa: PLC0415
    _deferred_registry_module = mod
return _deferred_registry_module
```

With:
```python
from codeintel.core.di import get_container

def _get_registry() -> TargetRegistryProtocol:
    return get_container().target_registry()
```

---

### Proposal 2: Structured Error Handling

**Eliminates**: BLE001 (4 blind excepts)

#### Problem Statement

Functions catch bare `Exception` because:
1. They don't know what specific errors might occur
2. They want to provide fallback behavior
3. They don't log or report the error type

#### Solution: Error Domain Types + Result Pattern

##### New Module: `codeintel.core.errors.schema_errors`

```python
"""Schema-specific error types for explicit exception handling.

This module defines a hierarchy of exceptions for schema operations,
enabling precise error handling without blind excepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.core.schemas.primitives import TableSchema

__all__ = [
    "SchemaError",
    "SchemaNotFoundError",
    "SchemaLoadError", 
    "SchemaValidationError",
    "SchemaDigestError",
]


class SchemaError(Exception):
    """Base exception for all schema-related errors."""

    def __init__(self, message: str, table_key: str | None = None) -> None:
        """Initialize schema error.
        
        Parameters
        ----------
        message
            Human-readable error description.
        table_key
            Optional table key associated with the error.
        """
        super().__init__(message)
        self.table_key = table_key


class SchemaNotFoundError(SchemaError):
    """Raised when a schema is not found for a table key."""

    def __init__(self, table_key: str) -> None:
        """Initialize with the missing table key.
        
        Parameters
        ----------
        table_key
            The table key that was not found.
        """
        super().__init__(f"No schema found for table key: {table_key}", table_key)


class SchemaLoadError(SchemaError):
    """Raised when a schema cannot be loaded from its source."""

    def __init__(self, table_key: str, cause: Exception) -> None:
        """Initialize with the table key and underlying cause.
        
        Parameters
        ----------
        table_key
            The table key that failed to load.
        cause
            The underlying exception that caused the load failure.
        """
        super().__init__(f"Failed to load schema for {table_key}: {cause}", table_key)
        self.__cause__ = cause


class SchemaValidationError(SchemaError):
    """Raised when schema validation fails."""

    def __init__(self, table_key: str, violations: list[str]) -> None:
        """Initialize with validation violations.
        
        Parameters
        ----------
        table_key
            The table key that failed validation.
        violations
            List of validation error messages.
        """
        msg = f"Schema validation failed for {table_key}: {'; '.join(violations)}"
        super().__init__(msg, table_key)
        self.violations = violations


class SchemaDigestError(SchemaError):
    """Raised when schema digest computation fails."""

    def __init__(self, table_key: str, cause: Exception) -> None:
        """Initialize with the table key and underlying cause.
        
        Parameters
        ----------
        table_key
            The table key for which digest computation failed.
        cause
            The underlying exception.
        """
        super().__init__(f"Failed to compute digest for {table_key}: {cause}", table_key)
        self.__cause__ = cause
```

##### Refactored Usage

**Before**:
```python
try:
    from codeintel.build.schemas.json_schema_registry import get_json_schema_for_dataset_name
    return get_json_schema_for_dataset_name(schema_name)
except Exception:  # noqa: BLE001
    return None
```

**After**:
```python
from codeintel.core.errors.schema_errors import SchemaNotFoundError, SchemaLoadError

try:
    from codeintel.build.schemas.json_schema_registry import get_json_schema_for_dataset_name
    return get_json_schema_for_dataset_name(schema_name)
except SchemaNotFoundError:
    log.debug("No generated schema for %s", schema_name)
    return None
except SchemaLoadError as e:
    log.warning("Schema load failed for %s: %s", schema_name, e)
    return None
except ImportError:
    log.debug("JSON schema registry not available")
    return None
```

##### Executor Error Handling

For the executor's catch-all pattern, use a **Result type**:

```python
"""Result type for explicit success/failure handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")

__all__ = ["Result", "Ok", "Err"]


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Successful result containing a value."""
    value: T


@dataclass(frozen=True)
class Err:
    """Failed result containing an exception."""
    error: Exception


Result = Ok[T] | Err


def try_execute(fn: Callable[[], T]) -> Result[T]:
    """Execute a function and capture any exception.
    
    This is the ONE place where we catch Exception broadly,
    with full logging and type preservation.
    """
    try:
        return Ok(fn())
    except Exception as exc:
        log.exception("Execution failed: %s", exc)
        return Err(exc)
```

---

### Proposal 3: Type Coercion via Singledispatch

**Eliminates**: PLR0911 (2 many-return functions)

#### Problem Statement

`optional_int()` and `optional_float()` have many return statements because they dispatch on runtime type.

#### Solution: Use `functools.singledispatch`

```python
"""Type coercion using singledispatch for clean multi-type handling."""

from __future__ import annotations

from functools import singledispatch
from typing import Any

__all__ = ["coerce_int", "coerce_float"]


@singledispatch
def coerce_int(value: Any) -> int | None:
    """Convert a value to int or None.
    
    This is the fallback for unregistered types.
    """
    if hasattr(value, "__int__"):
        try:
            return value.__int__()
        except (TypeError, ValueError):
            return None
    if hasattr(value, "__index__"):
        try:
            return value.__index__()
        except (TypeError, ValueError):
            return None
    return None


@coerce_int.register(type(None))
def _coerce_int_none(value: None) -> None:
    return None


@coerce_int.register(bool)
def _coerce_int_bool(value: bool) -> int:
    return int(value)


@coerce_int.register(int)
def _coerce_int_int(value: int) -> int:
    return value


@coerce_int.register(float)
def _coerce_int_float(value: float) -> int:
    return int(value)


@coerce_int.register(str)
def _coerce_int_str(value: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


# Wrapper function for backward compatibility
def optional_int(value: object | None) -> int | None:
    """Return an integer or None when value is not provided.
    
    Parameters
    ----------
    value
        Value to convert to integer.
        
    Returns
    -------
    int | None
        Converted integer or None when input is missing or invalid.
    """
    return coerce_int(value)
```

##### Benefits

1. **Single responsibility** — Each handler does one thing
2. **Extensible** — New types can register handlers
3. **No complex control flow** — Dispatch is automatic
4. **Testable** — Each handler can be tested in isolation

---

### Proposal 4: Immutability Testing Utility

**Eliminates**: PLC2801 (2 dunder calls)

#### Problem Statement

Tests use `object.__setattr__` to verify frozen dataclass immutability.

#### Solution: Dedicated Test Utility

##### New Module: `tests/_helpers/immutability.py`

```python
"""Utilities for testing frozen dataclass immutability."""

from __future__ import annotations

from typing import Any

import pytest

__all__ = ["assert_frozen", "assert_attribute_immutable"]


def assert_frozen(instance: Any, attribute: str, new_value: Any) -> None:
    """Assert that a frozen dataclass attribute cannot be modified.
    
    Parameters
    ----------
    instance
        The frozen dataclass instance to test.
    attribute
        Name of the attribute to attempt modification on.
    new_value
        Value to attempt to assign.
        
    Raises
    ------
    AssertionError
        If the instance is NOT frozen (mutation succeeded).
        
    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass(frozen=True)
    ... class MyFrozen:
    ...     name: str
    >>> obj = MyFrozen(name="original")
    >>> assert_frozen(obj, "name", "changed")  # Passes (is frozen)
    """
    with pytest.raises(
        (AttributeError, TypeError),
        match=r"(cannot assign|frozen|immutable)",
    ):
        setattr(instance, attribute, new_value)


def assert_attribute_immutable(instance: Any, attribute: str) -> None:
    """Assert that an attribute on an object cannot be modified.
    
    Uses a sentinel value that differs from current value.
    
    Parameters
    ----------
    instance
        The object instance to test.
    attribute
        Name of the attribute to test for immutability.
    """
    current = getattr(instance, attribute)
    # Create a distinct test value
    if isinstance(current, str):
        test_value = current + "_modified"
    elif isinstance(current, int):
        test_value = current + 1
    elif isinstance(current, bool):
        test_value = not current
    else:
        test_value = object()
    
    assert_frozen(instance, attribute, test_value)
```

##### Refactored Test

**Before**:
```python
with pytest.raises(AttributeError):
    object.__setattr__(cast("Any", meta), "name", "changed")  # noqa: PLC2801
```

**After**:
```python
from tests._helpers.immutability import assert_frozen

assert_frozen(meta, "name", "changed")
```

---

### Proposal 5: Named Constants Module

**Eliminates**: PLR2004 (1 magic value)

#### Solution: Cryptographic Constants

```python
"""Cryptographic constants used throughout the codebase."""

from __future__ import annotations

__all__ = [
    "SHA256_HEX_DIGEST_LENGTH",
    "SHA256_DIGEST_BYTES",
]

# SHA-256 produces 32 bytes = 64 hexadecimal characters
SHA256_DIGEST_BYTES: int = 32
SHA256_HEX_DIGEST_LENGTH: int = SHA256_DIGEST_BYTES * 2  # 64
```

**Refactored Usage**:
```python
from codeintel.core.constants.crypto import SHA256_HEX_DIGEST_LENGTH

if len(digest) != SHA256_HEX_DIGEST_LENGTH:
    raise ValueError(f"Invalid SHA-256 digest length: {len(digest)}")
```

---

### Proposal 6: Run Record Builder Pattern

**Eliminates**: PLR0913 (1 many-argument function)

#### Problem Statement

`create_run_record()` has 6 parameters with conditional requirements.

#### Solution: Builder with Type-Safe States

```python
"""Type-safe builder for TargetRunRecord creation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.manifest_hook import TargetRunRecord
    from codeintel.build.targets import OutputTarget


@dataclass
class RunRecordBuilder:
    """Builder for constructing TargetRunRecord with validation.
    
    Ensures that required parameters are provided for each status type.
    
    Examples
    --------
    >>> builder = RunRecordBuilder(target, input_hash)
    >>> record = builder.succeeded(env, row_counts={"table": 100})
    
    >>> record = builder.failed(error=ValueError("bad input"))
    
    >>> record = builder.skipped(env)
    """
    
    target: OutputTarget
    input_hash: str
    options_hash: str | None = None
    _start_time: float = field(default=0.0, repr=False)
    
    def succeeded(
        self,
        env: BuildEnv,
        *,
        duration_ms: float,
        row_counts: dict[str, int] | None = None,
    ) -> TargetRunRecord:
        """Create a success record.
        
        Parameters
        ----------
        env
            Build environment with gateway and paths.
        duration_ms
            Execution duration in milliseconds.
        row_counts
            Optional row counts per table key.
            
        Returns
        -------
        TargetRunRecord
            Record with status="succeeded".
        """
        return self._build("succeeded", env=env, duration_ms=duration_ms, row_counts=row_counts)
    
    def skipped(self, env: BuildEnv) -> TargetRunRecord:
        """Create a skipped record.
        
        Parameters
        ----------
        env
            Build environment.
            
        Returns
        -------
        TargetRunRecord
            Record with status="skipped".
        """
        return self._build("skipped", env=env, duration_ms=0.0)
    
    def failed(
        self,
        error: Exception,
        *,
        duration_ms: float = 0.0,
    ) -> TargetRunRecord:
        """Create a failure record.
        
        Parameters
        ----------
        error
            The exception that caused failure.
        duration_ms
            Execution duration before failure.
            
        Returns
        -------
        TargetRunRecord
            Record with status="failed".
        """
        return self._build("failed", error=error, duration_ms=duration_ms)
    
    def _build(
        self,
        status: Literal["succeeded", "skipped", "failed"],
        *,
        env: BuildEnv | None = None,
        duration_ms: float = 0.0,
        row_counts: dict[str, int] | None = None,
        error: Exception | None = None,
    ) -> TargetRunRecord:
        """Internal builder method."""
        # Import here to keep builder lightweight
        from codeintel.build.hamilton.native.runner import _build_record_impl
        
        return _build_record_impl(
            target=self.target,
            status=status,
            input_hash=self.input_hash,
            options_hash=self.options_hash,
            env=env,
            duration_ms=duration_ms,
            row_counts=row_counts,
            error=error,
        )
```

##### Benefits

1. **Type-safe states** — Each status method requires only its needed parameters
2. **Validated construction** — Can't create invalid combinations
3. **Fluent API** — Clear intent at call site
4. **Single responsibility** — Builder knows construction, not execution

---

## Implementation Plan

### Phase 1: Foundation (PR-79)

**Duration**: 2-3 days  
**Scope**: Core infrastructure changes

| Task | Files | Removes |
|------|-------|---------|
| Create `codeintel.core.di` module | `src/codeintel/core/di.py` | Foundation for PLC0415/PLW0603 |
| Create error hierarchy | `src/codeintel/core/errors/schema_errors.py` | Foundation for BLE001 |
| Create constants module | `src/codeintel/core/constants/crypto.py` | PLR2004 |
| Create immutability test helper | `tests/_helpers/immutability.py` | PLC2801 |

### Phase 2: Type Coercion (PR-80)

**Duration**: 1 day  
**Scope**: Singledispatch refactor

| Task | Files | Removes |
|------|-------|---------|
| Refactor `optional_int` | `src/codeintel/analytics/utilities/type_coercion.py` | PLR0911 ×1 |
| Refactor `optional_float` | `src/codeintel/analytics/utilities/type_coercion.py` | PLR0911 ×1 |
| Update tests | `tests/analytics/test_type_coercion.py` | — |

### Phase 3: Error Handling (PR-81)

**Duration**: 2 days  
**Scope**: Replace blind excepts

| Task | Files | Removes |
|------|-------|---------|
| Refactor validation.py | `src/codeintel/build/exports/validation.py` | BLE001 ×1 |
| Refactor conformance.py | `src/codeintel/storage/validation/conformance.py` | BLE001 ×1 |
| Refactor common.py | `src/codeintel/build/exports/common.py` | BLE001 ×1 |
| Refactor executor.py | `src/codeintel/build/hamilton/native/executor.py` | BLE001 ×1 |

### Phase 4: Dependency Injection (PR-82)

**Duration**: 3-4 days  
**Scope**: Migrate contract_provider.py and unified_registry.py

| Task | Files | Removes |
|------|-------|---------|
| Migrate contract_provider.py | `src/codeintel/build/schemas/contract_provider.py` | PLW0603 ×5, PLC0415 ×5 |
| Migrate unified_registry.py | `src/codeintel/build/unified_registry.py` | PLW0603 ×2, PLC0415 ×1 |
| Update consumers | Various | — |

### Phase 5: Registration Refactor (PR-83)

**Duration**: 2-3 days  
**Scope**: Migrate registrations.py and plugin imports

| Task | Files | Removes |
|------|-------|---------|
| Migrate registrations.py | `src/codeintel/build/registrations.py` | PLC0415 ×8 |
| Migrate provider_unified.py | `src/codeintel/build/schemas/provider_unified.py` | PLC0415 ×3 |
| Migrate remaining files | Various | PLC0415 ×remaining |

### Phase 6: Builder Pattern (PR-84)

**Duration**: 1 day  
**Scope**: Run record builder

| Task | Files | Removes |
|------|-------|---------|
| Create RunRecordBuilder | `src/codeintel/build/hamilton/native/runner.py` | PLR0913 ×1 |
| Update executor.py | `src/codeintel/build/hamilton/native/executor.py` | — |

### Phase 7: Test Migration (PR-85)

**Duration**: 1 day  
**Scope**: Update tests to use new patterns

| Task | Files | Removes |
|------|-------|---------|
| Use immutability helper | `tests/build/test_hamilton_phase0.py` | PLC2801 ×1 |
| Use immutability helper | `tests/build/test_hamilton_phase1.py` | PLC2801 ×1 |
| Use constants | `tests/build/hamilton/test_pr73_json_schema_generation.py` | PLR2004 ×1 |
| Update test imports | Various test files | PLC0415 in tests |

---

## Migration Strategy

### Backward Compatibility

1. **Deprecation shims** — Old function signatures remain but emit `DeprecationWarning`
2. **Dual export** — Both old and new APIs available during transition
3. **Version pinning** — Changes gated behind feature flags if needed

### Rollback Plan

Each PR is independently deployable and revertible:
- No single PR creates breaking changes
- Each PR has comprehensive tests
- Old code paths remain until explicitly deleted in final cleanup PR

### Testing Strategy

1. **Unit tests** — Each new module has 100% coverage
2. **Integration tests** — Verify DI container works end-to-end
3. **Regression tests** — Existing tests must pass unchanged
4. **Property tests** — Fuzz testing for type coercion

---

## Success Criteria

### Quantitative

| Metric | Current | Target |
|--------|---------|--------|
| Total noqa suppressions | 62 | 0 |
| PLC0415 suppressions | 44 | 0 |
| PLW0603 suppressions | 6 | 0 |
| BLE001 suppressions | 4 | 0 |
| PLR0911 suppressions | 2 | 0 |
| PLC2801 suppressions | 2 | 0 |
| PLR2004 suppressions | 1 | 0 |
| PLR0913 suppressions | 1 | 0 |

### Qualitative

1. **Testability** — All singletons can be reset/mocked in tests
2. **Type safety** — No `cast("Any", ...)` patterns remain
3. **Error clarity** — All error paths log with context
4. **Extensibility** — New types can be added without modifying existing code
5. **Documentation** — All new modules have docstrings and examples

### Verification

```bash
# Confirm zero suppressions
rg "# noqa" src/ tests/ tools/ --type py | wc -l
# Expected: 0

# Confirm all quality gates pass
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
# Expected: All checks pass

# Confirm tests pass
uv run pytest -q
# Expected: All tests pass
```

---

## Related Documents

- `Hamilton_consolidation_phase3.md` — Completed Phase 3 work
- `Hamilton_consolidation_remaining_scope.md` — TypedDict migration scope
- `AGENTS.md` — Development standards and quality gates

---

## Appendix: Full Suppression Inventory

### Source Files (48 suppressions)

| File | Line | Rule | Context |
|------|------|------|---------|
| `build/plugin.py` | 247 | PLC0415 | Plugin metadata import |
| `analytics/utilities/type_coercion.py` | 66 | PLR0911 | optional_int |
| `analytics/utilities/type_coercion.py` | 114 | PLR0911 | optional_float |
| `build/exports/validation.py` | 39 | PLC0415 | JSON schema import |
| `build/exports/validation.py` | 44 | BLE001 | Schema lookup fallback |
| `storage/validation/conformance.py` | 68 | PLC0415 | JSON schema import |
| `storage/validation/conformance.py` | 73 | BLE001 | Schema lookup fallback |
| `build/exports/common.py` | 285 | PLC0415 | JSON schema import |
| `build/exports/common.py` | 290 | BLE001 | Digest computation fallback |
| `build/exports/common.py` | 356 | PLC0415 | Export expressions import |
| `build/schemas/json_schema_registry.py` | 41 | PLC0415 | Schema provider import |
| `build/schemas/json_schema_registry.py` | 92 | PLC0415 | Contract provider import |
| `build/schemas/__init__.py` | 117 | PLC0415 | Dynamic importlib |
| `build/schemas/compile.py` | 141 | PLC0415 | Infer DuckDB schema |
| `build/schemas/compile.py` | 142 | PLC0415 | View names import |
| `build/schemas/compile.py` | 176 | PLC0415 | Contracts import |
| `build/schemas/contract_provider.py` | 54 | PLW0603 | Global registry module |
| `build/schemas/contract_provider.py` | 56 | PLC0415 | Registry import |
| `build/schemas/contract_provider.py` | 70 | PLW0603 | Global row registry |
| `build/schemas/contract_provider.py` | 72 | PLC0415 | Row registry import |
| `build/schemas/contract_provider.py` | 86 | PLW0603 | Global contracts module |
| `build/schemas/contract_provider.py` | 88 | PLC0415 | Contracts import |
| `build/schemas/contract_provider.py` | 102 | PLW0603 | Global composites module |
| `build/schemas/contract_provider.py` | 104 | PLC0415 | Composites import |
| `build/schemas/contract_provider.py` | 647 | PLW0603 | Global provider instance |
| `config/datasets/dataflow.py` | 85 | PLC0415 | Schema provider import |
| `storage/datasets/registry.py` | 207 | PLC0415 | Contracts import |
| `config/datasets/columns.py` | 24 | PLC0415 | Schema provider import |
| `build/schemas/provider_unified.py` | 40 | PLC0415 | Registry import |
| `build/schemas/provider_unified.py` | 133 | PLC0415 | Hamilton provider import |
| `build/schemas/provider_unified.py` | 244 | PLC0415 | Hamilton provider import |
| `build/schemas/registry.py` | 58 | PLC0415 | Unified provider import |
| `build/schemas/registry.py` | 114 | PLC0415 | Unified provider import |
| `build/registrations.py` | 82 | PLC0415 | Ingestion plugins import |
| `build/registrations.py` | 93 | PLC0415 | Registry targets import |
| `build/registrations.py` | 140 | PLC0415 | Analytics plugins import |
| `build/registrations.py` | 143 | PLC0415 | Graphs plugins import |
| `build/registrations.py` | 152 | PLC0415 | Registry targets import |
| `build/registrations.py` | 207 | PLC0415 | Analytics plugins import |
| `build/registrations.py` | 224 | PLC0415 | Registry targets import |
| `build/registrations.py` | 327 | PLC0415 | Registry targets import |
| `build/plugins/_helpers.py` | 145 | PLC0415 | DuckDB error import |
| `analytics/graphs/plugin_catalog.py` | 52 | PLC0415 | Unified registry import |
| `build/unified_registry.py` | 418 | PLW0603 | Global registry |
| `build/unified_registry.py` | 435 | PLC0415 | Registrations import |
| `build/unified_registry.py` | 448 | PLW0603 | Global registry reset |
| `build/hamilton/native/executor.py` | 207 | BLE001 | Execution catch-all |
| `build/hamilton/native/runner.py` | 94 | PLR0913 | Many function args |

### Test Files (14 suppressions)

| File | Line | Rule | Context |
|------|------|------|---------|
| `tests/docs_export/test_export_defaults.py` | 34 | PLC0415 | Test import |
| `tests/build/hamilton/test_pr73_json_schema_generation.py` | 448 | PLC0415 | Test import |
| `tests/build/hamilton/test_pr73_json_schema_generation.py` | 463 | PLC0415 | Test import |
| `tests/build/hamilton/test_pr73_json_schema_generation.py` | 479 | PLC0415 | Test import |
| `tests/build/hamilton/test_pr73_json_schema_generation.py` | 491 | PLC0415 | Test import |
| `tests/build/hamilton/test_pr73_json_schema_generation.py` | 500 | PLR2004 | Magic value (64) |
| `tests/build/hamilton/test_pr73_json_schema_generation.py` | 506 | PLC0415 | Test import |
| `tests/build/hamilton/test_pr73_json_schema_generation.py` | 528 | PLC0415 | Test import |
| `tests/build/hamilton/test_pr73_json_schema_generation.py` | 529 | PLC0415 | Test import |
| `tests/build/hamilton/test_pr73_json_schema_generation.py` | 531 | PLC0415 | Test import |
| `tests/build/hamilton/test_pr69_unified_schema_provider.py` | 161 | PLC0415 | Test import |
| `tests/build/hamilton/test_pr69_unified_schema_provider.py` | 300 | PLC0415 | Test import |
| `tests/build/test_hamilton_phase0.py` | 138 | PLC2801 | Immutability test |
| `tests/build/test_hamilton_phase1.py` | 292 | PLC2801 | Immutability test |

