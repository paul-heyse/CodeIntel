# Phase 5: Skip/Manifest Integration Implementation Plan

> **Scope**: Enable execution skipping based on input hash comparison
> **Duration**: 2-3 days
> **Risk Level**: Medium-High (affects execution flow)
> **Depends On**: Phase 4 (Profile Integration)

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Task 1: ManifestStore Implementation](#3-task-1-manifeststore-implementation)
4. [Task 2: Skip Decision Logic](#4-task-2-skip-decision-logic)
5. [Task 3: Executor Skip Integration](#5-task-3-executor-skip-integration)
6. [Task 4: Manifest Persistence](#6-task-4-manifest-persistence)
7. [Task 5: CLI Skip Controls](#7-task-5-cli-skip-controls)
8. [Verification](#8-verification)
9. [Rollback Plan](#9-rollback-plan)

---

## 1. Overview

Phase 5 completes the unified data abstraction by enabling intelligent skip decisions:

1. **ManifestStore implementation** - Persistent storage for execution records
2. **Skip decision logic** - Compare current vs. previous input hashes
3. **Executor integration** - Check for skip before executing plugins
4. **Manifest persistence** - Record execution results for future comparison
5. **CLI controls** - `--force` to bypass skip, `--dry-run` to preview

### Skip Decision Flow

```
┌─────────────────┐
│  Prepare Run    │
│  (options,      │
│   upstream)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│ Compute Hashes  │────▶│ Load Prior   │
│ (options_hash,  │     │ Record from  │
│  input_hash)    │     │ ManifestStore│
└────────┬────────┘     └──────┬───────┘
         │                     │
         ▼                     ▼
┌─────────────────────────────────────┐
│     Compare input_hash              │
│     current vs. prior               │
└────────┬───────────────────┬────────┘
         │                   │
    MATCH│              MISMATCH
         │                   │
         ▼                   ▼
┌─────────────────┐  ┌─────────────────┐
│  SKIP Plugin    │  │ EXECUTE Plugin  │
│  (return prior  │  │ (run compute,   │
│   result)       │  │  record result) │
└─────────────────┘  └─────────────────┘
```

---

## 2. Prerequisites

Verify Phase 4 is complete:

```bash
# Verify profile integration
uv run python -c "
from codeintel.build.config import create_build_config
from codeintel.core.plugins.execution.profiles import get_profile

config = create_build_config(profile='fast')
resolver = config.build_options_resolver()
print('Phase 4 profile integration verified')

# Check options flow
opts = resolver.config_source.get_plugin_options('analytics.function_metrics')
print(f'  Fast profile disables graph metrics: {opts.get(\"include_graph_metrics\") is False}')
"

# Run Phase 4 tests
uv run pytest tests/core/plugins/test_profiles.py tests/build/test_build_config.py -v
```

---

## 3. Task 1: ManifestStore Implementation

### 3.1 Create DuckDB ManifestStore

```python
# File: src/codeintel/core/plugins/execution/manifest_store.py
"""Manifest store implementations.

This module provides concrete ManifestStore implementations for
persisting and retrieving plugin execution records.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.core.plugins.execution.manifest import ManifestStore
from codeintel.core.plugins.types.result import (
    PluginExecutionRecord,
    PluginResult,
    PluginStatus,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

log = logging.getLogger(__name__)


class DuckDBManifestStore(ManifestStore):
    """ManifestStore backed by DuckDB.

    This implementation stores execution records in a DuckDB table,
    enabling fast lookups and persistence across runs.

    Attributes
    ----------
    con
        DuckDB connection.
    table_name
        Table name for storing records.

    Examples
    --------
    >>> from codeintel.storage.gateway import StorageGateway
    >>> gateway = StorageGateway.from_path(Path("db.duckdb"))
    >>> store = DuckDBManifestStore(gateway.con)
    >>> store.ensure_schema()
    """

    def __init__(
        self,
        con: DuckDBPyConnection,
        *,
        table_name: str = "core.plugin_execution_manifest",
    ) -> None:
        """Initialize store with DuckDB connection.

        Parameters
        ----------
        con
            DuckDB connection.
        table_name
            Table name for manifest records.
        """
        self._con = con
        self._table_name = table_name

    def ensure_schema(self) -> None:
        """Create the manifest table if it doesn't exist."""
        self._con.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                id INTEGER PRIMARY KEY,
                plugin_name VARCHAR NOT NULL,
                repo VARCHAR NOT NULL,
                commit VARCHAR NOT NULL,
                scope_id VARCHAR,
                variant VARCHAR,
                status VARCHAR NOT NULL,
                started_at TIMESTAMP NOT NULL,
                ended_at TIMESTAMP NOT NULL,
                duration_ms DOUBLE NOT NULL,
                options_hash VARCHAR,
                input_hash VARCHAR,
                error VARCHAR,
                meta_json VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for fast lookups
        self._con.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_manifest_lookup
            ON {self._table_name} (plugin_name, repo, commit, scope_id, variant)
        """)

    def load_last_record(
        self,
        *,
        plugin_name: str,
        repo: str,
        commit: str,
        scope_id: str | None,
        variant: str | None,
    ) -> PluginExecutionRecord | None:
        """Load the most recent record matching the criteria.

        Parameters
        ----------
        plugin_name
            Canonical plugin name.
        repo
            Repository identifier.
        commit
            Commit SHA.
        scope_id
            Scope hash or None.
        variant
            Profile/variant name or None.

        Returns
        -------
        PluginExecutionRecord | None
            Most recent matching record, or None.
        """
        # Handle NULL comparison for scope_id and variant
        scope_clause = "scope_id IS NULL" if scope_id is None else "scope_id = ?"
        variant_clause = "variant IS NULL" if variant is None else "variant = ?"

        params: list[Any] = [plugin_name, repo, commit]
        if scope_id is not None:
            params.append(scope_id)
        if variant is not None:
            params.append(variant)

        query = f"""
            SELECT
                plugin_name, status, started_at, ended_at, duration_ms,
                options_hash, input_hash, error, meta_json
            FROM {self._table_name}
            WHERE plugin_name = ?
              AND repo = ?
              AND commit = ?
              AND {scope_clause}
              AND {variant_clause}
            ORDER BY created_at DESC
            LIMIT 1
        """

        try:
            row = self._con.execute(query, params).fetchone()
        except Exception as e:  # noqa: BLE001
            log.debug("Error loading manifest record: %s", e)
            return None

        if row is None:
            return None

        (
            name,
            status,
            started_at,
            ended_at,
            duration_ms,
            options_hash,
            input_hash,
            error,
            meta_json,
        ) = row

        meta = json.loads(meta_json) if meta_json else {}
        meta["options_hash"] = options_hash
        meta["input_hash"] = input_hash
        meta["repo"] = repo
        meta["commit"] = commit
        meta["scope_id"] = scope_id
        meta["variant"] = variant

        return PluginExecutionRecord(
            plugin_name=name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            error=error,
            meta=meta,
        )

    def append_record(self, record: PluginExecutionRecord) -> None:
        """Persist a new execution record.

        Parameters
        ----------
        record
            Execution record to persist.
        """
        meta = dict(record.meta)
        repo = meta.pop("repo", "")
        commit = meta.pop("commit", "")
        scope_id = meta.pop("scope_id", None)
        variant = meta.pop("variant", None)
        options_hash = meta.pop("options_hash", None)
        input_hash = meta.pop("input_hash", None)

        self._con.execute(
            f"""
            INSERT INTO {self._table_name} (
                plugin_name, repo, commit, scope_id, variant,
                status, started_at, ended_at, duration_ms,
                options_hash, input_hash, error, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.plugin_name,
                repo,
                commit,
                scope_id,
                variant,
                record.status,
                record.started_at,
                record.ended_at,
                record.duration_ms,
                options_hash,
                input_hash,
                record.error,
                json.dumps(meta) if meta else None,
            ],
        )


class InMemoryManifestStore(ManifestStore):
    """In-memory ManifestStore for testing.

    This implementation stores records in memory without persistence.
    Useful for testing and single-run scenarios.
    """

    def __init__(self) -> None:
        """Initialize empty store."""
        self._records: dict[str, PluginExecutionRecord] = {}

    def _make_key(
        self,
        plugin_name: str,
        repo: str,
        commit: str,
        scope_id: str | None,
        variant: str | None,
    ) -> str:
        """Create lookup key from parameters."""
        return f"{plugin_name}:{repo}:{commit}:{scope_id}:{variant}"

    def load_last_record(
        self,
        *,
        plugin_name: str,
        repo: str,
        commit: str,
        scope_id: str | None,
        variant: str | None,
    ) -> PluginExecutionRecord | None:
        """Load record from memory.

        Returns
        -------
        PluginExecutionRecord | None
            Stored record or None.
        """
        key = self._make_key(plugin_name, repo, commit, scope_id, variant)
        return self._records.get(key)

    def append_record(self, record: PluginExecutionRecord) -> None:
        """Store record in memory.

        Parameters
        ----------
        record
            Record to store.
        """
        meta = record.meta
        key = self._make_key(
            record.plugin_name,
            str(meta.get("repo", "")),
            str(meta.get("commit", "")),
            meta.get("scope_id"),
            meta.get("variant"),
        )
        self._records[key] = record

    def clear(self) -> None:
        """Clear all stored records."""
        self._records.clear()


__all__ = [
    "DuckDBManifestStore",
    "InMemoryManifestStore",
]
```

### 3.2 Test File: `tests/core/plugins/test_manifest_store.py`

```python
# File: tests/core/plugins/test_manifest_store.py
"""Tests for ManifestStore implementations."""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import pytest

from codeintel.core.plugins.execution.manifest_store import (
    DuckDBManifestStore,
    InMemoryManifestStore,
)
from codeintel.core.plugins.types.result import PluginExecutionRecord


@pytest.fixture
def duckdb_store(tmp_path):
    """Create DuckDB manifest store."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS core")
    store = DuckDBManifestStore(con)
    store.ensure_schema()
    return store


@pytest.fixture
def memory_store():
    """Create in-memory manifest store."""
    return InMemoryManifestStore()


@pytest.fixture
def sample_record():
    """Create sample execution record."""
    now = datetime.now(tz=UTC)
    return PluginExecutionRecord(
        plugin_name="test.plugin",
        status="succeeded",
        started_at=now,
        ended_at=now,
        duration_ms=100.0,
        meta={
            "repo": "owner/repo",
            "commit": "abc123",
            "scope_id": None,
            "variant": "fast",
            "options_hash": "opts123",
            "input_hash": "input456",
        },
    )


class TestDuckDBManifestStore:
    """Tests for DuckDBManifestStore."""

    def test_append_and_load(
        self,
        duckdb_store: DuckDBManifestStore,
        sample_record: PluginExecutionRecord,
    ) -> None:
        """Verify record can be stored and loaded."""
        duckdb_store.append_record(sample_record)

        loaded = duckdb_store.load_last_record(
            plugin_name="test.plugin",
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant="fast",
        )

        assert loaded is not None
        assert loaded.plugin_name == "test.plugin"
        assert loaded.status == "succeeded"
        assert loaded.meta["input_hash"] == "input456"

    def test_load_missing_returns_none(
        self,
        duckdb_store: DuckDBManifestStore,
    ) -> None:
        """Verify loading missing record returns None."""
        loaded = duckdb_store.load_last_record(
            plugin_name="nonexistent",
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant=None,
        )
        assert loaded is None

    def test_loads_most_recent(
        self,
        duckdb_store: DuckDBManifestStore,
    ) -> None:
        """Verify most recent record is loaded."""
        now = datetime.now(tz=UTC)

        # Insert older record
        old_record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=100.0,
            meta={
                "repo": "owner/repo",
                "commit": "abc123",
                "scope_id": None,
                "variant": None,
                "input_hash": "old_hash",
            },
        )
        duckdb_store.append_record(old_record)

        # Insert newer record
        new_record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=50.0,
            meta={
                "repo": "owner/repo",
                "commit": "abc123",
                "scope_id": None,
                "variant": None,
                "input_hash": "new_hash",
            },
        )
        duckdb_store.append_record(new_record)

        loaded = duckdb_store.load_last_record(
            plugin_name="test.plugin",
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant=None,
        )

        assert loaded is not None
        assert loaded.meta["input_hash"] == "new_hash"


class TestInMemoryManifestStore:
    """Tests for InMemoryManifestStore."""

    def test_append_and_load(
        self,
        memory_store: InMemoryManifestStore,
        sample_record: PluginExecutionRecord,
    ) -> None:
        """Verify record can be stored and loaded."""
        memory_store.append_record(sample_record)

        loaded = memory_store.load_last_record(
            plugin_name="test.plugin",
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant="fast",
        )

        assert loaded is not None
        assert loaded.plugin_name == "test.plugin"

    def test_clear(
        self,
        memory_store: InMemoryManifestStore,
        sample_record: PluginExecutionRecord,
    ) -> None:
        """Verify clear removes all records."""
        memory_store.append_record(sample_record)
        memory_store.clear()

        loaded = memory_store.load_last_record(
            plugin_name="test.plugin",
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant="fast",
        )
        assert loaded is None
```

---

## 4. Task 2: Skip Decision Logic

### 4.1 Create Skip Decision Module

```python
# File: src/codeintel/core/plugins/execution/skip.py
"""Plugin skip decision logic.

This module provides functions for determining whether a plugin
execution can be skipped based on manifest comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.core.plugins.types.result import (
    PluginExecutionRecord,
    PluginResult,
)

if TYPE_CHECKING:
    from codeintel.core.plugins.execution.manifest import ManifestStore
    from codeintel.core.plugins.execution.run_context import PluginRunContext

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkipDecision:
    """Result of skip decision check.

    Attributes
    ----------
    should_skip
        Whether the plugin should be skipped.
    reason
        Reason for the decision.
    prior_record
        Prior execution record if skip is recommended.
    """

    should_skip: bool
    reason: str
    prior_record: PluginExecutionRecord | None = None

    @classmethod
    def execute(cls, reason: str) -> SkipDecision:
        """Create a decision to execute.

        Parameters
        ----------
        reason
            Reason for executing.

        Returns
        -------
        SkipDecision
            Decision to execute.
        """
        return cls(should_skip=False, reason=reason)

    @classmethod
    def skip(
        cls,
        reason: str,
        prior_record: PluginExecutionRecord,
    ) -> SkipDecision:
        """Create a decision to skip.

        Parameters
        ----------
        reason
            Reason for skipping.
        prior_record
            Prior execution record to reuse.

        Returns
        -------
        SkipDecision
            Decision to skip.
        """
        return cls(should_skip=True, reason=reason, prior_record=prior_record)


def should_skip_plugin(
    run_context: PluginRunContext,
    manifest_store: ManifestStore,
    *,
    repo: str,
    commit: str,
    scope_id: str | None,
    variant: str | None,
    force: bool = False,
) -> SkipDecision:
    """Determine whether a plugin execution should be skipped.

    Compare the current input_hash against the prior execution record
    to decide if computation can be skipped.

    Parameters
    ----------
    run_context
        Prepared plugin run context with computed hashes.
    manifest_store
        Store for loading prior execution records.
    repo
        Repository identifier.
    commit
        Commit SHA.
    scope_id
        Scope hash or None.
    variant
        Profile/variant name or None.
    force
        If True, never skip (always execute).

    Returns
    -------
    SkipDecision
        Decision with reasoning and prior record if skipping.

    Examples
    --------
    >>> decision = should_skip_plugin(
    ...     run_context=ctx,
    ...     manifest_store=store,
    ...     repo="owner/repo",
    ...     commit="abc123",
    ...     scope_id=None,
    ...     variant="fast",
    ... )
    >>> if decision.should_skip:
    ...     print(f"Skipping: {decision.reason}")
    ... else:
    ...     print(f"Executing: {decision.reason}")
    """
    plugin_name = run_context.plugin_name
    current_input_hash = run_context.input_hash

    # Force execution requested
    if force:
        log.debug("skip_check: %s force=True, executing", plugin_name)
        return SkipDecision.execute("force execution requested")

    # Load prior record
    prior_record = manifest_store.load_last_record(
        plugin_name=plugin_name,
        repo=repo,
        commit=commit,
        scope_id=scope_id,
        variant=variant,
    )

    # No prior record: must execute
    if prior_record is None:
        log.debug("skip_check: %s no prior record, executing", plugin_name)
        return SkipDecision.execute("no prior execution record")

    # Prior execution failed: re-execute
    if prior_record.status == "failed":
        log.debug("skip_check: %s prior failed, re-executing", plugin_name)
        return SkipDecision.execute("prior execution failed")

    # Compare input hashes
    prior_input_hash = prior_record.meta.get("input_hash")

    if prior_input_hash is None:
        log.debug("skip_check: %s prior has no input_hash, executing", plugin_name)
        return SkipDecision.execute("prior record missing input_hash")

    if prior_input_hash == current_input_hash:
        log.info(
            "skip_check: %s input_hash unchanged (%s), skipping",
            plugin_name,
            current_input_hash[:8],
        )
        return SkipDecision.skip(
            "input_hash unchanged",
            prior_record,
        )

    log.debug(
        "skip_check: %s input_hash changed (%s -> %s), executing",
        plugin_name,
        prior_input_hash[:8],
        current_input_hash[:8],
    )
    return SkipDecision.execute(
        f"input_hash changed: {prior_input_hash[:8]} -> {current_input_hash[:8]}"
    )


def create_skip_execution_record(
    run_context: PluginRunContext,
    prior_record: PluginExecutionRecord,
    *,
    repo: str,
    commit: str,
    scope_id: str | None,
    variant: str | None,
) -> PluginExecutionRecord:
    """Create an execution record for a skipped plugin.

    Parameters
    ----------
    run_context
        Current run context.
    prior_record
        Prior successful record being reused.
    repo
        Repository identifier.
    commit
        Commit SHA.
    scope_id
        Scope hash.
    variant
        Profile/variant.

    Returns
    -------
    PluginExecutionRecord
        Record marking execution as skipped.
    """
    now = datetime.now(tz=UTC)

    return PluginExecutionRecord(
        plugin_name=run_context.plugin_name,
        status="skipped",
        started_at=now,
        ended_at=now,
        duration_ms=0.0,
        result=prior_record.result,
        meta={
            "repo": repo,
            "commit": commit,
            "scope_id": scope_id,
            "variant": variant,
            "options_hash": run_context.options_hash,
            "input_hash": run_context.input_hash,
            "skip_reason": "input_hash_unchanged",
            "prior_input_hash": prior_record.meta.get("input_hash"),
        },
    )


__all__ = [
    "SkipDecision",
    "create_skip_execution_record",
    "should_skip_plugin",
]
```

### 4.2 Test File: `tests/core/plugins/test_skip.py`

```python
# File: tests/core/plugins/test_skip.py
"""Tests for skip decision logic."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from codeintel.core.plugins.execution.manifest_store import InMemoryManifestStore
from codeintel.core.plugins.execution.run_context import (
    PluginRunContext,
    prepare_plugin_run,
)
from codeintel.core.plugins.execution.skip import (
    SkipDecision,
    create_skip_execution_record,
    should_skip_plugin,
)
from codeintel.core.plugins.execution.options import (
    EmptyConfigSource,
    PluginOptionsResolver,
)
from codeintel.core.plugins.types.metadata import (
    CorePluginMetadata,
    PluginDomain,
)
from codeintel.core.plugins.types.result import PluginExecutionRecord


@dataclass(frozen=True)
class TestOptions:
    """Test options."""

    value: int = 10


@pytest.fixture
def sample_metadata() -> CorePluginMetadata:
    """Create sample metadata."""
    return CorePluginMetadata(
        name="test.plugin",
        version="1.0.0",
        description="Test plugin.",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        options_model=TestOptions,
    )


@pytest.fixture
def resolver() -> PluginOptionsResolver:
    """Create options resolver."""
    return PluginOptionsResolver(EmptyConfigSource())


@pytest.fixture
def run_context(
    sample_metadata: CorePluginMetadata,
    resolver: PluginOptionsResolver,
) -> PluginRunContext:
    """Create run context."""
    return prepare_plugin_run(
        metadata=sample_metadata,
        resolver=resolver,
        upstream_state={"test.input": "upstream123"},
    )


@pytest.fixture
def manifest_store() -> InMemoryManifestStore:
    """Create manifest store."""
    return InMemoryManifestStore()


class TestShouldSkipPlugin:
    """Tests for should_skip_plugin."""

    def test_force_always_executes(
        self,
        run_context: PluginRunContext,
        manifest_store: InMemoryManifestStore,
    ) -> None:
        """Verify force=True always returns execute."""
        decision = should_skip_plugin(
            run_context=run_context,
            manifest_store=manifest_store,
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant=None,
            force=True,
        )
        assert decision.should_skip is False
        assert "force" in decision.reason

    def test_no_prior_record_executes(
        self,
        run_context: PluginRunContext,
        manifest_store: InMemoryManifestStore,
    ) -> None:
        """Verify no prior record returns execute."""
        decision = should_skip_plugin(
            run_context=run_context,
            manifest_store=manifest_store,
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant=None,
        )
        assert decision.should_skip is False
        assert "no prior" in decision.reason

    def test_prior_failed_executes(
        self,
        run_context: PluginRunContext,
        manifest_store: InMemoryManifestStore,
    ) -> None:
        """Verify prior failed record returns execute."""
        now = datetime.now(tz=UTC)
        failed_record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="failed",
            started_at=now,
            ended_at=now,
            duration_ms=100.0,
            error="Some error",
            meta={
                "repo": "owner/repo",
                "commit": "abc123",
                "scope_id": None,
                "variant": None,
                "input_hash": run_context.input_hash,
            },
        )
        manifest_store.append_record(failed_record)

        decision = should_skip_plugin(
            run_context=run_context,
            manifest_store=manifest_store,
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant=None,
        )
        assert decision.should_skip is False
        assert "failed" in decision.reason

    def test_same_hash_skips(
        self,
        run_context: PluginRunContext,
        manifest_store: InMemoryManifestStore,
    ) -> None:
        """Verify matching input_hash returns skip."""
        now = datetime.now(tz=UTC)
        prior_record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=100.0,
            meta={
                "repo": "owner/repo",
                "commit": "abc123",
                "scope_id": None,
                "variant": None,
                "input_hash": run_context.input_hash,
            },
        )
        manifest_store.append_record(prior_record)

        decision = should_skip_plugin(
            run_context=run_context,
            manifest_store=manifest_store,
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant=None,
        )
        assert decision.should_skip is True
        assert "unchanged" in decision.reason
        assert decision.prior_record is not None

    def test_different_hash_executes(
        self,
        run_context: PluginRunContext,
        manifest_store: InMemoryManifestStore,
    ) -> None:
        """Verify different input_hash returns execute."""
        now = datetime.now(tz=UTC)
        prior_record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=100.0,
            meta={
                "repo": "owner/repo",
                "commit": "abc123",
                "scope_id": None,
                "variant": None,
                "input_hash": "different_hash",
            },
        )
        manifest_store.append_record(prior_record)

        decision = should_skip_plugin(
            run_context=run_context,
            manifest_store=manifest_store,
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant=None,
        )
        assert decision.should_skip is False
        assert "changed" in decision.reason


class TestCreateSkipExecutionRecord:
    """Tests for create_skip_execution_record."""

    def test_creates_skipped_record(
        self,
        run_context: PluginRunContext,
    ) -> None:
        """Verify skipped record is created correctly."""
        now = datetime.now(tz=UTC)
        prior_record = PluginExecutionRecord(
            plugin_name="test.plugin",
            status="succeeded",
            started_at=now,
            ended_at=now,
            duration_ms=100.0,
            meta={"input_hash": "prior123"},
        )

        record = create_skip_execution_record(
            run_context=run_context,
            prior_record=prior_record,
            repo="owner/repo",
            commit="abc123",
            scope_id=None,
            variant="fast",
        )

        assert record.plugin_name == "test.plugin"
        assert record.status == "skipped"
        assert record.duration_ms == 0.0
        assert record.meta["skip_reason"] == "input_hash_unchanged"
```

---

## 5. Task 3: Executor Skip Integration

### 5.1 Update Build Executor

```python
# File: src/codeintel/build/executor.py (additional modifications)
"""Build executor with skip integration.

Extends the executor to check skip conditions before executing plugins.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.core.plugins.execution.manifest import compute_scope_id
from codeintel.core.plugins.execution.run_context import prepare_plugin_run
from codeintel.core.plugins.execution.skip import (
    create_skip_execution_record,
    should_skip_plugin,
)
from codeintel.core.plugins.types.result import PluginExecutionRecord, PluginResult

if TYPE_CHECKING:
    from codeintel.build.config import BuildRunConfig
    from codeintel.build.plugin import TargetPlugin
    from codeintel.core.plugins.execution.manifest import ManifestStore
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.core.plugins.types.metadata import CorePluginMetadata

log = logging.getLogger(__name__)


class SkipAwareExecutor:
    """Executor that checks skip conditions before plugin execution.

    This executor extends build execution with:
    - Skip decision based on input hash comparison
    - Manifest recording for future skip decisions
    - Force execution support
    """

    def __init__(
        self,
        config: BuildRunConfig,
        manifest_store: ManifestStore,
        *,
        force: bool = False,
    ) -> None:
        """Initialize executor.

        Parameters
        ----------
        config
            Build run configuration.
        manifest_store
            Store for execution records.
        force
            If True, never skip plugins.
        """
        self._config = config
        self._manifest_store = manifest_store
        self._resolver = config.build_options_resolver()
        self._force = force
        self._scope_id = compute_scope_id(config.scope_paths)

    async def execute_plugin(
        self,
        plugin: TargetPlugin,
        upstream_state: dict[str, str],
    ) -> PluginExecutionRecord:
        """Execute a plugin with skip checking.

        Parameters
        ----------
        plugin
            Plugin to execute.
        upstream_state
            Capability → input hash mapping for dependencies.

        Returns
        -------
        PluginExecutionRecord
            Execution record (executed or skipped).
        """
        # Get core metadata
        core_metadata = self._get_core_metadata(plugin)

        # Prepare run context with hashes
        run_context = prepare_plugin_run(
            metadata=core_metadata,
            resolver=self._resolver,
            upstream_state=upstream_state,
        )

        # Check skip decision
        decision = should_skip_plugin(
            run_context=run_context,
            manifest_store=self._manifest_store,
            repo=self._config.repo,
            commit=self._config.commit,
            scope_id=self._scope_id,
            variant=self._config.profile_name,
            force=self._force,
        )

        if decision.should_skip and decision.prior_record is not None:
            log.info(
                "Skipping plugin %s: %s",
                plugin.plugin_name,
                decision.reason,
            )
            record = create_skip_execution_record(
                run_context=run_context,
                prior_record=decision.prior_record,
                repo=self._config.repo,
                commit=self._config.commit,
                scope_id=self._scope_id,
                variant=self._config.profile_name,
            )
            self._manifest_store.append_record(record)
            return record

        # Execute plugin
        log.info(
            "Executing plugin %s: %s",
            plugin.plugin_name,
            decision.reason,
        )
        record = await self._execute_and_record(
            plugin=plugin,
            run_context=run_context,
        )
        self._manifest_store.append_record(record)
        return record

    async def _execute_and_record(
        self,
        plugin: TargetPlugin,
        run_context: PluginRunContext,
    ) -> PluginExecutionRecord:
        """Execute plugin and create execution record.

        Parameters
        ----------
        plugin
            Plugin to execute.
        run_context
            Run context with hashes.

        Returns
        -------
        PluginExecutionRecord
            Record of execution.
        """
        started_at = datetime.now(tz=UTC)

        try:
            # Configure plugin with resolver
            configured_plugin = self._configure_plugin(plugin)

            # Execute
            from codeintel.build.context import TargetExecutionContext
            # Note: context creation would come from the build system
            # This is a simplified example
            result = await configured_plugin.execute(...)  # type: ignore[arg-type]

            ended_at = datetime.now(tz=UTC)
            duration_ms = (ended_at - started_at).total_seconds() * 1000

            return PluginExecutionRecord(
                plugin_name=run_context.plugin_name,
                status="succeeded",
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=duration_ms,
                result=PluginResult.ok(row_counts=result.row_counts),
                meta={
                    "repo": self._config.repo,
                    "commit": self._config.commit,
                    "scope_id": self._scope_id,
                    "variant": self._config.profile_name,
                    "options_hash": run_context.options_hash,
                    "input_hash": run_context.input_hash,
                },
            )
        except Exception as e:  # noqa: BLE001
            ended_at = datetime.now(tz=UTC)
            duration_ms = (ended_at - started_at).total_seconds() * 1000

            return PluginExecutionRecord(
                plugin_name=run_context.plugin_name,
                status="failed",
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=duration_ms,
                error=str(e),
                meta={
                    "repo": self._config.repo,
                    "commit": self._config.commit,
                    "scope_id": self._scope_id,
                    "variant": self._config.profile_name,
                    "options_hash": run_context.options_hash,
                    "input_hash": run_context.input_hash,
                },
            )

    def _get_core_metadata(self, plugin: TargetPlugin) -> CorePluginMetadata:
        """Get CorePluginMetadata from plugin."""
        if hasattr(plugin, "core_metadata"):
            return plugin.core_metadata
        # Fallback: create minimal metadata
        from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
        return CorePluginMetadata(
            name=plugin.plugin_name,
            version=plugin.plugin_version,
            description=plugin.plugin_description,
            domain=PluginDomain.ANALYTICS,  # Default
            kind="builder",
        )

    def _configure_plugin(self, plugin: TargetPlugin) -> TargetPlugin:
        """Configure plugin with options resolver."""
        if hasattr(plugin.__class__, "__init__"):
            import inspect
            sig = inspect.signature(plugin.__class__.__init__)
            if "options_resolver" in sig.parameters:
                return plugin.__class__(options_resolver=self._resolver)
        return plugin


__all__ = ["SkipAwareExecutor"]
```

---

## 6. Task 4: Manifest Persistence

### 6.1 Update __init__.py Exports

```python
# Add to: src/codeintel/core/plugins/execution/__init__.py

from codeintel.core.plugins.execution.manifest_store import (
    DuckDBManifestStore,
    InMemoryManifestStore,
)
from codeintel.core.plugins.execution.run_context import (
    PluginRunContext,
    prepare_plugin_run,
)
from codeintel.core.plugins.execution.skip import (
    SkipDecision,
    create_skip_execution_record,
    should_skip_plugin,
)

# Add to __all__:
# "DuckDBManifestStore",
# "InMemoryManifestStore",
# "PluginRunContext",
# "SkipDecision",
# "create_skip_execution_record",
# "prepare_plugin_run",
# "should_skip_plugin",
```

---

## 7. Task 5: CLI Skip Controls

### 7.1 Add Skip CLI Arguments

```python
# File: src/codeintel/cli/skip_args.py
"""Skip control CLI arguments."""

from __future__ import annotations

from typing import Annotated

import cyclopts

ForceArg = Annotated[
    bool,
    cyclopts.Parameter(
        name=["--force", "-f"],
        help="Force execution, skip nothing",
    ),
]

DryRunArg = Annotated[
    bool,
    cyclopts.Parameter(
        name=["--dry-run", "-n"],
        help="Show what would be executed/skipped without running",
    ),
]


__all__ = ["DryRunArg", "ForceArg"]
```

### 7.2 Example CLI Integration

```python
# Example: Updated build command with skip controls
# File: src/codeintel/cli/commands/build.py (example)

from codeintel.cli.profile_args import ProfileArg, ScopePathsArg
from codeintel.cli.skip_args import ForceArg, DryRunArg

@app.command()
def build(
    repo: str,
    commit: str,
    profile: ProfileArg = "full",
    scope: ScopePathsArg = None,
    force: ForceArg = False,
    dry_run: DryRunArg = False,
) -> None:
    """Run build with skip optimization.

    Examples
    --------
    # Normal execution with skip optimization
    codeintel build owner/repo abc123

    # Force re-execution of all plugins
    codeintel build owner/repo abc123 --force

    # Preview what would be executed
    codeintel build owner/repo abc123 --dry-run
    """
    config = build_config_from_cli(
        profile=profile,
        scope_paths=scope,
        repo=repo,
        commit=commit,
    )

    if dry_run:
        # Show skip decisions without executing
        _preview_skip_decisions(config)
        return

    # Execute with skip awareness
    executor = SkipAwareExecutor(config, manifest_store, force=force)
    # ...
```

---

## 8. Verification

### 8.1 Run Quality Checks

```bash
# Format and lint
uv run ruff format \
    src/codeintel/core/plugins/execution/manifest_store.py \
    src/codeintel/core/plugins/execution/skip.py \
    src/codeintel/cli/skip_args.py

uv run ruff check --fix src/codeintel/core/plugins/execution/ src/codeintel/cli/

# Type checking
uv run pyright \
    src/codeintel/core/plugins/execution/manifest_store.py \
    src/codeintel/core/plugins/execution/skip.py
```

### 8.2 Run Tests

```bash
# Run skip-related tests
uv run pytest tests/core/plugins/test_manifest_store.py -v
uv run pytest tests/core/plugins/test_skip.py -v

# Run integration tests
uv run pytest tests/build/ -v -k skip
```

### 8.3 Integration Test

```python
# File: tests/integration/test_skip_integration.py
"""Integration test for skip functionality."""

from __future__ import annotations

import pytest

from codeintel.build.config import create_build_config
from codeintel.core.plugins.execution.manifest_store import InMemoryManifestStore
from codeintel.core.plugins.execution.options import EmptyConfigSource, PluginOptionsResolver
from codeintel.core.plugins.execution.run_context import prepare_plugin_run
from codeintel.core.plugins.execution.skip import should_skip_plugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain


def test_skip_integration_same_commit():
    """Test that same commit with same options skips."""
    # Setup
    manifest_store = InMemoryManifestStore()
    resolver = PluginOptionsResolver(EmptyConfigSource())

    metadata = CorePluginMetadata(
        name="test.plugin",
        version="1.0.0",
        description="Test.",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
    )

    # First run: execute
    ctx1 = prepare_plugin_run(
        metadata=metadata,
        resolver=resolver,
        upstream_state={"dep": "hash1"},
    )

    decision1 = should_skip_plugin(
        run_context=ctx1,
        manifest_store=manifest_store,
        repo="owner/repo",
        commit="abc123",
        scope_id=None,
        variant="fast",
    )
    assert decision1.should_skip is False

    # Simulate execution by recording
    from datetime import UTC, datetime
    from codeintel.core.plugins.types.result import PluginExecutionRecord

    record = PluginExecutionRecord(
        plugin_name="test.plugin",
        status="succeeded",
        started_at=datetime.now(tz=UTC),
        ended_at=datetime.now(tz=UTC),
        duration_ms=100.0,
        meta={
            "repo": "owner/repo",
            "commit": "abc123",
            "scope_id": None,
            "variant": "fast",
            "input_hash": ctx1.input_hash,
        },
    )
    manifest_store.append_record(record)

    # Second run: should skip
    ctx2 = prepare_plugin_run(
        metadata=metadata,
        resolver=resolver,
        upstream_state={"dep": "hash1"},  # Same upstream
    )

    decision2 = should_skip_plugin(
        run_context=ctx2,
        manifest_store=manifest_store,
        repo="owner/repo",
        commit="abc123",
        scope_id=None,
        variant="fast",
    )
    assert decision2.should_skip is True
```

### 8.4 Verification Checklist

- [ ] `DuckDBManifestStore` can store and retrieve records
- [ ] `InMemoryManifestStore` works for testing
- [ ] `should_skip_plugin` returns correct decisions
- [ ] Force flag bypasses skip logic
- [ ] Skip records are created correctly
- [ ] Executor integrates skip checking
- [ ] CLI flags (--force, --dry-run) work

---

## 9. Rollback Plan

Phase 5 changes can be rolled back by:

1. **Revert execution modules**:
   - `src/codeintel/core/plugins/execution/manifest_store.py`
   - `src/codeintel/core/plugins/execution/skip.py`
2. **Revert CLI changes**:
   - `src/codeintel/cli/skip_args.py`
3. **Revert executor changes** to remove skip integration
4. **Delete test files**

---

## Appendix A: Complete Phase Summary

| Phase | Scope | Duration | Risk |
|-------|-------|----------|------|
| 1 | Core Infrastructure | 1-2 days | Low |
| 2 | Spine Plugin Migration | 2-3 days | Low-Medium |
| 3 | Full Rollout | 3-5 days | Medium |
| 4 | Profile Integration | 2-3 days | Medium |
| 5 | Skip/Manifest Integration | 2-3 days | Medium-High |

**Total Estimated Duration**: 10-16 days

---

## Appendix B: End State Architecture

After completing all phases, the system provides:

1. **Unified Metadata** - `CorePluginMetadata` for all plugins
2. **Profile-Driven Options** - Layered configuration (base → profile → CLI)
3. **Intelligent Skipping** - Input hash comparison for reuse
4. **Manifest Tracking** - Persistent execution history
5. **CLI Integration** - `--profile`, `--force`, `--dry-run` flags

This enables:
- **Faster iterations**: Fast profile skips expensive computations
- **Reproducible builds**: Same inputs produce same outputs
- **Incremental updates**: Only changed plugins re-execute
- **Policy-driven execution**: Centralized configuration management

---

**Migration Complete**: The unified data abstraction architecture is fully implemented and ready for production use.
