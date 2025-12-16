# Hamilton Phase 3 Wave 3 - Comprehensive Implementation Plan

> **Status**: Planning Document  
> **Date**: December 2024  
> **Scope**: Remaining Phase 3 deliverables (PR-24 through PR-27)

---

## Executive Summary

This document provides a comprehensive, detailed implementation plan for the remaining Hamilton Phase 3 work. Based on analysis of the current codebase, Wave 1 and Wave 2 have successfully delivered:

- **7 Native Targets**: risk_factors, coverage_functions, hotspots, subsystems, call_graph_views, export_jsonl, export_parquet
- **Core Infrastructure**: Native registry, auto driver mode, materializers, runner utilities, artifact materializers
- **Contract Parity**: Complete OutputContract definitions for key targets
- **Module Splitting**: Assets vs wrapper targets separation

The remaining work focuses on **advanced platform features** that transform CodeIntel's build system from "functional" to "best-in-class":

| PR | Feature | Priority | Complexity | Impact |
|----|---------|----------|------------|--------|
| PR-24 | Native Ingestion Refactor | High | High | Tool targets as Hamilton subgraphs |
| PR-25 | Asset Catalog + CLI | Medium | Medium | "What exists?" observability |
| PR-26 | Node-Level Telemetry | Medium | Medium | Fine-grained build profiling |
| PR-27 | Strict Contracts Mode | Low | Low | Contract enforcement + migration gate |

---

## Current State Analysis

### Completed Infrastructure (Wave 1 + Wave 2)

```
src/codeintel/build/hamilton/native/
├── __init__.py
├── registry.py              # NativeTargetSpec registry (7 targets)
├── outputs.py               # expected_datasets(), expected_artifacts()
├── materializer.py          # materialize_table(), materialize_tables()
├── runner.py                # should_skip_native_target(), create_*_record()
├── artifact_materializer.py # materialize_artifact() for file outputs
├── analytics/
│   ├── risk_factors.py      # Wave 1
│   ├── coverage_functions.py # Wave 2
│   ├── hotspots.py          # Wave 2
│   └── subsystems.py        # Wave 2
├── graphs/
│   └── call_graph_views.py  # Wave 2
└── export/
    ├── export_jsonl.py      # Wave 2
    └── export_parquet.py    # Wave 2
```

### Existing Build Tracking Tables

```sql
-- Already implemented in schemas.py
build.output_manifests  -- Target manifests for skip logic
build.runs              -- Build run tracking
build.run_targets       -- Per-target execution records

-- NOT YET IMPLEMENTED
build.run_nodes         -- Node-level telemetry (PR-26)
build.assets            -- Asset catalog (PR-25)
```

### Native Registry Status

```python
# Current NATIVE_TARGETS (7 targets)
NATIVE_TARGETS = (
    NativeTargetSpec("risk_factors", "...analytics.risk_factors"),
    NativeTargetSpec("coverage_functions", "...analytics.coverage_functions"),
    NativeTargetSpec("hotspots", "...analytics.hotspots"),
    NativeTargetSpec("subsystems", "...analytics.subsystems"),
    NativeTargetSpec("call_graph_views", "...graphs.call_graph_views"),
    NativeTargetSpec("export_jsonl", "...export.export_jsonl"),
    NativeTargetSpec("export_parquet", "...export.export_parquet"),
)
```

---

## PR-24: Native Ingestion Refactor — Tool Steps as Hamilton Subgraphs

### Problem Statement

Tool-based ingestion targets (SCIP, typing, tests_ingest, coverage_ingest) currently use wrapper execution, hiding their internal structure from the DAG. This prevents:

1. **Fine-grained observability**: Can't see tool execution vs parsing/ingest steps separately
2. **Partial caching**: Can't cache tool output independently from table materialization
3. **Parallel execution**: Can't parallelize independent tool steps

### Solution Architecture

Convert tool-based targets into Hamilton subgraphs with explicit separation of:

1. **Tool execution nodes** (`tool__<target>`) - Run external tools, produce ArtifactRef
2. **Parse/transform nodes** (`parse__<target>`) - Process tool output
3. **Materialize nodes** (`m__<target>`) - Write to DuckDB
4. **Target nodes** (`t__<target>`) - Orchestrate and return TargetRunRecord

### Implementation Details

#### 1. Tool Execution Abstraction

**New File**: `src/codeintel/build/hamilton/native/tools/__init__.py`

```python
"""Tool execution abstraction for native Hamilton targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.build.hamilton.io.artifact_ref import ArtifactRef

@dataclass(frozen=True)
class ToolExecutionSpec:
    """Specification for external tool execution."""
    
    tool_name: str
    command_args: tuple[str, ...]
    output_path: Path
    timeout_seconds: float = 300.0
    env_vars: dict[str, str] | None = None

@dataclass(frozen=True)
class ToolExecutionResult:
    """Result of external tool execution."""
    
    success: bool
    artifact: ArtifactRef | None
    duration_ms: float
    stdout: str
    stderr: str
    return_code: int
```

**New File**: `src/codeintel/build/hamilton/native/tools/executor.py`

```python
"""Tool execution for native Hamilton targets."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.native.tools import ToolExecutionResult, ToolExecutionSpec

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv

def execute_tool(
    spec: ToolExecutionSpec,
    env: BuildEnv,
) -> ToolExecutionResult:
    """Execute external tool and return result with ArtifactRef.
    
    Parameters
    ----------
    spec
        Tool execution specification.
    env
        Build environment with paths and snapshot.
        
    Returns
    -------
    ToolExecutionResult
        Execution result with artifact reference if successful.
    """
    start_time = time.perf_counter()
    
    try:
        result = subprocess.run(
            [spec.tool_name, *spec.command_args],
            capture_output=True,
            text=True,
            timeout=spec.timeout_seconds,
            env=spec.env_vars,
            cwd=str(env.paths.repo_root),
        )
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        artifact = None
        if result.returncode == 0 and spec.output_path.exists():
            artifact = ArtifactRef(
                name=spec.tool_name,
                artifact_type="file",
                repo=env.snapshot.repo,
                commit=env.snapshot.commit,
                path=str(spec.output_path),
            )
        
        return ToolExecutionResult(
            success=result.returncode == 0,
            artifact=artifact,
            duration_ms=duration_ms,
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
        )
        
    except subprocess.TimeoutExpired:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolExecutionResult(
            success=False,
            artifact=None,
            duration_ms=duration_ms,
            stdout="",
            stderr=f"Tool {spec.tool_name} timed out after {spec.timeout_seconds}s",
            return_code=-1,
        )
```

#### 2. SCIP Native Implementation (Primary Example)

**New File**: `src/codeintel/build/hamilton/native/ingestion/__init__.py`

```python
"""Native ingestion targets with tool execution subgraphs."""

from __future__ import annotations

__all__: list[str] = []
```

**New File**: `src/codeintel/build/hamilton/native/ingestion/scip.py`

```python
"""Native SCIP ingestion with Hamilton subgraph.

This module implements SCIP indexing as a native Hamilton pipeline with:
- tool__scip: Execute scip-python to generate index
- parse__scip: Parse SCIP index into tables
- m__scip: Materialize artifacts
- t__scip: Orchestrate execution and return TargetRunRecord
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.io.artifact_ref import ArtifactRef
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.outputs import expected_outputs
from codeintel.build.hamilton.native.runner import (
    create_failed_record,
    create_skipped_record,
    create_success_record,
    save_manifest,
    should_skip_native_target,
)
from codeintel.build.hamilton.native.tools import ToolExecutionResult, ToolExecutionSpec
from codeintel.build.hamilton.native.tools.executor import execute_tool
from codeintel.build.registry import get_target_graph

if TYPE_CHECKING:
    from pathlib import Path
    
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import TargetGraph


@tag(domain="ingestion", target="scip", node_kind="tool")
def tool__scip(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> ToolExecutionResult:
    """Execute scip-python tool to generate SCIP index.
    
    Parameters
    ----------
    env
        Build environment with paths and snapshot.
    t__modules
        Upstream modules target result (for dependency).
        
    Returns
    -------
    ToolExecutionResult
        Tool execution result with artifact reference.
    """
    if t__modules.status != "succeeded":
        return ToolExecutionResult(
            success=False,
            artifact=None,
            duration_ms=0.0,
            stdout="",
            stderr="Upstream modules target failed",
            return_code=-1,
        )
    
    output_path = env.paths.scip_dir / "index.scip"
    
    spec = ToolExecutionSpec(
        tool_name="scip-python",
        command_args=(
            "index",
            "--project-name", env.snapshot.repo,
            "--output", str(output_path),
            str(env.paths.repo_root),
        ),
        output_path=output_path,
        timeout_seconds=600.0,
    )
    
    return execute_tool(spec, env)


@tag(domain="ingestion", target="scip", node_kind="parse")
def parse__scip(
    env: BuildEnv,
    tool__scip: ToolExecutionResult,
) -> dict[str, object]:
    """Parse SCIP index into structured data.
    
    Parameters
    ----------
    env
        Build environment.
    tool__scip
        Tool execution result with SCIP index artifact.
        
    Returns
    -------
    dict[str, object]
        Parsed SCIP data for downstream processing.
    """
    if not tool__scip.success or tool__scip.artifact is None:
        return {"success": False, "error": tool__scip.stderr}
    
    # Parse SCIP index - implementation depends on existing SCIP parser
    # This is a placeholder for the actual parsing logic
    from codeintel.ingestion.scip.parser import parse_scip_index
    
    scip_path = tool__scip.artifact.path
    parsed = parse_scip_index(scip_path, env.snapshot)
    
    return {"success": True, "data": parsed}


@tag(domain="ingestion", target="scip", node_kind="target")
def t__scip(
    env: BuildEnv,
    graph: TargetGraph,
    tool__scip: ToolExecutionResult,
    parse__scip: dict[str, object],
    t__modules: TargetRunRecord,
) -> TargetRunRecord:
    """Orchestrate SCIP target execution.
    
    Parameters
    ----------
    env
        Build environment.
    graph
        Target graph for metadata.
    tool__scip
        Tool execution result.
    parse__scip
        Parsed SCIP data.
    t__modules
        Upstream modules result.
        
    Returns
    -------
    TargetRunRecord
        Complete target execution record.
    """
    start_time = time.perf_counter()
    target = graph.get("scip")
    
    # Check for upstream failure
    if t__modules.status != "succeeded":
        return TargetRunRecord(
            target="scip",
            plugin_name="native:scip",
            status="skipped",
            input_hash=None,
            options_hash=None,
            duration_ms=0.0,
            row_counts={},
            error="upstream_failed",
            datasets=(),
            artifacts=expected_outputs(target, env.snapshot),
        )
    
    # Check skip logic (simplified - actual implementation uses hash)
    # ... skip check implementation ...
    
    # Check tool execution
    if not tool__scip.success:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return create_failed_record(
            target=target,
            input_hash="",
            options_hash=None,
            duration_ms=duration_ms,
            error=Exception(tool__scip.stderr),
        )
    
    # Check parsing
    if not parse__scip.get("success"):
        duration_ms = (time.perf_counter() - start_time) * 1000
        return create_failed_record(
            target=target,
            input_hash="",
            options_hash=None,
            duration_ms=duration_ms,
            error=Exception(str(parse__scip.get("error", "Parse failed"))),
        )
    
    # Success case
    duration_ms = (time.perf_counter() - start_time) * 1000
    record = create_success_record(
        target=target,
        env=env,
        input_hash="",  # Computed from dependencies
        options_hash=None,
        duration_ms=duration_ms,
    )
    
    save_manifest(env, record)
    return record
```

#### 3. Typing Native Implementation

**New File**: `src/codeintel/build/hamilton/native/ingestion/typing.py`

Similar pattern to SCIP but with:
- `tool__typing__pyright`: Execute pyright for type checking
- `tool__typing__pyrefly`: Execute pyrefly for additional checks
- `tool__typing__ruff`: Execute ruff for static diagnostics
- `parse__typing`: Aggregate results into tables
- `t__typing`: Orchestrate and materialize

#### 4. Registry Updates

**Modified File**: `src/codeintel/build/hamilton/native/registry.py`

```python
# Add to NATIVE_TARGETS
NativeTargetSpec(
    target_name="scip",
    module_path="codeintel.build.hamilton.native.ingestion.scip",
),
NativeTargetSpec(
    target_name="typing",
    module_path="codeintel.build.hamilton.native.ingestion.typing",
),
```

### Tasks Checklist

- [ ] Create `native/tools/__init__.py` with ToolExecutionSpec and ToolExecutionResult
- [ ] Create `native/tools/executor.py` with execute_tool function
- [ ] Create `native/ingestion/__init__.py` package
- [ ] Implement `native/ingestion/scip.py` with tool/parse/target nodes
- [ ] Implement `native/ingestion/typing.py` with multi-tool pattern
- [ ] Update registry with new native targets
- [ ] Add artifact hash caching for tool outputs (optional)
- [ ] Add parallel tool execution hooks (optional)

### Tests Checklist

- [ ] `tests/build/hamilton/test_pr24_tool_executor.py`
  - Tool execution success case
  - Tool execution timeout case
  - Tool execution failure case
- [ ] `tests/build/hamilton/test_pr24_scip_native.py`
  - DAG shape includes tool/parse/target nodes
  - Upstream failure propagation
  - Artifact ref creation
- [ ] `tests/build/hamilton/test_pr24_typing_native.py`
  - Multi-tool orchestration
  - Result aggregation

### CLI Snapshots

Add to `manifest.yaml`:

```yaml
- name: "pr24_graph_scip_native"
  tags: ["pr24", "graph", "native", "mermaid", "ingestion"]
  args: ["build", "graph", "scip", "--format", "mermaid"]
  kind: "text"
  snapshot: "pr24_graph_scip_native.mmd"
```

---

## PR-25: Asset Catalog + "What Exists?" CLI

### Problem Statement

There is no unified view of what assets exist in the database for a given repo/commit. Users must query individual tables or use ad-hoc SQL to understand build state.

### Solution Architecture

Create a `build.assets` table and CLI commands that provide:

1. **Asset catalog**: Centralized record of all materialized assets
2. **"What exists?" CLI**: Quick visibility into current build state
3. **Lineage queries**: Which target produced which asset

### Implementation Details

#### 1. Schema Definition

**Modified File**: `src/codeintel/config/datasets/schemas.py`

Add after `build.run_targets`:

```python
"build.assets": TableSchema(
    schema="build",
    name="assets",
    columns=[
        Column("asset_key", "VARCHAR", nullable=False, description="Unique asset identifier"),
        Column("asset_type", "VARCHAR", nullable=False, description="table, view, or artifact"),
        Column("repo", "VARCHAR", nullable=False, description="Repository slug"),
        Column("commit", "VARCHAR", nullable=False, description="Commit SHA"),
        Column("owner_target", "VARCHAR", nullable=False, description="Target that produced this asset"),
        Column("schema_version", "VARCHAR", description="Schema version if applicable"),
        Column("row_count", "BIGINT", description="Row count for tables"),
        Column("file_size_bytes", "BIGINT", description="File size for artifacts"),
        Column("materialized_at", "TIMESTAMPTZ", nullable=False, description="When asset was created"),
        Column("input_hash", "VARCHAR", description="Input hash from manifest"),
        Column("metadata", "JSON", description="Additional metadata"),
    ],
    primary_key=("asset_key", "repo", "commit"),
    indexes=(
        Index("idx_build_assets_repo_commit", ("repo", "commit")),
        Index("idx_build_assets_owner_target", ("owner_target",)),
        Index("idx_build_assets_type", ("asset_type",)),
    ),
    description="Catalog of all materialized assets for observability",
),
```

#### 2. Asset Tracking Service

**New File**: `src/codeintel/storage/tracking/asset_tracking.py`

```python
"""Asset catalog tracking for build observability.

This module provides persistence and querying for the asset catalog,
enabling "what exists?" visibility into the build state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
from codeintel.storage.helpers.json import encode_json_compact

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.storage.gateway.protocol import StorageGateway


@dataclass(frozen=True)
class AssetRecord:
    """Record of a materialized asset."""
    
    asset_key: str
    asset_type: str  # "table", "view", "artifact"
    repo: str
    commit: str
    owner_target: str
    schema_version: str | None = None
    row_count: int | None = None
    file_size_bytes: int | None = None
    materialized_at: datetime | None = None
    input_hash: str | None = None
    metadata: dict[str, Any] | None = None


class AssetTracking:
    """Accessor for build asset catalog.
    
    Provides CRUD operations for the build.assets table,
    enabling observability into what has been materialized.
    """
    
    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize asset tracking accessor."""
        self._gateway = gateway
        self._con = gateway.con
        self._backend = DuckDBPolicyBackend(gateway)
    
    def record_asset(self, record: AssetRecord) -> None:
        """Record or update an asset in the catalog."""
        materialized_at = record.materialized_at or datetime.now(tz=UTC)
        metadata_json = encode_json_compact(record.metadata or {})
        
        self._con.execute(
            """
            INSERT OR REPLACE INTO build.assets (
                asset_key, asset_type, repo, commit, owner_target,
                schema_version, row_count, file_size_bytes,
                materialized_at, input_hash, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.asset_key,
                record.asset_type,
                record.repo,
                record.commit,
                record.owner_target,
                record.schema_version,
                record.row_count,
                record.file_size_bytes,
                materialized_at,
                record.input_hash,
                metadata_json,
            ],
        )
    
    def record_assets_batch(
        self,
        records: Sequence[AssetRecord],
    ) -> int:
        """Record multiple assets in a single batch."""
        if not records:
            return 0
        
        now = datetime.now(tz=UTC)
        rows = [
            (
                r.asset_key,
                r.asset_type,
                r.repo,
                r.commit,
                r.owner_target,
                r.schema_version,
                r.row_count,
                r.file_size_bytes,
                r.materialized_at or now,
                r.input_hash,
                encode_json_compact(r.metadata or {}),
            )
            for r in records
        ]
        
        return self._backend.upsert(
            "build.assets",
            rows,
            columns=(
                "asset_key", "asset_type", "repo", "commit", "owner_target",
                "schema_version", "row_count", "file_size_bytes",
                "materialized_at", "input_hash", "metadata",
            ),
            conflict_columns=("asset_key", "repo", "commit"),
            update_columns=(
                "asset_type", "owner_target", "schema_version", "row_count",
                "file_size_bytes", "materialized_at", "input_hash", "metadata",
            ),
        )
    
    def list_assets(
        self,
        repo: str,
        commit: str,
        *,
        asset_type: str | None = None,
        owner_target: str | None = None,
    ) -> list[AssetRecord]:
        """List assets for a repo/commit with optional filters."""
        query = """
            SELECT asset_key, asset_type, repo, commit, owner_target,
                   schema_version, row_count, file_size_bytes,
                   materialized_at, input_hash, metadata
            FROM build.assets
            WHERE repo = ? AND commit = ?
        """
        params: list[Any] = [repo, commit]
        
        if asset_type:
            query += " AND asset_type = ?"
            params.append(asset_type)
        
        if owner_target:
            query += " AND owner_target = ?"
            params.append(owner_target)
        
        query += " ORDER BY asset_key"
        
        results = self._con.execute(query, params).fetchall()
        return [self._parse_asset_row(row) for row in results]
    
    def _parse_asset_row(self, row: tuple[Any, ...]) -> AssetRecord:
        """Parse a DuckDB row into AssetRecord."""
        return AssetRecord(
            asset_key=str(row[0]),
            asset_type=str(row[1]),
            repo=str(row[2]),
            commit=str(row[3]),
            owner_target=str(row[4]),
            schema_version=str(row[5]) if row[5] else None,
            row_count=int(row[6]) if row[6] else None,
            file_size_bytes=int(row[7]) if row[7] else None,
            materialized_at=row[8],
            input_hash=str(row[9]) if row[9] else None,
            metadata=row[10] if row[10] else None,
        )
```

#### 3. CLI Commands

**Modified File**: `src/codeintel/cli/commands/build.py`

Add new command:

```python
@cli_command("build.assets", handler=build_assets_handler, config=_BUILD_CONFIG)
@build_app.command(name="assets")
@dataclass
class BuildAssetsCommand:
    """List materialized assets for the current snapshot."""
    
    target: Annotated[
        str | None,
        Parameter(
            name=["--target", "-t"],
            help="Filter to assets produced by a specific target.",
        ),
    ] = None
    asset_type: Annotated[
        str | None,
        Parameter(
            name=["--type"],
            help="Filter by asset type: table, view, or artifact.",
        ),
    ] = None
    output_format: Annotated[
        str,
        Parameter(
            name=["--format", "-f"],
            help="Output format: table (default), json, or csv.",
        ),
    ] = "table"
    flags: SharedFlags = field(default=SharedFlags(), metadata=SHARED_FLAGS_METADATA)
```

**New Handler**: `src/codeintel/cli/handlers/build.py`

```python
def build_assets_handler(ctx: CommandContext) -> CliResult[BuildAssetsResult]:
    """Handle build assets command."""
    gateway = ctx.require_gateway()
    runtime = ctx.require_runtime()
    
    target = ctx.params.get_str("target")
    asset_type = ctx.params.get_str("asset_type")
    output_format = ctx.params.get_str("output_format") or "table"
    
    assets = gateway.assets.list_assets(
        repo=runtime.snapshot.repo,
        commit=runtime.snapshot.commit,
        asset_type=asset_type,
        owner_target=target,
    )
    
    return CliResult.ok(
        BuildAssetsResult(
            assets=[a.to_dict() for a in assets],
            count=len(assets),
            format=output_format,
        )
    )
```

#### 4. Integration with Native Materializer

**Modified File**: `src/codeintel/build/hamilton/native/materializer.py`

Add asset recording after table materialization:

```python
def materialize_table(
    env: BuildEnv,
    table_key: str,
    expr: ir.Table,
    owner_target: str,
) -> int:
    """Materialize Ibis expression to DuckDB table.
    
    Also records the asset in the catalog for observability.
    """
    # ... existing materialization logic ...
    
    row_count = _execute_materialization(env.gateway, table_key, expr)
    
    # Record in asset catalog
    env.gateway.assets.record_asset(
        AssetRecord(
            asset_key=table_key,
            asset_type="table",
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
            owner_target=owner_target,
            row_count=row_count,
            input_hash=env.manifest_index.get(owner_target, {}).get("input_hash"),
        )
    )
    
    return row_count
```

### Tasks Checklist

- [ ] Add `build.assets` schema to `schemas.py`
- [ ] Create `storage/tracking/asset_tracking.py` with AssetTracking class
- [ ] Add `assets` accessor to StorageGateway
- [ ] Create `BuildAssetsCommand` CLI command
- [ ] Create `build_assets_handler` in handlers
- [ ] Integrate asset recording into `materializer.py`
- [ ] Integrate asset recording into `artifact_materializer.py`
- [ ] Add table bootstrap DDL for `build.assets`

### Tests Checklist

- [ ] `tests/build/hamilton/test_pr25_asset_catalog.py`
  - Asset recording and retrieval
  - Filter by target
  - Filter by asset type
- [ ] `tests/cli/test_pr25_assets_command.py`
  - CLI output formats (table, json, csv)
  - Filter flags

### CLI Snapshots

```yaml
- name: "pr25_assets_help"
  tags: ["pr25", "assets", "text", "tiny"]
  args: ["build", "assets", "--help"]
  kind: "text"
  snapshot: "pr25_assets_help.txt"
```

---

## PR-26: Node-Level Telemetry — build.run_nodes

### Problem Statement

With native targets expanding into subgraphs (compute, materialize, tool nodes), per-target telemetry (`build.run_targets`) is insufficient. Users need visibility into:

1. **Which internal nodes are slow?** - Identify bottlenecks within targets
2. **Where did failures occur?** - Pinpoint failure location in complex pipelines
3. **Cache hit rates** - Understand skip efficiency at node level

### Solution Architecture

Add `build.run_nodes` table and Hamilton execution hooks to record node-level telemetry.

### Implementation Details

#### 1. Schema Definition

**Modified File**: `src/codeintel/config/datasets/schemas.py`

```python
"build.run_nodes": TableSchema(
    schema="build",
    name="run_nodes",
    columns=[
        Column("run_id", "VARCHAR", nullable=False, description="Parent run identifier"),
        Column("node_name", "VARCHAR", nullable=False, description="Hamilton node name"),
        Column("target", "VARCHAR", description="Parent target if applicable"),
        Column("node_kind", "VARCHAR", description="Node kind: compute, materialize, tool, etc."),
        Column("status", "VARCHAR", nullable=False, description="succeeded, failed, skipped"),
        Column("started_at", "TIMESTAMPTZ", nullable=False),
        Column("completed_at", "TIMESTAMPTZ"),
        Column("duration_ms", "DOUBLE"),
        Column("error", "VARCHAR"),
        Column("tags", "JSON", description="Hamilton tags from node"),
    ],
    primary_key=("run_id", "node_name"),
    indexes=(
        Index("idx_build_run_nodes_run_id", ("run_id",)),
        Index("idx_build_run_nodes_target", ("target",)),
        Index("idx_build_run_nodes_status", ("status",)),
    ),
    description="Node-level execution telemetry for fine-grained profiling",
),
```

#### 2. Node Telemetry Hook

**New File**: `src/codeintel/build/hamilton/telemetry_hook.py`

```python
"""Hamilton execution hook for node-level telemetry.

This module provides a Hamilton adapter hook that records per-node
execution telemetry to build.run_nodes for profiling and debugging.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from hamilton.lifecycle.base import BasePostNodeExecute, BasePreNodeExecute

if TYPE_CHECKING:
    from hamilton.graph_types import HamiltonNode
    
    from codeintel.storage.gateway.protocol import StorageGateway


@dataclass
class NodeExecutionRecord:
    """Record of a single node execution."""
    
    run_id: str
    node_name: str
    target: str | None
    node_kind: str | None
    status: str
    started_at: datetime
    completed_at: datetime | None
    duration_ms: float | None
    error: str | None
    tags: dict[str, Any] | None


class NodeTelemetryHook(BasePreNodeExecute, BasePostNodeExecute):
    """Hamilton lifecycle hook for node telemetry.
    
    Records execution timing and status for each node to enable
    fine-grained build profiling.
    """
    
    def __init__(self, run_id: str, gateway: StorageGateway) -> None:
        """Initialize telemetry hook.
        
        Parameters
        ----------
        run_id
            Build run identifier for grouping.
        gateway
            Storage gateway for persistence.
        """
        self._run_id = run_id
        self._gateway = gateway
        self._node_starts: dict[str, datetime] = {}
        self._records: list[NodeExecutionRecord] = []
    
    def pre_node_execute(
        self,
        *,
        node_name: str,
        node_tags: dict[str, Any],
        node_kwargs: dict[str, Any],
        node_return_type: type,
        **kwargs: Any,
    ) -> None:
        """Record node execution start."""
        self._node_starts[node_name] = datetime.now(tz=UTC)
    
    def post_node_execute(
        self,
        *,
        node_name: str,
        node_tags: dict[str, Any],
        node_kwargs: dict[str, Any],
        node_return_type: type,
        result: Any,
        error: Exception | None,
        success: bool,
        **kwargs: Any,
    ) -> None:
        """Record node execution completion."""
        completed_at = datetime.now(tz=UTC)
        started_at = self._node_starts.pop(node_name, completed_at)
        duration_ms = (completed_at - started_at).total_seconds() * 1000
        
        record = NodeExecutionRecord(
            run_id=self._run_id,
            node_name=node_name,
            target=node_tags.get("target"),
            node_kind=node_tags.get("node_kind"),
            status="succeeded" if success else "failed",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=duration_ms,
            error=str(error) if error else None,
            tags=node_tags if node_tags else None,
        )
        
        self._records.append(record)
    
    def flush(self) -> int:
        """Persist all recorded telemetry and clear buffer.
        
        Returns
        -------
        int
            Number of records persisted.
        """
        if not self._records:
            return 0
        
        count = self._gateway.build.save_run_nodes(self._run_id, self._records)
        self._records.clear()
        return count
```

#### 3. Build Tracking Extension

**Modified File**: `src/codeintel/storage/tracking/build_tracking.py`

Add methods:

```python
def save_run_nodes(
    self,
    run_id: str,
    records: Sequence[NodeExecutionRecord],
) -> int:
    """Save node-level execution records for a build run.
    
    Parameters
    ----------
    run_id
        Parent run identifier.
    records
        Sequence of NodeExecutionRecord objects.
        
    Returns
    -------
    int
        Number of records inserted.
    """
    if not records:
        return 0
    
    rows = [
        (
            run_id,
            r.node_name,
            r.target,
            r.node_kind,
            r.status,
            r.started_at,
            r.completed_at,
            r.duration_ms,
            r.error,
            encode_json_compact(r.tags or {}),
        )
        for r in records
    ]
    
    return self._backend.bulk_insert(
        "build.run_nodes",
        rows,
        columns=(
            "run_id", "node_name", "target", "node_kind", "status",
            "started_at", "completed_at", "duration_ms", "error", "tags",
        ),
    )

def list_run_nodes(
    self,
    run_id: str,
    *,
    target: str | None = None,
) -> list[dict[str, Any]]:
    """List node records for a specific run.
    
    Parameters
    ----------
    run_id
        Run identifier to fetch nodes for.
    target
        Optional target filter.
        
    Returns
    -------
    list[dict[str, Any]]
        List of node record dictionaries.
    """
    query = """
        SELECT node_name, target, node_kind, status,
               started_at, completed_at, duration_ms, error, tags
        FROM build.run_nodes
        WHERE run_id = ?
    """
    params: list[Any] = [run_id]
    
    if target:
        query += " AND target = ?"
        params.append(target)
    
    query += " ORDER BY started_at"
    
    results = self._con.execute(query, params).fetchall()
    return [
        {
            "node_name": row[0],
            "target": row[1],
            "node_kind": row[2],
            "status": row[3],
            "started_at": row[4],
            "completed_at": row[5],
            "duration_ms": row[6],
            "error": row[7],
            "tags": row[8],
        }
        for row in results
    ]
```

#### 4. Executor Integration

**Modified File**: `src/codeintel/build/hamilton/executor.py`

Add hook registration:

```python
def run(
    self,
    env: BuildEnv,
    targets: list[str],
    *,
    enable_telemetry: bool = True,
) -> HamiltonExecutionResult:
    """Execute Hamilton build for requested targets.
    
    Parameters
    ----------
    enable_telemetry
        If True, record node-level telemetry to build.run_nodes.
    """
    run_id = self._generate_run_id()
    
    # Set up telemetry hook if enabled
    hooks = []
    telemetry_hook = None
    if enable_telemetry:
        from codeintel.build.hamilton.telemetry_hook import NodeTelemetryHook
        telemetry_hook = NodeTelemetryHook(run_id, env.gateway)
        hooks.append(telemetry_hook)
    
    # Execute with hooks
    result = self._driver.execute(
        final_vars=node_names,
        inputs=inputs,
        hooks=hooks,
    )
    
    # Flush telemetry
    if telemetry_hook:
        telemetry_hook.flush()
    
    return result
```

### Tasks Checklist

- [ ] Add `build.run_nodes` schema to `schemas.py`
- [ ] Create `telemetry_hook.py` with NodeTelemetryHook
- [ ] Add `save_run_nodes` and `list_run_nodes` to BuildTracking
- [ ] Integrate hook into HamiltonBuildExecutor
- [ ] Add `--enable-telemetry` flag to build run command
- [ ] Add node breakdown to `build history --run-id` output
- [ ] Add table bootstrap DDL for `build.run_nodes`

### Tests Checklist

- [ ] `tests/build/hamilton/test_pr26_node_telemetry.py`
  - Hook records pre/post execution
  - Records flushed to database
  - Target filtering works
- [ ] `tests/build/hamilton/test_pr26_history_nodes.py`
  - History command includes node breakdown
  - Node timing aggregation

### CLI Snapshots

```yaml
- name: "pr26_history_with_nodes"
  tags: ["pr26", "history", "json", "telemetry"]
  args: ["build", "history", "--help"]
  kind: "text"
  snapshot: "pr26_history_nodes_help.txt"
```

---

## PR-27: Strict Contracts Mode + Wrapper Deprecation Gate

### Problem Statement

Without enforcement, targets can write to tables outside their declared contract, causing:

1. **Silent drift**: Actual outputs diverge from documented contracts
2. **Lineage corruption**: Asset ownership becomes unclear
3. **Migration friction**: Hard to know when all targets are truly native

### Solution Architecture

Add `strict_contracts` mode that:

1. **Validates writes**: Fails target if it writes outside contract
2. **Tracks impl_kind**: Records wrapper vs native per execution
3. **Deprecation gate**: Warns/errors for wrapper targets in allowlist-only mode

### Implementation Details

#### 1. BuildEnv Extension

**Modified File**: `src/codeintel/build/hamilton/env.py`

```python
@dataclass(frozen=True)
class BuildEnv:
    """Bundled execution dependencies for Hamilton node execution."""
    
    # ... existing fields ...
    
    strict_contracts: bool = False
    wrapper_allowlist: frozenset[str] | None = None  # None = all allowed
```

#### 2. Contract Enforcement Context

**New File**: `src/codeintel/build/hamilton/contracts/enforcement.py`

```python
"""Contract enforcement for strict mode.

When strict_contracts is enabled, all writes are validated against
the target's declared OutputContract.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from codeintel.build.errors import ContractViolationError

if TYPE_CHECKING:
    from collections.abc import Iterator
    
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import OutputTarget


class ContractEnforcer:
    """Enforces write operations against target contracts.
    
    When active, intercepts table/artifact writes and validates
    they are within the current target's declared contract.
    """
    
    _current_target: OutputTarget | None = None
    _strict: bool = False
    
    @classmethod
    @contextmanager
    def for_target(
        cls,
        target: OutputTarget,
        *,
        strict: bool,
    ) -> Iterator[None]:
        """Context manager for contract enforcement during target execution.
        
        Parameters
        ----------
        target
            Target being executed.
        strict
            If True, raise on contract violations.
        """
        old_target = cls._current_target
        old_strict = cls._strict
        
        cls._current_target = target
        cls._strict = strict
        
        try:
            yield
        finally:
            cls._current_target = old_target
            cls._strict = old_strict
    
    @classmethod
    def validate_table_write(cls, table_key: str) -> None:
        """Validate that a table write is within contract.
        
        Parameters
        ----------
        table_key
            Table being written to.
            
        Raises
        ------
        ContractViolationError
            If strict mode and write is outside contract.
        """
        if not cls._strict or cls._current_target is None:
            return
        
        if table_key not in cls._current_target.table_keys:
            msg = (
                f"Target '{cls._current_target.name}' attempted to write "
                f"to '{table_key}' which is not in its contract. "
                f"Allowed tables: {cls._current_target.table_keys}"
            )
            raise ContractViolationError(msg)
    
    @classmethod
    def validate_artifact_write(cls, artifact_name: str) -> None:
        """Validate that an artifact write is within contract."""
        if not cls._strict or cls._current_target is None:
            return
        
        artifact_names = {a.name for a in cls._current_target.contract.artifacts}
        if artifact_name not in artifact_names:
            msg = (
                f"Target '{cls._current_target.name}' attempted to write "
                f"artifact '{artifact_name}' which is not in its contract."
            )
            raise ContractViolationError(msg)
```

#### 3. Wrapper Deprecation Validation

**Modified File**: `src/codeintel/build/hamilton/planner.py`

```python
def compute_plan(
    *,
    env: BuildEnv,
    graph: TargetGraph | None = None,
    requested: tuple[str, ...],
    mode: HamiltonNodeMode = "generated",
) -> HamiltonBuildPlan:
    """Compute build plan with wrapper deprecation checks."""
    # ... existing logic ...
    
    # Check wrapper deprecation
    if env.wrapper_allowlist is not None:
        for entry in entries:
            if entry.impl_kind == "wrapper" and entry.target not in env.wrapper_allowlist:
                # Mark as blocked or emit warning
                warnings.warn(
                    f"Target '{entry.target}' uses wrapper implementation "
                    f"but is not in allowlist. Consider migrating to native.",
                    DeprecationWarning,
                    stacklevel=2,
                )
```

#### 4. CLI Flags

**Modified File**: `src/codeintel/cli/commands/build.py`

```python
@dataclass
class BuildRunCommand:
    # ... existing fields ...
    
    strict_contracts: Annotated[
        bool,
        Parameter(
            name=["--strict-contracts"],
            help="Fail if target writes outside declared contract.",
            negative=(),
        ),
    ] = False
    
    wrapper_allowlist: Annotated[
        list[str] | None,
        Parameter(
            name=["--wrapper-allowlist"],
            help="Only allow wrapper implementation for these targets.",
        ),
    ] = None
```

### Tasks Checklist

- [ ] Add `strict_contracts` and `wrapper_allowlist` to BuildEnv
- [ ] Create `contracts/enforcement.py` with ContractEnforcer
- [ ] Integrate enforcer into materializer write operations
- [ ] Add deprecation warnings to planner for non-allowlisted wrappers
- [ ] Add CLI flags `--strict-contracts` and `--wrapper-allowlist`
- [ ] Create `ContractViolationError` in build errors
- [ ] Document migration path for strict mode adoption

### Tests Checklist

- [ ] `tests/build/hamilton/test_pr27_strict_contracts.py`
  - Write within contract succeeds
  - Write outside contract fails in strict mode
  - Non-strict mode allows all writes
- [ ] `tests/build/hamilton/test_pr27_wrapper_deprecation.py`
  - Allowlist enforcement
  - Deprecation warnings emitted
- [ ] `tests/build/hamilton/test_pr27_plan_impl_kind.py`
  - Plan correctly marks impl_kind for all targets

### CLI Snapshots

```yaml
- name: "pr27_run_strict_help"
  tags: ["pr27", "run", "text", "strict"]
  args: ["build", "run", "--help"]
  kind: "text"
  snapshot: "pr27_run_strict_help.txt"
```

---

## Implementation Sequence

### Recommended Order

```
Phase 3 Wave 3 Implementation Order:

1. PR-25: Asset Catalog (Foundation)
   └── Enables observability for all subsequent work
   
2. PR-26: Node Telemetry (Profiling)
   └── Required before PR-24 to debug tool execution
   
3. PR-24: Native Ingestion (Major Feature)
   └── Builds on observability from PR-25/26
   └── Highest complexity, highest impact
   
4. PR-27: Strict Contracts (Governance)
   └── Final gate before "native-only" migration
```

### Timeline Estimate

| PR | Estimated Effort | Dependencies |
|----|------------------|--------------|
| PR-25 | 2-3 days | None |
| PR-26 | 2-3 days | PR-25 (for asset recording) |
| PR-24 | 5-7 days | PR-26 (for telemetry) |
| PR-27 | 1-2 days | PR-24 (all native targets) |

**Total: ~12-15 days**

---

## Definition of Done (Wave 3)

### PR-24 Complete When:
- [ ] SCIP target runs as native Hamilton subgraph
- [ ] Typing target runs as native Hamilton subgraph
- [ ] Tool execution recorded separately from materialization
- [ ] All quality gates pass (ruff, pyright, pyrefly)
- [ ] Tests verify tool/parse/target node separation

### PR-25 Complete When:
- [ ] `build.assets` table created and populated
- [ ] `codeintel build assets` CLI works with filters
- [ ] Native materializer records assets automatically
- [ ] All quality gates pass

### PR-26 Complete When:
- [ ] `build.run_nodes` table created and populated
- [ ] NodeTelemetryHook integrated into executor
- [ ] `build history --run-id` shows node breakdown
- [ ] All quality gates pass

### PR-27 Complete When:
- [ ] Strict contracts mode available via flag
- [ ] Contract violations fail in strict mode
- [ ] Wrapper allowlist enforced with warnings
- [ ] All quality gates pass
- [ ] Documentation updated with migration guide

---

## Phase 3 Complete Definition of Done

By end of Phase 3, the following should be true:

1. **All key targets have native implementations** (ingestion, graphs, analytics, export)
2. **Tool execution is observable** via separate Hamilton nodes
3. **Asset catalog provides "what exists?" visibility**
4. **Node telemetry enables performance debugging**
5. **Strict contracts mode available for governance**
6. **Wrapper deprecation path documented and enforced**

---

## Appendix: File Inventory (Wave 3)

### New Files to Create

```
src/codeintel/build/hamilton/native/
├── tools/
│   ├── __init__.py          # ToolExecutionSpec, ToolExecutionResult
│   └── executor.py          # execute_tool()
├── ingestion/
│   ├── __init__.py          # Package init
│   ├── scip.py              # SCIP native target
│   └── typing.py            # Typing native target

src/codeintel/build/hamilton/
├── telemetry_hook.py        # NodeTelemetryHook
├── contracts/
│   └── enforcement.py       # ContractEnforcer

src/codeintel/storage/tracking/
└── asset_tracking.py        # AssetTracking

tests/build/hamilton/
├── test_pr24_tool_executor.py
├── test_pr24_scip_native.py
├── test_pr24_typing_native.py
├── test_pr25_asset_catalog.py
├── test_pr26_node_telemetry.py
├── test_pr27_strict_contracts.py
└── test_pr27_wrapper_deprecation.py
```

### Files to Modify

```
src/codeintel/config/datasets/schemas.py     # Add build.assets, build.run_nodes
src/codeintel/build/hamilton/native/registry.py  # Add scip, typing
src/codeintel/build/hamilton/env.py          # Add strict_contracts, wrapper_allowlist
src/codeintel/build/hamilton/executor.py     # Add telemetry hook integration
src/codeintel/build/hamilton/planner.py      # Add impl_kind population
src/codeintel/build/hamilton/native/materializer.py  # Add asset recording
src/codeintel/storage/tracking/build_tracking.py  # Add run_nodes methods
src/codeintel/cli/commands/build.py          # Add assets command, strict flags
src/codeintel/cli/handlers/build.py          # Add assets handler
tests/build/hamilton/snapshots/manifest.yaml # Add CLI snapshots
```

---

**End of Phase 3 Wave 3 Implementation Plan**

