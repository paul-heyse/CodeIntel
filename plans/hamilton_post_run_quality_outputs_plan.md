# Hamilton Post-Run Quality Outputs Plan

This plan migrates three quality/validation outputs out of the Hamilton DAG into
post-run dataset emission:
- `analytics.graph_validation`
- `analytics.scip_diagnostics_summary`, `analytics.scip_diagnostics_by_file`,
  `analytics.scip_diagnostics_top_messages`
- `analytics.py_cpg_quality_report`

The goal is to keep dataset delivery (parquet + manifest + schema hash) intact,
while reducing DAG surface area to "core analysis outputs" and pushing
non-critical diagnostics to post-run.

Each scope item below includes representative code patterns and a target file list.

## 0. Shared post-run quality output runner

Status: Planned

Goal:
- Provide a single post-run entry point that emits quality datasets after DAG
  execution using only persisted datasets and manifests.
- Centralize dataset writing behavior (schema hash + manifest extras) so the
  migrated outputs match existing delivery behavior.

Implementation:
1. Add a new module (or extend the diagnostics module) with helpers:
   - `_write_dataset_table(...)` for consistent dataset persistence.
   - `_scan_snapshot_reader(...)` for scoped dataset scanning.
   - `persist_post_run_quality_outputs(...)` to emit the three datasets.
2. Wire the post-run emitter into `_finalize_run` in
   `src/codeintel/build/hamilton/executor.py` after existing dataset persistence
   (`persist_contract_alignment_summary`, `persist_empty_dataset_issues`).
3. Reuse `schema_hash`, `ArrowDatasetWriteOptions`, and
   `TableSchema.to_json_obj()` for manifest extras.

Representative code pattern:
```python
from __future__ import annotations

from collections.abc import Iterable, Sequence

import pyarrow as pa

from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.core.datasets.scanning import ParquetScanOptions, scan_parquet_dataset
from codeintel.core.schemas.hashing import schema_hash
from codeintel.core.schemas.service import get_schema_service


def _scan_snapshot_reader(
    *,
    env: BuildEnv,
    table_key: str,
    columns: Sequence[str] | None,
) -> pa.RecordBatchReader | None:
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None:
        return None
    return scan_parquet_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=env.commit,
        options=ParquetScanOptions(
            columns=columns,
            repo=env.repo,
            commit=env.commit,
        ),
    )


def _write_dataset_table(
    *,
    env: BuildEnv,
    table_key: str,
    rows: Iterable[dict[str, object]],
) -> bool:
    dataset_root = env.paths.dataset_root_dir
    if dataset_root is None or not env.commit.strip():
        return False
    rows_list = list(rows)
    if rows_list:
        table, _ = table_for_rows(table_key, rows_list)
    else:
        table = empty_table_for_table(table_key)
    table_schema = get_schema_service().require_table_schema(table_key)
    options = ArrowDatasetWriteOptions(
        partition_columns=_partition_columns_for_schema(table_schema),
        schema_hash=schema_hash(table_schema),
        manifest_extras={"table_schema": table_schema.to_json_obj()},
    )
    write_dataset(
        dataset_root=dataset_root,
        table_key=table_key,
        snapshot_id=env.commit,
        data=table,
        options=options,
    )
    return True
```

File targets:
- `src/codeintel/build/hamilton/post_run_quality_outputs.py` (new)
- `src/codeintel/build/hamilton/executor.py` (invoke post-run emitter)
- `src/codeintel/build/hamilton/diagnostics.py` (optional: shared helpers)

## 1. Migrate analytics.graph_validation

Status: Planned

Goal:
- Compute graph validation rows post-run from persisted datasets and write the
  `analytics.graph_validation` dataset outside the DAG.

Implementation:
1. Extract a helper that runs `run_graph_validations_with_runner` and builds
   rows via `GraphValidationReporter`.
2. Use the shared `_write_dataset_table` helper to persist the table.
3. Remove the DAG target module and its import/export references.

Representative code pattern:
```python
from __future__ import annotations

from codeintel.build.graphs.runtime import GraphRuntimeOptions
from codeintel.build.graphs.validation.runner import (
    GraphValidationRunRequest,
    run_graph_validations_with_runner,
)
from codeintel.core.validation.reporters import GraphValidationReporter

GRAPH_VALIDATION_TABLE_KEY = "analytics.graph_validation"


def _graph_validation_rows(env: BuildEnv) -> list[dict[str, object]]:
    request = GraphValidationRunRequest(
        snapshot=env.snapshot,
        runtime=GraphRuntimeOptions(
            snapshot=env.snapshot,
            dataset_root_dir=env.paths.dataset_root_dir,
        ),
        dataset_root_dir=env.paths.dataset_root_dir,
    )
    report = run_graph_validations_with_runner(request=request)
    reporter = GraphValidationReporter(repo=env.repo, commit=env.commit)
    _report_findings(reporter, report.findings)
    return reporter.to_rows()
```

File targets:
- `src/codeintel/build/hamilton/post_run_quality_outputs.py` (new post-run emitter)
- `src/codeintel/build/hamilton/native/analytics/graph_validation.py` (remove)
- `src/codeintel/build/hamilton/native/analytics/__init__.py` (remove imports + __all__)

## 2. Migrate SCIP diagnostics rollups

Status: Planned

Goal:
- Build `analytics.scip_diagnostics_summary`, `analytics.scip_diagnostics_by_file`,
  and `analytics.scip_diagnostics_top_messages` post-run from
  `core.scip_diagnostics` dataset rows.

Implementation:
1. Move rollup helper logic (counts + row builders) into a non-DAG module,
   e.g., `src/codeintel/build/analytics/scip_diagnostics_rollups.py`.
2. Scan `core.scip_diagnostics` with `scan_parquet_dataset` scoped to repo/commit.
3. Compute summary/by-file/top-message rows and write each dataset using
   `_write_dataset_table`.
4. Remove the DAG target module and its import/export references.

Representative code pattern:
```python
from __future__ import annotations

from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.core.datasets.scanning import ParquetScanOptions, scan_parquet_dataset

SCIP_DIAGNOSTICS_TABLE_KEY = "core.scip_diagnostics"


def _scip_diagnostics_rows(env: BuildEnv) -> list[dict[str, object]]:
    reader = scan_parquet_dataset(
        dataset_root=env.paths.dataset_root_dir,
        table_key=SCIP_DIAGNOSTICS_TABLE_KEY,
        snapshot_id=env.commit,
        options=ParquetScanOptions(
            columns=("repo", "commit", "rel_path", "severity", "source", "code", "message"),
            repo=env.repo,
            commit=env.commit,
        ),
    )
    if reader is None:
        return []
    rows: list[dict[str, object]] = []
    for batch in reader.to_batches():
        rows.extend(iter_rows(batch))
    return rows
```

File targets:
- `src/codeintel/build/analytics/scip_diagnostics_rollups.py` (new or refactor target)
- `src/codeintel/build/hamilton/post_run_quality_outputs.py` (emit rollups)
- `src/codeintel/build/hamilton/native/analytics/scip_diagnostics.py` (remove)
- `src/codeintel/build/hamilton/native/analytics/__init__.py` (remove imports + __all__)

## 3. Migrate Python CPG quality report

Status: Planned

Goal:
- Compute `analytics.py_cpg_quality_report` post-run using persisted datasets:
  `core.py_bc_instructions`, `core.py_sym_scopes`, `core.py_bc_blocks`,
  `core.py_inspect_objects`, `core.py_bc_cfg_edges`,
  `core.py_bc_defuse_events`, `graph.cpg_edges`.

Implementation:
1. Move compute helpers (`_anchor_rate`, `_cfg_reachability`, `_scan_cpg_edges`)
   into a non-DAG module, e.g., `src/codeintel/build/analytics/py_cpg_quality_report.py`.
2. Scan the required datasets with `scan_parquet_dataset` scoped to repo/commit.
3. Convert readers to tables or iterate batches for counts, then build a
   single-row dataset with computed metrics.
4. Remove the DAG target module and its import/export references.

Representative code pattern:
```python
from __future__ import annotations

from codeintel.core.datasets.scanning import ParquetScanOptions, scan_parquet_dataset

PY_CPG_QUALITY_REPORT_TABLE_KEY = "analytics.py_cpg_quality_report"


def _scan_table(env: BuildEnv, table_key: str) -> pa.Table | None:
    reader = scan_parquet_dataset(
        dataset_root=env.paths.dataset_root_dir,
        table_key=table_key,
        snapshot_id=env.commit,
        options=ParquetScanOptions(
            columns=None,
            repo=env.repo,
            commit=env.commit,
        ),
    )
    if reader is None:
        return None
    return reader.read_all()


def _py_cpg_quality_rows(env: BuildEnv) -> list[dict[str, object]]:
    instructions = _scan_table(env, "core.py_bc_instructions")
    scopes = _scan_table(env, "core.py_sym_scopes")
    blocks = _scan_table(env, "core.py_bc_blocks")
    inspect_objects = _scan_table(env, "core.py_inspect_objects")
    cfg_edges = _scan_table(env, "core.py_bc_cfg_edges")
    defuse_events = _scan_table(env, "core.py_bc_defuse_events")
    cpg_edges = _scan_table(env, "graph.cpg_edges")
    if None in (
        instructions,
        scopes,
        blocks,
        inspect_objects,
        cfg_edges,
        defuse_events,
        cpg_edges,
    ):
        return []
    return _build_quality_rows(
        env=env,
        instructions=instructions,
        scopes=scopes,
        blocks=blocks,
        inspect_objects=inspect_objects,
        cfg_edges=cfg_edges,
        defuse_events=defuse_events,
        cpg_edges=cpg_edges,
    )
```

File targets:
- `src/codeintel/build/analytics/py_cpg_quality_report.py` (new or refactor target)
- `src/codeintel/build/hamilton/post_run_quality_outputs.py` (emit report)
- `src/codeintel/build/hamilton/native/analytics/py_cpg_quality_report.py` (remove)
- `src/codeintel/build/hamilton/native/analytics/__init__.py` (remove imports + __all__)

## 4. DAG cleanup and registry alignment

Status: Planned

Goal:
- Remove the three DAG targets and confirm no downstream DAG references remain.
- Keep `TableSchema` entries intact (post-run writer still relies on them).

Implementation:
1. Remove the three native analytics modules listed above.
2. Update `src/codeintel/build/hamilton/native/analytics/__init__.py` to drop
   imports and `__all__` entries.
3. Optionally trim any runtime schema override logic that only exists for the
   DAG targets (e.g., in `src/codeintel/runtime/compose.py`) if it is no longer
   required by DAG composition.

Representative code pattern:
```python
# __init__.py cleanup (remove target exports)
__all__ = [
    # ... keep existing analytics exports ...
    # "graph_validation__base",
    # "py_cpg_quality_report__base",
    # "scip_diagnostics__rollups",
]
```

File targets:
- `src/codeintel/build/hamilton/native/analytics/__init__.py`
- `src/codeintel/build/hamilton/native/analytics/graph_validation.py` (remove)
- `src/codeintel/build/hamilton/native/analytics/scip_diagnostics.py` (remove)
- `src/codeintel/build/hamilton/native/analytics/py_cpg_quality_report.py` (remove)
- `src/codeintel/runtime/compose.py` (optional cleanup if safe)

## 5. Verification & Validation

Status: Planned

Checklist:
1. Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
2. Run targeted tests covering graph validation and SCIP diagnostics rollups:
   - `tests/graphs/test_validation.py`
   - `tests/docs_export/test_graph_validation_export.py`
3. Run full build once to ensure post-run outputs appear under dataset root
   and manifests are created.

