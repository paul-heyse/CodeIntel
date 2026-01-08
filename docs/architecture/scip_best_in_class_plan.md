# SCIP Best-in-Class Implementation Plan

This plan translates the top five SCIP recommendations into concrete execution items. Each scope
item includes goal, rationale, implementation steps, representative code patterns, and target
files.

## 1. Align SCIP project root with repo root (rel_path normalization)

Status: Completed

Goal:
- Ensure `Document.relative_path` from SCIP matches repo-root relative paths used elsewhere
  (modules, file_line_index, scip_resolution joins).

Rationale:
- Current `resolve_target_base` prefers `repo_root/src`, so SCIP paths become `module.py` while the
  rest of the system uses `src/module.py`, leading to join misses and empty datasets.

Implementation:
1. Default SCIP project root to `repo_root` (no `repo_root/src` fallback).
2. Use `--target-only` to scope runs without altering the project root.
3. Rebase document + diagnostic paths when `Metadata.project_root` differs from `repo_root`.

Representative code pattern:
```python
from __future__ import annotations

from pathlib import Path

from codeintel.ingestion.scip.protobuf_parser import rebase_parsed_index


def rebase_index(parsed: ScipParsedIndex, repo_root: Path) -> ScipParsedIndex:
    return rebase_parsed_index(parsed, repo_root)
```

File targets:
- `src/codeintel/ingestion/scip/paths.py`
- `src/codeintel/ingestion/scip/protobuf_parser.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/ingestion/engine/scip.py`

## 2. External library resolution: explicit env JSON + pip preflight

Status: Completed

Goal:
- Make external symbol resolution deterministic under uv and avoid silent loss of
  `core.scip_external_symbols`.

Rationale:
- scip-python defaults to pip interrogation; uv environments can omit pip, causing external
  packages to be invisible unless an explicit `--environment` JSON is provided.

Implementation:
1. Added `ScipIngestOptions.environment_json` (optional path).
2. Passed `--environment` to scip-python when set.
3. Added pip preflight when `environment_json` is unset.
4. Documented + implemented `scripts/gen_scip_env.py` generator.

Representative code pattern:
```python
from __future__ import annotations

from pathlib import Path


def build_scip_python_args(
    *,
    target_base: Path,
    output_scip: Path,
    project_name: str,
    environment_json: Path | None,
    target_paths: list[str] | None = None,
) -> list[str]:
    args = [
        "index",
        str(target_base),
        "--output",
        str(output_scip),
        "--project-name",
        project_name,
    ]
    if environment_json is not None:
        args.extend(["--environment", str(environment_json)])
    for target_path in target_paths or []:
        args.extend(["--target-only", target_path])
    return args
```

File targets:
- `src/codeintel/build/hamilton/native/options/ingestion.py`
- `src/codeintel/ingestion/scip/cli.py`
- `src/codeintel/ingestion/engine/scip.py`
- `src/codeintel/ingestion/scip/incremental.py`
- `docs/python_library_reference/scip-python_environment_config.md`
- `scripts/gen_scip_env.py` (new helper)

## 3. Project identity contract (name + version + namespace)

Status: Completed

Goal:
- Stabilize SCIP symbol identity across runs and make project identity explicit.

Rationale:
- Only `--project-name` is wired today; the docs call for stable `(name, version, namespace)` and
  recommend using commit SHA or `_` for project version.

Implementation:
1. Added `project_version_mode`, `project_version_value`, and `project_namespace` to
   `ScipIngestOptions`, plus defaults in tools config.
2. Resolved project version/namespace in the ingestion pipeline and included them in options
   hashing + telemetry payloads.
3. Threaded `--project-version` and `--project-namespace` through the CLI argument builder and
   incremental runner.
4. Persisted project identity fields in run tracking records.

Representative code pattern:
```python
from __future__ import annotations


def resolve_project_version(
    *,
    mode: str,
    value: str | None,
    commit: str,
) -> str | None:
    if mode == "commit":
        return commit
    if mode == "constant" and value:
        return value
    return None
```

File targets:
- `src/codeintel/build/hamilton/native/options/ingestion.py`
- `src/codeintel/ingestion/scip/cli.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/config/models.py`
- `docs/python_library_reference/scip_python_overview.md`
- `src/codeintel/ingestion/scip/telemetry.py`
- `src/codeintel/ingestion/engine/scip.py`
- `src/codeintel/ingestion/engine/service.py`
- `src/codeintel/storage/tracking/build_tracking.py`
- `src/codeintel/core/schemas/table_registry.py`
- `src/codeintel/core/gateway.py`

## 4. Enforce scope_paths for full rebuilds

Status: Completed

Goal:
- Respect `scope_paths` even when a full rebuild is triggered so SCIP output matches
  intended scope.

Rationale:
- `scope_paths` is used only for incremental planning. When a full rebuild is triggered,
  scip-python is invoked with no path constraints.

Implementation:
1. Converted `scope_paths` to `--target-only` arguments.
2. Passed scoped targets for full rebuilds via the incremental runner.

Representative code pattern:
```python
from __future__ import annotations

from collections.abc import Sequence


def normalize_scope_paths(scope_paths: Sequence[str] | None) -> list[str]:
    if not scope_paths:
        return []
    return [path.strip("/").replace("\\", "/") for path in scope_paths if path]
```

File targets:
- `src/codeintel/ingestion/scip/incremental.py`
- `src/codeintel/ingestion/scip/cli.py`

## 5. Occurrence fidelity (syntax_kind, enclosing_range, override docs)

Status: Completed

Goal:
- Preserve SCIP occurrence details needed for best-in-class navigation and overlays.

Rationale:
- The parser drops `syntax_kind`, `enclosing_range`, and `override_documentation`, which
  makes it impossible to reconstruct full definition spans or fine-grained syntax classes.

Implementation:
1. Extended occurrence models to include syntax kind, enclosing range, and override docs.
2. Parsed the new fields from protobuf and propagated them through ingestion adapters.
3. Added schema columns to `core.scip_occurrences` and emitted them in row builders.
4. Updated `scip_resolution` to prefer enclosing ranges and coalesce override docs.

Representative code pattern:
```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScipOccurrence:
    symbol: str
    range_start_line: int
    range_start_col: int
    range_end_line: int
    range_end_col: int
    symbol_roles: int
    syntax_kind: int | None = None
    enclosing_start_line: int | None = None
    enclosing_start_col: int | None = None
    enclosing_end_line: int | None = None
    enclosing_end_col: int | None = None
    override_documentation: str | None = None
```

File targets:
- `src/codeintel/ingestion/ports/tools.py`
- `src/codeintel/ingestion/scip/protobuf_parser.py`
- `src/codeintel/ingestion/scip/rows.py`
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`
- `src/codeintel/ingestion/engine/results.py`
- `src/codeintel/ingestion/engine/scip.py`
- `src/codeintel/ingestion/adapters/tool_runner.py`
- `src/codeintel/core/data_models/rows.py`
