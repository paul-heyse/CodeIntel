# SCIP Best-in-Class Implementation Plan

This plan translates the top five SCIP recommendations into concrete execution items. Each scope
item includes goal, rationale, implementation steps, representative code patterns, and target
files.

## 1. Align SCIP project root with repo root (rel_path normalization)

Goal:
- Ensure `Document.relative_path` from SCIP matches repo-root relative paths used elsewhere
  (modules, file_line_index, scip_resolution joins).

Rationale:
- Current `resolve_target_base` prefers `repo_root/src`, so SCIP paths become `module.py` while the
  rest of the system uses `src/module.py`, leading to join misses and empty datasets.

Plan:
1. Default SCIP project root to `repo_root` instead of `repo_root/src`.
2. Pass `--target-only` for scoped runs instead of changing the project root.
3. Add an ingest-time rebase guard: if metadata `project_root` is not under `repo_root`,
   normalize paths so they remain repo-root relative.

Representative code pattern:
```python
from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse


def project_root_to_path(project_root: str | None) -> Path | None:
    if not project_root:
        return None
    parsed = urlparse(project_root)
    raw = parsed.path if parsed.scheme in {"file", ""} else project_root
    return Path(unquote(raw))


def rebase_scip_path(
    *,
    rel_path: str,
    project_root: Path | None,
    repo_root: Path,
) -> str:
    if project_root is None:
        return rel_path
    abs_path = project_root / rel_path
    try:
        return abs_path.relative_to(repo_root).as_posix()
    except ValueError:
        return rel_path
```

File targets:
- `src/codeintel/ingestion/scip/paths.py`
- `src/codeintel/ingestion/scip/protobuf_parser.py`
- `src/codeintel/ingestion/scip/rows.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`

## 2. External library resolution: explicit env JSON + pip preflight

Goal:
- Make external symbol resolution deterministic under uv and avoid silent loss of
  `core.scip_external_symbols`.

Rationale:
- scip-python defaults to pip interrogation; uv environments can omit pip, causing external
  packages to be invisible unless an explicit `--environment` JSON is provided.

Plan:
1. Add `ScipIngestOptions.environment_json` (optional path).
2. When set, pass `--environment` to scip-python.
3. Add a preflight: if `environment_json` is unset and pip is missing, emit a clear error with
   remediation (add pip to dev deps or generate env JSON).
4. Document the canonical env JSON generator script and wire it into the build profile.

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
    rel_paths: list[str] | None = None,
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
    for rel_path in rel_paths or []:
        args.extend(["--target-only", rel_path])
    return args
```

File targets:
- `src/codeintel/build/hamilton/native/options/ingestion.py`
- `src/codeintel/ingestion/scip/cli.py`
- `src/codeintel/ingestion/engine/scip.py`
- `docs/python_library_reference/scip-python_environment_config.md`
- `scripts/gen_scip_env.py` (new helper)

## 3. Project identity contract (name + version + namespace)

Goal:
- Stabilize SCIP symbol identity across runs and make project identity explicit.

Rationale:
- Only `--project-name` is wired today; the docs call for stable `(name, version, namespace)` and
  recommend using commit SHA or `_` for project version.

Plan:
1. Add `ScipIngestOptions.project_version_mode` (values: `commit`, `constant`, `unset`) and
   `project_version_value` for constant mode.
2. Add `ScipIngestOptions.project_namespace`.
3. Pass `--project-version` and `--project-namespace` when configured.
4. Include `project_version` and `project_namespace` in the options hash and telemetry payload.

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

## 4. Enforce scope_paths for full rebuilds

Goal:
- Respect `scope_paths` even when a full rebuild is triggered so SCIP output matches
  intended scope.

Rationale:
- `scope_paths` is used only for incremental planning. When a full rebuild is triggered,
  scip-python is invoked with no path constraints.

Plan:
1. Convert `scope_paths` to `--target-only` arguments.
2. Ensure `scope_paths` are passed for both incremental and full runs.
3. Add unit coverage for scoped full rebuild behavior.

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
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `tests/ingestion` (add coverage for scoped rebuild)

## 5. Occurrence fidelity (syntax_kind, enclosing_range, override docs)

Goal:
- Preserve SCIP occurrence details needed for best-in-class navigation and overlays.

Rationale:
- The parser drops `syntax_kind`, `enclosing_range`, and `override_documentation`, which
  makes it impossible to reconstruct full definition spans or fine-grained syntax classes.

Plan:
1. Extend `ScipOccurrence` with `syntax_kind`, `enclosing_range_*`, and
   `override_documentation`.
2. Parse these fields from protobuf.
3. Add columns to `core.scip_occurrences` and materialize them in row builders.
4. Update downstream joins (e.g., `scip_resolution`) to prefer `enclosing_range` when present.

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
