
Here’s a full Epic 11 implementation plan, aligned with your *current* codebase (with unified pipeline, op_planner, serving bootstrap, etc.) and the Epic 11 thinking you captured. 

I’ll structure it roughly as:

* 11.1 CLI structure (Typer app + command groups)
* 11.2 Project config (`codeintel.yaml`) & wiring
* 11.3 CLI tests

where each section includes concrete **code snippets** you can drop into the repo.

---

## 11.1 CLI structure – app-first, not module-first

### 11.1.1 New top-level package

Add a new top-level package:

* `src/codeintel/cli/__init__.py`
* `src/codeintel/cli/main.py`

You already have `codeintel.pipeline.cli.main` as a v1/module-first CLI; Epic 11 introduces a **new** app-first CLI around the unified pipeline & serving stack. 

#### `codeintel/cli/__init__.py`

```python
# src/codeintel/cli/__init__.py
from __future__ import annotations

import typer

from codeintel.cli.main import pipeline_app, op_app, dataset_app, serve_app

app = typer.Typer(help="CodeIntel application CLI")
app.add_typer(pipeline_app, name="pipeline", help="Pipeline orchestration commands")
app.add_typer(op_app, name="op", help="Operation catalog and calls")
app.add_typer(dataset_app, name="dataset", help="Dataset introspection and validation")
app.add_typer(serve_app, name="serve", help="Serve HTTP/MCP APIs")

def main() -> None:
    """Entry point for the `codeintel` CLI."""
    app()

__all__ = ["app", "main"]
```

Then wire the console entrypoint (e.g. in `pyproject.toml`):

```toml
[project.scripts]
codeintel = "codeintel.cli:main"
```

> This leaves the old pipeline CLI (`codeintel-pipeline` or similar) untouched and introduces `codeintel` as the “real app” entrypoint. 

---

### 11.1.2 Shared CLI helpers – project context

We need a small helper module to:

* discover `codeintel.yaml`
* build `SnapshotRef`, `BuildPaths`, `StorageGateway`
* provide `ToolsConfig`, `ServingConfig` and `QueryService` when needed.

Create `src/codeintel/cli/project.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import typer
import yaml

from codeintel.config import CliPathsInput, CodeIntelConfig, RepoConfig, ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.pipeline import run_pipeline, FULL_PIPELINE, build_pipeline_for_operation, ensure_prerequisites_for_operation
from codeintel.storage.config import StorageConfig
from codeintel.storage.gateway import StorageGateway, open_gateway

if TYPE_CHECKING:
    from codeintel.serving.services.query_service import QueryService
    from codeintel.serving.bootstrap import ServiceStack
```

#### Project config models (YAML mapping)

```python
from pydantic import BaseModel, Field

class IngestProjectConfig(BaseModel):
    recipe: str = Field("builtin.default", description="Ingestion recipe name")

class AnalyticsProjectConfig(BaseModel):
    profile: str = Field("full", description="Analytics profile name (reserved)")

class GraphsProjectConfig(BaseModel):
    recipe: str = Field("builtin.full", description="Graphs recipe name")

class StorageProjectConfig(BaseModel):
    db_path: Path = Field(Path(".codeintel/duckdb.db"), description="DuckDB path")

class ProjectConfig(BaseModel):
    """
    Project-level configuration loaded from codeintel.yaml.

    Example YAML
    ------------
    repo: github.com/org/repo
    default_profile: default
    ingest:
      recipe: builtin.default
    analytics:
      profile: full
    graphs:
      recipe: builtin.full
    storage:
      db_path: .codeintel/duckdb.db
    """

    repo: str
    default_profile: str = "default"
    ingest: IngestProjectConfig = IngestProjectConfig()
    analytics: AnalyticsProjectConfig = AnalyticsProjectConfig()
    graphs: GraphsProjectConfig = GraphsProjectConfig()
    storage: StorageProjectConfig = StorageProjectConfig()
```

#### Project discovery: `find_project_root` & `load_project_config`

```python
PROJECT_FILE = "codeintel.yaml"

def find_project_root(start: Path | None = None) -> Path:
    """
    Walk upwards from start (or CWD) to find codeintel.yaml.

    Raises
    ------
    RuntimeError
        If no project file is found.
    """
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / PROJECT_FILE
        if candidate.is_file():
            return parent
    message = f"Could not find {PROJECT_FILE} starting from {current}"
    raise RuntimeError(message)

def load_project_config(root: Path | None = None) -> ProjectConfig:
    """
    Load ProjectConfig from codeintel.yaml at the given root.

    Raises
    ------
    RuntimeError
        If the project file cannot be parsed.
    """
    root = root or find_project_root()
    path = root / PROJECT_FILE
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError as exc:
        msg = f"Project file {PROJECT_FILE} not found at {root}"
        raise RuntimeError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"Failed to parse {PROJECT_FILE}: {exc}"
        raise RuntimeError(msg) from exc
    return ProjectConfig.model_validate(raw)
```

#### Build CLI context: snapshot, paths, gateway, tools, serving config

```python
@dataclass(frozen=True)
class ProjectRuntime:
    """Runtime wiring derived from project config and current repo state."""

    root: Path
    project: ProjectConfig
    cfg: CodeIntelConfig
    snapshot: SnapshotRef
    paths: BuildPaths
    gateway: StorageGateway
    tools: ToolsConfig
    serving: ServingConfig

def detect_commit(root: Path) -> str:
    """
    Detect current commit (best-effort).

    Tries CODEINTEL_COMMIT env, then `git rev-parse HEAD`, then 'HEAD'.
    """
    env_commit = os.environ.get("CODEINTEL_COMMIT")
    if env_commit:
        return env_commit
    git_dir = root / ".git"
    if git_dir.exists():
        try:
            import subprocess

            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if out:
                return out
        except Exception:
            pass
    return "HEAD"

def build_project_runtime(root: Path | None = None) -> ProjectRuntime:
    """
    Build runtime context from project config and environment.

    Used by all CLI commands to construct SnapshotRef, BuildPaths,
    StorageGateway, ToolsConfig, and ServingConfig.
    """
    root = find_project_root(root)
    project = load_project_config(root)

    commit = detect_commit(root)
    repo_cfg = RepoConfig(repo=project.repo, commit=commit)
    paths_cfg = CliPathsInput(
        repo_root=root,
        build_dir=root / ".codeintel",
        db_path=(root / project.storage.db_path),
        document_output_dir=None,
    )
    cfg = CodeIntelConfig.from_cli_args(
        repo_cfg=repo_cfg,
        paths_cfg=paths_cfg,
        options=None,
    )

    snapshot = SnapshotRef(
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        repo_root=cfg.paths.repo_root,
    )
    paths = cfg.build_paths

    storage_cfg = StorageConfig.for_ingest(db_path=paths.db_path)
    gateway = open_gateway(storage_cfg)

    tools = cfg.tools

    serving = ServingConfig(
        mode="local_db",
        repo_root=cfg.paths.repo_root,
        repo=cfg.repo.repo,
        commit=cfg.repo.commit,
        db_path=paths.db_path,
        read_only=True,
    )

    return ProjectRuntime(
        root=root,
        project=project,
        cfg=cfg,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=tools,
        serving=serving,
    )
```

We’ll reuse this `ProjectRuntime` in all command groups.

---

### 11.1.3 `codeintel pipeline` commands

Define a sub-app in `codeintel/cli/main.py`:

```python
# src/codeintel/cli/main.py
from __future__ import annotations

import json
from typing import Optional

import typer

from codeintel.cli.project import ProjectRuntime, build_project_runtime
from codeintel.pipeline import (
    FULL_PIPELINE,
    build_pipeline_for_operation,
    ensure_prerequisites_for_operation,
    run_pipeline,
)
from codeintel.pipeline.run_registry import PipelineRunRecord
from codeintel.storage.run_tracking import PipelineRunTracking

pipeline_app = typer.Typer(help="Pipeline orchestration commands")
op_app = typer.Typer(help="Operation catalog and calls")
dataset_app = typer.Typer(help="Dataset introspection commands")
serve_app = typer.Typer(help="HTTP/MCP serving")
```

#### `codeintel pipeline run full`

```python
@pipeline_app.command("run-full")
def pipeline_run_full(
    repo_root: Optional[str] = typer.Option(
        None,
        help="Path to repo root (defaults to nearest codeintel.yaml).",
    ),
) -> None:
    """
    Run the full pipeline (ingestion + graphs + analytics) for the project.

    This builds the project runtime from codeintel.yaml, executes the FULL_PIPELINE
    spec, and prints a summary including run_id and status.
    """
    runtime = build_project_runtime(Path(repo_root) if repo_root else None)

    result = run_pipeline(
        spec=FULL_PIPELINE,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        gateway=runtime.gateway,
        tools=runtime.tools,
        trigger="cli",
    )

    typer.echo(f"Run ID: {result.run_id}")
    typer.echo(f"Status: {result.status}")
```

#### `codeintel pipeline run op <operation-id>`

We want operation-driven orchestration: basically `RunKind="op_prereqs"` behind the scenes. You already implemented `ensure_prerequisites_for_operation` in `pipeline.op_planner`. 

```python
@pipeline_app.command("run-op")
def pipeline_run_op(
    op_id: str = typer.Argument(..., help="Operation ID (e.g. 'function.summary')"),
    repo_root: Optional[str] = typer.Option(
        None, help="Repo root (defaults to nearest codeintel.yaml)"
    ),
) -> None:
    """
    Ensure all prerequisites for a single operation are computed.

    This uses the operation-driven planner to determine which stages
    need to run (ingest/graphs/analytics) and executes them as a single run.
    """
    runtime = build_project_runtime(Path(repo_root) if repo_root else None)

    result = ensure_prerequisites_for_operation(
        op_id=op_id,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        gateway=runtime.gateway,
        tools=runtime.tools,
        include_analytics=True,
        trigger="cli",
    )

    typer.echo(f"Run ID: {result.run_id}")
    typer.echo(f"Status: {result.status}")
```

#### `codeintel pipeline status [--run-id]`

This reads `metadata.pipeline_runs` and `metadata.pipeline_steps` via `gateway.runs` (which is `PipelineRunTracking` in the storage layer) and prints a simple status table. 

```python
def _print_run(run: PipelineRunRecord) -> None:
    typer.echo(
        f"{run.run_id}  repo={run.repo} commit={run.commit} "
        f"kind={run.kind} status={run.status} "
        f"operation={run.requested_operation or '-'}"
    )

def _print_steps(runs: PipelineRunTracking, run_id: str) -> None:
    steps = runs.fetch_steps(run_id)
    if not steps:
        typer.echo("  (no steps recorded)")
        return
    for step in steps:
        typer.echo(
            f"  [{step.module}] {step.stage}:{step.name} "
            f"{step.status} started={step.started_at} completed={step.completed_at}"
        )

@pipeline_app.command("status")
def pipeline_status(
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Run ID to inspect (defaults to last run).",
    ),
    repo_root: Optional[str] = typer.Option(
        None,
        help="Repo root (defaults to nearest codeintel.yaml).",
    ),
) -> None:
    """
    Show pipeline run status and step breakdown.

    If run-id is omitted, show the most recent run.
    """
    runtime = build_project_runtime(Path(repo_root) if repo_root else None)
    runs = runtime.gateway.runs

    if run_id is None:
        # naive "last run": fetch all and sort by started_at
        all_runs = runs.fetch_recent_runs(limit=10)  # you'd add this method if desired
        if not all_runs:
            typer.echo("No runs found.")
            raise typer.Exit(code=0)
        run = all_runs[0]
    else:
        run = runs.fetch_run(run_id)
        if run is None:
            typer.echo(f"Run not found: {run_id}")
            raise typer.Exit(code=1)

    _print_run(run)
    _print_steps(runs, run.run_id)
```

> If `fetch_recent_runs` doesn’t exist yet, you can implement it in `PipelineRunTracking` using a simple `ORDER BY started_at DESC LIMIT ?` query.

---

### 11.1.4 `codeintel op` commands

#### `codeintel op list`

Use `serving.operations.catalog.iter_operations()`. 

```python
from codeintel.serving.operations.catalog import iter_operations

@op_app.command("list")
def op_list() -> None:
    """
    List all operations from the canonical catalog.

    Columns: id, category, data_source, summary.
    """
    for op in iter_operations():
        typer.echo(
            f"{op.id:25}  [{op.category:<10}]  {op.data_source.value:<12}  {op.summary}"
        )
```

#### `codeintel op call <op-id> [params...]`

Implementation steps:

1. Build `ProjectRuntime` (snapshot, paths, gateway, tools, serving config).
2. Call `ensure_prerequisites_for_operation` to ensure data is ready.
3. Build a local `QueryService` via `serving.bootstrap.build_service_stack`.
4. Look up the operation via `get_operation(op_id)` and call `getattr(service, op.backend_method)(**kwargs)`.
5. Pretty-print the result (prefer JSON).

```python
from pathlib import Path
from typing import Any

from codeintel.serving.bootstrap import build_service_stack, ServiceStack
from codeintel.serving.operations.catalog import get_operation

def _parse_kv_params(params: list[str]) -> dict[str, Any]:
    """
    Parse name=value pairs into a kwargs dict with simple type coercion.
    """
    result: dict[str, Any] = {}
    for raw in params:
        if "=" not in raw:
            typer.echo(f"Invalid param '{raw}', expected name=value")
            raise typer.Exit(code=1)
        name, value = raw.split("=", 1)
        v: Any = value
        # naive type coercion
        if value.isdigit():
            v = int(value)
        elif value.lower() in {"true", "false"}:
            v = value.lower() == "true"
        result[name] = v
    return result

@op_app.command("call")
def op_call(
    op_id: str = typer.Argument(..., help="Operation ID (e.g. 'function.summary')"),
    params: list[str] = typer.Argument(
        [],
        help="Operation parameters as name=value pairs (e.g. goid_h128=123).",
    ),
    repo_root: Optional[str] = typer.Option(
        None, help="Repo root (defaults to nearest codeintel.yaml)"
    ),
) -> None:
    """
    Call an operation end-to-end:

    - Ensures prerequisites via operation-driven pipeline
    - Builds a local QueryService over DuckDB
    - Invokes the backend method and prints the result as JSON
    """
    runtime = build_project_runtime(Path(repo_root) if repo_root else None)

    op = get_operation(op_id)
    if op is None:
        typer.echo(f"Unknown operation: {op_id}")
        raise typer.Exit(code=1)

    # Ensure prerequisites
    ensure_prerequisites_for_operation(
        op_id=op_id,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        gateway=runtime.gateway,
        tools=runtime.tools,
        include_analytics=True,
        trigger="cli",
    )

    # Build service stack (local DB mode)
    stack: ServiceStack = build_service_stack(
        config=runtime.serving,
        gateway=runtime.gateway,
    )
    try:
        backend_method = op.backend_method
        if not hasattr(stack.service, backend_method):
            typer.echo(f"QueryService does not implement {backend_method}")
            raise typer.Exit(code=1)

        kwargs = _parse_kv_params(params)
        func = getattr(stack.service, backend_method)
        result = func(**kwargs)

        # Domain models are Pydantic or dataclasses; try to JSON-ify
        try:
            import json

            if hasattr(result, "model_dump"):
                payload = result.model_dump()
            elif hasattr(result, "dict"):
                payload = result.dict()
            elif hasattr(result, "__dict__"):
                payload = result.__dict__
            else:
                payload = result
            typer.echo(json.dumps(payload, indent=2, default=str))
        finally:
            stack.close()
    except Exception as exc:  # noqa: BLE001
        stack.close()
        typer.echo(f"Operation call failed: {exc}")
        raise typer.Exit(code=1)
```

---

### 11.1.5 `codeintel dataset` commands

We can use either:

* `DatasetQueryApi` via `QueryService`, or
* the storage layer’s `DatasetRegistry` (simpler for list/describe). 

#### `codeintel dataset list`

```python
from codeintel.storage.datasets import load_dataset_registry

@dataset_app.command("list")
def dataset_list(
    repo_root: Optional[str] = typer.Option(
        None, help="Repo root (defaults to nearest codeintel.yaml)"
    ),
) -> None:
    """
    List datasets with key metadata: name, table_key, owner_package, schema_version.
    """
    runtime = build_project_runtime(Path(repo_root) if repo_root else None)
    registry = runtime.gateway.datasets  # already a DatasetRegistry

    for name, contract in registry.by_name.items():
        typer.echo(
            f"{name:30}  table={contract.table_key:<32}  "
            f"owner={contract.owner_package or '-':<10}  "
            f"schema={contract.schema_version or '-'}"
        )
```

#### `codeintel dataset describe <table_key>`

```python
from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY

@dataset_app.command("describe")
def dataset_describe(
    table_key: str = typer.Argument(..., help="Dataset table key (e.g. 'analytics.function_profile')"),
    repo_root: Optional[str] = typer.Option(
        None, help="Repo root (defaults to nearest codeintel.yaml)"
    ),
) -> None:
    """
    Show schema + contract details for a specific dataset table_key.
    """
    runtime = build_project_runtime(Path(repo_root) if repo_root else None)
    contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
    if contract is None:
        typer.echo(f"No dataset contract found for table_key: {table_key}")
        raise typer.Exit(code=1)

    typer.echo(f"Dataset: {contract.name}")
    typer.echo(f"Table key: {contract.table_key}")
    typer.echo(f"Owner package: {contract.owner_package}")
    typer.echo(f"Family: {contract.family}")
    typer.echo(f"Schema version: {contract.schema_version}")
    typer.echo(f"Description: {contract.description or '-'}")
    typer.echo(f"Upstream dependencies: {', '.join(contract.upstream_dependencies or ()) or '-'}")

    if contract.schema is not None:
        typer.echo("\nColumns:")
        for col in contract.schema.columns:
            typer.echo(f"  - {col.name} : {col.type}")
```

#### `codeintel dataset verify <table_key>`

We’ll reuse `storage.contract_validation.collect_contract_issues` and filter down to the dataset of interest. 

```python
from codeintel.storage.contract_validation import collect_contract_issues

@dataset_app.command("verify")
def dataset_verify(
    table_key: str = typer.Argument(..., help="Dataset table key (e.g. 'analytics.function_profile')"),
    repo_root: Optional[str] = typer.Option(
        None, help="Repo root (defaults to nearest codeintel.yaml)"
    ),
) -> None:
    """
    Run lightweight contract validation for a specific dataset.

    This runs the standard contract validation and filters messages
    to those mentioning the dataset or table_key.
    """
    runtime = build_project_runtime(Path(repo_root) if repo_root else None)
    con = runtime.gateway.con

    issues = collect_contract_issues(con)
    filtered = [
        issue for issue in issues if table_key in issue or table_key.split(".", 1)[-1] in issue
    ]

    if not filtered:
        typer.echo(f"Dataset {table_key} passed contract checks.")
        raise typer.Exit(code=0)

    typer.echo(f"Dataset {table_key} has contract issues:")
    for issue in filtered:
        typer.echo(f"  - {issue}")
    raise typer.Exit(code=1)
```

---

### 11.1.6 `codeintel serve` commands

Goal: start the HTTP/MCP server with optional `--auto-pipeline` behavior. 

Minimum implementation for HTTP:

```python
import uvicorn
from codeintel.serving.http.fastapi import create_app

@serve_app.command("http")
def serve_http(
    host: str = typer.Option("127.0.0.1", help="Host to bind."),
    port: int = typer.Option(8080, help="Port to bind."),
    auto_pipeline: bool = typer.Option(
        False,
        "--auto-pipeline",
        help="Automatically run prerequisites for operations on first use.",
    ),
    repo_root: Optional[str] = typer.Option(
        None, help="Repo root (defaults to nearest codeintel.yaml)"
    ),
) -> None:
    """
    Start the HTTP server exposing CodeIntel operations backed by the local DB.

    When --auto-pipeline is set, the server will run operation prerequisites
    for a repo/commit that hasn't been indexed yet (via ensure_prerequisites_for_operation).
    """

    runtime = build_project_runtime(Path(repo_root) if repo_root else None)

    # You can thread auto_pipeline via ServingConfig or env var:
    if auto_pipeline:
        os.environ["CODEINTEL_AUTO_PIPELINE"] = "1"

    app = create_app(
        config_loader=lambda: runtime.serving,
        gateway=runtime.gateway,
    )

    uvicorn.run(app, host=host, port=port)
```

For MCP, you likely already have `serving.mcp.server.main` or equivalent; add a simple wrapper:

```python
from codeintel.serving.mcp.server import run_server as run_mcp_server

@serve_app.command("mcp")
def serve_mcp(
    auto_pipeline: bool = typer.Option(
        False,
        "--auto-pipeline",
        help="Automatically run prerequisites for operations on first use.",
    ),
    repo_root: Optional[str] = typer.Option(
        None, help="Repo root (defaults to nearest codeintel.yaml)"
    ),
) -> None:
    """
    Start the MCP server using local DuckDB with optional auto-pipeline.
    """
    runtime = build_project_runtime(Path(repo_root) if repo_root else None)
    if auto_pipeline:
        os.environ["CODEINTEL_AUTO_PIPELINE"] = "1"
    # Use ServingConfig+StorageGateway from runtime as needed in MCP bootstrap
    run_mcp_server(config=runtime.serving, gateway=runtime.gateway)
```

> The actual hook to `ensure_prerequisites_for_operation` in HTTP/MCP can be done inside the serving layer (e.g., in a request dependency that sees `CODEINTEL_AUTO_PIPELINE` and calls `ensure_prerequisites_for_operation` based on the Operation’s `required_datasets` / `required_graphs`). The Epic 11 plan only needs the CLI flag and wiring; deeper integration can be an Epic 12-style refinement.

---

## 11.2 Config & project file (`codeintel.yaml`)

You already saw the `ProjectConfig` model above. The expected YAML structure:

```yaml
repo: github.com/org/repo
default_profile: default

ingest:
  recipe: builtin.default

analytics:
  profile: full

graphs:
  recipe: builtin.full

storage:
  db_path: .codeintel/duckdb.db
```

**Mapping to runtime:**

* `repo` → `RepoConfig.repo`, `SnapshotRef.repo`.
* `commit` → discovered via env or Git (see `detect_commit`).
* `storage.db_path` → `CliPathsInput.db_path` → `BuildPaths.db_path` → `StorageConfig.for_ingest(db_path)`.
* `ingest.recipe` / `graphs.recipe` / `analytics.profile` → *for now*, we don’t override pipeline spec flavors (those still use `builtin.default` / `builtin.full`), but we can later extend:

  * `build_pipeline_for_operation` or `run_pipeline` to respect YAML-selected recipes by mapping `PipelineStage.name` accordingly.

This design keeps Epic 11 focused on **CLI & wiring** while leaving room for future use of these fields to tailor pipeline specs per project. 

---

## 11.3 CLI tests

Add new tests under `tests/cli`:

* `tests/cli/test_pipeline_cli_v2.py`
* `tests/cli/test_op_cli_v2.py`
* If you prefer the same names: adapt existing `tests/cli/test_pipeline_cli.py` and add `test_op_cli.py`.

Use Typer’s `CliRunner` (from `typer.testing`) and your existing repo fixtures. 

### 11.3.1 Example: pipeline CLI tests

```python
# tests/cli/test_pipeline_cli_v2.py
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from codeintel.cli import app

runner = CliRunner()

def test_pipeline_run_full_smoke(tmp_path: Path, monkeypatch) -> None:
    # Create a tiny repo + codeintel.yaml
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "codeintel.yaml").write_text(
        "repo: demo/repo\nstorage:\n  db_path: .codeintel/test.duckdb\n"
    )

    result = runner.invoke(app, ["pipeline", "run-full"], cwd=repo_root)

    assert result.exit_code == 0
    assert "Run ID:" in result.stdout
    assert "Status:" in result.stdout

def test_pipeline_run_op_smoke(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "codeintel.yaml").write_text(
        "repo: demo/repo\nstorage:\n  db_path: .codeintel/test.duckdb\n"
    )

    result = runner.invoke(app, ["pipeline", "run-op", "function.summary"], cwd=repo_root)
    assert result.exit_code == 0
    assert "Run ID:" in result.stdout
```

### 11.3.2 Example: op CLI tests

```python
# tests/cli/test_op_cli_v2.py
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from codeintel.cli import app

runner = CliRunner()

def test_op_list_shows_known_operation(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "codeintel.yaml").write_text(
        "repo: demo/repo\nstorage:\n  db_path: .codeintel/test.duckdb\n"
    )

    result = runner.invoke(app, ["op", "list"], cwd=repo_root)
    assert result.exit_code == 0
    assert "function.summary" in result.stdout

def test_op_call_produces_json(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "codeintel.yaml").write_text(
        "repo: demo/repo\nstorage:\n  db_path: .codeintel/test.duckdb\n"
    )
    # This will be a bit integration-heavy; start with a smoke test that it runs.
    result = runner.invoke(
        app,
        ["op", "call", "datasets.list"],
        cwd=repo_root,
    )
    assert result.exit_code == 0
    # Expect JSON object or list
    assert result.stdout.strip().startswith("[") or result.stdout.strip().startswith("{")
```

You can add similar tests for `dataset list / describe / verify` using an `open_memory_gateway` fixture seeded with a small dataset, and for `serve http` you can smoke-test that it boots and responds to `/health` using `httpx` or `requests` in a subprocess.

---

If you’d like, we can next:

* tighten the **auto-pipeline** integration on the serving side (add a decorator or dependency that calls `ensure_prerequisites_for_operation` using `Operation.required_datasets` / `required_graphs`), or
* refine the CLI parameter handling for `op call` by introspecting the backend method signatures and generating `typer.Option`/`Argument` definitions dynamically.


# detailed implementation plan and code snippets for decorator or dependency that calls ensure_prerequisites_for_operation #


Nice, this is a fun one. Let’s wire “auto-pipeline” into the serving layer in a way that:

* Is **opt-in** (via env / flag).
* Works for **local_db** mode (DuckDBBackend).
* Reuses your **Epic 10 op-planner** (`ensure_prerequisites_for_operation`).
* Avoids obviously re-running the pipeline if we already did a suitable run for this repo+commit+operation.

I’ll break it into:

1. Core helper module: `serving/auto_pipeline.py`
2. HTTP integration: FastAPI dependency factory
3. Route integration examples
4. Optional MCP integration hook (pattern)
5. Tests

I’ll give full code snippets you can drop in, with minimal assumptions about your existing Epic 10 API.

---

## 1. Core helper: `serving/auto_pipeline.py`

Create a new module:

```python
# src/codeintel/serving/auto_pipeline.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.config.serving_models import ServingConfig
from codeintel.pipeline.op_planner import ensure_prerequisites_for_operation
from codeintel.storage.run_tracking import PipelineRunTracking
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.serving.mcp.backend import QueryBackend, DuckDBBackend
```

### 1.1 Env flag & gate

We’ll control auto-pipeline via an env variable, which you already set in the CLI `serve` commands:

```python
AUTO_PIPELINE_ENV = "CODEINTEL_AUTO_PIPELINE"


def is_auto_pipeline_enabled(config: ServingConfig | None = None) -> bool:
    """
    Return True if auto-pipeline is enabled.

    Currently controlled by CODEINTEL_AUTO_PIPELINE env var:
    - '1', 'true', 'yes', 'on' (case-insensitive) => enabled.
    """
    value = os.environ.get(AUTO_PIPELINE_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}
```

### 1.2 Build `BuildPaths` from `ServingConfig`

HTTP/MCP server already knows `repo_root` and `db_path` in `ServingConfig`. We rebuild `BuildPaths` with the same defaults used by your CLI (`config.models.CodeIntelConfig.to_build_paths`). 

```python
def build_paths_for_serving(config: ServingConfig) -> BuildPaths:
    """
    Construct BuildPaths for a serving environment.

    Mirrors CodeIntelConfig/to_build_paths defaults:
    - build_dir: repo_root / '.codeintel'
    - db_path: config.db_path (required for local_db)
    - document_output_dir: repo_root / 'Document Output'
    - coverage_json: build_dir / 'coverage/coverage.json'
    - pytest_report: build_dir / 'test-results/pytest-report.json'
    - tool_cache: build_dir / 'tools'
    - log_db_path: build_dir / 'logs/codeintel_logs.duckdb'
    """
    root = config.repo_root
    build_dir = root / ".codeintel"
    db_path = config.db_path or (build_dir / "db" / "codeintel.duckdb")
    doc_dir = root / "Document Output"
    coverage_json = build_dir / "coverage" / "coverage.json"
    pytest_report = build_dir / "test-results" / "pytest-report.json"
    tool_cache = build_dir / "tools"
    log_db_path = build_dir / "logs" / "codeintel_logs.duckdb"

    return BuildPaths(
        build_dir=build_dir,
        db_path=db_path,
        document_output_dir=doc_dir,
        scip_dir=build_dir / "scip",
        coverage_json=coverage_json,
        pytest_report=pytest_report,
        tool_cache=tool_cache,
        log_db_path=log_db_path,
    )
```

### 1.3 Check if we already satisfied prereqs

We’ll use `metadata.pipeline_runs` via `PipelineRunTracking` to decide if we *need* to run the pipeline. Your runs table already captures `repo`, `commit`, `kind`, `requested_operation`, `status`.

We say “prereqs satisfied” if there exists a **successful** run for this repo+commit that is either:

* `kind='full'` (full pipeline), OR
* `kind='op_prereqs'` with `requested_operation=op_id`.

```python
def has_successful_prereq_run(
    runs: PipelineRunTracking,
    *,
    repo: str,
    commit: str,
    op_id: str,
) -> bool:
    """
    Return True if there is an existing successful run that satisfies
    the prerequisites for op_id at (repo, commit).

    Conditions:
    - status='succeeded' AND
      (kind='full' OR (kind='op_prereqs' AND requested_operation = op_id))
    """
    cur = runs.con.execute(
        """
        SELECT 1
        FROM metadata.pipeline_runs
        WHERE repo = ?
          AND commit = ?
          AND status = 'succeeded'
          AND (
                kind = 'full'
                OR (kind = 'op_prereqs' AND requested_operation = ?)
          )
        LIMIT 1
        """,
        [repo, commit, op_id],
    )
    return cur.fetchone() is not None
```

> This is intentionally conservative: a full pipeline run satisfies all operations; an op-specific run satisfies that op.

### 1.4 The serving-side auto-pipeline helper

Core helper called from HTTP/MCP: given an op_id, `ServingConfig`, and `QueryBackend`, decide whether to run prerequisites.

```python
def ensure_prereqs_for_http(
    *,
    op_id: str,
    config: ServingConfig,
    backend: "QueryBackend",
) -> None:
    """
    Ensure prerequisites for op_id are computed before handling the request.

    This is a lightweight gate that:
    - Checks CODEINTEL_AUTO_PIPELINE and ServingConfig.mode.
    - Only runs for local_db + DuckDBBackend.
    - Skips work if a prior successful run already exists.
    - Otherwise, runs ensure_prerequisites_for_operation via the unified pipeline.
    """
    from codeintel.serving.mcp.backend import DuckDBBackend  # local import to avoid cycles

    if not is_auto_pipeline_enabled(config):
        return
    if config.mode != "local_db":
        return
    if not isinstance(backend, DuckDBBackend):
        return

    gateway: StorageGateway = backend.gateway
    runs = gateway.runs

    repo = backend.repo or config.repo
    commit = backend.commit or config.commit

    if has_successful_prereq_run(runs, repo=repo, commit=commit, op_id=op_id):
        return

    snapshot = SnapshotRef(
        repo=repo,
        commit=commit,
        repo_root=config.repo_root,
    )
    paths = build_paths_for_serving(config)
    tools = ToolsConfig()  # default binaries; CLI can still set env PATH etc.

    # Important: we don't import this at module top to keep layering flexible
    ensure_prerequisites_for_operation(
        op_id=op_id,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=tools,
        include_analytics=True,
        trigger="http",
    )
```

That’s the “brains” of auto-pipeline: a single function that is:

* Cheap when prereqs are satisfied (one quick SQL query).
* Capability-gated (local_db + DuckDBBackend only).
* Pluggable into HTTP and MCP.

---

## 2. HTTP integration: dependency factory

Next we add a small dependency factory in `serving/http/dependencies.py` that wraps `ensure_prereqs_for_http` and exposes it as a FastAPI dependency.

Open **`serving/http/dependencies.py`**, and at the bottom add:

```python
from fastapi import Depends
from codeintel.serving.auto_pipeline import ensure_prereqs_for_http


def make_op_prereq_dependency(op_id: str):
    """
    Build a FastAPI dependency that ensures prerequisites for `op_id`.

    Usage
    -----
    @router.get(..., dependencies=[make_op_prereq_dependency("function.summary")])
    def function_summary(...):
        ...
    """

    def _dep(config: ConfigDep, backend: BackendDep) -> None:
        ensure_prereqs_for_http(
            op_id=op_id,
            config=config,
            backend=backend,
        )

    return Depends(_dep)
```

We reuse `ConfigDep` and `BackendDep` aliased earlier in this module:

```python
ConfigDep = Annotated[ServingConfig, Depends(get_app_config)]
BackendDep = Annotated[QueryBackend, Depends(get_backend)]
```

so FastAPI can inject `ServingConfig` and `QueryBackend` into our dependency.

---

## 3. Route integration: examples

Now we actually *use* this dependency in HTTP routes, mapping each HTTP endpoint to its corresponding operation ID.

### 3.1 Functions routes (canonical example)

In **`serving/http/routes/functions.py`**, we already have:

* `OperationSpec` mapping via `get_operation_spec`. 
* `_register_summary_and_risk_routes`, `_register_graph_and_tests_routes`, etc.

At the top, extend imports:

```python
from codeintel.serving.http.dependencies import ServiceDep, make_op_prereq_dependency
```

Then update `_register_summary_and_risk_routes` to attach auto-pipeline dependencies:

```python
from fastapi import APIRouter, Depends
...

def _register_summary_and_risk_routes(
    router: APIRouter,
    specs: dict[str, OperationSpec],
    paths: dict[str, str],
) -> None:
    summary_spec = specs["function.summary"]
    risk_spec = specs["functions.high_risk"]

    summary_prereqs = make_op_prereq_dependency(summary_spec.id)
    risk_prereqs = make_op_prereq_dependency(risk_spec.id)

    @router.get(
        paths["function.summary"],
        response_model=FunctionSummaryResponse,
        summary=summary_spec.summary,
        tags=[summary_spec.category],
        dependencies=[summary_prereqs],
    )
    def function_summary(
        *,
        service: ServiceDep,
        params: Annotated[FunctionSummaryParams, Depends(_function_summary_params)],
    ) -> FunctionSummaryResponse:
        ...
        # unchanged body
        domain_summary = service.function_summary(
            goid_h128=params.goid_h128,
            urn=params.urn,
            path=params.path,
            module=params.module,
            scope=params.scope,
        )
        ...

    @router.get(
        paths["functions.high_risk"],
        response_model=HighRiskFunctionsResponse,
        summary=risk_spec.summary,
        tags=[risk_spec.category],
        dependencies=[risk_prereqs],
    )
    def list_high_risk_functions(
        *,
        service: ServiceDep,
        min_risk: float | None = None,
        limit: int | None = None,
        tested_only: bool = False,
        scope: GraphScopePayload | None = None,
    ) -> HighRiskFunctionsResponse:
        ...
```

Similarly, in `_register_graph_and_tests_routes`, attach dependencies for:

* `graph.call_neighbors`
* `graph.call_neighborhood`
* `graph.import_boundary`
* `functions.tests`

Example snippet:

```python
def _register_graph_and_tests_routes(
    router: APIRouter, specs: dict[str, OperationSpec], paths: dict[str, str]
) -> None:
    neighbors_spec = specs["graph.call_neighbors"]
    neighborhood_spec = specs["graph.call_neighborhood"]
    import_boundary_spec = specs["graph.import_boundary"]
    tests_spec = specs["functions.tests"]

    neighbors_prereqs = make_op_prereq_dependency(neighbors_spec.id)
    neighborhood_prereqs = make_op_prereq_dependency(neighborhood_spec.id)
    import_boundary_prereqs = make_op_prereq_dependency(import_boundary_spec.id)
    tests_prereqs = make_op_prereq_dependency(tests_spec.id)

    @router.get(
        paths["graph.call_neighbors"],
        response_model=CallGraphNeighborsResponse,
        summary=neighbors_spec.summary,
        tags=[neighbors_spec.category],
        dependencies=[neighbors_prereqs],
    )
    def function_callgraph(...):
        ...

    @router.get(
        paths["graph.call_neighborhood"],
        response_model=GraphNeighborhoodResponse,
        summary=neighborhood_spec.summary,
        tags=[neighborhood_spec.category],
        dependencies=[neighborhood_prereqs],
    )
    def function_callgraph_neighborhood(...):
        ...

    @router.get(
        paths["graph.import_boundary"],
        response_model=ImportBoundaryResponse,
        summary=import_boundary_spec.summary,
        tags=[import_boundary_spec.category],
        dependencies=[import_boundary_prereqs],
    )
    def import_boundary(...):
        ...

    @router.get(
        paths["functions.tests"],
        response_model=TestsForFunctionResponse,
        summary=tests_spec.summary,
        tags=[tests_spec.category],
        dependencies=[tests_prereqs],
    )
    def list_tests_for_function(...):
        ...
```

> For other route modules (`profiles.py`, `datasets.py`, `architecture.py`, etc.) you can follow the same pattern: load `OperationSpec` for each endpoint, and attach `make_op_prereq_dependency(spec.id)` as a dependency.

We *don’t* add auto-pipeline to:

* `/health` routes.
* Non-operation endpoints (e.g. internal meta endpoints) unless they have a matching OperationSpec.

---

## 4. Optional MCP integration (pattern)

For MCP, operations are invoked via tools, not direct HTTP paths, but the pattern is the same:

* **Where**: in `serving/mcp/tools_base.py` (or wherever tools call `QueryBackend` / `QueryService` methods).

* **How**:

  * When a tool is invoked with tool id `op.id`, before calling `backend.service.<method>`, call:

    ```python
    from codeintel.serving.auto_pipeline import ensure_prereqs_for_http

    ensure_prereqs_for_http(
        op_id=op.id,
        config=config,
        backend=backend,
    )
    ```

    where `config` is your `ServingConfig` and `backend` is the local `DuckDBBackend`.

* **Gate**: `ensure_prereqs_for_http` already checks env + mode + backend type, so you don’t need extra guards.

You can later factor out a tiny helper `ensure_prereqs_for_mcp` if you want, but the logic is identical.

---

## 5. Tests

### 5.1 Unit test for `has_successful_prereq_run`

New file: **`tests/serving/test_auto_pipeline.py`**:

```python
from __future__ import annotations

from pathlib import Path

import duckdb

from codeintel.serving.auto_pipeline import (
    has_successful_prereq_run,
    build_paths_for_serving,
    ensure_prereqs_for_http,
)
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.gateway import StorageConfig, open_gateway

def test_has_successful_prereq_run_detects_full_run(tmp_path: Path) -> None:
    db_path = tmp_path / "codeintel.duckdb"
    cfg = StorageConfig.for_ingest(db_path)
    gw = open_gateway(cfg)
    runs = gw.runs

    # seed a successful full run into metadata.pipeline_runs
    runs.con.execute(
        """
        INSERT INTO metadata.pipeline_runs(
            run_id, repo, commit, kind, trigger,
            requested_operation, requested_datasets,
            started_at, completed_at, status, error_summary, pipeline_name
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 'succeeded', NULL, 'full')
        """,
        ["run-1", "demo/repo", "HEAD", "full", "cli", None, "[]"],
    )

    assert has_successful_prereq_run(
        runs,
        repo="demo/repo",
        commit="HEAD",
        op_id="function.summary",
    )
```

### 5.2 Integration-ish test: HTTP + auto-pipeline is invoked

You can mock `ensure_prerequisites_for_operation` to count calls:

```python
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from codeintel.serving.http.fastapi import create_app
from codeintel.serving.auto_pipeline import ensure_prereqs_for_http
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.gateway import StorageConfig, open_gateway

def test_http_auto_pipeline_runs_once(monkeypatch, tmp_path):
    # Enable auto-pipeline
    monkeypatch.setenv("CODEINTEL_AUTO_PIPELINE", "1")

    # Minimal ServingConfig + gateway
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    db_path = tmp_path / "codeintel.duckdb"
    gw = open_gateway(StorageConfig.for_ingest(db_path))

    config = ServingConfig(
        mode="local_db",
        repo_root=repo_root,
        repo="demo/repo",
        commit="HEAD",
        db_path=db_path,
    )

    app = create_app(config_loader=lambda: config, gateway=gw)

    client = TestClient(app)

    calls: list[str] = []

    def _spy(op_id, config, backend):
        calls.append(op_id)

    monkeypatch.setattr(
        "codeintel.serving.auto_pipeline.ensure_prereqs_for_http",
        _spy,
    )

    # First call: auto-pipeline should be invoked
    r1 = client.get("/functions/summary?goid_h128=123")
    # We don't assert on payload here; just check spy
    assert calls

    # Second call: still invoked unless has_successful_prereq_run marks it satisfied.
    # For a deeper test, we’d seed a successful run or let the real pipeline run once.
```

You can evolve this test to:

* Let the real pipeline run once.
* Then call the endpoint again and assert the spy isn’t called (because `has_successful_prereq_run` returns True).

---

## Summary

What we’ve added:

* **`serving/auto_pipeline.py`** – central logic to decide *when* and *how* to run operation prerequisites:

  * Env flag gate (`CODEINTEL_AUTO_PIPELINE`).
  * `BuildPaths` reconstruction for serving.
  * `has_successful_prereq_run` using `metadata.pipeline_runs`.
  * `ensure_prereqs_for_http` orchestrating `ensure_prerequisites_for_operation`.

* **HTTP dependency factory** in `serving/http/dependencies.py`:

  * `make_op_prereq_dependency(op_id)` that wires auto-pipeline into FastAPI routes.

* **Route integration** by attaching `dependencies=[make_op_prereq_dependency(spec.id)]` for each operation backed endpoint.

* **Testing hooks** so you can assert:

  * We don’t hammer the pipeline when a full/op_prereqs run already exists.
  * Auto-pipeline wiring actually runs before operation handlers when enabled.

If you’d like, the next refinement could be to make the **“has prerequisites”** check more data-aware (e.g., verifying required datasets actually exist for the repo/commit instead of just trusting a run record)—but the above is a solid, incremental step that plays nicely with your existing run tracking and op_planner.


# detailed implementation plan and code snippets for refine the CLI parameter handling for op call by introspecting the backend method signatures and generating typer.Option/Argument definitions dynamically. #

Yep, let’s make `codeintel op` feel “first-class” instead of a generic `name=value` dumping ground.

The key trick we’ll use: **dynamic command functions with a custom `__signature__`** so Typer can generate real `--goid-h128`, `--urn`, `--limit`, etc. options based on the backend method type hints, *without* having to hard-code every operation.

I’ll break it into:

1. Design overview
2. Introspection helpers (`cli/op_params.py`)
3. Dynamic command factory for operations
4. Wiring into `op_app`
5. Keeping the generic `op call` as a fallback
6. Tests

---

## 1. Design overview

Instead of:

```bash
codeintel op call function.summary goid_h128=123 urn=...
```

we’ll add **per-operation commands** like:

```bash
codeintel op function-summary --goid-h128 123
codeintel op function-summary --urn "urn:codeintel:py:..."
```

Technical approach:

* At import time, we:

  * Iterate over all `Operation`s from the canonical catalog.
  * For each, locate the corresponding backend API method (via the `query_api` Protocols).
  * Build an `inspect.Signature` that exposes typed CLI parameters (bool/int/str/etc).
  * Create a small generic handler function `op_command_impl(op_id, **kwargs)` and assign the **custom signature** to it via `func.__signature__ = ...`.
  * Register that function as a Typer command: `op_app.command(name=normalized_op_id)`.

Typer reads `__signature__` and uses it to define CLI arguments/options, so we don’t need to generate Python source code.

We will **keep** the previous `op call` (name=value) path as a fallback / power-user tool.

---

## 2. Introspection helpers (`codeintel/cli/op_params.py`)

Create a new file:

```python
# src/codeintel/cli/op_params.py
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, get_origin, get_args

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.backend import query_api as qa
from codeintel.serving.operations.catalog import Operation
```

### 2.1 Locate the backend method signature

The catalog gives you `Operation.backend_method` (e.g. `"get_function_summary"`). We don’t want to instantiate a real `DuckDBQueryService` just to inspect; instead we use the Protocol definitions in `serving/backend/query_api.py` (they carry type hints & defaults). 

Define which Protocol surfaces to search:

```python
_API_PROTOCOLS: tuple[type[Any], ...] = (
    qa.FunctionQueriesApi,
    qa.ModuleQueriesApi,
    qa.SubsystemQueriesApi,
    qa.DatasetQueriesApi,
    qa.ProfileQueriesApi,
    # add others as needed
)
```

Now locate the method:

```python
def get_backend_signature_for_operation(op: Operation) -> inspect.Signature:
    """
    Locate the backend method for an operation on one of the Query API protocols
    and return its signature.

    Raises
    ------
    ValueError
        If no matching method is found.
    """
    name = op.backend_method
    for api in _API_PROTOCOLS:
        if hasattr(api, name):
            fn = getattr(api, name)
            # Drop `self` if it appears in the signature
            sig = inspect.signature(fn)
            params = [
                p for p in sig.parameters.values()
                if p.name != "self"
            ]
            return sig.replace(parameters=params)
    msg = f"Could not find backend method {name!r} on any QueryApi protocol"
    raise ValueError(msg)
```

### 2.2 Map Python types → CLI types

We’ll create a small mapper for common types:

* `int`, `float`, `str`, `bool` → themselves.
* `GraphRunScope` and other complex types → accept JSON `str` and parse later (we’ll treat them as `str` at the CLI surface for now).
* `Sequence[...]` / `list[...]` → accept multiple values (`--tag foo --tag bar`) or comma-separated; we’ll start with comma-separated string for simplicity.

```python
@dataclass(frozen=True)
class CliParamSpec:
    name: str
    cli_name: str       # e.g. "goid-h128"
    annotation: Any
    cli_type: type
    default: Any
    required: bool
    is_flag: bool       # for bool fields with no default
    kind: inspect._ParameterKind
```

Helper to normalize Python names (snake_case) to CLI flag names:

```python
def _to_cli_name(name: str) -> str:
    return name.replace("_", "-")
```

Type mapping:

```python
def _cli_type_for_annotation(ann: Any) -> type:
    origin = get_origin(ann)
    args = get_args(ann)

    if ann is inspect._empty:
        return str

    if ann in {int, float, str, bool}:
        return ann

    # Optional[T] => T
    if origin is type(None):
        return str
    if origin is Union and len(args) == 2 and type(None) in args:  # type: ignore[name-defined]
        other = args[0] if args[1] is type(None) else args[1]
        return _cli_type_for_annotation(other)

    # GraphRunScope or other complex types -> accept JSON string
    if ann is GraphRunScope:
        return str

    # Fallback: string
    return str
```

Now extract CLI param specs from a signature:

```python
def build_cli_param_specs(sig: inspect.Signature) -> list[CliParamSpec]:
    """
    Convert a backend method signature into CLI parameter specs.

    Rules:
    - Positional-only is not supported; we treat all as keyword-only.
    - Required params: no default and not annotated Optional.
    - Bool parameters with default=False become --flag / --no-flag if needed.
    """
    specs: list[CliParamSpec] = []

    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        name = p.name
        cli_name = _to_cli_name(name)
        ann = p.annotation
        cli_type = _cli_type_for_annotation(ann)
        default = p.default
        required = default is inspect._empty

        is_flag = False
        if cli_type is bool and default is not inspect._empty:
            # bool with a default => simple --flag
            is_flag = True

        specs.append(
            CliParamSpec(
                name=name,
                cli_name=cli_name,
                annotation=ann,
                cli_type=cli_type,
                default=None if default is inspect._empty else default,
                required=required,
                is_flag=is_flag,
                kind=p.kind,
            )
        )

    return specs
```

We’ll use these `CliParamSpec`s to construct a *new* signature for the CLI handler.

### 2.3 Build a synthetic CLI signature

We want a function like:

```python
def op_command_impl(**kwargs): ...
op_command_impl.__signature__ = (...params based on CliParamSpec...)
```

We create parameters of kind `KEYWORD_ONLY` so they appear as options (`--param`), not as positional arguments.

```python
def build_cli_signature(op: Operation) -> inspect.Signature:
    """
    Build an inspect.Signature for the CLI command corresponding to `op`.
    """
    backend_sig = get_backend_signature_for_operation(op)
    param_specs = build_cli_param_specs(backend_sig)

    cli_params: list[inspect.Parameter] = [
        inspect.Parameter(
            name=spec.name,
            kind=inspect.Parameter.KEYWORD_ONLY,
            annotation=spec.cli_type,
            default=(
                inspect._empty if spec.required else spec.default
            ),
        )
        for spec in param_specs
    ]

    return inspect.Signature(parameters=cli_params)
```

> Note: we don’t embed the Operation into the signature; we’ll capture it via closure.

---

## 3. Dynamic command factory for operations

Now we build the actual handler factory.

Add to `op_params.py`:

```python
from typing import Any, Callable
from codeintel.cli.project import build_project_runtime
from codeintel.serving.bootstrap import build_service_stack
from codeintel.serving.operations.catalog import Operation
from codeintel.pipeline.op_planner import ensure_prerequisites_for_operation
import json
```

### 3.1 Common implementation function

This is essentially a refactor of your earlier `op call` body, but parameterized by `op_id` and kwargs:

```python
def op_command_impl(op: Operation, **kwargs: Any) -> None:
    """
    Common implementation for invoking an operation from the CLI.

    Steps:
    - Build project runtime (SnapshotRef, BuildPaths, Gateway, ToolsConfig, ServingConfig).
    - Run ensure_prerequisites_for_operation.
    - Build ServiceStack and call backend_method(**kwargs).
    - Pretty-print the result as JSON.
    """
    runtime = build_project_runtime()

    # 1) prereqs
    ensure_prerequisites_for_operation(
        op_id=op.id,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        gateway=runtime.gateway,
        tools=runtime.tools,
        include_analytics=True,
        trigger="cli",
    )

    # 2) service stack
    stack = build_service_stack(config=runtime.serving, gateway=runtime.gateway)
    try:
        service = stack.service
        method_name = op.backend_method
        if not hasattr(service, method_name):
            raise RuntimeError(f"QueryService does not implement {method_name!r}")

        fn = getattr(service, method_name)

        # Basic coercion is already done by Typer based on parameter types,
        # but we still need to post-process complex types (e.g., GraphRunScope from JSON).
        kwargs = _postprocess_kwargs_for_backend(fn, kwargs)

        result = fn(**kwargs)

        if hasattr(result, "model_dump"):
            payload = result.model_dump()
        elif hasattr(result, "dict"):
            payload = result.dict()
        elif hasattr(result, "__dict__"):
            payload = result.__dict__
        else:
            payload = result

        print(json.dumps(payload, indent=2, default=str))
    finally:
        stack.close()
```

Post-processing helper (e.g., `GraphRunScope` from JSON string):

```python
def _postprocess_kwargs_for_backend(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Convert CLI values into backend types that Typer couldn't infer safely.

    Example:
    - GraphRunScope passed as JSON string -> GraphRunScope.from_dict(...) or custom builder.
    """
    sig = inspect.signature(fn)
    new_kwargs: dict[str, Any] = {}

    for name, value in kwargs.items():
        param = sig.parameters.get(name)
        if param is None:
            new_kwargs[name] = value
            continue
        ann = param.annotation

        if ann is GraphRunScope and isinstance(value, str):
            # Expect JSON object with appropriate fields
            try:
                data = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON for GraphRunScope {name}: {value}") from exc

            new_kwargs[name] = GraphRunScope(
                paths=tuple(data.get("paths", ())),
                modules=tuple(data.get("modules", ())),
                time_window=None,  # or parse if provided
            )
        else:
            new_kwargs[name] = value

    return new_kwargs
```

### 3.2 Command factory that attaches `__signature__`

Now we build the per-op command:

```python
def make_op_cli_command(op: Operation) -> Callable[..., None]:
    """
    Create a CLI command function for an operation with a dynamic Typer signature.

    The returned function:
    - Captures the Operation in a closure.
    - Delegates to op_command_impl(op, **kwargs).
    - Exposes a synthetic inspect.Signature via __signature__ for Typer.
    """
    def command(**kwargs: Any) -> None:
        op_command_impl(op, **kwargs)

    # For Typer: set the signature and docstring
    command.__name__ = f"op_{op.id.replace('.', '_')}"
    command.__doc__ = op.summary or op.description or ""
    command.__signature__ = build_cli_signature(op)  # type: ignore[attr-defined]

    return command
```

---

## 4. Wiring dynamic commands into `op_app`

Now we integrate into `codeintel/cli/main.py`.

In `main.py`, add:

```python
from codeintel.serving.operations.catalog import iter_operations
from codeintel.cli.op_params import make_op_cli_command
```

Then, after `op_app = typer.Typer(...)`, register commands dynamically:

```python
def register_operation_commands(app: typer.Typer) -> None:
    """
    Dynamically register one CLI command per Operation with typed parameters.

    For an operation id like 'function.summary', we create a command named
    'function-summary', so users can call:

        codeintel op function-summary --goid-h128 123
    """
    for op in iter_operations():
        # Keep generic-only ops or internal ops out if needed
        if op.backend_method is None:
            continue

        cmd_name = op.id.replace(".", "-")
        cmd = make_op_cli_command(op)

        # Use Typer's decorator API directly
        app.command(name=cmd_name)(cmd)

# After op_app definition:
register_operation_commands(op_app)
```

The resulting CLI usage:

* For `function.summary`:

  ```bash
  codeintel op function-summary --goid-h128 123
  # or
  codeintel op function-summary --urn "urn:codeintel:py:..." --scope '{"paths":["src"],"modules":[]}'
  ```

* For `functions.high_risk`:

  ```bash
  codeintel op functions-high_risk --min-risk 0.8 --limit 10
  ```

Typer will automatically:

* Show `--goid-h128 INTEGER`, `--urn TEXT`, etc. in `--help`.
* Validate argument types (int/bool/float/str) using `cli_type` we set in the signature.

---

## 5. Keeping `op call` as a fallback

Your original Epic 11 design had:

```bash
codeintel op call <op-id> [params...]
```

We should keep that, but now back it with the same introspection logic for:

* Better error messages (unknown param name / missing required param).
* Type coercion (int, bool) based on annotation.

In `codeintel/cli/main.py` keep:

```python
@op_app.command("call")
def op_call_generic(
    op_id: str = typer.Argument(...),
    params: list[str] = typer.Argument(
        [],
        help="name=value pairs to pass as parameters",
    ),
) -> None:
    """
    Generic operation invocation using name=value pairs.

    Useful for experimentation or when dynamic operation commands are not available.
    """
    op = get_operation(op_id)
    if op is None:
        typer.echo(f"Unknown operation: {op_id}")
        raise typer.Exit(code=1)

    sig = get_backend_signature_for_operation(op)
    specs = build_cli_param_specs(sig)

    # parse name=value pairs
    raw_kwargs: dict[str, Any] = {}
    for raw in params:
        if "=" not in raw:
            typer.echo(f"Invalid param {raw!r}, expected name=value")
            raise typer.Exit(code=1)
        name, value = raw.split("=", 1)
        raw_kwargs[name] = value

    # validate against specs and coerce types
    kwargs: dict[str, Any] = {}
    spec_by_name = {s.name: s for s in specs}
    for spec in specs:
        if spec.name in raw_kwargs:
            v_str = raw_kwargs.pop(spec.name)
            # simple coercion based on cli_type
            if spec.cli_type is bool:
                kwargs[spec.name] = v_str.lower() in {"1", "true", "yes", "on"}
            elif spec.cli_type is int:
                kwargs[spec.name] = int(v_str)
            elif spec.cli_type is float:
                kwargs[spec.name] = float(v_str)
            else:
                kwargs[spec.name] = v_str
        elif spec.required:
            typer.echo(f"Missing required parameter: {spec.name}")
            raise typer.Exit(code=1)
        else:
            # optional; let backend default handle it
            pass

    if raw_kwargs:
        # user provided unknown params
        typer.echo(f"Unknown parameters: {', '.join(raw_kwargs.keys())}")
        raise typer.Exit(code=1)

    # Delegate to shared impl
    op_command_impl(op, **kwargs)
```

So:

* **Dynamic commands** (`codeintel op function-summary ...`) give you *fully typed* flags.
* **Generic call** remains handy as a backdoor but now uses the same signature knowledge for validation and coercion.

---

## 6. Tests

Add new tests under `tests/cli/test_op_dynamic_cli.py`.

### 6.1 Dynamic command exists

```python
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from codeintel.cli import app

runner = CliRunner()

def test_function_summary_dynamic_command_help(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "codeintel.yaml").write_text(
        "repo: demo/repo\nstorage:\n  db_path: .codeintel/test.duckdb\n"
    )

    result = runner.invoke(app, ["op", "function-summary", "--help"], cwd=repo_root)
    assert result.exit_code == 0
    # Expect some of the backend parameters to show up
    assert "--goid-h128" in result.stdout or "--urn" in result.stdout
```

### 6.2 Dynamic command type coercion

```python
def test_function_summary_dynamic_command_coerces_types(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "codeintel.yaml").write_text(
        "repo: demo/repo\nstorage:\n  db_path: .codeintel/test.duckdb\n"
    )

    # We don't assert on payload; just ensure it parses and calls backend.
    result = runner.invoke(
        app,
        ["op", "function-summary", "--goid-h128", "123"],
        cwd=repo_root,
    )
    assert result.exit_code == 0
```

### 6.3 Generic `op call` validation

```python
def test_op_call_generic_rejects_unknown_param(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "codeintel.yaml").write_text(
        "repo: demo/repo\nstorage:\n  db_path: .codeintel/test.duckdb\n"
    )

    result = runner.invoke(
        app,
        ["op", "call", "function.summary", "bogus=123"],
        cwd=repo_root,
    )
    assert result.exit_code != 0
    assert "Unknown parameters" in result.stdout
```

---

### Recap

This plan gives you:

* **Dynamic, typed, ergonomic commands per operation** (using Typer + `__signature__` hack).
* A shared, robust `op_command_impl` that handles project runtime, prerequisites, QueryService, and JSON output.
* A **smarter generic `op call`** that validates & coerces parameters based on backend signatures instead of blindly accepting `name=value`.

If you’d like, the next incremental refinement could be:

* Using operation metadata (`Operation.required_datasets`, `required_graphs`) and domain models to enrich the CLI help text (e.g., annotating which parameters are “key selectors” vs “filters”, marking `scope` as advanced, etc.).


# enriched cli help text impelementation plan and code snippets #

Nice, we’re getting into the UX polish layer now 🎯

We’ll make `codeintel op` commands *self-documenting* by:

* Using `Operation` metadata (`required_datasets`, `required_graphs`, etc.) to enrich help text.
* Classifying parameters as **“Key selectors”**, **“Filters”**, or **“Advanced”**.
* Using that classification to:

  * Add better per-param help text.
  * Group options into help panels (“Key selectors”, “Filters”, “Advanced options”) in `--help`.

We’ll build this on top of the dynamic per-operation commands we already designed (Epic 11 refinement #1).

---

## High-level design

1. **Extend `CliParamSpec`** (in `codeintel/cli/op_params.py`) with:

   * `role: Literal["selector", "filter", "advanced"]`
   * `help_text: str`
   * `help_panel: str | None` (for grouping in help)

2. **Introduce a classifier** that:

   * Looks at parameter **name** and **type**.
   * Looks at `Operation` metadata (`category`, `required_graphs`, `required_datasets`) to tweak messages.
   * Marks things like `goid_h128`, `urn`, `path` as **selectors**; `limit`, `min_risk` as **filters**; `scope`/graph-related params as **advanced**.

3. **Use Typer `Annotated` + `typer.Option`** when constructing the synthetic signature:

   * We set each parameter’s annotation to `Annotated[cli_type, typer.Option(..., help=help_text, rich_help_panel=help_panel)]`.
   * Typer + Click will show sections in `--help` grouped by `rich_help_panel`.

4. **Enrich the command docstring** with operation metadata:

   * Summary + description.
   * Category, data source, required datasets, required graphs.

5. Keep generic `op call` as-is (or lightly improved), but focus the enrichment on the dynamic commands.

---

## 1. Extend `CliParamSpec`

In `src/codeintel/cli/op_params.py`, extend the dataclass we already created:

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class CliParamSpec:
    name: str
    cli_name: str       # e.g. "goid-h128"
    annotation: Any
    cli_type: type
    default: Any
    required: bool
    is_flag: bool       # for bool fields with default
    kind: inspect._ParameterKind

    # New fields for enriched help
    role: Literal["selector", "filter", "advanced"] = "filter"
    help_text: str = ""
    help_panel: str | None = None
```

We’ll fill `role`, `help_text`, `help_panel` via a new classifier that takes both `Operation` and the parameter into account.

---

## 2. Classify parameters using Operation metadata + heuristics

Still in `op_params.py`, add:

```python
from codeintel.serving.operations.catalog import Operation
```

Define panel names:

```python
SELECTOR_PANEL = "Key selectors"
FILTER_PANEL = "Filters"
ADVANCED_PANEL = "Advanced options"
```

Add a classifier function:

```python
def classify_param_for_operation(
    op: Operation,
    spec: CliParamSpec,
) -> CliParamSpec:
    """
    Annotate a CliParamSpec with role, help_text, and help_panel using
    a combination of heuristics and Operation metadata.

    We try to keep rules simple and predictable:
    - 'Key selectors' are parameters that identify the primary entity.
    - 'Filters' are optional refinements (limit, offset, min/max, flags).
    - 'Advanced options' are graph/config/scoping parameters.
    """
    name = spec.name
    role = spec.role
    help_text = spec.help_text
    panel = spec.help_panel

    # 1) selectors: names indicating "what" to operate on
    selector_names = {
        "goid_h128", "urn", "path", "module",
        "file_path", "subsystem", "dataset_name", "table_key",
        "id", "function_id", "entrypoint",
    }
    if name in selector_names:
        role = "selector"
        panel = SELECTOR_PANEL
        # Basic description by default
        if "function" in op.category:
            help_text = help_text or "Target function to operate on."
        elif "datasets" in op.category:
            help_text = help_text or "Target dataset or table."
        else:
            help_text = help_text or "Target entity for this operation."

        return spec.__class__(**{**spec.__dict__, "role": role, "help_text": help_text, "help_panel": panel})

    # 2) filters: typical limit/offset/bounds/toggles
    if any(
        name.startswith(prefix)
        for prefix in ("min_", "max_", "has_", "include_", "exclude_")
    ) or name in {"limit", "offset", "tested_only", "include_tests"}:
        role = "filter"
        panel = FILTER_PANEL
        help_text = help_text or "Optional filter or control parameter."
        return spec.__class__(**{**spec.__dict__, "role": role, "help_text": help_text, "help_panel": panel})

    # 3) advanced: scopes, graph options, diagnostics, etc.
    if name in {"scope", "graph_scope", "graph_options"} or spec.cli_type is str and "scope" in name:
        role = "advanced"
        panel = ADVANCED_PANEL

        # Use required_graphs to enrich the message
        if op.required_graphs:
            graphs = ", ".join(op.required_graphs)
            help_text = (
                help_text
                or f"Optional graph scope; relevant because this operation depends on graph(s): {graphs}."
            )
        else:
            help_text = help_text or "Advanced scope parameter; most users can omit."

        return spec.__class__(**{**spec.__dict__, "role": role, "help_text": help_text, "help_panel": panel})

    # 4) default classification based on type
    if spec.cli_type is bool:
        role = "filter"
        panel = FILTER_PANEL
        help_text = help_text or "Optional flag controlling behavior."

    elif spec.required:
        # Required but not obviously a selector; treat as selector
        role = "selector"
        panel = SELECTOR_PANEL
        help_text = help_text or "Required parameter for this operation."

    else:
        role = "filter"
        panel = FILTER_PANEL
        help_text = help_text or "Optional parameter."

    return spec.__class__(**{**spec.__dict__, "role": role, "help_text": help_text, "help_panel": panel})
```

We’ve now got a way to classify each parameter based on:

* Name patterns (e.g. `min_`, `max_`, `limit` → filters).
* Type (bool flags).
* Operation metadata (`op.category`, `op.required_graphs`).

We’ll use this during spec building.

---

## 3. Integrate classification into `build_cli_param_specs`

Previously we had:

```python
def build_cli_param_specs(sig: inspect.Signature) -> list[CliParamSpec]:
    ...
```

We now need operation context, so we introduce a new function:

```python
def build_cli_param_specs_for_operation(
    op: Operation,
    sig: inspect.Signature,
) -> list[CliParamSpec]:
    """
    Convert backend signature into a list of CliParamSpec enriched
    with roles and help text using Operation metadata.
    """
    specs: list[CliParamSpec] = []

    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        name = p.name
        cli_name = _to_cli_name(name)
        ann = p.annotation
        cli_type = _cli_type_for_annotation(ann)
        default = p.default
        required = default is inspect._empty

        is_flag = False
        if cli_type is bool and default is not inspect._empty:
            is_flag = True

        base_spec = CliParamSpec(
            name=name,
            cli_name=cli_name,
            annotation=ann,
            cli_type=cli_type,
            default=None if default is inspect._empty else default,
            required=required,
            is_flag=is_flag,
            kind=p.kind,
        )

        enriched = classify_param_for_operation(op, base_spec)
        specs.append(enriched)

    return specs
```

Then adjust `build_cli_signature` to use it:

```python
def build_cli_signature(op: Operation) -> inspect.Signature:
    backend_sig = get_backend_signature_for_operation(op)
    param_specs = build_cli_param_specs_for_operation(op, backend_sig)

    cli_params: list[inspect.Parameter] = []
    for spec in param_specs:
        param = _make_typer_parameter_from_spec(spec)
        cli_params.append(param)

    return inspect.Signature(parameters=cli_params)
```

We’ll implement `_make_typer_parameter_from_spec` next.

---

## 4. Embed Typer `Option` metadata via `Annotated`

We want Typer to know:

* The type (`int`, `str`, etc.).
* The help text (per parameter).
* The help panel (selectors vs filters vs advanced).

We achieve that by using `typing.Annotated` + `typer.Option` as the annotation when constructing the parameter.

```python
from typing import Annotated
import typer

def _make_typer_parameter_from_spec(spec: CliParamSpec) -> inspect.Parameter:
    """
    Build an inspect.Parameter that Typer understands as an Option with
    rich help metadata.
    """
    option_kwargs: dict[str, Any] = {
        "help": spec.help_text,
        "show_default": True,
    }

    # Rich help panels group options in --help
    if spec.help_panel:
        option_kwargs["rich_help_panel"] = spec.help_panel

    # Use cli_name ("goid-h128") as the flag name
    param_decorator = typer.Option(
        None if spec.required else spec.default,
        f"--{spec.cli_name}",
        **option_kwargs,
    )

    annotated_type = Annotated[spec.cli_type, param_decorator]

    return inspect.Parameter(
        name=spec.name,
        kind=inspect.Parameter.KEYWORD_ONLY,
        annotation=annotated_type,
        default=inspect._empty if spec.required else spec.default,
    )
```

Now when we set:

```python
command.__signature__ = build_cli_signature(op)
```

Typer sees:

* a parameter annotated as `Annotated[int, typer.Option(...)]`
* with a KEYWORD_ONLY kind and optionally default, so it becomes an `--option`.

And it will reflect:

* `help` text.
* `rich_help_panel` grouping in the help output.

---

## 5. Enrich command docstring with Operation metadata

When we build the command function in `make_op_cli_command`, we previously did:

```python
command.__name__ = ...
command.__doc__ = op.summary or op.description or ""
command.__signature__ = build_cli_signature(op)
```

Now we’ll make the docstring more informative:

```python
def make_op_cli_command(op: Operation) -> Callable[..., None]:
    def command(**kwargs: Any) -> None:
        op_command_impl(op, **kwargs)

    command.__name__ = f"op_{op.id.replace('.', '_')}"

    # Build a rich docstring using op metadata
    required_datasets = ", ".join(op.required_datasets) if op.required_datasets else "none"
    required_graphs = ", ".join(op.required_graphs) if op.required_graphs else "none"

    lines = [
        op.summary or op.description or "",
        "",
        f"Category: {op.category}",
        f"Data source: {op.data_source.value} ({op.source_name})",
        f"Required datasets: {required_datasets}",
        f"Required graphs: {required_graphs}",
    ]
    command.__doc__ = "\n".join(lines).strip()

    command.__signature__ = build_cli_signature(op)  # type: ignore[attr-defined]
    return command
```

Now `codeintel op function-summary --help` might show:

```text
Function summary for a single function.

Category: functions
Data source: docs (docs.v_function_summary)
Required datasets: analytics.function_profile, analytics.function_metrics
Required graphs: callgraph
```

before listing sections:

* `Key selectors`
* `Filters`
* `Advanced options`

---

## 6. What the CLI help looks like after this

For `function.summary`, the help will roughly look like:

```text
Usage: codeintel op function-summary [OPTIONS]

  Function summary for a single function.

  Category: functions
  Data source: docs (docs.v_function_summary)
  Required datasets: analytics.function_profile, analytics.function_metrics
  Required graphs: callgraph

Key selectors:
  --goid-h128 INTEGER   Target function to operate on. [required]
  --urn TEXT            Target function to operate on.
  --path TEXT           Target function to operate on.
  --module TEXT         Target function to operate on.

Filters:
  --limit INTEGER       Optional filter or control parameter.
  --tested-only         Optional flag controlling behavior.

Advanced options:
  --scope TEXT          Optional graph scope; relevant because this
                        operation depends on graph(s): callgraph.
```

(all shaped by your actual signatures and `required_*` metadata).

---

## 7. Tests

Add tests in `tests/cli/test_op_dynamic_help.py` to validate the enriched help.

### 7.1 Help includes metadata and panels

```python
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from codeintel.cli import app

runner = CliRunner()

def test_function_summary_help_includes_operation_metadata(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "codeintel.yaml").write_text(
        "repo: demo/repo\nstorage:\n  db_path: .codeintel/test.duckdb\n"
    )

    result = runner.invoke(
        app,
        ["op", "function-summary", "--help"],
        cwd=repo_root,
    )
    assert result.exit_code == 0

    # Operation metadata
    assert "Category:" in result.stdout
    assert "Required datasets:" in result.stdout
    assert "Required graphs:" in result.stdout

    # Panels
    assert "Key selectors:" in result.stdout
    assert "Advanced options:" in result.stdout
```

### 7.2 A param classified as selector shows in Key selectors panel

```python
def test_selector_params_grouped_in_key_selectors(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "codeintel.yaml").write_text(
        "repo: demo/repo\nstorage:\n  db_path: .codeintel/test.duckdb\n"
    )

    result = runner.invoke(
        app,
        ["op", "function-summary", "--help"],
        cwd=repo_root,
    )
    assert result.exit_code == 0

    # Basic assertion that goid-h128 is listed under Key selectors
    help_text = result.stdout
    idx_panel = help_text.find("Key selectors:")
    idx_param = help_text.find("--goid-h128", idx_panel)
    assert idx_panel != -1
    assert idx_param != -1
```

### 7.3 Advanced param uses required_graphs in its help

```python
def test_scope_help_mentions_required_graphs(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "codeintel.yaml").write_text(
        "repo: demo/repo\nstorage:\n  db_path: .codeintel/test.duckdb\n"
    )

    result = runner.invoke(
        app,
        ["op", "function-summary", "--help"],
        cwd=repo_root,
    )

    assert result.exit_code == 0
    # The scope parameter should mention required graphs in its help
    assert "scope" in result.stdout
    assert "Required graphs:" not in the option line itself, but we expect
    assert "depends on graph(s)" in result.stdout
```

(You can fine-tune the expected string based on the exact help message you built.)

---

## Summary

This refinement adds a “semantic layer” to your `codeintel op` CLI:

* **Operation-level context** in the help header (category, data source, required datasets/graphs).
* **Parameter-level meaning**:

  * Key selectors vs filters vs advanced options.
  * Parameter help text that references Operation metadata when appropriate (e.g. scope mentions `required_graphs`).
* **Structured help layout** using `rich_help_panel` so users can quickly see what matters most.

It builds entirely on your existing Epic 10/11 structures, without hard-coding special cases per operation, and remains extensible if you later add new Query APIs or more nuanced roles (e.g. “pagination”, “sorting”, etc.).


# has prerequisites” check more data-aware (e.g., verifying required datasets actually exist for the repo/commit) implementation plan and detailed code snippets #

Love this refinement—this is where the system starts acting “smart” instead of just “logged something once so I guess it’s fine.”

We’ll add a **data-aware prerequisite check** so auto-pipeline:

* Looks at **Operation.required_datasets / required_graphs**.
* Expands upstream dependencies using dataset contracts.
* Verifies that the **required tables actually have data for (repo, commit)**.
* Only runs the pipeline if those checks fail.

We’ll still keep run-tracking in play, mainly as a fallback for operations with no dataset requirements.

I’ll break it into:

1. Design & behavior rules
2. New helpers in `pipeline.op_planner` (reuse Epic 10 logic)
3. Data-aware checks in `serving.auto_pipeline`
4. Integrating into `ensure_prereqs_for_http`
5. Optional tuning knobs (strict vs relaxed)
6. Tests

---

## 1. Design & behavior rules

**Current behavior:**

* Auto-pipeline checks `metadata.pipeline_runs` for a **successful** run with:

  * `kind='full'` OR
  * `kind='op_prereqs'` + `requested_operation = op_id`.
* If present, it **skips** running the pipeline again.

**New behavior (data-aware):**

For a given `op_id`, `repo`, `commit`:

1. Inspect `Operation.required_datasets` and **expand** transitive dependencies via `DatasetContract.upstream_dependencies`.
2. For each required table_key:

   * Use `DatasetContract` to:

     * Find the table name,
     * Determine whether it has `repo` + `commit` columns.
   * Run a cheap **LIMIT 1** query to see if there is *any* row for `(repo, commit)` (only for datasets partitioned by repo/commit).
3. If **all required datasets** are “present” for `(repo, commit)`:

   * Treat prerequisites as satisfied, **regardless** of whether a pipeline run record exists.
4. If there are **no required datasets** (and no obvious graph tables) then:

   * Fall back to the existing run-based check (for things like `datasets.list` that are computed from metadata).
5. Only when **both**:

   * Data-aware checks fail, and
   * (when applicable) run-tracking indicates no successful prereq run
     do we actually call `ensure_prerequisites_for_operation`.

This makes auto-pipeline smarter in two ways:

* It doesn’t trust *only* a run record; it looks at actual data.
* If someone bulk-loaded data into DuckDB without using the pipeline, auto-pipeline will see it and **not** re-run ingestion/graphs/analytics unnecessarily.

---

## 2. New helper in `pipeline.op_planner`: required dataset table keys

We want a single shared place that knows “what tables does op X *actually* depend on,” including upstream datasets.

Add to **`src/codeintel/pipeline/op_planner.py`** (building on the functions from Epic 10):

### 2.1 Existing building blocks (from Epic 10 plan)

From earlier, you likely already have:

* `_get_required_from_operation(op_id) -> (Operation, set[str], set[str])`
* `_expand_dataset_dependencies(required_tables: set[str]) -> set[str]`

If not, you can add them as described in Epic 10; here we’ll just assume they exist.

### 2.2 New API: `get_required_table_keys_for_operation`

```python
# src/codeintel/pipeline/op_planner.py

from __future__ import annotations
from typing import Set

from codeintel.serving.operations.catalog import get_operation

def get_required_table_keys_for_operation(op_id: str) -> set[str]:
    """
    Return the set of dataset table_keys required for op_id, including
    transitive upstream dependencies via DatasetContract.upstream_dependencies.

    This is the canonical source of truth for "which tables must contain
    data for this operation to be safe to execute".
    """
    op = get_operation(op_id)
    if op is None:
        raise ValueError(f"Unknown operation id: {op_id}")

    # Direct table_keys from Operation.required_datasets
    direct = set(op.required_datasets)

    # Expand via dataset contracts (from Epic 10 helpers)
    expanded = _expand_dataset_dependencies(direct)

    return expanded
```

> If you’d like to distinguish between **base tables** and **views**, you can keep `(contract.is_view)` in mind later; we’ll handle that in the data-layer check.

---

## 3. Data-aware checks in `serving.auto_pipeline`

Now we enhance `serving/auto_pipeline.py` to actually look at the DuckDB tables.

Assume you already have:

* `is_auto_pipeline_enabled(config)`
* `build_paths_for_serving(config)`
* `has_successful_prereq_run(runs, repo, commit, op_id)`
* `ensure_prereqs_for_http(...)` skeleton.

We’ll add:

1. A helper to check if a **single dataset** has data for a snapshot.
2. A helper to check **all required datasets** for an operation.
3. A higher-level `operation_prereqs_satisfied(...)` that decides whether to skip pipeline run.

### 3.1 Imports

At the top of `serving/auto_pipeline.py`:

```python
from duckdb import DuckDBPyConnection

from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY, DatasetContract
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.sql_builder import SafeTable
from codeintel.pipeline.op_planner import get_required_table_keys_for_operation
from codeintel.serving.operations.catalog import get_operation
```

### 3.2 Single-dataset check

```python
def dataset_has_rows_for_snapshot(
    con: DuckDBPyConnection,
    contract: DatasetContract,
    snapshot: SnapshotRef,
) -> bool:
    """
    Return True if the dataset has any row for the given (repo, commit).

    Behavior:
    - If the contract schema has 'repo' and 'commit' columns:
        SELECT 1 FROM <table> WHERE repo=? AND commit=? LIMIT 1
    - Otherwise:
        SELECT 1 FROM <table> LIMIT 1
      (we can't partition by repo/commit, so presence of any row counts)
    - For views (is_view=True), we treat them as satisfied if the query runs.
    """
    table_key = contract.table_key
    safe_table = SafeTable(table_key)

    # No schema? just test that the view/table can be queried
    if contract.schema is None:
        try:
            con.execute(f"SELECT 1 FROM {safe_table} LIMIT 1")
            return True
        except Exception:
            return False

    col_names = contract.schema.column_names()
    try:
        if "repo" in col_names and "commit" in col_names:
            # S608: SafeTable validates table key, and values are parameterized.
            cur = con.execute(
                f"SELECT 1 FROM {safe_table} WHERE repo = ? AND commit = ? LIMIT 1",  # noqa: S608
                [snapshot.repo, snapshot.commit],
            )
        else:
            cur = con.execute(
                f"SELECT 1 FROM {safe_table} LIMIT 1",  # noqa: S608
            )
        return cur.fetchone() is not None
    except Exception:
        return False
```

> This is intentionally conservative and cheap (`LIMIT 1`). If a repo legitimately has zero relevant rows, this test will come back `False`, and we’ll err on the side of running the pipeline.

### 3.3 Required datasets check

```python
def has_required_data_for_operation(
    *,
    op_id: str,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
) -> bool:
    """
    Return True if all dataset prerequisites for op_id are satisfied
    at the data level for this snapshot.

    We:
    - Fetch the required table_keys (direct + upstream) via op_planner.
    - For each table_key, load its DatasetContract.
    - Check for at least one row for (repo, commit) (or any row if not partitioned).
    """
    con: DuckDBPyConnection = gateway.con

    required_tables = get_required_table_keys_for_operation(op_id)
    if not required_tables:
        # No dataset prerequisites expressed; nothing to check at the data layer
        return False

    for table_key in required_tables:
        contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if contract is None:
            # Contract missing: be conservative, treat as not satisfied
            return False
        if not dataset_has_rows_for_snapshot(con, contract, snapshot):
            return False

    return True
```

> Note the subtle design choice: we return **False** when there are no required tables; that signals to the caller “I can’t prove prerequisites from data alone, fall back to run logs or pipeline.” We’ll handle that in the next function.

### 3.4 Combined “prereqs satisfied” logic

Now we encapsulate the full heuristic:

```python
def operation_prereqs_satisfied(
    *,
    op_id: str,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
) -> bool:
    """
    Decide whether operation prerequisites are satisfied for a snapshot.

    Logic:
    1. If the operation declares dataset prerequisites:
        - Use has_required_data_for_operation.
        - If True, consider prereqs satisfied (data is ground truth).
        - If False, treat as unsatisfied (run pipeline).
    2. If the operation declares NO dataset prerequisites:
        - Fall back to run-based heuristic has_successful_prereq_run.
    """
    op = get_operation(op_id)
    if op is None:
        # Unknown op; better to skip auto-pipeline than crash
        return False

    # Data-aware path if explicit datasets are declared
    if op.required_datasets:
        if has_required_data_for_operation(
            op_id=op_id,
            snapshot=snapshot,
            gateway=gateway,
        ):
            return True
        # Data is missing => prereqs not satisfied; DO NOT trust run record alone
        return False

    # No datasets declared: fall back to run-based heuristic
    runs = gateway.runs
    return has_successful_prereq_run(
        runs,
        repo=snapshot.repo,
        commit=snapshot.commit,
        op_id=op_id,
    )
```

This captures the “more data-aware” behavior:

* For operations that **actually declare dataset dependencies**, we insist on verifying the underlying tables.
* Only when an operation has **no dataset prerequisites** do we rely solely on `pipeline_runs`.

---

## 4. Integrate into `ensure_prereqs_for_http`

Finally, tie this into `ensure_prereqs_for_http` (the function the HTTP dependency calls).

Previously (simplified) we had:

```python
def ensure_prereqs_for_http(*, op_id, config, backend):
    if not is_auto_pipeline_enabled(config): return
    if config.mode != "local_db": return
    if not isinstance(backend, DuckDBBackend): return

    gateway = backend.gateway
    runs = gateway.runs
    repo = backend.repo or config.repo
    commit = backend.commit or config.commit

    if has_successful_prereq_run(runs, repo=repo, commit=commit, op_id=op_id):
        return

    snapshot = SnapshotRef(...)
    paths = build_paths_for_serving(config)
    tools = ToolsConfig()

    ensure_prerequisites_for_operation(...)
```

Update it to use the new data-aware logic:

```python
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.config.models import ToolsConfig

def ensure_prereqs_for_http(
    *,
    op_id: str,
    config: ServingConfig,
    backend: "QueryBackend",
) -> None:
    """
    Ensure prerequisites for op_id are computed before handling the request.

    Uses:
    - data-aware checks for operations that declare dataset prerequisites, and
    - run-based checks as a fallback for dataset-free operations.
    """
    if not is_auto_pipeline_enabled(config):
        return
    if config.mode != "local_db":
        return
    if not isinstance(backend, DuckDBBackend):
        return

    gateway: StorageGateway = backend.gateway

    repo = backend.repo or config.repo
    commit = backend.commit or config.commit
    snapshot = SnapshotRef(
        repo=repo,
        commit=commit,
        repo_root=config.repo_root,
    )

    # New: data-aware prerequisite check
    if operation_prereqs_satisfied(
        op_id=op_id,
        snapshot=snapshot,
        gateway=gateway,
    ):
        return

    # If we reach here, prerequisites are not satisfied for this op+snapshot
    paths = build_paths_for_serving(config)
    tools = ToolsConfig()

    ensure_prerequisites_for_operation(
        op_id=op_id,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=tools,
        include_analytics=True,
        trigger="http",
    )
```

So:

* For operations with `required_datasets`, we now **never** skip a pipeline run just because a run record exists; we actually check the data.
* For operations without dataset prerequisites, we behave as before.

---

## 5. Optional tuning knobs (strict vs relaxed)

If you’re worried about performance or false negatives (e.g., some repos legitimately having no rows), you can gate strict behavior with another env var:

* `CODEINTEL_AUTO_PIPELINE_STRICT=1` → use data-aware path.
* Absent → fall back to run-based only (current behavior).

You’d tweak `operation_prereqs_satisfied` accordingly:

```python
def is_strict_data_check_enabled() -> bool:
    return os.environ.get("CODEINTEL_AUTO_PIPELINE_STRICT", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
```

Then:

```python
if op.required_datasets and is_strict_data_check_enabled():
    ...
elif op.required_datasets:
    # optional hybrid: first check run record, only fall back to data check when
    # there's no successful run.
    ...
```

But for now, the simpler always-data-aware behavior is fine and closer to what you asked.

---

## 6. Tests

Add a new test file: **`tests/serving/test_auto_pipeline_data_aware.py`**.

### 6.1 Data present → no pipeline run

We’ll monkeypatch `ensure_prerequisites_for_operation` to detect whether the pipeline was run.

```python
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from codeintel.serving.auto_pipeline import ensure_prereqs_for_http
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.sql_builder import SafeTable
from codeintel.serving.mcp.backend import DuckDBBackend

from codeintel.serving.operations.catalog import get_operation


def _seed_required_table_for_op(
    con: duckdb.DuckDBPyConnection,
    op_id: str,
    repo: str,
    commit: str,
) -> None:
    op = get_operation(op_id)
    assert op is not None

    # For test, use a simple known required dataset, e.g. analytics.function_profile
    # or pick the first required_datasets table_key.
    table_key = op.required_datasets[0]
    safe_table = SafeTable(table_key)

    # Create table and insert a row for (repo, commit)
    con.execute(f"CREATE TABLE {safe_table}(repo TEXT, commit TEXT, x INT)")  # noqa: S608
    con.execute(f"INSERT INTO {safe_table} VALUES (?, ?, 1)", [repo, commit])


def test_auto_pipeline_skips_when_data_present(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "codeintel.duckdb"
    gw = open_gateway(StorageConfig.for_ingest(db_path))

    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="demo/repo",
        commit="HEAD",
        db_path=db_path,
    )

    backend = DuckDBBackend(gateway=gw, repo=config.repo, commit=config.commit)

    # Seed required dataset with a row for this repo/commit
    _seed_required_table_for_op(gw.con, "function.summary", config.repo, config.commit)

    # Spy on ensure_prerequisites_for_operation
    calls: list[str] = []

    def _spy(*, op_id, snapshot, paths, gateway, tools, include_analytics, trigger) -> None:
        calls.append(op_id)

    monkeypatch.setenv("CODEINTEL_AUTO_PIPELINE", "1")
    monkeypatch.setattr(
        "codeintel.serving.auto_pipeline.ensure_prerequisites_for_operation",
        _spy,
    )

    ensure_prereqs_for_http(
        op_id="function.summary",
        config=config,
        backend=backend,
    )

    # Because data is present, we expect no pipeline run
    assert calls == []
```

### 6.2 Data missing → pipeline run triggered

```python
def test_auto_pipeline_runs_when_data_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "codeintel.duckdb"
    gw = open_gateway(StorageConfig.for_ingest(db_path))

    config = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="demo/repo",
        commit="HEAD",
        db_path=db_path,
    )

    backend = DuckDBBackend(gateway=gw, repo=config.repo, commit=config.commit)

    calls: list[str] = []

    def _spy(*, op_id, snapshot, paths, gateway, tools, include_analytics, trigger) -> None:
        calls.append(op_id)

    monkeypatch.setenv("CODEINTEL_AUTO_PIPELINE", "1")
    monkeypatch.setattr(
        "codeintel.serving.auto_pipeline.ensure_prerequisites_for_operation",
        _spy,
    )

    ensure_prereqs_for_http(
        op_id="function.summary",
        config=config,
        backend=backend,
    )

    assert calls == ["function.summary"]
```

You can add more tests for operations without `required_datasets` to confirm that they still rely on `has_successful_prereq_run`.

---

### Wrap-up

With these changes, auto-pipeline is now **data-aware**:

* For operations with explicit dataset dependencies, it actually looks at those tables and confirms there’s data for `(repo, commit)` before assuming prerequisites are satisfied.
* Run records become a secondary heuristic instead of the sole source of truth.
* The system gracefully handles both “schema-driven” automation and future manual ingestion scenarios with minimal extra complexity.

If you want, we can next layer in a small `/debug/pipeline/prereqs` endpoint or `codeintel op debug-prereqs <op-id>` CLI command that prints exactly which tables and runs the system considered when making its decision—that’s a nice complement to this refinement.



# /debug/pipeline/prereqs endpoint that tells you exactly why auto-pipeline did or did not run for an operation detailed implementation plan and code snippets #

Nice, this is a really natural “observability” add-on to everything we just built. Let’s give ourselves a `/debug/pipeline/prereqs` endpoint that tells you *exactly* why auto-pipeline did or did not run for an operation.

I’ll walk through:

1. What the endpoint should return (shape & semantics)
2. New Pydantic response models
3. Debug info helpers in `serving.auto_pipeline`
4. HTTP route implementation in `serving/http/routes/meta.py`
5. Example responses
6. Tests

---

## 1. Endpoint goal & response shape

**Endpoint**:

```http
GET /debug/pipeline/prereqs?op_id=function.summary&repo=...&commit=...
```

**Goal**: For a given operation and snapshot:

* Show *which* datasets & graphs are considered prerequisites.
* Show for each dataset:

  * Whether a contract exists
  * Whether we checked it
  * Whether it has any rows (for repo+commit, when applicable)
* Show which pipeline runs we considered:

  * `full` runs
  * `op_prereqs` runs for this op_id
* Show whether:

  * `data_satisfied` (data-aware check passed)
  * `run_satisfied` (run-based check passed)
  * `overall_satisfied` (what auto-pipeline uses to skip/rerun)

A concrete response might look like:

```json
{
  "op_id": "function.summary",
  "repo": "github.com/acme/project",
  "commit": "abc123",
  "required_datasets": ["analytics.function_profile"],
  "expanded_datasets": [
    "analytics.function_profile",
    "analytics.function_metrics"
  ],
  "required_graphs": ["callgraph"],
  "dataset_statuses": [
    {
      "table_key": "analytics.function_profile",
      "name": "function_profile",
      "owner_package": "analytics",
      "schema_version": "1",
      "is_view": false,
      "has_repo_commit_rows": true,
      "checked": true,
      "error": null
    },
    {
      "table_key": "analytics.function_metrics",
      "name": "function_metrics",
      "owner_package": "analytics",
      "schema_version": "1",
      "is_view": false,
      "has_repo_commit_rows": false,
      "checked": true,
      "error": null
    }
  ],
  "runs_considered": [
    {
      "run_id": "full-123",
      "kind": "full",
      "status": "succeeded",
      "pipeline_name": "full",
      "requested_operation": null,
      "started_at": "2025-03-10T10:30:00Z",
      "completed_at": "2025-03-10T10:31:00Z"
    }
  ],
  "data_satisfied": false,
  "run_satisfied": true,
  "overall_satisfied": true
}
```

---

## 2. New Pydantic response models

Add these to `src/codeintel/serving/mcp/models.py` near `DatasetMetaResponse` / `OperationMetaResponse`:

```python
from datetime import datetime
from pydantic import BaseModel, Field
```

```python
class OperationPrereqDatasetStatus(BaseModel):
    """Debug status for a single dataset prerequisite."""

    table_key: str
    name: str | None = None
    owner_package: str | None = None
    schema_version: str | None = None
    is_view: bool = False
    has_repo_commit_rows: bool | None = Field(
        default=None,
        description="True if any row exists for (repo, commit). "
                    "False if none. None if repo/commit columns not present.",
    )
    checked: bool = Field(
        default=False,
        description="True if we attempted to query this dataset; False if contract missing.",
    )
    error: str | None = Field(
        default=None,
        description="Any error encountered while querying the dataset.",
    )


class OperationPrereqRunSummary(BaseModel):
    """Summary of a pipeline run relevant to prerequisites."""

    run_id: str
    kind: str
    status: str
    pipeline_name: str | None = None
    requested_operation: str | None = None
    started_at: datetime
    completed_at: datetime | None = None


class OperationPrereqDebugResponse(BaseModel):
    """Explain why auto-pipeline considered prerequisites satisfied or not."""

    op_id: str
    repo: str
    commit: str

    required_datasets: list[str] = Field(default_factory=list)
    expanded_datasets: list[str] = Field(default_factory=list)
    required_graphs: list[str] = Field(default_factory=list)

    dataset_statuses: list[OperationPrereqDatasetStatus] = Field(default_factory=list)
    runs_considered: list[OperationPrereqRunSummary] = Field(default_factory=list)

    data_satisfied: bool
    run_satisfied: bool
    overall_satisfied: bool
```

(And add these to the module’s `__all__` if you keep one.)

---

## 3. Debug helpers in `serving.auto_pipeline`

We’ll reuse the functions we just designed in the data-aware refinement:

* `get_required_table_keys_for_operation(op_id)` – from `pipeline.op_planner`
* `dataset_has_rows_for_snapshot(con, contract, snapshot)`
* `has_required_data_for_operation(op_id, snapshot, gateway)`
* `has_successful_prereq_run(runs, repo, commit, op_id)`
* `operation_prereqs_satisfied(op_id, snapshot, gateway)`

We’ll add a small set of **dataclasses** to hold raw debug info (to avoid importing Pydantic into `serving.auto_pipeline`, which is nice for layering), and a function to populate them.

### 3.1 Dataclasses

In `src/codeintel/serving/auto_pipeline.py`:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from duckdb import DuckDBPyConnection

from codeintel.config.datasets import DATASET_CONTRACTS_BY_TABLE_KEY, DatasetContract
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.run_tracking import PipelineRunTracking
from codeintel.pipeline.op_planner import get_required_table_keys_for_operation
from codeintel.serving.operations.catalog import get_operation
```

```python
@dataclass
class DatasetDebugInfo:
    table_key: str
    name: str | None
    owner_package: str | None
    schema_version: str | None
    is_view: bool
    has_repo_commit_rows: bool | None
    checked: bool
    error: str | None = None


@dataclass
class RunDebugInfo:
    run_id: str
    kind: str
    status: str
    pipeline_name: str | None
    requested_operation: str | None
    started_at: datetime
    completed_at: datetime | None


@dataclass
class PrereqDebugInfo:
    op_id: str
    repo: str
    commit: str
    required_datasets: list[str]
    expanded_datasets: list[str]
    required_graphs: list[str]
    dataset_statuses: list[DatasetDebugInfo]
    runs_considered: list[RunDebugInfo]
    data_satisfied: bool
    run_satisfied: bool
    overall_satisfied: bool
```

### 3.2 Helper: collect run info

We’ll reuse the same WHERE clause as `has_successful_prereq_run`, but return all matching runs:

```python
def get_prereq_runs_for_operation(
    runs: PipelineRunTracking,
    *,
    repo: str,
    commit: str,
    op_id: str,
) -> list[RunDebugInfo]:
    """
    Fetch runs that could satisfy prerequisites for op_id at (repo, commit).

    Includes:
    - full runs
    - op_prereqs runs with requested_operation = op_id
    """
    cur = runs.con.execute(
        """
        SELECT
            run_id,
            kind,
            status,
            pipeline_name,
            requested_operation,
            started_at,
            completed_at
        FROM metadata.pipeline_runs
        WHERE repo = ?
          AND commit = ?
          AND (
                kind = 'full'
                OR (kind = 'op_prereqs' AND requested_operation = ?)
          )
        ORDER BY started_at DESC
        """,
        [repo, commit, op_id],
    )

    results: list[RunDebugInfo] = []
    for (
        run_id,
        kind,
        status,
        pipeline_name,
        requested_operation,
        started_at,
        completed_at,
    ) in cur.fetchall():
        results.append(
            RunDebugInfo(
                run_id=run_id,
                kind=str(kind),
                status=str(status),
                pipeline_name=pipeline_name,
                requested_operation=requested_operation,
                started_at=started_at,
                completed_at=completed_at,
            )
        )
    return results
```

### 3.3 Helper: dataset debug info

We’ll wrap `dataset_has_rows_for_snapshot` carefully to capture errors instead of raising:

```python
def collect_dataset_debug_info(
    *,
    required_tables: list[str],
    snapshot: SnapshotRef,
    gateway: StorageGateway,
) -> list[DatasetDebugInfo]:
    con: DuckDBPyConnection = gateway.con
    results: list[DatasetDebugInfo] = []

    for table_key in required_tables:
        contract = DATASET_CONTRACTS_BY_TABLE_KEY.get(table_key)
        if contract is None:
            results.append(
                DatasetDebugInfo(
                    table_key=table_key,
                    name=None,
                    owner_package=None,
                    schema_version=None,
                    is_view=False,
                    has_repo_commit_rows=None,
                    checked=False,
                    error="No DatasetContract found for table_key",
                )
            )
            continue

        error: str | None = None
        has_rows: bool | None = None
        try:
            has_rows = dataset_has_rows_for_snapshot(con, contract, snapshot)
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            has_rows = None

        results.append(
            DatasetDebugInfo(
                table_key=table_key,
                name=contract.name,
                owner_package=contract.owner_package,
                schema_version=contract.schema_version,
                is_view=contract.is_view,
                has_repo_commit_rows=has_rows,
                checked=True,
                error=error,
            )
        )

    return results
```

### 3.4 Main debug builder

Now a single function to produce the structured debug info:

```python
def build_prereq_debug_info(
    *,
    op_id: str,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
) -> PrereqDebugInfo:
    """
    Build a detailed debug view of prerequisites for op_id at snapshot.

    Combines:
    - Operation metadata (required datasets/graphs)
    - Dataset-level checks for repo/commit
    - Pipeline run history relevant to this operation
    - Final booleans used by auto-pipeline
    """
    op = get_operation(op_id)
    if op is None:
        raise ValueError(f"Unknown operation id: {op_id}")

    repo = snapshot.repo
    commit = snapshot.commit

    # Required datasets: direct and expanded
    direct_required = list(op.required_datasets)
    expanded_required = sorted(get_required_table_keys_for_operation(op_id))

    dataset_statuses = collect_dataset_debug_info(
        required_tables=expanded_required,
        snapshot=snapshot,
        gateway=gateway,
    )

    runs = gateway.runs
    run_summaries = get_prereq_runs_for_operation(
        runs,
        repo=repo,
        commit=commit,
        op_id=op_id,
    )

    data_satisfied = has_required_data_for_operation(
        op_id=op_id,
        snapshot=snapshot,
        gateway=gateway,
    )
    run_satisfied = has_successful_prereq_run(
        runs,
        repo=repo,
        commit=commit,
        op_id=op_id,
    )
    overall_satisfied = operation_prereqs_satisfied(
        op_id=op_id,
        snapshot=snapshot,
        gateway=gateway,
    )

    return PrereqDebugInfo(
        op_id=op_id,
        repo=repo,
        commit=commit,
        required_datasets=direct_required,
        expanded_datasets=expanded_required,
        required_graphs=list(op.required_graphs),
        dataset_statuses=dataset_statuses,
        runs_considered=run_summaries,
        data_satisfied=data_satisfied,
        run_satisfied=run_satisfied,
        overall_satisfied=overall_satisfied,
    )
```

We now have a single call that can power both the HTTP endpoint and any future CLI `codeintel op debug-prereqs`.

---

## 4. HTTP route in `serving/http/routes/meta.py`

Now we expose this via FastAPI.

### 4.1 Imports

At the top of `src/codeintel/serving/http/routes/meta.py`, extend imports:

```python
from fastapi import APIRouter, HTTPException, Query

from codeintel.serving.http.dependencies import BackendDep, ConfigDep, ServiceDep
from codeintel.serving.auto_pipeline import build_prereq_debug_info
from codeintel.serving.mcp.models import (
    ...,
    OperationPrereqDebugResponse,
    OperationPrereqDatasetStatus,
    OperationPrereqRunSummary,
)
```

### 4.2 Router function signature

`build_meta_router()` already exists and attaches meta routes; we’ll add a new GET handler inside:

```python
def build_meta_router() -> APIRouter:
    router = APIRouter(tags=["meta"])

    # ... existing dataset / operation meta routes ...

    @router.get(
        "/debug/pipeline/prereqs",
        response_model=OperationPrereqDebugResponse,
        summary="Explain auto-pipeline prerequisites for an operation.",
    )
    def debug_pipeline_prereqs(
        op_id: str = Query(..., description="Operation id (e.g. 'function.summary')."),
        repo: str | None = Query(
            None,
            description="Override repo slug (defaults to serving config repo).",
        ),
        commit: str | None = Query(
            None,
            description="Override commit (defaults to serving config commit).",
        ),
        cfg: ConfigDep = None,
        backend: BackendDep = None,
    ) -> OperationPrereqDebugResponse:
        """
        Return detailed information about how auto-pipeline decides whether
        prerequisites are satisfied for op_id at the given snapshot.

        Uses the same logic as auto-pipeline:
        - required_datasets / required_graphs from OperationCatalog
        - dataset contracts (DATASET_CONTRACTS_BY_TABLE_KEY)
        - data-aware checks (rows for repo/commit)
        - run-based checks (pipeline_runs)
        """
        if cfg is None or backend is None:
            raise HTTPException(status_code=500, detail="Serving dependencies not available")

        snapshot_repo = repo or cfg.repo
        snapshot_commit = commit or cfg.commit

        snapshot = SnapshotRef(
            repo=snapshot_repo,
            commit=snapshot_commit,
            repo_root=cfg.repo_root,
        )

        try:
            debug_info = build_prereq_debug_info(
                op_id=op_id,
                snapshot=snapshot,
                gateway=backend.gateway,
            )
        except ValueError as exc:
            # Unknown operation
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # Map dataclasses -> Pydantic models
        dataset_statuses = [
            OperationPrereqDatasetStatus(
                table_key=d.table_key,
                name=d.name,
                owner_package=d.owner_package,
                schema_version=d.schema_version,
                is_view=d.is_view,
                has_repo_commit_rows=d.has_repo_commit_rows,
                checked=d.checked,
                error=d.error,
            )
            for d in debug_info.dataset_statuses
        ]

        run_summaries = [
            OperationPrereqRunSummary(
                run_id=r.run_id,
                kind=r.kind,
                status=r.status,
                pipeline_name=r.pipeline_name,
                requested_operation=r.requested_operation,
                started_at=r.started_at,
                completed_at=r.completed_at,
            )
            for r in debug_info.runs_considered
        ]

        return OperationPrereqDebugResponse(
            op_id=debug_info.op_id,
            repo=debug_info.repo,
            commit=debug_info.commit,
            required_datasets=debug_info.required_datasets,
            expanded_datasets=debug_info.expanded_datasets,
            required_graphs=debug_info.required_graphs,
            dataset_statuses=dataset_statuses,
            runs_considered=run_summaries,
            data_satisfied=debug_info.data_satisfied,
            run_satisfied=debug_info.run_satisfied,
            overall_satisfied=debug_info.overall_satisfied,
        )

    return router
```

Because `register_routes(app)` already includes `build_meta_router()`, this new endpoint will be exposed automatically.

You can tweak the path to include your `LOG_ROUTE_PREFIX` if you’ve standardized meta paths, e.g.:

```python
@router.get(
    f"{LOG_ROUTE_PREFIX}/pipeline/prereqs",
    ...
)
```

instead of `/debug/pipeline/prereqs`.

---

## 5. Example usage

Once wired:

```bash
# For current serving repo/commit
curl 'http://localhost:8080/debug/pipeline/prereqs?op_id=function.summary' | jq

# For a specific repo/commit
curl 'http://localhost:8080/debug/pipeline/prereqs?op_id=function.summary&repo=github.com/acme/project&commit=abc123'
```

Typical workflow:

* You hit a 500 from `/functions/summary`.
* You call `/debug/pipeline/prereqs?op_id=function.summary` and see:

  * Which datasets are missing rows.
  * Whether auto-pipeline already tried a prereq run and what it did.
* You can then decide whether:

  * Your config (`codeintel.yaml`) is wrong.
  * The pipeline isn’t producing datasets you expect.
  * Or auto-pipeline logic needs adjustment.

---

## 6. Tests

Add **`tests/http/test_debug_pipeline_prereqs.py`**.

### 6.1 Basic shape / 200 response

```python
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from codeintel.serving.http.fastapi import create_app
from codeintel.config.serving_models import ServingConfig
from codeintel.storage.gateway import StorageConfig, open_gateway

def _make_app(tmp_path: Path):
    db_path = tmp_path / "codeintel.duckdb"
    gw = open_gateway(StorageConfig.for_ingest(db_path))

    cfg = ServingConfig(
        mode="local_db",
        repo_root=tmp_path,
        repo="demo/repo",
        commit="HEAD",
        db_path=db_path,
    )

    app = create_app(config_loader=lambda: cfg, gateway=gw)
    return app, gw, cfg

def test_debug_prereqs_endpoint_basic(tmp_path: Path) -> None:
    app, gw, cfg = _make_app(tmp_path)
    client = TestClient(app)

    # Even with empty DB, the endpoint should respond with a structured payload.
    r = client.get("/debug/pipeline/prereqs", params={"op_id": "function.summary"})
    assert r.status_code == 200

    payload = r.json()
    assert payload["op_id"] == "function.summary"
    assert payload["repo"] == cfg.repo
    assert payload["commit"] == cfg.commit
    assert "required_datasets" in payload
    assert "dataset_statuses" in payload
    assert "runs_considered" in payload
```

### 6.2 Data-aware check reflected in debug response

Use the same seeding technique as in the previous refinement tests:

```python
import duckdb
from codeintel.serving.operations.catalog import get_operation
from codeintel.storage.sql_builder import SafeTable

def _seed_required_table(con: duckdb.DuckDBPyConnection, op_id: str, repo: str, commit: str) -> None:
    op = get_operation(op_id)
    assert op is not None
    table_key = op.required_datasets[0]
    table = SafeTable(table_key)
    con.execute(f"CREATE TABLE {table} (repo TEXT, commit TEXT, x INT)")  # noqa: S608
    con.execute(f"INSERT INTO {table} VALUES (?, ?, 1)", [repo, commit])

def test_debug_prereqs_shows_data_satisfied_when_row_present(tmp_path: Path) -> None:
    app, gw, cfg = _make_app(tmp_path)
    client = TestClient(app)

    _seed_required_table(gw.con, "function.summary", cfg.repo, cfg.commit)

    r = client.get("/debug/pipeline/prereqs", params={"op_id": "function.summary"})
    assert r.status_code == 200
    payload = r.json()

    assert payload["data_satisfied"] is True
    # At least one dataset_status should show has_repo_commit_rows=True
    assert any(
        ds.get("has_repo_commit_rows") is True
        for ds in payload["dataset_statuses"]
    )
```

You can add more tests:

* When you insert a successful full run into `metadata.pipeline_runs`, `run_satisfied` becomes `True` and `runs_considered` contains a `kind="full"` run.
* When op_id is unknown → 404.

---

### Wrap-up

This `/debug/pipeline/prereqs` endpoint gives you an “X-ray” of auto-pipeline’s decision:

* Exactly which datasets and runs it looked at.
* Whether the data-aware and run-based checks passed.
* And how that rolled up into the final `overall_satisfied` used to skip or trigger `ensure_prerequisites_for_operation`.

It’s also a solid foundation for future tooling:

* You can build UI panels around it (“Why is this op slow / failing?”).
* You can wire a `codeintel op debug-prereqs` CLI that simply calls this endpoint or reuses `build_prereq_debug_info` directly.


