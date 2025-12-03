
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

