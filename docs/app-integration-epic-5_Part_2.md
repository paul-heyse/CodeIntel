
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


