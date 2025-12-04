Here’s an aggressive, “no-compat-left” plan to remove all argparse-style helpers from the CLI and leave only the Typer / explicit APIs. 

I’ll focus on the three concrete helpers plus the docstring cleanup, and show before/after snippets you can apply.

---

## 1. What we’re removing

There are three argparse-style helpers that exist purely for “test compatibility”:

* `cli/cli/commands/docs.py`

  * `cmd_docs_export(args: object) -> int`
* `cli/cli/commands/datasets.py`

  * `build_scaffold_options(args: object, registry: DatasetRegistry | None = None) -> ScaffoldOptions`
  * `run_datasets_catalog(args: object) -> int`

They are:

* Not called anywhere in the codebase (only defined + exported).
* Duplicating logic that already exists in Typer commands and internal helpers.

Goal: **delete them outright**, update `__all__`, and update docs/tests to use the canonical Typer / helper APIs.

---

## 2. `docs.py`: remove `cmd_docs_export` and clean up exports

### 2.1. Delete `cmd_docs_export`

In `cli/cli/commands/docs.py`, locate:

```python
def cmd_docs_export(
    args: object,
    validator: Callable[[StorageGateway], None] = validate_dataset_registry,
    export_runner: ExportRunner = run_validated_exports,
    gateway_factory: GatewayFactory | None = None,
) -> int:
    """Execute docs export from argparse namespace (test compatibility).

    Parameters
    ----------
    args
        Argparse namespace with CLI options.
    ...
    """
    ...
    cfg = CodeIntelConfig.from_cli_args(...)
    ...
    try:
        _run_docs_export(
            cfg=cfg,
            validate_exports=bool(getattr(args, "validate_exports", False)),
            schemas=list(schemas_raw) if schemas_raw else None,
            datasets=list(datasets_raw) if datasets_raw else None,
            require_normalized_macros=bool(
                getattr(args, "require_normalized_macros", False)
            ),
            validator=validator,
            export_runner=export_runner,
        )
    except ExportError as exc:
        log_problem(LOG, exc.detail)
        return 1

    LOG.info("Export complete.")
    return 0
```

**Action:** delete this function entirely (docstring + body).

All behaviour it provided is already covered by:

* The Typer command `docs_export(...)` (for real CLI usage).
* The internal helper `_run_docs_export` (which tests can call directly with a `CodeIntelConfig`).

### 2.2. Clean up `__all__` and the alias

At the bottom of `docs.py` you currently have:

```python
__all__ = [
    "GatewayFactory",
    "cmd_docs_export",
    "docs_app",
    "run_docs_export",
]

# Alias for test compatibility
run_docs_export = _run_docs_export
```

We want no compat alias and no argparse helper.

**Change to:**

1. Rename `_run_docs_export` → `run_docs_export`:

   ```python
   def run_docs_export(
       cfg: CodeIntelConfig,
       validate_exports: bool,
       schemas: list[str] | None,
       datasets: list[str] | None,
       require_normalized_macros: bool,
       validator: Callable[[StorageGateway], None],
       export_runner: ExportRunner,
   ) -> None:
       """Execute the docs export with provided configuration and callbacks."""
       ...
   ```

2. Update the Typer command to call the new name:

   ```python
   @docs_app.command("export")
   def docs_export(...):
       ...
       cfg = _build_runtime_config_from_cli(...)
       run_docs_export(
           cfg=cfg,
           validate_exports=validate,
           schemas=list(schemas) if schemas else None,
           datasets=list(datasets) if datasets else None,
           require_normalized_macros=require_normalized_macros,
           validator=validate_dataset_registry,
           export_runner=run_validated_exports,
       )
   ```

3. Update `__all__` and remove the alias:

   ```python
   __all__ = [
       "GatewayFactory",
       "docs_app",
       "run_docs_export",
   ]
   ```

   Delete the `# Alias for test compatibility` comment and the `run_docs_export = _run_docs_export` line.

**Tests update (what you’ll need to change):**

* Any test that previously imported `cmd_docs_export` and passed an argparse namespace should instead:

  * Build a `CodeIntelConfig` using the same helpers as the Typer command (or re-use a helper you already have in tests), then
  * Call `run_docs_export(cfg, ...)` directly with explicit keyword args.

---

## 3. `datasets.py`: remove `build_scaffold_options` and `run_datasets_catalog`

### 3.1. Delete `build_scaffold_options`

In `cli/cli/commands/datasets.py`, find:

```python
def build_scaffold_options(
    args: object,
    registry: DatasetRegistry | None = None,
) -> ScaffoldOptions:
    """Build scaffold options from argparse namespace (test compatibility).

    Parameters
    ----------
    args
        Argparse namespace with scaffold options.
    registry
        Optional registry for validation.

    Returns
    -------
    ScaffoldOptions
    """
    return _build_scaffold_options(
        name=str(getattr(args, "name")),
        kind=str(getattr(args, "kind", "table")),
        table_key=getattr(args, "table_key", None),
        owner=getattr(args, "owner", None),
        freshness_sla=getattr(args, "freshness_sla", None),
        retention_policy=getattr(args, "retention_policy", None),
        schema_version=str(getattr(args, "schema_version", "v1")),
        validation_profile=str(getattr(args, "validation_profile", "strict")),
        schema_id=getattr(args, "schema_id", None),
        jsonl_filename=getattr(args, "jsonl_filename", None),
        parquet_filename=getattr(args, "parquet_filename", None),
        stable_id=getattr(args, "stable_id", None),
        specs_snapshot=Path(getattr(args, "specs_snapshot", "specs.json")),
        output_dir=Path(getattr(args, "output_dir", "datasets")),
        overwrite=bool(getattr(args, "overwrite", False)),
        dry_run=bool(getattr(args, "dry_run", False)),
        emit_bootstrap_snippet=bool(getattr(args, "emit_bootstrap_snippet", False)),
        registry=registry,
    )
```

**Action:** delete this function entirely.

The canonical builder is `_build_scaffold_options(...)` and is already used by the Typer command `datasets_scaffold`.

### 3.2. Delete `run_datasets_catalog`

Still in `datasets.py`, find:

```python
def run_datasets_catalog(args: object) -> int:
    """Run dataset catalog generation from argparse namespace (test compatibility).

    Parameters
    ----------
    args
        Argparse namespace with catalog options.
    """
    from duckdb import DuckDBError
    from codeintel.storage.config import StorageConfig
    from codeintel.storage.gateway import open_gateway

    warnings_seen: set[str] = set()
    ...
    db_path = Path(getattr(args, "db_path", ""))
    output_dir = Path(getattr(args, "output_dir", ""))
    sample_rows = int(getattr(args, "sample_rows", 0))
    sample_rows_strict = bool(getattr(args, "sample_rows_strict", False))
    ...
    entries = build_catalog_entries(
        gateway,
        registry,
        options=CatalogOptions(
            include_datasets=None,
            sample_rows=sample_rows,
            sample_rows_strict=sample_rows_strict,
        ),
        warn=_warn,
    )
    ...
    write_markdown_catalog(output_dir, entries)
    write_html_catalog(output_dir, entries)
    ...
    return 0
```

**Action:** delete this function entirely.

The Typer `datasets_catalog` command already wraps `build_catalog_entries` and the same output functions; there is no unique functionality here.

### 3.3. Update `__all__` in `datasets.py`

At the bottom you have:

```python
__all__ = [
    "ScaffoldConfigError",
    "build_scaffold_options",
    "datasets_ext_app",
    "run_datasets_catalog",
]
```

**Change to:**

```python
__all__ = [
    "ScaffoldConfigError",
    "datasets_ext_app",
]
```

No other code in the repo imports `build_scaffold_options` or `run_datasets_catalog`, so this is safe.

**Tests update (what you’ll need to change):**

* Tests that called `build_scaffold_options(args)` should instead:

  * Call `_build_scaffold_options(...)` directly with explicit arguments, **or**
  * Invoke the Typer command `datasets_scaffold` via Typer’s test runner if you want an end-to-end test of CLI parsing.
* Tests that called `run_datasets_catalog(args)` should instead:

  * Use `build_catalog_entries(...)` directly to get the catalog, and assert on the entries, or
  * Invoke `datasets_catalog` via Typer.

---

## 4. Clean up CLI docstrings mentioning the “legacy argparse CLI”

Since we just removed the last remaining argparse-based shims, the docs can also stop talking about “legacy argparse.”

### 4.1. `cli/cli/__init__.py`

Current top docstring:

```python
"""CodeIntel unified CLI entry point.

This module provides the unified Typer-based CLI for CodeIntel, consolidating
all functionality from both the legacy argparse CLI and the newer Typer-based
application CLI into a single coherent interface.
...
"""
```

**Change to something like:**

```python
"""CodeIntel unified CLI entry point.

This module provides the Typer-based CLI for CodeIntel, exposing all functional
areas (pipeline, ingest, graph, docs, datasets, etc.) under a single interface.
...
"""
```

Just remove references to “legacy argparse CLI.”

### 4.2. `cli/cli/commands/__init__.py`

Current:

```python
"""CLI command modules for the unified CodeIntel CLI.

This package contains Typer command groups migrated from the legacy argparse CLI,
organized by functional area.
...
"""
```

**Change to:**

```python
"""CLI command modules for the unified CodeIntel CLI.

This package contains Typer command groups organized by functional area.
...
"""
```

Again, drop “migrated from the legacy argparse CLI.”

---

## 5. Optional: tighten up docs export API naming

This goes slightly beyond “argparse helpers,” but if you want **zero** “test compatibility” references:

* We already renamed `_run_docs_export` → `run_docs_export` and removed the alias + comment.
* That means there is no “test compatibility” language anywhere in `docs.py`.

If you want to keep a strict public/private split, you could:

* Leave the function named `run_docs_export` and treat it as the supported programmatic API, or
* Prefix it with `_` again and only expose the Typer CLI; in that case, just remove `run_docs_export` from `__all__` as well.

Either way, you’ve removed the compatibility alias and all argparse-based shims.

---

## 6. Final checklist

You’re done when:

* [ ] No functions named `cmd_docs_export`, `build_scaffold_options`, or `run_datasets_catalog` exist.
* [ ] No references to those names remain in the codebase (including `__all__`).
* [ ] The only ways to drive docs export and dataset management are:

  * Typer commands (`docs_export`, `datasets_catalog`, `datasets_scaffold`, etc.), and/or
  * Their explicit helper functions (`run_docs_export`, `_build_scaffold_options`, `build_catalog_entries`, etc.).
* [ ] CLI docstrings no longer mention “legacy argparse CLI.”
* [ ] Tests are updated to call the new explicit APIs or Typer commands rather than passing an `argparse.Namespace` into helpers.

Once this is done, the CLI layer is **purely Typer-based with explicit helpers** and has **no remaining argparse compatibility code**.
