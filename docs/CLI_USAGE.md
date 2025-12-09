# CodeIntel CLI usage

## Configuration precedence

CLI arguments > environment variables > TOML config > library defaults:

1. **CLI arguments** override everything else.
2. **Environment variables** with the `CODEINTEL_` prefix (including `CODEINTEL_CONFIG_PATH`)
  are applied next. The default prefix-based loader also honors `codeintel.toml` when
  present in the working directory or at the configured path.
3. **Library defaults** are used only when neither CLI nor env/config provide a value.

Errors are normalized via `handle_cli_error`, yielding usage errors as exit 2 and
  validation errors as exit 1 with rich-friendly stderr output.

## Runtime and output defaults

The CLI provides shared dataclass bundles for common options:

### RuntimeCLI defaults

```python
RuntimeCLI(
    project_root=None,   # Explicit project root directory (--root/-r)
    repo=None,           # Repository slug (--repo)
    commit=None,         # Commit SHA (--commit)
    db_path=None,        # DuckDB database path (--db-path)
    build_dir=None,      # Build directory (--build-dir)
    repo_root=None,      # Repository root path (--repo-root)
    document_output_dir=None,  # Document output override (--document-output-dir)
    verbose=0,           # Verbosity level (--verbose/-v)
)
```

Use `runtime_field()` in command dataclasses to inherit these options with Cyclopts
parameter metadata for nested flattening.

### OutputFormatCLI defaults

```python
OutputFormatCLI(
    output_format=OutputFormat.TEXT,  # Output format (--output-format)
    json=False,                       # JSON shortcut (--json)
)
```

Use `output_field()` for commands needing output format control. The `--json` flag
takes precedence over `--output-format`.

## Result handling and embedding

The root Cyclopts app defaults `result_action` to `["call_if_callable", "return_value"]`
  so commands can be embedded in tests or agents without invoking `sys.exit`.

### Parse-only invocation

  ```python
from codeintel.cli.cyclopts_ops import app_proxy

# Parse arguments and return namespace without executing command
ns = app_proxy(
    ["op", "list", "--category", "core"],
      result_action="return_value",
      exit_on_error=False,
      print_error=False,
  )

# Access parsed values from the namespace
category = ns.kwargs.get("category")
```

### Getting the root app

```python
from codeintel.cli.cyclopts_ops import get_app

# Get the initialized root Cyclopts application
app = get_app()
```

### Asynchronous invocation

For embedding in event loops:

```python
await app.run_async(args, result_action="return_value")
```

## Path validator semantics

Dynamic CLI commands apply path validators based on parameter naming heuristics:

| Parameter Pattern | Default | Validation |
|------------------|---------|------------|
| `*_env*`, `env*`, `*venv*` | `.venv` | Must exist as directory |
| `*output*`, `*dest*` | (none) | Parent directory must exist; file can be missing |
| Other paths | (none) | Must exist (file or directory) |

Use the `path_validator()` helper to create custom validators:

```python
from codeintel.cli.cyclopts_ops import path_validator

# Require existing directory
validator = path_validator(require_exists=True, require_dir=True)

# Allow missing file (output path)
validator = path_validator(require_exists=False)
```

## Shared helper contract

### make_handler_context

Extract runtime options, verbosity, and output format from CLI dataclasses:

```python
from codeintel.cli.cyclopts_common import make_handler_context

runtime_opts, verbose, output_format = make_handler_context(
    runtime_cli,     # RuntimeCLI instance
    output_cli,      # OutputFormatCLI instance
    default_output=OutputFormat.TEXT,
)
```

### runtime_required

Validate that required runtime fields are present:

```python
from codeintel.cli.cli_errors import runtime_required, ValidationError

# Raises ValidationError if repo or commit is None
runtime_required(
    cli_runtime,
    "history command",
    require_repo=True,
    require_commit=True,
    require_db_path=False,
)
```

## Parse-time validation semantics

- Build `run` requires exactly one of targets, `--module`, or `--all`; conflicting inputs
  fail before handlers execute with a clear error message.
- Docs export enforces exclusivity pairs: `validation_mode` vs `--validate`,
  `run_mode` vs `--dry-run`, and `prereq_mode` vs `--skip-prereqs`.
- Dynamic operation commands group parameters by intent (Target Selection, Filtering,
  Advanced) with choice-aware flags (enums/literals render `show_choices=True`).
- Structured types validate at parse time: paths verify existence (or parent for output
  paths), boolean flags have negative forms disabled where inappropriate, and numeric
  options coerce from strings with early errors on invalid input.

## Help robustness

- Help output is patched per-app to render nested/grouped defaults without leaking
  internal `SimpleNamespace` representations, and to keep choice/default metadata readable
  for grouped parameters.
