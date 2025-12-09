# CodeIntel CLI usage

## Configuration precedence
- CLI arguments override everything else.
- Environment variables with the `CODEINTEL_` prefix (including `CODEINTEL_CONFIG_PATH`)
  are applied next. The default prefix-based loader also honors `codeintel.toml` when
  present in the working directory or at the configured path.
- Library defaults are used only when neither CLI nor env/config provide a value.
- Errors are normalized via `handle_cli_error`, yielding usage errors as exit 2 and
  validation errors as exit 1 with rich-friendly stderr output.

## Result handling and embedding
- The root Cyclopts app defaults `result_action` to `["call_if_callable", "return_value"]`
  so commands can be embedded in tests or agents without invoking `sys.exit`.
- Parse-only harness example:
  ```python
  from codeintel.cli.cyclopts_app import app

  ns = app(
      ["op", "call", "--operation-id", "example"],
      result_action="return_value",
      exit_on_error=False,
      print_error=False,
  )
  ```
- Asynchronous invocation is also supported: `await app.run_async(args, result_action="return_value")`
  for embedding in event loops.

## Parse-time validation semantics
- Build `run` requires exactly one of targets, `--module`, or `--all`; conflicting inputs
  fail before handlers execute.
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
