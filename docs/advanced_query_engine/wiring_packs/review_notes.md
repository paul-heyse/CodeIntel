# Wiring Packs Review Notes

Scope: `docs/advanced_query_engine/wiring_packs` (packs and executor code).

## Functional Errors (Blocking)

1) rpygrep import failure halts candidate discovery.
   - `DEFAULT_EXCLUDED_TYPES` is not exported by rpygrep 0.2.1, so the import fails and
     `RipGrepSearch` is set to `None`, making every pack execution raise at runtime.
   - File: `docs/advanced_query_engine/wiring_packs/wiring_executor/engines/rpygrep_runner.py:11`

2) rpygrep JSON parsing is enabled in the code path but never requested on the command.
   - `run_direct()` emits plain text unless `as_json()` is set. Presets define `as_json`,
     but `_apply_preset` does not apply it, so `json.loads(line)` fails and yields no hits.
   - File: `docs/advanced_query_engine/wiring_packs/wiring_executor/engines/rpygrep_runner.py:40`
   - File: `docs/advanced_query_engine/wiring_packs/wiring_executor/engines/rpygrep_runner.py:151`

## Improvement Opportunities (Non-Blocking)

1) Argparse entry keys can degrade to "<missing:ARGPARSE_CMD>" when join fails.
   - Consider a fallback entry key for `py.argparse.subparser.set_defaults` when no
     `ARGPARSE_CMD` is derived (e.g., use `{HANDLER}` or `{enclosing_name}`).
   - File: `docs/advanced_query_engine/wiring_packs/packs/wiring_packs/python/argparse_cli.json:41`

2) FastAPI and Flask HTTP method coverage can be improved.
   - Capture `methods=[...]` or `methods=...` in `route/api_route/add_api_route` calls
     to include method semantics when decorators are not used.
   - File: `docs/advanced_query_engine/wiring_packs/packs/wiring_packs/python/fastapi_routes.json`
   - File: `docs/advanced_query_engine/wiring_packs/packs/wiring_packs/python/flask_routes.json`

3) Plugin entrypoint detection should include newer importlib patterns.
   - Add patterns for `importlib.metadata.entry_points().select(group=...)` and the
     positional `entry_points("group")` variant.
   - File: `docs/advanced_query_engine/wiring_packs/packs/wiring_packs/python/entrypoints.json`
   - File: `docs/advanced_query_engine/wiring_packs/packs/ast_grep/python/entrypoints.yaml`

4) Typer wiring precision can be tightened.
   - Link `@app.command` matches to a captured `app = typer.Typer(...)` instance to
     reduce false positives in repos that use multiple CLI patterns.
   - File: `docs/advanced_query_engine/wiring_packs/packs/wiring_packs/python/typer_cli.json`
   - File: `docs/advanced_query_engine/wiring_packs/packs/ast_grep/python/typer.yaml`
