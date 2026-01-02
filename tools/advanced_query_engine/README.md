# Advanced Query Engine - Usage Guide

This guide explains how to run the advanced query engine CLI, what each query returns,
how to control scope and budgets, and how to validate outputs.

## What it is

The advanced query engine runs a small set of search handlers that map to eight search
intents:

1) symbol.resolve
2) refs.find
3) callgraph.slice
4) pattern.scan
5) contract.lookup
6) wiring.map
7) precedent.search
8) impact.slice

The CLI prints JSON to stdout. You can view it directly in the terminal or pipe it to
`jq` or a file.

## Pack roots

- Query packs (rpygrep, ast-grep, tree-sitter):
  `docs/advanced_query_engine/query_packs`
- Wiring packs (framework wiring maps):
  `docs/advanced_query_engine/wiring_packs/packs`

## CLI quickstart

```bash
uv run python -m tools.advanced_query_engine.cli \\
  --repo . \\
  --type refs.find \\
  --text compose_runtime
```

The JSON is printed to stdout. To save:

```bash
uv run python -m tools.advanced_query_engine.cli --repo . --type refs.find --text X \\
  > /tmp/aqe.json
```

## Common request fields

All query types accept:

- `--repo` repository root.
- `--type` one of the eight query types.
- `--text` query text (symbol name, pattern, or target).
- `--scope` repeatable repo-relative paths to constrain scanning.
- `--budget-json` JSON string matching `QueryBudget`:
  - `max_files` max candidate files (0 means no cap).
  - `max_matches` max matches (0 means no cap).
  - `max_depth` max directory depth (0 means unlimited).
  - `max_seconds` optional time budget (seconds).
  - `context_lines` context lines for snippets.
- `--options-json` or `--options-file` for query-specific options.

Default scope is the repo root. For performance, prefer `--scope src --scope tests`.

## Output shape

Every response includes:

- `summary`: human-readable summary.
- `primary`: main results list.
- `related`: auxiliary groups or secondary outputs.
- `debug`: engine metadata and partial/budget hints.

If any backend hits limits, `debug` includes flags such as `rg_partial`,
`ast_partial`, `ts_partial`, or `budget_exhausted`.

## Query-specific guidance

### 1) symbol.resolve

Finds definition records for a symbol name.

Example:
```bash
uv run python -m tools.advanced_query_engine.cli \\
  --repo . --type symbol.resolve --text SearchService
```

Output highlights:
- `primary`: list of symbol records with `symbol_id`, `def_span`, `signature`, `docstring`.

### 2) refs.find

Enumerates usage sites and classifies them as call/import/read/write/inherit.

Example:
```bash
uv run python -m tools.advanced_query_engine.cli \\
  --repo . --type refs.find --text compose_runtime --scope src
```

Output highlights:
- `primary`: records with `role`, `confidence`, `rank`, `snippet`, `enclosing`.
- `related.groups`: counts by module and by kind (prod/test/doc/example).
- `related.overrides`: method overrides with signature/docstring.

### 3) callgraph.slice

Returns incoming and outgoing call edges with callsite spans and arg maps.

Example:
```bash
uv run python -m tools.advanced_query_engine.cli \\
  --repo . --type callgraph.slice --text compose_runtime --scope src
```

Output highlights:
- `related.calls_in` and `related.calls_out`: each edge includes `call_span`,
  `callsite` snippet, `arguments`, and `arg_map` (when signature is available).

### 4) pattern.scan

Runs a lexical or structured scan. Options choose backends.

Common options:
- `pattern_group_id`: rpygrep group id.
- `ast_grep_pack_id`, `rule_ids`: ast-grep pack + rule ids.
- `tree_sitter_pack_id`: tree-sitter pack id.

Example (rpygrep):
```bash
uv run python -m tools.advanced_query_engine.cli \\
  --repo . --type pattern.scan --text \"TODO\" \\
  --options-json '{\"pattern_group_id\":\"rg.precedent.search\"}'
```

Example (ast-grep):
```bash
uv run python -m tools.advanced_query_engine.cli \\
  --repo . --type pattern.scan --text \"\" \\
  --options-json '{\"ast_grep_pack_id\":\"ag.python.policy_security\", \"rule_ids\":[\"py.security.eval\"]}'
```

### 5) contract.lookup

Finds tests/docs/examples that reference a symbol, and extracts test assertions.

Example:
```bash
uv run python -m tools.advanced_query_engine.cli \\
  --repo . --type contract.lookup --text compose_runtime --scope tests --scope docs
```

Output highlights:
- `primary`: test records with `assertions`, `references`, and a test snippet.
- `related.docs` and `related.examples`: documentation/example hits.

### 6) wiring.map

Runs wiring packs (FastAPI, Flask, Click, Typer, argparse, env, entrypoints).

Options:
- `pack_ids`: list of wiring pack ids to run.
- `allow_cross_file_resolution`: resolve handler symbols across files.

Example:
```bash
uv run python -m tools.advanced_query_engine.cli \\
  --repo . --type wiring.map --text \"\" \\
  --options-json '{\"pack_ids\":[\"wire.python.fastapi.routes\"]}'
```

Output highlights:
- `primary`: wiring edges with `entry_key`, `hook_span`, `target`, and `config`.
- `related.by_pack`: per-pack results and validation findings.

### 7) precedent.search

Finds similar definitions based on signature/decorators/docstring overlap.

Example:
```bash
uv run python -m tools.advanced_query_engine.cli \\
  --repo . --type precedent.search --text \"compose_runtime\" --scope src
```

Options:
- `k`: top-K results (default 5).
- `pattern_group_id`: optional rpygrep pattern group for candidate pool.

Output highlights:
- `primary`: scored exemplars with `why` explanations.

### 8) impact.slice

Builds a bounded transitive slice of callers/callees plus boundary crossings.

Options:
- `slice`: object with `caller_depth`, `callee_depth`.
- `package_depth`: depth for boundary crossings.

Example:
```bash
uv run python -m tools.advanced_query_engine.cli \\
  --repo . --type impact.slice --text compose_runtime \\
  --options-json '{\"slice\":{\"caller_depth\":2, \"callee_depth\":1}, \"package_depth\":2}'
```

Output highlights:
- `primary`: node list in the slice.
- `related.edges`: transitive call edges.
- `related.calls_in/out`: 1-hop edges from callgraph.slice.
- `related.boundary_crossings` and `related.risk`.

## Persistence and analytics

The CLI can persist results and optionally run analytics:

```bash
uv run python -m tools.advanced_query_engine.cli \\
  --repo . --type pattern.scan --text TODO \\
  --persist --analytics --validate-persisted
```

Relevant flags:
- `--persist`, `--persist-path`, `--persist-partition-by`
- `--analytics`, `--analytics-profile`, `--analytics-chunk-size`, `--analytics-max-rows`
- `--validate-persisted`

## Validation harness

Run independent checks across the eight query types:

```bash
uv run python -m tools.advanced_query_engine.validation_harness \\
  --repo-root . --scope src --scope tests
```

The default report is:
`build/advanced_query_engine/validation_report.json`

You can override cases with `--case` or provide a JSON config via `--config`.

## Troubleshooting

- If a query is slow, add `--scope` (prefer `src` and `tests`) and reduce `max_files`.
- If you see `rg_partial` or `budget_exhausted`, raise budgets or tighten scope.
- If wiring packs emit validation errors, fix the pack captures or templates.
