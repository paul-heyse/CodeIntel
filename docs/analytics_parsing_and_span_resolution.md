# Analytics parsing, span resolution, and validation

## Parser registry
- Function analytics selects parsers via `FunctionParserKind` (enum) and `FunctionParserRegistry`.
- Default is Python AST; new languages/parsers should register in the registry rather than passing ad-hoc callables.
- CLI flag `--function-parser` and config overrides map strings to the enum; invalid values fail fast.

## Span resolution
- `resolve_span` (in `src/codeintel/analytics/span_resolver.py`) provides structured outcomes for GOID spans: `ok`, `missing_span`, `missing_index`.
- Analytics should use this helper instead of direct AST lookups to keep validation consistent and cache-friendly.

## Validation reporting
- `ValidationReporter` emits structured counts (`function_validation.*`) to logs and can be extended to metrics sinks.
- Prefect orchestration passes a reporter so per-run validation gaps are visible; hook into your metrics/alerting as needed.
