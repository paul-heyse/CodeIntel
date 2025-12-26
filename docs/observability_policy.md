# Observability Policy

## Overview

The observability policy defines how CodeIntel shapes telemetry attributes, enforces
cardinality budgets, and applies redaction. It is the single source of truth for
attribute allowlists and per-surface limits used across traces, metrics, logs, and
DB spans.

## Policy Contract

The policy is implemented in `codeintel.observability.policy.ObservabilityPolicy` and
includes:

- Operation attribute allowlist and per-operation overrides.
- DB attribute prefix allowlist.
- Budgets for CLI arg names, HTTP routes, and MCP tool names.
- Redaction rules for command/path values in teardown telemetry.

## Attribute Taxonomy

### Operation Attributes

Default allowlist (low-cardinality):

- `codeintel.correlation_id`
- `codeintel.output_format`
- `http.method`
- `http.route`
- `mcp.method`
- `mcp.tool_name`

Per-operation overrides may be provided to expand or narrow this list.

### DB Attributes

DB span attributes allow `codeintel.*` and `db.*` prefixes.

## Budgets and Cardinality Guardrails

Default budgets:

- CLI arg names: `25`
- HTTP route length: `120`
- MCP tool name length: `80`

Truncation uses a trailing `.` to indicate a truncated value.

## Redaction

Teardown telemetry redacts sensitive values while preserving a short suffix:

- Commands keep `1` trailing segment.
- Paths keep `1` trailing segment.

Both values are configurable via the policy.

## Per-Operation Overrides

Overrides are keyed by either:

- `component` (applies to all operations in that component), or
- `component.operation` (highest priority).

Example JSON payload:

```json
{
  "cli": ["http.method"],
  "cli.health": ["codeintel.output_format", "codeintel.correlation_id"]
}
```

## Configuration and Precedence

### Runtime Settings

Settings map into the policy during bootstrap:

- `CODEINTEL_OBSERVABILITY_CLI_ARG_NAMES_MAX`
- `CODEINTEL_OBSERVABILITY_HTTP_ROUTE_MAX_LEN`
- `CODEINTEL_OBSERVABILITY_MCP_TOOL_NAME_MAX_LEN`
- `CODEINTEL_OBSERVABILITY_OPERATION_ALLOWLIST_OVERRIDES` (JSON mapping)

### Config File Mode

If `OTEL_EXPERIMENTAL_CONFIG_FILE` is set, the SDK is enabled regardless of
`OTEL_SDK_DISABLED`. The config file is validated before being applied and will
fail fast with explicit error messages if malformed or missing required sections.
