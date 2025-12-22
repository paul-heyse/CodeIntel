## Context
Serving, storage, and CLI currently handle observability and SQL fingerprinting in separate
modules with overlapping responsibilities. Export payload handling and ID generation also
vary across components. This change consolidates these cross-cutting concerns into canonical
modules and applies consistent patterns across CLI, HTTP, MCP, and storage.

## Goals / Non-Goals
- Goals:
  - Centralize observability bootstrap and context propagation across CLI, HTTP, and MCP.
  - Make OpenTelemetry metrics the canonical metrics pipeline.
  - Enable DuckDB tracing by default with safe SQL redaction and explicit env toggles.
  - Unify SQL fingerprinting and provide semantic diffs for upgrade diagnostics.
  - Standardize ID generation with UUIDv7 when available.
  - Improve export responses (ResourceContent + UTF-8 NDJSON streaming).
- Non-Goals:
  - Rework the serving API surface beyond the additions specified here.
  - Introduce new third-party dependencies beyond those already in pyproject.toml.
  - Change external compatibility contracts (project is in design phase and can migrate fully).

## Decisions
- Observability is bootstrapped once per process via a shared module that configures OTel
  tracing and metrics; CLI, HTTP, and MCP call the same bootstrap.
- OpenTelemetry metrics are canonical; Prometheus exposure is via the OTel Prometheus
  exporter and /metrics endpoints, not direct prometheus_client counters.
- DuckDB tracing is default-on when OTel is enabled. SQL statements are redacted by
  default to an operation + hash display, with explicit env toggles for full/operation/none.
- SQL fingerprinting uses storage SQLGlot canonicalization (normalize -> qualify ->
  optimize -> render) with a safe fallback to hashing raw SQL when parsing fails.
- SemanticQueryResponse includes sql_fingerprint derived from canonical SQL when compiled
  SQL is available; computed in the kernel so HTTP and MCP share the same value.
- UUID generation uses a single factory that prefers UUIDv7 (uuid6) and falls back to
  UUIDv4; correlation/debug/instance IDs and run/job IDs migrate to this factory.
- NDJSON encoding uses a shared encoder with a msgspec fast path and stdlib fallback;
  output is UTF-8 with compact separators and stringified non-JSON types (HTTP + MCP).
- MCP export reads return content with registry MIME; export metadata is provided via
  codeintel://exports/{export_id}/meta and MAY be surfaced in resource listing _meta.
  Per-read metadata requires a custom read handler or FastMCP extension if needed.

## Risks / Trade-offs
- SQL fingerprint values will change after canonicalization unification; acceptable for a
  design-phase migration.
- OTel instrumentation adds overhead; mitigated by opt-out env toggles and no-op behavior
  when exporters are absent.
- SQL redaction may reduce debugging detail; mitigated by configurable statement modes.
- /metrics endpoints can expose operational data; mitigated by optional auth gating.
- NDJSON encoding changes may affect downstream consumers; mitigated by parity tests.
- FastMCP read_resource does not currently surface per-read metadata; metadata is provided
  via export meta resource and resource listings instead.

## Migration Plan
1. Introduce shared observability and context modules.
2. Wire CLI/HTTP/MCP to the shared bootstrap and metrics helpers.
3. Implement DuckDB tracing with redaction and hook into DuckDBSession.
4. Unify SQL fingerprinting and add semantic diff helper, then update call sites.
5. Add UUID factory and migrate call sites across the codebase.
6. Update export handling (shared NDJSON encoder; MCP metadata via export meta resource
   and resource listing) and add tests.
7. Add sql_fingerprint to SemanticQueryResponse and update HTTP/MCP usage + tests.
8. Run quality checks and targeted tests for observability, SQL diffs, and exports.

## Open Questions
- None.
