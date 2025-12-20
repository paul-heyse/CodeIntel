# DAG-First Boundaries and Settings Injection

## Contract derivation

- Dataset contracts are derived through the core factory in
  `codeintel.core.schemas.contract_factory`.
- Build and storage providers should not duplicate view/owner/tag logic.
- For tests, `ContractResolutionSettings` accepts `target_metadata_provider`
  and `output_inventory` to avoid loading the Hamilton metadata service.

## Output inventory

- Output inventory is computed without initializing the Hamilton driver via
  `codeintel.build.target_inventory.get_output_inventory`.
- Use `OutputInventory` injection when you need deterministic filtering in
  schema-only contract resolution.

## Export boundary

- Build exports use `gateway.exports` for relation creation and audit logging.
- DuckDB connection access stays within `codeintel.storage.exports.service`.

## Settings injection

- Canonical settings live in `codeintel.core.config.settings`.
- CLI entrypoints load environment configuration (`get_build_settings`,
  `get_serving_settings`) and pass settings explicitly to runtime builders.
- HTTP serving factories require a `ServingSettings` instance; for multi-worker
  Uvicorn runs, use `codeintel.cli.serving_factory:create_serving_app_from_env`.

## Error payloads

- Serving HTTP and MCP error responses use the core RFC 9457 `ProblemDetail`.
- The adapter in `codeintel.serving.errors.problem_adapter` ensures consistent
  extensions (`code`, `kind`, `retryable`, `hint`, `correlation_id`).

## Export formats

- Canonical registry: `codeintel.core.exports.formats`.
- `ndjson` is treated as an alias of `jsonl`; serving defaults remain `ndjson`.
