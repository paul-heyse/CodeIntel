# Change: Remove Legacy and Compatibility Code Paths

## Why
Legacy shims and compatibility aliases remain across build, storage, serving, and core layers,
creating ambiguous APIs, extra maintenance surface, and behavior that no longer matches the
current architecture. Removing these paths clarifies the canonical interfaces and enforces
modern boundaries.

## What Changes
- **BREAKING** Remove legacy row model migration APIs and RowBinding in favor of
  schema-generated row bindings with provenance metadata.
- **BREAKING** Remove the build-layer full declared schema provider and require source-only
  providers for build contract enumeration.
- **BREAKING** Remove wrapper-based build target implementations and allowlists; require
  native Hamilton targets only.
- **BREAKING** Remove legacy execution profile alias "default" and the ValidationResult
  alias; enforce canonical configuration identifiers and types.
- **BREAKING** Remove the FastMCP compatibility shim and use direct fastmcp imports.
- **BREAKING** Remove storage exception re-export compatibility and DuckDB fallback stubs;
  use canonical error types and require DuckDB at runtime.
- **BREAKING** Remove graph compatibility fields from core parsing models and rely on
  graph-specific parsing types.
- Update tests and docs to reflect the canonical surfaces.

## Impact
- Affected specs: build-execution (new), config-injection, contract-resolution,
  parsing-models (new), schema-contracts, serving-interfaces (new), storage-boundaries.
- Affected code:
  - Build: `src/codeintel/build/hamilton/planner.py`,
    `src/codeintel/build/hamilton/contracts/schemas/row_migration.py`,
    `src/codeintel/build/schemas/provider_declared.py`.
  - Core: `src/codeintel/core/options/protocol.py`,
    `src/codeintel/core/plugins/execution/profiles.py`,
    `src/codeintel/core/parsing/models.py`,
    `src/codeintel/core/schemas/contract_primitives.py`,
    `src/codeintel/core/schemas/row_models.py`.
  - Storage: `src/codeintel/storage/exceptions.py`,
    `src/codeintel/storage/gateway/protocol.py`.
  - Serving: `src/codeintel/serving/mcp/_compat.py` and MCP import sites.
