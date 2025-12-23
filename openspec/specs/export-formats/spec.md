# export-formats Specification

## Purpose
TBD - created by archiving change refactor-dag-first-consolidation. Update Purpose after archive.
## Requirements
### Requirement: Canonical export format registry
The system SHALL define a single export format registry shared by build and serving that
specifies supported formats, MIME types, and default file suffixes.

#### Scenario: Build and serving share format definitions
- **WHEN** build or serving resolves an export format
- **THEN** both layers use the same registry values for MIME type and suffix

### Requirement: Alias normalization for line-delimited JSON
The system SHALL treat `jsonl` and `ndjson` as aliases for the same line-delimited JSON
format and normalize them to a canonical format ID internally.

#### Scenario: Alias inputs resolve to the same format
- **WHEN** a client requests export format `ndjson`
- **THEN** the system normalizes it to the canonical line-delimited JSON format

### Requirement: NDJSON exports are UTF-8 and deterministic
NDJSON exports SHALL be UTF-8 encoded line-delimited JSON and SHALL use msgspec encoding
when available with a stdlib json fallback. Unknown types MUST be stringified and JSON
output MUST use compact separators without ASCII escaping. This applies to HTTP streaming
responses and MCP export artifacts written by the resource store.

#### Scenario: NDJSON line preserves unicode and stringifies types
- **WHEN** a row with unicode text and datetime values is streamed as NDJSON
- **THEN** the output line is UTF-8, preserves unicode, and stringifies non-JSON types

### Requirement: Export payload MIME types are registry-driven
Export payload responses SHALL use MIME types from the canonical export format registry for
both HTTP and MCP payloads.

#### Scenario: MIME type matches registry
- **WHEN** an export payload is returned from HTTP or MCP
- **THEN** the payload MIME type matches the canonical registry value

### Requirement: NDJSON datetime encoding is RFC3339
NDJSON exports SHALL serialize datetime values as RFC3339 UTC strings with a Z suffix and
SHALL preserve UTF-8 while stringifying non-JSON types consistently across msgspec and
stdlib json encoders.

#### Scenario: NDJSON line uses RFC3339 and preserves unicode
- **WHEN** a row with datetime, UUID, bytes, and unicode text is encoded as NDJSON
- **THEN** the datetime is formatted as 2024-01-01T00:00:00Z, unicode is preserved, and
  non-JSON types are stringified

### Requirement: Export serialization is core-serializer backed
Export serialization SHALL use the core serialization utilities for JSON-compatible
conversion and the canonical export format registry for MIME/suffix handling. Build,
serving, and storage exports SHALL share the same coercion rules for non-JSON types.

#### Scenario: Export serialization is consistent across layers
- **WHEN** the same row is exported from build and serving
- **THEN** both outputs apply identical JSON/NDJSON coercion rules and MIME types

