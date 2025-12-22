## ADDED Requirements
### Requirement: NDJSON exports are UTF-8 and deterministic
NDJSON exports SHALL be UTF-8 encoded line-delimited JSON and SHALL use msgspec encoding
when available with a stdlib json fallback. Unknown types MUST be stringified and JSON
output MUST use compact separators without ASCII escaping.

#### Scenario: NDJSON line preserves unicode and stringifies types
- **WHEN** a row with unicode text and datetime values is streamed as NDJSON
- **THEN** the output line is UTF-8, preserves unicode, and stringifies non-JSON types

### Requirement: Export payload MIME types are registry-driven
Export payload responses SHALL use MIME types from the canonical export format registry for
both HTTP and MCP payloads.

#### Scenario: MIME type matches registry
- **WHEN** an export payload is returned from HTTP or MCP
- **THEN** the payload MIME type matches the canonical registry value
