## ADDED Requirements
### Requirement: Export serialization is core-serializer backed
Export serialization SHALL use the core serialization utilities for JSON-compatible
conversion and the canonical export format registry for MIME/suffix handling. Build,
serving, and storage exports SHALL share the same coercion rules for non-JSON types.

#### Scenario: Export serialization is consistent across layers
- **WHEN** the same row is exported from build and serving
- **THEN** both outputs apply identical JSON/NDJSON coercion rules and MIME types
