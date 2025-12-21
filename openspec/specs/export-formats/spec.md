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

