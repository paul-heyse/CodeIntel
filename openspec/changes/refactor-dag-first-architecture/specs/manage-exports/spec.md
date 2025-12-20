## ADDED Requirements

### Requirement: Canonical export format registry
The system SHALL define a canonical export format registry that includes ndjson, json, parquet,
and arrow, with ndjson as the default interactive format. The canonical registry MUST NOT include
jsonl.

#### Scenario: Default export format
- **WHEN** an export format is not specified
- **THEN** the system selects ndjson as the default

#### Scenario: Canonical registry contents
- **WHEN** the export format registry is inspected
- **THEN** jsonl is absent and ndjson is present

### Requirement: Export format normalization
The system SHALL normalize any input format aliases to the canonical registry or reject them.

#### Scenario: Alias normalization
- **WHEN** a request specifies format "jsonl"
- **THEN** the system normalizes it to ndjson or returns a validation error

### Requirement: Export planning uses canonical registry
Export planning SHALL use the canonical format registry to determine MIME types, suffixes, and
delivery strategies for both build and serving.

#### Scenario: MIME type resolution
- **WHEN** an export plan is created for parquet
- **THEN** the plan uses the canonical parquet MIME type and suffix

### Requirement: Export artifacts are cataloged
DatasetCatalog SHALL include export artifact specifications tied to table_key and format.

#### Scenario: Cataloged export artifacts
- **WHEN** a dataset is exportable
- **THEN** its DatasetCatalog entry includes export artifact metadata for each supported format
