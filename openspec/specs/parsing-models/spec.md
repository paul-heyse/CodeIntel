# parsing-models Specification

## Purpose
TBD - created by archiving change remove-legacy-compat-code. Update Purpose after archive.
## Requirements
### Requirement: Core parsing models are graph-agnostic
Core parsing models SHALL include only canonical parsing metadata, and graph-specific
fields SHALL live in graph parsing models.

#### Scenario: Core ParsedFunction excludes graph compatibility fields
- **WHEN** a core ParsedFunction is instantiated
- **THEN** it does not include graph-specific fields such as is_async or decorator_names

