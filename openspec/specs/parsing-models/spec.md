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

### Requirement: Parsing validation reporters import from core
Analytics parsing modules SHALL import validation reporters from
codeintel.core.validation.reporters and SHALL NOT provide compatibility re-exports.

#### Scenario: Validation reporters are imported from core
- **WHEN** analytics parsing modules import validation reporters
- **THEN** the import path is the core validation reporters module

