## ADDED Requirements
### Requirement: Parsing validation reporters import from core
Analytics parsing modules SHALL import validation reporters from
codeintel.core.validation.reporters and SHALL NOT provide compatibility re-exports.

#### Scenario: Validation reporters are imported from core
- **WHEN** analytics parsing modules import validation reporters
- **THEN** the import path is the core validation reporters module
