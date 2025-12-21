## ADDED Requirements
### Requirement: Canonical configuration identifiers
The system SHALL expose canonical execution profile names and option result types only,
and SHALL NOT provide legacy aliases or compatibility shims.

#### Scenario: Legacy profile alias rejected
- **WHEN** configuration requests the "default" profile alias
- **THEN** profile resolution fails with an unknown-profile error

#### Scenario: ValidationOutcome is the only options result type
- **WHEN** options validation is performed
- **THEN** the result type is ValidationOutcome and no ValidationResult alias is exported
