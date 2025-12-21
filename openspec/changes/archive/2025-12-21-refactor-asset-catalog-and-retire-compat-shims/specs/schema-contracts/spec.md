## ADDED Requirements
### Requirement: Schema compilation is native-only
Schema compilation SHALL consider only native targets and SHALL NOT expose compatibility
flags such as --only-native.

#### Scenario: Schema diff CLI omits only-native
- **WHEN** the schema diff command help is displayed
- **THEN** no only-native flag is exposed

### Requirement: Structured schema diff is the only output
Schema diff SHALL emit structured summaries with breaking-change detection and SHALL NOT
provide legacy unified diff output or a toggle to enable it.

#### Scenario: Schema diff uses structured output
- **WHEN** schema diff detects changes
- **THEN** the output includes a structured summary rather than a unified diff
