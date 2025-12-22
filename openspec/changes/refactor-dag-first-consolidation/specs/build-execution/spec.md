## ADDED Requirements
### Requirement: Hamilton-native analytics and graph execution
Analytics, graph metrics, and history computations SHALL be executed via Hamilton DAG targets
or materializers only, and non-DAG orchestration functions/modules SHALL NOT be used by build
or CLI entrypoints. CLI/debug outputs SHALL be derived from DAG-produced datasets or cached
DAG artifacts.

#### Scenario: Graph metrics run through Hamilton
- **WHEN** graph metrics are executed for a snapshot
- **THEN** the execution plan schedules Hamilton targets only and no non-DAG orchestrators run

#### Scenario: CLI history uses DAG-derived data
- **WHEN** a CLI history command is invoked
- **THEN** it reads DAG-produced datasets or triggers DAG targets and no direct compute path runs
