## ADDED Requirements
### Requirement: Error taxonomy is imported from core
CLI error handling SHALL import taxonomy definitions from the core taxonomy module and SHALL
NOT re-export or shadow taxonomy types in CLI-specific modules.

#### Scenario: CLI taxonomy uses core module
- **WHEN** CLI error handlers need taxonomy definitions
- **THEN** they import from the core taxonomy module
