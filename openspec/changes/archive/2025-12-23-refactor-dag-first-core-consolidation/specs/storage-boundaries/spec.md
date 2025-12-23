## ADDED Requirements
### Requirement: StorageFacade is the non-storage entrypoint
Non-storage modules SHALL access storage via a single StorageFacade that exposes
read, write, and export capabilities. Direct use of gateways, repositories, or
view builders outside storage SHALL NOT be permitted.

#### Scenario: Non-storage code uses the facade
- **WHEN** analytics or serving code needs storage access
- **THEN** it uses StorageFacade APIs instead of gateway or repository classes
