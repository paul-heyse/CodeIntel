## ADDED Requirements
### Requirement: Canonical registry service is the public discovery surface
Public interfaces SHALL expose a single RegistryService for dataset, semantic, and
export discovery, and SHALL NOT expose legacy registry modules or duplicate catalog
implementations.

#### Scenario: Discovery uses the registry service
- **WHEN** a caller requests dataset or semantic catalog data
- **THEN** the registry service is used and no legacy registries are imported

### Requirement: Core utility helpers are centralized
Shared helpers for hashing, time normalization, and serialization SHALL live in core
utility modules and SHALL NOT be reimplemented in build, storage, serving, or analytics
packages.

#### Scenario: Utility usage is centralized
- **WHEN** a module needs hashing or serialization helpers
- **THEN** it imports from the core utility modules and no package-local helpers exist
