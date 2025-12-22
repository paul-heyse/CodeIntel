## ADDED Requirements
### Requirement: NDJSON datetime encoding is RFC3339
NDJSON exports SHALL serialize datetime values as RFC3339 UTC strings with a Z suffix and
SHALL preserve UTF-8 while stringifying non-JSON types consistently across msgspec and
stdlib json encoders.

#### Scenario: NDJSON line uses RFC3339 and preserves unicode
- **WHEN** a row with datetime, UUID, bytes, and unicode text is encoded as NDJSON
- **THEN** the datetime is formatted as 2024-01-01T00:00:00Z, unicode is preserved, and
  non-JSON types are stringified
