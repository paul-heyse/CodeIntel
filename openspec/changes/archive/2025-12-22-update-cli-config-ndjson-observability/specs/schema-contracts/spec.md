## ADDED Requirements
### Requirement: JSON columns are excluded from numeric non-negative checks
Pandera constraint generation SHALL apply non-negative checks only to numeric columns and
SHALL NOT apply them to JSON columns such as functions_covered. Corresponding count columns
(e.g., functions_covered_count) SHALL continue to enforce non-negative constraints.

#### Scenario: Test profile JSON columns pass validation
- **WHEN** analytics.test_profile includes functions_covered as a JSON list and
  functions_covered_count as 1
- **THEN** Pandera validation succeeds and only the count column is evaluated for
  non-negative checks
