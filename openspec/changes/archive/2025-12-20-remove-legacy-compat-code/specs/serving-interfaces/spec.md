## ADDED Requirements
### Requirement: Direct FastMCP imports
Serving MCP components SHALL import FastMCP types directly from fastmcp packages and
SHALL NOT rely on local compatibility shims.

#### Scenario: MCP server uses direct imports
- **WHEN** the MCP server is constructed
- **THEN** FastMCP, Context, and EventStore are imported from fastmcp directly
