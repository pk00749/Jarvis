# mcp-tool-registry Specification

## Purpose
TBD - created by archiving change refactor-mcp-wrapper-extensible. Update Purpose after archive.
## Requirements
### Requirement: Dynamic MCP tool registration from configuration

The system SHALL automatically register MCP tools from a centralized configuration file without requiring manual class creation.

#### Scenario: Adding a new MCP tool via configuration
Given an MCP configuration file exists at `app/tools/mcp_config.yaml`
And the configuration contains a new MCP tool definition with name, description, endpoint_url, and parameters
When the `mcp_wrapper` module is imported
Then the tool class shall be dynamically created and registered with the qwen-agent tool registry
And the tool shall be available for use in the Assistant

#### Scenario: Listing all registered MCP tools
Given the MCP tool registry has loaded the configuration
When `get_all_mcp_tools()` is called
Then a dictionary of all registered MCP tool classes shall be returned
And the dictionary keys shall match the configured tool names

### Requirement: Configuration-based MCP tool factory

The system SHALL provide a factory that creates MCP tool classes from configuration data.

#### Scenario: Creating tool class from configuration
Given a tool configuration dictionary with name, description, endpoint_url, and parameters
When `MCPToolFactory.create_tool_class()` is called with the configuration
Then a tool class extending `MCPToolBase` shall be returned
And the class shall have the correct `description` and `parameters` attributes
And the class `__name__` shall match the configured name

#### Scenario: Instantiating tool from factory-generated class
Given a tool class created by the factory
When the tool class is instantiated with no arguments
Then the tool shall have the correct `endpoint_url` set from configuration
And the tool shall be callable via the `call()` method

### Requirement: Backward compatibility with existing code

The refactored system SHALL maintain backward compatibility with existing code that imports MCP tool classes.

#### Scenario: Existing code continues to work
Given existing code imports `AmapMapsTool` and `EdgeOneTool` from `app.tools.mcp_wrapper`
When the refactored module is imported
Then the imports shall succeed
And the imported classes shall behave identically to the pre-refactor implementation

#### Scenario: Existing Assistant integration continues to work
Given an existing Assistant instance with MCP tools in `function_list`
When the Assistant processes a query requiring an MCP tool
Then the tool call shall succeed
And the result shall be identical to the pre-refactor implementation

