# Spec: MCP Config Loader

## ADDED Requirements

### Requirement: YAML-based MCP configuration

The system SHALL load MCP tool definitions from a YAML configuration file.

#### Scenario: Loading valid configuration
Given a valid YAML file exists at `app/tools/mcp_config.yaml`
And the file contains an `mcp_tools` key with tool definitions
When `MCPToolRegistry.load_config()` is called
Then the configuration shall be parsed and returned as a Python dictionary
And the dictionary shall contain all tool definitions under `mcp_tools` key

#### Scenario: Validating configuration schema
Given a YAML configuration file
When the configuration is loaded
Then each tool definition shall contain required fields: `name`, `description`, `endpoint_url`, `parameters`
And invalid configurations shall raise a descriptive error

### Requirement: Environment variable substitution in configuration

The configuration SHALL support environment variable substitution for sensitive data like API keys or custom endpoints.

#### Scenario: Substituting environment variable in endpoint URL
Given a configuration with `endpoint_url: ${MCP_BASE_URL}/maps/sse`
And the environment variable `MCP_BASE_URL` is set to `https://api.example.com`
When the configuration is loaded and tool is created
Then the tool's `endpoint_url` shall be `https://api.example.com/maps/sse`

#### Scenario: Missing environment variable
Given a configuration with `endpoint_url: ${UNDEFINED_VAR}/maps/sse`
And the environment variable `UNDEFINED_VAR` is not set
When the configuration is loaded
Then an informative error shall be raised indicating the missing variable

### Requirement: Configuration file location flexibility

The MCP configuration file SHALL be configurable to support different environments.

#### Scenario: Using default config location
Given no config path is specified when creating MCPToolRegistry
When the registry is initialized
Then the default config path `app/tools/mcp_config.yaml` shall be used

#### Scenario: Using custom config location
Given a config path is specified when creating MCPToolRegistry
When the registry is initialized
Then the specified config path shall be used for loading configuration

## MODIFIED Requirements

None

## REMOVED Requirements

None
