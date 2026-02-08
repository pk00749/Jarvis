# MCP Tool Configuration Guide

## Overview

MCP tools are now configured via YAML configuration files, enabling zero-code addition of new tools.

## Configuration File

Location: `app/tools/mcp_config.yaml`

### Configuration Schema

```yaml
mcp_tools:
  tool_name:
    name: 'tool_name'              # Tool identifier (used for registration)
    description: 'Tool description' # Human-readable description
    endpoint_url: 'https://...'    # MCP SSE endpoint URL
    parameters:
      - name: 'param_name'         # Parameter name
        type: 'string'              # Parameter type (string, number, boolean, etc.)
        description: 'Param desc'  # Parameter description
        required: true              # Whether parameter is required
```

### Required Fields

- `name`: Tool identifier (used with `@register_tool`)
- `description`: Human-readable description for the LLM
- `endpoint_url`: Full URL to the MCP SSE endpoint
- `parameters`: Array of parameter definitions

### Parameter Fields

- `name`: Parameter name
- `type`: Parameter type (string, number, boolean, array, object)
- `description`: What the parameter does
- `required`: Boolean indicating if parameter is required

## Adding a New MCP Tool

### Step 1: Add to Configuration

Edit `app/tools/mcp_config.yaml` and add your tool:

```yaml
mcp_tools:
  amap_maps:
    name: 'amap_maps'
    description: '高德地图位置搜索，查找地点、地址、POI等信息'
    endpoint_url: 'https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse'
    parameters:
      - name: 'query'
        type: 'string'
        description: '要搜索的地点、地址或POI名称，如"北京天安门"'
        required: true

  # Add your new tool here
  my_new_tool:
    name: 'my_new_tool'
    description: 'My new MCP tool description'
    endpoint_url: 'https://api.example.com/mcp/sse'
    parameters:
      - name: 'input'
        type: 'string'
        description: 'Input parameter'
        required: true
```

### Step 2: Restart Application

The tool is automatically registered when `app.tools.mcp_wrapper` is imported. No code changes needed.

### Step 3: Use in Assistant

The tool is automatically available in the Assistant's function_list:

```python
from app.tools import get_all_mcp_tools

# Get all MCP tools
mcp_tools = get_all_mcp_tools()

# Access specific tool
my_tool = mcp_tools.get('my_new_tool')
```

## Environment Variable Substitution

For security and flexibility, endpoint URLs can use environment variables:

```yaml
mcp_tools:
  my_tool:
    name: 'my_tool'
    description: 'Tool with dynamic endpoint'
    endpoint_url: '${MCP_BASE_URL}/api/sse'  # Uses MCP_BASE_URL env var
    parameters: []
```

Set the environment variable:

```bash
export MCP_BASE_URL=https://api.example.com
python app/playback_qwen3_tts_flash.py
```

### Error Handling

If an environment variable is referenced but not set, the system raises a descriptive error:

```
ValueError: Environment variable 'MCP_BASE_URL' is not set but referenced in configuration: ${MCP_BASE_URL}/api/sse
```

## Configuration Validation

The system validates the configuration on module load and will raise errors for:

- Missing configuration file
- Invalid YAML syntax
- Missing required fields (`name`, `description`, `endpoint_url`, `parameters`)
- Invalid parameter definitions

## Backward Compatibility

Existing code continues to work without changes:

```python
# Old way still works
from app.tools.mcp_wrapper import AmapMapsTool, EdgeOneTool

tool = AmapMapsTool()
result = tool.call({'query': '北京天安门'})
```

## Advanced Usage

### Custom Configuration Path

Use a custom configuration file path:

```python
from app.tools.mcp_wrapper import MCPToolRegistry

registry = MCPToolRegistry('/path/to/custom/config.yaml')
tools = registry.register_tools()
```

### Accessing All MCP Tools

```python
from app.tools import get_all_mcp_tools

all_tools = get_all_mcp_tools()
for name, tool_class in all_tools.items():
    print(f"Tool: {name}, Description: {tool_class.description}")
```

## Architecture

The new MCP system consists of:

1. **MCPToolFactory**: Dynamically creates tool classes from configuration
2. **MCPToolRegistry**: Loads YAML config and registers tools with qwen-agent
3. **mcp_config.yaml**: Centralized configuration for all MCP tools

## Migration from Old Pattern

Before (manual class creation):
```python
@register_tool('my_tool')
class MyTool(MCPToolBase):
    description = 'Description'
    parameters = [{'name': 'query', 'type': 'string', ...}]
    def __init__(self, cfg=None):
        default_cfg = {'endpoint_url': 'https://...'}
        if cfg: default_cfg.update(cfg)
        super().__init__(default_cfg)
```

After (configuration-only):
```yaml
mcp_tools:
  my_tool:
    name: 'my_tool'
    description: 'Description'
    endpoint_url: 'https://...'
    parameters:
      - name: 'query'
        type: 'string'
        ...
```

No Python code needed!
