# Design: MCP Wrapper Extensibility

## Current Architecture

### MCP Tool Pattern
```python
@register_tool('tool_name')
class ToolName(MCPToolBase):
    description = 'Tool description'
    parameters = [{'name': 'param', 'type': 'string', ...}]
    def __init__(self, cfg=None):
        default_cfg = {'endpoint_url': '...'}
        if cfg: default_cfg.update(cfg)
        super().__init__(default_cfg)
```

**Issues:**
- Hardcoded endpoint URLs
- Manual class creation for each MCP
- Manual registration with decorator
- No centralized configuration
- Repetitive boilerplate code

## Proposed Architecture

### Configuration-Based Registration

```yaml
# app/tools/mcp_config.yaml
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

  edgeone_search:
    name: 'edgeone_search'
    description: 'EdgeOne页面内容获取和位置相关信息搜索'
    endpoint_url: 'https://dashscope.aliyuncs.com/api/v1/mcps/EdgeOne/sse'
    parameters:
      - name: 'query'
        type: 'string'
        description: '位置或页面查询内容'
        required: true
```

### Dynamic MCP Tool Factory

```python
# app/tools/mcp_wrapper.py (refactored)

class MCPToolFactory:
    @staticmethod
    def create_tool_class(name: str, config: dict) -> Type[MCPToolBase]:
        """Dynamically create MCP tool class from config"""
        class DynamicMCPTool(MCPToolBase):
            description = config['description']
            parameters = config['parameters']

            def __init__(self, cfg=None):
                default_cfg = {'endpoint_url': config['endpoint_url']}
                if cfg:
                    default_cfg.update(cfg)
                super().__init__(default_cfg)

        DynamicMCPTool.__name__ = name
        return DynamicMCPTool

class MCPToolRegistry:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.tools = {}

    def load_config(self) -> dict:
        """Load MCP configuration from YAML"""
        import yaml
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def register_tools(self):
        """Register all MCP tools from configuration"""
        config = self.load_config()
        for tool_name, tool_config in config.get('mcp_tools', {}).items():
            tool_class = MCPToolFactory.create_tool_class(tool_name, tool_config)
            registered_tool = register_tool(tool_name)(tool_class)
            self.tools[tool_name] = registered_tool
            return self.tools
```

### Module-Level Auto-Registration

```python
# app/tools/mcp_wrapper.py (bottom of file)

# Auto-register all MCP tools from config
registry = MCPToolRegistry('app/tools/mcp_config.yaml')
mcp_tools_dict = registry.register_tools()

# Export individual tools for backward compatibility
AmapMapsTool = mcp_tools_dict.get('amap_maps')
EdgeOneTool = mcp_tools_dict.get('edgeone_search')
```

## Benefits

1. **Zero-Code Addition**: New MCP tools added via YAML config only
2. **Centralized Configuration**: All MCP endpoints in one place
3. **Backward Compatible**: Existing code continues to work
4. **Type Safety**: Tool classes still type-checked
5. **Dynamic Discovery**: Easy to list available MCPs
6. **Environment-Specific Config**: Different configs for dev/prod

## Trade-offs

| Aspect | Current Approach | Proposed Approach |
|--------|------------------|-------------------|
| Simplicity | Simple classes | Config parsing overhead |
| Type Safety | Explicit class definitions | Dynamic class creation |
| Debugging | Static stack traces | Dynamic stack traces |
| Extensibility | Requires code change | Config-only changes |
| Testing | Direct class instantiation | Test factory logic |

## Implementation Notes

1. Add `pyyaml` dependency for config parsing
2. Preserve existing `MCPToolBase` logic unchanged
3. Keep `MCPToolBase` as importable base for custom implementations
4. Support environment variable substitution in endpoint URLs
5. Validate config on module load (fail fast)

## Migration Path

1. **Phase 1**: Add factory and registry alongside existing classes
2. **Phase 2**: Create config file with existing tools
3. **Phase 3**: Update `__init__.py` to export from registry
4. **Phase 4**: Deprecate hardcoded classes (keep for compatibility)
5. **Phase 5**: Documentation update
