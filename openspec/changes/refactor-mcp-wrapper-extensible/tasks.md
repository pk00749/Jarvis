# Tasks: MCP Wrapper Extensibility

## Tasks

- [x] **Add configuration loading capability**
   - Install `pyyaml` dependency in `requirements.txt` or `pyproject.toml`
   - Create `app/tools/mcp_config.yaml` with existing MCP tools (AmapMaps, EdgeOne)
   - Validate YAML syntax and required fields

- [x] **Implement MCPToolFactory class**
   - Create `MCPToolFactory.create_tool_class()` method
   - Generate tool class with correct `description` and `parameters`
   - Generate `__init__()` method with `endpoint_url` from config
   - Set tool class `__name__` property correctly
   - Add unit tests for factory method

- [x] **Implement MCPToolRegistry class**
   - Create `MCPToolRegistry.__init__(config_path)` constructor
   - Implement `load_config()` method with YAML parsing
   - Implement `register_tools()` method that:
     - Iterates through config `mcp_tools`
     - Calls `MCPToolFactory.create_tool_class()` for each
     - Applies `@register_tool()` decorator dynamically
     - Stores registered tools in `self.tools` dict
   - Add error handling for missing config file or invalid YAML
   - Add unit tests for registry

- [x] **Update mcp_wrapper.py module**
   - Import factory and registry classes
   - Add auto-registration at module level:
     ```python
     registry = MCPToolRegistry('app/tools/mcp_config.yaml')
     mcp_tools_dict = registry.register_tools()
     ```
   - Export tools from registry for backward compatibility:
     ```python
     AmapMapsTool = mcp_tools_dict.get('amap_maps')
     EdgeOneTool = mcp_tools_dict.get('edgeone_search')
     ```
   - Keep original `MCPToolBase` class unchanged
   - Add `get_all_mcp_tools()` helper function

- [x] **Update app/tools/__init__.py**
   - Export dynamic tools from registry:
     ```python
     from .mcp_wrapper import mcp_tools_dict, get_all_mcp_tools
     ```
   - Keep individual exports for backward compatibility

- [x] **Add validation and tests**
   - Validate config schema on load (name, description, endpoint_url, parameters)
   - Test auto-registration on module import
   - Test that tools can be instantiated and called
   - Test backward compatibility with existing code
   - Test error handling (missing config, invalid YAML, missing fields)

- [x] **Update documentation**
   - Document how to add new MCP via YAML config
   - Document configuration schema
   - Document environment variable substitution in endpoints
   - Update `openspec/project.md` with new MCP pattern

## Dependencies

- Task 1 must complete before Task 2 (config file needed)
- Task 2 must complete before Task 3 (factory needed for registry)
- Task 3 must complete before Task 4 (registry needed for module update)
- Task 4 must complete before Task 5 (module exports needed for __init__)
- Task 6 can run in parallel with Task 4 and 5

## Validation

After completing all tasks:
- Run existing integration tests with AmapMaps and EdgeOne tools
- Verify tools work identically to pre-refactor implementation
- Add a test MCP to config and verify auto-registration
- Confirm no changes required in `playback_qwen3_tts_flash.py`
