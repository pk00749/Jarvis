# MCP and Function Call Support Implementation

## Overview
Implementation of MCP (Model Context Protocol) and function call support for `playback_qwen3_tts_flash.py` LLM component.

## Files Created

### 1. `app/tools/mcp_wrapper.py`
- **MCPToolBase**: Base class for MCP tools with SSE connection handling
- **AmapMapsTool**: Wraps amap-maps MCP server for location search
- **EdgeOneTool**: Wraps EdgeOne MCP server for location search
- Uses httpx for async SSE connections
- Handles authentication with DASHSCOPE_API_KEY

### 2. `app/tools/custom_tools.py`
- **WeatherTool**: Simple weather query tool with mock data
- **TimeTool**: Current time/date query tool
- Both tools follow qwen-agent BaseTool pattern

### 3. `app/tools/__init__.py`
- Exports all custom tools for easy import

## Files Modified

### 1. `playback_qwen3_tts_flash.py`
**Changes:**
- Added imports for custom MCP tools (lines 9-10)
- Replaced invalid `mcpServers` configuration with proper `function_list` (lines 129-135)
- Updated Assistant initialization to use new tool list (line 143)
- Improved tool call logging for better debugging (lines 155, 158)

### 2. `requirements.txt`
- Added `httpx>=0.24.0` dependency for SSE connections

## Tool Configuration

The `agent_llm` method now supports 5 tools:

1. **amap_weather** (built-in): Official qwen-agent weather tool
2. **AmapMapsTool** (MCP): High德地图 location search via MCP
3. **EdgeOneTool** (MCP): EdgeOne location search via MCP
4. **WeatherTool** (custom): Simple weather query with mock data
5. **TimeTool** (custom): Current time/date query

## Usage Examples

Query examples that will trigger tools:

```bash
# Weather queries
"今天北京天气怎么样"
"帮我查一下上海的天气"

# Location search
"帮我查一下天安门的地址"
"搜索一下故宫的位置信息"

# Time queries
"现在几点了"
"今天是什么时候"
```

## Technical Details

### MCP Tool Architecture

```python
MCPToolBase (extends BaseTool)
├── __init__(cfg): Initialize with endpoint URL
├── _call_mcp_sse(params): Async SSE connection handler
└── call(params): Synchronous wrapper for qwen-agent
```

### Tool Registration

All tools use `@register_tool('name')` decorator to register with qwen-agent's TOOL_REGISTRY.

### SSE Connection Pattern

1. Send HTTP POST request to MCP endpoint
2. Process Server-Sent Events stream
3. Parse JSON data chunks
4. Return formatted result to agent

## Testing

### Manual Testing

```python
# Test individual tools
from app.tools.mcp_wrapper import AmapMapsTool
from app.tools.custom_tools import WeatherTool

tool = AmapMapsTool()
print(tool.call({'query': '北京天安门'}))

tool = WeatherTool()
print(tool.call({'location': '北京'}))
```

### Integration Testing

Run `playback_qwen3_tts_flash.py` with queries that trigger tool calls:
- Weather queries
- Location searches
- Time queries

## Error Handling

- Missing DASHSCOPE_API_KEY: Returns error message
- MCP endpoint unavailable: Returns error with exception details
- Invalid tool parameters: Handled by qwen-agent's parameter validation
- SSE connection failures: Graceful degradation with error messages

## Future Enhancements

1. Add more MCP tools (route planning, POI details, etc.)
2. Implement tool result caching
3. Add more sophisticated custom tools
4. Support for streaming tool responses
5. Add unit tests for all tools

## Notes

- MCP servers require valid DASHSCOPE_API_KEY
- SSE timeout set to 30 seconds
- Custom tools use mock data (WeatherTool randomizes responses)
- Tool results are formatted for TTS synthesis
