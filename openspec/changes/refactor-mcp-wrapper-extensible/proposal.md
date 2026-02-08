# Proposal: Refactor MCP Wrapper for Extensibility

## Summary
Optimize `mcp_wrapper.py` to support adding new MCP tools with minimal code changes through a configuration-based approach.

## Motivation
Currently, adding a new MCP tool requires:
- Creating a new class extending `MCPToolBase`
- Hardcoding `endpoint_url`, `description`, and `parameters`
- Manually adding to `__init__.py` exports
- Manually adding to `playback_qwen3_tts_flash.py` function_list

This repetitive process is error-prone and does not scale as more MCP tools are added.

## Goals
1. Add new MCP tools via configuration only
2. Auto-register tools from configuration
3. Maintain backward compatibility with existing tools
4. Reduce boilerplate code for each MCP tool

## Scope
**In Scope:**
- Refactor `app/tools/mcp_wrapper.py` to support config-based registration
- Create MCP configuration file (YAML/JSON)
- Auto-discovery and registration of MCP tools
- Update `app/tools/__init__.py` to export dynamic tools

**Out of Scope:**
- Changes to MCP protocol implementation
- Changes to SSE communication logic
- Changes to existing tool behavior
- Changes to qwen-agent integration

## Non-Goals
- Dynamic endpoint discovery from external services
- Runtime endpoint configuration without restart
- Hot-reloading of MCP configurations
