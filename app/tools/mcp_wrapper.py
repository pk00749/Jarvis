import os
import re
import json
import asyncio
from typing import Optional, Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Type

try:
    from qwen_agent.tools.base import BaseTool, register_tool
except ImportError:
    BaseTool = object
    register_tool = lambda x: x


class MCPToolBase(BaseTool):
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.endpoint_url = self.cfg.get('endpoint_url')
        self.api_key = os.getenv('DASHSCOPE_API_KEY')
        
    async def _call_mcp_sse(self, params: dict) -> str:
        import httpx
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream('POST', self.endpoint_url, 
                                          json=params, 
                                          headers=headers) as response:
                    response.raise_for_status()
                    result = []
                    async for line in response.aiter_lines():
                        if line.startswith('data: '):
                            data_str = line[6:].strip()
                            if data_str:
                                try:
                                    data = json.loads(data_str)
                                    result.append(data)
                                except json.JSONDecodeError:
                                    pass
                    
                    if result:
                        return json.dumps(result, ensure_ascii=False)
                    else:
                        return json.dumps({"status": "success", "message": "查询完成，无返回数据"}, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"status": "error", "message": f"MCP调用失败: {str(e)}"}, ensure_ascii=False)
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params) if hasattr(self, '_verify_json_format_args') else params
        return asyncio.run(self._call_mcp_sse(params))


class MCPToolFactory:
    @staticmethod
    def create_tool_class(name: str, config: dict):
        """Dynamically create MCP tool class from config"""
        tool_name = config.get('name', name)
        
        class DynamicMCPTool(MCPToolBase):
            name = tool_name
            description = config['description']
            parameters = config['parameters']

            def __init__(self, cfg: Optional[Dict] = None):
                endpoint_url = MCPToolFactory._substitute_env_vars(config['endpoint_url'])
                default_cfg = {'endpoint_url': endpoint_url}
                if cfg:
                    default_cfg.update(cfg)
                super().__init__(default_cfg)

        DynamicMCPTool.__name__ = name
        return DynamicMCPTool

    @staticmethod
    def _substitute_env_vars(value: str) -> str:
        """Replace ${VAR_NAME} patterns with environment variable values"""
        if not isinstance(value, str):
            return value
        
        pattern = r'\$\{([^}]+)\}'
        
        def replacer(match):
            var_name = match.group(1)
            env_value = os.getenv(var_name)
            if env_value is None:
                raise ValueError(f"Environment variable '{var_name}' is not set but referenced in configuration: {value}")
            return env_value
        
        return re.sub(pattern, replacer, value)


class MCPToolRegistry:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or 'app/tools/mcp_config.yaml'
        self.tools = {}

    def load_config(self) -> dict:
        """Load MCP configuration from YAML"""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required. Install it with: pip install pyyaml")

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"MCP configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")

        if not config:
            raise ValueError(f"Configuration file is empty: {self.config_path}")

        if 'mcp_tools' not in config:
            raise ValueError("Configuration must contain 'mcp_tools' key")

        self._validate_config(config)
        return config

    def _validate_config(self, config: dict):
        """Validate configuration schema"""
        required_fields = ['name', 'description', 'endpoint_url', 'parameters']
        
        for tool_key, tool_config in config.get('mcp_tools', {}).items():
            if not isinstance(tool_config, dict):
                raise ValueError(f"Tool '{tool_key}' must be a dictionary")

            for field in required_fields:
                if field not in tool_config:
                    raise ValueError(f"Tool '{tool_key}' missing required field: '{field}'")

            if not isinstance(tool_config['parameters'], list):
                raise ValueError(f"Tool '{tool_key}' 'parameters' must be a list")

            for param in tool_config['parameters']:
                if not isinstance(param, dict):
                    raise ValueError(f"Tool '{tool_key}' parameter must be a dictionary")
                if 'name' not in param or 'type' not in param:
                    raise ValueError(f"Tool '{tool_key}' parameter missing 'name' or 'type' field")

    def register_tools(self) -> dict:
        """Register all MCP tools from configuration"""
        config = self.load_config()
        
        for tool_key, tool_config in config.get('mcp_tools', {}).items():
            tool_name = tool_config.get('name', tool_key)
            tool_class = MCPToolFactory.create_tool_class(tool_name, tool_config)
            registered_tool = register_tool(tool_name)(tool_class)
            self.tools[tool_name] = registered_tool
        
        return self.tools


def get_all_mcp_tools() -> dict:
    """Get all registered MCP tools from the registry"""
    return mcp_tools_dict


registry = MCPToolRegistry()
mcp_tools_dict = registry.register_tools()

AmapMapsTool = mcp_tools_dict.get('amap_maps')
EdgeOneTool = mcp_tools_dict.get('edgeone_search')
