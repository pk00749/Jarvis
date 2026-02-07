import os
import json
import asyncio
from typing import Optional, Dict, Union

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


@register_tool('amap_maps')
class AmapMapsTool(MCPToolBase):
    description = '高德地图位置搜索，查找地点、地址、POI等信息'
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': '要搜索的地点、地址或POI名称，如"北京天安门"',
        'required': True
    }]
    
    def __init__(self, cfg: Optional[Dict] = None):
        default_cfg = {
            'endpoint_url': 'https://dashscope.aliyuncs.com/api/v1/mcps/amap-maps/sse'
        }
        if cfg:
            default_cfg.update(cfg)
        super().__init__(default_cfg)


@register_tool('edgeone_search')
class EdgeOneTool(MCPToolBase):
    description = 'EdgeOne页面内容获取和位置相关信息搜索'
    parameters = [{
        'name': 'query',
        'type': 'string',
        'description': '位置或页面查询内容',
        'required': True
    }]
    
    def __init__(self, cfg: Optional[Dict] = None):
        default_cfg = {
            'endpoint_url': 'https://dashscope.aliyuncs.com/api/v1/mcps/EdgeOne/sse'
        }
        if cfg:
            default_cfg.update(cfg)
        super().__init__(default_cfg)
