import random
import json
from datetime import datetime
from typing import Optional, Dict, Union

try:
    from qwen_agent.tools.base import BaseTool, register_tool
except ImportError:
    BaseTool = object
    register_tool = lambda x: x


@register_tool('get_weather')
class WeatherTool(BaseTool):
    description = '查询指定城市的天气情况'
    parameters = [{
        'name': 'location',
        'type': 'string',
        'description': '城市或地区名称，如北京、上海、广州等',
        'required': True
    }]
    
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        
    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            params = json.loads(params) if params.startswith('{') else {'location': params}
        location = params.get('location', '未知') if isinstance(params, dict) else params
        
        weather_conditions = ["晴天", "多云", "阴天", "小雨", "大雨", "雷阵雨"]
        temperatures = range(-5, 40)
        
        weather = random.choice(weather_conditions)
        temperature = random.choice(temperatures)
        
        return f"{location}今天的天气是{weather}，温度{temperature}度。"


@register_tool('get_time')
class TimeTool(BaseTool):
    description = '查询当前时间'
    parameters = [{
        'name': 'timezone',
        'type': 'string',
        'description': '时区名称，如Asia/Shanghai、Asia/Tokyo等，默认为本地时间',
        'required': False
    }]
    
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        
    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            params = json.loads(params) if params.startswith('{') else {'timezone': params}
        
        timezone = params.get('timezone') if isinstance(params, dict) else params
        
        if timezone and timezone != '本地':
            try:
                import pytz
                tz = pytz.timezone(timezone)
                current_time = datetime.now(tz)
                return f"{timezone}现在的时间是{current_time.strftime('%Y-%m-%d %H:%M:%S')}。"
            except Exception as e:
                return f"时区{timezone}查询失败，使用本地时间。"
        else:
            current_time = datetime.now()
            return f"现在的时间是{current_time.strftime('%Y年%m月%d日 %H时%M分%S秒')}。"
