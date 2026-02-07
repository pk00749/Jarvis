import random
import json
from typing import Optional, Dict, Union

try:
    from qwen_agent.tools.base import BaseTool, register_tool
except ImportError:
    BaseTool = object
    register_tool = lambda x: x


@register_tool('calculator')
class CalculatorTool(BaseTool):
    description = '执行数学计算，包括加减乘除、幂运算等'
    parameters = [{
        'name': 'expression',
        'type': 'string',
        'description': '要计算的表达式，如"125加378"、"23乘以56"、"100减去35"等',
        'required': True
    }]
    
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        
    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            params = json.loads(params) if params.startswith('{') else {'expression': params}
        
        expression = params.get('expression', '') if isinstance(params, dict) else params
        
        if not expression:
            return "你唔係想算什么呀？请告诉我你要计算嘅数字。"
        
        try:
            result = self._calculate(expression)
            return f"{expression}等于{result}。"
        except Exception as e:
            return f"抱歉，我唔识算{expression}，可能系讲得唔清楚，可以再讲一次咩？"
    
    def _calculate(self, expression: str) -> float:
        expression = expression.lower()
        
        if '加' in expression or '加' in expression:
            parts = expression.replace('加', '+').replace('加', '+').split('+')
            if len(parts) == 2:
                return float(parts[0].strip()) + float(parts[1].strip())
        
        if '减' in expression:
            parts = expression.split('减')
            if len(parts) == 2:
                return float(parts[0].strip()) - float(parts[1].strip())
        
        if '乘' in expression or '乘以' in expression:
            expr = expression.replace('乘以', '乘').replace('乘', '*')
            parts = expr.split('*')
            if len(parts) == 2:
                return float(parts[0].strip()) * float(parts[1].strip())
        
        if '除' in expression or '除以' in expression:
            expr = expression.replace('除以', '除').replace('除', '/')
            parts = expr.split('/')
            if len(parts) == 2:
                denominator = float(parts[1].strip())
                if denominator == 0:
                    raise ValueError("唔可以除以零㗎")
                return float(parts[0].strip()) / denominator
        
        if '的' in expression and '次方' in expression:
            parts = expression.replace('次方', '^').replace('的', '').split('^')
            if len(parts) == 2:
                return float(parts[0].strip()) ** float(parts[1].strip())
        
        raise ValueError("唔识得呢个计算方式")
