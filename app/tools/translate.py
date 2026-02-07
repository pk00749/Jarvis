import os
import json
from typing import Optional, Dict, Union

try:
    from qwen_agent.tools.base import BaseTool, register_tool
except ImportError:
    BaseTool = object
    register_tool = lambda x: x


@register_tool('translate')
class TranslateTool(BaseTool):
    description = '文本翻译，支持中英文互译'
    parameters = [{
        'name': 'text',
        'type': 'string',
        'description': '要翻译的文本内容',
        'required': True
    }, {
        'name': 'target_language',
        'type': 'string',
        'description': '目标语言，如：中文、英文、English、Chinese，不指定则自动检测',
        'required': False
    }]
    
    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        
    def call(self, params: Union[str, dict], **kwargs) -> str:
        if isinstance(params, str):
            params = json.loads(params) if params.startswith('{') else {'text': params}
        
        text = params.get('text', '') if isinstance(params, dict) else params
        target_lang = params.get('target_language', '') if isinstance(params, dict) else ''
        
        if not text:
            return "你要翻译什么内容呀？"
        
        import dashscope
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
        
        detected_lang = self._detect_language(text)
        
        if not target_lang:
            if detected_lang == '中文':
                target_lang = 'English'
                target_name = '英文'
            else:
                target_lang = 'Chinese'
                target_name = '中文'
        else:
            target_name = target_lang if '中文' in target_lang or '英文' in target_lang else target_lang
            if '中文' in target_lang:
                target_lang = 'Chinese'
            elif '英文' in target_lang or 'English' in target_lang:
                target_lang = 'English'
        
        try:
            from dashscope import Generation
            response = Generation.call(
                model='qwen-plus',
                messages=[{
                    'role': 'user',
                    'content': f'请将以下文本翻译成{target_name}，只返回翻译结果，不要有其他内容：{text}'
                }],
                result_format='message'
            )
            
            if response.status_code == 200:
                translation = response.output.choices[0]['message']['content']
                return f"翻译结果：{translation}"
            else:
                return f"翻译失败，请再试一次。"
        except Exception as e:
            return f"抱歉，翻译出问题，可能系网络唔稳定。"
    
    def _detect_language(self, text: str) -> str:
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        total_chars = len(text)
        
        if chinese_chars / total_chars > 0.5:
            return '中文'
        else:
            return '英文'
