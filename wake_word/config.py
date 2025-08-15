from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import json
import os

@dataclass
class WakeWordConfig:
    """唤醒词配置类"""
    # 基础配置
    enabled: bool = True
    wake_word: str = "喂"
    language: str = "yue"  # 粤语

    # 敏感度配置
    sensitivity_threshold: float = 1.2  # 触发阈值，默认中等敏感度
    min_threshold: float = 0.8         # 最小阈值（高敏感度）
    max_threshold: float = 2.0         # 最大阈值（低敏感度）

    # 权重配置
    single_word_weight: float = 1.0    # 单个"喂"权重
    double_word_weight: float = 1.5    # 连续"喂喂"权重
    triple_word_weight: float = 2.0    # 三个及以上"喂"权重

    # 音频配置
    sample_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    format_type: str = "paInt16"

    # 历史记录配置
    max_history_size: int = 5

    # 超时配置
    conversation_timeout: float = 30.0  # 对话超时时间（秒）

    def save_to_file(self, file_path: str) -> None:
        """保存配置到文件"""
        config_dict = {
            'enabled': self.enabled,
            'wake_word': self.wake_word,
            'language': self.language,
            'sensitivity_threshold': self.sensitivity_threshold,
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,
            'single_word_weight': self.single_word_weight,
            'double_word_weight': self.double_word_weight,
            'triple_word_weight': self.triple_word_weight,
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'channels': self.channels,
            'format_type': self.format_type,
            'max_history_size': self.max_history_size,
            'conversation_timeout': self.conversation_timeout
        }

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'WakeWordConfig':
        """从文件加载配置"""
        if not os.path.exists(file_path):
            return cls()

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return cls()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'enabled': self.enabled,
            'wake_word': self.wake_word,
            'language': self.language,
            'sensitivity_threshold': self.sensitivity_threshold,
            'min_threshold': self.min_threshold,
            'max_threshold': self.max_threshold,
            'single_word_weight': self.single_word_weight,
            'double_word_weight': self.double_word_weight,
            'triple_word_weight': self.triple_word_weight,
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'channels': self.channels,
            'format_type': self.format_type,
            'max_history_size': self.max_history_size,
            'conversation_timeout': self.conversation_timeout
        }
