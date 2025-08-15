"""关键词匹配模块."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from .config import WakeWordConfig


@dataclass
class MatchResult:
    """匹配结果数据类."""
    text: str
    weight: float
    word_count: int
    is_triggered: bool
    timestamp: datetime
    confidence: float = 0.0


class KeywordMatcher:
    """精简的关键词匹配器."""

    def __init__(self, config: WakeWordConfig):
        """初始化关键词匹配器."""
        self.config = config
        self.history: List[MatchResult] = []

    def match_wake_word(self, text: str) -> MatchResult:
        """匹配唤醒词并计算权重."""
        if not text:
            return self._create_empty_result("")

        # 清理文本，只保留中文字符
        cleaned_text = re.sub(r'[^\u4e00-\u9fff]', '', text)

        # 计算连续"喂"的数量
        wake_word = self.config.wake_word
        word_count = cleaned_text.count(wake_word)

        # 计算权重
        if word_count == 0:
            weight = 0.0
        elif word_count == 1:
            weight = self.config.single_word_weight
        elif word_count == 2:
            weight = self.config.double_word_weight
        else:
            weight = self.config.triple_word_weight

        # 判断是否触发
        is_triggered = weight >= self.config.sensitivity_threshold
        confidence = (min(weight / self.config.max_threshold, 1.0) if weight > 0 else 0.0)

        result = MatchResult(
            text=text,
            weight=weight,
            word_count=word_count,
            is_triggered=is_triggered,
            timestamp=datetime.now(),
            confidence=confidence
        )

        # 添加到历史记录
        self._add_to_history(result)
        return result

    def _create_empty_result(self, text: str) -> MatchResult:
        """创建空结果."""
        return MatchResult(
            text=text,
            weight=0.0,
            word_count=0,
            is_triggered=False,
            timestamp=datetime.now()
        )

    def _add_to_history(self, result: MatchResult):
        """添加结果到历史记录."""
        self.history.append(result)

        # 保持历史记录大小限制
        if len(self.history) > self.config.max_history_size:
            self.history = self.history[-self.config.max_history_size:]

    def get_statistics(self) -> Dict:
        """获取匹配统计信息."""
        if not self.history:
            return {
                'total_attempts': 0,
                'successful_triggers': 0,
                'success_rate': 0.0
            }

        total_attempts = len(self.history)
        successful_triggers = sum(1 for r in self.history if r.is_triggered)
        success_rate = successful_triggers / total_attempts

        return {'total_attempts': total_attempts, 'successful_triggers': successful_triggers,
                'success_rate': success_rate}

    def clear_history(self):
        """清空历史记录."""
        self.history.clear()

    def update_config(self, new_config: WakeWordConfig):
        """更新配置."""
        self.config = new_config
