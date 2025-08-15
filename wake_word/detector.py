"""唤醒词检测器模块."""

import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

# 导入现有的语音识别功能
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from influence.influence import Influence

from .audio_stream import AudioStreamProcessor
from .config import WakeWordConfig
from .keyword_matcher import KeywordMatcher, MatchResult


class WakeWordDetector:
    """精简的唤醒词检测器."""

    def __init__(self, config: Optional[WakeWordConfig] = None):
        """初始化唤醒词检测器."""
        self.config = config or WakeWordConfig()

        # 核心组件
        self.keyword_matcher = KeywordMatcher(self.config)
        self.audio_processor = AudioStreamProcessor(self.config)

        # 回调和状态
        self._wake_detected_callback: Optional[Callable] = None
        self._is_running = False
        self._detection_thread: Optional[threading.Thread] = None

        # 识别结果
        self._current_recognition = ""
        self._last_recognition_time = datetime.now()
        self._history: List[Dict[str, Any]] = []

        # 设置音频回调
        self.audio_processor.set_audio_callback(self._process_voice_input)

    def set_wake_detected_callback(self, callback: Callable[[MatchResult], None]):
        """设置唤醒检测回调函数."""
        self._wake_detected_callback = callback

    def start_listening(self) -> bool:
        """开始监听唤醒词."""
        if self._is_running:
            return True

        try:
            self._is_running = True
            self.audio_processor.start_streaming()

            # 启动检测线程
            self._detection_thread = threading.Thread(
                target=self._detection_loop, daemon=True
            )
            self._detection_thread.start()

            return True
        except Exception as e:
            print(f"❌ 启动监听失败: {e}")
            self._is_running = False
            return False

    def stop_listening(self):
        """停止监听."""
        self._is_running = False
        self.audio_processor.stop_streaming()

    def _detection_loop(self):
        """检测循环."""
        while self._is_running:
            time.sleep(0.1)

    def _process_voice_input(self, audio_file_path):
        """处理语音输入."""
        if not self._is_running:
            return

        try:
            # 语音识别
            recognition_result = Influence.voice_to_text(audio_file_path)
            if not recognition_result:
                return

            # 提取文本
            text_result = self._extract_text(recognition_result)
            if not text_result:
                return

            self._current_recognition = text_result
            self._last_recognition_time = datetime.now()

            print(f"🎤 识别结果: {text_result}")

            # 检测唤醒词
            match_result = self.keyword_matcher.match_wake_word(text_result)
            self._add_to_history(text_result, match_result)

            if match_result.is_triggered:
                print(f"✅ 匹配成功，权重: {match_result.weight}")
                self._on_wake_detected(match_result)

        except Exception as e:
            print(f"❌ 语音处理出错: {e}")

    def _extract_text(self, recognition_result):
        """提取识别结果中的文本."""
        if isinstance(recognition_result, list) and recognition_result:
            return recognition_result[0].get('text', '') if recognition_result[0] else ''
        elif isinstance(recognition_result, dict):
            return recognition_result.get('text', '')
        elif isinstance(recognition_result, str):
            return recognition_result
        return str(recognition_result) if recognition_result else ''

    def _on_wake_detected(self, match_result: MatchResult):
        """唤醒词检测成功回调."""
        if self._wake_detected_callback:
            self._wake_detected_callback(match_result)

    def _add_to_history(self, text: str, match_result: MatchResult):
        """添加识别结果到历史记录."""
        history_item = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'text': text,
            'weight': match_result.weight,
            'word_count': match_result.word_count,
            'is_triggered': match_result.is_triggered
        }

        self._history.append(history_item)

        # 保持历史记录限制
        max_history = getattr(self.config, 'max_history_size', 10)
        if len(self._history) > max_history:
            self._history = self._history[-max_history:]

    def get_status(self) -> Dict[str, Any]:
        """获取检测器状态."""
        return {
            'enabled': self.config.enabled,
            'is_running': self._is_running,
            'current_recognition': self._current_recognition,
            'last_recognition_time': self._last_recognition_time.isoformat(),
            'wake_word': self.config.wake_word,
            'sensitivity': self.config.sensitivity_threshold
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史记录."""
        return self._history.copy()

    def clear_history(self):
        """清空历史记录."""
        self._history.clear()

    def reset(self):
        """重置检测器."""
        self._current_recognition = ""
        self._last_recognition_time = datetime.now()
        self._history.clear()

    # 兼容接口
    def start_detection(self) -> bool:
        """开始检测（兼容接口）."""
        return self.start_listening()

    def stop_detection(self):
        """停止检测（兼容接口）."""
        return self.stop_listening()

    def enable(self):
        """启用检测器."""
        self.config.enabled = True

    def disable(self):
        """禁用检测器."""
        self.config.enabled = False

    def update_sensitivity(self, sensitivity: float):
        """更新敏感度."""
        self.config.sensitivity_threshold = sensitivity
