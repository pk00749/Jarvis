#!/usr/bin/env python3
"""
唤醒词与语音对话集成测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from wake_word import WakeWordDetector, WakeWordConfig
from wake_word.keyword_matcher import MatchResult
import time
import threading

class WakeWordIntegrator:
    """唤醒词与语音对话集成器"""

    def __init__(self):
        self.config = WakeWordConfig()
        self.detector = WakeWordDetector(self.config)
        self.is_in_conversation = False

        # 设置回调
        self.detector.set_wake_detected_callback(self._on_wake_detected)
        self.detector.set_state_change_callback(self._on_state_change)

    def _on_wake_detected(self, result: MatchResult):
        """唤醒词检测成功回调"""
        print(f"🎉 唤醒成功! 文本: '{result.text}', 权重: {result.weight}")

        # 模拟进入对话流程
        self._start_conversation()

    def _on_state_change(self, state, reason):
        """状态变化回调"""
        print(f"状态变化: {state.value} ({reason})")

    def _start_conversation(self):
        """开始对话流程"""
        print("🗣️ 开始对话流程...")
        self.is_in_conversation = True

        # 模拟对话处理
        def conversation_thread():
            # 模拟语音识别 + LLM + TTS的时间
            print("🎤 正在识别用户语音...")
            time.sleep(1)

            print("🧠 LLM正在生成回答...")
            time.sleep(2)

            print("🔊 TTS正在播放回答...")
            # 播放时暂停唤醒检测
            self.detector.pause_detection()
            time.sleep(3)

            # 播放完成后恢复监听
            self.detector.resume_detection()
            self.is_in_conversation = False
            print("✅ 对话完成，继续监听唤醒词...")

        # 启动对话线程
        thread = threading.Thread(target=conversation_thread, daemon=True)
        thread.start()

    def start_integrated_test(self):
        """启动集成测试"""
        print("=== 唤醒词与语音对话集成测试 ===")

        # 启动唤醒检测
        if self.detector.start_detection():
            print("✅ 唤醒词检测已启动")

            # 模拟语音输入测试
            test_inputs = [
                "你好",           # 不触发
                "喂",             # 单个喂，可能不触发（阈值1.2）
                "喂喂",           # 双喂，应该触发
                "没有关键词",      # 不触发
                "喂喂喂"          # 三喂，应该触发
            ]

            for i, test_input in enumerate(test_inputs):
                print(f"\n--- 测试 {i+1}: '{test_input}' ---")

                # 直接调用关键词匹配进行测试
                result = self.detector.keyword_matcher.match_wake_word(test_input)

                if result.is_triggered:
                    self._on_wake_detected(result)
                    # 等待对话完成
                    while self.is_in_conversation:
                        time.sleep(0.5)
                else:
                    print(f"未触发: 权重{result.weight} < 阈值{self.config.sensitivity_threshold}")

                time.sleep(1)

            print("\n=== 测试完成 ===")
            self.detector.stop_detection()
        else:
            print("❌ 唤醒词检测启动失败")

if __name__ == "__main__":
    integrator = WakeWordIntegrator()
    integrator.start_integrated_test()
