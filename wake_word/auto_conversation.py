"""自动对话处理模块."""

import logging
import threading
from typing import Optional

import numpy as np
import pyaudio

# 设置日志
logger = logging.getLogger(__name__)


class AutoConversationHandler:
    """精简的自动对话处理器."""

    def __init__(self, jarvis_brain_streaming_func):
        """初始化自动对话处理器."""
        self.brain_streaming = jarvis_brain_streaming_func
        self.is_recording = False
        self._conversation_lock = threading.Lock()

    def handle_wake_up(self, wake_result):
        """处理唤醒事件，自动开始录音和对话."""
        with self._conversation_lock:
            if self.is_recording:
                logger.warning("⚠️ 对话进行中，忽略新唤醒")
                return

            logger.info("🎤 唤醒成功！开始录音...")
            self.is_recording = True

            # 在新线程中处理对话，避免阻塞
            threading.Thread(target=self._handle_conversation, args=(wake_result,), daemon=True).start()

    def _handle_conversation(self, wake_result):
        """处理完整的对话流程."""
        try:
            # 录音
            audio_data = self._record_audio()
            if not audio_data:
                logger.warning("⚠️ 未检测到有效语音")
                return

            logger.info("🧠 处理语音中...")

            # 调用Jarvis语音处理
            response = self.brain_streaming(audio_data)

            # 处理响应（支持生成器和普通返回值）
            if hasattr(response, '__iter__'):
                # 消费生成器但不处理具体内容（音频会自动播放）
                list(response)

            logger.info("✅ 对话完成")

        except Exception as e:
            logger.error(f"❌ 对话处理失败: {e}")
        finally:
            with self._conversation_lock:
                self.is_recording = False

    def _record_audio(self):
        """录制音频 - 精简版."""
        # 音频参数
        CHUNK, FORMAT, CHANNELS, RATE = 1024, pyaudio.paInt16, 1, 16000
        RECORD_SECONDS = 5

        audio_instance = None
        stream = None

        try:
            audio_instance = pyaudio.PyAudio()

            # 简化的流初始化
            stream = audio_instance.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

            logger.info("🎤 录音中，请说话...")

            # 录制音频数据
            frames = []
            for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                except Exception:
                    # 忽略单次读取错误，继续录音
                    continue

            logger.info("🎤 录音结束")

            # 转换为numpy格式
            audio_data = b''.join(frames)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            return (RATE, audio_array)

        except Exception as e:
            logger.error(f"❌ 录音失败: {e}")
            return None
        finally:
            # 清理资源
            if stream:
                stream.stop_stream()
                stream.close()
            if audio_instance:
                audio_instance.terminate()

    def stop_conversation(self):
        """停止当前对话."""
        with self._conversation_lock:
            if self.is_recording:
                logger.info("🛑 停止对话")
                self.is_recording = False
