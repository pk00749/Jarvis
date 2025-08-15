"""实时音频流处理模块."""

import os
import queue
import tempfile
import threading
import wave
from typing import Callable, Optional

import pyaudio

from .config import WakeWordConfig


class AudioStreamProcessor:
    """精简的实时音频流处理器."""

    def __init__(self, config: WakeWordConfig):
        """初始化音频流处理器."""
        self.config = config
        self._audio = None
        self._stream = None
        self._recording = False
        self._audio_queue = queue.Queue(maxsize=50)  # 限制队列大小防止内存泄漏
        self._processing_thread: Optional[threading.Thread] = None
        self._audio_callback: Optional[Callable] = None

    def set_audio_callback(self, callback: Callable[[str], None]):
        """设置音频处理回调函数."""
        self._audio_callback = callback

    def start_streaming(self) -> bool:
        """开始音频流采集."""
        if self._recording:
            return True

        try:
            self._audio = pyaudio.PyAudio()
            self._stream = self._audio.open(
                format=getattr(pyaudio, self.config.format_type),
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size,
                stream_callback=self._audio_callback_handler
            )

            self._recording = True
            self._processing_thread = threading.Thread(
                target=self._process_audio, daemon=True
            )
            self._processing_thread.start()

            print("🎤 音频流已启动")
            return True

        except Exception as e:
            print(f"❌ 启动音频流失败: {e}")
            self.stop_streaming()
            return False

    def stop_streaming(self):
        """停止音频流采集."""
        self._recording = False

        # 清理音频流
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._audio:
            try:
                self._audio.terminate()
            except Exception:
                pass
            self._audio = None

        # 等待处理线程结束
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=1.0)

        # 清空队列
        self._clear_queue()
        print("🛑 音频流已停止")

    def _audio_callback_handler(self, in_data, frame_count, time_info, status):
        """音频流回调处理器."""
        if self._recording:
            try:
                self._audio_queue.put_nowait(in_data)
            except queue.Full:
                # 队列满时移除最老的数据
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.put_nowait(in_data)
                except queue.Empty:
                    pass
        return (in_data, pyaudio.paContinue)

    def _process_audio(self):
        """音频处理主循环."""
        # 简化：每2秒处理一次音频
        chunk_duration = 2.0
        chunks_needed = int(
            chunk_duration * self.config.sample_rate / self.config.chunk_size
        )
        audio_chunks = []

        while self._recording:
            try:
                chunk = self._audio_queue.get(timeout=1.0)
                audio_chunks.append(chunk)

                if len(audio_chunks) >= chunks_needed:
                    self._process_and_callback(audio_chunks)
                    audio_chunks = []

            except queue.Empty:
                # 超时时处理已有数据
                if audio_chunks:
                    self._process_and_callback(audio_chunks)
                    audio_chunks = []
            except Exception as e:
                print(f"❌ 音频处理错误: {e}")
                break

    def _process_and_callback(self, audio_chunks):
        """处理音频块并调用回调."""
        if not audio_chunks or not self._audio_callback:
            return

        try:
            # 合并音频数据
            audio_data = b''.join(audio_chunks)

            # 保存为临时文件并调用回调
            temp_file = self._create_temp_wav(audio_data)
            if temp_file:
                self._audio_callback(temp_file)
                # 清理临时文件
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

        except Exception as e:
            print(f"❌ 音频回调错误: {e}")

    def _create_temp_wav(self, audio_data: bytes) -> Optional[str]:
        """创建临时WAV文件."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_filename = f.name

            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.config.sample_rate)
                wf.writeframes(audio_data)

            return temp_filename

        except Exception as e:
            print(f"❌ 创建临时音频文件失败: {e}")
            return None

    def _clear_queue(self):
        """清空音频队列."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def is_recording(self) -> bool:
        """检查是否正在录音."""
        return self._recording

    def get_audio_info(self) -> dict:
        """获取音频流信息."""
        return {
            'recording': self._recording,
            'sample_rate': self.config.sample_rate,
            'channels': self.config.channels,
            'chunk_size': self.config.chunk_size,
            'queue_size': self._audio_queue.qsize()
        }
