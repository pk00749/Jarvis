import gradio as gr
import numpy as np
import soundfile as sf
import queue
import threading
from config.performance import AUDIO_CONFIG, MEMORY_CONFIG
import gc

class AudioStreamProcessor:
    def __init__(self):
        self.audio_queue = queue.Queue(maxsize=AUDIO_CONFIG['stream_buffer_count'])
        self.buffer = np.zeros(AUDIO_CONFIG['buffer_size'], dtype='float32')
        self.overlap_buffer = np.zeros(AUDIO_CONFIG['overlap_size'], dtype='float32')
        self._stop_flag = False

    def preload_audio(self, file_path):
        """预加载音频数据到队列"""
        def _preload():
            try:
                with sf.SoundFile(file_path) as f:
                    while not self._stop_flag:
                        data = f.read(AUDIO_CONFIG['chunk_size'], dtype='float32')
                        if len(data) == 0:
                            break
                        if data.ndim > 1:
                            data = data[:, 0]
                        self.audio_queue.put(data)
            except Exception as e:
                print(f"预加载错误: {e}")

        thread = threading.Thread(target=_preload, daemon=True)
        thread.start()
        return thread

    def process_stream(self, dummy):
        """处理音频流"""
        self._stop_flag = False
        file = "record_output.wav"
        sr = 16000

        # 启动预加载线程
        preload_thread = self.preload_audio(file)

        try:
            while not self._stop_flag:
                # 获取音频块
                try:
                    chunk = self.audio_queue.get(timeout=1.0)
                except queue.Empty:
                    break

                # 应用重叠
                if len(chunk) < AUDIO_CONFIG['chunk_size']:
                    chunk = np.pad(chunk, (0, AUDIO_CONFIG['chunk_size'] - len(chunk)))

                # 混合重叠区域
                output = np.zeros(len(chunk))
                output[:AUDIO_CONFIG['overlap_size']] = self.overlap_buffer
                output[AUDIO_CONFIG['overlap_size']:] = chunk[AUDIO_CONFIG['overlap_size']:]

                # 保存新的重叠区域
                self.overlap_buffer = chunk[-AUDIO_CONFIG['overlap_size']:]

                # 定期进行垃圾回收
                if MEMORY_CONFIG['gc_interval'] > 0 and self.audio_queue.qsize() % MEMORY_CONFIG['gc_interval'] == 0:
                    gc.collect()

                yield (sr, output.astype(np.float32))

        finally:
            self._stop_flag = True
            preload_thread.join(timeout=1.0)
            # 清理
            self.audio_queue.queue.clear()
            gc.collect()

# 创建全局处理器实例
stream_processor = AudioStreamProcessor()

demo = gr.Interface(
    fn=stream_processor.process_stream,
    inputs=gr.Button("流式播放WAV"),
    outputs=gr.Audio(
        label="流式音频",
        streaming=True,
        autoplay=True,
        speed=1.0,  # 保持原始速度
    ),
    title="Gradio流式播放本地WAV",
    theme="default"
)

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        max_threads=MEMORY_CONFIG['max_threads']
    )
