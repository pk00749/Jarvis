import logging
from typing import Generator, Optional, Tuple, List
import gradio as gr
import os
import sys
import numpy as np
from dataclasses import dataclass
from influence.influence import Influence
from listen.listen import Listen
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from snownlp import SnowNLP

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """配置类"""
    ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))
    SAMPLE_RATE: int = 24000
    PROMPT_WAV_PATH: str = 'asset/zero_shot_prompt.wav'
    COSYVOICE_MODEL_PATH: str = 'pretrained_models/CosyVoice2-0.5B'
    BUFFER_SIZE: int = 48000  # 2秒的音频长度
    OVERLAP_SIZE: int = 2400  # 0.1秒的重叠长度
    CHUNK_SIZE: int = 24000   # 1秒的音频长度
    MAX_AUDIO_LENGTH: int = 480000  # 20秒的最大音频长度

class AudioBuffer:
    """改进的音频缓冲器类，用于平滑处理音频流"""
    def __init__(self, buffer_size: int, overlap_size: int):
        self.buffer_size = buffer_size
        self.overlap_size = overlap_size
        self.buffer = np.zeros(0)
        self.fade_in = np.linspace(0, 1, overlap_size)
        self.fade_out = np.linspace(1, 0, overlap_size)

    def process(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """处理音频块，返回处理后的数据"""
        # 确保chunk是一维数组
        if chunk.ndim > 1:
            chunk = chunk.flatten()

        # 将新的chunk添加到缓冲区
        self.buffer = np.concatenate([self.buffer, chunk])

        # 如果缓冲区数据不够，返回None
        if len(self.buffer) < self.buffer_size:
            return None

        # 准备输出数据
        output = self.buffer[:self.buffer_size]

        # 应用交叉淡入淡出
        if len(output) >= self.overlap_size:
            output[:self.overlap_size] *= self.fade_in

        # 保留重叠部分
        self.buffer = self.buffer[self.buffer_size - self.overlap_size:]
        if len(self.buffer) >= self.overlap_size:
            self.buffer[:self.overlap_size] *= self.fade_out

        return output

class AudioProcessor:
    def __init__(self, sample_rate: int, chunk_size: int):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_buffer = []
        self.total_length = 0

    def add_chunk(self, chunk: np.ndarray) -> None:
        """添加音频块到缓冲区"""
        if chunk.ndim > 1:
            chunk = chunk.flatten()
        self.audio_buffer.append(chunk)
        self.total_length += len(chunk)

    def get_audio(self) -> Optional[np.ndarray]:
        """获取处理后的完整音频"""
        if not self.audio_buffer:
            return None

        # 合并所有音频块
        audio = np.concatenate(self.audio_buffer)

        # 应用淡入淡出效果
        fade_length = min(2048, len(audio) // 10)
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)

        audio[:fade_length] *= fade_in
        audio[-fade_length:] *= fade_out

        # 重置缓冲区
        self.audio_buffer = []
        self.total_length = 0

        return audio

class JarvisApp:
    def __init__(self):
        self.config = Config()
        self._init_paths()
        self.cosyvoice = self._init_cosyvoice()
        self.prompt_speech = self._load_prompt_speech()
        self.audio_buffer = AudioBuffer(self.config.BUFFER_SIZE, self.config.OVERLAP_SIZE)
        self.audio_processor = AudioProcessor(self.config.SAMPLE_RATE, self.config.CHUNK_SIZE)

    def _init_paths(self) -> None:
        """初始化路径"""
        logger.info(f'Root path: {self.config.ROOT_DIR}')
        sys.path.append(f'{self.config.ROOT_DIR}/third_party/AcademiCodec')
        sys.path.append(f'{self.config.ROOT_DIR}/third_party/Matcha-TTS')

    def _init_cosyvoice(self) -> CosyVoice2:
        """初始化CosyVoice模型"""
        return CosyVoice2(
            f'{self.config.ROOT_DIR}/{self.config.COSYVOICE_MODEL_PATH}',
            load_jit=False,
            load_trt=False,
            fp16=False
        )

    def _load_prompt_speech(self) -> np.ndarray:
        """加载提示音频"""
        return load_wav(
            f'{self.config.ROOT_DIR}/{self.config.PROMPT_WAV_PATH}',
            16000
        )

    @staticmethod
    def process_audio(audio: Optional[np.ndarray]) -> str:
        """处理输入音频"""
        try:
            if audio is None:
                return "No voice to be recorded."
            filename = Listen.save_voice(audio)
            return Influence.voice_to_text(filename)
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return f"Failed to record voice: {e}"

    @staticmethod
    def split_text(text: str) -> List[str]:
        """使用NLP分割文本"""
        try:
            result = SnowNLP(text)
            logger.info(f"Split sentences: {result.sentences}")
            return result.sentences
        except Exception as e:
            logger.error(f"Error splitting text: {e}")
            return [text]

    @staticmethod
    def text_generator(texts: List[str]) -> Generator[str, None, None]:
        """文本生成器"""
        for text in texts:
            logger.info(f"Processing text: {text}")
            yield text

    def process_audio_stream(self,
                           audio: Optional[np.ndarray]) -> Generator[Tuple[int, np.ndarray], None, None]:
        """处理音频流"""
        try:
            prompt_text = self.process_audio(audio)
            answer_text = Influence.llm(prompt_text)
            text_chunks = self.split_text(answer_text)

            # 收集完整的音频数据
            for chunk in self.cosyvoice.inference_instruct2(
                tts_text=self.text_generator(text_chunks),
                instruct_text='用粤语说这句话',
                prompt_speech_16k=self.prompt_speech,
                stream=True
            ):
                audio_chunk = self._process_audio_chunk(chunk['tts_speech'].cpu().numpy())
                self.audio_processor.add_chunk(audio_chunk)

                # 当累积的音频长度超过阈值时，输出完整的音频段
                if self.audio_processor.total_length >= self.config.BUFFER_SIZE:
                    processed_audio = self.audio_processor.get_audio()
                    if processed_audio is not None:
                        yield (self.config.SAMPLE_RATE, processed_audio)

            # 处理剩余的音频数据
            remaining_audio = self.audio_processor.get_audio()
            if remaining_audio is not None:
                yield (self.config.SAMPLE_RATE, remaining_audio)

        except Exception as e:
            logger.error(f"Error in audio stream processing: {e}")
            yield (self.config.SAMPLE_RATE, np.zeros(1))

    @staticmethod
    def _process_audio_chunk(chunk: np.ndarray) -> np.ndarray:
        """处理音频块"""
        chunk = np.asarray(chunk, dtype=np.float32)
        chunk = np.nan_to_num(chunk)

        # 应用音频增强
        chunk = np.clip(chunk, -1.0, 1.0)
        if chunk.max() > 1e-6:
            chunk = chunk / np.abs(chunk).max() * 0.95

        # 应用简单的降噪
        noise_threshold = 0.01
        chunk[np.abs(chunk) < noise_threshold] = 0

        return chunk.flatten() if chunk.ndim > 1 else chunk

    def create_ui(self) -> None:
        """Create Gradio UI"""
        with gr.Blocks() as ui:
            output_audio = gr.Audio(
                label="Jarvis",
                sources=["microphone"],
                autoplay=True,
                streaming=True,
                elem_id="output_audio",
                show_label=True
            )
            input_audio = gr.Audio(
                sources=["microphone"],
                label="Me"
            )

            interface = gr.Interface(
                fn=self.process_audio_stream,
                inputs=[input_audio],
                outputs=[output_audio],
                title="Jarvis👾",
                description="Talk with Jarvis, your AI assistant.",
                concurrency_limit=1
            )

        ui.launch(
            share=False,
            debug=True,
            server_port=7860,
            server_name="0.0.0.0",
            max_threads=4  # 限制最大工作线程数
        )

def main():
    app = JarvisApp()
    app.create_ui()

if __name__ == "__main__":
    main()
