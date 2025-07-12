import logging
from typing import Generator, Optional, Tuple, List
import gradio as gr
import os
import sys
import numpy as np
from dataclasses import dataclass
from influence.influence import Influence
from listen.listen import Listen
from speak.speak import Speak
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

class JarvisApp:
    def __init__(self):
        self.config = Config()
        self._init_paths()
        self.cosyvoice = self._init_cosyvoice()
        self.prompt_speech = self._load_prompt_speech()

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

            for chunk in self.cosyvoice.inference_instruct2(
                tts_text=self.text_generator(text_chunks),
                instruct_text='用粤语说这句话',
                prompt_speech_16k=self.prompt_speech,
                stream=True
            ):
                audio_chunk = self._process_audio_chunk(chunk['tts_speech'].cpu().numpy())
                yield (self.config.SAMPLE_RATE, audio_chunk)
        except Exception as e:
            logger.error(f"Error in audio stream processing: {e}")
            yield (self.config.SAMPLE_RATE, np.zeros(1))

    @staticmethod
    def _process_audio_chunk(chunk: np.ndarray) -> np.ndarray:
        """处理音频块"""
        chunk = np.asarray(chunk, dtype=np.float32)
        chunk = np.nan_to_num(chunk)
        chunk = np.clip(chunk, -1.0, 1.0)
        return chunk.flatten() if chunk.ndim > 1 else chunk

    def create_ui(self) -> None:
        """创建用户界面"""
        with gr.Blocks() as ui:
            output_audio = gr.Audio(sources=["microphone"], autoplay=True)
            gr.Interface(
                fn=self.process_audio_stream,
                inputs=gr.Audio(sources=["microphone"]),
                outputs=[output_audio],
                title="Jarvis👾",
                description=""
            )
        ui.launch()

def main():
    app = JarvisApp()
    app.create_ui()

if __name__ == "__main__":
    main()
