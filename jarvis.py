import logging
from typing import Generator, Optional, Tuple, List
import gradio as gr
import os
import sys
import numpy as np
import torch
from dataclasses import dataclass
from influence.influence import Influence
from listen.listen import Listen
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import threading

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """配置类 - 针对Mac M3优化"""
    ROOT_DIR: str = os.path.dirname(os.path.abspath(__file__))
    SAMPLE_RATE: int = 24000
    PROMPT_WAV_PATH: str = 'asset/zero_shot_prompt.wav'
    COSYVOICE_MODEL_PATH: str = 'pretrained_models/CosyVoice2-0.5B'
    BUFFER_SIZE: int = 12000  # 500ms 缓冲，适合M3处理速度
    CHUNK_SIZE: int = 6000    # 250ms 块大小
    OVERLAP_SIZE: int = 1200  # 50ms 重叠
    MAX_CHUNKS_IN_MEMORY: int = 2
    MAX_TEXT_LENGTH: int = 50
    RECORDINGS_DIR: str = 'recordings'

    def __post_init__(self):
        """优化Mac M3性能配置"""
        # 为Apple Silicon优化线程配置
        os.environ['MKL_NUM_THREADS'] = '4'  # M3可以支持更多线程
        os.environ['NUMEXPR_NUM_THREADS'] = '4'
        os.environ['OMP_NUM_THREADS'] = '4'
        os.environ['OPENBLAS_NUM_THREADS'] = '4'

        # 创建录音目录
        recordings_path = os.path.join(self.ROOT_DIR, self.RECORDINGS_DIR)
        os.makedirs(recordings_path, exist_ok=True)

    @staticmethod
    def split_text(text: str) -> List[str]:
        """智能文本分割，支持中文和粤语"""
        import re

        # 确保输入是字符串类型
        if not isinstance(text, str):
            text = str(text)

        # 清理文本
        text = ' '.join(text.split())

        if not text.strip():
            return [""]

        # 按句号、感叹号、问号分割
        sentences = []
        split_parts = re.split(r'([。！？!?])', text)

        for i, part in enumerate(split_parts):
            # 确保part是字符串并且不为空
            if isinstance(part, str) and part.strip():
                # 如果是标点符号，与前一个句子合并
                if part in '。！？!?' and sentences:
                    sentences[-1] += part
                else:
                    # 长句按逗号分割
                    if len(part) > Config.MAX_TEXT_LENGTH:
                        sub_parts = re.split(r'([,，、])', part)
                        for j, sub_part in enumerate(sub_parts):
                            if isinstance(sub_part, str) and sub_part.strip():
                                if sub_part in ',，、' and sentences:
                                    sentences[-1] += sub_part
                                else:
                                    sentences.append(sub_part.strip())
                    else:
                        sentences.append(part.strip())

        # 过滤掉空字符串和无效内容
        valid_sentences = []
        for sentence in sentences:
            if isinstance(sentence, str) and sentence.strip() and len(sentence.strip()) > 0:
                # 确保句子不只是标点符号
                clean_sentence = sentence.strip()
                if not re.match(r'^[。！？!?，、,]+$', clean_sentence):
                    valid_sentences.append(clean_sentence)

        return valid_sentences if valid_sentences else [""]

class AudioBuffer:
    """音频缓冲器，优化流式处理"""
    def __init__(self, buffer_size: int, overlap_size: int):
        self.buffer_size = buffer_size
        self.overlap_size = overlap_size
        self.buffer = np.zeros(0)
        self.fade_in = np.linspace(0, 1, overlap_size)
        self.fade_out = np.linspace(1, 0, overlap_size)

    def process(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """处理音频块"""
        if chunk.ndim > 1:
            chunk = chunk.flatten()

        self.buffer = np.concatenate([self.buffer, chunk])

        if len(self.buffer) < self.buffer_size:
            return None

        output = self.buffer[:self.buffer_size]

        # 应用淡入淡出效果
        if len(output) >= self.overlap_size:
            output[:self.overlap_size] *= self.fade_in

        self.buffer = self.buffer[self.buffer_size - self.overlap_size:]
        if len(self.buffer) >= self.overlap_size:
            self.buffer[:self.overlap_size] *= self.fade_out

        return output

class JarvisApp:
    def __init__(self):
        self.config = Config()
        self._init_paths()
        self.cosyvoice = self._init_cosyvoice()
        self.prompt_speech = self._load_prompt_speech()
        self.audio_buffer = AudioBuffer(self.config.BUFFER_SIZE, self.config.OVERLAP_SIZE)
        self.is_processing = False
        self.processing_lock = threading.Lock()

    def _init_paths(self) -> None:
        """初始化路径"""
        logger.info(f'Root path: {self.config.ROOT_DIR}')
        sys.path.append(f'{self.config.ROOT_DIR}/third_party/AcademiCodec')
        sys.path.append(f'{self.config.ROOT_DIR}/third_party/Matcha-TTS')

    def _init_cosyvoice(self) -> CosyVoice2:
        """初始化CosyVoice��型，针对Mac M3优化"""
        import torch

        # Mac M3优化配置
        torch.set_num_threads(4)  # M3可以支持更多线程

        # 如果有MPS（Metal Performance Shaders）支持，使用它
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Using device: {device}")

        model = CosyVoice2(
            f'{self.config.ROOT_DIR}/{self.config.COSYVOICE_MODEL_PATH}',
            load_jit=True,
            load_trt=False,
            fp16=True,
        )

        return model

    def _load_prompt_speech(self) -> np.ndarray:
        """加载提示音频"""
        return load_wav(
            f'{self.config.ROOT_DIR}/{self.config.PROMPT_WAV_PATH}',
            16000
        )

    def process_audio(self, audio: Optional[np.ndarray]) -> str:
        """处理输入音频"""
        try:
            if audio is None:
                return "没有录制到语音"

            # 处理Gradio音频数据格式
            if isinstance(audio, tuple):
                # 如果是元组格式 (sample_rate, audio_data)
                sample_rate, audio_data = audio
                audio = audio_data

            # 保存音频文件
            filename = Listen.save_voice(audio)

            # 语音转文字
            text = Influence.voice_to_text(filename)
            logger.info(f"识别到的文本: {text}")

            return text

        except Exception as e:
            logger.error(f"处理音频时出错: {e}")
            return f"处理失败: {e}"

    def generate_response(self, prompt_text: str) -> str:
        """生成回应"""
        try:
            # 使用大语言模型生成回应
            response = Influence.llm(prompt_text)
            logger.info(f"生成的回应: {response}")
            return response

        except Exception as e:
            logger.error(f"生成回应时出�����: {e}")
            return "抱歉，我现在无法处理您的请求。"

    def text_to_speech_stream(self, text: str) -> Generator[Tuple[int, np.ndarray], None, None]:
        """文本转语音流 - 优化粤语语音合成"""
        try:
            # 确保text是字符串类型
            if not isinstance(text, str):
                text = str(text)

            # 如果文本为空，返回静默
            if not text.strip():
                yield (self.config.SAMPLE_RATE, np.zeros(1000))
                return

            # 获取文本块
            raw_text_chunks = Config.split_text(text)

            # 彻底清理text_chunks，确保每个元素都是有效字符串
            text_chunks = []
            for chunk in raw_text_chunks:
                # 确保chunk是字符串类型
                if isinstance(chunk, str):
                    cleaned_chunk = chunk.strip()
                    if cleaned_chunk:  # 不为空
                        text_chunks.append(cleaned_chunk)
                elif chunk is not None:  # 如果不是字符串但不为None，转换为字符串
                    str_chunk = str(chunk).strip()
                    if str_chunk:
                        text_chunks.append(str_chunk)

            # 如果没���有效的文本块，返回静默
            if not text_chunks:
                logger.warning("没有有效的文本块可以合成语音")
                yield (self.config.SAMPLE_RATE, np.zeros(1000))
                return

            logger.info(f"处理粤语文本块: {text_chunks}")

            # 再次验证所有文本块都是字符串
            def generate_valid_chunks():
                """生成器：产生有效的文本块"""
                for i, chunk in enumerate(text_chunks):
                    if isinstance(chunk, str) and chunk.strip():
                        yield chunk.strip()
                    else:
                        logger.warning(f"跳过无效文本块 {i}: {chunk} (类型: {type(chunk)})")

            # 创建生成器
            valid_chunks = generate_valid_chunks()

            # ���查是否有有效文本块（需要先转换为列表来检查）
            valid_chunks_list = list(valid_chunks)
            if not valid_chunks_list:
                logger.warning("所有文本块都无效")
                yield (self.config.SAMPLE_RATE, np.zeros(1000))
                return

            logger.info(f"最终有效文本块: {valid_chunks_list}")

            # 重新创建生成器用于语音合成
            def text_generator():
                """为语音合成创建文��生成器"""
                for chunk in valid_chunks_list:
                    yield chunk

            with torch.inference_mode():
                # 使用粤语语调进行语音合成
                for chunk in self.cosyvoice.inference_instruct2(
                    tts_text=text_generator(),
                    instruct_text='用粤语语调自然地说出这句话，要有粤语的韵味和语气',
                    prompt_speech_16k=self.prompt_speech,
                    stream=True
                ):
                    audio_data = chunk['tts_speech'].cpu().numpy()
                    processed_audio = self.audio_buffer.process(audio_data)

                    if processed_audio is not None:
                        yield (self.config.SAMPLE_RATE, processed_audio)

                    # 清理内存
                    del audio_data

        except Exception as e:
            logger.error(f"粤语语音合成出错: {e}")
            logger.error(f"输入文本: {text}")
            logger.error(f"文本类型: {type(text)}")
            import traceback
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            # 生成静默音频作为错误处理
            yield (self.config.SAMPLE_RATE, np.zeros(1000))

    def process_conversation(self, audio: Optional[np.ndarray]) -> Generator[Tuple[int, np.ndarray], None, None]:
        """处理完整对话流程"""
        with self.processing_lock:
            if self.is_processing:
                yield (self.config.SAMPLE_RATE, np.zeros(1000))
                return

            self.is_processing = True

        try:
            # 1. 语音转文字
            if audio is None:
                yield (self.config.SAMPLE_RATE, np.zeros(1000))
                return

            prompt_text = self.process_audio(audio)
            if not prompt_text or prompt_text == "没有录制到语音":
                yield (self.config.SAMPLE_RATE, np.zeros(1000))
                return

            # 2. 生成回应
            response_text = self.generate_response(prompt_text)

            # 3. 文本转语音流
            for audio_chunk in self.text_to_speech_stream(response_text):
                yield audio_chunk

        except Exception as e:
            logger.error(f"对话处理出错: {e}")
            yield (self.config.SAMPLE_RATE, np.zeros(1000))
        finally:
            self.is_processing = False

    def create_ui(self) -> None:
        """创建优化的Gradio界面"""
        with gr.Blocks(
            title="Jarvis AI Assistant",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 800px !important;
                margin: auto !important;
            }
            .audio-container {
                margin: 20px 0;
            }
            """
        ) as demo:

            gr.Markdown(
                """
                # 🤖 Jarvis AI Assistant (粤语版)
                
                **使用方法:**
                1. 点击"录音"按钮开始录音
                2. 用任何语言说出您的问题或指令
                3. 停止录音后，Jarvis会自动用**粤语**回应
                4. 回应将以粤语语音形式播放
                
                **核心特性:**
                - 🎤 多语言语音识别 (iic/SenseVoiceSmall)
                - 🧠 智能对话生成 (DeepSeek-Coder-V2-Lite)
                - 🔊 粤语语音合成 (CosyVoice2-0.5B)
                - 🌟 **无论您用什么语言提问，Jarvis都会用粤语回答**
                
                **支持输入**: 中文、英文、粤语等多种语言
                **输出语言**: 100% 粤语回答
                """,
                elem_id="header"
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🎤 您的语音输入")
                    input_audio = gr.Audio(
                        sources=["microphone"],
                        # type="numpy",
                        label="点击录音",
                        elem_classes=["audio-container"]
                    )

                with gr.Column():
                    gr.Markdown("### 🔊 Jarvis的回应")
                    output_audio = gr.Audio(
                        label="Jarvis回应",
                        autoplay=True,
                        streaming=True,
                        elem_classes=["audio-container"]
                    )

            with gr.Row():
                process_btn = gr.Button("🚀 开始对话", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ 清除", variant="secondary")

            # 状���指示器
            status = gr.Textbox(
                label="状态",
                value="准备就绪",
                interactive=False,
                elem_id="status"
            )

            # 事件处理
            process_btn.click(
                fn=self.process_conversation,
                inputs=[input_audio],
                outputs=[output_audio],
                show_progress="minimal"
            )

            clear_btn.click(
                fn=lambda: (None, None, "已清除"),
                outputs=[input_audio, output_audio, status]
            )

            # 自动处理
            input_audio.change(
                fn=self.process_conversation,
                inputs=[input_audio],
                outputs=[output_audio]
            )

        # 启动界面
        demo.launch(
            share=False,
            debug=True,
            server_port=7860,
            server_name="0.0.0.0",
            max_threads=8,  # M3可以支持更多线程
            inbrowser=True,
            show_error=True
        )

def main():
    """主函数"""
    try:
        logger.info("启动 Jarvis AI Assistant...")
        app = JarvisApp()
        app.create_ui()
    except Exception as e:
        logger.error(f"启动失败: {e}")
        raise

if __name__ == "__main__":
    main()
