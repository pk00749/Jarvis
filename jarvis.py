"""Jarvis语音助手主程序模块."""

import gc
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from typing import Tuple

import gradio as gr
import numpy as np
import pyaudio
from dashscope import Generation
from dashscope.audio.tts_v2 import *
from snownlp import SnowNLP

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from influence.influence import Influence
from listen.listen import Listen

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f'Root path: {ROOT_DIR}')
sys.path.append(f'{ROOT_DIR}/third_party/AcademiCodec')
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

# 配置阿里云API
MODEL = "cosyvoice-v2"
VOICE = "longjiayi_v2"

# 简化模式配置：True使用API模式（LLM+TTS），False使用本地模式（LLM+TTS）
USE_API_MODE = True

# MacBook Air M3 优化: 预加载本地模型到GPU内存，使用fp16减少内存占用（仅在本地模式下）
cosyvoice = None
if not USE_API_MODE:
    cosyvoice = CosyVoice2(
        f'{ROOT_DIR}/pretrained_models/CosyVoice2-0.5B',
        load_jit=False,
        load_trt=False,
        fp16=True
    )

# MacBook Air M3 优化: 创建线程池用于并发处理
executor = ThreadPoolExecutor(max_workers=2)


class APICallbackWithPlayback(ResultCallback):
    """阿里云API回调类，支持实时音频播放."""

    def __init__(self):
        """初始化回调类."""
        self._audio_chunks = []
        self._player = None
        self._stream = None

    def on_open(self):
        """WebSocket连接打开时的回调."""
        print("websocket is open.")
        # 启用实时播放
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=22050, output=True
        )

    def on_complete(self):
        """语音合成完成时的回调."""
        print("speech synthesis task success and completed.")

    def on_error(self, message: str):
        """语音合成错误时的回调."""
        print(f"speech synthesis task failed, {message}")

    def on_close(self):
        """WebSocket连接关闭时的回调."""
        print("websocket is closed.")
        # 清理资源
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._player:
            self._player.terminate()

    def on_event(self, message):
        """接收到事件消息时的回调."""
        print(f"Receive speech synthesis message: {message}")

    def on_data(self, data: bytes) -> None:
        """接收到音频数据时的回调."""
        # 同时收集和播放音频数据
        self._audio_chunks.append(data)
        # 实时播放音频
        if self._stream:
            self._stream.write(data)

    def get_audio_chunks(self):
        """获取音频数据块."""
        return self._audio_chunks


def listener(audio):
    """语音识别函数."""
    # MacBook Air M3 优化: 添加内存清理
    gc.collect()
    try:
        if audio is None:
            return "No voice to be recorded."
        filename = Listen.save_voice(audio)
        return Influence.voice_to_text(filename)
    except Exception as e:
        return f"Fail to record voice: {e}"


def _nlp_generator(text):
    """NLP文本分句生成器."""
    print("Split answer text by NLP...")
    result = SnowNLP(text)
    for sen in result.sentences:
        print(sen)
        yield sen


def brain_streaming_api_with_realtime_playback(audio):
    """使用API模式的语音处理流程，支持实时播放."""
    gc.collect()

    user_voice_to_text = listener(audio)

    # 创建支持实时播放的语音合成器
    callback = APICallbackWithPlayback()
    synthesizer = SpeechSynthesizer(model=MODEL, voice=VOICE, format=AudioFormat.PCM_22050HZ_MONO_16BIT, callback=callback,)

    # 使用阿里云LLM API
    inf = Influence()
    messages = [{"role": "user", "content": inf._create_cantonese_prompt(user_voice_to_text)}]
    responses = Generation.call(model="qwen-turbo", messages=messages, result_format="message",
                                stream=True, incremental_output=True,
    )

    print("Using API mode with real-time playback (LLM + TTS)...")
    for response in responses:
        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0]["message"]["content"]
            print(content, end="")
            # 流式调用TTS API，音频会通过callback实时播放
            synthesizer.streaming_call(content)
        else:
            print(f"Request id:{response.request_id}, Status code:{response.status_code}, "
                  f"error code:{response.code}, error message:{response.message}"
            )

    # 完成流式合成
    synthesizer.streaming_complete()
    print('\nRequestId: ', synthesizer.get_last_request_id())

    # 同时也返回音频数据给Gradio（实际上会被实时播放覆盖）
    audio_chunks = callback.get_audio_chunks()
    for chunk in audio_chunks:
        audio_array = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_array = np.clip(audio_array, -1.0, 1.0)
        yield (22050, audio_array)


def brain_streaming_local(audio):
    """使用本地模式的语音处理流程（LLM本地 + TTS本地）."""
    # MacBook Air M3 优化: 智能内存管理
    gc.collect()

    user_voice_to_text = listener(audio)
    inf = Influence()

    print("Using local mode (LLM + TTS)...")
    # 使用本地LLM
    jarvis_answer_text = inf.llm(user_voice_to_text)
    text_generator = _nlp_generator(jarvis_answer_text)
    print("Using local TTS model for streaming...")
    prompt_speech_16k = load_wav(f'{ROOT_DIR}/asset/zero_shot_prompt.wav', 16000)

    # MacBook Air M3 优化: 预测性资源预加载
    chunk_count = 0

    # 使用本地TTS模型
    for i, j in enumerate(cosyvoice.inference_instruct2(
            tts_text=text_generator,
            instruct_text='用粤语说这句话',
            prompt_speech_16k=prompt_speech_16k,
            stream=True)):

        # MacBook Air M3 优化: 优化音频处理流水线
        audio_chunk = j['tts_speech'].cpu().numpy()
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        audio_chunk = np.nan_to_num(audio_chunk)  # Replace NaN/Inf with 0
        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()  # Make it 1D

        # MacBook Air M3 优化: 缓冲策略减少延迟
        chunk_count += 1
        if chunk_count % 3 == 0:  # 每3个chunk清理一次内存
            gc.collect()

        yield (24000, audio_chunk)


def brain_streaming(audio):
    """根据配置选择使用API模式还是本地模式."""
    if USE_API_MODE:
        return brain_streaming_api_with_realtime_playback(audio)
    else:
        return brain_streaming_local(audio)


def ui_launch():
    """启动Gradio界面."""
    def process_audio_with_mode(audio, mode):
        global USE_API_MODE, cosyvoice
        # 更新模式
        USE_API_MODE = (mode == "API模式")

        # 如果切换到本地模式且模型未加载，则加载模型
        if not USE_API_MODE and cosyvoice is None:
            print("Loading local CosyVoice model...")
            cosyvoice = CosyVoice2(f'{ROOT_DIR}/pretrained_models/CosyVoice2-0.5B',
                                 load_jit=False, load_trt=False, fp16=True)

        # 处理音频并返回结果
        if audio is None:
            return None

        # 收集所有音频chunks并合并
        audio_chunks = []
        sample_rate = 22050  # 默认采样率

        for chunk in brain_streaming(audio):
            if chunk:
                sample_rate, audio_data = chunk
                audio_chunks.append(audio_data)

        if audio_chunks:
            # 合并所有音频块
            combined_audio = np.concatenate(audio_chunks)
            return (sample_rate, combined_audio)
        else:
            return None

    # 创建带标签页的界面
    with gr.Blocks(title="Jarvis 语音助手 👾") as demo:
        gr.Markdown("# Jarvis 语音助手 👾")

        with gr.Tabs():
            # 主功能标签页
            with gr.TabItem("语音助手"):
                gr.Markdown("API模式：LLM API + TTS API | 本地模式：LLM 本地 + TTS 本地")

                with gr.Row():
                    audio_input = gr.Audio(
                        sources=["microphone"],
                        label="我的语音输入",
                        type="numpy"
                    )
                    mode_radio = gr.Radio(
                        choices=["API模式", "本地模式"],
                        value="API模式" if USE_API_MODE else "本地模式",
                        label="选择运行模式"
                    )

                audio_output = gr.Audio(label="Jarvis回复", autoplay=False)

                # 处理按钮
                process_btn = gr.Button("处理语音", variant="primary")
                process_btn.click(
                    fn=process_audio_with_mode,
                    inputs=[audio_input, mode_radio],
                    outputs=audio_output
                )

            # 唤醒词设置标签页
            with gr.TabItem("唤醒词设置"):
                # 直接使用迁移后的WakeWordUIComponents类
                wake_word_ui = WakeWordUIComponents()
                wake_word_ui.create_interface()

    demo.launch(inbrowser=True)


def get_wake_word_detector():
    """获取唤醒词检测器实例"""
    if not hasattr(get_wake_word_detector, '_instance'):
        try:
            from wake_word import WakeWordDetector, WakeWordConfig
            from wake_word.auto_conversation import AutoConversationHandler

            detector = WakeWordDetector(WakeWordConfig())
            auto_handler = AutoConversationHandler(brain_streaming)

            def handle_wake_detected(result):
                print(f"🎉 检测到唤醒词: {result.text} (权重: {result.weight})")
                auto_handler.handle_wake_up(result)

            detector.set_wake_detected_callback(handle_wake_detected)
            get_wake_word_detector._instance = detector
            print("✅ 唤醒词检测器初始化成功")
        except Exception as e:
            print(f"❌ 唤醒词检测器初始化失败: {e}")
            get_wake_word_detector._instance = None

    return getattr(get_wake_word_detector, '_instance', None)


class WakeWordUIComponents:
    """精简的唤醒词设置界面组件"""

    def __init__(self):
        self.detector = get_wake_word_detector()

    def create_interface(self):
        """创建精简的唤醒词设置界面"""
        with gr.Blocks(title="唤醒词设置") as interface:
            gr.Markdown("# 🎤 唤醒词设置")
            gr.Markdown("基于粤语「喂」的语音唤醒功能")

            with gr.Row():
                # 控制面板
                with gr.Column():
                    enabled_checkbox = gr.Checkbox(
                        label="启用唤醒词检测",
                        value=True,
                        info="说「喂」来唤醒Jarvis"
                    )

                    sensitivity_slider = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        value=1.2,
                        step=0.1,
                        label="敏感度",
                        info="数值越低越敏感"
                    )

                    with gr.Row():
                        start_btn = gr.Button("开始监听", variant="primary")
                        stop_btn = gr.Button("停止监听")

                # 状态显示
                with gr.Column():
                    status_text = gr.Textbox(
                        label="状态",
                        value="未启动",
                        interactive=False
                    )

                    recognition_text = gr.Textbox(
                        label="识别结果",
                        value="暂无",
                        interactive=False
                    )

                    weight_text = gr.Textbox(
                        label="权重���息",
                        value="权重: 0.0 | 触发: 否",
                        interactive=False
                    )

            # 事件绑定
            enabled_checkbox.change(
                fn=self._toggle_enabled,
                inputs=[enabled_checkbox],
                outputs=[status_text]
            )

            sensitivity_slider.change(
                fn=self._update_sensitivity,
                inputs=[sensitivity_slider],
                outputs=[status_text]
            )

            start_btn.click(
                fn=self._start_listening,
                outputs=[status_text]
            )

            stop_btn.click(
                fn=self._stop_listening,
                outputs=[status_text]
            )

            # 定期更新状态
            interface.load(
                fn=self._update_status,
                outputs=[status_text, recognition_text, weight_text],
                every=3.0
            )

    def _toggle_enabled(self, enabled: bool) -> str:
        """切换启用状态"""
        if not self.detector:
            return "❌ 检测器未初始化"

        if enabled:
            self.detector.enable()
            return "✅ 已启用"
        else:
            self.detector.disable()
            return "❌ 已禁用"

    def _update_sensitivity(self, sensitivity: float) -> str:
        """更新敏感度"""
        if self.detector:
            self.detector.update_sensitivity(sensitivity)
        return f"🎚️ 敏感度: {sensitivity:.1f}"

    def _start_listening(self) -> str:
        """开始监听"""
        if not self.detector:
            return "❌ 检测器未初始化"

        if self.detector.start_detection():
            return "🎤 正在监听中..."
        else:
            return "❌ 启动失败"

    def _stop_listening(self) -> str:
        """停止监听"""
        if self.detector:
            self.detector.stop_detection()
        return "⏹️ 已停止监听"

    def _update_status(self) -> Tuple[str, str, str]:
        """更新状态显示"""
        if not self.detector:
            return "❌ 检测器未初始化", "暂无", "权重: 0.0"

        status = self.detector.get_status()

        # 状态文本
        if status['is_running']:
            status_text = "🎤 监听中"
        else:
            status_text = "⭕ 空闲"

        # 识别结果
        recognition_text = status.get('current_recognition', '') or "暂无"

        # 权重信息
        history = self.detector.get_history()
        if history:
            latest = history[-1]
            weight_text = f"权重: {latest['weight']:.1f} | 触发: {'是' if latest['is_triggered'] else '否'}"
        else:
            weight_text = "权重: 0.0 | 触发: 否"

        return status_text, recognition_text, weight_text


if __name__ == "__main__":
    ui_launch()
