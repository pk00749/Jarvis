import gradio as gr
import numpy as np
import time


# 模拟一个函数，将文本转换为音频
def text_to_audio(text):
    # 这里可以使用 TTS（Text-to-Speech）模型将文本转换为音频
    # 为了简化示例，我们生成一个简单的正弦波音频
    sample_rate = 44100  # 采样率
    duration = 3  # 持续时间（秒）
    frequency = 440  # 音频频率（Hz）

    # 生成正弦波音频
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = np.sin(2 * np.pi * frequency * t)

    # 返回音频数据和采样率
    return (sample_rate, audio_data.astype(np.float32))


# 创建 Text 控件
text_input = gr.Textbox(label="输入文本")

# 创建 Audio 控件
audio_output = gr.Audio(label="音频输出", autoplay=True)


# 使用 change 方法监听 Text 控件的变化
# def update_audio(text):
#     if text:
#         audio_data = text_to_audio(text)
#         return audio_data
#     return None

with gr.Blocks() as ui:
    text_input.change(
        fn=text_to_audio,  # 要执行的函数
        inputs=[text_input],  # 输入组件
        outputs=[audio_output]  # 输出组件
    )

    # 创建 Gradio 接口
    gr.Interface(
        fn=lambda x: x,  # 这里可以是一个空函数，因为我们主要使用 change 事件
        inputs=[text_input],
        outputs=[audio_output]
    )

# 启动应用
ui.launch()
