import gradio as gr
import numpy as np
import soundfile as sf

def stream_wav(dummy):
    file = "record_output.wav"  # 替换为你的音频文件
    sr = 16000
    block_size = 16000  # 1秒
    with sf.SoundFile(file) as f:
        while True:
            data = f.read(block_size, dtype='float32')
            if len(data) == 0:
                break
            # 保证shape为(N,)
            if data.ndim > 1:
                data = data[:, 0]
            yield (sr, data)

demo = gr.Interface(
    fn=stream_wav,
    inputs=gr.Button("流式播放WAV"),
    outputs=gr.Audio(label="流式音频", streaming=True, autoplay=True),
    title="Gradio流式播放本地WAV"
)

if __name__ == "__main__":
    demo.launch()