import gradio as gr
import sys, os
import numpy as np
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/third_party/AcademiCodec')
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

prompt_speech_16k = load_wav('./asset/zero_shot_prompt.wav', 16000)

def stream_tts(text):
    text_gen = (char for char in text)  # 或你的分句生成器
    for i, j in enumerate(cosyvoice.inference_instruct2(
            tts_text=text_gen, instruct_text='用粤语说这句话', prompt_speech_16k=prompt_speech_16k, stream=True)):
        # j['tts_speech'] 是 torch.Tensor, 需要转 numpy
        audio_chunk = j['tts_speech'].cpu().numpy()
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        if audio_chunk.ndim == 1:
            pass  # (N,)
        elif audio_chunk.ndim == 2 and audio_chunk.shape[1] in [1, 2]:
            pass  # (N, 1) or (N, 2)
        elif audio_chunk.ndim == 2 and audio_chunk.shape[0] in [1, 2]:
            audio_chunk = audio_chunk.T  # 转置
        else:
            audio_chunk = audio_chunk.flatten()
        if audio_chunk.size == 0:
            continue  # 跳过空块
        if np.any(np.isnan(audio_chunk)) or np.any(np.isinf(audio_chunk)):
            continue  # 跳过异常块


demo = gr.Interface(
    fn=stream_tts,
    inputs=gr.Textbox(label="输入文本"),
    outputs=gr.Audio(label="流式语音", streaming=True),
    title="流式TTS Demo"
)
demo.launch()
