import gradio as gr
import os, sys
import numpy as np
from influence.influence import Influence
from listen.listen import Listen
from speak.speak import Speak
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from snownlp import SnowNLP

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f'Root path: {ROOT_DIR}')
sys.path.append(f'{ROOT_DIR}/third_party/AcademiCodec')
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')
cosyvoice = CosyVoice2(f'{ROOT_DIR}/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

def listener(audio):
    try:
        if audio is None:
            return "No voice to be recorded."
        filename = Listen.save_voice(audio)
        return Influence.voice_to_text(filename)
    except Exception as e:
        return f"Fail to record voice: {e}"

def _nlp_generator(text):
    print("Split answer text by NLP...")
    result = SnowNLP(text)
    print(result.sentences)
    return result.sentences

def _list_to_generator(texts):
    print("Convert List to Generator...")
    texts = ["你好",
             "在这两天的美股市场中",
             "最亮眼的无疑就是英伟达和CEO黄仁勋",
             "随着英伟达市值周四突破4万亿美元大关、黄仁勋继续按计划减持公司股票"]
    if len(texts) > 0:
        for text in texts:
            print(text)
            yield text

def brain_streaming(audio):
    prompt_text = listener(audio)
    answer_text = Influence.llm(prompt_text)
    answer_text_list = _nlp_generator(answer_text)
    text_generator = _list_to_generator(answer_text_list)
    print("Streaming...")
    prompt_speech_16k = load_wav(f'{ROOT_DIR}/asset/zero_shot_prompt.wav', 16000)
    # instruct usage
    for i, j in enumerate(cosyvoice.inference_instruct2(
            tts_text=text_generator, instruct_text='用粤语说这句话', prompt_speech_16k=prompt_speech_16k,
            stream=True)):
        audio_chunk = j['tts_speech'].cpu().numpy()
        audio_chunk = np.asarray(audio_chunk, dtype=np.float32)
        audio_chunk = np.nan_to_num(audio_chunk)  # Replace NaN/Inf with 0
        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
        # print("audio_chunk min:", np.min(audio_chunk))
        # print("audio_chunk max:", np.max(audio_chunk))
        # print("audio_chunk shape:", audio_chunk.shape)
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.flatten()  # Make it 1D
        yield (24000, audio_chunk)

def ui_launch():
    with gr.Blocks() as ui:
        # output_me = gr.Textbox(label="You")
        # output_jarvis_text = gr.Textbox(label="Jarvis")
        output_jarvis_audio = gr.Audio(sources=["microphone"], autoplay=True)

        # output_jarvis_text.change(fn=Speak.text_to_voice_stream,
        #                           inputs=[output_jarvis_text],
        #                           outputs=[output_jarvis_audio])
        gr.Interface(
            fn=brain_streaming,
            inputs=gr.Audio(sources=["microphone"]),
            outputs=[output_jarvis_audio],
            title="Jarvis👾",
            description=""
        )


    ui.launch()


if __name__ == "__main__":
    ui_launch()
