import sys, os
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# set environment variable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(ROOT_DIR)
sys.path.append(f'{ROOT_DIR}/third_party/AcademiCodec')
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')
cosyvoice = CosyVoice2(f'{ROOT_DIR}/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

class Speak:
    def __init__(self):
        pass

    @staticmethod
    def text_to_voice(text):
        prompt_speech_16k = load_wav(f'{ROOT_DIR}/asset/zero_shot_prompt.wav', 16000)
        # instruct usage
        voice_file_list = []
        for i, j in enumerate(cosyvoice.inference_instruct2(
                tts_text=text, instruct_text='用粤语说这句话', prompt_speech_16k=prompt_speech_16k, stream=False)):
            print(f'{i}:{j}')
            voice_file = f'{ROOT_DIR}/tests/jarvis_{i}.wav'
            voice_file_list.append(voice_file)
            torchaudio.save(voice_file, j['tts_speech'], cosyvoice.sample_rate)
        print("Success to generate voice.")
        return voice_file_list[0]
