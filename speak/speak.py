import sys, os
import torchaudio
from typing import Generator
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from snownlp import SnowNLP


# set environment variable
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(f'{ROOT_DIR}/third_party/AcademiCodec')
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')
cosyvoice = CosyVoice2(f'{ROOT_DIR}/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)

class Speak:
    def __init__(self):
        pass

    @staticmethod
    def _string_to_generator(text):
        length = 12
        if length <= 0:
            print("Length should be greater than zero")
        else:
           return (char for char in text[:length])

    @staticmethod
    def nlp_generator(text):
        result = SnowNLP(text)
        print(result.sentences)
        return result.sentences

    # def text_to_voice_file(self, text):
    #     text_generator = self._string_to_generator(text)
    #     print("Save as file.")
    #     prompt_speech_16k = load_wav(f'{ROOT_DIR}/asset/zero_shot_prompt.wav', 16000)
    #     # instruct usage
    #     voice_file_list = []
    #
    #     if isinstance(text_generator, Generator):
    #         for i, j in enumerate(cosyvoice.inference_instruct2(
    #                 tts_text=text_generator, instruct_text='用粤语说这句话', prompt_speech_16k=prompt_speech_16k, stream=True)):
    #             print(f'{i}:{j}')
    #             voice_file = f'{ROOT_DIR}/tests/jarvis_{i}.wav'
    #             voice_file_list.append(voice_file)
    #             torchaudio.save(voice_file, j['tts_speech'], cosyvoice.sample_rate)
    #         print("Success to generate voice.")
    #         return voice_file_list[0]
