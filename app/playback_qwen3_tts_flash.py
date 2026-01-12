import time
import threading
import dashscope
import pyaudio
from dashscope.audio.qwen_tts_realtime import *

from dashscope import Generation
from B64PCMPlayer import B64PCMPlayer

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

p = pyaudio.PyAudio()
# 创建音频流
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True)

voice = 'Kiki'

pya = None
b64_player: B64PCMPlayer = None
qwen_tts_realtime: QwenTtsRealtime = None
DO_VIDEO_TEST = False


class MyCallback(QwenTtsRealtimeCallback):
    def __init__(self):
        super().__init__()
        self.finish_event = threading.Event()

    def on_open(self) -> None:
        global pya
        global b64_player
        print('connection opened, init player')
        pya = pyaudio.PyAudio()
        b64_player = B64PCMPlayer(pya, save_file=True)

    def on_close(self, close_status_code, close_msg) -> None:
        print('connection closed with code: {}, msg: {}, destroy player'.format(close_status_code, close_msg))
        global pya
        global b64_player
        b64_player.wait_for_complete()
        b64_player.shutdown()
        if pya:
            pya.terminate()
            pya = None

    def on_event(self, response: str) -> None:
        try:
            global qwen_tts_realtime
            global b64_player
            response_type = response['type']
            if 'session.created' == response_type:
                print('start session: {}'.format(response['session']['id']))
            if 'response.audio.delta' == response_type:
                recv_audio_b64 = response['delta']
                b64_player.add_data(recv_audio_b64)
            if 'response.done' == response_type:
                print(f'response {qwen_tts_realtime.get_last_response_id()} done')
            if 'session.finished' == response_type:
                print('session finished')
                self.finish_event.set()
        except Exception as e:
            print('[Error] {}'.format(e))
            self.finish_event.set()
            return

    def wait_for_complete(self):
        self.finish_event.wait()


class SynthesizeSpeechFromLlm:
    def llm(self, query_text: str):
        system_text = '你是一个闲聊型语音AI助手，来自广东广州，日常说粤语，主要任务是和用户展开日常性的友善聊天，用粤语回复。请不要回复使用任何格式化文本，回复要求口语化，不要使用markdown格式或者列表。'
        messages = [{
            'role': 'system',
            'content': system_text
        }, {
            'role': 'user',
            'content': query_text
        }]
        print('>>> query: ' + query_text)
        response = Generation.call(
            model='qwen-plus',
            messages=messages,
            result_format='message',  # set result format as 'message'
            stream=False,  # enable stream output
            incremental_output=True,  # enable incremental output
        )
        print('>>> answer: ', end='')
        print(response.output.choices[0]['message']['content'])
        return response.output.choices[0]['message']['content']

    @staticmethod
    def run(text: str):
        text_to_synthesize = str.split(text, ",")

        callback = MyCallback()
        qwen_tts_realtime = QwenTtsRealtime(
            model='qwen3-tts-flash-realtime',
            callback=callback,
        )

        qwen_tts_realtime.connect()
        qwen_tts_realtime.update_session(
            voice=voice,
            response_format=AudioFormat.PCM_24000HZ_MONO_16BIT,
            mode='server_commit'
        )
        for text_chunk in text_to_synthesize:
            print(f'send texd: {text_chunk}')
            qwen_tts_realtime.append_text(text_chunk)
            time.sleep(0.1)
        qwen_tts_realtime.finish()
        callback.wait_for_complete()
        qwen_tts_realtime.close()
        print('[Metric] session: {}, first audio delay: {}'.format(
            qwen_tts_realtime.get_session_id(),
            qwen_tts_realtime.get_first_audio_delay(),
        ))


if __name__ == '__main__':
    synthesizer = SynthesizeSpeechFromLlm()
    llm = synthesizer.run