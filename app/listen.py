# For prerequisites running following sample, visit https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen
import signal
import sys
import threading
from queue import Queue
from typing import Optional

import pyaudio
from dashscope.audio.asr import *

sample_rate = 16000
format_pcm = 'pcm'


class RealTimeRecognizer:
    def __init__(self, text_queue: Optional[Queue] = None):
        self.recognition = None
        self.text_queue = text_queue
        self._should_stop = False

    def signal_handler(self, sig, frame):
        print('Ctrl+C pressed, stop translation ...')
        self.recognition.stop()
        print('Translation stopped.')
        print(
            'RequestId: {}, first package delay ms: {}, last package delay ms: {}'
            .format(
                self.recognition.get_last_request_id(),
                self.recognition.get_first_package_delay(),
                self.recognition.get_last_package_delay(),
            ))
        sys.exit(0)

    def run(self):
        print("===================== Real-Time Speech Recognition =====================")
        print('Initializing ...')

        class Callback(RecognitionCallback):
            def __init__(self, outer):
                self.outer = outer

            def on_open(self) -> None:
                global mic, stream
                print('Listening open.')
                mic = pyaudio.PyAudio()
                stream = mic.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True)

            def on_close(self) -> None:
                global mic, stream
                print('Listening closed.')
                stream.stop_stream()
                stream.close()
                mic.terminate()
                stream = None
                mic = None

            def on_complete(self) -> None:
                print('Listening completed.')

            def on_error(self, message) -> None:
                print('Listening task_id: ', message.request_id)
                print('Listening error: ', message.message)
                if 'stream' in globals() and stream.active:
                    stream.stop()
                    stream.close()
                sys.exit(1)

            def on_event(self, result: RecognitionResult) -> None:
                sentence = result.get_sentence()
                if 'text' in sentence and RecognitionResult.is_sentence_end(sentence):
                    print(f'The content heard: {sentence["text"]}')
                    print(f'Listening request_id:{result.get_request_id()}, usage:{result.get_usage(sentence)}')
                    if self.outer.text_queue:
                        self.outer.text_queue.put(sentence['text'])
                    self.outer._should_stop = True

        callback = Callback(self)

        self.recognition = Recognition(
            model='fun-asr-realtime',
            format=format_pcm,
            sample_rate=sample_rate,
            semantic_punctuation_enabled=False,
            callback=callback)

        self.recognition.start()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.signal_handler)
        print("Press 'Ctrl+C' to stop recording and translation...")

        while True:
            if self._should_stop:
                break
            if not stream:
                break
            if hasattr(stream, 'is_active') and not stream.is_active():
                break
            try:
                data = stream.read(3200, exception_on_overflow=False)
            except Exception:
                break
            if self._should_stop:
                break
            self.recognition.send_audio_frame(data)

        if self.recognition:
            self.recognition.stop()
