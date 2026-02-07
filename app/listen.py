import signal
import sys
import threading
from queue import Queue
from typing import Optional

import pyaudio
from dashscope.audio.asr import *


class RealTimeRecognizer:
    def __init__(self, text_queue: Optional[Queue] = None):
        self.recognition = None
        self.text_queue = text_queue
        self._should_stop = False
        self.mic = None
        self.stream = None

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
                print('Listening open.')
                self.outer.mic = pyaudio.PyAudio()
                self.outer.stream = self.outer.mic.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True
                )

            def on_close(self) -> None:
                print('Listening closed.')
                if self.outer.stream:
                    self.outer.stream.stop_stream()
                    self.outer.stream.close()
                if self.outer.mic:
                    self.outer.mic.terminate()
                self.outer.stream = None
                self.outer.mic = None

            def on_complete(self) -> None:
                print('Listening completed.')

            def on_error(self, message) -> None:
                print('Listening task_id: ', message.request_id)
                print('Listening error: ', message.message)
                if self.outer.stream and hasattr(self.outer.stream, 'active') and self.outer.stream.active:
                    self.outer.stream.stop()
                    self.outer.stream.close()
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
            format='pcm',
            sample_rate=16000,
            semantic_punctuation_enabled=False,
            callback=callback)

        self.recognition.start()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.signal_handler)
        print("Press 'Ctrl+C' to stop recording and translation...")

        while True:
            if self._should_stop:
                break
            if not self.stream:
                break
            if hasattr(self.stream, 'is_active') and not self.stream.is_active():
                break
            try:
                data = self.stream.read(3200, exception_on_overflow=False)
            except Exception:
                break
            if self._should_stop:
                break
            self.recognition.send_audio_frame(data)

        if self.recognition:
            self.recognition.stop()
