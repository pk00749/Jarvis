# For prerequisites running the following sample, visit https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen
import signal  # for keyboard events handling (press "Ctrl+C" to terminate recording and translation)
import sys
import threading
from queue import Queue
from typing import Optional

import pyaudio
from dashscope.audio.asr import *

mic = None
stream = None

# Set recording parameters
sample_rate = 16000  # sampling rate (Hz)
channels = 1  # mono channel
dtype = 'int16'  # data type
format_pcm = 'pcm'  # the format of the audio data
block_size = 3200  # number of frames per buffer

text = None


class RealTimeRecognizer:
    def __init__(self, text_queue: Optional[Queue] = None):
        self.recognition = None
        self.text_queue = text_queue
        self._should_stop = False

    def signal_handler(self, sig, frame):
        print('Ctrl+C pressed, stop translation ...')
        # Stop translation
        self.recognition.stop()
        print('Translation stopped.')
        print(
            '[Metric] requestId: {}, first package delay ms: {}, last package delay ms: {}'
            .format(
                self.recognition.get_last_request_id(),
                self.recognition.get_first_package_delay(),
                self.recognition.get_last_package_delay(),
            ))
        # Forcefully exit the program
        sys.exit(0)

    def run(self):
        print('Initializing ...')

        # Real-time speech recognition callback
        outer = self

        class Callback(RecognitionCallback):
            def on_open(self) -> None:
                global mic
                global stream
                print('RecognitionCallback open.')
                mic = pyaudio.PyAudio()
                stream = mic.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=16000,
                                  input=True)

            def on_close(self) -> None:
                global mic
                global stream
                print('RecognitionCallback close.')
                stream.stop_stream()
                stream.close()
                mic.terminate()
                stream = None
                mic = None

            def on_complete(self) -> None:
                print('RecognitionCallback completed.')  # translation completed

            def on_error(self, message) -> None:
                print('RecognitionCallback task_id: ', message.request_id)
                print('RecognitionCallback error: ', message.message)
                # Stop and close the audio stream if it is running
                if 'stream' in globals() and stream.active:
                    stream.stop()
                    stream.close()
                # Forcefully exit the program
                sys.exit(1)

            def on_event(self, result: RecognitionResult) -> None:
                sentence = result.get_sentence()
                if 'text' in sentence:
                    if RecognitionResult.is_sentence_end(sentence):
                        print('RecognitionCallback text: ', sentence['text'])
                        print(
                            f'RecognitionCallback sentence end, request_id:{result.get_request_id()}, usage:{result.get_usage(sentence)}')
                        if outer.text_queue:
                            outer.text_queue.put(sentence['text'])
                        outer._should_stop = True

        # Create the translation callback
        callback = Callback()

        # Call recognition service by async mode, you can customize the recognition parameters, like model, format,
        # sample_rate For more information, please refer to https://help.aliyun.com/document_detail/2712536.html
        self.recognition = Recognition(
            model='fun-asr-realtime',
            # 'paraformer-realtime-v1'、'paraformer-realtime-8k-v1'
            format=format_pcm,
            # 'pcm'、'wav'、'opus'、'speex'、'aac'、'amr', you can check the supported formats in the document
            sample_rate=sample_rate,
            # support 8000, 16000
            semantic_punctuation_enabled=False,
            callback=callback)

        # Start translation
        self.recognition.start()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.signal_handler)
        print("Press 'Ctrl+C' to stop recording and translation...")
        # Create a keyboard listener until "Ctrl+C" is pressed

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


# main function
# if __name__ == '__main__':
#     recognizer = RealTimeRecognizer()
#     while True:
#         recognizer.run()
