import threading
import time
import os
from queue import Queue
from app.utils.simple_logger import SimpleLogger
from listen import RealTimeRecognizer
from playback_qwen3_tts_flash import SynthesizeSpeechFromLlm


class Jarvis:
    def __init__(self):
        self.cond = threading.Condition()
        self.turn = 'A'
        self.queue = Queue()
        self.logger = SimpleLogger.get_logger("jarvis")

    def init_dashscope_api_key(self):
        if 'DASHSCOPE_API_KEY' in os.environ:
            import dashscope
            dashscope.api_key = os.environ['DASHSCOPE_API_KEY']

    def worker_listen(self):
        count_a = 0
        while True:
            with self.cond:
                while self.turn != 'A':
                    self.cond.wait()
                count_a += 1
                self.logger.info(f'[Listen]: Round-{count_a}')
                recognizer = RealTimeRecognizer(self.queue)
                recognizer.run()
                time.sleep(0.3)
                self.turn = 'B'
                self.cond.notify()

    def worker_brain(self):
        count_b = 0
        while True:
            with self.cond:
                while self.turn != 'B':
                    self.cond.wait()
                count_b += 1
                self.logger.info(f'[Brain]: Round {count_b}')
                content_heard = self.queue.get()
                print(f'[Brain] 听到 [Listen] 的内容 → {content_heard}')
                synthesize_speech = SynthesizeSpeechFromLlm()
                synthesize_speech.run(content_heard)
                time.sleep(0.3)
                self.turn = 'A'
                self.cond.notify()

    def run(self):
        self.init_dashscope_api_key()
        threading.Thread(target=self.worker_listen, daemon=True).start()
        threading.Thread(target=self.worker_brain, daemon=True).start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print('主线程收到中断信号，退出...')


if __name__ == '__main__':
    jarvis = Jarvis()
    jarvis.run()
