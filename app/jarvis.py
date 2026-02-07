import threading
import time
import os
from queue import Queue
from app.utils.simple_logger import SimpleLogger
from listen import RealTimeRecognizer
from playback_qwen3_tts_flash import SynthesizeSpeechFromLlm

cond = threading.Condition()
turn = 'A'
queue = Queue()
logger = SimpleLogger.get_logger("jarvis")


def init_dashscope_api_key():
    if 'DASHSCOPE_API_KEY' in os.environ:
        import dashscope
        dashscope.api_key = os.environ['DASHSCOPE_API_KEY']


def worker_listen():
    global turn
    count_a = 0
    while True:
        with cond:
            while turn != 'A':
                cond.wait()
            count_a += 1
            logger.info(f'[Listen]: Round-{count_a}')
            recognizer = RealTimeRecognizer(queue)
            recognizer.run()
            time.sleep(0.3)
            turn = 'B'
            cond.notify()


def worker_brain():
    global turn
    count_b = 0
    while True:
        with cond:
            while turn != 'B':
                cond.wait()
            count_b += 1
            logger.info(f'[Brain]: Round {count_b}')
            content_heard = queue.get()
            print(f'[Brain] 听到 [Listen] 的内容 → {content_heard}')
            synthesize_speech = SynthesizeSpeechFromLlm()
            synthesize_speech.run(content_heard)
            time.sleep(0.3)
            turn = 'A'
            cond.notify()


threading.Thread(target=worker_listen, daemon=True).start()
threading.Thread(target=worker_brain, daemon=True).start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('主线程收到中断信号，退出...')
