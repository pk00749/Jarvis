import threading
import time
import dashscope
import os
from queue import Queue
from app.utils.simple_logger import SimpleLogger
from listen import RealTimeRecognizer
from playback_qwen3_tts_flash import SynthesizeSpeechFromLlm
lock = threading.Lock()
cond = threading.Condition(lock)

turn = 'A'          # 谁该跑
queue = Queue()

logger = SimpleLogger.get_logger("jarvis")

def init_dashscope_api_key():
    if 'DASHSCOPE_API_KEY' in os.environ:
        dashscope.api_key = os.environ['DASHSCOPE_API_KEY']  # load API-key from environment variable DASHSCOPE_API_KEY

def worker_listen():
    global turn, result
    count_a = 0
    while True:
        with cond:
            while turn != 'A':
                cond.wait()
            # ---- Listen 的临界区 ----
            count_a += 1
            worker_a_tag = f'[Listen]: Round-{count_a}'      # 1. 算出结果
            logger.info(worker_a_tag)
            recognizer = RealTimeRecognizer(queue)
            recognizer.run()
            time.sleep(0.3)

            turn = 'B'                      # 2. 把令牌给 B
            cond.notify()                   # 3. 唤醒 B（同时仍持锁，出 with 才放锁）

def worker_brain():
    global turn, result
    count_b = 0
    while True:
        with cond:
            while turn != 'B':
                cond.wait()
            # ---- Brain 的临界区 ----
            count_b += 1
            worker_b_tag = f'[Brain]: Round {count_b}'               # B 也可以把新结果写回去
            print(worker_b_tag)
            content_heard = queue.get()          # 1. 取 A 的结果
            print(f'[Brain] 听到 [Listen] 的内容 → {content_heard}')  # 读结果
            # synthesize_speech_from_llm_by_streaming_mode(a_result)
            synthesize_speech = SynthesizeSpeechFromLlm()
            synthesize_speech.run(content_heard)

            time.sleep(0.3)

            turn = 'A'                      # 把令牌还给 A
            cond.notify()

threading.Thread(target=worker_listen, daemon=True).start()
threading.Thread(target=worker_brain, daemon=True).start()

# 让主线程保持运行，直到外部终止（Ctrl+C）
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('主线程收到中断信号，退出...')
