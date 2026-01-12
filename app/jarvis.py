import threading
import time
import dashscope
import os
from queue import Queue
from playback import synthesize_speech_from_llm_by_streaming_mode
from listen import RealTimeRecognizer
from playback_qwen3_tts_flash import SynthesizeSpeechFromLlm
lock = threading.Lock()
cond = threading.Condition(lock)

turn = 'A'          # 谁该跑
queue = Queue()

def init_dashscope_api_key():
    if 'DASHSCOPE_API_KEY' in os.environ:
        dashscope.api_key = os.environ['DASHSCOPE_API_KEY']  # load API-key from environment variable DASHSCOPE_API_KEY

def worker_a():
    global turn, result
    i = 0
    while True:
        with cond:
            while turn != 'A':
                cond.wait()
            # ---- A 的临界区 ----
            i += 1
            a_result = f'A-result-{i}'      # 1. 算出结果
            print(f'[A] 产生结果 → {a_result}')
            recog = RealTimeRecognizer(queue)
            recog.run()
            # queue.put(a_result)
            time.sleep(0.3)

            turn = 'B'                      # 2. 把令牌给 B
            cond.notify()                   # 3. 唤醒 B（同时仍持锁，出 with 才放锁）

def worker_b():
    global turn, result
    i = 0
    while True:
        with cond:
            while turn != 'B':
                cond.wait()
            # ---- B 的临界区 ----
            i += 1
            a_result = queue.get()          # 1. 取 A 的结果
            print(f'[B] 收到 A 的结果 → {a_result}')  # 读结果
            # synthesize_speech_from_llm_by_streaming_mode(a_result)
            synthesize_speech = SynthesizeSpeechFromLlm()
            llm_result = synthesize_speech.llm(a_result)
            synthesize_speech.run(llm_result)
            b_result = f'B-result-{i}'               # B 也可以把新结果写回去
            print(f'[B] 产生结果 → {b_result}')
            time.sleep(0.3)

            turn = 'A'                      # 把令牌还给 A
            cond.notify()

threading.Thread(target=worker_a, daemon=True).start()
threading.Thread(target=worker_b, daemon=True).start()

# 让主线程保持运行，直到外部终止（Ctrl+C）
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print('主线程收到中断信号，退出...')
