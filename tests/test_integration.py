import time
import pytest
from jarvis import JarvisApp

def test_end_to_end():
    app = JarvisApp()
    # 1. 录音输入
    audio = app.record_audio(duration=3)
    # 2. 语音识别
    text = app.voice_to_text(audio)
    assert isinstance(text, str) and len(text) > 0
    # 3. LLM生成
    reply = app.llm(text)
    assert reply and "咩" in reply or "啦" in reply  # 粤语特征词
    # 4. 语音合成
    speech = app.text_to_speech(reply)
    assert speech is not None and len(speech) > 1000
    # 5. 界面交互
    result = app.ui_play_audio(speech)
    assert result is True

def test_performance():
    app = JarvisApp()
    start = time.time()
    for _ in range(10):
        audio = app.record_audio(duration=1)
        text = app.voice_to_text(audio)
        reply = app.llm(text)
        speech = app.text_to_speech(reply)
    elapsed = time.time() - start
    assert elapsed < 100  # 10轮总响应时间小于100秒

# 其他测试可参考 tests/test_streaming_optimization.py、tests/test_cantonese_llm.py 等