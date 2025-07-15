import os
import pyaudio
import wave
from datetime import datetime
import soundfile as sf


class Listen:
    def __init__(self):
        pass

    @staticmethod
    def record():
        # 定义一些参数
        CHUNK = 1024  # 每个缓冲区的帧数
        FORMAT = pyaudio.paInt16  # 音频格式（16位PCM）
        CHANNELS = 1  # 单声道
        RATE = 16000  # 采样率（Hz）
        RECORD_SECONDS = 5  # 录制时长（秒）
        WAVE_OUTPUT_FILENAME = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp_record/output.wav")  # 输出文件名

        # 初始化PyAudio
        p = pyaudio.PyAudio()

        # 打开麦克风流
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("正在录音...")

        # 录制音频
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("录音结束。")

        # 关闭流
        stream.stop_stream()
        stream.close()
        p.terminate()

        # 保存为WAV文件
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        print(f"Audio saved as {WAVE_OUTPUT_FILENAME}.")
        return WAVE_OUTPUT_FILENAME

    @staticmethod
    def save_voice(audio):
        # Create recordings directory if it doesn't exist
        os.makedirs("recordings", exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/recording_{timestamp}.mp3"  # waw

        # Save the audio file
        sample_rate, audio_data = audio
        sf.write(filename, audio_data, sample_rate)

        print(f"Voice saved as {filename}")
        return filename # "./tests/yue.mp3"
