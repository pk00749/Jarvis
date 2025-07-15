import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import time
import os

def audio_callback(outdata, frames, time_info, status):
    """回调函数，用于流式播放"""
    if status:
        print(f"Stream callback error: {status}")
    
    # 从音频数据中读取指定帧数
    if not hasattr(audio_callback, 'position'):
        audio_callback.position = 0
    
    if audio_callback.position >= len(audio_callback.audio_data):
        raise sd.CallbackStop()
    
    # 计算还需要多少帧
    remaining = len(audio_callback.audio_data) - audio_callback.position
    valid_frames = min(frames, remaining)
    
    if valid_frames < frames:
        # 如果剩余的帧数不够，用零填充
        outdata[:] = 0
        outdata[:valid_frames] = audio_callback.audio_data[audio_callback.position:audio_callback.position + valid_frames]
    else:
        # 正常复制数据
        outdata[:] = audio_callback.audio_data[audio_callback.position:audio_callback.position + frames]
    
    audio_callback.position += valid_frames

def play_audio_file(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件不存在: {file_path}")
    
    print(f"正在读取音频文件: {file_path}")
    # 读取音频文件
    try:
        sample_rate, audio_data = wavfile.read(file_path)
    except Exception as e:
        raise Exception(f"读取音频文件失败: {str(e)}")
    
    print(f"音频采样率: {sample_rate}Hz")
    print(f"音频数据形状: {audio_data.shape}")
    
    # 确保数据是浮点型的并且在 [-1, 1] 范围内
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    
    # 如果是单声道，转换为二维数组
    if len(audio_data.shape) == 1:
        audio_data = audio_data.reshape(-1, 1)
    
    # 检查数据是否为空
    if len(audio_data) == 0:
        raise ValueError("音频数据为空")
    
    # 存储音频数据供回调函数使用
    audio_callback.audio_data = audio_data
    
    # 设置音频流
    try:
        stream = sd.OutputStream(
            channels=audio_data.shape[1],
            samplerate=sample_rate,
            callback=audio_callback,
            finished_callback=lambda: print("播放完成")
        )
    except Exception as e:
        raise Exception(f"创建音频流失败: {str(e)}")
    
    print("开始播放音频文件...")
    try:
        with stream:
            while stream.active:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n播放被用户中断")
    except Exception as e:
        print(f"播放过程中出错: {str(e)}")
    finally:
        print("播放结束")

if __name__ == "__main__":
    try:
        # 替换为你的音频文件路径
        audio_file = "./instruct_0.wav"
        play_audio_file(audio_file)
    except Exception as e:
        print(f"错误: {str(e)}")
