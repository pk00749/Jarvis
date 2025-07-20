#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from listen.listen import Listen
from influence.influence import Influence

def check_prompt_audio(audio_path):
    """检查提示音频的实际内容"""
    print(f"🔍 正在检查提示音频: {audio_path}")

    try:
        # 使用语音识别来检查音频内容
        from cosyvoice.utils.file_utils import load_wav

        # 加载音频
        audio_data = load_wav(audio_path, 16000)
        print(f"✅ 音频加载成功，长度: {len(audio_data)/16000:.2f} 秒")

        # 转换为临时文件格式用于识别
        import tempfile
        import scipy.io.wavfile as wavfile

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            wavfile.write(tmp_file.name, 16000, audio_data.numpy())

            # 使用语音识别
            recognized_text = Influence.voice_to_text(tmp_file.name)
            print(f"📝 识别到的文本: {recognized_text}")

            # 清理临时文件
            os.unlink(tmp_file.name)

            return recognized_text

    except Exception as e:
        print(f"❌ 检查音频失败: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = "asset/cross_lingual_prompt.wav"

    print("=" * 60)
    print("🎵 提示音频内容检查器")
    print("=" * 60)

    result = check_prompt_audio(audio_path)

    if result:
        print(f"\n📊 检查结果:")
        print(f"   🎵 音频文件: {audio_path}")
        print(f"   📝 实际内容: {result}")
        print(f"\n💡 建议:")
        print(f"   - 如果这不是粤语，需要替换为粤语提示音频")
        print(f"   - 确保提示文本与音频内容完全匹配")
    else:
        print("\n❌ 无法识别音频内容")
