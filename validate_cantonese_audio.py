#!/usr/bin/env python3
"""
粤语音频验证脚本
用于检测生成的音频是否为粤语，并分析音频质量
"""

import os
import sys
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import torchaudio
import soundfile as sf
from datetime import datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

class CantoneseAudioValidator:
    def __init__(self):
        self.sample_rate = 22050
        self.cantonese_indicators = {
            # 粤语音频特征指标（基于频谱分析）
            'tone_range': (80, 400),  # 粤语声调频率范围
            'formant_ratios': {
                'f1_mean': (300, 800),    # 第一共振峰
                'f2_mean': (1000, 2500),  # 第二共振峰
                'f3_mean': (2000, 3500)   # 第三共振峰
            }
        }

    def analyze_audio_spectrum(self, audio_path):
        """分析音频频谱特征"""
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)

            # 计算基本音频特征
            duration = len(audio) / sr
            rms_energy = np.sqrt(np.mean(audio**2))

            # 计算频谱
            frequencies, times, Sxx = spectrogram(audio, sr)

            # 分析频谱特征
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]

            return {
                'duration': duration,
                'rms_energy': rms_energy,
                'spectral_centroid_mean': np.mean(spectral_centroid),
                'spectral_rolloff_mean': np.mean(spectral_rolloff),
                'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
                'frequencies': frequencies,
                'times': times,
                'spectrogram': Sxx
            }
        except Exception as e:
            print(f"❌ 音频分析失败: {e}")
            return None

    def detect_language_features(self, audio_features):
        """检测语言特征，判断是否可能是粤语"""
        if not audio_features:
            return None

        # 粤语检测指标
        cantonese_score = 0
        max_score = 5

        # 1. 检查频谱质心（粤语通常有特定的频率分布）
        centroid = audio_features['spectral_centroid_mean']
        if 800 < centroid < 3000:  # 粤语语音的典型频率范围
            cantonese_score += 1
            print(f"✅ 频谱质心正常: {centroid:.1f} Hz")
        else:
            print(f"⚠️  频谱质心异常: {centroid:.1f} Hz (期望: 800-3000 Hz)")

        # 2. 检查音频时长（是否有合理的语音长度）
        duration = audio_features['duration']
        if 1.0 < duration < 60.0:  # 合理的语音时长
            cantonese_score += 1
            print(f"✅ 音频时长正常: {duration:.2f} 秒")
        else:
            print(f"⚠️  音频时长异常: {duration:.2f} 秒")

        # 3. 检查能量水平
        energy = audio_features['rms_energy']
        if 0.01 < energy < 0.5:  # 正常的语音能量范围
            cantonese_score += 1
            print(f"✅ 音频能量正常: {energy:.4f}")
        else:
            print(f"⚠️  音频能量异常: {energy:.4f}")

        # 4. 检查过零率（语音的动态特性）
        zcr = audio_features['zero_crossing_rate_mean']
        if 0.05 < zcr < 0.3:  # 语音的典型过零率
            cantonese_score += 1
            print(f"✅ 过零率正常: {zcr:.4f}")
        else:
            print(f"⚠️  过零率异常: {zcr:.4f}")

        # 5. 检查频谱滚降点
        rolloff = audio_features['spectral_rolloff_mean']
        if 2000 < rolloff < 8000:  # 语音的典型滚降点
            cantonese_score += 1
            print(f"✅ 频谱滚降正常: {rolloff:.1f} Hz")
        else:
            print(f"⚠️  频谱滚降异常: {rolloff:.1f} Hz")

        confidence = (cantonese_score / max_score) * 100
        is_likely_cantonese = confidence >= 60

        return {
            'is_likely_cantonese': is_likely_cantonese,
            'confidence': confidence,
            'score': cantonese_score,
            'max_score': max_score,
            'details': audio_features
        }

    def generate_report(self, audio_path, validation_result):
        """生成验证报告"""
        print("\n" + "="*60)
        print(f"🔍 粤语音频验证报告")
        print("="*60)
        print(f"📁 文件路径: {audio_path}")

        if not validation_result:
            print("❌ 验证失败：无法分析音频文件")
            return

        print(f"📊 验证结果: {'✅ 可能是粤语' if validation_result['is_likely_cantonese'] else '❌ 可能不是粤语'}")
        print(f"🎯 置信度: {validation_result['confidence']:.1f}%")
        print(f"📈 得分: {validation_result['score']}/{validation_result['max_score']}")

        details = validation_result['details']
        print(f"\n📋 详细信息:")
        print(f"   ⏱  时长: {details['duration']:.2f} 秒")
        print(f"   🔊 能量: {details['rms_energy']:.4f}")
        print(f"   📊 频谱质心: {details['spectral_centroid_mean']:.1f} Hz")
        print(f"   📈 频谱滚降: {details['spectral_rolloff_mean']:.1f} Hz")
        print(f"   🌊 过零率: {details['zero_crossing_rate_mean']:.4f}")

        # 生成建议
        print(f"\n💡 建议:")
        if validation_result['confidence'] < 40:
            print("   ⚠️  音频可能不是语音或质量很差")
            print("   🔧 建议检查音频生成参数")
        elif validation_result['confidence'] < 60:
            print("   ⚠️  音频质量一般，可能需要优化")
            print("   🔧 建议检查提示音频和指令参数")
        else:
            print("   ✅ 音频质量良好，可能是有效的语音")

    def validate_audio_file(self, audio_path):
        """验证单个音频文件"""
        if not os.path.exists(audio_path):
            print(f"❌ 文件不存在: {audio_path}")
            return None

        print(f"🔍 开始验证音频文件: {os.path.basename(audio_path)}")

        # 分析音频
        features = self.analyze_audio_spectrum(audio_path)
        if not features:
            return None

        # 检测语言特征
        result = self.detect_language_features(features)

        # 生成报告
        self.generate_report(audio_path, result)

        return result

def main():
    """主函数 - 可以验证指定音频文件或最新生成的音频"""
    validator = CantoneseAudioValidator()

    # 检查命令行参数
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        validator.validate_audio_file(audio_path)
    else:
        # 查找最新的录音文件进行验证
        recordings_dir = os.path.join(ROOT_DIR, 'recordings')
        if os.path.exists(recordings_dir):
            audio_files = [f for f in os.listdir(recordings_dir) if f.endswith(('.wav', '.mp3'))]
            if audio_files:
                # 按修改时间排序，获取最新的文件
                audio_files.sort(key=lambda x: os.path.getmtime(os.path.join(recordings_dir, x)), reverse=True)
                latest_file = os.path.join(recordings_dir, audio_files[0])
                print(f"🎵 验证最新录音文件: {audio_files[0]}")
                validator.validate_audio_file(latest_file)
            else:
                print("❌ recordings 目录中没有找到音频文件")
        else:
            print("❌ 找不到 recordings 目录")
            print("💡 用法: python validate_cantonese_audio.py [音频文件路径]")

if __name__ == "__main__":
    main()
