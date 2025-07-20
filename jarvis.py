import gradio as gr
import os, sys
import numpy as np
from influence.influence import Influence
from listen.listen import Listen
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from snownlp import SnowNLP
import torch
import gc
from concurrent.futures import ThreadPoolExecutor
import time
import scipy.signal  # 信号处理库
from scipy.io import wavfile  # 添加WAV文件处理

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f'Root path: {ROOT_DIR}')
sys.path.append(f'{ROOT_DIR}/third_party/AcademiCodec')
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

# macOS M3性能优化：启用硬件加速和优化设置
def initialize_cosyvoice_optimized():
    """
    针对MacBook Air M3优化的CosyVoice2初始化
    """
    print("🚀 正在初始化CosyVoice2，启用Apple Silicon优化...")

    # 检测Apple Silicon并设置优化参数
    is_apple_silicon = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    if is_apple_silicon:
        print("✅ 检测到Apple Silicon，启用MPS加速")
        # 设置MPS为默认设备
        try:
            torch.backends.mps.empty_cache()
        except:
            pass  # 忽略MPS缓存清理错误

        # 针对Apple Silicon优化的参数
        cosyvoice = CosyVoice2(
            f'{ROOT_DIR}/pretrained_models/CosyVoice2-0.5B',
            load_jit=True,      # 启用JIT编译优化
            load_trt=False,     # TensorRT在Apple Silicon上不可用
            fp16=True           # 启用半精度浮点，减少内存使用
        )
    else:
        print("⚠️  未检测到Apple Silicon，使用默认设置")
        cosyvoice = CosyVoice2(
            f'{ROOT_DIR}/pretrained_models/CosyVoice2-0.5B',
            load_jit=False,
            load_trt=False,
            fp16=False
        )

    return cosyvoice

# 全局变量和缓存管理
cosyvoice = None
model_cache = {}
executor = ThreadPoolExecutor(max_workers=3)  # 利用多核处理能力

def get_cosyvoice():
    """获取CosyVoice2实例，使用单例模式避免重复初始化"""
    global cosyvoice
    if cosyvoice is None:
        cosyvoice = initialize_cosyvoice_optimized()
    return cosyvoice

def optimize_memory():
    """内存优化：清理不必要的缓存"""
    global model_cache

    # 强制垃圾回收
    gc.collect()

    # 清理MPS缓存（如果可用）- 修复API调用错误
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # 检查是否有empty_cache方法（不同PyTorch版本可能不同）
            if hasattr(torch.backends.mps, 'empty_cache'):
                torch.backends.mps.empty_cache()
            else:
                # 备选方案：使用torch.mps.empty_cache()（如果可用）
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
    except Exception as ex:
        # 忽略MPS缓存清理错误，不影响主要功能
        print(f"⚠️ MPS缓存清理跳过: {ex}")


def listener(audio):
    try:
        if audio is None:
            return "No voice to be recorded."
        filename = Listen.save_voice(audio)
        return Influence.voice_to_text(filename)
    except Exception as e:
        return f"Fail to record voice: {e}"

def _nlp_generator(text):
    print("Split answer text by NLP...")
    result = SnowNLP(text)
    print(result.sentences)
    for sen in result.sentences:
        print(sen)
        yield sen

def brain_streaming_optimized(audio):
    """优化的语音处理函数 - 使用指令模式生成粤语语音 - 流式返回音频块"""
    start_time = time.time()

    try:
        # 步骤1: 语音识别
        print("🎤 开始语音识别...")
        prompt_text = listener(audio)
        print(f"✅ 识别结果: {prompt_text}")

        # 步骤2: LLM生成回复
        print("🧠 开始LLM生成回复...")
        answer_text = Influence.llm(prompt_text)
        print(f"✅ LLM回复: {answer_text}")

        # 步骤3: 使用指令模式语音合成 - 生成粤语
        print("🎵 开始流式粤语语音合成...")
        cosyvoice_instance = get_cosyvoice()

        # 🔧 直接加载提示音频
        print("🎵 加载提示音频...")
        prompt_speech_16k = load_wav(f'{ROOT_DIR}/asset/zero_shot_prompt.wav', 16000)
        print("✅ 提示音频加载完成")

        # 🔧 关键：使用指令文本控制语音特性
        instruct_text = "请用粤语说话"  # 明确指令：使用粤语

        # 🔧 预先生成句子列表，然后创建生成器函数
        print("📝 开始分句处理...")
        sentences = list(_nlp_generator_optimized(answer_text))
        print(f"✅ 分句完成，共 {len(sentences)} 个句子:")
        for i, sentence in enumerate(sentences, 1):
            print(f"   句子{i}: 「{sentence}」")

        if not sentences:
            print("❌ 分句结果为空")
            return

        def create_text_generator():
            """创建文本生成器 - 确保只生成一次"""
            print("🎯 生成器开始工作...")
            for i, sentence in enumerate(sentences, 1):
                print(f"🎯 生成器产出句子 {i}: 「{sentence}」")
                yield sentence.strip()
            print("🎯 生成器工作完成")

        # 流式音频生成和播放
        audio_chunk_count = 0
        min_chunk_duration = 0.5  # 最小音频块时长（秒），避免播放过于频繁
        accumulated_chunk = []

        print("🎵 开始流式音频生成...")
        print(f"📋 指令文本: {instruct_text}")
        print("=" * 60)

        try:
            for i, output in enumerate(cosyvoice_instance.inference_instruct2(
                    create_text_generator(),
                    instruct_text,
                    prompt_speech_16k,
                    stream=True)):

                audio_chunk_count += 1
                chunk_start_time = time.time()

                print(f"🎵 正在处理第 {audio_chunk_count} 个音频块...")

                # 处理音频块
                if 'tts_speech' not in output:
                    print(f"⚠️ 音频块 {audio_chunk_count} 缺少 tts_speech")
                    continue

                audio_chunk = output['tts_speech'].cpu().numpy()

                # 确保数据类型为float32
                if audio_chunk.dtype != np.float32:
                    audio_chunk = audio_chunk.astype(np.float32)

                # 确保音频为1D数组
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.flatten()

                # 音频质量优化
                audio_chunk = np.clip(audio_chunk, -0.95, 0.95)

                if len(audio_chunk) > 0:
                    # 将音频块加入累积缓冲区
                    accumulated_chunk.append(audio_chunk)

                    # 检查累积的音频是否达到最小播放时长
                    total_samples = sum(len(chunk) for chunk in accumulated_chunk)
                    total_duration = total_samples / 22050

                    chunk_duration = len(audio_chunk) / 22050
                    chunk_time = time.time() - chunk_start_time

                    print(f"✅ 音频块 {audio_chunk_count} 已处理")
                    print(f"   - 当前块长度: {len(audio_chunk)} 样本 ({chunk_duration:.2f}秒)")
                    print(f"   - 累积时长: {total_duration:.2f}秒")
                    print(f"   - 处理用时: {chunk_time:.2f}秒")

                    # 🔧 关键：当累积音频达到最小播放时长时，yield返回音频块
                    if total_duration >= min_chunk_duration or audio_chunk_count % 3 == 0:
                        # 合并累积的音频块
                        if len(accumulated_chunk) == 1:
                            playback_audio = accumulated_chunk[0]
                        else:
                            playback_audio = np.concatenate(accumulated_chunk)

                        # 应用基本的音频优化
                        playback_audio = np.clip(playback_audio, -0.95, 0.95)
                        playback_audio = trim_silence(playback_audio, threshold=0.005)

                        # 轻量级音频增强（避免过度处理）
                        max_val = np.max(np.abs(playback_audio))
                        if max_val > 0:
                            playback_audio = playback_audio / max_val * 0.7

                        playback_duration = len(playback_audio) / 22050

                        print(f"🎵 返回音频块用于播放:")
                        print(f"   - 播放时长: {playback_duration:.2f}秒")
                        print(f"   - 包含 {len(accumulated_chunk)} 个原始音频块")

                        # 🔧 关键：yield返回音频块供界面流式播放
                        yield (22050, playback_audio)

                        # 清空累积缓冲区，为下一轮做准备
                        accumulated_chunk = []

                        # 添加短暂停顿，让界面有时间处理
                        time.sleep(0.1)

                else:
                    print(f"⚠️ 音频块 {audio_chunk_count} 为空，跳过")

                # 内存管理
                if audio_chunk_count % 5 == 0:
                    optimize_memory()

            # 处理剩余的累积音频（如果有的话）
            if accumulated_chunk:
                print(f"🔗 处理剩余的 {len(accumulated_chunk)} 个音频块...")

                if len(accumulated_chunk) == 1:
                    final_audio = accumulated_chunk[0]
                else:
                    final_audio = np.concatenate(accumulated_chunk)

                # 最终音频优化
                final_audio = np.clip(final_audio, -0.95, 0.95)
                final_audio = trim_silence(final_audio, threshold=0.005)

                # 轻量级增强
                max_val = np.max(np.abs(final_audio))
                if max_val > 0:
                    final_audio = final_audio / max_val * 0.7

                final_duration = len(final_audio) / 22050
                print(f"🎵 返回最终音频块: {final_duration:.2f}秒")

                yield (22050, final_audio)

            total_time = time.time() - start_time
            print("=" * 60)
            print(f"✅ 流式音频生成完成:")
            print(f"   - 总音频块数: {audio_chunk_count}")
            print(f"   - 总处理时间: {total_time:.2f}秒")
            print(f"   - 使用指令: {instruct_text}")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 流式音频生成失败: {e}")
            import traceback
            traceback.print_exc()
            return

    except Exception as e:
        print(f"❌ 流式音频处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

def _nlp_generator_optimized(text):
    """优化的NLP文本分句处理 - 针对中文优化"""
    print(f"📝 开始智能分句，原文: {text}")

    # 智能缓存
    cache_key = hash(text)
    if cache_key in model_cache:
        sentences = model_cache[cache_key]
        print(f"✅ 使用缓存的分句结果: {len(sentences)} 个句子")
    else:
        # 使用SnowNLP进行中文分句
        result = SnowNLP(text)
        sentences = []

        for sentence in result.sentences:
            sentence = sentence.strip()
            # 过滤掉过短的句子（少于2个字符）
            if len(sentence) >= 2:
                sentences.append(sentence)

        # 如果分句结果为空，使���原文
        if not sentences:
            sentences = [text.strip()]

        # 缓存结果
        if len(model_cache) > 50:  # 限制缓存大小
            oldest_keys = list(model_cache.keys())[:10]
            for key in oldest_keys:
                del model_cache[key]

        model_cache[cache_key] = sentences
        print(f"✅ 新分句结果: {len(sentences)} 个句子")

    # 输出分句结果
    for i, sentence in enumerate(sentences, 1):
        print(f"   句子{i}: {sentence}")
        yield sentence

def trim_silence(audio, threshold=0.01):
    """去除音频开头和结尾的静音部分"""
    # 找到开头第一个非静音样本
    start_idx = 0
    for i, sample in enumerate(audio):
        if abs(sample) > threshold:
            start_idx = i
            break

    # 找到结尾最后一个非静音样本
    end_idx = len(audio) - 1
    for i in range(len(audio) - 1, -1, -1):
        if abs(audio[i]) > threshold:
            end_idx = i + 1
            break

    return audio[start_idx:end_idx] if start_idx < end_idx else audio

def enhance_audio_clarity(audio, sample_rate=22050):
    """
    音频清晰度增强函��
    针对语音清晰度进行多重优化
    """
    print("🔧 开始音频清晰度增强...")

    try:
        # 1. 音频归一化 - 提升整体音量
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            # 使用较高的归一化目标值，提升音量
            normalized_audio = audio / max_val * 0.8  # 从0.95降到0.8避免过度增益
        else:
            normalized_audio = audio

        # 2. 动态范围压缩 - 让小声音更清楚，大声音不过载
        def soft_compress(x, threshold=0.3, ratio=4.0):
            """软压缩算法"""
            abs_x = np.abs(x)
            sign_x = np.sign(x)

            # 对超过阈值的部分进行压缩
            compressed = np.where(
                abs_x > threshold,
                threshold + (abs_x - threshold) / ratio,
                abs_x
            )

            return sign_x * compressed

        compressed_audio = soft_compress(normalized_audio)

        # 3. 高频增强 - 提升语音清晰度关键频率
        # 设计一个简单的高频增强滤波器
        nyquist = sample_rate / 2

        # 增强1-4kHz频段（语音清晰度关键频段）
        b, a = scipy.signal.butter(2, [1000/nyquist, 4000/nyquist], btype='band')
        high_freq = scipy.signal.filtfilt(b, a, compressed_audio)

        # 将增强的高频成分混合回原始音频
        enhanced_audio = compressed_audio + high_freq * 0.3

        # 4. 去噪处理 - 移除低频噪音
        # 高通滤波器，移除50Hz以下的噪音
        b_hp, a_hp = scipy.signal.butter(2, 50/nyquist, btype='high')
        denoised_audio = scipy.signal.filtfilt(b_hp, a_hp, enhanced_audio)

        # 5. 最终限幅和平滑
        final_audio = np.clip(denoised_audio, -0.9, 0.9)

        # 6. 音频增益调整 - 再次提升整体音量
        final_max = np.max(np.abs(final_audio))
        if final_max > 0:
            # 目标RMS值调整，让音频更响亮
            current_rms = np.sqrt(np.mean(final_audio**2))
            target_rms = 0.2  # 目标RMS值，提升音量
            gain = target_rms / current_rms if current_rms > 0 else 1.0
            final_audio = final_audio * min(gain, 3.0)  # 最大3倍增益

        # 最终限幅
        final_audio = np.clip(final_audio, -0.95, 0.95)

        print("✅ 音频清晰度增强完成")
        print(f"   - 原始RMS: {np.sqrt(np.mean(audio**2)):.4f}")
        print(f"   - 增强后RMS: {np.sqrt(np.mean(final_audio**2)):.4f}")
        print(f"   - 增益倍数: {np.sqrt(np.mean(final_audio**2))/np.sqrt(np.mean(audio**2)):.2f}x")

        return final_audio

    except Exception as e:
        print(f"❌ 音频增强失败: {e}")
        # 如果增强失败，返回原始音频但进行基本的音量提升
        return np.clip(audio * 2.0, -0.95, 0.95)

def brain_streaming_with_realtime_playback(audio):
    """真正的流式音频生成函数 - 逐个返回音频块供实时播放"""
    start_time = time.time()

    try:
        # 步骤1: 语音识别
        print("🎤 开始语音识别...")
        prompt_text = listener(audio)
        print(f"✅ 识别结果: {prompt_text}")

        # 步骤2: LLM生成回复
        print("🧠 开始LLM生成回复...")
        answer_text = Influence.llm(prompt_text)
        print(f"✅ LLM回复: {answer_text}")

        # 步骤3: 使用指令模式语音合成 - 生成粤语
        print("🎵 开始流式粤语语音合成...")
        cosyvoice_instance = get_cosyvoice()

        # 🔧 直接加载提示音频
        print("🎵 加载提示音频...")
        prompt_speech_16k = load_wav(f'{ROOT_DIR}/asset/zero_shot_prompt.wav', 16000)
        print("✅ 提示音频加载完成")

        # 🔧 关键：使用指令文本控制语音特性
        instruct_text = "请用粤语说话"  # 明确指令：使用粤语

        # 🔧 预先生成句子列表，然后创建生成器函数
        print("📝 开始分句处理...")
        sentences = list(_nlp_generator_optimized(answer_text))
        print(f"✅ 分句完成，共 {len(sentences)} 个句子:")
        for i, sentence in enumerate(sentences, 1):
            print(f"   句子{i}: 「{sentence}」")

        if not sentences:
            print("❌ 分句结果为空")
            return

        def create_text_generator():
            """创建文本生成器 - 确保只生成一次"""
            print("🎯 生成器开始工作...")
            for i, sentence in enumerate(sentences, 1):
                print(f"🎯 生成器产出句子 {i}: 「{sentence}」")
                yield sentence.strip()
            print("🎯 生成器工作完成")

        # 流式音频生成和播放
        audio_chunk_count = 0
        min_chunk_duration = 0.5  # 最小音频块时长（秒），避免播放过于频繁
        accumulated_chunk = []

        print("🎵 开始流式音频生成...")
        print(f"📋 指令文本: {instruct_text}")
        print("=" * 60)

        try:
            for i, output in enumerate(cosyvoice_instance.inference_instruct2(
                    create_text_generator(),
                    instruct_text,
                    prompt_speech_16k,
                    stream=True)):

                audio_chunk_count += 1
                chunk_start_time = time.time()

                print(f"🎵 正在处理第 {audio_chunk_count} 个音频块...")

                # 处理音频块
                if 'tts_speech' not in output:
                    print(f"⚠️ 音频块 {audio_chunk_count} 缺少 tts_speech")
                    continue

                audio_chunk = output['tts_speech'].cpu().numpy()

                # 确保数据类型为float32
                if audio_chunk.dtype != np.float32:
                    audio_chunk = audio_chunk.astype(np.float32)

                # 确保音频为1D数组
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.flatten()

                # 音频质量优化
                audio_chunk = np.clip(audio_chunk, -0.95, 0.95)

                if len(audio_chunk) > 0:
                    # 将音频块加入累积缓冲区
                    accumulated_chunk.append(audio_chunk)

                    # 检查累积的音频是否达到最小播放时长
                    total_samples = sum(len(chunk) for chunk in accumulated_chunk)
                    total_duration = total_samples / 22050

                    chunk_duration = len(audio_chunk) / 22050
                    chunk_time = time.time() - chunk_start_time

                    print(f"✅ 音频块 {audio_chunk_count} 已处理")
                    print(f"   - 当前块长度: {len(audio_chunk)} 样本 ({chunk_duration:.2f}秒)")
                    print(f"   - 累积时长: {total_duration:.2f}秒")
                    print(f"   - 处理用时: {chunk_time:.2f}秒")

                    # 🔧 关键：当累积音频达到最小播放时长时，返回给界面播放
                    if total_duration >= min_chunk_duration or audio_chunk_count % 3 == 0:
                        # 合并累积的音频块
                        if len(accumulated_chunk) == 1:
                            playback_audio = accumulated_chunk[0]
                        else:
                            playback_audio = np.concatenate(accumulated_chunk)

                        # 应用基本的音频优化
                        playback_audio = np.clip(playback_audio, -0.95, 0.95)
                        playback_audio = trim_silence(playback_audio, threshold=0.005)

                        # 轻量级音频增强（避免过度处理）
                        max_val = np.max(np.abs(playback_audio))
                        if max_val > 0:
                            playback_audio = playback_audio / max_val * 0.7

                        playback_duration = len(playback_audio) / 22050

                        print(f"🎵 返回音频块用于播放:")
                        print(f"   - 播放时长: {playback_duration:.2f}秒")
                        print(f"   - 包含 {len(accumulated_chunk)} 个原始音频块")

                        # 返回音频块供界面播放
                        yield (22050, playback_audio)

                        # 清空累积缓冲区，为下一轮做准备
                        accumulated_chunk = []

                        # 添加短暂停顿，让界面有时间处理
                        time.sleep(0.1)

                else:
                    print(f"⚠️ 音频块 {audio_chunk_count} 为空，跳过")

                # 内存管理
                if audio_chunk_count % 5 == 0:
                    optimize_memory()

            # 处理剩余的累积音频（如果有的话）
            if accumulated_chunk:
                print(f"🔗 处理剩余的 {len(accumulated_chunk)} 个音频块...")

                if len(accumulated_chunk) == 1:
                    final_audio = accumulated_chunk[0]
                else:
                    final_audio = np.concatenate(accumulated_chunk)

                # 最终音频优化
                final_audio = np.clip(final_audio, -0.95, 0.95)
                final_audio = trim_silence(final_audio, threshold=0.005)

                # 轻量级增强
                max_val = np.max(np.abs(final_audio))
                if max_val > 0:
                    final_audio = final_audio / max_val * 0.7

                final_duration = len(final_audio) / 22050
                print(f"🎵 返回最终音频块: {final_duration:.2f}秒")

                yield (22050, final_audio)

            total_time = time.time() - start_time
            print("=" * 60)
            print(f"✅ 流式音频生成完成:")
            print(f"   - 总音频块数: {audio_chunk_count}")
            print(f"   - 总处理时间: {total_time:.2f}秒")
            print(f"   - 使用指令: {instruct_text}")
            print("=" * 60)

        except Exception as e:
            print(f"❌ 流式音频生成失败: {e}")
            import traceback
            traceback.print_exc()
            return

    except Exception as e:
        print(f"❌ 流式音频处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return

# 简化的CSS样式 - 黑色背景，白色字体，双音频控件设计 - 13寸屏幕适配
def create_custom_css():
    return """
    /* 全局样式 - 黑色背景主题 */
    .gradio-container {
        background: #000000 !important;
        color: #ffffff !important;
        min-height: 100vh !important;
        padding: 10px !important;
        font-family: 'Arial', sans-serif !important;
        font-size: 14px !important;
    }
    
    /* 主容器样式 - 13寸屏幕优化 */
    .main-container {
        max-width: 900px !important;
        margin: 0 auto !important;
        background: linear-gradient(135deg, #000000, #1a1a1a) !important;
        border-radius: 15px !important;
        padding: 20px !important;
        box-shadow: 0 5px 20px rgba(0,0,0,0.5) !important;
        max-height: 95vh !important;
        overflow-y: auto !important;
    }
    
    /* 标题样式 - 缩小尺寸 */
    .main-title {
        font-size: 2rem !important;
        font-weight: bold !important;
        color: #ffffff !important;
        text-align: center !important;
        margin-bottom: 20px !important;
        text-shadow: 0 0 15px rgba(255,255,255,0.5) !important;
    }
    
    /* 状态显示样式 - 缩小尺寸 */
    .status-display {
        background: rgba(255,255,255,0.1) !important;
        border: 2px solid #ffffff !important;
        border-radius: 10px !important;
        padding: 12px 15px !important;
        margin: 10px 0 !important;
        font-size: 1rem !important;
        color: #ffffff !important;
        text-align: center !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* 音频区域容器 - 缩小间距 */
    .audio-section {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        backdrop-filter: blur(5px) !important;
    }
    
    /* 音频区域标题 - 缩小字体 */
    .audio-title {
        font-size: 1.2rem !important;
        font-weight: bold !important;
        color: #4CAF50 !important;
        text-align: center !important;
        margin-bottom: 10px !important;
        display: block !important;
    }
    
    /* 录音控制按钮 - 缩小高度 */
    .record-button {
        background: linear-gradient(135deg, #4CAF50, #45a049) !important;
        color: white !important;
        border: none !important;
        min-height: 50px !important;
        width: 100% !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
        margin: 8px 0 !important;
    }
    
    .record-button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.4) !important;
    }
    
    /* 停止录音按钮 */
    .stop-button {
        background: linear-gradient(135deg, #F44336, #d32f2f) !important;
        animation: pulse-red 1.5s infinite !important;
    }
    
    @keyframes pulse-red {
        0% { box-shadow: 0 4px 8px rgba(244,67,54,0.3); }
        50% { box-shadow: 0 4px 8px rgba(244,67,54,0.7), 0 0 15px rgba(244,67,54,0.5); }
        100% { box-shadow: 0 4px 8px rgba(244,67,54,0.3); }
    }
    
    /* 处理中按钮 */
    .processing-button {
        background: linear-gradient(135deg, #FF9800, #F57C00) !important;
        cursor: not-allowed !important;
        animation: pulse-orange 1s infinite !important;
    }
    
    @keyframes pulse-orange {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }
    
    /* 音频播放器样式 - 缩小尺寸 */
    .audio-player {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        margin: 8px 0 !important;
    }
    
    /* 音频播放器内部控件调整 */
    .audio-player audio {
        width: 100% !important;
        height: 40px !important;
    }
    
    /* 🔧 新增：重置按钮样式 */
    .reset-button {
        background: linear-gradient(135deg, #FF9800, #F57C00) !important;
        color: white !important;
        border: none !important;
        min-height: 35px !important;
        width: 100% !important;
        font-size: 0.9rem !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 6px rgba(255,152,0,0.3) !important;
        margin-top: 8px !important;
    }
    
    .reset-button:hover {
        background: linear-gradient(135deg, #F57C00, #E65100) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(255,152,0,0.4) !important;
    }
    
    .reset-button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 4px rgba(255,152,0,0.3) !important;
    }
    
    /* Jarvis输出音频标题 */
    .jarvis-title {
        color: #2196F3 !important;
    }
    
    /* 用户录音音频标题 */
    .user-title {
        color: #4CAF50 !important;
    }
    
    /* 流式播放提示 - 缩小尺寸 */
    .streaming-indicator {
        background: linear-gradient(135deg, #2196F3, #1976D2) !important;
        color: white !important;
        padding: 8px 15px !important;
        border-radius: 15px !important;
        font-size: 0.9rem !important;
        text-align: center !important;
        margin: 8px 0 !important;
        animation: pulse-blue 2s infinite !important;
    }
    
    @keyframes pulse-blue {
        0% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.01); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }
    
    /* Gradio默认样式调整 */
    .gradio-container .gradio-row {
        margin: 5px 0 !important;
    }
    
    .gradio-container .gradio-column {
        padding: 5px !important;
    }
    
    /* 音频组件标签样式 */
    .gradio-container label {
        font-size: 0.9rem !important;
        margin-bottom: 5px !important;
    }
    
    /* 滚动条样式 */
    ::-webkit-scrollbar {
        width: 8px !important;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1) !important;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255,255,255,0.3) !important;
        border-radius: 4px !important;
    }
    
    /* 13寸屏幕特殊优化 */
    @media (max-height: 800px) {
        .main-container {
            padding: 15px !important;
            max-height: 98vh !important;
        }
        
        .main-title {
            font-size: 1.8rem !important;
            margin-bottom: 15px !important;
        }
        
        .audio-section {
            padding: 12px !important;
            margin: 8px 0 !important;
        }
        
        .status-display {
            padding: 10px 12px !important;
            font-size: 0.95rem !important;
        }
    }
    
    /* 小屏幕响应式设计 */
    @media (max-width: 768px) {
        .main-container {
            max-width: 95% !important;
            padding: 12px !important;
        }
        
        .main-title {
            font-size: 1.5rem !important;
        }
        
        .record-button {
            min-height: 45px !important;
            font-size: 1rem !important;
        }
        
        .audio-title {
            font-size: 1.1rem !important;
        }
    }
    """

def create_elderly_friendly_interface():
    """创建使用Gradio原生音频控件的简化界面"""

    custom_css = create_custom_css()

    with gr.Blocks(
        theme=gr.themes.Base(),
        css=custom_css,
        title="Jarvis 粤语语音助手"
    ) as interface:

        # 主容器
        with gr.Column(elem_classes=["main-container"]):

            # 标题
            gr.HTML("""
            <div class="main-title">
                🤖 Jarvis 粤语语音助手
            </div>
            """)

            # 全局状态显示
            status_display = gr.HTML(
                value='<div class="status-display">👋 准备就绪，请点击下方录音按钮开始对话</div>'
            )

            # 用户录音区域
            with gr.Column(elem_classes=["audio-section"]):
                gr.HTML('<div class="audio-title user-title">🎤 您的语音输入</div>')

                # 使用Gradio原生音频录制控件
                user_audio = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="点击🎤录音按钮开始说话",
                    show_label=True,
                    elem_classes=["audio-player"],
                    interactive=True
                )

                # 🔧 新增：重置按钮
                with gr.Row():
                    reset_button = gr.Button(
                        value="🔄 重置录音",
                        variant="secondary",
                        size="sm",
                        elem_classes=["reset-button"]
                    )

            # Jarvis回复区域
            with gr.Column(elem_classes=["audio-section"]):
                gr.HTML('<div class="audio-title jarvis-title">🔊 Jarvis 语音回复</div>')

                # 流式播放状态指示
                streaming_indicator = gr.HTML(
                    value="",
                    visible=False
                )

                # Jarvis输出音频组件
                jarvis_audio = gr.Audio(
                    label="Jarvis回复音频",
                    show_label=True,
                    elem_classes=["audio-player"],
                    autoplay=True,
                    interactive=False
                )

        # 音频处理函数 - 当用户完成录音后自动触发
        def process_audio_input(audio_data):
            print(f"🎵 收到音频数据: {type(audio_data) if audio_data else 'None'}")

            if audio_data is None:
                print("❌ 音频数据为空")
                return (
                    gr.update(value='<div class="status-display">❌ 录音失败，请重试</div>'),
                    "",
                    None
                )

            try:
                # 更新状态为处理中
                yield (
                    gr.update(value='<div class="status-display">🧠 正在理解您的话...</div>'),
                    "",
                    None
                )

                # 步骤1: 语音识别
                print("🎤 开始语音识别...")
                prompt_text = listener(audio_data)
                print(f"✅ 识别结果: {prompt_text}")

                yield (
                    gr.update(value=f'<div class="status-display">📝 识别到：{prompt_text}</div>'),
                    "",
                    None
                )

                # 步骤2: LLM生成回复
                print("🧠 开始LLM生成回复...")
                answer_text = Influence.llm(prompt_text)
                print(f"✅ LLM回复: {answer_text}")

                yield (
                    gr.update(value='<div class="status-display">🎵 正在生成语音回复...</div>'),
                    '<div class="streaming-indicator">🔄 正在流式生成语音，请稍候...</div>',
                    None
                )

                # 步骤3: 真正的流式语音合成并播放
                print("🎵 开始流式语音合成...")

                # 🔧 恢复流式播放：每个音频块都实时推送给Gradio播放
                streaming_audio_generator = brain_streaming_with_realtime_playback(audio_data)

                chunk_count = 0

                # 真正的流式播放：每个音频块都立即播放
                for audio_chunk_result in streaming_audio_generator:
                    if audio_chunk_result is None:
                        continue

                    chunk_count += 1
                    sample_rate, audio_chunk = audio_chunk_result

                    # 实时流式播放每个音频块
                    yield (
                        gr.update(value=f'<div class="status-display">🎵 正在播放第 {chunk_count} 段语音...</div>'),
                        '<div class="streaming-indicator">🎵 流式播放中...</div>',
                        (sample_rate, audio_chunk)  # 实时播放当前音频块
                    )

                    print(f"✅ 流式播放：第 {chunk_count} 段音频已推送到界面播放")

                # 播放完成状态
                yield (
                    gr.update(value='<div class="status-display">✅ 对话完成！可以继续录音进行下一轮对话</div>'),
                    "",  # 隐藏流式指示器
                    None  # 清空音频组件
                )

            except Exception as e:
                print(f"❌ 处理异常: {e}")
                import traceback
                traceback.print_exc()
                yield (
                    gr.update(value=f'<div class="status-display">❌ 处理失败: {str(e)}</div>'),
                    "",
                    None
                )

        # 🔧 新增：重置录音功能
        def reset_recording():
            """重置录音和状态"""
            print("🔄 用户点击重置录音")
            return (
                None,  # 清空用户录音
                gr.update(value='<div class="status-display">🔄 录音已重置，请重新录音</div>'),
                "",    # 清空流式指示器
                None   # 清空Jarvis音频
            )

        # 绑定重置按钮事件
        reset_button.click(
            fn=reset_recording,
            inputs=[],
            outputs=[user_audio, status_display, streaming_indicator, jarvis_audio]
        )

        # 绑定音频录制完成事件 - 当用户录音完成后自动处理
        user_audio.change(
            fn=process_audio_input,
            inputs=[user_audio],
            outputs=[status_display, streaming_indicator, jarvis_audio],
            show_progress="hidden"
        )

    return interface

def ui_launch():
    """启动优化后的用户界面"""
    print("🚀 启动 Jarvis 粤语语音助手...")
    print("✨ 界面优化特性:")
    print("   - 适合老年用户的大按钮设计")
    print("   - 清晰的状态指示器")
    print("   - 一键录音开始/停止功能")
    print("   - 简洁的色彩搭配")
    print("   - 实时状态反馈")
    print("   - 响应式布局适配")
    print("   - 无障碍优化支持")

    interface = create_elderly_friendly_interface()

    # 启动界面
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
        quiet=False
    )

if __name__ == "__main__":
    """主函数入口"""
    try:
        print("🎯 Jarvis 粤语语音交互系统 v2.0")
        print("=" * 50)
        print("🚀 系统启动中...")
        print("📝 正在加载组件...")

        # 预初始化关键组件以加快响应速度
        print("⚡ 预初始化组件中...")

        # 启动用户界面
        ui_launch()

    except KeyboardInterrupt:
        print("\n👋 用户终止程序")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        try:
            executor.shutdown(wait=False)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.backends.mps.empty_cache()
        except:
            pass
        print("🔚 程序已退出")
