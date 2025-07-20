#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段7：系统集成测试 - 简化版本
逐步执行各个测试项目
"""

import os
import sys
import time
import json
from datetime import datetime

# 添加项目路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

def test_basic_imports():
    """测试基本导入"""
    print("🔍 测试1：基本组件导入测试")
    try:
        from influence.influence import Influence
        print("  ✅ Influence导入成功")

        from listen.listen import Listen
        print("  ✅ Listen导入成功")

        from cosyvoice.cli.cosyvoice import CosyVoice2
        print("  ✅ CosyVoice2导入成功")

        return True
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False

def test_component_initialization():
    """测试组件初始化"""
    print("\n🚀 测试2：组件初始化测试")
    try:
        from influence.influence import Influence
        influence = Influence()
        print("  ✅ Influence初始化成功")

        from listen.listen import Listen
        listen = Listen()
        print("  ✅ Listen初始化成功")

        # CosyVoice2初始化需要更多时间
        print("  ⏳ 正在初始化CosyVoice2...")
        from jarvis import initialize_cosyvoice_optimized
        cosyvoice = initialize_cosyvoice_optimized()
        print("  ✅ CosyVoice2初始化成功")

        return influence, listen, cosyvoice
    except Exception as e:
        print(f"  ❌ 初始化失败: {e}")
        return None, None, None

def test_llm_functionality(influence):
    """测试LLM功能"""
    print("\n💬 测试3：LLM功能测试")
    if influence is None:
        print("  ❌ Influence未初始化，跳过测试")
        return False

    test_cases = [
        "你好",
        "今日天气点样？",
        "帮我讲个故事"
    ]

    results = []
    for i, test_text in enumerate(test_cases):
        try:
            start_time = time.time()
            response = influence.llm(test_text)
            response_time = time.time() - start_time

            print(f"  📝 测试 {i+1}: {test_text}")
            print(f"     回复: {response[:100]}...")
            print(f"     响应时间: {response_time:.2f}s")

            results.append({
                'input': test_text,
                'output': response,
                'time': response_time,
                'success': len(response) > 0
            })

        except Exception as e:
            print(f"  ❌ 测试 {i+1} 失败: {e}")
            results.append({
                'input': test_text,
                'output': "",
                'time': 0,
                'success': False
            })

    success_count = sum(1 for r in results if r['success'])
    success_rate = success_count / len(results) * 100
    print(f"  ✅ LLM测试成功率: {success_rate:.1f}%")

    return success_rate > 80

def test_speech_synthesis(cosyvoice):
    """测试语音合成功能"""
    print("\n🔊 测试4：语音合成测试")
    if cosyvoice is None:
        print("  ❌ CosyVoice2未初始化，跳过测试")
        return False

    test_texts = [
        "你好，我係Jarvis",
        "今日天气好好",
        "多谢你"
    ]

    results = []
    for i, text in enumerate(test_texts):
        try:
            start_time = time.time()
            audio_chunks = []

            print(f"  🎵 合成文本 {i+1}: {text}")

            # 测试流式合成
            for chunk in cosyvoice.inference_instruct2(
                text, "女性，温柔，粤语", stream=True
            ):
                if chunk is not None:
                    audio_chunks.append(chunk)

            synthesis_time = time.time() - start_time
            print(f"     合成时间: {synthesis_time:.2f}s")
            print(f"     音频块数: {len(audio_chunks)}")

            results.append({
                'text': text,
                'time': synthesis_time,
                'chunks': len(audio_chunks),
                'success': len(audio_chunks) > 0
            })

        except Exception as e:
            print(f"  ❌ 合成失败 {i+1}: {e}")
            results.append({
                'text': text,
                'time': 0,
                'chunks': 0,
                'success': False
            })

    success_count = sum(1 for r in results if r['success'])
    success_rate = success_count / len(results) * 100
    print(f"  ✅ 语音合成成功率: {success_rate:.1f}%")

    return success_rate > 80

def test_end_to_end_flow(influence, cosyvoice):
    """测试端到端流程"""
    print("\n🔄 测试5：端到端流程测试")
    if influence is None or cosyvoice is None:
        print("  ❌ 组件未完全初始化，跳过测试")
        return False

    test_input = "你好，今日天气点样？"

    try:
        total_start = time.time()

        # LLM处理
        llm_start = time.time()
        response = influence.llm(test_input)
        llm_time = time.time() - llm_start

        # 语音合成
        synthesis_start = time.time()
        chunk_count = 0
        for chunk in cosyvoice.inference_instruct2(
            response[:50], "女性，温柔，粤语", stream=True
        ):
            if chunk is not None:
                chunk_count += 1
        synthesis_time = time.time() - synthesis_start

        total_time = time.time() - total_start

        print(f"  📝 输入: {test_input}")
        print(f"  📝 LLM回复: {response[:100]}...")
        print(f"  ⏱️  LLM处理时间: {llm_time:.2f}s")
        print(f"  ⏱️  语音合成时间: {synthesis_time:.2f}s")
        print(f"  ⏱️  总处理时间: {total_time:.2f}s")
        print(f"  🎵 生成音频块数: {chunk_count}")

        # 检查性能目标
        performance_ok = total_time < 10
        functionality_ok = len(response) > 0 and chunk_count > 0

        print(f"  ✅ 性能达标: {'是' if performance_ok else '否'} (目标<10s)")
        print(f"  ✅ 功能正常: {'是' if functionality_ok else '否'}")

        return performance_ok and functionality_ok

    except Exception as e:
        print(f"  ❌ 端到端测试失败: {e}")
        return False

def test_speech_recognition(influence):
    """测试语音识别功能"""
    print("\n🎤 测试6：语音识别测试")
    if influence is None:
        print("  ❌ Influence未初始化，跳过测试")
        return False

    # 查找录音文件
    recordings_dir = os.path.join(ROOT_DIR, 'recordings')
    if not os.path.exists(recordings_dir):
        print("  ⚠️  recordings目录不存在，跳过语音识别测试")
        return True  # 不算失败

    audio_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')][:3]
    if not audio_files:
        print("  ⚠️  未找到音频文件，跳过语音识别测试")
        return True

    results = []
    for audio_file in audio_files:
        audio_path = os.path.join(recordings_dir, audio_file)
        try:
            start_time = time.time()
            result = influence.voice_to_text(audio_path)
            recognition_time = time.time() - start_time

            print(f"  🎵 {audio_file}: {result} ({recognition_time:.2f}s)")

            results.append({
                'file': audio_file,
                'result': result,
                'time': recognition_time,
                'success': len(result.strip()) > 0
            })

        except Exception as e:
            print(f"  ❌ 识别失败 {audio_file}: {e}")
            results.append({
                'file': audio_file,
                'result': "",
                'time': 0,
                'success': False
            })

    if results:
        success_count = sum(1 for r in results if r['success'])
        success_rate = success_count / len(results) * 100
        print(f"  ✅ 语音识别成功率: {success_rate:.1f}%")
        return success_rate > 60

    return True

def main():
    """主测试函数"""
    print("🧪 启动Jarvis系统集成测试 (阶段7)")
    print("=" * 50)

    test_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }

    # 测试1：基本导入
    test_results['tests']['imports'] = test_basic_imports()

    if not test_results['tests']['imports']:
        print("❌ 基本导入测试失败，终止测试")
        return

    # 测试2：组件初始化
    influence, listen, cosyvoice = test_component_initialization()
    test_results['tests']['initialization'] = all([influence, listen, cosyvoice])

    # 测试3：LLM功能
    test_results['tests']['llm'] = test_llm_functionality(influence)

    # 测试4：语音合成
    test_results['tests']['synthesis'] = test_speech_synthesis(cosyvoice)

    # 测试5：端到端流程
    test_results['tests']['end_to_end'] = test_end_to_end_flow(influence, cosyvoice)

    # 测试6：语音识别
    test_results['tests']['speech_recognition'] = test_speech_recognition(influence)

    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)

    passed_tests = sum(1 for result in test_results['tests'].values() if result)
    total_tests = len(test_results['tests'])
    success_rate = passed_tests / total_tests * 100

    for test_name, result in test_results['tests'].items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name:20}: {status}")

    print(f"\n总体通过率: {success_rate:.1f}% ({passed_tests}/{total_tests})")

    # 保存测试报告
    report_path = os.path.join(ROOT_DIR, 'tests', 'stage7_test_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)

    print(f"📄 测试报告已保存: {report_path}")

    if success_rate >= 80:
        print("🎉 系统集成测试基本通过！")
    else:
        print("⚠️  系统集成测试存在问题，需要进一步优化")

if __name__ == '__main__':
    main()
