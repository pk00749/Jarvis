#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试阶段5流式语音处理优化效果
验证低延迟流式合成、音频块优化和预测性资源预加载
"""

import sys
import os
import time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_streaming_performance():
    """测试流式处理性能"""
    print("🧪 测试阶段5流式语音处理优化...")

    # 测试音频块优化
    print("\n1. 测试音频块处理优化:")
    from jarvis import process_audio_chunk_optimized
    import torch

    # 创建测试音频张量
    test_audio = torch.randn(1, 4096, dtype=torch.float32)

    # 测试处理时间
    start_time = time.time()
    for i in range(100):
        processed = process_audio_chunk_optimized(test_audio)
    processing_time = time.time() - start_time

    print(f"✅ 音频块处理优化测试完成")
    print(f"   - 100次处理用时: {processing_time:.3f}秒")
    print(f"   - 平均处理时间: {processing_time/100:.6f}秒")
    print(f"   - 输出音频形状: {processed.shape}")

    return processing_time < 0.1  # 期望100次处理在0.1秒内完成

def test_nlp_optimization():
    """测试NLP分句优化"""
    print("\n2. 测试NLP分句优化:")
    from jarvis import _nlp_generator_optimized

    test_text = "今天天气好好呀！你想去边度玩呢？我哋可以去公园散步，或者去茶餐厅食嘢。"

    # 测试首次分句
    start_time = time.time()
    sentences1 = list(_nlp_generator_optimized(test_text))
    first_time = time.time() - start_time

    # 测试缓存分句
    start_time = time.time()
    sentences2 = list(_nlp_generator_optimized(test_text))
    cached_time = time.time() - start_time

    print(f"✅ NLP分句优化测试完成")
    print(f"   - 首次分句用时: {first_time:.3f}秒")
    print(f"   - 缓存分句用时: {cached_time:.3f}秒")
    print(f"   - 缓存加速比: {first_time/cached_time:.1f}x")
    print(f"   - 分句数量: {len(sentences1)}")

    return cached_time < first_time * 0.5  # 期望缓存至少快50%

def test_memory_optimization():
    """测试内存优化"""
    print("\n3. 测试内存优化:")
    from jarvis import optimize_memory, model_cache, audio_cache

    # 填充缓存
    for i in range(25):
        model_cache[f"test_key_{i}"] = f"test_value_{i}"
        audio_cache[f"audio_key_{i}"] = np.random.randn(1000)

    initial_model_cache_size = len(model_cache)
    initial_audio_cache_size = len(audio_cache)

    print(f"   - 清理前模型缓存: {initial_model_cache_size} 项")
    print(f"   - 清理前音频缓存: {initial_audio_cache_size} 项")

    # 执行内存优化
    optimize_memory()

    final_model_cache_size = len(model_cache)
    final_audio_cache_size = len(audio_cache)

    print(f"   - 清理后模型缓存: {final_model_cache_size} 项")
    print(f"   - 清理后音频缓存: {final_audio_cache_size} 项")
    print(f"✅ 内存优化测试完成")

    return final_model_cache_size < initial_model_cache_size and final_audio_cache_size < initial_audio_cache_size

def test_preload_resources():
    """测试预测性资源预加载"""
    print("\n4. 测试预测性资源预加载:")
    from jarvis import preload_resources

    start_time = time.time()
    result = preload_resources()
    preload_time = time.time() - start_time

    print(f"✅ 预测性资源预加载测试完成")
    print(f"   - 预加载用时: {preload_time:.3f}秒")
    print(f"   - 预加载结果: {'成功' if result else '失败'}")

    return result and preload_time < 3.0  # 期望3秒内完成

def test_cosyvoice_optimizations():
    """测试CosyVoice2优化"""
    print("\n5. 测试CosyVoice2流式优化:")
    from jarvis import get_cosyvoice

    try:
        # 测试模型实例获取
        start_time = time.time()
        cosyvoice_instance = get_cosyvoice()
        init_time = time.time() - start_time

        print(f"✅ CosyVoice2实例获取测试完成")
        print(f"   - 初始化用时: {init_time:.3f}秒")
        print(f"   - 实例类型: {type(cosyvoice_instance).__name__}")

        # 测试预热功能
        if hasattr(cosyvoice_instance, '_warmup_model_if_needed'):
            start_time = time.time()
            cosyvoice_instance._warmup_model_if_needed()
            warmup_time = time.time() - start_time

            print(f"   - 模型预热用时: {warmup_time:.3f}秒")
            print(f"   - 预热状态: {getattr(cosyvoice_instance, '_model_warmed', False)}")

        return True

    except Exception as e:
        print(f"❌ CosyVoice2优化测试失败: {e}")
        return False

def performance_benchmark():
    """性能基准测试"""
    print("\n6. 性能基准测试:")

    # 模拟完整的流式处理流程
    metrics = {
        'audio_processing': 0,
        'nlp_processing': 0,
        'memory_cleanup': 0,
        'resource_preload': 0
    }

    # 音频处理基准
    start_time = time.time()
    test_streaming_performance()
    metrics['audio_processing'] = time.time() - start_time

    # NLP处理基准
    start_time = time.time()
    test_nlp_optimization()
    metrics['nlp_processing'] = time.time() - start_time

    # 内存清理基准
    start_time = time.time()
    test_memory_optimization()
    metrics['memory_cleanup'] = time.time() - start_time

    # 资源预加载基准
    start_time = time.time()
    test_preload_resources()
    metrics['resource_preload'] = time.time() - start_time

    print(f"\n📊 性能基准测试结果:")
    for component, time_taken in metrics.items():
        status = "🚀" if time_taken < 1.0 else "⚡" if time_taken < 3.0 else "🐌"
        print(f"   - {component}: {time_taken:.3f}秒 {status}")

    total_time = sum(metrics.values())
    print(f"   - 总体性能: {total_time:.3f}秒")

    return total_time < 10.0  # 期望总体测试在10秒内完成

def run_all_tests():
    """运行所有测试"""
    print("🎯 开始阶段5测试 - 流式语音处理优化\n")

    tests = [
        ("音频块处理优化", test_streaming_performance),
        ("NLP分句优化", test_nlp_optimization),
        ("内存优化", test_memory_optimization),
        ("预测性资源预加载", test_preload_resources),
        ("CosyVoice2优化", test_cosyvoice_optimizations),
        ("性能基准测试", performance_benchmark)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*50}")
            print(f"测试项目: {test_name}")
            print(f"{'='*50}")

            if test_func():
                passed += 1
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")

        except Exception as e:
            print(f"❌ {test_name} 测试出错: {e}")

    print(f"\n{'='*50}")
    print(f"🎉 阶段5测试完成: {passed}/{total} 测试通过")
    print(f"{'='*50}")

    if passed == total:
        print("🎯 阶段5任务完成 - 流式语音处理优化成功")
        print("\n🚀 优化成果:")
        print("   - 智能音频块处理和缓冲")
        print("   - 模型预热减少冷启动时间")
        print("   - 预测性资源预加载")
        print("   - 优化的NLP分句处理")
        print("   - 动态内存管理")
        print("   - 详细的性能监控")
        return True
    else:
        print("⚠️  部分测试未通过，需要进一步调试")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
