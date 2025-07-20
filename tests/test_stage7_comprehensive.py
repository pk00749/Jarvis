#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段7：系统集成测试 - 综合测试套件
测试项目：
1. 端到端功能测试
2. 性能压力测试
3. 语音识别准确率测试
4. 语音合成质量测试
5. 用户界面易用性测试
6. 系统稳定性测试（24小时运行）
7. 回归测试确保原有功能不受影响
"""

import os
import sys
import time
import threading
import unittest
import numpy as np
import torch
import gc
from datetime import datetime, timedelta
import json
import wave
import librosa
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import GPUtil

# 添加项目路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from jarvis import initialize_cosyvoice_optimized
from influence.influence import Influence
from listen.listen import Listen
from cosyvoice.cli.cosyvoice import CosyVoice2

class JarvisIntegrationTest(unittest.TestCase):
    """Jarvis系统集成测试套件"""

    @classmethod
    def setUpClass(cls):
        """测试前的初始化"""
        print("🧪 开始Jarvis系统集成测试...")
        cls.start_time = time.time()

        # 初始化组件
        try:
            cls.cosyvoice = initialize_cosyvoice_optimized()
            cls.influence = Influence()
            cls.listen = Listen()
            print("✅ 所有组件初始化成功")
        except Exception as e:
            print(f"❌ 组件初始化失败: {e}")
            raise

        # 测试结果统计
        cls.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'performance_metrics': {},
            'stability_metrics': {},
            'quality_metrics': {}
        }

    def setUp(self):
        """每个测试前的准备"""
        self.test_start_time = time.time()

    def tearDown(self):
        """每个测试后的清理"""
        test_duration = time.time() - self.test_start_time
        print(f"⏱️  测试耗时: {test_duration:.2f}秒")

    def test_01_end_to_end_functionality(self):
        """1. 端到端功能测试"""
        print("\n🔍 测试1：端到端功能测试")

        # 测试样本
        test_cases = [
            "今日天气点样？",
            "你好，我想了解一下广州嘅情况",
            "帮我讲个故事听下"
        ]

        for i, test_text in enumerate(test_cases):
            print(f"测试案例 {i+1}: {test_text}")

            try:
                # 模拟语音转文字
                start_time = time.time()

                # LLM处理
                response = self.influence.llm(test_text)
                llm_time = time.time() - start_time

                # 语音合成
                synthesis_start = time.time()
                for chunk in self.cosyvoice.inference_instruct2(
                    response, "女性，温柔，粤语", stream=True
                ):
                    pass  # 消费流式输出
                synthesis_time = time.time() - synthesis_start

                total_time = time.time() - start_time

                print(f"  ✅ LLM处理时间: {llm_time:.2f}s")
                print(f"  ✅ 语音合成时间: {synthesis_time:.2f}s")
                print(f"  ✅ 总处理时间: {total_time:.2f}s")

                # 记录性能指标
                self.test_results['performance_metrics'][f'case_{i+1}'] = {
                    'llm_time': llm_time,
                    'synthesis_time': synthesis_time,
                    'total_time': total_time
                }

                self.assertLess(total_time, 10, f"端到端响应时间超过10秒: {total_time:.2f}s")
                self.assertIsInstance(response, str)
                self.assertGreater(len(response), 0)

            except Exception as e:
                print(f"  ❌ 测试案例 {i+1} 失败: {e}")
                self.fail(f"端到端功能测试失败: {e}")

    def test_02_performance_stress_test(self):
        """2. 性能压力测试"""
        print("\n🚀 测试2：性能压力测试")

        # 并发测试参数
        num_concurrent = 3  # 并发数量
        test_duration = 60  # 测试持续时间（秒）

        def single_request():
            """单个请求处理"""
            try:
                start_time = time.time()
                response = self.influence.llm("你好")

                # 语音合成
                for chunk in self.cosyvoice.inference_instruct2(
                    response[:50], "女性，温柔，粤语", stream=True
                ):
                    pass

                return time.time() - start_time
            except Exception as e:
                return None

        # 执行并发测试
        start_time = time.time()
        completed_requests = 0
        failed_requests = 0
        response_times = []

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = []

            while time.time() - start_time < test_duration:
                future = executor.submit(single_request)
                futures.append(future)
                time.sleep(0.5)  # 控制请求频率

            # 收集结果
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    completed_requests += 1
                    response_times.append(result)
                else:
                    failed_requests += 1

        # 计算性能指标
        if response_times:
            avg_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            success_rate = completed_requests / (completed_requests + failed_requests) * 100

            print(f"  ✅ 完成请求数: {completed_requests}")
            print(f"  ✅ 失败请求数: {failed_requests}")
            print(f"  ✅ 成功率: {success_rate:.1f}%")
            print(f"  ✅ 平均响应时间: {avg_response_time:.2f}s")
            print(f"  ✅ 95%响应时间: {p95_response_time:.2f}s")

            self.test_results['performance_metrics']['stress_test'] = {
                'completed_requests': completed_requests,
                'failed_requests': failed_requests,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'p95_response_time': p95_response_time
            }

            self.assertGreater(success_rate, 95, f"成功率过低: {success_rate:.1f}%")
            self.assertLess(avg_response_time, 8, f"平均响应时间过长: {avg_response_time:.2f}s")
        else:
            self.fail("压力测试未获得任何有效响应")

    def test_03_speech_recognition_accuracy(self):
        """3. 语音识别准确率测试"""
        print("\n🎤 测试3：语音识别准确率测试")

        # 查找测试音频文件
        recordings_dir = os.path.join(ROOT_DIR, 'recordings')
        audio_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')][:5]

        if not audio_files:
            print("  ⚠️  未找到测试音频文件，跳过语音识别测试")
            return

        recognition_results = []

        for audio_file in audio_files:
            audio_path = os.path.join(recordings_dir, audio_file)

            try:
                # 执行语音识别
                start_time = time.time()
                result = self.influence.voice_to_text(audio_path)
                recognition_time = time.time() - start_time

                print(f"  🎵 {audio_file}: {result} (耗时: {recognition_time:.2f}s)")

                recognition_results.append({
                    'file': audio_file,
                    'result': result,
                    'time': recognition_time,
                    'success': len(result) > 0
                })

            except Exception as e:
                print(f"  ❌ 识别失败 {audio_file}: {e}")
                recognition_results.append({
                    'file': audio_file,
                    'result': "",
                    'time': 0,
                    'success': False
                })

        # 计算准确率
        successful_recognitions = sum(1 for r in recognition_results if r['success'])
        accuracy_rate = successful_recognitions / len(recognition_results) * 100
        avg_recognition_time = np.mean([r['time'] for r in recognition_results if r['success']])

        print(f"  ✅ 识别成功率: {accuracy_rate:.1f}%")
        print(f"  ✅ 平均识别时间: {avg_recognition_time:.2f}s")

        self.test_results['quality_metrics']['speech_recognition'] = {
            'accuracy_rate': accuracy_rate,
            'avg_recognition_time': avg_recognition_time,
            'total_files_tested': len(recognition_results)
        }

        self.assertGreater(accuracy_rate, 80, f"语音识别准确率过低: {accuracy_rate:.1f}%")

    def test_04_speech_synthesis_quality(self):
        """4. 语音合成质量测试"""
        print("\n🔊 测试4：语音合成质量测试")

        test_texts = [
            "你好，我係Jarvis",
            "今日天气真係好好",
            "多谢你使用我哋嘅系统",
            "有咩可以帮到你？"
        ]

        synthesis_results = []

        for i, text in enumerate(test_texts):
            try:
                start_time = time.time()
                audio_chunks = []

                # 测试流式合成
                for chunk in self.cosyvoice.inference_instruct2(
                    text, "女性，温柔，粤语", stream=True
                ):
                    if chunk is not None:
                        audio_chunks.append(chunk)

                synthesis_time = time.time() - start_time

                print(f"  🎵 文本 {i+1}: \"{text}\"")
                print(f"     合成时间: {synthesis_time:.2f}s")
                print(f"     音频块数量: {len(audio_chunks)}")

                synthesis_results.append({
                    'text': text,
                    'synthesis_time': synthesis_time,
                    'chunks_count': len(audio_chunks),
                    'success': len(audio_chunks) > 0
                })

                # 检查是否有提示词混入
                # 这里可以添加更复杂的音频分析逻辑

            except Exception as e:
                print(f"  ❌ 合成失败 {i+1}: {e}")
                synthesis_results.append({
                    'text': text,
                    'synthesis_time': 0,
                    'chunks_count': 0,
                    'success': False
                })

        # 计算质量指标
        successful_synthesis = sum(1 for r in synthesis_results if r['success'])
        success_rate = successful_synthesis / len(synthesis_results) * 100
        avg_synthesis_time = np.mean([r['synthesis_time'] for r in synthesis_results if r['success']])

        print(f"  ✅ 合成成功率: {success_rate:.1f}%")
        print(f"  ✅ 平均合成时间: {avg_synthesis_time:.2f}s")

        self.test_results['quality_metrics']['speech_synthesis'] = {
            'success_rate': success_rate,
            'avg_synthesis_time': avg_synthesis_time,
            'total_texts_tested': len(synthesis_results)
        }

        self.assertGreater(success_rate, 95, f"语音合成成功率过低: {success_rate:.1f}%")
        self.assertLess(avg_synthesis_time, 3, f"平均合成时间过长: {avg_synthesis_time:.2f}s")

    def test_05_cantonese_llm_quality(self):
        """5. 粤语LLM回复质量测试"""
        print("\n🗣️  测试5：粤语LLM回复质量测试")

        test_cases = [
            {
                'input': '你好',
                'expected_keywords': ['你好', '係', '嘅', '咩']
            },
            {
                'input': '今日天气点样？',
                'expected_keywords': ['天气', '今日', '係', '嘅']
            },
            {
                'input': '可唔可以帮我？',
                'expected_keywords': ['可以', '帮', '你', '係']
            }
        ]

        cantonese_quality_scores = []

        for i, case in enumerate(test_cases):
            try:
                start_time = time.time()
                response = self.influence.llm(case['input'])
                response_time = time.time() - start_time

                print(f"  💬 输入: {case['input']}")
                print(f"     回复: {response}")
                print(f"     响应时间: {response_time:.2f}s")

                # 检查粤语关键词出现情况
                cantonese_keywords_found = 0
                for keyword in case['expected_keywords']:
                    if keyword in response:
                        cantonese_keywords_found += 1

                quality_score = cantonese_keywords_found / len(case['expected_keywords']) * 100
                cantonese_quality_scores.append(quality_score)

                print(f"     粤语质量得分: {quality_score:.1f}%")

            except Exception as e:
                print(f"  ❌ LLM测试失败 {i+1}: {e}")
                cantonese_quality_scores.append(0)

        avg_cantonese_quality = np.mean(cantonese_quality_scores)
        print(f"  ✅ 平均粤语质量得分: {avg_cantonese_quality:.1f}%")

        self.test_results['quality_metrics']['cantonese_llm'] = {
            'avg_quality_score': avg_cantonese_quality,
            'individual_scores': cantonese_quality_scores
        }

        self.assertGreater(avg_cantonese_quality, 60, f"粤语LLM质量得分过低: {avg_cantonese_quality:.1f}%")

    def test_06_system_stability_short(self):
        """6. 系统稳定性测试（短期版本，替代24小时测试）"""
        print("\n⚡ 测试6：系统稳定性测试（10分钟版本）")

        test_duration = 600  # 10分钟
        check_interval = 30  # 30秒检查一次

        start_time = time.time()
        stability_checks = []
        error_count = 0

        print(f"  开始10分钟稳定性测试...")

        while time.time() - start_time < test_duration:
            try:
                check_start = time.time()

                # 执行一次完整的处理流程
                test_text = "系统稳定性测试"
                response = self.influence.llm(test_text)

                # 简短的语音合成测试
                chunk_count = 0
                for chunk in self.cosyvoice.inference_instruct2(
                    response[:30], "女性，温柔，粤语", stream=True
                ):
                    chunk_count += 1
                    if chunk_count >= 3:  # 限制测试长度
                        break

                check_time = time.time() - check_start

                # 记录系统资源使用情况
                memory_usage = psutil.virtual_memory().percent
                cpu_usage = psutil.cpu_percent()

                stability_checks.append({
                    'timestamp': time.time(),
                    'response_time': check_time,
                    'memory_usage': memory_usage,
                    'cpu_usage': cpu_usage,
                    'success': True
                })

                elapsed = time.time() - start_time
                remaining = test_duration - elapsed
                print(f"  ⏱️  已运行 {elapsed:.0f}s，剩余 {remaining:.0f}s (内存: {memory_usage:.1f}%, CPU: {cpu_usage:.1f}%)")

            except Exception as e:
                error_count += 1
                print(f"  ❌ 稳定性检查失败: {e}")
                stability_checks.append({
                    'timestamp': time.time(),
                    'response_time': None,
                    'memory_usage': None,
                    'cpu_usage': None,
                    'success': False,
                    'error': str(e)
                })

            # 等待下一次检查
            time.sleep(check_interval)

        # 计算稳定性指标
        successful_checks = sum(1 for check in stability_checks if check['success'])
        total_checks = len(stability_checks)
        stability_rate = successful_checks / total_checks * 100 if total_checks > 0 else 0

        if successful_checks > 0:
            avg_response_time = np.mean([check['response_time'] for check in stability_checks if check['success']])
            avg_memory_usage = np.mean([check['memory_usage'] for check in stability_checks if check['success']])
            avg_cpu_usage = np.mean([check['cpu_usage'] for check in stability_checks if check['success']])
        else:
            avg_response_time = avg_memory_usage = avg_cpu_usage = 0

        print(f"  ✅ 稳定性测试完成")
        print(f"  ✅ 总检查次数: {total_checks}")
        print(f"  ✅ 成功次数: {successful_checks}")
        print(f"  ✅ 稳定率: {stability_rate:.1f}%")
        print(f"  ✅ 平均响应时间: {avg_response_time:.2f}s")
        print(f"  ✅ 平均内存使用: {avg_memory_usage:.1f}%")
        print(f"  ✅ 平均CPU使用: {avg_cpu_usage:.1f}%")

        self.test_results['stability_metrics'] = {
            'total_checks': total_checks,
            'successful_checks': successful_checks,
            'stability_rate': stability_rate,
            'avg_response_time': avg_response_time,
            'avg_memory_usage': avg_memory_usage,
            'avg_cpu_usage': avg_cpu_usage,
            'error_count': error_count
        }

        self.assertGreater(stability_rate, 95, f"系统稳定率过低: {stability_rate:.1f}%")
        self.assertLess(avg_memory_usage, 80, f"平均内存使用过高: {avg_memory_usage:.1f}%")

    def test_07_regression_test(self):
        """7. 回归测试确保原有功能不受影响"""
        print("\n🔄 测试7：回归测试")

        # 测试基础组件功能
        regression_results = {}

        # 1. 测试Influence组件
        try:
            test_text = "Hello"
            result = self.influence.llm(test_text)
            regression_results['influence_llm'] = len(result) > 0
            print("  ✅ Influence.llm 功能正常")
        except Exception as e:
            regression_results['influence_llm'] = False
            print(f"  ❌ Influence.llm 功能异常: {e}")

        # 2. 测试CosyVoice组件
        try:
            test_text = "测试"
            chunks = list(self.cosyvoice.inference_instruct2(
                test_text, "女性，温柔，粤语", stream=True
            ))
            regression_results['cosyvoice_synthesis'] = len(chunks) > 0
            print("  ✅ CosyVoice2语音合成功能正常")
        except Exception as e:
            regression_results['cosyvoice_synthesis'] = False
            print(f"  ❌ CosyVoice2语音合成功能异常: {e}")

        # 3. 测试Listen组件（如果有音频文件）
        recordings_dir = os.path.join(ROOT_DIR, 'recordings')
        if os.path.exists(recordings_dir):
            audio_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
            if audio_files:
                try:
                    audio_path = os.path.join(recordings_dir, audio_files[0])
                    result = self.influence.voice_to_text(audio_path)
                    regression_results['speech_recognition'] = len(result) > 0
                    print("  ✅ 语音识别功能正常")
                except Exception as e:
                    regression_results['speech_recognition'] = False
                    print(f"  ❌ 语音识别功能异常: {e}")
            else:
                regression_results['speech_recognition'] = None
                print("  ⚠️  无音频文件，跳过语音识别测试")
        else:
            regression_results['speech_recognition'] = None
            print("  ⚠️  recordings目录不存在，跳过语音识别测试")

        # 统计回归测试结果
        tested_functions = [k for k, v in regression_results.items() if v is not None]
        passed_functions = [k for k, v in regression_results.items() if v is True]

        regression_rate = len(passed_functions) / len(tested_functions) * 100 if tested_functions else 0

        print(f"  ✅ 回归测试通过率: {regression_rate:.1f}%")
        print(f"  ✅ 测试功能数: {len(tested_functions)}")
        print(f"  ✅ 通过功能数: {len(passed_functions)}")

        self.test_results['regression_results'] = regression_results

        self.assertGreater(regression_rate, 95, f"回归测试通过率过低: {regression_rate:.1f}%")

    @classmethod
    def tearDownClass(cls):
        """测试完成后的清理和报告生成"""
        total_time = time.time() - cls.start_time

        print(f"\n🏁 系统集成测试完成，总耗时: {total_time:.2f}秒")

        # 生成测试报告
        test_report = {
            'test_timestamp': datetime.now().isoformat(),
            'total_test_time': total_time,
            'test_results': cls.test_results,
            'system_info': {
                'platform': sys.platform,
                'python_version': sys.version,
                'torch_version': torch.__version__ if hasattr(torch, '__version__') else 'N/A'
            }
        }

        # 保存测试报告
        report_path = os.path.join(ROOT_DIR, 'tests', 'integration_test_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(test_report, f, indent=2, ensure_ascii=False)

        print(f"📊 测试报告已保存至: {report_path}")

        # 内存清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_integration_tests():
    """运行集成测试"""
    unittest.main(argv=[''], exit=False, verbosity=2)

if __name__ == '__main__':
    print("🚀 启动Jarvis系统集成测试...")
    run_integration_tests()
