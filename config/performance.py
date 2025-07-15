"""
Mac M3 性能优化配置
针对 Apple Silicon 的 Jarvis 语音助手性能调优
"""
import os
import torch
import multiprocessing as mp
from typing import Dict, Any

class M3PerformanceConfig:
    """Mac M3 性能优化配置类"""

    def __init__(self):
        self.cpu_count = mp.cpu_count()
        self.memory_gb = self._get_memory_info()
        self.configure_environment()

    @staticmethod
    def _get_memory_info() -> int:
        """获取系统内存信息"""
        try:
            import psutil
            return int(psutil.virtual_memory().total / (1024**3))
        except ImportError:
            return 16  # 默认假设16GB内存

    def configure_environment(self):
        """配置环境变量以优化性能"""
        # PyTorch 优化
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

        # 线程配置 - M3 有8个性能核心 + 4个能效核心
        optimal_threads = min(8, self.cpu_count)
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
        os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)

        # 内存优化
        os.environ['MALLOC_MMAP_THRESHOLD_'] = '65536'
        os.environ['MALLOC_TRIM_THRESHOLD_'] = '131072'

        # Gradio 优化
        os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
        os.environ['GRADIO_SERVER_NAME'] = '0.0.0.0'

        print(f"🔧 已配置 M3 优化: {optimal_threads} 线程, {self.memory_gb}GB 内存")

    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
            'dtype': torch.float16 if torch.backends.mps.is_available() else torch.float32,
            'batch_size': 1,
            'max_memory_gb': max(4, self.memory_gb // 4),  # 使用25%的系统内存
            'num_threads': min(8, self.cpu_count),
            'enable_optimization': True
        }

    @staticmethod
    def get_audio_config() -> Dict[str, Any]:
        """获取音频处理配置"""
        return {
            'sample_rate': 16000,
            'chunk_size': 1024,
            'buffer_size': 4096,
            'channels': 1,
            'format': 'float32',
            'enable_noise_reduction': True,
            'enable_echo_cancellation': True
        }

    def get_gradio_config(self) -> Dict[str, Any]:
        """获取Gradio界面配置"""
        return {
            'max_threads': min(8, self.cpu_count),
            'queue_max_size': 10,
            'enable_queue': True,
            'show_error': True,
            'debug': False,
            'analytics_enabled': False,
            'auth': None,
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': False,
            'inbrowser': True
        }

# 全局配置实例
m3_config = M3PerformanceConfig()

def optimize_torch_for_m3():
    """优化PyTorch以适配M3芯片"""
    if torch.backends.mps.is_available():
        print("✅ 启用 MPS (Metal Performance Shaders)")
        # 修复：使用正确的MPS内存清理方法
        try:
            torch.mps.empty_cache()
        except AttributeError:
            # 兼容旧版本PyTorch
            pass
    else:
        print("⚠️  MPS 不可用，使用 CPU")

    # 设置线程数
    torch.set_num_threads(m3_config.get_model_config()['num_threads'])

    # 启用优化
    torch.backends.cudnn.benchmark = False  # M3不使用CUDNN
    torch.backends.cudnn.deterministic = True

def get_optimal_worker_count() -> int:
    """获取最佳工作线程数"""
    return min(4, m3_config.cpu_count // 2)

def monitor_performance():
    """监控性能指标"""
    try:
        import psutil
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent

        print(f"📊 性能监控: CPU {cpu_percent:.1f}%, 内存 {memory_percent:.1f}%")

        if cpu_percent > 80:
            print("⚠️  CPU使用率过高，建议降低处理负载")
        if memory_percent > 85:
            print("⚠️  内存使用率过高，建议重启应用")

    except ImportError:
        print("💡 安装 psutil 以启用性能监控: pip install psutil")

def cleanup_memory():
    """清理内存"""
    import gc
    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("🧹 已清理内存缓存")

if __name__ == "__main__":
    print("🚀 Mac M3 性能配置测试")
    print(f"CPU核心数: {m3_config.cpu_count}")
    print(f"系统内存: {m3_config.memory_gb}GB")
    print(f"模型配置: {m3_config.get_model_config()}")
    print(f"音频配置: {m3_config.get_audio_config()}")
    print(f"Gradio配置: {m3_config.get_gradio_config()}")

    optimize_torch_for_m3()
    monitor_performance()
