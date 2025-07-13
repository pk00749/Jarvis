"""性能优化配置"""
import os

# 内存相关配置
MEMORY_CONFIG = {
    'max_threads': 2,              # 限制线程数
    'batch_size': 1,              # 最小批处理大小
    'max_chunks_in_memory': 4,    # 内存中最大音频块数量
    'enable_fp16': True,          # 启用半精度
    'enable_jit': True,           # 启用 JIT
    'use_cpu': True               # 强制使用 CPU (Mac上更稳定)
}

# 音频处理配置
AUDIO_CONFIG = {
    'buffer_size': 12000,         # 500ms 音频长度
    'chunk_size': 6000,          # 250ms 音频长度
    'overlap_size': 1200,        # 50ms 重叠长度
    'max_text_length': 50        # 单次处理的最大文本长度
}

# 环境变量配置
ENV_CONFIG = {
    'MKL_NUM_THREADS': '2',
    'NUMEXPR_NUM_THREADS': '2',
    'OMP_NUM_THREADS': '2',
    'OPENBLAS_NUM_THREADS': '2',
    'TORCH_NUM_THREADS': '2'
}

def apply_performance_config():
    """应用性能优化配置"""
    # 设置环境变量
    for key, value in ENV_CONFIG.items():
        os.environ[key] = value

    # 配置 NumPy
    try:
        import numpy as np
        np.set_printoptions(threshold=1000)
    except ImportError:
        pass

    # 配置 PyTorch
    try:
        import torch
        torch.set_num_threads(MEMORY_CONFIG['max_threads'])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
