"""性能优化配置"""
import os

# 内存相关配置
MEMORY_CONFIG = {
    'max_threads': 1,              # MacBook优化：降低线程数
    'batch_size': 1,              # 保持最小批处理以减少内存占用
    'max_chunks_in_memory': 2,     # 减少内存中的音频块数量
    'enable_fp16': True,           # 启用半精度以降低内存占用
    'enable_jit': True,            # 启用JIT优化性能
    'use_cpu': True,              # MacBook上使用CPU
    'gc_interval': 10,            # 垃圾回收间隔
    'model_offload': True         # 启用模型卸载
}

# 音频处理配置
AUDIO_CONFIG = {
    'buffer_size': 24000,         # 增加到1000ms提高连贯性
    'chunk_size': 12000,          # 增加到500ms
    'overlap_size': 2400,         # 增加到100ms提高平滑度
    'max_text_length': 30,        # 减少单次处理文本长度
    'prefetch_size': 2,          # 预加载块数量
    'stream_buffer_count': 3      # 流式处理缓冲区数量
}

# 环境变量配置
ENV_CONFIG = {
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'OMP_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'TORCH_NUM_THREADS': '1',
    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512'  # 限制CUDA内存分配
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
        import gc

        # 设置线程数
        torch.set_num_threads(MEMORY_CONFIG['max_threads'])

        # 启用内存优化
        if MEMORY_CONFIG['enable_fp16']:
            if hasattr(torch.cuda, 'amp'):
                torch.cuda.amp.autocast(enabled=True)

        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except ImportError:
        pass
