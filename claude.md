# Jarvis 多模型集成说明

## 项目概述

Jarvis 是一个多模态智能助手系统，集成了语音识别(ASR)、大语言模型(LLM)和语音合成(TTS)三大核心模型：

- ASR: SenseVoiceSmall (语音识别)
- LLM: DeepSeek-Coder-V2-Lite-Instruct (对话生成)
- TTS: CosyVoice2-0.5B (语音合成)

## 系统要求

- macOS 操作系统（支持 macOS Sonoma 14.0 及以上版本）
- Python 3.10
- Anaconda/Miniconda 环境
- Apple Silicon M3 处理器
- 推荐 8GB 及以上内存

## 模型配置说明

### 1. 语音识别模型 (ASR)
```
模型：iic/SenseVoiceSmall
用途：将用户语音输入转换为文本
特点：
- 轻量级语音识别模型
- 支持实时语音转写
- 针对中文场景优化
```

### 2. 大语言模型 (LLM)
```
模型：DeepSeek-Coder-V2-Lite-Instruct
路径：~/.lmstudio/models/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
特点：
- 基于 MLX 框架优化
- 4bit 量化减少内存占用
- 支持代码生成和通用对话
```

### 3. 语音合成模型 (TTS)
```
模型：CosyVoice2-0.5B
路径：pretrained_models/CosyVoice2-0.5B/
特点：
- 高质量语音合成
- 支持情感表达
- 多说话人支持
```

## 性能优化配置

1. MLX 加速支持：
   - 使用 Apple 的 Metal 引擎加速
   - 模型量化优化
   - 异步推理支持

2. 内存管理：
   - 动态加载模型
   - 使用流式处理减少内存占用
   - 合理设置批处理大小

3. 推理优化：
   - 模型预热
   - 结果缓存
   - 流式处理

## 使用示例

1. 语音识别：
```python
from listen.listen import Listen

listen = Listen()
text = listen.process_audio(audio_data)
```

2. 对话生成：
```python
from influence.influence import Influence

influence = Influence()
response = influence.chat(text)
```

3. 语音合成：
```python
from cosyvoice.cli.cosyvoice import CosyVoice2

tts = CosyVoice2(model_dir="pretrained_models/CosyVoice2-0.5B")
audio = tts.synthesize(text)
```

## 注意事项

1. 模型加载
   - 按需加载模型减少内存占用
   - 注意模型文件完整性
   - 预热模型减少首次延迟

2. 流式处理
   - ASR 支持实时转写
   - TTS 支持流式合成
   - 合理设置缓冲区大小

3. 错误处理
   - 实现模型加载失败的容错
   - 处理网络超时情况
   - 内存不足时的降级方案

## 开发建议

1. 本地开发
   - 使用 MLX 开发工具
   - 监控 Metal 性能
   - 进行性能分析

2. 模型优化
   - 使用量化模型
   - 优化推理参数
   - 实现结果缓存

## Mac OS 模型加载优化

### 1. Core ML 转换与加载
- 可以将模型转换为 Core ML 格式，利用 Apple 原生的机器学习框架
- 使用 Core ML Tools 进行模型转换：
  ```bash
  pip install coremltools
  ```
- Core ML 优势：
  - 原生支持 Metal 性能优化
  - 自动内存管理
  - 更好的电池续航

### 2. MLX 框架优化
- 使用 Apple 开源的 MLX 机器学习框架
- 安装 MLX：
  ```bash
  pip install mlx
  ```
- MLX 特性：
  - 专为 Apple Silicon 芯片优化
  - 支持动态图计算
  - 高效的内存管理

### 3. Metal 性能优化
- 启用 Metal Performance Shaders (MPS)：
  ```python
  import torch
  if torch.backends.mps.is_available():
      device = torch.device("mps")
  ```
- 优化建议：
  - 使用批处理减少 CPU-GPU 数据传输
  - 启用 Metal 内存池
  - 使用异步推理

## 具体模型优化建议

### 1. SenseVoiceSmall (ASR)
```python
# 检查并使用 MPS 加速
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# 使用 Core ML 转换（如果支持）
import coremltools as ct
model_coreml = ct.convert(model, inputs=[ct.TensorType(shape=input_shape)])
```

### 2. DeepSeek-Coder-V2-Lite (LLM)
```python
# 使用 MLX 框架加载和优化
import mlx.core as mx
import mlx.nn as nn

# MLX 原生支持 Apple Silicon
model = load_mlx_model("~/.lmstudio/models/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx")
```

### 3. CosyVoice2-0.5B (TTS)
```python
# 优化 ONNX 模型加载
import onnxruntime as ort

# 配置 ONNX Runtime 使用 CoreML
providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("pretrained_models/CosyVoice2-0.5B/model.onnx", 
                             providers=providers)
```

## 性能监控

1. 使用 Metal System Trace
   - 通过 Xcode Instruments 监控 GPU 使用
   - 分析内存访问模式
   - 识别性能瓶颈

2. 内存优化
   - 使用 Memory Graph Debugger
   - 监控内存压力
   - 实现智能缓存策略

3. 能耗优化
   - 使用 Energy Impact 监控
   - 批量处理减少 CPU 唤醒
   - 优化后台任务

## 参考资源

- [MLX 框架文档](https://ml-explore.github.io/mlx/build/html/index.html)
- [DeepSeek Coder 文档](https://github.com/deepseek-ai/DeepSeek-Coder)
- [CosyVoice2 文档](https://github.com/CosyVoice/CosyVoice2)
- [SenseVoice 文档](https://github.com/iic-project/SenseVoice)
