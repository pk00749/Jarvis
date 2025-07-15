# Jarvis 语音助手项目文档

## 项目概述
基于Mac M3芯片开发的智能语音助手，支持语音识别、粤语对话和语音合成。使用Gradio创建用户界面，专为Mac OS原生支持优化。

## 核心功能
- 🎤 **语音识别**: 使用 iic/SenseVoiceSmall 模型
- 🧠 **智能对话**: 基于 DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx，**专门返回粤语回答**
- 🔊 **语音合成**: 使用 CosyVoice2-0.5B 模型
- 💻 **用户界面**: 基于 Gradio 的 Web 界面

## 技术架构

### 模型配置
1. **语音识别模型**: iic/SenseVoiceSmall
   - 设备: CPU (Mac M3 优化)
   - 支持多语言识别 (中文、英文、粤语等)

2. **大语言模型**: DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
   - 框架: MLX (Apple Silicon 专用)
   - 路径: /Users/yorkhxli/.lmstudio/models/mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx
   - **特殊配置**: 强制返回粤语回答

3. **语音合成模型**: CosyVoice2-0.5B
   - 支持中文和粤语语音合成
   - 使用 zero-shot 提示音频

### Mac M3 优化
- 移除 CUDA 依赖，使用 CPU 和 MPS
- 优化线程配置利用 M3 的性能核心
- 使用 MLX 框架获得最佳性能
- 智能内存管理和缓存清理

## 核心特性

### 粤语对话支持
- **输入**: 支持多语言语音输入
- **处理**: LLM 自动生成粤语回答
- **输出**: 粤语语音合成播放

### 界面功能
- 录音按钮: 开始/停止语音输入
- 播放功能: 自动播放 Jarvis 的语音回应
- 状态指示器: 显示处理进度
- 清除功能: 重置对话状态

### 性能优化
- 实时语音流处理
- 智能音频缓冲
- 内存优化管理
- 错误恢复机制

## 文件结构
```
Jarvis/
├── jarvis.py              # 主应用程序
├── requirements.txt       # 依赖包列表
├── config/
│   └── performance.py     # Mac M3 性能配置
├��─ influence/
│   └── influence.py       # 语音识别和LLM处理
├── listen/
│   └── listen.py          # 音频录制和保存
├── cosyvoice/             # 语音合成模块
├── pretrained_models/     # 预训练模型
├── asset/                 # 音频资源
└── recordings/            # 录音文件存储
```

## 使用方法

### 环境准备
1. 确保使用 Mac M3 设备
2. 安装 Homebrew 和系统依赖
3. 安装 Python 依赖包

### 安装步骤
```bash
# 1. 安装系统依赖
brew install portaudio ffmpeg

# 2. 安装 Python 依赖
pip install -r requirements.txt

# 3. 启动应用
python jarvis.py
```

### 使用流程
1. 在浏览器中打开 http://localhost:7860
2. 点击"录音"按钮开始语音输入
3. 说出您的问题或指令
4. 停止录音后，Jarvis 会自动生成粤语回答
5. 回答将以语音形式播放

## 关键配置

### 粤语回答配置
- LLM 提示词配置为强制返回粤语
- 语音合成使用粤语语调
- 支持粤语语音识别输入

### 性能配置
- 线程数: 4 (适配 M3 性能核心)
- 内存管理: 智能清理和缓存
- 设备选择: 自动检测 MPS 或 CPU

## 错误处理
- 音频格式兼容性处理
- 网络连接异常恢复
- 模型加载失败处理
- 内存不足保护机制

## 开发说明
- 代码遵循 Python 最佳实践
- 完整的错误处理和日志记录
- 模块化设计便于维护
- 性能优化适配 Apple Silicon

## 更新日志
- 2025-07-15: 添加粤语回答强制配置
- 2025-07-15: 完成 Mac M3 优化版本
- 2025-07-15: 实现完整的语音助手功能
