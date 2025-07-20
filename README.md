# Jarvis - 粤语语音交互系统

## 项目简介

Jarvis 是一个专为粤语用户设计的智能语音交互系统，特别适合老年用户群体。系统能够：

- 🎤 **听懂粤语**：通过SenseVoiceSmall模型进行语音识别
- 🧠 **智能对话**：使用DeepSeek-Coder-V2-Lite-Instruct生成纯正粤语回复
- 🔊 **粤语发音**：通过CosyVoice2生成自然的粤语语音
- 🖥️ **简洁界面**：基于Gradio构建，适合老年用户的大按钮设计

## 系统架构

```
用户语音输入 → 语音识别(SenseVoice) → LLM处理(DeepSeek) → 语音合成(CosyVoice2) → 语音输出
```

## 主要特性

### ✨ 核心功能
- **一键录音**：点击开始录音，点击停止自动处理
- **纯粤语交互**：LLM确保生成纯正粤语回复
- **流式语音合成**：低延迟的实时语音生成
- **智能缓存**：优化性能，减少等待时间

### 🚀 性能优化
- **Apple Silicon优化**：针对M系列芯片专门优化
- **并发处理**：异步执行语音识别和LLM生成
- **模型预热**：减少冷启动时间
- **内存管理**：智能缓存和垃圾回收

### 🎯 用户体验
- **老年友好界面**：大按钮、清晰状态指示
- **实时反馈**：显示处理进度和状态
- **错误恢复**：自动重试和错误处理

## 技术栈

- **语音识别**: SenseVoiceSmall (FunAudioLLM)
- **大语言模型**: DeepSeek-Coder-V2-Lite-Instruct (MLX)
- **语音合成**: CosyVoice2 (阿里巴巴)
- **用户界面**: Gradio
- **加速框架**: MLX (Apple Silicon), PyTorch
- **开发语言**: Python 3.10

## 系统要求

### 硬件要求
- **推荐配置**: MacBook Air M3 (24GB内存, 10核GPU)
- **最低配置**: Apple Silicon Mac (16GB内存)
- **存储空间**: 至少20GB可用空间（用于模型文件）

### 软件要求
- macOS 12.0+ (建议macOS 13.0+)
- Anaconda 或 Miniconda
- Python 3.10

## 安装指南

### 1. 环境准备

```bash
# 创建虚拟环境
conda create -n jarvis python=3.10
conda activate jarvis
```

### 2. 安装依赖

#### WeTextProcessing (必须先安装)
```shell
conda install -c conda-forge pynini==2.1.5
pip install WeTextProcessing==1.0.3 --no-deps
```

#### 其他依赖
```bash
pip install -r requirements.txt
```

### 3. 模型下载
系统会在首次运行时自动下载所需模型：
- SenseVoiceSmall: 约1GB
- DeepSeek-Coder-V2-Lite-Instruct: 约8GB
- CosyVoice2-0.5B: 约2GB

## 快速开始

### 启动系统
```bash
python jarvis.py
```

启动后会看到Gradio界面，通常在 `http://localhost:7860`

### 基本使用
1. 点击"开始录音"按钮
2. 说出粤语内容（如："你好"、"今日天气点样？"）
3. 点击"停止录音"
4. 系统会自动处理并播放粤语回复

### 示例对话
```
用户: "你好"
Jarvis: "你好！有咩可以幫到你嘅？"

用户: "今日天气点样？"
Jarvis: "今日天气幾好，陽光普照，適合出街行下。"

用户: "帮我讲个故事"
Jarvis: "好啊！從前有個小朋友..."
```

## 性能指标

### 响应时间（在MacBook Air M3上测试）
- **语音识别**: 1-3秒
- **LLM生成**: 2-4秒  
- **语音合成**: 1-2秒
- **端到端总时间**: < 10秒

### 准确率
- **粤语识别准确率**: > 85%
- **粤语回复率**: > 95%
- **系统稳定性**: > 99%

## 故障排除

### 常见问题

#### 1. 模型加载失败
```bash
# 清理缓存重新下载
rm -rf ~/.cache/huggingface
rm -rf pretrained_models/
python jarvis.py
```

#### 2. 语音识别无反应
- 检查麦克风权限
- 确保音频文件格式正确(.wav, 16kHz)

#### 3. 内存不足
```bash
# 清理缓存
python -c "import torch; torch.mps.empty_cache()"
```

#### 4. 启动速度慢
首次启动需要下载模型，请耐心等待。后续启动会明显加快。

## 开发指南

### 项目结构
```
Jarvis/
├── jarvis.py              # 主程序和界面
├── influence/             # LLM处理模块
│   ├── influence.py       # 核心逻辑
│   └── model.py          # 模型配置
├── listen/                # 语音识别模块
│   └── listen.py         # 录音和识别
├── cosyvoice/            # 语音合成模块
│   └── cli/cosyvoice.py  # 合成接口
├── tests/                # 测试脚本
├── docs/                 # 项目文档
└── recordings/           # 录音文件存储
```

### 测试
```bash
# 运行所有测试
python -m pytest tests/

# 系统集成测试
python tests/test_stage7_quick.py

# 单独测试组件
python tests/test_cantonese_llm.py
python tests/test_listen.py
```

### 性能监控
系统内置了详细的性能监控，包括：
- 各阶段处理时间
- 内存使用情况
- GPU利用率（Apple Silicon）
- 音频块生成统计

## 更新日志

### v2.0.0 (2025-07-19)
- ✨ 新增纯粤语LLM回复功能
- 🚀 Apple Silicon M3专项性能优化
- 🔧 修复语音合成中提示词混入问题
- 🎨 全新老年友好界面设计
- ⚡ 流式语音处理优化，延迟显著降低
- 📱 一键录音功能
- 🧪 完整的系统集成测试套件

### v1.0.0 (2025-07-15)
- 🎤 基础语音识别功能
- 🧠 LLM对话生成
- 🔊 语音合成功能
- 🖥️ Gradio界面

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

### 开发流程
1. Fork项目
2. 创建功能分支
3. 提交更改
4. 运行测试确保通过
5. 提交Pull Request

### 代码规范
- 遵循PEP 8代码规范
- 添加必要的注释和文档字符串
- 确保所有测试通过

## 许可证

本项目使用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢

- [FunAudioLLM](https://github.com/FunAudioLLM) - SenseVoiceSmall模型
- [DeepSeek](https://github.com/deepseek-ai) - DeepSeek-Coder模型  
- [阿里巴巴](https://github.com/FunAudioLLM/CosyVoice) - CosyVoice语音合成
- [MLX](https://github.com/ml-explore/mlx) - Apple Silicon优化框架
- [Gradio](https://gradio.app/) - 用户界面框架

## 联系方式

如有问题或建议，请通过GitHub Issues联系我们。

---

**Jarvis - 让粤语对话更自然** 🗣️✨
